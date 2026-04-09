import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from flask import (
    Blueprint, current_app, jsonify, make_response, render_template, render_template_string, request
)

from .tools.data_tool import *
from . import recommender_original as original_system
from .hybrid_support import (
    DEFAULT_COLLABORATIVE_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    get_hybrid_recommendation_results,
    get_liked_similar_results,
)

bp = Blueprint('main', __name__, url_prefix='/')

COLLABORATIVE_WEIGHT = DEFAULT_COLLABORATIVE_WEIGHT
SEMANTIC_WEIGHT = DEFAULT_SEMANTIC_WEIGHT


def main():
    from flaskr import create_app
    app = create_app()
    app.run(debug=True)

movies, genres, rates = loadData()


@bp.route('/', methods=('GET', 'POST'))
def index():
    participant_id = request.args.get('participant_id', '').strip()
    variant = _get_ab_variant()
    ui_variant = _get_ui_variant()
    if variant is None:
        return render_template_string(
            """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Choose Test Route</title>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
            </head>
            <body>
                <section class="hero is-fullheight is-light">
                    <div class="hero-body">
                        <div class="container" style="max-width: 720px;">
                            <div class="box">
                                <h1 class="title">Choose a Test Route</h1>
                                <p class="subtitle">Select one of the three test combinations.</p>
                                <form method="get" action="/">
                                    <input type="hidden" id="variant_input" name="variant" value="">
                                    <input type="hidden" id="ui_input" name="ui" value="classic">
                                    <div class="field">
                                        <label class="label" for="participant_id">Participant ID</label>
                                        <div class="control">
                                            <input class="input" id="participant_id" name="participant_id" value="{{ participant_id }}" placeholder="e.g. user01">
                                        </div>
                                    </div>
                                    <div class="buttons">
                                        <button class="button is-link" type="button" onclick="submitRoute('A', 'classic')">1) Same UI + Same Algorithm</button>
                                        <button class="button is-info" type="button" onclick="submitRoute('B', 'classic')">2) Same UI + Different Algorithm</button>
                                        <button class="button is-warning" type="button" onclick="submitRoute('A', 'v2')">3) Different UI + Same Algorithm</button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </section>
                <script>
                    function submitRoute(variant, ui) {
                        document.getElementById('variant_input').value = variant;
                        document.getElementById('ui_input').value = ui;
                        document.querySelector('form').submit();
                    }
                </script>
            </body>
            </html>
            """,
            participant_id=participant_id,
        )
    default_genres = genres.to_dict('records')
    user_genres = _parse_cookie_list('user_genres')
    user_rates = _parse_cookie_list('user_rates')
    user_likes = _parse_cookie_list('user_likes')
    if variant == 'A':
        default_genres_movies = original_system.getMoviesByGenres(user_genres)[:10]
        recommendations_movies, recommendations_message = original_system.getRecommendationBy(user_rates)
        likes_similar_movies, likes_similar_message = original_system.getLikedSimilarBy(
            [int(numeric_string) for numeric_string in user_likes]
        )
        likes_movies = original_system.getUserLikesBy(user_likes)
    else:
        default_genres_movies = getMoviesByGenres(user_genres)[:10]
        recommendations_movies, recommendations_message = getRecommendationBy(user_rates)
        likes_similar_movies, likes_similar_message = getLikedSimilarBy([int(numeric_string) for numeric_string in user_likes])
        likes_movies = getUserLikesBy(user_likes)

    selected_genre_names = []
    if user_genres:
        try:
            selected_ids = [int(genre_id) for genre_id in user_genres]
            selected_rows = genres[genres['id'].isin(selected_ids)]
            selected_genre_names = selected_rows['name'].tolist()
        except ValueError:
            selected_genre_names = []

    feedback_state = {
        'genres_count': len(user_genres),
        'ratings_count': len(user_rates),
        'likes_count': len(user_likes),
        'selected_genres': selected_genre_names,
    }

    response = make_response(render_template('index.html',
                                             genres=default_genres,
                                             user_genres=user_genres,
                                             user_rates=user_rates,
                                             user_likes=user_likes,
                                             default_genres_movies=default_genres_movies,
                                             recommendations=recommendations_movies,
                                             recommendations_message=recommendations_message,
                                             likes_similars=likes_similar_movies,
                                             likes_similar_message=likes_similar_message,
                                             likes=likes_movies,
                                             ab_variant=variant,
                                             ui_variant=ui_variant,
                                             participant_id=participant_id,
                                             collaborative_weight=COLLABORATIVE_WEIGHT,
                                             semantic_weight=SEMANTIC_WEIGHT,
                                             feedback_state=feedback_state,
                                             ))
    _log_ab_event(
        participant_id,
        variant,
        user_genres,
        user_rates,
        user_likes,
        recommendations_movies,
        likes_similar_movies,
    )
    return response


@bp.route('/ab-summary', methods=('GET',))
def ab_summary():
    summary = _build_ab_summary()
    return jsonify(summary)


def _get_ab_variant():
    forced_variant = request.args.get('variant', '').upper()
    if forced_variant in {'A', 'B'}:
        return forced_variant
    return None


def _get_ui_variant():
    requested_ui = request.args.get('ui', 'classic').strip().lower()
    if requested_ui in {'classic', 'v2'}:
        return requested_ui
    return 'classic'


def _parse_cookie_list(cookie_name):
    raw_value = request.cookies.get(cookie_name, '')
    if not raw_value:
        return []
    decoded_value = unquote(raw_value)
    return [item for item in decoded_value.split(',') if item]


def _ab_log_path():
    log_dir = Path(current_app.root_path).parent / 'ab_test_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / 'ab_test_events.jsonl'


def _log_ab_event(participant_id, variant, user_genres, user_rates, user_likes, recommendations, likes_similars):
    log_path = _ab_log_path()
    liked_movie_ids = [int(movie_id) for movie_id in user_likes]
    recommendation_ids = [movie['movieId'] for movie in recommendations]
    event = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'participant_id': participant_id,
        'variant': variant,
        'genres_count': len(user_genres),
        'ratings_count': len(user_rates),
        'likes_count': len(user_likes),
        'user_like_ids': liked_movie_ids,
        'recommendation_ids': recommendation_ids,
        'similar_like_ids': [movie['movieId'] for movie in likes_similars],
        'liked_recommendation_count': len(set(liked_movie_ids).intersection(recommendation_ids)),
    }
    with log_path.open('a', encoding='utf-8') as file_handle:
        file_handle.write(json.dumps(event) + '\n')


def _build_ab_summary():
    log_path = _ab_log_path()
    summary = {
        'A': {
            'page_views': 0,
            'unique_participants': 0,
            'avg_ratings': 0.0,
            'avg_likes': 0.0,
            'recommendation_hit_rate': 0.0,
            'recommendation_like_rate': 0.0,
        },
        'B': {
            'page_views': 0,
            'unique_participants': 0,
            'avg_ratings': 0.0,
            'avg_likes': 0.0,
            'recommendation_hit_rate': 0.0,
            'recommendation_like_rate': 0.0,
        },
    }

    if not log_path.exists():
        return summary

    participants = {'A': set(), 'B': set()}
    recommendation_hits = {'A': 0, 'B': 0}
    recommendation_slots = {'A': 0, 'B': 0}

    with log_path.open('r', encoding='utf-8') as file_handle:
        for raw_line in file_handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            event = json.loads(raw_line)
            variant = event.get('variant')
            if variant not in summary:
                continue
            summary[variant]['page_views'] += 1
            summary[variant]['avg_ratings'] += event.get('ratings_count', 0)
            summary[variant]['avg_likes'] += event.get('likes_count', 0)
            liked_recommendation_count = event.get('liked_recommendation_count', 0)
            recommendation_ids = event.get('recommendation_ids', [])
            recommendation_hits[variant] += 1 if liked_recommendation_count > 0 else 0
            recommendation_slots[variant] += len(recommendation_ids)
            summary[variant]['recommendation_like_rate'] += liked_recommendation_count
            participant_id = event.get('participant_id')
            if participant_id:
                participants[variant].add(participant_id)

    for variant_name, variant in summary.items():
        variant['unique_participants'] = len(participants[variant_name])
        if variant['page_views'] == 0:
            continue
        variant['avg_ratings'] /= float(variant['page_views'])
        variant['avg_likes'] /= float(variant['page_views'])
        variant['recommendation_hit_rate'] = recommendation_hits[variant_name] / float(variant['page_views'])
        if recommendation_slots[variant_name] > 0:
            variant['recommendation_like_rate'] /= float(recommendation_slots[variant_name])
        else:
            variant['recommendation_like_rate'] = 0.0

    return summary


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    if len(results) > 0:
        return results.to_dict('records') # type: ignore
    return results

def is_genre_match(movie_genres, interested_genres):
    return bool(set(movie_genres).intersection(set(interested_genres)))

def getMoviesByGenres(user_genres):
    results = []
    if len(user_genres) > 0:
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres,columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        results = movies[movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records') # type: ignore
    return results

# Modify this function
def getRecommendationBy(user_rates):
    results = []
    if len(user_rates) > 0:
        results = get_hybrid_recommendation_results(
            movies,
            rates,
            user_rates,
            k=12,
            collaborative_weight=COLLABORATIVE_WEIGHT,
            semantic_weight=SEMANTIC_WEIGHT,
        )

    if len(results) > 0:
        return results.to_dict('records'), "These movies are recommended by a TimeSVD++-style model that uses rating timestamps plus movie text features."  # type: ignore
    return results, "No recommendations."


# Modify this function
def getLikedSimilarBy(user_likes):
    results = []
    if len(user_likes) > 0:
        results = get_liked_similar_results(movies, user_likes, k=12)
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked movies based on semantic text embeddings." # type: ignore
    return results, "No similar movies found."

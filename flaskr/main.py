import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
from flask import (
    Blueprint, current_app, jsonify, make_response, render_template, render_template_string, request
)

from .tools.data_tool import *
from . import recommender_original as original_system

from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

bp = Blueprint('main', __name__, url_prefix='/')

_MOVIE_EMBEDDING_CACHE = None
_MOVIE_TIME_CACHE = None

ACTOR_COLUMNS = ('actors', 'actor', 'cast', 'casts', 'starring')
DIRECTOR_COLUMNS = ('director', 'directors', 'filmmaker', 'crew')
DEFAULT_USER_ID = 611


def main():
    from flaskr import create_app
    app = create_app()
    app.run(debug=True)

movies, genres, rates = loadData()


def _to_string_list(raw_values):
    if raw_values is None:
        return []
    if isinstance(raw_values, list):
        return [str(item).strip() for item in raw_values if str(item).strip()]
    if isinstance(raw_values, str):
        if not raw_values.strip():
            return []
        return [item.strip() for item in raw_values.split(',') if item.strip()]
    return []


def _safe_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _safe_int_list(values):
    parsed_values = []
    for value in values:
        parsed_value = _safe_int(value)
        if parsed_value is not None:
            parsed_values.append(parsed_value)
    return parsed_values


def _to_dataframe(records):
    if isinstance(records, pd.DataFrame):
        return records.copy(deep=True)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _to_record_list(records):
    if isinstance(records, pd.DataFrame):
        if len(records) == 0:
            return []
        return records.to_dict('records') # type: ignore
    return list(records) if records else []


def _parse_user_rate_records(user_rates):
    parsed_records = []
    for record in user_rates:
        if isinstance(record, dict):
            movie_id = _safe_int(record.get('movieId'))
            rating = record.get('rating')
            try:
                rating_value = float(rating)
            except (TypeError, ValueError):
                continue
            parsed_records.append({
                'userId': _safe_int(record.get('userId')) or DEFAULT_USER_ID,
                'movieId': movie_id,
                'rating': rating_value,
            })
            continue

        parts = str(record).split('|')
        if len(parts) < 3:
            continue
        user_id = _safe_int(parts[0]) or DEFAULT_USER_ID
        movie_id = _safe_int(parts[1])
        try:
            rating_value = float(parts[2])
        except ValueError:
            continue
        if movie_id is None:
            continue
        parsed_records.append({
            'userId': user_id,
            'movieId': movie_id,
            'rating': rating_value,
        })
    return parsed_records


def _split_name_tokens(raw_text):
    if raw_text is None or pd.isna(raw_text):
        return []
    if isinstance(raw_text, list):
        source = [str(item) for item in raw_text]
    else:
        source = re.split(r'[|,;/]', str(raw_text))
    tokens = []
    for token in source:
        cleaned = token.strip()
        if cleaned and cleaned.lower() not in {'nan', 'none'}:
            tokens.append(cleaned)
    return tokens


def _get_people_from_movie(movie_record, columns):
    for column in columns:
        if column in movie_record:
            tokens = _split_name_tokens(movie_record.get(column))
            if tokens:
                return tokens
    return []


def _normalize_movie_genres(movie_record):
    genres_value = movie_record.get('genres', [])
    if isinstance(genres_value, list):
        return [str(name) for name in genres_value if str(name).strip()]
    if isinstance(genres_value, str):
        return [name.strip() for name in genres_value.split('|') if name.strip()]
    return []


def _movie_lookup_map():
    lookup = {}
    for _, row in movies.iterrows():
        row_record = row.to_dict()
        movie_id = _safe_int(row_record.get('movieId'))
        if movie_id is not None:
            lookup[movie_id] = row_record
    return lookup


def _selected_genre_names(user_genres):
    selected_ids = _safe_int_list(user_genres)
    if not selected_ids:
        return []
    selected_rows = genres[genres['id'].isin(selected_ids)]
    return selected_rows['name'].tolist()


def _genre_counter_from_movie_ids(movie_ids):
    lookup = _movie_lookup_map()
    genre_counter = Counter()
    for movie_id in movie_ids:
        movie_record = lookup.get(movie_id)
        if not movie_record:
            continue
        for genre_name in _normalize_movie_genres(movie_record):
            genre_counter[genre_name] += 1
    return genre_counter


def _ensure_base_scores(recommendations_df):
    if len(recommendations_df) == 0:
        return recommendations_df

    ranked_df = recommendations_df.copy(deep=True)
    if 'final_score' in ranked_df.columns:
        ranked_df['base_score'] = ranked_df['final_score'].astype(float)
        return ranked_df
    if 'similarity' in ranked_df.columns:
        ranked_df['base_score'] = ranked_df['similarity'].astype(float)
        return ranked_df

    total = len(ranked_df)
    ranked_df['base_score'] = [1.0 - (index / max(total, 1)) for index in range(total)]
    return ranked_df


def _filter_excluded_movies(items_df, excluded_movie_ids):
    if len(items_df) == 0 or not excluded_movie_ids:
        return items_df
    filtered = items_df[~items_df['movieId'].isin(list(excluded_movie_ids))]
    return filtered.copy(deep=True)


def _apply_feedback_adjustments(recommendations_df, user_genres, user_likes, user_dislikes, user_not_interested):
    if len(recommendations_df) == 0:
        return recommendations_df

    adjusted_df = _ensure_base_scores(recommendations_df)
    adjusted_df['feedback_bonus'] = 0.0
    adjusted_df['feedback_penalty'] = 0.0
    adjusted_df['feedback_adjustment'] = 0.0
    adjusted_df['matched_preferred_genres'] = ''
    adjusted_df['matched_avoided_genres'] = ''

    excluded_movie_ids = set(_safe_int_list(user_likes))
    excluded_movie_ids.update(_safe_int_list(user_dislikes))
    excluded_movie_ids.update(_safe_int_list(user_not_interested))
    adjusted_df = _filter_excluded_movies(adjusted_df, excluded_movie_ids)

    selected_genres = set(_selected_genre_names(user_genres))
    disliked_genre_counter = _genre_counter_from_movie_ids(
        _safe_int_list(user_dislikes) + _safe_int_list(user_not_interested)
    )
    disliked_genres = set(disliked_genre_counter.keys())

    for row_index, row in adjusted_df.iterrows():
        movie_genres = set(_normalize_movie_genres(row.to_dict()))
        preferred_matches = sorted(movie_genres.intersection(selected_genres))
        avoided_matches = sorted(movie_genres.intersection(disliked_genres))

        bonus = 0.03 * len(preferred_matches)
        penalty = 0.08 * len(avoided_matches)
        adjustment = bonus - penalty

        adjusted_df.at[row_index, 'feedback_bonus'] = round(bonus, 4)
        adjusted_df.at[row_index, 'feedback_penalty'] = round(penalty, 4)
        adjusted_df.at[row_index, 'feedback_adjustment'] = round(adjustment, 4)
        adjusted_df.at[row_index, 'matched_preferred_genres'] = ', '.join(preferred_matches)
        adjusted_df.at[row_index, 'matched_avoided_genres'] = ', '.join(avoided_matches)

    adjusted_df['adjusted_score'] = adjusted_df['base_score'] + adjusted_df['feedback_adjustment']
    return adjusted_df.sort_values(by=['adjusted_score'], ascending=False)


def _build_recommendation_explanations(recommendations, user_genres, user_rates, user_likes):
    explanation_map = {}
    if not recommendations:
        return explanation_map

    selected_genres = set(_selected_genre_names(user_genres))
    rate_records = _parse_user_rate_records(user_rates)
    high_rating_movie_ids = [
        int(record['movieId']) for record in rate_records
        if record.get('movieId') is not None and float(record.get('rating', 0.0)) >= 4.0
    ]
    history_movie_ids = list(dict.fromkeys(_safe_int_list(user_likes) + high_rating_movie_ids))
    movie_lookup = _movie_lookup_map()

    actor_preferences = Counter()
    director_preferences = Counter()
    for history_movie_id in history_movie_ids:
        history_movie = movie_lookup.get(history_movie_id)
        if not history_movie:
            continue
        for actor_name in _get_people_from_movie(history_movie, ACTOR_COLUMNS):
            actor_preferences[actor_name] += 1
        for director_name in _get_people_from_movie(history_movie, DIRECTOR_COLUMNS):
            director_preferences[director_name] += 1

    for movie in recommendations:
        movie_id = _safe_int(movie.get('movieId'))
        if movie_id is None:
            continue
        movie_title = movie.get('title', f'Movie {movie_id}')
        movie_genres = set(_normalize_movie_genres(movie))

        reasons = []

        preferred_matches = sorted(movie_genres.intersection(selected_genres))
        if preferred_matches:
            reasons.append(f"Matches your preferred genres: {', '.join(preferred_matches[:3])}.")

        best_history_match = None
        best_overlap_size = -1
        for history_movie_id in history_movie_ids:
            history_movie = movie_lookup.get(history_movie_id)
            if not history_movie:
                continue
            history_genres = set(_normalize_movie_genres(history_movie))
            overlap = sorted(movie_genres.intersection(history_genres))
            if len(overlap) > best_overlap_size:
                best_overlap_size = len(overlap)
                best_history_match = (history_movie, overlap)

        if best_history_match and best_history_match[0]:
            history_movie, overlap = best_history_match
            if overlap:
                reasons.append(
                    f"Similar in style to '{history_movie.get('title', 'one of your liked titles')}' with overlap in {', '.join(overlap[:2])}."
                )

        if rate_records:
            reasons.append('Users with rating patterns similar to yours also engaged positively with this title.')

        movie_actors = _get_people_from_movie(movie, ACTOR_COLUMNS)
        actor_overlap = [actor for actor in movie_actors if actor in actor_preferences]
        if actor_overlap:
            reasons.append(f"Includes performers you often engage with: {', '.join(actor_overlap[:2])}.")

        movie_directors = _get_people_from_movie(movie, DIRECTOR_COLUMNS)
        director_overlap = [director for director in movie_directors if director in director_preferences]
        if director_overlap:
            reasons.append(f"Directed by creators aligned with your preferences: {', '.join(director_overlap[:2])}.")

        if not reasons:
            reasons.append('Overall content signals are close to your recent preference behavior.')

        explanation_map[str(movie_id)] = {
            'movieId': movie_id,
            'movieTitle': movie_title,
            'summary': reasons[0],
            'details': reasons,
            'signals': {
                'collaborative': round(COLLABORATIVE_WEIGHT if rate_records else 0.0, 2),
                'semantic': round(SEMANTIC_WEIGHT if rate_records else 1.0, 2),
                'feedback_adjustment': round(float(movie.get('feedback_adjustment', 0.0)), 3),
            },
        }

    return explanation_map


def _build_preference_dashboard(user_genres, user_rates, user_likes, user_dislikes, user_not_interested):
    rate_records = _parse_user_rate_records(user_rates)
    selected_genres = _selected_genre_names(user_genres)
    like_ids = _safe_int_list(user_likes)
    dislike_ids = _safe_int_list(user_dislikes)
    not_interested_ids = _safe_int_list(user_not_interested)

    genre_scores = Counter()
    for genre_name in selected_genres:
        genre_scores[genre_name] += 1.2

    movie_lookup = _movie_lookup_map()

    for movie_id in like_ids:
        movie = movie_lookup.get(movie_id)
        if not movie:
            continue
        for genre_name in _normalize_movie_genres(movie):
            genre_scores[genre_name] += 1.6

    for record in rate_records:
        movie_id = _safe_int(record.get('movieId'))
        if movie_id is None:
            continue
        movie = movie_lookup.get(movie_id)
        if not movie:
            continue
        rating_value = float(record.get('rating', 0.0))
        rating_effect = rating_value - 3.0
        for genre_name in _normalize_movie_genres(movie):
            genre_scores[genre_name] += rating_effect

    for movie_id in dislike_ids + not_interested_ids:
        movie = movie_lookup.get(movie_id)
        if not movie:
            continue
        for genre_name in _normalize_movie_genres(movie):
            genre_scores[genre_name] -= 1.4

    positive_genres = [(name, score) for name, score in genre_scores.items() if score > 0]
    positive_genres.sort(key=lambda item: item[1], reverse=True)
    total_positive = sum(score for _, score in positive_genres) or 1.0
    genre_preferences = [
        {
            'name': name,
            'score': round(float(score), 3),
            'ratio': round(float(score) / total_positive, 3),
        }
        for name, score in positive_genres[:8]
    ]

    suppressed_genres = [
        {'name': name, 'score': round(float(abs(score)), 3)}
        for name, score in sorted(genre_scores.items(), key=lambda item: item[1])
        if score < 0
    ][:6]

    positive_history_ids = list(dict.fromkeys(like_ids + [
        int(record['movieId']) for record in rate_records if float(record.get('rating', 0.0)) >= 4.0
    ]))

    actor_counter = Counter()
    director_counter = Counter()
    for movie_id in positive_history_ids:
        movie = movie_lookup.get(movie_id)
        if not movie:
            continue
        for actor_name in _get_people_from_movie(movie, ACTOR_COLUMNS):
            actor_counter[actor_name] += 1
        for director_name in _get_people_from_movie(movie, DIRECTOR_COLUMNS):
            director_counter[director_name] += 1

    preferred_actors = [
        {'name': actor_name, 'score': score}
        for actor_name, score in actor_counter.most_common(5)
    ]
    preferred_directors = [
        {'name': director_name, 'score': score}
        for director_name, score in director_counter.most_common(5)
    ]

    focus_topics = [item['name'] for item in genre_preferences[:3]]
    focus_summary = ', '.join(focus_topics) if focus_topics else 'Still learning your preference focus'

    return {
        'summary': {
            'selected_genres': len(user_genres),
            'rated_movies': len(rate_records),
            'likes': len(like_ids),
            'dislikes': len(dislike_ids),
            'not_interested': len(not_interested_ids),
        },
        'genre_preferences': genre_preferences,
        'suppressed_genres': suppressed_genres,
        'preferred_actors': preferred_actors,
        'preferred_directors': preferred_directors,
        'focus_summary': focus_summary,
    }


def _build_feedback_loop_summary(dashboard):
    summary = dashboard.get('summary', {})
    return {
        'headline': 'Every action you take helps train the recommendation model.',
        'steps': [
            {
                'title': '1. You provide behavior',
                'description': (
                    f"You have provided {summary.get('rated_movies', 0)} ratings, "
                    f"{summary.get('likes', 0)} likes, and {summary.get('dislikes', 0) + summary.get('not_interested', 0)} negative feedback actions."
                ),
            },
            {
                'title': '2. System updates preferences',
                'description': 'The system increases weights for preferred topics and lowers exposure for unwanted content.',
            },
            {
                'title': '3. Ranking adjusts',
                'description': 'The next recommendation list moves better-fitting items higher.',
            },
            {
                'title': '4. Continuous loop',
                'description': 'As you provide new feedback, the model keeps learning and recommendations keep improving.',
            },
        ],
    }


def _build_feedback_impact(feedback_event):
    if not isinstance(feedback_event, dict):
        return {
            'message': 'The system will update preference weights immediately using your new feedback.',
            'details': [],
        }

    feedback_type = str(feedback_event.get('type', '')).strip().lower()
    movie_id = _safe_int(feedback_event.get('movieId'))
    movie_lookup = _movie_lookup_map()
    movie_record = movie_lookup.get(movie_id) if movie_id is not None else None
    movie_title = movie_record.get('title', 'this movie') if movie_record else 'this movie'
    movie_genres = _normalize_movie_genres(movie_record or {})
    genre_text = ', '.join(movie_genres[:2]) if movie_genres else 'related themes'

    if feedback_type == 'like':
        return {
            'message': f"Marked '{movie_title}' as liked and increased recommendation weight for {genre_text}.",
            'details': [
                'More similar content will appear earlier.',
                'Ranking for related genres has been raised.',
            ],
        }
    if feedback_type == 'dislike':
        return {
            'message': f"Marked '{movie_title}' as disliked and reduced recommendation frequency for {genre_text}.",
            'details': [
                'Exposure for similar titles will decrease.',
                'Future results will prioritize avoiding related content.',
            ],
        }
    if feedback_type == 'not_interested':
        return {
            'message': f"Marked '{movie_title}' as not interested and temporarily deprioritized similar items.",
            'details': [
                'Content with similar themes will be downweighted for now.',
                'The system will explore other directions to learn new preferences.',
            ],
        }
    if feedback_type == 'rating_save':
        return {
            'message': 'Your new ratings are active and recommendations were recalculated from your rating pattern.',
            'details': [
                'Collaborative signals were refreshed.',
                'Ranking is now closer to your scoring preference.',
            ],
        }
    if feedback_type == 'genre_save':
        return {
            'message': 'Genre preferences were updated, and recommendations now prioritize your new genre choices.',
            'details': [
                'Genre weights were recalculated.',
                'Future ratings and feedback will continue refining this profile.',
            ],
        }

    return {
        'message': 'Feedback received. Recommendation state was updated with the new signal.',
        'details': [],
    }


def _build_recommendation_state(
    variant,
    user_genres,
    user_rates,
    user_likes,
    user_dislikes=None,
    user_not_interested=None,
    feedback_event=None,
):
    user_dislikes = user_dislikes or []
    user_not_interested = user_not_interested or []

    if variant == 'A':
        default_genres_movies = original_system.getMoviesByGenres(user_genres)[:10]
        recommendations_movies, recommendations_message = original_system.getRecommendationBy(user_rates)
        likes_similar_movies, likes_similar_message = original_system.getLikedSimilarBy(
            _safe_int_list(user_likes)
        )
        likes_movies = original_system.getUserLikesBy(user_likes)
    else:
        default_genres_movies = getMoviesByGenres(user_genres)[:10]
        recommendations_movies, recommendations_message = getRecommendationBy(user_rates)
        likes_similar_movies, likes_similar_message = getLikedSimilarBy(_safe_int_list(user_likes))
        likes_movies = getUserLikesBy(user_likes)

    recommendations_df = _to_dataframe(recommendations_movies)
    recommendations_df = _apply_feedback_adjustments(
        recommendations_df,
        user_genres,
        user_likes,
        user_dislikes,
        user_not_interested,
    )
    recommendations_movies = _to_record_list(recommendations_df.head(12))

    excluded_ids = set(_safe_int_list(user_dislikes) + _safe_int_list(user_not_interested))
    likes_similar_df = _filter_excluded_movies(_to_dataframe(likes_similar_movies), excluded_ids)
    likes_similar_movies = _to_record_list(likes_similar_df.head(12))

    selected_genre_names = _selected_genre_names(user_genres)
    recommendation_explanations = _build_recommendation_explanations(
        recommendations_movies,
        user_genres,
        user_rates,
        user_likes,
    )
    preference_dashboard = _build_preference_dashboard(
        user_genres,
        user_rates,
        user_likes,
        user_dislikes,
        user_not_interested,
    )
    feedback_loop = _build_feedback_loop_summary(preference_dashboard)
    feedback_impact = _build_feedback_impact(feedback_event)

    feedback_state = {
        'genres_count': len(user_genres),
        'ratings_count': len(user_rates),
        'likes_count': len(user_likes),
        'dislikes_count': len(user_dislikes),
        'not_interested_count': len(user_not_interested),
        'selected_genres': selected_genre_names,
    }

    return {
        'default_genres_movies': default_genres_movies,
        'recommendations': recommendations_movies,
        'recommendations_message': recommendations_message,
        'likes_similars': likes_similar_movies,
        'likes_similar_message': likes_similar_message,
        'likes': likes_movies,
        'feedback_state': feedback_state,
        'recommendation_explanations': recommendation_explanations,
        'preference_dashboard': preference_dashboard,
        'feedback_loop': feedback_loop,
        'feedback_impact': feedback_impact,
    }


@bp.route('/', methods=('GET', 'POST'))
def index():
    participant_id = request.args.get('participant_id', '').strip()
    variant = _get_ab_variant()
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
                                <p class="subtitle">Select which system version this participant should use for the algorithm A/B test.</p>
                                <form method="get" action="/">
                                    <div class="field">
                                        <label class="label" for="participant_id">Participant ID</label>
                                        <div class="control">
                                            <input class="input" id="participant_id" name="participant_id" value="{{ participant_id }}" placeholder="e.g. user01">
                                        </div>
                                    </div>
                                    <div class="buttons">
                                        <button class="button is-link" type="submit" name="variant" value="A">Route A: Original</button>
                                        <button class="button is-primary" type="submit" name="variant" value="B">Route B: Enhanced</button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </section>
            </body>
            </html>
            """,
            participant_id=participant_id,
        )
    default_genres = genres.to_dict('records')
    user_genres = _parse_cookie_list('user_genres')
    user_rates = _parse_cookie_list('user_rates')
    user_likes = _parse_cookie_list('user_likes')
    user_dislikes = _parse_cookie_list('user_dislikes')
    user_not_interested = _parse_cookie_list('user_not_interested')

<<<<<<< HEAD
=======
    state = _build_recommendation_state(
        variant,
        user_genres,
        user_rates,
        user_likes,
        user_dislikes=user_dislikes,
        user_not_interested=user_not_interested,
    )

>>>>>>> ui
    response = make_response(render_template('index.html',
                                             genres=default_genres,
                                             user_genres=user_genres,
                                             user_rates=user_rates,
                                             user_likes=user_likes,
                                             user_dislikes=user_dislikes,
                                             user_not_interested=user_not_interested,
                                             default_genres_movies=state['default_genres_movies'],
                                             recommendations=state['recommendations'],
                                             recommendations_message=state['recommendations_message'],
                                             likes_similars=state['likes_similars'],
                                             likes_similar_message=state['likes_similar_message'],
                                             likes=state['likes'],
                                             ab_variant=variant,
                                             participant_id=participant_id,
<<<<<<< HEAD
=======
                                             collaborative_weight=COLLABORATIVE_WEIGHT,
                                             semantic_weight=SEMANTIC_WEIGHT,
                                             feedback_state=state['feedback_state'],
                                             recommendation_explanations=state['recommendation_explanations'],
                                             preference_dashboard=state['preference_dashboard'],
                                             feedback_loop=state['feedback_loop'],
                                             feedback_impact=state['feedback_impact'],
>>>>>>> ui
                                             ))
    _log_ab_event(
        participant_id,
        variant,
        user_genres,
        user_rates,
        user_likes,
        state['recommendations'],
        state['likes_similars'],
    )
    return response


@bp.route('/api/recommendation-state', methods=('POST',))
def recommendation_state():
    payload = request.get_json(silent=True) or {}
    variant = str(payload.get('variant', 'B')).strip().upper()
    if variant not in {'A', 'B'}:
        variant = 'B'

    user_genres = _to_string_list(payload.get('user_genres', []))
    user_rates = _to_string_list(payload.get('user_rates', []))
    user_likes = _to_string_list(payload.get('user_likes', []))
    user_dislikes = _to_string_list(payload.get('user_dislikes', []))
    user_not_interested = _to_string_list(payload.get('user_not_interested', []))
    feedback_event = payload.get('feedback_event')

    state = _build_recommendation_state(
        variant,
        user_genres,
        user_rates,
        user_likes,
        user_dislikes=user_dislikes,
        user_not_interested=user_not_interested,
        feedback_event=feedback_event,
    )

    return jsonify(state)


@bp.route('/ab-summary', methods=('GET',))
def ab_summary():
    summary = _build_ab_summary()
    return jsonify(summary)


def _get_ab_variant():
    forced_variant = request.args.get('variant', '').upper()
    if forced_variant in {'A', 'B'}:
        return forced_variant
    return None


<<<<<<< HEAD
=======
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


>>>>>>> ui
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
    global movies, rates, _MOVIE_EMBEDDING_CACHE, _MOVIE_TIME_CACHE
    results = []
    if len(user_rates) > 0:
        movies, _, rates = loadData()
        _MOVIE_EMBEDDING_CACHE = None
        _MOVIE_TIME_CACHE = None
        reader = Reader(rating_scale=(0.5, 5.0))
        algo = SVDpp(
            n_factors=80,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42,
        )
        user_rates = ratesFromUser(user_rates)
        training_rates = pd.concat([rates[['userId', 'movieId', 'rating']], user_rates], ignore_index=True)
        training_data = Dataset.load_from_df(training_rates, reader=reader)
        trainset = training_data.build_full_trainset()
        algo.fit(trainset)

        user_id = int(user_rates['userId'].iloc[0])
        rated_movie_ids = set(user_rates[user_rates['userId'] == user_id]['movieId'].tolist())

        embedding_cache = _get_movie_embedding_cache()
        movie_vectors = embedding_cache['vectors']
        movie_id_to_index = embedding_cache['movie_id_to_index']
        time_scores = _get_movie_time_scores()
        user_profile = _build_user_profile_from_ratings(user_rates, movie_vectors, movie_id_to_index)

        candidate_movie_ids = []
        cf_scores = []
        semantic_scores = []

        for movie_id in movies['movieId'].tolist():
            if movie_id in rated_movie_ids:
                continue
            candidate_movie_ids.append(movie_id)
            cf_scores.append(algo.predict(user_id, movie_id).est)
            movie_index = movie_id_to_index.get(movie_id)
            semantic_scores.append(0.0 if user_profile is None or movie_index is None else float(np.dot(user_profile, movie_vectors[movie_index])))

        if candidate_movie_ids:
            candidate_indices = [movie_id_to_index[movie_id] for movie_id in candidate_movie_ids]
            time_candidate_scores = time_scores[candidate_indices]
            cf_scores = _minmax_scale(cf_scores)
            semantic_scores = _minmax_scale(semantic_scores)
            time_candidate_scores = _minmax_scale(time_candidate_scores)

            result_frame = movies[movies['movieId'].isin(candidate_movie_ids)].copy(deep=True)
            result_frame['final_score'] = (
                0.65 * cf_scores
                + 0.25 * semantic_scores
                + 0.10 * time_candidate_scores
            )
            results = result_frame.sort_values(by=['final_score'], ascending=False).head(12)

    if len(results) > 0:
        return results.to_dict('records'), "These movies are recommended by an SVD++ + TF-IDF + time-aware hybrid."  # type: ignore
    return results, "No recommendations."



# Modify this function
def getLikedSimilarBy(user_likes):
    global movies, rates, _MOVIE_EMBEDDING_CACHE, _MOVIE_TIME_CACHE
    results = []
    if len(user_likes) > 0:
        movies, _, rates = loadData()
        _MOVIE_EMBEDDING_CACHE = None
        _MOVIE_TIME_CACHE = None
        embedding_cache = _get_movie_embedding_cache()
        movie_vectors = embedding_cache['vectors']
        movie_id_to_index = embedding_cache['movie_id_to_index']
        liked_movie_ids = [int(movie_id) for movie_id in user_likes]
        user_profile = _build_semantic_profile(liked_movie_ids, movie_vectors, movie_id_to_index)
        if user_profile is not None:
            results = _semantic_recommendation_results(
                user_profile,
                movie_vectors,
                liked_movie_ids,
                embedding_cache['movie_ids'],
                12,
            )
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked movies based on semantic text embeddings." # type: ignore
    return results, "No similar movies found."


# Step 1: Representing items with multi-hot vectors
def item_representation_based_movie_genres(movies_df):
    movies_with_genres = movies_df.copy(deep=True)
    genre_list = []
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)

    movies_with_genres = movies_with_genres.fillna(0)

    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()
    
    return movies_genre_matrix, movies_with_genres, genre_list

# Step 2: Building user profile
def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T
    
    if normalized:
        user_profile = user_profile / sum(user_profile.values)
        
    return user_profile
# Step 3: Predicting user preference for items
def generate_recommendation_results(user_profile,item_rep_matrix, movies_data, k=12):
    u_v = user_profile.values
    u_v_matrix =  [u_v]
    recommendation_table =  cosine_similarity(u_v_matrix,item_rep_matrix) # type: ignore
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result


def _movie_text(row):
    genres_text = ' '.join(row['genres']) if isinstance(row.get('genres'), list) else ''
    overview = '' if pd.isna(row.get('overview')) else str(row.get('overview'))
    title = '' if pd.isna(row.get('title')) else str(row.get('title'))
    year = '' if pd.isna(row.get('year')) else str(row.get('year'))
    return f"{title} {year} {genres_text} {overview}".strip().lower()


def _l2_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _minmax_scale(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    min_value = arr.min()
    max_value = arr.max()
    if np.isclose(min_value, max_value):
        return np.zeros_like(arr)
    return (arr - min_value) / (max_value - min_value)


def _get_movie_embedding_cache():
    global _MOVIE_EMBEDDING_CACHE
    if _MOVIE_EMBEDDING_CACHE is not None:
        return _MOVIE_EMBEDDING_CACHE

    movie_frame = movies.copy(deep=True)
    texts = movie_frame.apply(_movie_text, axis=1).tolist()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=6000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    if tfidf_matrix.shape[1] <= 2:
        vectors = tfidf_matrix.toarray()
    else:
        n_components = min(100, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if n_components < 2:
            vectors = tfidf_matrix.toarray()
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            vectors = svd.fit_transform(tfidf_matrix)

    vectors = _l2_normalize(vectors)
    movie_ids = movie_frame['movieId'].astype(int).tolist()
    _MOVIE_EMBEDDING_CACHE = {
        'movie_ids': movie_ids,
        'vectors': vectors,
        'movie_id_to_index': {movie_id: index for index, movie_id in enumerate(movie_ids)},
    }
    return _MOVIE_EMBEDDING_CACHE


def _build_semantic_profile(movie_ids, movie_vectors, movie_id_to_index):
    profile_vectors = []
    weights = []
    for movie_id in movie_ids:
        index = movie_id_to_index.get(int(movie_id))
        if index is None:
            continue
        profile_vectors.append(movie_vectors[index])
        weights.append(1.0)

    if not profile_vectors:
        return None

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    profile = np.average(np.vstack(profile_vectors), axis=0, weights=weights)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return None
    return profile / norm


def _build_user_profile_from_ratings(user_rates_df, movie_vectors, movie_id_to_index):
    rated_vectors = []
    weights = []
    for _, row in user_rates_df.iterrows():
        movie_index = movie_id_to_index.get(int(row['movieId']))
        if movie_index is None:
            continue
        rated_vectors.append(movie_vectors[movie_index])
        weights.append(max(float(row['rating']), 0.5))

    if not rated_vectors:
        return None

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    profile = np.average(np.vstack(rated_vectors), axis=0, weights=weights)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return None
    return profile / norm


def _movie_semantic_similarity(user_rates_df, movie_id, movie_vectors, movie_id_to_index):
    user_profile = _build_user_profile_from_ratings(user_rates_df, movie_vectors, movie_id_to_index)
    if user_profile is None:
        return 0.0
    movie_index = movie_id_to_index.get(int(movie_id))
    if movie_index is None:
        return 0.0
    return float(np.dot(user_profile, movie_vectors[movie_index]))


def _semantic_recommendation_results(user_profile, movie_vectors, liked_movie_ids, movie_ids, k=12):
    movie_ids = np.asarray(movie_ids, dtype=int)
    scores = movie_vectors @ user_profile
    mask = ~np.isin(movie_ids, np.asarray(liked_movie_ids, dtype=int))
    filtered_movie_ids = movie_ids[mask]
    filtered_scores = scores[mask]
    ranking = np.argsort(filtered_scores)[::-1][:k]
    ranked_movie_ids = filtered_movie_ids[ranking]
    result_frame = movies[movies['movieId'].isin(ranked_movie_ids)].copy(deep=True)
    score_map = {movie_id: score for movie_id, score in zip(filtered_movie_ids, filtered_scores)}
    result_frame['similarity'] = result_frame['movieId'].map(score_map)
    return result_frame.sort_values(by=['similarity'], ascending=False).head(k)


def _get_movie_time_scores():
    global _MOVIE_TIME_CACHE
    if _MOVIE_TIME_CACHE is not None:
        return _MOVIE_TIME_CACHE

    movie_last_rating = rates.groupby('movieId')['timestamp'].max()
    aligned = movie_last_rating.reindex(movies['movieId']).fillna(movie_last_rating.min())
    min_value = float(movie_last_rating.min())
    max_value = float(movie_last_rating.max())
    if np.isclose(min_value, max_value):
        _MOVIE_TIME_CACHE = np.zeros(len(movies), dtype=float)
    else:
        _MOVIE_TIME_CACHE = ((aligned.astype(float) - min_value) / (max_value - min_value)).to_numpy(dtype=float)
    return _MOVIE_TIME_CACHE

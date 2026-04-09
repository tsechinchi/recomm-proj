import re
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (
    Blueprint, current_app, jsonify, make_response, render_template, render_template_string, request
)

from .tools.data_tool import *
from . import recommender_original as original_system

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover - optional dependency
    Word2Vec = None

from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

bp = Blueprint('main', __name__, url_prefix='/')

_MOVIE_EMBEDDING_CACHE = None
_MOVIE_TIME_CACHE = None


def main():
    from flaskr import create_app
    app = create_app()
    app.run(debug=True)

movies, genres, rates = loadData()


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
    user_genres = request.cookies.get('user_genres')
    if user_genres:
        user_genres = user_genres.split(",")
    else:
        user_genres = []
    user_rates = request.cookies.get('user_rates')
    if user_rates:
        user_rates = user_rates.split(",")
    else:
        user_rates = []
    user_likes = request.cookies.get('user_likes')
    if user_likes:
        user_likes = user_likes.split(",")
    else:
        user_likes = []
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
                                             participant_id=participant_id,
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
        return results.to_dict('records'), "These movies are recommended by an SVD++ + semantic + time-aware hybrid."  # type: ignore
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


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


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

    if Word2Vec is not None:
        tokenized_texts = [_tokenize(text) for text in texts]
        tokenized_texts = [tokens if tokens else ['movie'] for tokens in tokenized_texts]
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=1,
            workers=1,
            sg=1,
            epochs=25,
            seed=42,
        )
        vectors = []
        for tokens in tokenized_texts:
            token_vectors = [model.wv[token] for token in tokens if token in model.wv]
            vectors.append(np.mean(token_vectors, axis=0) if token_vectors else np.zeros(model.vector_size))
        vectors = np.asarray(vectors, dtype=float)
    else:
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

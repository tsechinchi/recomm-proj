import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover - optional dependency
    Word2Vec = None


# Default weight applied to the collaborative-filtering component when
# computing hybrid scores. Value should be in [0.0, 1.0]. Higher values give
# more importance to the collaborative model outputs.
DEFAULT_COLLABORATIVE_WEIGHT = 0.85

# Default weight for the semantic (content-based) component. Typically
# complementary to the collaborative weight; both weights can be tuned.
DEFAULT_SEMANTIC_WEIGHT = 0.15

# Penalty factor applied to item popularity (higher reduces scores for
# more popular items). This is subtracted from the final score.
DEFAULT_POPULARITY_PENALTY = 0.05

# Default parameters passed to the TimeSVDpp recommender. Keys:
# - "n_factors": number of latent factors
# - "n_epochs": number of training epochs
# - "lr_all": base learning rate for factors/biases
# - "reg_all": regularization strength
# - "temporal_epochs": extra epochs for temporal components
# - "temporal_lr": learning rate for temporal components
# - "temporal_reg": regularization for temporal components
# - "time_bin_days": days per time bin for temporal effects
# - "random_state": RNG seed for reproducibility
MODEL_CONFIG = {
    "n_factors": 50,
    "n_epochs": 30,
    "lr_all": 0.005,
    "reg_all": 0.05,
    "temporal_epochs": 4,
    "temporal_lr": 0.003,
    "temporal_reg": 0.02,
    "time_bin_days": 30,
    "random_state": 42,
}

# Grid of collaborative weights used for quick tuning/validation runs.
TUNING_WEIGHT_GRID = [0.8, 0.85, 0.9]

# Maximum number of users to sample during validation to keep evaluation
# runtimes bounded.
MAX_VALIDATION_USERS = 200

# Seed used for reproducible sampling of validation users.
VALIDATION_SAMPLE_SEED = 42

# Internal caches (process-global) to avoid recomputing embeddings or
# recommender instances on repeated requests. These are intentionally
# module-level mutable objects.
_MOVIE_EMBEDDING_CACHE = None
_RECOMMENDER_CACHE = {}

# Limit on how many recommender instances to keep in `_RECOMMENDER_CACHE`.
_RECOMMENDER_CACHE_LIMIT = 6


def movie_text(row):
    genres_text = " ".join(row["genres"]) if isinstance(row.get("genres"), list) else ""
    overview = "" if pd.isna(row.get("overview")) else str(row.get("overview"))
    title = "" if pd.isna(row.get("title")) else str(row.get("title"))
    year = "" if pd.isna(row.get("year")) else str(row.get("year"))
    return f"{title} {year} {genres_text} {overview}".strip().lower()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def l2_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def minmax_scale(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    min_value = arr.min()
    max_value = arr.max()
    if np.isclose(min_value, max_value):
        return np.zeros_like(arr)
    return (arr - min_value) / (max_value - min_value)


def build_popularity_lookup(ratings_df):
    if len(ratings_df) == 0:
        return {}

    popularity = ratings_df.groupby("movieId")["rating"].count().astype(float)
    max_count = float(popularity.max())
    if np.isclose(max_count, 0.0):
        return {int(movie_id): 0.0 for movie_id in popularity.index}
    return {
        int(movie_id): float(count / max_count)
        for movie_id, count in popularity.items()
    }


def build_movie_embeddings(movies):
    texts = movies.apply(movie_text, axis=1).tolist()
    if Word2Vec is not None:
        tokenized_texts = [tokenize(text) or ["movie"] for text in texts]
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
        return l2_normalize(np.asarray(vectors, dtype=float))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    if tfidf_matrix.shape[1] <= 2:
        return l2_normalize(tfidf_matrix.toarray())

    n_components = min(100, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    if n_components < 2:
        return l2_normalize(tfidf_matrix.toarray())

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return l2_normalize(svd.fit_transform(tfidf_matrix))


def build_profile_from_ratings(user_history, movie_vectors, movie_id_to_index):
    rated_vectors = []
    weights = []
    for _, row in user_history.iterrows():
        movie_index = movie_id_to_index.get(int(row["movieId"]))
        if movie_index is None:
            continue
        rated_vectors.append(movie_vectors[movie_index])
        weights.append(max(float(row["rating"]), 0.5))

    if not rated_vectors:
        return None

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    profile = np.average(np.vstack(rated_vectors), axis=0, weights=weights)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return None
    return profile / norm


def build_semantic_profile(movie_ids, movie_vectors, movie_id_to_index):
    profile_vectors = []
    weights = []
    for movie_id in movie_ids:
        movie_index = movie_id_to_index.get(int(movie_id))
        if movie_index is None:
            continue
        profile_vectors.append(movie_vectors[movie_index])
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


def add_synthetic_timestamps(user_rates, latest_timestamp):
    if len(user_rates) == 0:
        return user_rates

    user_rates = user_rates.copy(deep=True)
    user_rates["timestamp"] = np.linspace(
        float(latest_timestamp) + 1.0,
        float(latest_timestamp) + float(len(user_rates)),
        num=len(user_rates),
        dtype=float,
    )
    return user_rates


def get_movie_embedding_cache(movies_df):
    global _MOVIE_EMBEDDING_CACHE
    if _MOVIE_EMBEDDING_CACHE is not None:
        return _MOVIE_EMBEDDING_CACHE

    movie_ids = movies_df["movieId"].astype(int).tolist()
    _MOVIE_EMBEDDING_CACHE = {
        "movie_ids": movie_ids,
        "vectors": build_movie_embeddings(movies_df.copy(deep=True)),
        "movie_id_to_index": {
            movie_id: index for index, movie_id in enumerate(movie_ids)
        },
    }
    return _MOVIE_EMBEDDING_CACHE


def prepare_request_ratings(user_rates, ratings_df):
    from .tools.data_tool import ratesFromUser

    user_rates_df = ratesFromUser(user_rates)
    latest_timestamp = float(ratings_df["timestamp"].max()) if len(ratings_df) > 0 else 0.0
    return add_synthetic_timestamps(user_rates_df, latest_timestamp)


def get_recommender_cache_key(user_rates_df):
    return tuple(
        sorted(
            (
                int(row["movieId"]),
                round(float(row["rating"]), 2),
            )
            for _, row in user_rates_df.iterrows()
        )
    )


def get_cached_recommender(ratings_df, user_rates_df):
    cache_key = get_recommender_cache_key(user_rates_df)
    cached_model = _RECOMMENDER_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model

    from .time_svdpp import TimeSVDppRecommender

    training_rates = pd.concat(
        [
            ratings_df[["userId", "movieId", "rating", "timestamp"]],
            user_rates_df[["userId", "movieId", "rating", "timestamp"]],
        ],
        ignore_index=True,
    )
    model = TimeSVDppRecommender(**MODEL_CONFIG)
    model.fit(training_rates)
    _RECOMMENDER_CACHE[cache_key] = model
    if len(_RECOMMENDER_CACHE) > _RECOMMENDER_CACHE_LIMIT:
        oldest_key = next(iter(_RECOMMENDER_CACHE))
        _RECOMMENDER_CACHE.pop(oldest_key, None)
    return model


def score_hybrid_candidates(
    model,
    user_history,
    candidate_movie_ids,
    reference_timestamp,
    movie_vectors,
    movie_id_to_index,
    popularity_lookup=None,
    collaborative_weight=DEFAULT_COLLABORATIVE_WEIGHT,
    semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
    popularity_penalty=DEFAULT_POPULARITY_PENALTY,
):
    collaborative_scores = model.score_candidates(
        int(user_history["userId"].iloc[0]),
        candidate_movie_ids,
        timestamp=reference_timestamp,
    )
    user_profile = build_profile_from_ratings(user_history, movie_vectors, movie_id_to_index)
    semantic_scores = []
    for movie_id in candidate_movie_ids:
        movie_index = movie_id_to_index.get(int(movie_id))
        semantic_scores.append(
            0.0 if user_profile is None or movie_index is None else float(np.dot(user_profile, movie_vectors[movie_index]))
        )

    popularity_scores = np.asarray(
        [
            0.0 if popularity_lookup is None else popularity_lookup.get(int(movie_id), 0.0)
            for movie_id in candidate_movie_ids
        ],
        dtype=float,
    )

    return (
        collaborative_weight * minmax_scale(collaborative_scores)
        + semantic_weight * minmax_scale(semantic_scores)
        - popularity_penalty * popularity_scores
    )


def get_hybrid_recommendation_results(
    movies_df,
    ratings_df,
    user_rates,
    k=12,
    collaborative_weight=DEFAULT_COLLABORATIVE_WEIGHT,
    semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
):
    results = []
    if len(user_rates) == 0:
        return results

    user_rates_df = prepare_request_ratings(user_rates, ratings_df)
    model = get_cached_recommender(ratings_df, user_rates_df)
    rated_movie_ids = set(user_rates_df["movieId"].tolist())

    embedding_cache = get_movie_embedding_cache(movies_df)
    popularity_lookup = build_popularity_lookup(ratings_df)
    candidate_movie_ids = [
        movie_id for movie_id in movies_df["movieId"].tolist() if movie_id not in rated_movie_ids
    ]
    if not candidate_movie_ids:
        return results

    reference_timestamp = float(user_rates_df["timestamp"].max())
    final_scores = score_hybrid_candidates(
        model,
        user_rates_df,
        candidate_movie_ids,
        reference_timestamp,
        embedding_cache["vectors"],
        embedding_cache["movie_id_to_index"],
        popularity_lookup=popularity_lookup,
        collaborative_weight=collaborative_weight,
        semantic_weight=semantic_weight,
    )
    score_map = {
        movie_id: score for movie_id, score in zip(candidate_movie_ids, final_scores)
    }
    results = movies_df[movies_df["movieId"].isin(candidate_movie_ids)].copy(deep=True)
    results["final_score"] = results["movieId"].map(score_map)
    return results.sort_values(by=["final_score"], ascending=False).head(k)


def get_liked_similar_results(movies_df, user_likes, k=12):
    results = []
    if len(user_likes) == 0:
        return results

    embedding_cache = get_movie_embedding_cache(movies_df)
    liked_movie_ids = [int(movie_id) for movie_id in user_likes]
    user_profile = build_semantic_profile(
        liked_movie_ids,
        embedding_cache["vectors"],
        embedding_cache["movie_id_to_index"],
    )
    if user_profile is None:
        return results

    movie_ids = np.asarray(embedding_cache["movie_ids"], dtype=int)
    scores = embedding_cache["vectors"] @ user_profile
    mask = ~np.isin(movie_ids, np.asarray(liked_movie_ids, dtype=int))
    filtered_movie_ids = movie_ids[mask]
    filtered_scores = scores[mask]
    ranking = np.argsort(filtered_scores)[::-1][:k]
    ranked_movie_ids = filtered_movie_ids[ranking]
    results = movies_df[movies_df["movieId"].isin(ranked_movie_ids)].copy(deep=True)
    score_map = {
        movie_id: score for movie_id, score in zip(filtered_movie_ids, filtered_scores)
    }
    results["similarity"] = results["movieId"].map(score_map)
    return results.sort_values(by=["similarity"], ascending=False).head(k)


def sample_validation_users(validation_users, limit=MAX_VALIDATION_USERS, seed=VALIDATION_SAMPLE_SEED):
    if len(validation_users) <= limit:
        return validation_users

    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(validation_users), size=limit, replace=False))
    return [validation_users[index] for index in selected_indices]

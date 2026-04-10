import json
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from flaskr.evaluation import evaluate_ranking_batch
from flaskr.tools.data_tool import loadData


TOP_K = 20
MIN_USER_HISTORY = 5
POSITIVE_RATING = 4.0


def movie_text(row):
    genres_text = " ".join(row["genres"]) if isinstance(row["genres"], list) else ""
    overview = "" if pd.isna(row.get("overview")) else str(row.get("overview"))
    title = "" if pd.isna(row.get("title")) else str(row.get("title"))
    year = "" if pd.isna(row.get("year")) else str(row.get("year"))
    return f"{title} {year} {genres_text} {overview}".strip().lower()


def l2_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
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


def build_movie_embeddings(movies):
    texts = movies.apply(movie_text, axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] <= 2:
        return l2_normalize(tfidf.toarray())
    n_components = min(100, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if n_components < 2:
        return l2_normalize(tfidf.toarray())
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return l2_normalize(svd.fit_transform(tfidf))


def build_user_profile(user_history, movie_vectors, movie_id_to_index):
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


def build_time_prior(train_ratings, movies):
    movie_last_rating = train_ratings.groupby("movieId")["timestamp"].max()
    aligned = movie_last_rating.reindex(movies["movieId"]).fillna(movie_last_rating.min())
    min_value = float(movie_last_rating.min())
    max_value = float(movie_last_rating.max())
    if np.isclose(min_value, max_value):
        return np.zeros(len(movies), dtype=float)
    return ((aligned.astype(float) - min_value) / (max_value - min_value)).to_numpy(dtype=float)


def temporal_holdout_split(ratings):
    train_parts = []
    eval_users = []

    for user_id, user_frame in ratings.sort_values(["userId", "timestamp"]).groupby("userId"):
        if len(user_frame) < MIN_USER_HISTORY:
            continue

        positives = user_frame[user_frame["rating"] >= POSITIVE_RATING]
        if positives.empty:
            continue

        held_out = positives.sort_values("timestamp").tail(1)
        held_out_movie_id = int(held_out.iloc[0]["movieId"])

        train_frame = user_frame[user_frame["movieId"] != held_out_movie_id]
        if len(train_frame) < MIN_USER_HISTORY - 1:
            continue

        train_parts.append(train_frame)
        eval_users.append(
            {
                "user_id": int(user_id),
                "train_history": train_frame,
                "relevant_items": [held_out_movie_id],
            }
        )

    if not train_parts:
        raise RuntimeError("No eligible users found for temporal evaluation.")

    train_ratings = pd.concat(train_parts, ignore_index=True)
    return train_ratings, eval_users


def build_collaborative_model(train_ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    train_df = train_ratings[["userId", "movieId", "rating"]]
    surprise_data = Dataset.load_from_df(train_df, reader=reader)
    algo = SVDpp(
        n_factors=80,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )
    algo.fit(surprise_data.build_full_trainset())
    return algo


def recommend_for_user(user_id, user_history, movies, movie_vectors, movie_id_to_index, time_prior, algo):
    seen_movie_ids = set(user_history["movieId"].astype(int).tolist())
    user_profile = build_user_profile(user_history, movie_vectors, movie_id_to_index)

    candidate_movie_ids = []
    collaborative_scores = []
    semantic_scores = []

    for movie_id in movies["movieId"].astype(int).tolist():
        if movie_id in seen_movie_ids:
            continue
        candidate_movie_ids.append(movie_id)
        collaborative_scores.append(algo.predict(user_id, movie_id).est)
        movie_index = movie_id_to_index.get(movie_id)
        semantic_scores.append(
            0.0 if user_profile is None or movie_index is None else float(np.dot(user_profile, movie_vectors[movie_index]))
        )

    candidate_indices = [movie_id_to_index[movie_id] for movie_id in candidate_movie_ids]
    time_scores = time_prior[candidate_indices]

    final_scores = (
        0.65 * minmax_scale(collaborative_scores)
        + 0.25 * minmax_scale(semantic_scores)
        + 0.10 * minmax_scale(time_scores)
    )

    ranked_indices = np.argsort(final_scores)[::-1][:TOP_K]
    return [candidate_movie_ids[index] for index in ranked_indices]


def run_evaluation():
    movies, _, ratings = loadData()
    train_ratings, eval_users = temporal_holdout_split(ratings)

    movie_vectors = build_movie_embeddings(movies)
    movie_id_to_index = {int(movie_id): index for index, movie_id in enumerate(movies["movieId"].astype(int).tolist())}
    time_prior = build_time_prior(train_ratings, movies)
    collaborative_model = build_collaborative_model(train_ratings)

    recommendation_lists = []
    relevant_lists = []

    for user in eval_users:
        recommendation_lists.append(
            recommend_for_user(
                user["user_id"],
                user["train_history"],
                movies,
                movie_vectors,
                movie_id_to_index,
                time_prior,
                collaborative_model,
            )
        )
        relevant_lists.append(user["relevant_items"])

    metrics = evaluate_ranking_batch(recommendation_lists, relevant_lists, k=TOP_K)
    summary = {
        "users_evaluated": len(eval_users),
        "split": "temporal holdout (last positive interaction per eligible user)",
        "model": "SVD++ + TF-IDF semantic embeddings + time-aware reranking",
        "metrics": {
            "ndcg@20": metrics["ndcg@20"],
            "map@20": metrics["map@20"],
            "recall@20": metrics["recall@20"],
            "hit_rate@20": metrics["hit_rate@20"],
        },
    }
    return summary


if __name__ == "__main__":
    summary = run_evaluation()
    print(json.dumps(summary, indent=2))

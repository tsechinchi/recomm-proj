import json
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from flaskr.evaluation import evaluate_ranking_batch
from flaskr.time_svdpp import TimeSVDppRecommender
from flaskr.tools.data_tool import loadData

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover - optional dependency
    Word2Vec = None


TOP_K = 20
MIN_USER_HISTORY = 5
POSITIVE_RATING = 4.0
WEIGHT_GRID = [round(weight, 2) for weight in np.linspace(0.0, 1.0, 11)]


def movie_text(row):
    genres_text = " ".join(row["genres"]) if isinstance(row["genres"], list) else ""
    overview = "" if pd.isna(row.get("overview")) else str(row.get("overview"))
    title = "" if pd.isna(row.get("title")) else str(row.get("title"))
    year = "" if pd.isna(row.get("year")) else str(row.get("year"))
    return f"{title} {year} {genres_text} {overview}".strip().lower()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


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
    if Word2Vec is not None:
        tokenized = [tokenize(text) or ["movie"] for text in texts]
        model = Word2Vec(
            sentences=tokenized,
            vector_size=100,
            window=5,
            min_count=1,
            workers=1,
            sg=1,
            epochs=25,
            seed=42,
        )
        vectors = []
        for tokens in tokenized:
            token_vectors = [model.wv[token] for token in tokens if token in model.wv]
            vectors.append(np.mean(token_vectors, axis=0) if token_vectors else np.zeros(model.vector_size))
        return l2_normalize(np.asarray(vectors, dtype=float))

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


def temporal_validation_test_split(ratings):
    train_parts = []
    validation_users = []
    test_users = []

    ordered_ratings = ratings.sort_values(["userId", "timestamp"])
    for user_id, user_frame in ordered_ratings.groupby("userId"):
        if len(user_frame) < MIN_USER_HISTORY:
            continue

        positives = user_frame[user_frame["rating"] >= POSITIVE_RATING]
        if len(positives) < 2:
            continue

        validation_row = positives.tail(2).head(1)
        test_row = positives.tail(1)
        validation_movie_id = int(validation_row.iloc[0]["movieId"])
        validation_timestamp = float(validation_row.iloc[0]["timestamp"])
        test_movie_id = int(test_row.iloc[0]["movieId"])
        test_timestamp = float(test_row.iloc[0]["timestamp"])

        held_out_movie_ids = {validation_movie_id, test_movie_id}
        train_frame = user_frame[~user_frame["movieId"].isin(held_out_movie_ids)]
        if len(train_frame) < MIN_USER_HISTORY - 1:
            continue

        train_parts.append(train_frame)
        validation_users.append(
            {
                "user_id": int(user_id),
                "seen_movie_ids": set(train_frame["movieId"].astype(int).tolist()),
                "reference_timestamp": validation_timestamp,
                "train_history": train_frame,
                "relevant_items": [validation_movie_id],
            }
        )
        validation_seen = set(train_frame["movieId"].astype(int).tolist())
        validation_seen.add(validation_movie_id)
        test_history = pd.concat([train_frame, validation_row], ignore_index=True)
        test_users.append(
            {
                "user_id": int(user_id),
                "seen_movie_ids": validation_seen,
                "reference_timestamp": test_timestamp,
                "train_history": test_history,
                "relevant_items": [test_movie_id],
            }
        )

    if not train_parts:
        raise RuntimeError("No eligible users found for temporal evaluation.")

    train_ratings = pd.concat(train_parts, ignore_index=True)
    return train_ratings, validation_users, test_users


def build_rankings(model, user_groups, all_movie_ids, movie_vectors, movie_id_to_index, collaborative_weight):
    recommendation_lists = []
    relevant_lists = []
    semantic_weight = 1.0 - collaborative_weight

    for user in user_groups:
        candidate_movie_ids = [
            movie_id for movie_id in all_movie_ids if movie_id not in user["seen_movie_ids"]
        ]
        collaborative_scores = model.score_candidates(
            user["user_id"],
            candidate_movie_ids,
            timestamp=user["reference_timestamp"],
        )
        user_profile = build_user_profile(
            user["train_history"],
            movie_vectors,
            movie_id_to_index,
        )
        semantic_scores = []
        for movie_id in candidate_movie_ids:
            movie_index = movie_id_to_index.get(movie_id)
            semantic_scores.append(
                0.0 if user_profile is None or movie_index is None else float(np.dot(user_profile, movie_vectors[movie_index]))
            )
        final_scores = (
            collaborative_weight * minmax_scale(collaborative_scores)
            + semantic_weight * minmax_scale(semantic_scores)
        )
        ranking = np.argsort(final_scores)[::-1][:TOP_K]
        recommendation_lists.append(
            [candidate_movie_ids[index] for index in ranking]
        )
        relevant_lists.append(user["relevant_items"])

    return recommendation_lists, relevant_lists


def tune_fusion_weight(model, validation_users, all_movie_ids, movie_vectors, movie_id_to_index):
    best_weight = None
    best_metrics = None

    for collaborative_weight in WEIGHT_GRID:
        recommendation_lists, relevant_lists = build_rankings(
            model,
            validation_users,
            all_movie_ids,
            movie_vectors,
            movie_id_to_index,
            collaborative_weight,
        )
        metrics = evaluate_ranking_batch(recommendation_lists, relevant_lists, k=TOP_K)
        if best_metrics is None or metrics["ndcg@20"] > best_metrics["ndcg@20"]:
            best_weight = collaborative_weight
            best_metrics = metrics

    return best_weight, best_metrics


def run_evaluation():
    movies, _, ratings = loadData()
    train_ratings, validation_users, test_users = temporal_validation_test_split(ratings)

    model = TimeSVDppRecommender(
        n_factors=80,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        temporal_epochs=8,
        temporal_lr=0.003,
        temporal_reg=0.02,
        time_bin_days=30,
        random_state=42,
    )
    model.fit(train_ratings[["userId", "movieId", "rating", "timestamp"]])

    all_movie_ids = movies["movieId"].astype(int).tolist()
    movie_vectors = build_movie_embeddings(movies)
    movie_id_to_index = {
        int(movie_id): index for index, movie_id in enumerate(all_movie_ids)
    }
    best_weight, validation_metrics = tune_fusion_weight(
        model,
        validation_users,
        all_movie_ids,
        movie_vectors,
        movie_id_to_index,
    )
    recommendation_lists, relevant_lists = build_rankings(
        model,
        test_users,
        all_movie_ids,
        movie_vectors,
        movie_id_to_index,
        best_weight,
    )
    metrics = evaluate_ranking_batch(recommendation_lists, relevant_lists, k=TOP_K)
    return {
        "users_evaluated": len(test_users),
        "split": "temporal train/validation/test split (last 2 positive interactions per eligible user)",
        "model": "TimeSVD++-style recommender with movie text embeddings",
        "tuned_weights": {
            "collaborative": best_weight,
            "semantic": round(1.0 - best_weight, 2),
        },
        "validation_metrics": {
            "ndcg@20": validation_metrics["ndcg@20"],
            "map@20": validation_metrics["map@20"],
            "recall@20": validation_metrics["recall@20"],
            "hit_rate@20": validation_metrics["hit_rate@20"],
        },
        "metrics": {
            "ndcg@20": metrics["ndcg@20"],
            "map@20": metrics["map@20"],
            "recall@20": metrics["recall@20"],
            "hit_rate@20": metrics["hit_rate@20"],
        },
    }


if __name__ == "__main__":
    summary = run_evaluation()
    print(json.dumps(summary, indent=2))

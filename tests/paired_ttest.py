import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from surprise import KNNWithMeans

from flaskr.evaluation import average_precision_at_k
from flaskr.evaluation import hit_rate_at_k
from flaskr.evaluation import ndcg_at_k
from flaskr.evaluation import recall_at_k
from flaskr.tools.data_tool import loadData

from flaskr.main import (
    _build_user_profile_from_ratings,
    _get_movie_embedding_cache,
    _get_movie_time_scores,
    _minmax_scale,
)


TOP_K = 20
MIN_USER_HISTORY = 5
POSITIVE_RATING = 4.0


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


def build_original_model(train_ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    train_df = train_ratings[["userId", "movieId", "rating"]]
    surprise_data = Dataset.load_from_df(train_df, reader=reader)
    algo = KNNWithMeans(
        sim_options={"name": "pearson", "user_based": True},
        k=40,
        min_k=1,
    )
    algo.fit(surprise_data.build_full_trainset())
    return algo


def build_enhanced_model(train_ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    training_rates = train_ratings[["userId", "movieId", "rating"]]
    surprise_data = Dataset.load_from_df(training_rates, reader=reader)
    algo = SVDpp(
        n_factors=80,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )
    algo.fit(surprise_data.build_full_trainset())
    return algo


def recommend_original_for_user(user_id, user_history, movies, algo):
    seen_movie_ids = set(user_history["movieId"].astype(int).tolist())
    candidate_movie_ids = []
    collaborative_scores = []

    for movie_id in movies["movieId"].astype(int).tolist():
        if movie_id in seen_movie_ids:
            continue
        candidate_movie_ids.append(movie_id)
        collaborative_scores.append(algo.predict(user_id, movie_id).est)

    ranked_indices = np.argsort(collaborative_scores)[::-1][:TOP_K]
    return [candidate_movie_ids[index] for index in ranked_indices]


def recommend_enhanced_for_user(user_id, user_history, movies, algo, movie_vectors, movie_id_to_index, time_scores):
    seen_movie_ids = set(user_history["movieId"].astype(int).tolist())
    user_profile = _build_user_profile_from_ratings(user_history, movie_vectors, movie_id_to_index)

    candidate_movie_ids = []
    cf_scores = []
    semantic_scores = []

    for movie_id in movies["movieId"].astype(int).tolist():
        if movie_id in seen_movie_ids:
            continue
        candidate_movie_ids.append(movie_id)
        cf_scores.append(algo.predict(user_id, movie_id).est)
        movie_index = movie_id_to_index.get(movie_id)
        semantic_scores.append(
            0.0 if user_profile is None or movie_index is None else float(np.dot(user_profile, movie_vectors[movie_index]))
        )

    if not candidate_movie_ids:
        return []

    candidate_indices = [movie_id_to_index[movie_id] for movie_id in candidate_movie_ids]
    time_candidate_scores = time_scores[candidate_indices]
    final_scores = (
        0.65 * _minmax_scale(cf_scores)
        + 0.25 * _minmax_scale(semantic_scores)
        + 0.10 * _minmax_scale(time_candidate_scores)
    )
    ranked_indices = np.argsort(final_scores)[::-1][:TOP_K]
    return [candidate_movie_ids[index] for index in ranked_indices]


def run_paired_ttest():
    movies, _, ratings = loadData()
    train_ratings, eval_users = temporal_holdout_split(ratings)

    original_model = build_original_model(train_ratings)
    movie_cache = _get_movie_embedding_cache()
    movie_vectors = movie_cache["vectors"]
    movie_id_to_index = movie_cache["movie_id_to_index"]
    time_scores = _get_movie_time_scores()

    enhanced_model = build_enhanced_model(train_ratings)

    per_user_original = {"ndcg": [], "map": [], "recall": [], "hit_rate": []}
    per_user_enhanced = {"ndcg": [], "map": [], "recall": [], "hit_rate": []}

    for user in eval_users:
        relevant = user["relevant_items"]
        original_recommendations = recommend_original_for_user(
            user["user_id"],
            user["train_history"],
            movies,
            original_model,
        )
        enhanced_recommendations = recommend_enhanced_for_user(
            user["user_id"],
            user["train_history"],
            movies,
            enhanced_model,
            movie_vectors,
            movie_id_to_index,
            time_scores,
        )

        per_user_original["ndcg"].append(ndcg_at_k(original_recommendations, relevant, TOP_K))
        per_user_original["map"].append(average_precision_at_k(original_recommendations, relevant, TOP_K))
        per_user_original["recall"].append(recall_at_k(original_recommendations, relevant, TOP_K))
        per_user_original["hit_rate"].append(hit_rate_at_k(original_recommendations, relevant, TOP_K))

        per_user_enhanced["ndcg"].append(ndcg_at_k(enhanced_recommendations, relevant, TOP_K))
        per_user_enhanced["map"].append(average_precision_at_k(enhanced_recommendations, relevant, TOP_K))
        per_user_enhanced["recall"].append(recall_at_k(enhanced_recommendations, relevant, TOP_K))
        per_user_enhanced["hit_rate"].append(hit_rate_at_k(enhanced_recommendations, relevant, TOP_K))

    results = {}
    for metric_name in ["ndcg", "map", "recall", "hit_rate"]:
        stat = ttest_rel(per_user_enhanced[metric_name], per_user_original[metric_name], nan_policy="omit")
        results[f"{metric_name}@{TOP_K}"] = {
            "original_mean": float(np.mean(per_user_original[metric_name])),
            "enhanced_mean": float(np.mean(per_user_enhanced[metric_name])),
            "mean_diff": float(np.mean(np.asarray(per_user_enhanced[metric_name]) - np.asarray(per_user_original[metric_name]))),
            "t_statistic": float(stat.statistic),
            "p_value": float(stat.pvalue),
        }

    return {
        "users_evaluated": len(eval_users),
        "test": "paired t-test",
        "split": "temporal holdout (same users for Route A and Route B)",
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_paired_ttest(), indent=2))

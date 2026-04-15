import json
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans

from flaskr.evaluation import evaluate_ranking_batch
from flaskr.tools.data_tool import loadData


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


def build_collaborative_model(train_ratings):
    """Build KNNWithMeans collaborative filtering model (original algorithm approach)"""
    reader = Reader(rating_scale=(0.5, 5.0))
    train_df = train_ratings[["userId", "movieId", "rating"]]
    surprise_data = Dataset.load_from_df(train_df, reader=reader)
    algo = KNNWithMeans(
        sim_options={'name': 'pearson', 'user_based': True},
        k=40,
        min_k=1,
    )
    algo.fit(surprise_data.build_full_trainset())
    return algo


def recommend_for_user(
    user_id,
    user_history,
    movies,
    algo,
):
    """Generate recommendations using original KNNWithMeans collaborative filtering only"""
    seen_movie_ids = set(user_history["movieId"].astype(int).tolist())

    candidate_movie_ids = []
    collaborative_scores = []

    for movie_id in movies["movieId"].astype(int).tolist():
        if movie_id in seen_movie_ids:
            continue
        candidate_movie_ids.append(movie_id)
        collaborative_scores.append(algo.predict(user_id, movie_id).est)

    # Rank by collaborative score only
    ranked_indices = np.argsort(collaborative_scores)[::-1][:TOP_K]
    return [candidate_movie_ids[index] for index in ranked_indices]


def run_evaluation():
    movies, _, ratings = loadData()
    train_ratings, eval_users = temporal_holdout_split(ratings)

    # Build collaborative model only (original algorithm)
    collaborative_model = build_collaborative_model(train_ratings)

    recommendation_lists = []
    relevant_lists = []

    for user in eval_users:
        recommendation_lists.append(
            recommend_for_user(
                user["user_id"],
                user["train_history"],
                movies,
                collaborative_model,
            )
        )
        relevant_lists.append(user["relevant_items"])

    metrics = evaluate_ranking_batch(recommendation_lists, relevant_lists, k=TOP_K)
    summary = {
        "users_evaluated": len(eval_users),
        "split": "temporal holdout (last positive interaction per eligible user)",
        "model": "KNNWithMeans collaborative filtering",
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

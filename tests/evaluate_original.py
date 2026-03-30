import json

import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import Reader

from flaskr.evaluation import evaluate_ranking_batch
from flaskr.tools.data_tool import loadData


TOP_K = 20
MIN_USER_HISTORY = 5
POSITIVE_RATING = 4.0
SYNTHETIC_USER_ID = 611


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

        train_frame = user_frame[~user_frame["movieId"].isin({validation_movie_id, test_movie_id})]
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

        test_history = pd.concat([train_frame, validation_row], ignore_index=True)
        test_seen_movie_ids = set(train_frame["movieId"].astype(int).tolist())
        test_seen_movie_ids.add(validation_movie_id)
        test_users.append(
            {
                "user_id": int(user_id),
                "seen_movie_ids": test_seen_movie_ids,
                "reference_timestamp": test_timestamp,
                "train_history": test_history,
                "relevant_items": [test_movie_id],
            }
        )

    if not train_parts:
        raise RuntimeError("No eligible users found for temporal evaluation.")

    return pd.concat(train_parts, ignore_index=True), validation_users, test_users


def build_synthetic_user_history(user_history):
    synthetic_history = user_history[["movieId", "rating"]].copy(deep=True)
    synthetic_history["userId"] = SYNTHETIC_USER_ID
    return synthetic_history[["userId", "movieId", "rating"]]


def fit_original_model(train_ratings, user_history, original_user_id):
    reader = Reader(rating_scale=(0.5, 5.0))
    synthetic_history = build_synthetic_user_history(user_history)
    base_ratings = train_ratings[train_ratings["userId"] != int(original_user_id)]
    training_df = pd.concat(
        [base_ratings[["userId", "movieId", "rating"]], synthetic_history],
        ignore_index=True,
    )
    surprise_data = Dataset.load_from_df(training_df, reader=reader)
    algo = KNNWithMeans(sim_options={"name": "pearson", "user_based": True})
    algo.fit(surprise_data.build_full_trainset())
    return algo


def build_rankings(train_ratings, user_groups, all_movie_ids):
    recommendation_lists = []
    relevant_lists = []

    for user in user_groups:
        model = fit_original_model(
            train_ratings,
            user["train_history"],
            user["user_id"],
        )
        candidate_movie_ids = [
            movie_id for movie_id in all_movie_ids if movie_id not in user["seen_movie_ids"]
        ]
        predictions = [model.predict(SYNTHETIC_USER_ID, movie_id) for movie_id in candidate_movie_ids]
        ranking = np.argsort([prediction.est for prediction in predictions])[::-1][:TOP_K]
        recommendation_lists.append([candidate_movie_ids[index] for index in ranking])
        relevant_lists.append(user["relevant_items"])

    return recommendation_lists, relevant_lists


def run_evaluation():
    movies, _, ratings = loadData()
    train_ratings, validation_users, test_users = temporal_validation_test_split(ratings)

    all_movie_ids = movies["movieId"].astype(int).tolist()

    validation_recommendations, validation_relevant = build_rankings(
        train_ratings,
        validation_users,
        all_movie_ids,
    )
    validation_metrics = evaluate_ranking_batch(
        validation_recommendations,
        validation_relevant,
        k=TOP_K,
    )

    recommendation_lists, relevant_lists = build_rankings(
        train_ratings,
        test_users,
        all_movie_ids,
    )
    metrics = evaluate_ranking_batch(recommendation_lists, relevant_lists, k=TOP_K)

    return {
        "users_evaluated": len(test_users),
        "split": "temporal train/validation/test split (last 2 positive interactions per eligible user)",
        "model": "Original demo recommender (KNNWithMeans user-based collaborative filtering)",
        "validation_users_total": len(validation_users),
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
    print(json.dumps(run_evaluation(), indent=2))

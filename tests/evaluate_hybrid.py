import json

import numpy as np
import pandas as pd

from flaskr.evaluation import evaluate_ranking_batch
from flaskr.hybrid_support import (
    MODEL_CONFIG,
    TUNING_WEIGHT_GRID,
    build_movie_embeddings,
    build_popularity_lookup,
    score_hybrid_candidates,
    sample_validation_users,
)
from flaskr.time_svdpp import TimeSVDppRecommender
from flaskr.tools.data_tool import loadData


TOP_K = 20
MIN_USER_HISTORY = 5
POSITIVE_RATING = 4.0


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


def build_rankings(
    model,
    user_groups,
    all_movie_ids,
    movie_vectors,
    movie_id_to_index,
    popularity_lookup,
    collaborative_weight,
):
    recommendation_lists = []
    relevant_lists = []
    semantic_weight = round(1.0 - collaborative_weight, 2)

    for user in user_groups:
        candidate_movie_ids = [
            movie_id for movie_id in all_movie_ids if movie_id not in user["seen_movie_ids"]
        ]
        final_scores = score_hybrid_candidates(
            model,
            user["train_history"],
            candidate_movie_ids,
            user["reference_timestamp"],
            movie_vectors,
            movie_id_to_index,
            popularity_lookup=popularity_lookup,
            collaborative_weight=collaborative_weight,
            semantic_weight=semantic_weight,
        )
        ranking = np.argsort(final_scores)[::-1][:TOP_K]
        recommendation_lists.append([candidate_movie_ids[index] for index in ranking])
        relevant_lists.append(user["relevant_items"])

    return recommendation_lists, relevant_lists


def tune_fusion_weight(
    model,
    validation_users,
    all_movie_ids,
    movie_vectors,
    movie_id_to_index,
    popularity_lookup,
):
    sampled_validation_users = sample_validation_users(validation_users)
    best_weight = None
    best_metrics = None

    for collaborative_weight in TUNING_WEIGHT_GRID:
        recommendation_lists, relevant_lists = build_rankings(
            model,
            sampled_validation_users,
            all_movie_ids,
            movie_vectors,
            movie_id_to_index,
            popularity_lookup,
            collaborative_weight,
        )
        metrics = evaluate_ranking_batch(recommendation_lists, relevant_lists, k=TOP_K)
        if best_metrics is None or metrics["ndcg@20"] > best_metrics["ndcg@20"]:
            best_weight = collaborative_weight
            best_metrics = metrics

    return best_weight, best_metrics, len(sampled_validation_users)


def run_evaluation():
    movies, _, ratings = loadData()
    train_ratings, validation_users, test_users = temporal_validation_test_split(ratings)

    model = TimeSVDppRecommender(**MODEL_CONFIG)
    model.fit(train_ratings[["userId", "movieId", "rating", "timestamp"]])

    all_movie_ids = movies["movieId"].astype(int).tolist()
    movie_vectors = build_movie_embeddings(movies)
    movie_id_to_index = {int(movie_id): index for index, movie_id in enumerate(all_movie_ids)}
    popularity_lookup = build_popularity_lookup(train_ratings)

    best_weight, validation_metrics, sampled_validation_count = tune_fusion_weight(
        model,
        validation_users,
        all_movie_ids,
        movie_vectors,
        movie_id_to_index,
        popularity_lookup,
    )
    recommendation_lists, relevant_lists = build_rankings(
        model,
        test_users,
        all_movie_ids,
        movie_vectors,
        movie_id_to_index,
        popularity_lookup,
        best_weight,
    )
    metrics = evaluate_ranking_batch(recommendation_lists, relevant_lists, k=TOP_K)

    return {
        "users_evaluated": len(test_users),
        "split": "temporal train/validation/test split (last 2 positive interactions per eligible user)",
        "model": "TimeSVD++-style recommender with movie text embeddings and popularity penalty",
        "tuned_weights": {
            "collaborative": best_weight,
            "semantic": round(1.0 - best_weight, 2),
        },
        "validation_users_total": len(validation_users),
        "validation_users_sampled": sampled_validation_count,
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

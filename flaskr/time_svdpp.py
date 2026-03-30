import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp


SECONDS_PER_DAY = 86400.0


class TimeSVDppRecommender:
    def __init__(
        self,
        n_factors=80,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        temporal_epochs=8,
        temporal_lr=0.003,
        temporal_reg=0.02,
        time_bin_days=30,
        random_state=42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.temporal_epochs = temporal_epochs
        self.temporal_lr = temporal_lr
        self.temporal_reg = temporal_reg
        self.time_bin_days = time_bin_days
        self.random_state = random_state
        self.algo = None
        self.global_max_timestamp = None
        self.global_min_timestamp = None
        self.user_mean_timestamp = {}
        self.user_time_drift = {}
        self.user_bin_bias = {}
        self.item_bin_bias = {}

    def fit(self, ratings_df):
        ratings_df = ratings_df.copy(deep=True)
        if "timestamp" not in ratings_df.columns:
            raise ValueError("ratings_df must include a timestamp column")

        ratings_df["timestamp"] = ratings_df["timestamp"].astype(float)
        ratings_df["userId"] = ratings_df["userId"].astype(int)
        ratings_df["movieId"] = ratings_df["movieId"].astype(int)
        ratings_df["rating"] = ratings_df["rating"].astype(float)

        reader = Reader(rating_scale=(0.5, 5.0))
        training_df = ratings_df[["userId", "movieId", "rating"]]
        surprise_data = Dataset.load_from_df(training_df, reader=reader)
        self.algo = SVDpp(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        self.algo.fit(surprise_data.build_full_trainset())

        self.global_min_timestamp = float(ratings_df["timestamp"].min())
        self.global_max_timestamp = float(ratings_df["timestamp"].max())
        self.user_mean_timestamp = (
            ratings_df.groupby("userId")["timestamp"].mean().astype(float).to_dict()
        )
        self.user_time_drift = {int(user_id): 0.0 for user_id in self.user_mean_timestamp}
        self.user_bin_bias = {}
        self.item_bin_bias = {}

        rating_rows = ratings_df.sort_values(["timestamp", "userId", "movieId"]).to_dict("records")
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.temporal_epochs):
            rng.shuffle(rating_rows)
            for row in rating_rows:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])
                timestamp = float(row["timestamp"])

                base_estimate = float(self.algo.predict(user_id, movie_id).est)
                user_dev = self._time_deviation(user_id, timestamp)
                bin_id = self._time_bin(timestamp)

                drift = self.user_time_drift.get(user_id, 0.0)
                user_key = (user_id, bin_id)
                item_key = (movie_id, bin_id)
                user_bias = self.user_bin_bias.get(user_key, 0.0)
                item_bias = self.item_bin_bias.get(item_key, 0.0)

                estimate = base_estimate + drift * user_dev + user_bias + item_bias
                err = rating - estimate

                self.user_time_drift[user_id] = drift + self.temporal_lr * (
                    err * user_dev - self.temporal_reg * drift
                )
                self.user_bin_bias[user_key] = user_bias + self.temporal_lr * (
                    err - self.temporal_reg * user_bias
                )
                self.item_bin_bias[item_key] = item_bias + self.temporal_lr * (
                    err - self.temporal_reg * item_bias
                )

        return self

    def predict(self, user_id, movie_id, timestamp=None):
        if self.algo is None:
            raise RuntimeError("Model must be fitted before prediction")

        user_id = int(user_id)
        movie_id = int(movie_id)
        if timestamp is None:
            timestamp = self.global_max_timestamp
        timestamp = float(timestamp)

        base_estimate = float(self.algo.predict(user_id, movie_id).est)
        user_dev = self._time_deviation(user_id, timestamp)
        bin_id = self._time_bin(timestamp)

        return (
            base_estimate
            + self.user_time_drift.get(user_id, 0.0) * user_dev
            + self.user_bin_bias.get((user_id, bin_id), 0.0)
            + self.item_bin_bias.get((movie_id, bin_id), 0.0)
        )

    def rank(self, user_id, candidate_movie_ids, timestamp=None, top_k=20):
        candidate_movie_ids = [int(movie_id) for movie_id in candidate_movie_ids]
        if not candidate_movie_ids:
            return []

        scores = np.asarray(
            [self.predict(user_id, movie_id, timestamp=timestamp) for movie_id in candidate_movie_ids],
            dtype=float,
        )
        order = np.argsort(scores)[::-1][:top_k]
        return [candidate_movie_ids[index] for index in order]

    def score_candidates(self, user_id, candidate_movie_ids, timestamp=None):
        candidate_movie_ids = [int(movie_id) for movie_id in candidate_movie_ids]
        scores = [self.predict(user_id, movie_id, timestamp=timestamp) for movie_id in candidate_movie_ids]
        return np.asarray(scores, dtype=float)

    def _time_bin(self, timestamp):
        timestamp = float(timestamp)
        bin_size = max(float(self.time_bin_days) * SECONDS_PER_DAY, SECONDS_PER_DAY)
        return int((timestamp - self.global_min_timestamp) // bin_size)

    def _time_deviation(self, user_id, timestamp):
        mean_timestamp = self.user_mean_timestamp.get(int(user_id))
        if mean_timestamp is None:
            mean_timestamp = self.global_max_timestamp
        delta_days = (float(timestamp) - float(mean_timestamp)) / SECONDS_PER_DAY
        if np.isclose(delta_days, 0.0):
            return 0.0
        return np.sign(delta_days) * (abs(delta_days) ** 0.4)

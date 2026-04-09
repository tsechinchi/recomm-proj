from __future__ import annotations

from collections import defaultdict

import numpy as np
from surprise import AlgoBase
from surprise.prediction_algorithms.predictions import Prediction
from surprise.prediction_algorithms.predictions import PredictionImpossible


class TimeSVDpp(AlgoBase):
    def __init__(
        self,
        n_factors=40,
        n_epochs=10,
        lr_all=0.0005,
        reg_all=0.05,
        n_bins=10,
        beta=0.5,
        random_state=42,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.n_bins = n_bins
        self.beta = beta
        self.random_state = random_state
        self.max_abs_dev = 10.0
        self.max_abs_err = 10.0
        self.max_abs_grad = 5.0
        self.max_abs_param = 10.0

    def fit(self, trainset, timestamps=None):
        AlgoBase.fit(self, trainset)
        self._fit_model(timestamps or {})
        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False, timestamp=None):
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = "UKN__" + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = "UKN__" + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid, timestamp=timestamp)
            if clip:
                lower_bound, upper_bound = self.trainset.rating_scale
                est = min(upper_bound, max(lower_bound, est))
            details["was_impossible"] = False
        except PredictionImpossible as exc:
            est = self.default_prediction()
            details["was_impossible"] = True
            details["reason"] = str(exc)

        pred = Prediction(uid, iid, r_ui, est, details)
        if verbose:
            print(pred)
        return pred

    def estimate(self, u, i, timestamp=None):
        user_known = self.trainset.knows_user(u) if isinstance(u, int) else False
        item_known = self.trainset.knows_item(i) if isinstance(i, int) else False

        if not user_known and not item_known:
            raise PredictionImpossible("User and item are unknown.")

        ts = self._resolve_prediction_timestamp(u, timestamp)
        bin_index = self._time_bin(ts)

        baseline = self.trainset.global_mean

        if item_known:
            baseline += self.bi[i] + self.bi_bin[i, bin_index]

        if user_known:
            dev = self._dev(u, ts)
            baseline += self.bu[u] + self.au[u] * dev + self.bu_bin[u, bin_index]
            user_state = self.pu[u] + self.alpha_pu[u] * dev + self._implicit_sum(u)
        else:
            user_state = np.zeros(self.n_factors, dtype=float)

        if not item_known:
            return baseline

        return baseline + float(np.dot(self.qi[i], user_state))

    def _fit_model(self, timestamps):
        rng = np.random.default_rng(self.random_state)
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        self.pu = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.qi = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.yj = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.alpha_pu = np.zeros((n_users, self.n_factors), dtype=float)

        self.bu = np.zeros(n_users, dtype=float)
        self.bi = np.zeros(n_items, dtype=float)
        self.au = np.zeros(n_users, dtype=float)
        self.bu_bin = np.zeros((n_users, self.n_bins), dtype=float)
        self.bi_bin = np.zeros((n_items, self.n_bins), dtype=float)

        rating_entries = []
        per_user_times = defaultdict(list)

        for u, i, r in self.trainset.all_ratings():
            raw_u = int(self.trainset.to_raw_uid(u))
            raw_i = int(self.trainset.to_raw_iid(i))
            timestamp = timestamps.get((raw_u, raw_i))
            if timestamp is None:
                timestamp = timestamps.get((str(raw_u), str(raw_i)), 0)
            ts_days = float(timestamp) / 86400.0 if timestamp is not None else 0.0
            rating_entries.append((u, i, float(r), ts_days))
            per_user_times[u].append(ts_days)

        if rating_entries:
            all_times = np.asarray([entry[3] for entry in rating_entries], dtype=float)
            self.min_time = float(all_times.min())
            self.max_time = float(all_times.max())
        else:
            self.min_time = 0.0
            self.max_time = 1.0

        self.user_mean_time = np.full(n_users, self.min_time, dtype=float)
        for u, user_times in per_user_times.items():
            self.user_mean_time[u] = float(np.mean(user_times))

        self.user_rated_items = {
            u: [i for i, _ in ratings]
            for u, ratings in self.trainset.ur.items()
        }
        self.user_norm = {
            u: 1.0 / np.sqrt(len(items)) if items else 0.0
            for u, items in self.user_rated_items.items()
        }

        for _ in range(self.n_epochs):
            rng.shuffle(rating_entries)
            for u, i, rating, ts in rating_entries:
                bin_index = self._time_bin(ts)
                dev = self._dev(u, ts)
                dev = float(np.clip(dev, -self.max_abs_dev, self.max_abs_dev))

                implicit_sum = self._implicit_sum(u)
                user_state = self.pu[u] + self.alpha_pu[u] * dev + implicit_sum
                qi_old = self.qi[i].copy()

                pred = (
                    self.trainset.global_mean
                    + self.bu[u]
                    + self.au[u] * dev
                    + self.bu_bin[u, bin_index]
                    + self.bi[i]
                    + self.bi_bin[i, bin_index]
                    + float(np.dot(qi_old, user_state))
                )
                err = rating - pred
                if not np.isfinite(err):
                    continue
                err = float(np.clip(err, -self.max_abs_err, self.max_abs_err))

                grad_bu = np.clip(err - self.reg_all * self.bu[u], -self.max_abs_grad, self.max_abs_grad)
                grad_au = np.clip(err * dev - self.reg_all * self.au[u], -self.max_abs_grad, self.max_abs_grad)
                grad_bu_bin = np.clip(err - self.reg_all * self.bu_bin[u, bin_index], -self.max_abs_grad, self.max_abs_grad)
                grad_bi = np.clip(err - self.reg_all * self.bi[i], -self.max_abs_grad, self.max_abs_grad)
                grad_bi_bin = np.clip(err - self.reg_all * self.bi_bin[i, bin_index], -self.max_abs_grad, self.max_abs_grad)

                self.bu[u] += self.lr_all * grad_bu
                self.au[u] += self.lr_all * grad_au
                self.bu_bin[u, bin_index] += self.lr_all * grad_bu_bin
                self.bi[i] += self.lr_all * grad_bi
                self.bi_bin[i, bin_index] += self.lr_all * grad_bi_bin

                grad_qi = np.clip(err * user_state - self.reg_all * qi_old, -self.max_abs_grad, self.max_abs_grad)
                grad_pu = np.clip(err * qi_old - self.reg_all * self.pu[u], -self.max_abs_grad, self.max_abs_grad)
                grad_alpha_pu = np.clip(err * qi_old * dev - self.reg_all * self.alpha_pu[u], -self.max_abs_grad, self.max_abs_grad)

                self.qi[i] += self.lr_all * grad_qi
                self.pu[u] += self.lr_all * grad_pu
                self.alpha_pu[u] += self.lr_all * grad_alpha_pu

                norm = self.user_norm.get(u, 0.0)
                if norm > 0.0:
                    implicit_gradient = np.clip(err * qi_old * norm, -self.max_abs_grad, self.max_abs_grad)
                    for j in self.user_rated_items.get(u, []):
                        grad_yj = np.clip(implicit_gradient - self.reg_all * self.yj[j], -self.max_abs_grad, self.max_abs_grad)
                        self.yj[j] += self.lr_all * grad_yj

                self.bu[u] = float(np.clip(self.bu[u], -self.max_abs_param, self.max_abs_param))
                self.au[u] = float(np.clip(self.au[u], -self.max_abs_param, self.max_abs_param))
                self.bi[i] = float(np.clip(self.bi[i], -self.max_abs_param, self.max_abs_param))
                self.bu_bin[u, bin_index] = float(np.clip(self.bu_bin[u, bin_index], -self.max_abs_param, self.max_abs_param))
                self.bi_bin[i, bin_index] = float(np.clip(self.bi_bin[i, bin_index], -self.max_abs_param, self.max_abs_param))
                self.pu[u] = np.clip(self.pu[u], -self.max_abs_param, self.max_abs_param)
                self.qi[i] = np.clip(self.qi[i], -self.max_abs_param, self.max_abs_param)
                self.alpha_pu[u] = np.clip(self.alpha_pu[u], -self.max_abs_param, self.max_abs_param)

    def _resolve_prediction_timestamp(self, u, timestamp):
        if timestamp is not None:
            return float(timestamp) / 86400.0
        if isinstance(u, int) and self.trainset.knows_user(u):
            return float(self.user_mean_time[u])
        return float(self.max_time)

    def _implicit_sum(self, u):
        rated_items = self.user_rated_items.get(u, [])
        if not rated_items:
            return np.zeros(self.n_factors, dtype=float)
        norm = self.user_norm.get(u, 0.0)
        return np.sum(self.yj[rated_items], axis=0) * norm

    def _dev(self, u, timestamp_days):
        if not self.trainset.knows_user(u):
            return 0.0
        delta = float(timestamp_days) - float(self.user_mean_time[u])
        if np.isclose(delta, 0.0):
            return 0.0
        return np.sign(delta) * (abs(delta) ** self.beta)

    def _time_bin(self, timestamp_days):
        if self.n_bins <= 1 or np.isclose(self.max_time, self.min_time):
            return 0
        scaled = (float(timestamp_days) - self.min_time) / (self.max_time - self.min_time)
        scaled = min(0.999999, max(0.0, scaled))
        return int(scaled * self.n_bins)

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "flaskr" / "evaluation.py"
SPEC = importlib.util.spec_from_file_location("evaluation_module", MODULE_PATH)
evaluation = importlib.util.module_from_spec(SPEC) # type: ignore
assert SPEC.loader is not None # type: ignore
SPEC.loader.exec_module(evaluation) # type: ignore


class RankingMetricTests(unittest.TestCase):
    def test_precision_recall_hit_rate_and_mrr(self):
        recommended = [10, 20, 30]
        relevant = {20, 50}

        self.assertAlmostEqual(evaluation.precision_at_k(recommended, relevant, 3), 1 / 3)
        self.assertAlmostEqual(evaluation.recall_at_k(recommended, relevant, 3), 0.5)
        self.assertEqual(evaluation.hit_rate_at_k(recommended, relevant, 3), 1.0)
        self.assertAlmostEqual(evaluation.mrr_at_k(recommended, relevant, 3), 0.5)

    def test_average_precision_and_ndcg(self):
        recommended = [7, 8, 9, 10]
        relevant = {8, 10}

        self.assertAlmostEqual(evaluation.average_precision_at_k(recommended, relevant, 4), 0.5)
        self.assertAlmostEqual(evaluation.ndcg_at_k(recommended, relevant, 4), 0.6509209298)

    def test_batch_metrics(self):
        recommendation_lists = [[1, 2, 3], [4, 5, 6]]
        relevant_lists = [{1, 3}, {7}]

        metrics = evaluation.evaluate_ranking_batch(recommendation_lists, relevant_lists, 3)

        self.assertIn("precision@3", metrics)
        self.assertIn("recall@3", metrics)
        self.assertIn("hit_rate@3", metrics)
        self.assertIn("mrr@3", metrics)
        self.assertIn("map@3", metrics)
        self.assertIn("ndcg@3", metrics)
        self.assertAlmostEqual(metrics["hit_rate@3"], 0.5)

    def test_batch_metrics_at_20_keys_exist(self):
        metrics = evaluation.evaluate_ranking_batch([[1, 2, 3]], [{1}], 20)

        self.assertIn("recall@20", metrics)
        self.assertIn("hit_rate@20", metrics)
        self.assertIn("map@20", metrics)
        self.assertIn("ndcg@20", metrics)

    def test_empty_relevant_items_return_zero(self):
        recommended = [1, 2, 3]
        relevant = set()

        self.assertEqual(evaluation.recall_at_k(recommended, relevant, 3), 0.0)
        self.assertEqual(evaluation.average_precision_at_k(recommended, relevant, 3), 0.0)
        self.assertEqual(evaluation.ndcg_at_k(recommended, relevant, 3), 0.0)


if __name__ == "__main__":
    unittest.main()

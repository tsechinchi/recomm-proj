"""
Compare accuracy metrics between original and improved recommendation algorithms.
"""
import json
import subprocess
import sys
from pathlib import Path


def run_evaluation(script_path):
    """Run an evaluation script and return parsed JSON output"""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stderr)
        sys.exit(1)

    # Extract JSON from output (skip non-JSON lines like progress messages)
    lines = result.stdout.strip().split('\n')
    json_start = None
    for i, line in enumerate(lines):
        if line.startswith('{'):
            json_start = i
            break

    if json_start is None:
        print(f"No JSON found in output from {script_path}")
        print(result.stdout)
        sys.exit(1)

    json_str = '\n'.join(lines[json_start:])
    return json.loads(json_str)


def compare_algorithms():
    tests_dir = Path(__file__).resolve().parent

    print("Evaluating recommendation algorithms...")
    print("-" * 70)

    # Run original algorithm evaluation
    print("\n1. Evaluating original algorithm (KNNWithMeans + genre-based content)...")
    original_summary = run_evaluation(tests_dir / "evaluate_original.py")

    # Run improved algorithm evaluation
    print("2. Evaluating improved algorithm (SVD++ + TF-IDF + temporal)...")
    improved_summary = run_evaluation(tests_dir / "evaluate_hybrid.py")

    # Print results
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nUsers evaluated: {original_summary['users_evaluated']}")
    print(f"Evaluation split: {original_summary['split']}")

    print("\n" + "-" * 70)
    print(f"{'Metric':<20} {'Original':<20} {'Improved':<20} {'Difference':<15}")
    print("-" * 70)

    original_metrics = original_summary["metrics"]
    improved_metrics = improved_summary["metrics"]

    for metric_name in ["ndcg@20", "map@20", "recall@20", "hit_rate@20"]:
        original_val = original_metrics[metric_name]
        improved_val = improved_metrics[metric_name]
        diff = improved_val - original_val
        diff_pct = (diff / original_val * 100) if original_val > 0 else 0

        print(
            f"{metric_name:<20} {original_val:<20.4f} {improved_val:<20.4f} "
            f"{diff:+.4f} ({diff_pct:+.1f}%)"
        )

    print("\n" + "=" * 70)
    print(f"Original model:  {original_summary['model']}")
    print(f"Improved model:  {improved_summary['model']}")
    print("=" * 70)


if __name__ == "__main__":
    compare_algorithms()

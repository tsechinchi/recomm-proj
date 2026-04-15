"""
Analyze dataset sparsity to understand the challenge of the recommendation task.
"""
import json
import numpy as np
import pandas as pd
from flaskr.tools.data_tool import loadData


def analyze_sparsity():
    movies, genres, ratings = loadData()

    print("=" * 70)
    print("DATASET SPARSITY ANALYSIS")
    print("=" * 70)

    # Basic counts
    n_users = ratings["userId"].nunique()
    n_movies = len(movies)
    n_ratings = len(ratings)
    max_possible_ratings = n_users * n_movies

    print(f"\nBasic Statistics:")
    print(f"  Users: {n_users:,}")
    print(f"  Movies: {n_movies:,}")
    print(f"  Ratings: {n_ratings:,}")
    print(f"  Max possible ratings: {max_possible_ratings:,}")

    # Sparsity metrics
    density = (n_ratings / max_possible_ratings) * 100
    sparsity = 100 - density

    print(f"\nSparsity Metrics:")
    print(f"  Density: {density:.4f}%")
    print(f"  Sparsity: {sparsity:.4f}%")
    print(f"  (Only {density:.4f}% of possible user-movie pairs are rated)")

    # User statistics
    user_ratings = ratings.groupby("userId").size()
    print(f"\nUser Statistics:")
    print(f"  Avg ratings per user: {user_ratings.mean():.1f}")
    print(f"  Median ratings per user: {user_ratings.median():.0f}")
    print(f"  Min ratings per user: {user_ratings.min()}")
    print(f"  Max ratings per user: {user_ratings.max()}")
    print(f"  Std dev: {user_ratings.std():.1f}")

    # Movie statistics
    movie_ratings = ratings.groupby("movieId").size()
    print(f"\nMovie Statistics:")
    print(f"  Avg ratings per movie: {movie_ratings.mean():.1f}")
    print(f"  Median ratings per movie: {movie_ratings.median():.0f}")
    print(f"  Min ratings per movie: {movie_ratings.min()}")
    print(f"  Max ratings per movie: {movie_ratings.max()}")
    print(f"  Std dev: {movie_ratings.std():.1f}")

    # Distribution analysis
    print(f"\nUser Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(user_ratings, p)
        print(f"  {p}th percentile: {val:.0f} ratings")

    print(f"\nMovie Distribution:")
    for p in percentiles:
        val = np.percentile(movie_ratings, p)
        print(f"  {p}th percentile: {val:.0f} ratings")

    # Cold start problem
    users_with_lt5_ratings = (user_ratings < 5).sum()
    movies_with_lt5_ratings = (movie_ratings < 5).sum()

    print(f"\nCold Start Problem:")
    print(f"  Users with <5 ratings: {users_with_lt5_ratings} ({users_with_lt5_ratings/n_users*100:.1f}%)")
    print(f"  Movies with <5 ratings: {movies_with_lt5_ratings} ({movies_with_lt5_ratings/n_movies*100:.1f}%)")
    print(f"  Users with <10 ratings: {(user_ratings < 10).sum()} ({(user_ratings < 10).sum()/n_users*100:.1f}%)")
    print(f"  Movies with <10 ratings: {(movie_ratings < 10).sum()} ({(movie_ratings < 10).sum()/n_movies*100:.1f}%)")

    # Rating distribution
    print(f"\nRating Distribution:")
    for rating in sorted(ratings["rating"].unique()):
        count = (ratings["rating"] == rating).sum()
        pct = count / n_ratings * 100
        print(f"  {rating}: {count:,} ({pct:.1f}%)")

    # Temporal coverage
    print(f"\nTemporal Coverage:")
    min_ts = ratings["timestamp"].min()
    max_ts = ratings["timestamp"].max()
    min_date = pd.to_datetime(min_ts, unit='s').strftime('%Y-%m-%d')
    max_date = pd.to_datetime(max_ts, unit='s').strftime('%Y-%m-%d')
    days_span = (max_ts - min_ts) / (86400)
    print(f"  Date range: {min_date} to {max_date}")
    print(f"  Time span: {days_span:.0f} days ({days_span/365.25:.1f} years)")
    print(f"  Avg ratings per day: {n_ratings/days_span:.0f}")

    # Impact on evaluation
    print(f"\n" + "=" * 70)
    print("IMPACT ON RECOMMENDATION EVALUATION")
    print("=" * 70)
    print(f"  With {n_movies:,} movies and top-20 ranking:")
    print(f"    - Random hit probability: {20/n_movies*100:.3f}%")
    print(f"    - 32 hits (improved algo): {32/485*100:.1f}% hit rate")
    print(f"    - Improvement over random: {(32/485)/(20/n_movies):.1f}x")
    print(f"\n  Dataset is {sparsity:.2f}% sparse → Hard to learn good models")
    print(f"  Collaborative filtering needs dense interactions → Struggle here")
    print(f"  Content filtering helps but still limited by sparse user profiles")

    # Summary statistics
    summary = {
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "n_ratings": int(n_ratings),
        "density_percent": float(density),
        "sparsity_percent": float(sparsity),
        "avg_ratings_per_user": float(user_ratings.mean()),
        "avg_ratings_per_movie": float(movie_ratings.mean()),
        "users_with_lt5_ratings_percent": float(users_with_lt5_ratings / n_users * 100),
        "movies_with_lt5_ratings_percent": float(movies_with_lt5_ratings / n_movies * 100),
        "time_span_days": float(days_span),
    }

    return summary


if __name__ == "__main__":
    summary = analyze_sparsity()
    print("\n" + "=" * 70)
    print("SUMMARY (JSON)")
    print("=" * 70)
    print(json.dumps(summary, indent=2))

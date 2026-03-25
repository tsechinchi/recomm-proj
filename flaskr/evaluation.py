from math import log2


def precision_at_k(recommended_items, relevant_items, k=20):
    if k <= 0:
        return 0.0
    recommended = list(recommended_items)[:k]
    relevant = set(relevant_items)
    if not recommended:
        return 0.0
    hits = sum(1 for item in recommended if item in relevant)
    return hits / float(k)


def recall_at_k(recommended_items, relevant_items, k=20):
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    recommended = list(recommended_items)[:k]
    hits = sum(1 for item in recommended if item in relevant)
    return hits / float(len(relevant))


def hit_rate_at_k(recommended_items, relevant_items, k=20):
    relevant = set(relevant_items)
    recommended = list(recommended_items)[:k]
    return 1.0 if any(item in relevant for item in recommended) else 0.0


def mrr_at_k(recommended_items, relevant_items, k=20):
    relevant = set(relevant_items)
    for rank, item in enumerate(list(recommended_items)[:k], start=1):
        if item in relevant:
            return 1.0 / float(rank)
    return 0.0


def average_precision_at_k(recommended_items, relevant_items, k=20):
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0
    seen = set()
    for rank, item in enumerate(list(recommended_items)[:k], start=1):
        if item in relevant and item not in seen:
            hits += 1
            seen.add(item)
            score += hits / float(rank)

    return score / float(min(len(relevant), k))


def dcg_at_k(recommended_items, relevant_items, k=20):
    relevant = set(relevant_items)
    dcg = 0.0
    for rank, item in enumerate(list(recommended_items)[:k], start=1):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)
    return dcg


def ndcg_at_k(recommended_items, relevant_items, k=20):
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    actual_dcg = dcg_at_k(recommended_items, relevant, k)
    ideal_length = min(len(relevant), k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_length + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def map_at_k(recommendation_lists, relevant_lists, k=20):
    scores = [
        average_precision_at_k(recommended, relevant, k)
        for recommended, relevant in zip(recommendation_lists, relevant_lists)
    ]
    if not scores:
        return 0.0
    return sum(scores) / float(len(scores))


def evaluate_ranking_batch(recommendation_lists, relevant_lists, k=20):
    if len(recommendation_lists) != len(relevant_lists):
        raise ValueError("recommendation_lists and relevant_lists must have the same length")

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    mrr_scores = []
    ndcg_scores = []

    for recommended, relevant in zip(recommendation_lists, relevant_lists):
        precision_scores.append(precision_at_k(recommended, relevant, k))
        recall_scores.append(recall_at_k(recommended, relevant, k))
        hit_rate_scores.append(hit_rate_at_k(recommended, relevant, k))
        mrr_scores.append(mrr_at_k(recommended, relevant, k))
        ndcg_scores.append(ndcg_at_k(recommended, relevant, k))

    total = len(recommendation_lists) or 1
    return {
        f"precision@{k}": sum(precision_scores) / float(total),
        f"recall@{k}": sum(recall_scores) / float(total),
        f"hit_rate@{k}": sum(hit_rate_scores) / float(total),
        f"mrr@{k}": sum(mrr_scores) / float(total),
        f"map@{k}": map_at_k(recommendation_lists, relevant_lists, k),
        f"ndcg@{k}": sum(ndcg_scores) / float(total),
    }

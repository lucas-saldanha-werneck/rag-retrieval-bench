def evaluate(retriever, queries: list[dict], k: int = 3) -> dict:
    """queries: [{"q": str, "relevant": doc_id}]. Returns hit@1, hit@k, MRR and misses."""
    hit1 = hitk = rr = 0.0
    misses = []
    for item in queries:
        ranking = retriever.rank(item["q"])
        pos = ranking.index(item["relevant"]) + 1
        hit1 += pos == 1
        hitk += pos <= k
        rr += 1.0 / pos
        if pos > k:
            misses.append({"q": item["q"], "relevant": item["relevant"], "rank": pos})
    n = len(queries)
    return {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n, "misses": misses}

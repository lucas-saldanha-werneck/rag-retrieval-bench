import argparse
import json
import sys
from pathlib import Path

from .metrics import evaluate
from .retrievers import BM25, Hybrid, NgramVectorIndex


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_report(docs: list[dict], queries: list[dict], k: int = 3) -> str:
    corpus = {d["id"]: d["text"] for d in docs}
    unknown = {q["relevant"] for q in queries} - corpus.keys()
    if unknown:
        raise ValueError(f"queries point to unknown doc ids: {sorted(unknown)}")
    bm25, vec = BM25(corpus), NgramVectorIndex(corpus)
    systems = {"bm25": bm25, "ngram-vector": vec, "hybrid (RRF)": Hybrid([bm25, vec])}
    lines = [
        f"# Retrieval report ({len(docs)} docs, {len(queries)} queries)",
        "",
        f"| retriever | hit@1 | hit@{k} | MRR |",
        "|---|---|---|---|",
    ]
    results = {name: evaluate(r, queries, k) for name, r in systems.items()}
    for name, m in results.items():
        lines.append(f"| {name} | {m['hit@1']:.2f} | {m[f'hit@{k}']:.2f} | {m['mrr']:.2f} |")
    for name, m in results.items():
        lines += ["", f"## Misses for {name} (relevant doc below rank {k})"]
        lines += [f"- `{x['relevant']}` ranked {x['rank']}: {x['q']}" for x in m["misses"]] or ["- none"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent.parent / "examples"
    p = argparse.ArgumentParser(description="Compare retrievers on a doc set and a query set.")
    p.add_argument("--docs", default=str(here / "faq_pt.jsonl"))
    p.add_argument("--queries", default=str(here / "queries_pt.jsonl"))
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args(argv)
    sys.stdout.write(build_report(load_jsonl(a.docs), load_jsonl(a.queries), a.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

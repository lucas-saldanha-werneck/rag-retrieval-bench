import json

import pytest

from ragbench.cli import build_report, load_jsonl, main
from ragbench.metrics import evaluate
from ragbench.retrievers import BM25, Hybrid, NgramVectorIndex
from ragbench.text import char_ngrams, fold, tokens

DOCS = {
    "a": "Para cancelar a assinatura clique em Cancelar plano.",
    "b": "A segunda via do boleto fica em Financeiro.",
    "c": "Ative a autenticação em dois fatores em Segurança.",
}


def test_fold_removes_accents_and_case():
    assert fold("Cobrança AÇÃO") == "cobranca acao"


def test_tokens_drop_stopwords():
    assert tokens("O boleto da fatura") == ["boleto", "fatura"]


def test_char_ngrams_pad_word_edges():
    assert char_ngrams("sol") == [" so", "sol", "ol "]


def test_bm25_finds_exact_words():
    assert BM25(DOCS).rank("segunda via do boleto")[0] == "b"


def test_bm25_ignores_accent_differences():
    assert BM25(DOCS).rank("autenticacao")[0] == "c"


def test_bm25_zero_overlap_scores_zero_and_ties_by_id():
    assert BM25(DOCS).rank("zzz") == ["a", "b", "c"]


def test_ngram_matches_word_variants():
    # "cancelamento" shares no whole word with doc "a", but shares n-grams.
    assert NgramVectorIndex(DOCS).rank("cancelamento")[0] == "a"
    assert BM25(DOCS).score("cancelamento", "a") == 0.0


def test_hybrid_returns_every_doc_once():
    r = Hybrid([BM25(DOCS), NgramVectorIndex(DOCS)]).rank("boleto")
    assert sorted(r) == ["a", "b", "c"] and r[0] == "b"


def test_ranking_is_deterministic():
    r = Hybrid([BM25(DOCS), NgramVectorIndex(DOCS)])
    assert r.rank("plano") == r.rank("plano")


class Fixed:
    def __init__(self, ranking):
        self.ranking = ranking

    def rank(self, q):
        return self.ranking


def test_metrics_values():
    queries = [{"q": "x", "relevant": "a"}, {"q": "y", "relevant": "c"}]
    m = evaluate(Fixed(["a", "b", "c", "d"]), queries, k=2)
    assert m["hit@1"] == 0.5
    assert m["hit@2"] == 0.5
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert m["misses"] == [{"q": "y", "relevant": "c", "rank": 3}]


def test_report_rejects_unknown_doc_id():
    with pytest.raises(ValueError):
        build_report([{"id": "a", "text": "x"}], [{"q": "q", "relevant": "zz"}])


def test_report_contains_all_retrievers():
    rep = build_report(
        [{"id": k, "text": v} for k, v in DOCS.items()],
        [{"q": "boleto", "relevant": "b"}],
    )
    for name in ("bm25", "ngram-vector", "hybrid (RRF)"):
        assert name in rep


def test_example_data_is_consistent():
    from pathlib import Path

    ex = Path(__file__).resolve().parent.parent / "examples"
    docs = load_jsonl(str(ex / "faq_pt.jsonl"))
    queries = load_jsonl(str(ex / "queries_pt.jsonl"))
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids)) == 24
    assert all(q["relevant"] in ids for q in queries)


def test_example_numbers_are_stable():
    # Pins the numbers quoted in the README. If these change, update the README.
    from pathlib import Path

    ex = Path(__file__).resolve().parent.parent / "examples"
    docs = {d["id"]: d["text"] for d in load_jsonl(str(ex / "faq_pt.jsonl"))}
    queries = load_jsonl(str(ex / "queries_pt.jsonl"))
    bm25, vec = BM25(docs), NgramVectorIndex(docs)
    got = {
        "bm25": evaluate(bm25, queries)["hit@3"],
        "vec": evaluate(vec, queries)["hit@3"],
        "hyb": evaluate(Hybrid([bm25, vec]), queries)["hit@3"],
    }
    assert got == {"bm25": 20 / 28, "vec": 22 / 28, "hyb": 22 / 28}


def test_cli_runs(tmp_path, capsys):
    d = tmp_path / "d.jsonl"
    q = tmp_path / "q.jsonl"
    d.write_text(json.dumps({"id": "a", "text": "boleto vencido"}) + "\n")
    q.write_text(json.dumps({"q": "boleto", "relevant": "a"}) + "\n")
    assert main(["--docs", str(d), "--queries", str(q)]) == 0
    assert "| bm25 | 1.00" in capsys.readouterr().out

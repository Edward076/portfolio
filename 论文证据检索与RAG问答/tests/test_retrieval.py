from scientific_rag.bm25_retriever import BM25Retriever
from scientific_rag.evaluation import evaluate_retrieval, retrieval_metrics
from scientific_rag.hybrid_retriever import reciprocal_rank_fusion
from scientific_rag.reranker import RerankedHybridRetriever
from scientific_rag.statistics import cluster_bootstrap_ci, paired_cluster_bootstrap_ci


def test_bm25_returns_ranked_paper_local_passages():
    chunks = [
        {"chunk_id": "p1:a", "paper_id": "p1", "text": "Dense retrieval uses vector embeddings."},
        {"chunk_id": "p1:b", "paper_id": "p1", "text": "BM25 BM25 keyword keyword scoring."},
        {"chunk_id": "p1:d", "paper_id": "p1", "text": "Unrelated methods use another model."},
        {"chunk_id": "p1:e", "paper_id": "p1", "text": "A separate paper describes training."},
        {"chunk_id": "p2:c", "paper_id": "p2", "text": "BM25 keyword score."},
    ]
    hits = BM25Retriever(chunks).retrieve("BM25 keyword", "p1", top_k=2)
    assert len(hits) == 2
    assert all(hit["paper_id"] == "p1" for hit in hits)
    assert hits[0]["chunk_id"] == "p1:b"
    assert [hit["rank"] for hit in hits] == [1, 2]


def test_multi_annotator_evidence_uses_best_alternative_set():
    metrics = retrieval_metrics(["a", "b"], [{"a", "c"}, {"a", "b"}], k=2)
    assert metrics["evidence_recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["hit_rate"] == 1.0
    assert metrics["complete_evidence_coverage"] == 1.0


def test_no_mapped_evidence_is_not_a_retrieval_miss():
    metrics = retrieval_metrics(["a"], [], k=5)
    assert metrics["evidence_recall"] is None
    assert metrics["hit_rate"] is None


def test_unscored_questions_can_receive_retrieval_predictions_for_generation():
    questions = [
        {
            "question_id": "answerable",
            "paper_id": "p1",
            "question": "Where is evidence?",
            "annotations": [{"unanswerable": False, "mapped_evidence_ids": ["p1:e"]}],
        },
        {
            "question_id": "unanswerable",
            "paper_id": "p1",
            "question": "Can the paper answer this?",
            "annotations": [{"unanswerable": True, "mapped_evidence_ids": []}],
        },
        {
            "question_id": "unmapped",
            "paper_id": "p1",
            "question": "Where is table evidence?",
            "annotations": [{"unanswerable": False, "mapped_evidence_ids": []}],
        },
    ]

    class FakeRetriever:
        method_name = "fake"

        def retrieve(self, query, paper_id, top_k):
            return [
                {
                    "chunk_id": "p1:e",
                    "paragraph_id": "p1:e",
                    "paper_id": paper_id,
                    "text": query,
                }
            ][:top_k]

    rows, predictions = evaluate_retrieval(
        questions, FakeRetriever(), top_k=1, include_unscored_predictions=True
    )

    assert [row["question_id"] for row in rows] == ["answerable"]
    assert [row["question_id"] for row in predictions] == [
        "answerable",
        "unanswerable",
        "unmapped",
    ]
    assert predictions[0]["gold_evidence_sets"] == [["p1:e"]]
    assert predictions[1]["gold_evidence_sets"] == []
    assert predictions[2]["gold_evidence_sets"] == []


def test_cluster_bootstrap_and_paired_difference_are_reproducible():
    baseline = [
        {"paper_id": "p1", "question_id": "q1", "score": 0.0},
        {"paper_id": "p1", "question_id": "q2", "score": 1.0},
        {"paper_id": "p2", "question_id": "q3", "score": 0.0},
    ]
    improved = [{**row, "score": 1.0} for row in baseline]
    ci = cluster_bootstrap_ci(improved, "score", n_bootstrap=100)
    paired = paired_cluster_bootstrap_ci(baseline, improved, "score", n_bootstrap=100)
    assert ci["n_questions"] == 3
    assert ci["n_clusters"] == 2
    assert ci["mean"] == 1.0
    assert paired["mean_difference"] == 2 / 3


def test_rrf_deduplicates_passages_and_combines_rank_evidence():
    dense = [
        {"chunk_id": "a", "paper_id": "p1", "text": "A"},
        {"chunk_id": "b", "paper_id": "p1", "text": "B"},
    ]
    lexical = [
        {"chunk_id": "b", "paper_id": "p1", "text": "B"},
        {"chunk_id": "c", "paper_id": "p1", "text": "C"},
    ]
    fused = reciprocal_rank_fusion([dense, lexical], rrf_constant=60, top_k=3)
    assert [item["chunk_id"] for item in fused] == ["b", "a", "c"]
    assert len({item["chunk_id"] for item in fused}) == 3
    assert fused[0]["source_ranks"] == [2, 1]


def test_reranked_hybrid_only_scores_retrieval_candidates():
    candidates = [
        {"chunk_id": "a", "paragraph_id": "a", "text": "A", "rank": 1},
        {"chunk_id": "b", "paragraph_id": "b", "text": "B", "rank": 2},
    ]

    class FakeHybrid:
        def retrieve(self, query, paper_id, top_k):
            return candidates[:top_k]

    class FakeReranker:
        def rerank(self, query, given, top_k):
            assert given == candidates
            assert top_k == len(candidates)
            return [
                {**given[1], "pre_rerank_rank": 2, "rank": 1},
                {**given[0], "pre_rerank_rank": 1, "rank": 2},
            ]

    results = RerankedHybridRetriever(FakeHybrid(), FakeReranker()).retrieve("q", "p1", top_k=1)
    assert [item["chunk_id"] for item in results] == ["b", "a"]

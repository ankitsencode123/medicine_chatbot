"""
Threshold Calibration Diagnostic
─────────────────────────────────
Runs each eval query through the retrieval pipeline and prints raw scores
at every stage (BM25, Vector, RRF, Cross-Encoder) so you can empirically
validate that chosen thresholds produce optimal precision without hurting recall.

Usage:
    python calibrate_thresholds.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rag_engine import (
    bm25_retrieve, vector_retrieve,
    reciprocal_rank_fusion, cross_encoder_rerank,
    retrieve,
)
from rag_metrics import EVAL_DATASET
from config import (
    CROSS_ENCODER_THRESHOLD, RRF_MIN_SCORE_THRESHOLD, RRF_SCORE_DROP_RATIO,
)


def calibrate():
    print("=" * 70)
    print("  THRESHOLD CALIBRATION DIAGNOSTIC")
    print(f"  RRF floor={RRF_MIN_SCORE_THRESHOLD}  "
          f"RRF drop_ratio={RRF_SCORE_DROP_RATIO}  "
          f"CE threshold={CROSS_ENCODER_THRESHOLD}")
    print("=" * 70)

    total_precision = 0.0
    total_recall = 0.0
    n = len(EVAL_DATASET)

    for tc in EVAL_DATASET:
        query = tc["query"]
        relevant = set(tc["relevant_medicine_ids"])

        print(f"\n{'─' * 70}")
        print(f"🔍 Query: {query}")
        print(f"   Relevant: {sorted(relevant)}")

        # Stage 1: raw candidates
        bm25 = bm25_retrieve(query, top_k=5)
        vec = vector_retrieve(query, top_k=5)

        # Stage 2: RRF fusion (NO threshold — show all scores)
        from collections import Counter
        rrf_scores = {}
        doc_map = {}
        k = 60
        for rlist in [bm25, vec]:
            for rank, doc in enumerate(rlist, 1):
                doc_id = doc["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        top_score = rrf_scores[sorted_ids[0]] if sorted_ids else 0.0
        gap_cutoff = top_score * RRF_SCORE_DROP_RATIO

        print(f"\n   RRF scores (top_score={top_score:.6f}, gap_cutoff={gap_cutoff:.6f}):")
        for doc_id in sorted_ids[:7]:
            score = rrf_scores[doc_id]
            is_relevant = "✅" if doc_id in relevant else "  "
            passes_gap = "PASS" if score >= gap_cutoff else "DROP"
            name = doc_map[doc_id]["metadata"]["medicine_name"]
            print(f"     {is_relevant} {doc_id:>8s}  {name:>20s}  "
                  f"RRF={score:.6f}  [{passes_gap}]")

        # Stage 2b: actual RRF with thresholds
        fused = reciprocal_rank_fusion(
            [bm25, vec], top_n=5,
            min_score_threshold=RRF_MIN_SCORE_THRESHOLD,
            score_drop_ratio=RRF_SCORE_DROP_RATIO,
        )
        print(f"\n   After RRF filtering: {len(fused)} docs survive")

        # Stage 3: Cross-Encoder
        from sentence_transformers import CrossEncoder
        from config import CROSS_ENCODER_MODEL as CE_MODEL
        model = CrossEncoder(CE_MODEL)
        pairs = [(query, doc["document"]) for doc in fused]
        if pairs:
            scores = model.predict(pairs)
            print(f"\n   Cross-Encoder scores (threshold={CROSS_ENCODER_THRESHOLD}):")
            for doc, ce_score in zip(fused, scores):
                is_relevant = "✅" if doc["id"] in relevant else "  "
                passes = "PASS" if float(ce_score) >= CROSS_ENCODER_THRESHOLD else "DROP"
                print(f"     {is_relevant} {doc['id']:>8s}  "
                      f"{doc['metadata']['medicine_name']:>20s}  "
                      f"CE={float(ce_score):>8.4f}  [{passes}]")

        # Final output
        final = retrieve(query)
        final_ids = [d["id"] for d in final]
        final_relevant = set(final_ids) & relevant

        precision = len(final_relevant) / len(final_ids) if final_ids else 0.0
        recall = len(final_relevant) / len(relevant) if relevant else 1.0
        total_precision += precision
        total_recall += recall

        print(f"\n   Final: {len(final)} doc(s) → "
              f"Precision={precision:.2f}, Recall={recall:.2f}")
        for d in final:
            tag = "✅" if d["id"] in relevant else "❌"
            print(f"     {tag} {d['metadata']['medicine_name']} "
                  f"(RRF={d.get('rrf_score', 'N/A')}, "
                  f"CE={d.get('cross_encoder_score', 'N/A')})")

    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE: Precision={total_precision/n:.4f}  "
          f"Recall={total_recall/n:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    calibrate()

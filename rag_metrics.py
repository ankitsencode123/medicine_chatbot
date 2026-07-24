"""
RAG Evaluation Metrics
──────────────────────
Comprehensive metrics for evaluating the RAG pipeline:

RETRIEVAL METRICS:
  • Context Precision  – fraction of retrieved docs that are relevant
  • Context Recall     – fraction of relevant docs that were retrieved
  • Mean Reciprocal Rank (MRR) – 1/rank of first relevant doc
  • Hit Rate (HR@K)    – whether any relevant doc appears in top-K

GENERATION METRICS (LLM-as-judge via Groq):
  • Faithfulness       – does the answer stick to retrieved context?
  • Answer Relevancy   – does the answer address the user's question?
  • Correctness        – does the answer match the expected ground truth?

LEXICAL METRICS:
  • BLEU Score         – n-gram overlap with reference answer
  • ROUGE-L (F1)       – longest common subsequence overlap

LATENCY:
  • Retrieval latency (seconds)
  • Generation latency (seconds)
  • Total end-to-end latency
"""

import sys, pathlib, time, json, re, math
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rag_engine import retrieve, generate_answer, _call_groq


# ═══════════════════════════════════════════════════════════
# 1. GROUND-TRUTH BENCHMARK DATASET
# ═══════════════════════════════════════════════════════════
EVAL_DATASET = [
    {
        "query": "Do you have Ibuprofen for pain relief?",
        "relevant_medicine_ids": ["med_5"],
        "expected_keywords": ["ibuprofen", "400mg", "out of stock", "diclofenac", "pain"],
        "reference_answer": (
            "Ibuprofen 400mg is used for pain and inflammation. "
            "It is currently out of stock. An alternative is Diclofenac. "
            "Dosage: 1 tablet every 8 hours."
        ),
    },
    {
        "query": "What medicine is recommended for fever?",
        "relevant_medicine_ids": ["med_1"],
        "expected_keywords": ["paracetamol", "500mg", "fever", "crocin", "dolo"],
        "reference_answer": (
            "Paracetamol 500mg is recommended for fever and headache. "
            "It is in stock. Alternatives: Crocin, Dolo. "
            "Dosage: 1 tablet every 6 hours."
        ),
    },
    {
        "query": "Is Paracetamol available? Any alternative?",
        "relevant_medicine_ids": ["med_1"],
        "expected_keywords": ["paracetamol", "yes", "stock", "crocin", "dolo"],
        "reference_answer": (
            "Yes, Paracetamol 500mg is currently in stock. "
            "Alternatives include Crocin and Dolo. "
            "Dosage: 1 tablet every 6 hours."
        ),
    },
    {
        "query": "Do you have Amoxicillin?",
        "relevant_medicine_ids": ["med_2"],
        "expected_keywords": ["amoxicillin", "250mg", "out of stock", "augmentin", "bacterial"],
        "reference_answer": (
            "Amoxicillin 250mg is used for bacterial infections. "
            "It is currently out of stock. An alternative is Augmentin. "
            "Dosage: 1 capsule every 8 hours."
        ),
    },
    {
        "query": "What can I take for allergy?",
        "relevant_medicine_ids": ["med_3"],
        "expected_keywords": ["cetirizine", "10mg", "allergy", "levocetirizine"],
        "reference_answer": (
            "Cetirizine 10mg is used for allergy and cold. "
            "It is in stock. Alternative: Levocetirizine. "
            "Dosage: 1 tablet at night."
        ),
    },
    {
        "query": "Medicine for diabetes?",
        "relevant_medicine_ids": ["med_4", "med_10"],
        "expected_keywords": ["metformin", "diabetes", "insulin"],
        "reference_answer": (
            "Metformin 500mg is used for diabetes and is in stock. "
            "Alternative: Glimepiride. Dosage: 1 tablet after meals. "
            "Insulin 10ml is also used for diabetes but is currently out of stock."
        ),
    },
    {
        "query": "I have acidity, what should I take?",
        "relevant_medicine_ids": ["med_6"],
        "expected_keywords": ["ranitidine", "150mg", "acidity", "famotidine"],
        "reference_answer": (
            "Ranitidine 150mg is used for acidity and is in stock. "
            "Alternative: Famotidine. "
            "Dosage: 1 tablet before meals."
        ),
    },
    {
        "query": "Is Azithromycin in stock for throat infection?",
        "relevant_medicine_ids": ["med_9"],
        "expected_keywords": ["azithromycin", "500mg", "throat", "stock", "clarithromycin"],
        "reference_answer": (
            "Azithromycin 500mg is used for throat infection and is in stock. "
            "Alternative: Clarithromycin. "
            "Dosage: 1 tablet daily for 3 days."
        ),
    },
    {
        "query": "What is available for dehydration?",
        "relevant_medicine_ids": ["med_7"],
        "expected_keywords": ["ors", "dehydration", "electral", "stock"],
        "reference_answer": (
            "ORS Solution 200ml is used for dehydration and is in stock. "
            "Alternative: Electral Powder. "
            "Dosage: As directed, sip slowly."
        ),
    },
    {
        "query": "Any vitamin supplement available?",
        "relevant_medicine_ids": ["med_8"],
        "expected_keywords": ["vitamin c", "500mg", "immunity", "zincovit"],
        "reference_answer": (
            "Vitamin C 500mg is used for immunity boost and is in stock. "
            "Alternative: Zincovit. "
            "Dosage: 1 tablet daily."
        ),
    },
]


# ═══════════════════════════════════════════════════════════
# 2. RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════════

def context_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of retrieved documents that are relevant."""
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in retrieved_ids if rid in relevant_set)
    return hits / len(retrieved_ids)


def context_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of relevant documents that were retrieved."""
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for rid in relevant_ids if rid in retrieved_set)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / rank of the first relevant document."""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 if any relevant document is in retrieved set, else 0."""
    relevant_set = set(relevant_ids)
    return 1.0 if any(rid in relevant_set for rid in retrieved_ids) else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 3) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0
    # ideal DCG: all relevant docs at top
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ═══════════════════════════════════════════════════════════
# 3. LEXICAL METRICS
# ═══════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r'\w+', text.lower())


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    """Compute BLEU score (unigram to max_n-gram) with brevity penalty."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0

    # brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(cand_tokens))) if cand_tokens else 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        cand_ngrams = Counter(tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1))
        overlap = sum((cand_ngrams & ref_ngrams).values())
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(overlap / total)

    # geometric mean of precisions (smoothed)
    log_avg = 0.0
    for p in precisions:
        if p == 0:
            return 0.0
        log_avg += math.log(p) / len(precisions)

    return bp * math.exp(log_avg)


def rouge_l_f1(reference: str, candidate: str) -> float:
    """ROUGE-L F1 using longest common subsequence."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]

    precision = lcs_len / n
    recall = lcs_len / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0


# ═══════════════════════════════════════════════════════════
# 4. LLM-AS-JUDGE METRICS (via Groq)
# ═══════════════════════════════════════════════════════════

def _llm_judge(prompt: str) -> float:
    """Ask the LLM to rate something 1-5, return normalized 0-1 score."""
    try:
        result = _call_groq([
            {"role": "system", "content": (
                "You are a strict evaluator. You MUST respond with ONLY a single integer "
                "from 1 to 5. No explanation, no text, just the number."
            )},
            {"role": "user", "content": prompt},
        ])
        score = int(re.search(r'[1-5]', result.strip()).group())
        return (score - 1) / 4.0  # normalize to 0-1
    except Exception:
        return 0.5  # fallback neutral score


def faithfulness_score(query: str, context: str, answer: str) -> float:
    """LLM judges: does the answer only use info from the context?"""
    prompt = (
        f"Rate 1-5 how faithful the ANSWER is to the CONTEXT. "
        f"5 = answer only uses info from context, 1 = answer fabricates information.\n\n"
        f"QUERY: {query}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}\n\nScore (1-5):"
    )
    return _llm_judge(prompt)


def answer_relevancy_score(query: str, answer: str) -> float:
    """LLM judges: does the answer address the user's question?"""
    prompt = (
        f"Rate 1-5 how relevant the ANSWER is to the QUERY. "
        f"5 = perfectly relevant, 1 = completely off-topic.\n\n"
        f"QUERY: {query}\n\nANSWER: {answer}\n\nScore (1-5):"
    )
    return _llm_judge(prompt)


def correctness_score(query: str, answer: str, reference: str) -> float:
    """LLM judges: does the answer match the expected ground-truth?"""
    prompt = (
        f"Rate 1-5 how correct the ANSWER is compared to the REFERENCE answer. "
        f"5 = fully correct, 1 = completely wrong.\n\n"
        f"QUERY: {query}\n\nREFERENCE: {reference}\n\nANSWER: {answer}\n\nScore (1-5):"
    )
    return _llm_judge(prompt)


# ═══════════════════════════════════════════════════════════
# 5. FULL EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════

def evaluate_single(test_case: dict, run_llm_judge: bool = True) -> dict:
    """
    Run the full RAG pipeline on one test case and compute all metrics.
    Returns a dict with all metric values.
    """
    query = test_case["query"]
    relevant_ids = test_case["relevant_medicine_ids"]
    expected_kw = test_case["expected_keywords"]
    reference = test_case["reference_answer"]

    # ── Retrieval ──
    t0 = time.time()
    docs = retrieve(query)
    retrieval_latency = time.time() - t0
    retrieved_ids = [d["id"] for d in docs]
    context_text = "\n".join(d["document"] for d in docs)

    # ── Generation ──
    t1 = time.time()
    answer, _ = generate_answer(query)
    generation_latency = time.time() - t1
    total_latency = retrieval_latency + generation_latency

    # ── Compute retrieval metrics ──
    cp = context_precision(retrieved_ids, relevant_ids)
    cr = context_recall(retrieved_ids, relevant_ids)
    mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    hr = hit_rate(retrieved_ids, relevant_ids)
    ndcg = ndcg_at_k(retrieved_ids, relevant_ids)

    # ── Compute lexical metrics ──
    bleu = bleu_score(reference, answer)
    rouge = rouge_l_f1(reference, answer)
    kw_cov = keyword_coverage(answer, expected_kw)

    result = {
        "query": query,
        "answer": answer,
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,
        # Retrieval
        "context_precision": round(cp, 4),
        "context_recall": round(cr, 4),
        "mrr": round(mrr, 4),
        "hit_rate": round(hr, 4),
        "ndcg@3": round(ndcg, 4),
        # Lexical
        "bleu": round(bleu, 4),
        "rouge_l_f1": round(rouge, 4),
        "keyword_coverage": round(kw_cov, 4),
        # Latency
        "retrieval_latency_s": round(retrieval_latency, 3),
        "generation_latency_s": round(generation_latency, 3),
        "total_latency_s": round(total_latency, 3),
    }

    # ── LLM-as-judge (optional, costs API calls) ──
    if run_llm_judge:
        result["faithfulness"] = round(faithfulness_score(query, context_text, answer), 4)
        result["answer_relevancy"] = round(answer_relevancy_score(query, answer), 4)
        result["correctness"] = round(correctness_score(query, answer, reference), 4)

    return result


def run_full_evaluation(run_llm_judge: bool = True, progress_callback=None) -> dict:
    """
    Run evaluation on all test cases in EVAL_DATASET.
    Returns:
      {
        "per_query": [list of per-query results],
        "aggregate": {metric_name: avg_value, ...}
      }
    """
    per_query = []
    for i, tc in enumerate(EVAL_DATASET):
        result = evaluate_single(tc, run_llm_judge=run_llm_judge)
        per_query.append(result)
        if progress_callback:
            progress_callback(i + 1, len(EVAL_DATASET), tc["query"])

    # ── Aggregate averages ──
    metric_keys = [
        "context_precision", "context_recall", "mrr", "hit_rate", "ndcg@3",
        "bleu", "rouge_l_f1", "keyword_coverage",
        "retrieval_latency_s", "generation_latency_s", "total_latency_s",
    ]
    if run_llm_judge:
        metric_keys += ["faithfulness", "answer_relevancy", "correctness"]

    aggregate = {}
    for key in metric_keys:
        values = [r[key] for r in per_query if key in r]
        aggregate[key] = round(sum(values) / len(values), 4) if values else 0.0

    return {"per_query": per_query, "aggregate": aggregate}


# ═══════════════════════════════════════════════════════════
# CLI runner
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG evaluation metrics")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip LLM-as-judge metrics (faster, fewer API calls)")
    args = parser.parse_args()

    def _progress(i, total, q):
        print(f"  [{i}/{total}] {q}")

    print("🔬 Running RAG Evaluation …\n")
    results = run_full_evaluation(
        run_llm_judge=not args.no_llm_judge,
        progress_callback=_progress,
    )

    print("\n" + "═" * 60)
    print("  AGGREGATE RESULTS")
    print("═" * 60)
    for k, v in results["aggregate"].items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20)) if v <= 1.0 else ""
        label = f"{v:.4f}"
        if "latency" in k:
            print(f"  {k:<25s} {label:>8s}s")
        else:
            print(f"  {k:<25s} {bar} {label}")
    print("═" * 60)

    # save results to JSON
    out_path = pathlib.Path(__file__).resolve().parent / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Full results saved to {out_path}")

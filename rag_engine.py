"""
RAG Engine – Precision-Optimised Hybrid Search
────────────────────────────────────────────────
Pipeline:
  1. BM25 keyword search (with Levenshtein fuzzy matching for typos)
  2. Dense vector search via ChromaDB Cloud
  3. Reciprocal Rank Fusion (RRF) with score-gap dynamic cutoff
  4. Cross-Encoder reranking with calibrated relevance cutoff
  5. Strict grounding prompt → LLM answers ONLY from retrieved context
  6. Automatic Groq API key rotation on 429 rate-limit errors
"""

import sys, pathlib, time, random, re, math
from collections import Counter
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq, RateLimitError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import (
    GROQ_API_KEYS, GROQ_MODEL,
    CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE,
    CHROMA_COLLECTION, EMBEDDING_MODEL, TOP_K_RESULTS,
    CSV_PATH,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_THRESHOLD, CROSS_ENCODER_SCORE_GAP,
    RRF_MIN_SCORE_THRESHOLD, RRF_SCORE_DROP_RATIO,
)

# ═══════════════════════════════════════════════════════════
# Singleton resources
# ═══════════════════════════════════════════════════════════
_embed_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None
_chroma_collection = None
_bm25_index = None
_key_index = 0


# ═══════════════════════════════════════════════════════════
# Strict Grounding System Prompt
# ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """\
You are a **medical inventory assistant** for a hospital pharmacy.

RULES — follow these strictly, no exceptions:
1. Answer ONLY using the CONTEXT provided below. Do NOT use any outside medical knowledge.
2. If the user asks about a medicine, state its exact Name, Strength, Use Case, Stock status, and Dosage Instruction as they appear in the context.
3. If a medicine is **out of stock**, mention that clearly and suggest the alternative ONLY if one is explicitly listed in the 'Alternative' field of the context.
4. When quoting dosage, use the EXACT 'Dosage_Instruction' text from the context. Do not paraphrase or invent dosage information.
5. If the requested medicine or condition is NOT found in the context, reply: "I do not have information on that medication in our inventory."
6. Never recommend a medicine, alternative, or dosage that is not explicitly present in the context.
7. End every response that includes dosage information with: "Please consult a doctor before use."

─── CONTEXT ───
{context}
────────────────
"""


# ═══════════════════════════════════════════════════════════
# BM25 Keyword Search Engine
# ═══════════════════════════════════════════════════════════

def _tokenize_bm25(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric. Keeps numbers together."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _build_bm25_index():
    """Build an in-memory BM25 index from the CSV data."""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    csv_path = pathlib.Path(__file__).resolve().parent / CSV_PATH
    df = pd.read_csv(csv_path)

    docs = []
    for _, row in df.iterrows():
        stock_status = "currently in stock" if str(row["Stock"]).strip().lower() == "yes" else "currently out of stock"
        text = (
            f"{row['Medicine_Name']} {row['Strength']} is used for {row['Use_Case']}. "
            f"It is {stock_status}. "
            f"Alternative medicines: {row['Alternative']}. "
            f"Dosage instruction: {row['Dosage_Instruction']}."
        )
        docs.append({
            "id": f"med_{int(row['Medicine_ID'])}",
            "document": text,
            "metadata": {
                "medicine_name": str(row["Medicine_Name"]),
                "strength": str(row["Strength"]),
                "use_case": str(row["Use_Case"]),
                "alternative": str(row["Alternative"]),
                "stock": str(row["Stock"]).strip(),
                "dosage": str(row["Dosage_Instruction"]),
            },
            "tokens": _tokenize_bm25(text),
        })

    N = len(docs)
    df_counts = Counter()
    for d in docs:
        unique_terms = set(d["tokens"])
        for t in unique_terms:
            df_counts[t] += 1

    idf = {}
    for term, freq in df_counts.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    avg_dl = sum(len(d["tokens"]) for d in docs) / N if N else 1

    _bm25_index = {"docs": docs, "idf": idf, "avg_dl": avg_dl}
    return _bm25_index


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                idf: dict, avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    dl = len(doc_tokens)
    doc_tf = Counter(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        if qt not in idf:
            continue
        tf = doc_tf.get(qt, 0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf[qt] * (numerator / denominator)
    return score


def _fuzzy_match_tokens(query_tokens: list[str], doc_tokens: list[str],
                        max_edit_dist: int = 2) -> list[str]:
    index = _build_bm25_index()
    all_vocab = set(index["idf"].keys())
    expanded = []
    for qt in query_tokens:
        if qt in all_vocab:
            expanded.append(qt)
        else:
            best_term, best_dist = None, max_edit_dist + 1
            for vocab_term in all_vocab:
                if abs(len(vocab_term) - len(qt)) > max_edit_dist:
                    continue
                d = _edit_distance(qt, vocab_term)
                if d < best_dist:
                    best_dist = d
                    best_term = vocab_term
            if best_term is not None and best_dist <= max_edit_dist:
                expanded.append(best_term)
            else:
                expanded.append(qt)
    return expanded


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def bm25_retrieve(query: str, top_k: int = 10) -> list[dict]:
    index = _build_bm25_index()
    query_tokens = _tokenize_bm25(query)
    query_tokens = _fuzzy_match_tokens(query_tokens, [], max_edit_dist=2)
    scored = []
    for doc in index["docs"]:
        score = _bm25_score(query_tokens, doc["tokens"], index["idf"], index["avg_dl"])
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        results.append({
            "id": doc["id"],
            "document": doc["document"],
            "metadata": doc["metadata"],
            "bm25_score": score,
        })
    return results


# ═══════════════════════════════════════════════════════════
# Dense Vector Search (ChromaDB Cloud)
# ═══════════════════════════════════════════════════════════

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def vector_retrieve(query: str, top_k: int = 10) -> list[dict]:
    model = _get_embed_model()
    query_embedding = model.encode([query]).tolist()
    collection = _get_collection()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return docs


# ═══════════════════════════════════════════════════════════
# Reciprocal Rank Fusion (RRF) with Dynamic Thresholding
# ═══════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 5,
    min_score_threshold: float = RRF_MIN_SCORE_THRESHOLD,
    score_drop_ratio: float = RRF_SCORE_DROP_RATIO,
) -> list[dict]:
    """
    Merge multiple ranked result lists using RRF with dual dynamic cutoff:

    1. Absolute floor: documents below min_score_threshold are dropped.
    2. Score-gap cutoff: documents scoring < score_drop_ratio × top_score
       are dropped (catches the cliff between relevant and bycatch docs).

    Output list may be shorter than top_n.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rlist in ranked_lists:
        for rank, doc in enumerate(rlist, 1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    # Determine the score-gap cutoff from the top-scoring document
    top_score = rrf_scores[sorted_ids[0]] if sorted_ids else 0.0
    gap_cutoff = top_score * score_drop_ratio

    results = []
    for doc_id in sorted_ids[:top_n]:
        score = round(rrf_scores[doc_id], 6)
        if score >= min_score_threshold and score >= gap_cutoff:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = score
            results.append(doc)

    return results


# ═══════════════════════════════════════════════════════════
# Cross-Encoder Reranker
# ═══════════════════════════════════════════════════════════

def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load the Cross-Encoder model (singleton)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def cross_encoder_rerank(
    query: str,
    docs: list[dict],
    threshold: float = CROSS_ENCODER_THRESHOLD,
    score_gap: float = CROSS_ENCODER_SCORE_GAP,
) -> list[dict]:
    """
    Re-score each document against the query using a Cross-Encoder.

    Dual pruning strategy:
    1. Absolute floor: documents with score < threshold are dropped
       (safety net for extreme mismatches).
    2. Score-gap: documents scoring > score_gap points below the best
       candidate are dropped (catches irrelevant bycatch even when
       absolute scores vary by query type — which they do: the same
       cross-encoder scores relevant pairs from 0.3 to 9.2 depending
       on lexical alignment).

    Returns a list sorted by cross-encoder score (desc), possibly shorter
    than input.
    """
    if not docs:
        return []

    model = _get_cross_encoder()

    # build pairs for scoring
    pairs = [(query, doc["document"]) for doc in docs]
    scores = model.predict(pairs)

    # find the top score for gap-based pruning
    top_ce_score = float(max(scores))
    gap_cutoff = top_ce_score - score_gap

    # attach scores and filter with dual criteria
    reranked = []
    for doc, score in zip(docs, scores):
        ce_score = float(score)
        if ce_score >= threshold and ce_score >= gap_cutoff:
            doc_copy = doc.copy()
            doc_copy["cross_encoder_score"] = round(ce_score, 4)
            reranked.append(doc_copy)

    # sort by cross-encoder score descending
    reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return reranked


# ═══════════════════════════════════════════════════════════
# Full Hybrid Retrieve Pipeline
# ═══════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Precision-optimised hybrid retrieval pipeline:
      1. BM25 + Vector → 5 candidates each
      2. RRF fusion with score-gap + absolute threshold → prune bycatch
      3. Cross-Encoder reranking with calibrated logit cutoff → only truly relevant docs
      4. Safety cap at top_k (but usually fewer after pruning)
    """
    # Stage 1: fetch wide candidate pool
    bm25_results = bm25_retrieve(query, top_k=5)
    vector_results = vector_retrieve(query, top_k=5)

    # Stage 2: RRF fusion with dual dynamic cutoff
    fused = reciprocal_rank_fusion(
        [bm25_results, vector_results],
        top_n=5,
        min_score_threshold=RRF_MIN_SCORE_THRESHOLD,
        score_drop_ratio=RRF_SCORE_DROP_RATIO,
    )

    # Stage 3: Cross-Encoder reranking with calibrated threshold
    reranked = cross_encoder_rerank(
        query, fused,
        threshold=CROSS_ENCODER_THRESHOLD,
    )

    # Safety cap (Cross-Encoder pruning controls precision, not this cap)
    return reranked[:top_k]


# ═══════════════════════════════════════════════════════════
# Groq LLM Caller (with key rotation)
# ═══════════════════════════════════════════════════════════

def _call_groq(messages: list[dict], retries: int = len(GROQ_API_KEYS)) -> str:
    global _key_index
    last_error = None
    for attempt in range(retries):
        key = GROQ_API_KEYS[_key_index % len(GROQ_API_KEYS)]
        client = Groq(api_key=key)
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            last_error = e
            _key_index += 1
            wait = 1 + random.random()
            time.sleep(wait)
        except Exception as e:
            last_error = e
            _key_index += 1
            if attempt < retries - 1:
                time.sleep(0.5)
    raise RuntimeError(
        f"All {retries} Groq API keys exhausted or errored. Last error: {last_error}"
    )


# ═══════════════════════════════════════════════════════════
# End-to-End RAG
# ═══════════════════════════════════════════════════════════

def generate_answer(query: str) -> tuple[str, list[dict]]:
    """
    End-to-end Precision RAG:
      1. Hybrid retrieve → RRF → Cross-Encoder rerank
      2. Build strict grounding prompt
      3. Generate answer via Groq LLM
    Returns (answer_text, retrieved_docs).
    """
    docs = retrieve(query)
    context_block = "\n\n".join(
        f"• {d['document']}" for d in docs
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context_block)},
        {"role": "user", "content": query},
    ]
    answer = _call_groq(messages)
    return answer, docs


if __name__ == "__main__":
    print("=" * 60)
    print("  MedAssist AI – Precision Hybrid Search")
    print("  (BM25 + Vector + RRF + Cross-Encoder)")
    print("=" * 60)

    test_queries = [
        "Do you have Paracetmol?",
        "Ibuprofin for pain",
        "my head hurts what should i take",
        "Amoxicilin for infection",
        "Do you have Ibuprofen for pain relief?",
    ]
    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        docs = retrieve(q)
        print(f"   Returned {len(docs)} doc(s):")
        for d in docs:
            print(f"     ✅ {d['metadata']['medicine_name']} "
                  f"(RRF={d.get('rrf_score', 'N/A')}, "
                  f"CE={d.get('cross_encoder_score', 'N/A')})")

"""
RAG Engine – Hybrid Search Edition
────────────────────────────────────
1. BM25 keyword search  → catches typos & exact product names.
2. Dense vector search   → catches semantic intent ("my head hurts" → Headache).
3. Reciprocal Rank Fusion (RRF) → merges both ranked lists into one.
4. Strict grounding prompt → LLM answers ONLY from retrieved context.
5. Automatic Groq API key rotation on 429 rate-limit errors.
"""

import sys, pathlib, time, random, re, math
from collections import Counter
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq, RateLimitError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import (
    GROQ_API_KEYS, GROQ_MODEL,
    CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE,
    CHROMA_COLLECTION, EMBEDDING_MODEL, TOP_K_RESULTS,
    CSV_PATH,
)

# ═══════════════════════════════════════════════════════════
# Singleton resources
# ═══════════════════════════════════════════════════════════
_embed_model: SentenceTransformer | None = None
_chroma_collection = None
_bm25_index = None  # will hold {"docs": [...], "idf": {...}, "tf": [...]}
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

    # compute IDF
    N = len(docs)
    df_counts = Counter()  # document frequency for each term
    for d in docs:
        unique_terms = set(d["tokens"])
        for t in unique_terms:
            df_counts[t] += 1

    idf = {}
    for term, freq in df_counts.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    # average document length
    avg_dl = sum(len(d["tokens"]) for d in docs) / N if N else 1

    _bm25_index = {"docs": docs, "idf": idf, "avg_dl": avg_dl}
    return _bm25_index


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                idf: dict, avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """Compute BM25 score for a single document."""
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
    """
    For each query token that has NO exact match in the index,
    try to find the closest token via edit distance (Levenshtein).
    Returns an expanded list of query tokens with corrections applied.
    """
    index = _build_bm25_index()
    all_vocab = set(index["idf"].keys())

    expanded = []
    for qt in query_tokens:
        if qt in all_vocab:
            expanded.append(qt)
        else:
            # find the closest vocabulary term
            best_term, best_dist = None, max_edit_dist + 1
            for vocab_term in all_vocab:
                # quick length filter to skip obviously bad matches
                if abs(len(vocab_term) - len(qt)) > max_edit_dist:
                    continue
                d = _edit_distance(qt, vocab_term)
                if d < best_dist:
                    best_dist = d
                    best_term = vocab_term
            if best_term is not None and best_dist <= max_edit_dist:
                expanded.append(best_term)
            else:
                expanded.append(qt)  # keep original if no close match
    return expanded


def _edit_distance(s1: str, s2: str) -> int:
    """Classic Levenshtein edit distance."""
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
    """BM25 keyword search with fuzzy matching for typo tolerance."""
    index = _build_bm25_index()
    query_tokens = _tokenize_bm25(query)

    # expand tokens with fuzzy matching (corrects typos)
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
    """Dense vector search via ChromaDB Cloud."""
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
# Reciprocal Rank Fusion (RRF)
# ═══════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_n: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Merge multiple ranked result lists using RRF.

    For each document across all lists:
        RRF_score = Σ  1 / (k + rank_i)

    where rank_i is the 1-based rank of the document in list i
    (and 0 contribution if not present in that list).

    Args:
        ranked_lists: list of ranked result lists, each a list of dicts with "id"
        k: RRF constant (default 60 per the original paper)
        top_n: number of final results to return
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}  # id → full doc dict

    for rlist in ranked_lists:
        for rank, doc in enumerate(rlist, 1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    # sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids[:top_n]:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        results.append(doc)
    return results


# ═══════════════════════════════════════════════════════════
# Hybrid Retrieve (BM25 + Vector + RRF)
# ═══════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Hybrid retrieval: runs BOTH BM25 and dense vector search,
    then fuses results via Reciprocal Rank Fusion (RRF).
    """
    # run both retrieval strategies (fetch more candidates for better fusion)
    bm25_results = bm25_retrieve(query, top_k=10)
    vector_results = vector_retrieve(query, top_k=10)

    # fuse with RRF
    fused = reciprocal_rank_fusion([bm25_results, vector_results], top_n=top_k)
    return fused


# ═══════════════════════════════════════════════════════════
# Groq LLM Caller (with key rotation)
# ═══════════════════════════════════════════════════════════

def _call_groq(messages: list[dict], retries: int = len(GROQ_API_KEYS)) -> str:
    """Call Groq LLM with automatic key rotation on rate-limit."""
    global _key_index
    last_error = None
    for attempt in range(retries):
        key = GROQ_API_KEYS[_key_index % len(GROQ_API_KEYS)]
        client = Groq(api_key=key)
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,  # lower for stricter grounding
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
    End-to-end Hybrid RAG:
      1. Hybrid retrieve (BM25 + Vector + RRF)
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
    # quick CLI test
    print("=" * 60)
    print("  MedAssist AI – Hybrid Search Engine")
    print("  (BM25 + Dense Vector + RRF)")
    print("=" * 60)

    # test with a misspelled query
    test_queries = [
        "Do you have Paracetmol?",       # typo: Paracetmol → Paracetamol
        "Ibuprofin for pain",              # typo: Ibuprofin → Ibuprofen
        "my head hurts what should i take", # intent-based
        "Amoxicilin for infection",         # typo: Amoxicilin → Amoxicillin
    ]
    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        docs = retrieve(q)
        print(f"   Top result: {docs[0]['metadata']['medicine_name']} "
              f"(RRF={docs[0].get('rrf_score', 'N/A')})")
        for d in docs:
            print(f"     - {d['id']}: {d['metadata']['medicine_name']}")

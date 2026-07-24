"""
Ingestion Pipeline
──────────────────
Reads medicines.csv → converts each row to a searchable text document →
generates embeddings with sentence-transformers → upserts into ChromaDB Cloud.
"""

import os, sys, pathlib
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ── make project root importable ──
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import (
    CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE,
    CHROMA_COLLECTION, EMBEDDING_MODEL, CSV_PATH,
)


def _row_to_document(row: pd.Series) -> str:
    """Convert a single CSV row into a rich text paragraph for embedding."""
    stock_status = "currently in stock" if row["Stock"].strip().lower() == "yes" else "currently out of stock"
    return (
        f"{row['Medicine_Name']} {row['Strength']} is used for {row['Use_Case']}. "
        f"It is {stock_status}. "
        f"Alternative medicines: {row['Alternative']}. "
        f"Dosage instruction: {row['Dosage_Instruction']}."
    )


def ingest(csv_path: str | None = None):
    """Main ingestion function."""
    csv_path = csv_path or CSV_PATH
    # resolve relative to project root
    csv_full = pathlib.Path(__file__).resolve().parent / csv_path
    print(f"📄 Reading CSV from {csv_full} …")
    df = pd.read_csv(csv_full)
    print(f"   Found {len(df)} rows.")

    # ── build documents ──
    documents = []
    ids = []
    metadatas = []
    for _, row in df.iterrows():
        doc_text = _row_to_document(row)
        documents.append(doc_text)
        ids.append(f"med_{int(row['Medicine_ID'])}")
        metadatas.append({
            "medicine_name": str(row["Medicine_Name"]),
            "strength": str(row["Strength"]),
            "use_case": str(row["Use_Case"]),
            "alternative": str(row["Alternative"]),
            "stock": str(row["Stock"]).strip(),
            "dosage": str(row["Dosage_Instruction"]),
        })

    # ── generate embeddings ──
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL} …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(documents).tolist()
    print(f"   Generated {len(embeddings)} embeddings (dim={len(embeddings[0])}).")

    # ── connect to ChromaDB Cloud ──
    print("☁️  Connecting to ChromaDB Cloud …")
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

    # get or create collection
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # ── upsert ──
    print(f"⬆️  Upserting {len(documents)} documents into collection '{CHROMA_COLLECTION}' …")
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print("✅ Ingestion complete!")
    return len(documents)


if __name__ == "__main__":
    ingest()

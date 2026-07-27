"""
Configuration for the Hospital Chatbot AI Agent.
- Groq API keys with rotation
- ChromaDB Cloud credentials
- Model settings
"""

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Groq API Keys – rotated automatically on 429 rate-limit errors
# ---------------------------------------------------------------------------
# Dynamically load all keys starting with GROQ_API_KEY_
GROQ_API_KEYS = [
    val for key, val in os.environ.items() if key.startswith("GROQ_API_KEY_") and val
]

# Fallback if no keys found
if not GROQ_API_KEYS:
    GROQ_API_KEYS = ["gsk_placeholder"]

# ---------------------------------------------------------------------------
# Groq Model
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# ChromaDB Cloud
# ---------------------------------------------------------------------------
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY", "placeholder")
CHROMA_TENANT = os.environ.get("CHROMA_TENANT", "placeholder")
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE", "placeholder")
CHROMA_COLLECTION = "medicines"

# ---------------------------------------------------------------------------
# Embedding Model (runs locally – no API key needed)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# RAG Settings
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 5  # max documents after all filtering (safety cap)

# Cross-Encoder reranking (ms-marco outputs raw logits, not 0-1 probabilities)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_THRESHOLD = -4.0  # absolute floor: safety net only (blocks extreme mismatches)
CROSS_ENCODER_SCORE_GAP = 5.0   # drop docs scoring ≥5 points below the best candidate

# RRF fusion thresholds
RRF_MIN_SCORE_THRESHOLD = 0.02   # absolute floor
RRF_SCORE_DROP_RATIO = 0.6       # drop docs scoring < 60% of top doc's score

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
CSV_PATH = "data/medicines.csv"

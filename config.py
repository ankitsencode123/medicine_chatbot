"""
Configuration for the Hospital Chatbot AI Agent.
- Groq API keys with rotation
- ChromaDB Cloud credentials
- Model settings
"""

# ---------------------------------------------------------------------------
# Groq API Keys – rotated automatically on 429 rate-limit errors
# ---------------------------------------------------------------------------
GROQ_API_KEYS = [
    "gsk_your_groq_api_key_1",
    "gsk_your_groq_api_key_2",
    # Add more keys here if you want to use the rotation feature
]

# ---------------------------------------------------------------------------
# Groq Model
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# ChromaDB Cloud
# ---------------------------------------------------------------------------
CHROMA_API_KEY = "ck-cFbLLBXnFRApkqX9Uzm2wWBgjy9dWju68XheA89QeiT"
CHROMA_TENANT = "9c758a5a-143c-49b5-871e-b4e4fa1ddc66"
CHROMA_DATABASE = "hospital_agent"
CHROMA_COLLECTION = "medicines"

# ---------------------------------------------------------------------------
# Embedding Model (runs locally – no API key needed)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# RAG Settings
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 3  # number of documents to retrieve

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
CSV_PATH = "data/medicines.csv"

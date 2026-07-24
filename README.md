# 🏥 MedAssist AI – Hospital & Medical Shop Prescription Guidance

An AI-powered chatbot that helps hospital staff and patients with medicine queries using **Retrieval Augmented Generation (RAG)**.

## Features
- 💊 **Stock Check** – Instantly check if a medicine is in stock
- 🔄 **Alternatives** – Get generic/alternative medicine suggestions  
- 📋 **Dosage Guidance** – View dosage instructions for any medicine
- 🤖 **Natural Language** – Ask questions in plain English
- ⚡ **Fast Responses** – Powered by Groq API (LLaMA 3.3 70B)

## Tech Stack
| Component | Tool |
|-----------|------|
| LLM | Groq API (LLaMA 3.3 70B Versatile) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB Cloud |
| Framework | LangChain |
| UI | Streamlit |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest medicine data into ChromaDB Cloud
python ingest.py

# 3. Launch the chatbot
streamlit run app.py
```

## Sample Queries
- *"Do you have Paracetamol 500mg?"*
- *"What medicine is recommended for fever?"*
- *"Is Ibuprofen available? Any alternative?"*
- *"What antibiotic is used for throat infection?"*

## Project Structure
```
├── app.py           # Streamlit UI
├── config.py        # API keys & settings
├── ingest.py        # CSV → ChromaDB ingestion
├── rag_engine.py    # RAG retrieval + generation
├── data/
│   └── medicines.csv
├── requirements.txt
└── README.md
```

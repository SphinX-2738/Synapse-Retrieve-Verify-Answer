import os
from dotenv import load_dotenv

load_dotenv()

# ─── App Identity ────────────────────────────────────────────
APP_NAME = "Synapse"
APP_TAGLINE = "Retrieve. Verify. Answer."
VERSION = "1.0.0"

# ─── LLM Settings ────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1        # low = deterministic, better for RAG
LLM_MAX_TOKENS = 1024

# ─── Embedding Settings ──────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "gemini-embedding-001"   # latest stable Gemini embedding model
EMBED_MODE = os.getenv("EMBED_MODE", "gemini")
# "gemini" = Google Gemini API (~0MB RAM) ← production default
# "local"  = sentence-transformers (~400MB RAM, dev only)
# Why Gemini:
# - HF Inference API blocked by some Indian ISPs
# - Gemini free tier: 1500 requests/day, no ISP blocks in India
# - gemini-embedding-001: 3072 dims, state-of-the-art accuracy
# - Uses new google-genai SDK (google-generativeai is deprecated)

# ─── ChromaDB Settings ───────────────────────────────────────
CHROMA_PATH = "./chroma_store"
SESSION_EXPIRY_HOURS = 24

# ─── Chunking Settings ───────────────────────────────────────
CHUNK_SIZE = 300             # words per chunk
CHUNK_OVERLAP = 50           # word overlap between chunks

# ─── Agent Settings ──────────────────────────────────────────
MAX_AGENT_RETRIES = 2
RELEVANCE_THRESHOLD = 0.6    # min cosine similarity to accept a chunk
HALLUCINATION_PASS_SCORE = 7 # min score /10 to accept answer
TOP_K_CHUNKS = 5             # chunks retrieved per query

# ─── Memory Settings ─────────────────────────────────────────
MEMORY_WINDOW = 6
MEMORY_SUMMARY_THRESHOLD = 10

# ─── Sample Documents ────────────────────────────────────────
SAMPLE_DOCS_PATH = "./sample_documents"
SAMPLE_DOCS = [
    "attention_is_all_you_need.pdf",
    "rag_paper.pdf"
]

# ─── Output / Logging ────────────────────────────────────────
EVAL_LOGS_PATH = "./outputs/eval_logs"

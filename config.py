"""Central configuration for ScholarPanel."""

# --- Models (Groq) ---
MODEL_PRIMARY = "openai/gpt-oss-120b"   # deep reasoning agents + chair
MODEL_LIGHT = "openai/gpt-oss-20b"      # lighter agents (structure, writing)

# --- Embeddings (must match scripts/build_embeddings.py) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_PATH = "data/faiss_index/index.faiss"
META_PATH = "data/metadata.json"

# --- Limits (protect the free Groq quota) ---
MAX_UPLOAD_MB = 8
MIN_DOC_CHARS = 400
MAX_DOC_CHARS = 120_000       # hard cap on extracted text
EXCERPT_CHARS = 3500          # max characters of the document sent to each agent
TOP_K = 3                     # retrieved guideline chunks per agent
CHUNK_CHARS = 600             # max characters of each retrieved chunk sent to an LLM
MAX_TOKENS_REVIEWER = 1300
MAX_TOKENS_CHAIR = 1800

# --- Scoring weights (sum = 1.0) ---
WEIGHTS = {"structure": 0.15, "problem": 0.30, "method": 0.30, "writing": 0.25}

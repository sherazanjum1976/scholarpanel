# ScholarPanel - INDEXING PHASE (run once in Google Colab, NOT on Streamlit)
# ---------------------------------------------------------------------------
# Colab usage: paste each "# %%" block into its own cell, or upload this file
# and run:  %run build_embeddings.py
# Output: faiss.index + metadata.json packaged in scholarpanel_index.zip
# Then upload them to GitHub as:
#   data/faiss_index/index.faiss   (rename faiss.index -> index.faiss)
#   data/metadata.json
# ---------------------------------------------------------------------------

# %% 1. Install (same pinned versions as requirements.txt)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastembed==0.7.3",
                "faiss-cpu==1.12.0", "pymupdf==1.26.4", "python-docx==1.2.0", "numpy<2.3"], check=True)

# %% 2. Settings (EMBED_MODEL must equal config.EMBED_MODEL in the app)
import io, json, os, re, zipfile
import numpy as np
import faiss
import fitz
from docx import Document
from fastembed import TextEmbedding

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_WORDS, OVERLAP_WORDS = 130, 25      # ~600 characters per chunk
DEFAULT_CATEGORY = "Graduate Studies"

# %% 3. Upload documents (PDF / DOCX / TXT): handbook, thesis guidelines, ethics policy...
try:
    from google.colab import files
    print("Select your university documents...")
    uploaded = files.upload()
    DOCS = {name: data for name, data in uploaded.items()}
except ImportError:  # not in Colab: read ./docs_source
    DOCS = {f: open(os.path.join("docs_source", f), "rb").read() for f in os.listdir("docs_source")}

# %% 4. Extract text with page numbers
def pages_of(name, data):
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        return [(i + 1, p.get_text()) for i, p in enumerate(doc)]
    if ext == "docx":
        d = Document(io.BytesIO(data))
        return [(1, "\n".join(p.text for p in d.paragraphs))]
    return [(1, data.decode("utf-8", errors="ignore"))]

def guess_section(text):
    for line in text.split("\n"):
        s = line.strip()
        if 3 < len(s) < 80 and not s.endswith((".", ",")) and (s.isupper() or re.match(r"^\d+(\.\d+)*[.)\s]", s)):
            return s
    return "General"

# %% 5. Chunk
chunks = []
for name, data in DOCS.items():
    current_section = "General"
    for page, text in pages_of(name, data):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        current_section = guess_section(text) if guess_section(text) != "General" else current_section
        words = text.split()
        step = CHUNK_WORDS - OVERLAP_WORDS
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if len(piece) < 80:
                continue
            chunks.append({"id": f"c{len(chunks):05d}", "doc": name, "category": DEFAULT_CATEGORY,
                           "section": current_section, "page": page, "text": piece})
print(f"{len(chunks)} chunks from {len(DOCS)} documents")
assert chunks, "No text extracted - are the PDFs scanned images?"

# %% 6. Embed (plain embed() for BOTH indexing and querying, so preprocessing is identical)
model = TextEmbedding(EMBED_MODEL)
vecs = np.array(list(model.embed([c["text"] for c in chunks])), dtype="float32")
vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

# %% 7. Build FAISS index (inner product on normalised vectors = cosine similarity)
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, "faiss.index")
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=1)

# %% 8. Package + download
with zipfile.ZipFile("scholarpanel_index.zip", "w") as z:
    z.write("faiss.index", "index.faiss")
    z.write("metadata.json", "metadata.json")
print("Done. Download scholarpanel_index.zip, unzip it, then upload to GitHub:")
print("  index.faiss   -> data/faiss_index/index.faiss")
print("  metadata.json -> data/metadata.json")
try:
    files.download("scholarpanel_index.zip")
except NameError:
    pass

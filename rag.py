"""Lightweight RAG: pre-built FAISS index + metadata (runtime only; never rebuilds)."""
import json
import math
import os
import re

from config import CHUNK_CHARS, EMBED_MODEL, INDEX_PATH, META_PATH, TOP_K

_STOP = set("the and for with that this from are was were have has will shall must should their there which "
            "what when where into than then also such been being each other these those about over under".split())


def _tok(s):
    return [w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in _STOP]


class Retriever:
    """mode: 'faiss' (semantic) | 'keyword' (fallback over metadata chunks) | 'none'."""

    def __init__(self):
        self.chunks, self.index, self.embedder = [], None, None
        self.mode, self.note = "none", ""
        if not os.path.exists(META_PATH):
            self.note = "No knowledge base found (data/metadata.json missing). Reviews run without guideline retrieval."
            return
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        except Exception as e:  # corrupted metadata
            self.note = f"Could not read metadata.json ({e}). Running without retrieval."
            self.chunks = []
            return
        self.mode = "keyword"
        self.note = "Keyword retrieval over the bundled guideline chunks (no FAISS index found)."
        if os.path.exists(INDEX_PATH):
            try:
                import faiss
                from fastembed import TextEmbedding
                idx = faiss.read_index(INDEX_PATH)
                if idx.ntotal != len(self.chunks):
                    raise ValueError("index/metadata size mismatch - rebuild with scripts/build_embeddings.py")
                self.embedder = TextEmbedding(EMBED_MODEL)
                self.index, self.mode = idx, "faiss"
                self.note = f"Semantic retrieval: FAISS index ({idx.ntotal} chunks) + {EMBED_MODEL}."
            except Exception as e:
                self.index, self.embedder, self.mode = None, None, "keyword"
                self.note = f"FAISS unavailable ({str(e)[:120]}). Using keyword retrieval fallback."

    def search(self, query: str, k: int = TOP_K):
        if not self.chunks or not query.strip():
            return []
        try:
            if self.mode == "faiss":
                import numpy as np
                v = np.array(list(self.embedder.embed([query])), dtype="float32")
                v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
                scores, ids = self.index.search(v, k)
                hits = [(float(s), int(i)) for s, i in zip(scores[0], ids[0]) if i >= 0]
            else:
                q = _tok(query)
                hits = []
                for i, c in enumerate(self.chunks):
                    toks = _tok(c.get("text", "") + " " + c.get("section", ""))
                    if not toks:
                        continue
                    s = sum(toks.count(w) for w in set(q)) / math.sqrt(len(toks))
                    if s > 0:
                        hits.append((s, i))
                hits = sorted(hits, reverse=True)[:k]
        except Exception as e:
            self.note = f"Retrieval error: {str(e)[:120]}"
            return []
        out = []
        for s, i in hits:
            c = self.chunks[i]
            out.append({"id": c.get("id", str(i)), "doc": c.get("doc", "Unknown"),
                        "category": c.get("category", ""), "section": c.get("section", ""),
                        "page": c.get("page", "-"), "score": round(s, 3),
                        "text": c.get("text", "")[:CHUNK_CHARS]})
        return out

# 🎓 ScholarPanel - AI Review Panel for MS/PhD Synopses & Theses

Multi-agent system built with **CrewAI + Groq**. Four specialist reviewers run in parallel; a Panel Chair merges
their verdicts into a scored readiness report and a prioritised revision roadmap.
*Generative and Agentic AI Online Hackathon-II.*

## Agents (CrewAI)
| Agent | Job | Model |
|---|---|---|
| Document Structure Auditor | Format and required-section compliance | `openai/gpt-oss-20b` |
| Problem & Objectives Analyst | Gap, questions, objectives coherence | `openai/gpt-oss-120b` |
| Methodology & Ethics Evaluator | Design fit, sampling, ethics, timeline | `openai/gpt-oss-120b` |
| Academic Writing Reviewer | Clarity, literature depth, citations | `openai/gpt-oss-20b` |
| Panel Chair | Resolves conflicts, writes roadmap | `openai/gpt-oss-120b` |

The first four tasks use `async_execution=True`; the Chair task takes them as `context`. **5 LLM calls per run.**
Python (no LLM) does: text extraction, doc-type detection, section checklist, writing statistics, excerpt selection,
guideline retrieval, score aggregation and the advisory decision.

## Architecture
```
Upload/Paste -> Python pre-analysis (sections, stats, excerpts) -> RAG (FAISS) per agent
   -> [Structure | Problem | Method | Writing]  (parallel CrewAI tasks)
   -> Panel Chair (context = 4 reviews) -> Python scoring + decision -> Report (Word/MD)
```

## Deploy (browser only)
1. **GitHub:** New repository -> *Add file -> Upload files* -> drag in the **contents** of the extracted ZIP
   (so `app.py` is at the repo root) -> Commit.
2. **Streamlit Community Cloud:** share.streamlit.io -> *Create app* -> pick the repo, branch `main`,
   main file `app.py`.
3. **Advanced settings:** Python version **3.12**; in *Secrets* paste: `GROQ_API_KEY = "your_groq_key"`
4. **Deploy.** First start takes a few minutes (installs dependencies).
   (Add or change the secret later: app ⋮ -> Settings -> Secrets.)
5. **Test:** choose *Use sample synopsis* -> *Convene the panel*.

## Knowledge base (RAG)
* The repo ships `data/metadata.json` with **14 generic sample guideline chunks** so the app works immediately
  (keyword retrieval). **Replace them with your university's real rules** for the intended behaviour.
* **Indexing phase (once, Google Colab):** open Colab -> upload/paste `scripts/build_embeddings.py` -> run cells ->
  upload your handbook/thesis-guideline/ethics PDFs -> download `scholarpanel_index.zip`.
  Put `index.faiss` in `data/faiss_index/` and `metadata.json` in `data/` (replace existing) on GitHub.
* **Runtime phase:** the app only *loads* the index (`rag.py`); original documents are not needed.
  The same embedding model (`BAAI/bge-small-en-v1.5` via fastembed/ONNX, no PyTorch) is used for both phases.
  If the index and metadata sizes mismatch, the app falls back to keyword retrieval and says so in the UI.

## Free-tier safeguards
Per-agent excerpt ≤ 3,500 characters, 3 retrieved chunks (≤ 600 chars each), 5 calls per run, `max_iter=2`,
low reasoning effort, result cache per document (re-running the same document costs nothing),
automatic sequential retry after a rate-limit error, upload cap 8 MB. Use the sidebar *Sequential* mode if you
see rate-limit warnings. Check your current limits at console.groq.com.

## 5-minute demo
1. (30s) Show the 5 agent cards and explain parallel panel + chair.
2. (30s) *Use sample synopsis* (deliberately weak: vague title, non-SMART objectives, no ethics, no timeline).
3. (60s) Click *Convene the panel*; show live agent status.
4. (90s) Walk through score, radar chart, Panel Report roadmap, then the Problem and Methodology tabs.
5. (60s) *Sources & Checks* tab: Python section check (❌ ethics, timeline) and retrieved guideline citations.
6. (30s) Download the Word report; mention that the same document re-run uses the cache.

## Files
`app.py` UI · `crew.py` orchestration · `agent_*.py` agents/tasks · `rag.py` retrieval · `utils.py` analysis,
scoring, reports · `config.py` limits/models · `scripts/build_embeddings.py` Colab indexer.

*AI-generated advisory feedback only; final decisions rest with supervisors and committees.*

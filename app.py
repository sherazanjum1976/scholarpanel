"""ScholarPanel - Multi-agent MS/PhD synopsis & thesis review panel (Streamlit UI)."""
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import threading
import time

import streamlit as st

from groq_llm import get_key, key_name, ping, provider
from config import EXCERPT_CHARS, MAX_DOC_CHARS, MIN_DOC_CHARS, MODEL_LIGHT, MODEL_PRIMARY
from utils import (AGENT_LABELS, build_report_md, check_sections, compute_overall, decide,
                   detect_doc_type, extract_text, finalize_text, input_hash, parse_review,
                   pick_excerpt, report_to_docx, word_count, writing_stats)

st.set_page_config(page_title="ScholarPanel - AI Review Panel", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#312E81,#4F46E5 60%,#7C3AED);padding:1.6rem 2rem;border-radius:16px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:2.1rem;color:#fff}.hero p{margin:.3rem 0 0;opacity:.92;font-size:1.05rem}
.card{background:#F3F4F8;border-radius:12px;padding:.8rem .9rem;height:100%;border-top:4px solid #4F46E5;font-size:.85rem}
.card b{font-size:.95rem}.chair{border-top-color:#7C3AED;background:#EEEAFB}
.big{font-size:2.2rem;font-weight:700;color:#4F46E5;line-height:1.1}
</style>""", unsafe_allow_html=True)

REVIEWERS = ["structure", "problem", "method", "writing"]
ORDER = REVIEWERS + ["chair"]
ICONS = {"structure": "📑", "problem": "🎯", "method": "🔬", "writing": "✍️", "chair": "👨‍⚖️"}
AGENT_LABELS_ALL = {**AGENT_LABELS, "chair": "Panel Chair"}


# ------------------------------------------------------------------ helpers
def load_key() -> bool:
    for name in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "LLM_PROVIDER"):
        try:
            v = st.secrets.get(name)
        except Exception:
            v = None
        if v:
            os.environ[name] = str(v)
    return bool(get_key())


@st.cache_resource(show_spinner="Loading guideline knowledge base...")
def get_retriever():
    from rag import Retriever
    return Retriever()


def build_ctx(text, doc_type, level, retriever):
    checks, coverage = check_sections(text, doc_type)
    excerpts = {
        "structure": text[:2000],
        "problem": pick_excerpt(text, ["abstract", "introduction", "problem", "gap", "questions", "objectives"], (0, 0.35)),
        "method": pick_excerpt(text, ["method", "ethics", "timeline"], (0.3, 0.75)),
        "writing": pick_excerpt(text, ["abstract", "introduction", "literature"], (0, 0.5)),
    }
    queries = {
        "structure": f"{doc_type} required sections format length title abstract",
        "problem": f"{doc_type} problem statement research gap objectives research questions {level} originality",
        "method": "methodology sampling data collection ethics approval consent plagiarism timeline",
        "writing": "writing style citation referencing language formal academic similarity",
    }
    tags, sources, hits = {}, [], {}
    for k, q in queries.items():
        hits[k] = []
        for h in retriever.search(q):
            if h["id"] not in tags:
                tags[h["id"]] = f"S{len(tags) + 1}"
                sources.append({**h, "tag": tags[h["id"]]})
            hits[k].append({**h, "tag": tags[h["id"]]})
    return {"doc_type": doc_type, "level": level, "checks": checks, "coverage": coverage,
            "excerpts": excerpts, "stats": writing_stats(text), "hits": hits}, sources


def status_md(events, parallel):
    done = {e[1] for e in events if e[0] == "done"}
    notes = [e[1] for e in events if e[0] == "notice"]
    first_pending = next((k for k in REVIEWERS if k not in done), None)
    lines = []
    for k in ORDER:
        if k in done:
            s = "✅ finished"
        elif k == "chair":
            s = "⏳ synthesising panel report..." if all(r in done for r in REVIEWERS) else "⌛ waiting for the 4 reviewers"
        elif parallel or k == first_pending:
            s = "⏳ reviewing..."
        else:
            s = "⌛ queued"
        lines.append(f"{ICONS[k]} **{AGENT_LABELS_ALL[k]}** - {s}")
    return "\n\n".join(lines + [f"⚠️ {n}" for n in notes])


def radar_svg(scores: dict) -> str:
    cx = cy = 130
    R = 85
    axes = [("structure", 0, -1), ("problem", 1, 0), ("method", 0, 1), ("writing", -1, 0)]

    def pt(dx, dy, f):
        return f"{cx + dx * R * f:.1f},{cy + dy * R * f:.1f}"

    grid = "".join(
        f'<polygon points="{" ".join(pt(dx, dy, f) for _, dx, dy in axes)}" fill="none" stroke="#CBD5E1"/>'
        for f in (0.5, 1.0))
    lines = "".join(f'<line x1="{cx}" y1="{cy}" x2="{cx + dx * R}" y2="{cy + dy * R}" stroke="#CBD5E1"/>'
                    for _, dx, dy in axes)
    poly = " ".join(pt(dx, dy, max(scores.get(k, 0), 0.3) / 10) for k, dx, dy in axes)
    labels = [("Structure", cx, 18), ("Problem", 232, cy + 4), ("Method", cx, 252), ("Writing", 28, cy + 4)]
    lab = "".join(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="12" fill="#334155" '
                  f'font-family="sans-serif">{t}</text>' for t, x, y in labels)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 270" width="260" height="270">'
            f'{grid}{lines}<polygon points="{poly}" fill="#4F46E5" fill-opacity="0.35" stroke="#4F46E5" '
            f'stroke-width="2"/>{lab}</svg>')


def render_result(res):
    m = res["meta"]
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.3])
    c1.markdown(f"<div class='big'>{m['overall']}/10</div>Overall readiness", unsafe_allow_html=True)
    c2.markdown(f"<div class='big' style='font-size:1.3rem;padding-top:.5rem'>{m['decision']}</div>Advisory decision",
                unsafe_allow_html=True)
    c3.markdown(f"<div class='big'>{round(m['coverage'] * 100)}%</div>Required-section coverage", unsafe_allow_html=True)
    c4.image(radar_svg(res["scores"]), width=230)
    st.download_button("⬇️ Download report (Word)", report_to_docx(res["md"]), "ScholarPanel_Report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    st.download_button("⬇️ Download report (Markdown)", res["md"], "ScholarPanel_Report.md", "text/markdown")

    tabs = st.tabs(["👨‍⚖️ Panel Report"] + [f"{ICONS[k]} {AGENT_LABELS[k].split(' & ')[0]}" for k in REVIEWERS]
                   + ["📚 Sources & Checks"])
    with tabs[0]:
        st.markdown(res["chair"])
    for t, k in zip(tabs[1:5], REVIEWERS):
        with t:
            r = res["reviews"][k]
            st.subheader(f"{AGENT_LABELS[k]} - {res['scores'][k]}/10")
            if not r["parsed"]:
                st.warning("This agent's output could not be parsed as structured JSON; raw text shown. "
                           "A neutral score of 5 was used in aggregation.")
            st.write(r["summary"])
            for s in r["strengths"]:
                st.markdown(f"✅ {s}")
            for i in r["issues"]:
                sev = str(i.get("severity", "")).lower()
                icon = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(sev, "⚪")
                src = f" · source: {i['source']}" if i.get("source") else ""
                st.markdown(f"{icon} **{i.get('issue', '')}**{src}  \n"
                            f"*Evidence:* {i.get('evidence', '-')}  \n*Fix:* {i.get('fix', '-')}")
    with tabs[5]:
        st.markdown("**Automated section check (Python, no LLM)**")
        st.dataframe([{"Required section": r["label"], "Found": "✅" if r["present"] else "❌"}
                      for r in res["checks"]], hide_index=True, use_container_width=True)
        st.markdown("**Measured writing statistics**")
        st.json(res["stats"])
        st.markdown(f"**Retrieved guideline sources** - _{res['retrieval_note']}_")
        if not res["sources"]:
            st.info("No guideline chunks were retrieved for this run.")
        for s in res["sources"]:
            with st.expander(f"[{s['tag']}] {s['doc']} - {s['section']} (p. {s['page']}) · score {s['score']}"):
                st.write(s["text"])


# ------------------------------------------------------------------ sidebar
has_key = load_key()
with st.sidebar:
    st.header("⚙️ Settings")
    type_choice = st.selectbox("Document type", ["Auto-detect", "Synopsis / Proposal", "Thesis / Chapters"])
    level = st.radio("Degree level", ["MS", "PhD"], horizontal=True)
    mode = st.radio("Execution", ["Parallel (faster)", "Sequential (gentler on rate limits)"])
    st.divider()
    st.markdown("**API status**")
    if has_key:
        st.success(f"{key_name()} detected (provider: {provider()})")
        if st.button("🔌 Test connection"):
            with st.spinner("Contacting the LLM provider..."):
                ok, msg = ping()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    else:
        st.error(f"{key_name()} missing")
        st.caption("Streamlit Cloud → your app → ⋮ → Settings → Secrets, then add:\n\n"
                   "`GROQ_API_KEY = \"your_key_here\"`")
    st.markdown("**Models**")
    st.caption(f"Primary: `{MODEL_PRIMARY}` (Problem, Methodology, Chair)\n\n"
               f"Light: `{MODEL_LIGHT}` (Structure, Writing)")
    st.caption(f"Free-tier safeguards: ≤{EXCERPT_CHARS} chars per agent, 3 guideline chunks, "
               "5 LLM calls per run, results cached per document.")
    st.caption("🔒 Documents are processed in memory and never stored. Relevant excerpts are sent to Groq.")

# ------------------------------------------------------------------ header
st.markdown("""<div class='hero'><h1>🎓 ScholarPanel</h1>
<p>An AI review committee for MS/PhD synopses and theses - four specialist reviewers work in parallel,
a Panel Chair merges their verdicts into a scored, prioritised revision roadmap.</p></div>""", unsafe_allow_html=True)

cols = st.columns(5)
cards = [("structure", "Checks format & required sections"), ("problem", "Tests gap, questions & objectives"),
         ("method", "Tests design, sampling, ethics"), ("writing", "Reviews clarity, literature & citations"),
         ("chair", "Resolves conflicts, writes the roadmap")]
for col, (k, d) in zip(cols, cards):
    col.markdown(f"<div class='card {'chair' if k == 'chair' else ''}'>{ICONS[k]} <b>{AGENT_LABELS_ALL[k]}</b><br>{d}</div>",
                 unsafe_allow_html=True)

with st.expander("ℹ️ Problem & how to use", expanded=False):
    st.markdown("Supervisors and committees repeat the same feedback on weak research gaps, vague objectives, "
                "mismatched methods and inconsistent writing. ScholarPanel gives students a structured, "
                "guideline-grounded pre-review. **Steps:** 1) upload/paste a document (or use the sample), "
                "2) click *Convene the panel*, 3) read the report and download it. "
                "This is an advisory tool; final decisions remain with the supervisor and committee.")

# ------------------------------------------------------------------ input
st.subheader("1 · Provide the document")
src = st.radio("Source", ["Upload file", "Paste text", "Use sample synopsis"], horizontal=True, label_visibility="collapsed")
text, truncated = None, False
try:
    if src == "Upload file":
        up = st.file_uploader("PDF, DOCX or TXT (max 8 MB)", type=["pdf", "docx", "txt"])
        if up is not None:
            text, truncated = extract_text(up.name, up.getvalue())
    elif src == "Paste text":
        pasted = st.text_area("Paste your synopsis or thesis text", height=220)
        if pasted.strip():
            text, truncated = finalize_text(pasted)
    else:
        with open("data/sample_synopsis.txt", "r", encoding="utf-8") as f:
            text, truncated = finalize_text(f.read())
        st.info("Sample synopsis loaded. It deliberately contains weaknesses so you can see the panel at work.")
        with st.expander("Preview sample"):
            st.text(text)
except ValueError as e:
    st.error(str(e))
except OSError:
    st.error("Sample file is missing from the repository (data/sample_synopsis.txt).")

if text:
    auto = detect_doc_type(text)
    doc_type = auto if type_choice == "Auto-detect" else ("synopsis" if type_choice.startswith("Synopsis") else "thesis")
    st.caption(f"📄 {word_count(text):,} words · type: **{doc_type}**"
               f"{' (auto-detected)' if type_choice == 'Auto-detect' else ''} · level: **{level}**")
    if truncated:
        st.warning(f"Document was long; only the first {MAX_DOC_CHARS:,} characters are analysed.")

# ------------------------------------------------------------------ run
st.subheader("2 · Convene the panel")
go = st.button("🚀 Convene the panel", type="primary", disabled=not text)
if not text:
    st.caption(f"Provide at least {MIN_DOC_CHARS} characters of text to enable the panel.")

if go and text:
    if not has_key:
        st.error(f"{key_name()} is not configured. See the sidebar for instructions.")
    else:
        parallel = mode.startswith("Parallel")
        key = input_hash(text, doc_type, level, parallel)
        cache = st.session_state.setdefault("cache", {})
        if key in cache:
            st.session_state["result"] = cache[key]
            st.success("Same document already reviewed - showing the saved result (no API calls used).")
        else:
            try:
                from crew import run_panel
            except Exception as e:
                st.error(f"Could not load the CrewAI module: {e}")
                st.stop()
            retriever = get_retriever()
            ctx, sources = build_ctx(text, doc_type, level, retriever)
            st.info(f"Pre-analysis done in Python: {round(ctx['coverage'] * 100)}% required-section coverage, "
                    f"{len(sources)} guideline chunks retrieved ({retriever.mode} mode). Starting agents...")
            events, box, holder = [], st.empty(), {}

            def worker():
                try:
                    holder["out"] = run_panel(ctx, events, parallel=parallel)
                except Exception as e:  # PanelError or unexpected
                    holder["err"] = str(e)

            th = threading.Thread(target=worker, daemon=True)
            th.start()
            while th.is_alive():
                box.markdown(status_md(events, parallel))
                time.sleep(0.7)
            th.join()
            box.markdown(status_md(events, parallel))
            if "err" in holder:
                st.error(holder["err"])
            else:
                out = holder["out"]
                reviews = {k: parse_review(out.get(k, "")) for k in REVIEWERS}
                overall, scores = compute_overall({k: reviews[k]["score"] for k in REVIEWERS}, ctx["coverage"])
                decision = decide(overall, ctx["coverage"])
                chair = (out.get("chair") or "").strip() or \
                    "## Panel Summary\nThe Panel Chair returned no output. Please see the individual reviewer findings."
                meta = {"doc_type": doc_type, "level": level, "words": word_count(text), "overall": overall,
                        "decision": decision, "coverage": ctx["coverage"]}
                res = {"meta": meta, "scores": scores, "reviews": reviews, "chair": chair, "sources": sources,
                       "checks": ctx["checks"], "stats": ctx["stats"], "retrieval_note": retriever.note,
                       "md": build_report_md(meta, scores, reviews, chair, sources)}
                cache[key] = res
                st.session_state["result"] = res

if "result" in st.session_state:
    st.subheader("3 · Panel verdict")
    render_result(st.session_state["result"])

st.divider()
st.caption("ScholarPanel · CrewAI + Groq · AI-generated advisory feedback only - not a substitute for supervisor "
           "or committee judgment.")

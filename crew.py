"""CrewAI orchestration: 4 reviewers (parallel) -> Panel Chair."""
import time

from crewai import Crew, Process

import agent_chair
import agent_method
import agent_problem
import agent_structure
import agent_writing
from groq_llm import GroqLLM, get_key, key_name
from config import MAX_TOKENS_CHAIR, MAX_TOKENS_REVIEWER, MODEL_LIGHT, MODEL_PRIMARY

# CrewAI shows an interactive 20s "view traces?" prompt on its first run. There is no terminal on
# Streamlit Cloud, so record "first run done, no consent" up front to skip the prompt entirely.
try:
    from crewai.events.listeners.tracing.utils import mark_first_execution_done
    mark_first_execution_done(user_consented=False)
except Exception:  # never let this cosmetic step break the app
    pass

# Which model each agent uses (and why): see README / UI.
AGENT_MODELS = {
    "structure": MODEL_LIGHT,    # mostly checklist logic already computed in Python
    "problem": MODEL_PRIMARY,    # deepest reasoning
    "method": MODEL_PRIMARY,     # deepest reasoning
    "writing": MODEL_LIGHT,      # language feedback, lighter model is sufficient
    "chair": MODEL_PRIMARY,      # synthesis and conflict resolution
}


class PanelError(Exception):
    """Error with a message that is safe to show to the user."""


def _llm(model: str, max_tokens: int) -> GroqLLM:
    key = get_key()
    if not key:
        raise PanelError(f"{key_name()} is missing. Add it in Streamlit Cloud > App settings > Secrets.")
    return GroqLLM(model=model, api_key=key, max_tokens=max_tokens)


def _friendly(e: Exception) -> str:
    m = str(e).lower()
    if "access denied" in m or "network settings" in m or "403" in m:
        return ("Groq blocked the request (HTTP 403 'Access denied'). This is a network/IP block on Groq's side, "
                "not a bug in the app and not a key problem. Use the sidebar 'Test connection' button, reboot the app, "
                "or switch provider/host (see README). Details: " + str(e)[:160])
    if "401" in m or "invalid api key" in m or "authentication" in m:
        return "Groq rejected the API key. Check GROQ_API_KEY in your Streamlit secrets."
    if "429" in m or "rate limit" in m or "rate_limit" in m:
        return "Groq free-tier rate limit reached. Wait about a minute and try again."
    if "timeout" in m or "connection" in m:
        return "Network/timeout error while contacting Groq. Please try again."
    return f"The agent crew failed: {str(e)[:300]}"


def _is_rate_limit(e: Exception) -> bool:
    m = str(e).lower()
    return "429" in m or "rate limit" in m or "rate_limit" in m


def _run(ctx: dict, parallel: bool, events: list) -> dict:
    def cb(key):
        return lambda _out: events.append(("done", key, time.time()))

    llm_small = {k: _llm(m, MAX_TOKENS_REVIEWER) for k, m in AGENT_MODELS.items() if k != "chair"}
    chair_llm = _llm(AGENT_MODELS["chair"], MAX_TOKENS_CHAIR)

    mods = {"structure": agent_structure, "problem": agent_problem,
            "method": agent_method, "writing": agent_writing}
    reviewers, tasks = [], {}
    for key, mod in mods.items():
        agent = mod.build_agent(llm_small[key])
        reviewers.append(agent)
        tasks[key] = mod.build_task(agent, ctx, callback=cb(key), parallel=parallel)
    chair = agent_chair.build_agent(chair_llm)
    tasks["chair"] = agent_chair.build_task(chair, ctx, list(tasks.values()), callback=cb("chair"))

    crew = Crew(agents=reviewers + [chair], tasks=list(tasks.values()), process=Process.sequential,
                verbose=False, memory=False, cache=False, max_rpm=20, tracing=False)
    crew.kickoff()
    out = {}
    for key, t in tasks.items():
        out[key] = t.output.raw if getattr(t, "output", None) is not None else ""
    return out


def run_panel(ctx: dict, events: list, parallel: bool = True) -> dict:
    """Run the crew. Returns {agent_key: raw_text}. Raises PanelError on failure."""
    try:
        return _run(ctx, parallel, events)
    except PanelError:
        raise
    except Exception as e:
        if parallel and _is_rate_limit(e):
            events.append(("notice", "Rate limit hit - waiting 30s, then retrying agents one at a time...", time.time()))
            time.sleep(30)
            try:
                return _run(ctx, False, events)
            except Exception as e2:
                raise PanelError(_friendly(e2))
        raise PanelError(_friendly(e))

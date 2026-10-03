"""CrewAI LLM adapter. Default provider: Groq (official Groq SDK).

Optional fallback provider (only if you set it in Streamlit Secrets): Cerebras (OpenAI-compatible API).
  LLM_PROVIDER = "groq"      (default)  needs GROQ_API_KEY
  LLM_PROVIDER = "cerebras"             needs CEREBRAS_API_KEY  (hosts gpt-oss-120b only)
"""
import os

from crewai import BaseLLM
from groq import Groq

CEREBRAS_URL = "https://api.cerebras.ai/v1"


def provider() -> str:
    p = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    return p if p in ("groq", "cerebras") else "groq"


def key_name() -> str:
    return "CEREBRAS_API_KEY" if provider() == "cerebras" else "GROQ_API_KEY"


def get_key() -> str:
    return os.environ.get(key_name(), "")


def model_id(model: str) -> str:
    # Cerebras uses plain ids and only hosts the 120b model of the two we use.
    return "gpt-oss-120b" if provider() == "cerebras" else model


class GroqLLM(BaseLLM):
    def __init__(self, model: str, max_tokens: int = 1300, temperature: float = 0.2,
                 reasoning_effort: str = "low", api_key: str = None):
        super().__init__(model=model_id(model), temperature=temperature)
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        key = api_key or get_key()
        if provider() == "cerebras":
            from openai import OpenAI
            self.client = OpenAI(api_key=key, base_url=CEREBRAS_URL, max_retries=3, timeout=90.0)
        else:
            self.client = Groq(api_key=key, max_retries=3, timeout=90.0)

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        clean = [{"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages]
        kwargs = dict(model=self.model, messages=clean, temperature=self.temperature,
                      max_completion_tokens=self.max_tokens)
        if self.reasoning_effort and provider() == "groq":
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128000


def ping():
    """Tiny connectivity test. Returns (ok: bool, message: str)."""
    if not get_key():
        return False, f"{key_name()} is not set."
    try:
        out = GroqLLM("openai/gpt-oss-20b", max_tokens=50).call("Reply with the single word: ok")
        return True, f"{provider().title()} reachable. Model replied: {out.strip()[:40] or '(empty)'}"
    except Exception as e:
        return False, f"{provider().title()} call failed: {str(e)[:300]}"

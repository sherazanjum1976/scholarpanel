"""CrewAI LLM that talks to Groq through the OFFICIAL Groq Python SDK (no litellm in the call path)."""
import os

from crewai import BaseLLM
from groq import Groq


class GroqLLM(BaseLLM):
    def __init__(self, model: str, max_tokens: int = 1300, temperature: float = 0.2,
                 reasoning_effort: str = "low", api_key: str = None):
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"), max_retries=3, timeout=90.0)

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        clean = [{"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages]
        kwargs = dict(model=self.model, messages=clean, temperature=self.temperature,
                      max_completion_tokens=self.max_tokens)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128000

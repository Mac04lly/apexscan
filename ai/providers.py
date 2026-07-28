"""ai/providers.py — thin OpenAI wrapper. Never raises; returns "" on any failure."""
from __future__ import annotations
import logging
import requests

log = logging.getLogger("apexscan.ai.providers")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 temperature: float = 0.2, max_tokens: int = 600):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str, system: str = None) -> str:
        if not self.api_key:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = requests.post(
                OPENAI_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=25,
            )
            if resp.status_code == 401:
                log.warning("OpenAI auth failed — check openai_api_key. AI disabled for this call.")
                return ""
            if resp.status_code == 429:
                log.warning("OpenAI rate limited — skipping this AI call, scanner continues.")
                return ""
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"OpenAI call failed, degrading gracefully: {e}")
            return ""

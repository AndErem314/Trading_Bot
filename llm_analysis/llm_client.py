import json
from typing import Optional, Dict, Any
import os

from .env_loader import LLMConfig

class LLMClient:
    """Unified client for OpenAI and Gemini."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def _gen_openai(self, prompt: str, model: str) -> str:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("OpenAI Python SDK is not installed. pip install openai") from e
        api_key = self.config.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Create a .env in the project root with OPENAI_API_KEY=... or export it in your shell."
            )
        client = OpenAI(api_key=api_key)
        # Use chat completions for broad compatibility
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""
        return content

    def _gen_gemini(self, prompt: str, model: str) -> str:
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("Google Generative AI SDK is not installed. pip install google-generativeai") from e
        api_key = self.config.gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Create a .env in the project root or export it in your shell."
            )
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(prompt)
        # gemini returns candidates; take the text
        try:
            return resp.text or ""
        except Exception:
            return ""

    def generate(self, prompt: str, provider: Optional[str] = None, model_override: Optional[str] = None) -> str:
        prov = (provider or self.config.provider).lower()
        if prov not in ("openai", "gemini"):
            # fallback based on available keys
            prov = "openai" if self.config.openai_api_key else "gemini"
        if prov == "openai":
            model = model_override or self.config.openai_model or "gpt-4o-mini"
            return self._gen_openai(prompt, model)
        else:
            model = model_override or self.config.gemini_model or "gemini-2.5-pro"
            return self._gen_gemini(prompt, model)

from typing import Optional


def _approx_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token for English-like text
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _openai_count(text: str, model: Optional[str]) -> int:
    try:
        import tiktoken  # type: ignore
        enc = None
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                pass
        if enc is None:
            # Fallback to a common encoding compatible with GPT-3.5/4 families
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return _approx_tokens(text or "")


def _gemini_count(text: str, model: Optional[str]) -> int:
    # Prefer SDK token counting if available/configured; otherwise approximate
    try:
        import google.generativeai as genai  # type: ignore
        # If not configured here, assume caller configured before generate();
        # count_tokens should still work with global configuration.
        m = genai.GenerativeModel(model or "gemini-2.5-pro")
        resp = m.count_tokens(text or "")
        # Some SDK versions return .total_tokens; fallback to summing fields if needed
        total = getattr(resp, "total_tokens", None)
        if total is None:
            try:
                total = int(resp["total_tokens"])  # type: ignore[index]
            except Exception:
                total = None
        return int(total) if total is not None else _approx_tokens(text or "")
    except Exception:
        return _approx_tokens(text or "")


def count_tokens(text: str, provider: str, model: Optional[str]) -> int:
    prov = (provider or "openai").lower()
    if prov == "openai":
        return _openai_count(text, model)
    elif prov == "gemini":
        return _gemini_count(text, model)
    # Default fallback
    return _approx_tokens(text or "")

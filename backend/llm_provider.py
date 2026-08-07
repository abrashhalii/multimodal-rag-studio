"""
LLM factory.

Two providers behind one interface so the demo has a fallback. If the venue's
wifi dies during the live screen share you flip LLM_PROVIDER=ollama in .env and
restart - you lose some answer quality, not the whole evaluation.

get_llm()      -> the strong model, for answering
get_fast_llm() -> the cheap model, for mechanical work (query rewriting, routing)
"""

import functools

import config


class LLMNotConfigured(RuntimeError):
    pass


def _build(model_name: str):
    provider = config.LLM_PROVIDER

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise LLMNotConfigured(
                "GEMINI_API_KEY is empty. Put your key in backend/.env, or set "
                "LLM_PROVIDER=ollama to run fully offline."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
            timeout=90,
            max_retries=3,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            num_predict=config.LLM_MAX_TOKENS,
        )

    raise LLMNotConfigured(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'ollama'.")


@functools.lru_cache(maxsize=1)
def get_llm():
    name = config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini" else config.OLLAMA_MODEL
    return _build(name)


@functools.lru_cache(maxsize=1)
def get_fast_llm():
    """Cheaper model for query rewriting and classification.

    Rewriting a follow-up into a standalone question is a mechanical task - it
    does not need the strong model, and on Gemini's free tier Flash-Lite has a
    higher RPM ceiling than Flash. This is where model tiering actually pays.
    """
    name = (config.GEMINI_FAST_MODEL if config.LLM_PROVIDER == "gemini"
            else config.OLLAMA_FAST_MODEL)
    return _build(name)


def describe() -> str:
    if config.LLM_PROVIDER == "gemini":
        return f"gemini ({config.GEMINI_MODEL} / {config.GEMINI_FAST_MODEL})"
    return f"ollama ({config.OLLAMA_MODEL} @ {config.OLLAMA_BASE_URL})"

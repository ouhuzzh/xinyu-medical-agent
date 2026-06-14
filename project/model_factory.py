import config


def _require_setting(value, name, provider):
    if value:
        return value
    raise ValueError(f"Missing required setting `{name}` for provider `{provider}`.")


def _create_openai_chat(model, temperature, api_key, base_url=None, *,
                         max_tokens: int = 0, timeout: float = 0):
    from langchain_openai import ChatOpenAI

    timeout_sec = timeout if timeout > 0 else config.LLM_TIMEOUT_SECONDS
    max_tok = max_tokens if max_tokens > 0 else config.LLM_MAX_TOKENS

    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "timeout": timeout_sec,
        "max_retries": 2,
    }
    # OpenAI renamed max_tokens → max_completion_tokens; send both for compat
    if max_tok > 0:
        kwargs["max_tokens"] = max_tok
        kwargs["max_completion_tokens"] = max_tok
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        kwargs.pop("max_tokens", None)
        kwargs.pop("max_completion_tokens", None)
        kwargs.pop("timeout", None)
        kwargs.pop("max_retries", None)
        return ChatOpenAI(**kwargs)


def _create_openai_embeddings(model, api_key, base_url=None):
    from langchain_openai import OpenAIEmbeddings

    kwargs = {
        "model": model,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


def get_chat_model(provider=None):
    provider_name = (provider or config.ACTIVE_LLM_PROVIDER).lower()

    if provider_name == "deepseek":
        api_key = _require_setting(config.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY", provider_name)
        return _create_openai_chat(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=api_key,
            base_url=config.DEEPSEEK_BASE_URL,
        )

    if provider_name == "openai":
        api_key = _require_setting(config.OPENAI_API_KEY, "OPENAI_API_KEY", provider_name)
        return _create_openai_chat(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=api_key,
            base_url=config.OPENAI_BASE_URL or None,
        )

    if provider_name == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "base_url": config.OLLAMA_BASE_URL,
            "timeout": config.LLM_TIMEOUT_SECONDS,
        }
        try:
            return ChatOllama(**kwargs)
        except TypeError:
            kwargs.pop("timeout", None)
            return ChatOllama(**kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider_name}")


def get_chat_model_for_tier(*, provider: str, model: str, temperature: float,
                            timeout: float, max_tokens: int):
    """Create a chat model with explicit tier parameters.

    Reuses the same provider logic as :func:`get_chat_model` but allows
    overriding ``model``, ``temperature``, ``timeout`` and ``max_tokens``
    per tier — without touching global config.
    """
    provider_name = provider.lower()

    if provider_name == "deepseek":
        api_key = _require_setting(config.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY", provider_name)
        return _create_openai_chat(
            model=model, temperature=temperature, api_key=api_key,
            base_url=config.DEEPSEEK_BASE_URL,
            max_tokens=max_tokens, timeout=timeout,
        )

    if provider_name == "openai":
        api_key = _require_setting(config.OPENAI_API_KEY, "OPENAI_API_KEY", provider_name)
        return _create_openai_chat(
            model=model, temperature=temperature, api_key=api_key,
            base_url=config.OPENAI_BASE_URL or None,
            max_tokens=max_tokens, timeout=timeout,
        )

    if provider_name == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": model,
            "temperature": temperature,
            "base_url": config.OLLAMA_BASE_URL,
            "timeout": timeout,
        }
        try:
            return ChatOllama(**kwargs)
        except TypeError:
            kwargs.pop("timeout", None)
            return ChatOllama(**kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider_name}")


_embedding_singleton = None
_embedding_singleton_lock = __import__("threading").Lock()


def get_embedding_model(provider=None):
    """Return the global embedding model singleton. ~450 MB RAM savings
    when shared across all stores (vs each store creating its own)."""
    global _embedding_singleton
    if _embedding_singleton is not None:
        return _embedding_singleton
    with _embedding_singleton_lock:
        if _embedding_singleton is not None:
            return _embedding_singleton
        _embedding_singleton = _create_embedding_model(provider)
        return _embedding_singleton


def _create_embedding_model(provider=None):
    provider_name = (provider or config.ACTIVE_EMBEDDING_PROVIDER).lower()

    if provider_name in {"openai", "openai_compatible"}:
        api_key = _require_setting(config.OPENAI_API_KEY, "OPENAI_API_KEY", provider_name)
        return _create_openai_embeddings(
            model=config.EMBEDDING_MODEL,
            api_key=api_key,
            base_url=config.OPENAI_BASE_URL or None,
        )

    if provider_name == "deepseek":
        api_key = _require_setting(config.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY", provider_name)
        return _create_openai_embeddings(
            model=config.EMBEDDING_MODEL,
            api_key=api_key,
            base_url=config.DEEPSEEK_BASE_URL,
        )

    if provider_name == "huggingface_local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        except ImportError as exc:
            raise RuntimeError(
                "Embedding provider `huggingface_local` requires local ML dependencies. "
                "Install the full requirements or build Docker with INSTALL_LOCAL_ML=true."
            ) from exc

    if provider_name == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)

    raise ValueError(f"Unsupported embedding provider: {provider_name}")

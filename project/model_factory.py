import config


def _require_setting(value, name, provider):
    if value:
        return value
    raise ValueError(f"Missing required setting `{name}` for provider `{provider}`.")


def _create_openai_chat(model, temperature, api_key, base_url=None):
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "timeout": config.LLM_TIMEOUT_SECONDS,
        "max_retries": 2,
        "extra_body": {
            "enable_thinking": config.OPENAI_ENABLE_THINKING,
            "thinking_budget": config.OPENAI_THINKING_BUDGET,
        },
    }
    if config.LLM_MAX_TOKENS > 0:
        kwargs["max_tokens"] = config.LLM_MAX_TOKENS
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        # Some OpenAI-compatible wrappers reject provider-specific kwargs.
        # Keep startup robust and fall back to the minimum portable set.
        kwargs.pop("max_tokens", None)
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
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=config.DEEPSEEK_BASE_URL,
        )

    if provider_name == "openai":
        api_key = _require_setting(config.OPENAI_API_KEY, "OPENAI_API_KEY", provider_name)
        return _create_openai_chat(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=config.OPENAI_BASE_URL or None,
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


def get_embedding_model(provider=None):
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
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    if provider_name == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)

    raise ValueError(f"Unsupported embedding provider: {provider_name}")

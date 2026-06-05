"""Tiered LLM router with per-provider circuit-breaker protection.

When ``LLM_TIERS_JSON`` is configured in the environment, different graph
nodes can request a *light* or *strong* model tier.  If a provider's circuit
breaker is open, the router automatically falls back to the
``LLM_FALLBACK_PROVIDER``.

If ``LLM_TIERS_JSON`` is empty or invalid, the router degrades to a
single-tier mode that simply returns ``get_chat_model()``, keeping full
backward compatibility.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Three-state circuit breaker: *closed* → *open* → *half_open* → *closed*.

    Thread-safe via ``threading.Lock``.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._state: str = "closed"  # closed | open | half_open
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    # -- public helpers -----------------------------------------------------

    @property
    def state(self) -> str:
        """Current circuit state (read-only snapshot)."""
        with self._lock:
            # Auto-transition open → half_open after recovery timeout
            if self._state == "open":
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "half_open"
            return self._state

    def allow_request(self) -> bool:
        """Return *True* if a request may proceed.

        - *closed*: always allowed
        - *open*: never allowed (but auto-transitions to *half_open* after timeout)
        - *half_open*: allowed (probe request)
        """
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "half_open"
                    return True  # probe
                return False
            # half_open: let one probe through
            return True

    def record_success(self) -> None:
        """Record a successful call; resets failure count, closes if *half_open*."""
        with self._lock:
            self._failure_count = 0
            if self._state == "half_open":
                self._state = "closed"
                logger.info("Circuit breaker recovered → closed")

    def record_failure(self) -> None:
        """Record a failure; opens the circuit when threshold is reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == "half_open":
                self._state = "open"
                logger.warning("Circuit breaker probe failed → open")
            elif self._failure_count >= self._failure_threshold:
                self._state = "open"
                logger.warning(
                    "Circuit breaker opened after %d consecutive failures",
                    self._failure_count,
                )


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMTierConfig:
    """Configuration for one model tier."""

    name: str  # "light", "strong", "default"
    provider: str  # "deepseek", "openai", "ollama"
    model: str  # e.g. "Qwen/Qwen3-8B"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 45.0


# ---------------------------------------------------------------------------
# Tiered LLM router
# ---------------------------------------------------------------------------

class TieredLLMRouter:
    """Routes LLM calls to model tiers with circuit-breaker protection."""

    def __init__(
        self,
        tiers: dict[str, LLMTierConfig],
        fallback_provider: Optional[str] = None,
    ):
        self._tiers = tiers
        self._fallback_provider = fallback_provider
        self._breakers: dict[str, CircuitBreaker] = {}
        self._llm_cache: dict[str, object] = {}
        self._lock = threading.Lock()

        # Pre-create one breaker per distinct provider
        for tier_cfg in tiers.values():
            if tier_cfg.provider not in self._breakers:
                self._breakers[tier_cfg.provider] = CircuitBreaker()
        if fallback_provider and fallback_provider not in self._breakers:
            self._breakers[fallback_provider] = CircuitBreaker()

    # -- factory -----------------------------------------------------------

    @classmethod
    def from_env(cls) -> "TieredLLMRouter":
        """Build router from environment variables.

        If ``LLM_TIERS_JSON`` is empty or unparseable, returns a single-tier
        router that delegates to ``get_chat_model()`` for full backward compat.
        """
        import config

        raw = config.LLM_TIERS_JSON.strip()
        fallback = config.LLM_FALLBACK_PROVIDER.strip() or None

        if not raw:
            # Single-tier mode — one "default" tier using current settings
            default_tier = LLMTierConfig(
                name="default",
                provider=config.ACTIVE_LLM_PROVIDER,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                timeout_seconds=config.LLM_TIMEOUT_SECONDS,
            )
            return cls(tiers={"default": default_tier}, fallback_provider=fallback)

        # Parse JSON array of tier configs
        try:
            tier_list = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "LLM_TIERS_JSON is not valid JSON; falling back to single-tier mode"
            )
            default_tier = LLMTierConfig(
                name="default",
                provider=config.ACTIVE_LLM_PROVIDER,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                timeout_seconds=config.LLM_TIMEOUT_SECONDS,
            )
            return cls(tiers={"default": default_tier}, fallback_provider=fallback)

        if not isinstance(tier_list, list) or not tier_list:
            logger.warning("LLM_TIERS_JSON must be a non-empty array; using single-tier")
            default_tier = LLMTierConfig(
                name="default",
                provider=config.ACTIVE_LLM_PROVIDER,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                timeout_seconds=config.LLM_TIMEOUT_SECONDS,
            )
            return cls(tiers={"default": default_tier}, fallback_provider=fallback)

        tiers: dict[str, LLMTierConfig] = {}
        for item in tier_list:
            try:
                cfg = LLMTierConfig(
                    name=str(item["name"]),
                    provider=str(item["provider"]),
                    model=str(item["model"]),
                    temperature=float(item.get("temperature", 0.0)),
                    max_tokens=int(item.get("max_tokens", 2048)),
                    timeout_seconds=float(item.get("timeout_seconds", 45.0)),
                )
                tiers[cfg.name] = cfg
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid tier config %s: %s", item, exc)

        # Ensure a "default" tier always exists
        if "default" not in tiers:
            first_name = next(iter(tiers), None)
            if first_name:
                tiers["default"] = tiers[first_name]

        return cls(tiers=tiers, fallback_provider=fallback)

    # -- public API --------------------------------------------------------

    @property
    def has_tiers(self) -> bool:
        """Whether multiple tiers are configured (not just single "default")."""
        return len(self._tiers) > 1

    def get_llm(self, tier: str = "default"):
        """Get an LLM instance for *tier*, with circuit-breaker protection.

        If the primary provider's circuit is open, attempts the fallback
        provider.  Falls back to ``get_chat_model()`` when no tier config
        matches.
        """
        tier_cfg = self._tiers.get(tier)
        if tier_cfg is None:
            # Unknown tier — fall back to default
            if tier != "default":
                logger.debug("Unknown tier %r, falling back to default", tier)
                return self.get_llm("default")
            # Should not happen, but safety net
            from model_factory import get_chat_model
            return get_chat_model()

        provider = tier_cfg.provider
        breaker = self._breakers.get(provider)

        if breaker and not breaker.allow_request():
            # Primary provider circuit is open — try fallback
            if self._fallback_provider and self._fallback_provider != provider:
                logger.info(
                    "Provider %r circuit open, falling back to %r for tier %r",
                    provider,
                    self._fallback_provider,
                    tier,
                )
                return self._get_or_create_llm(
                    tier_cfg, override_provider=self._fallback_provider
                )
            # No fallback available — return the model anyway and let it fail
            logger.warning(
                "Provider %r circuit open and no fallback; request will likely fail",
                provider,
            )

        return self._get_or_create_llm(tier_cfg)

    def get_status(self) -> dict:
        """Return circuit-breaker status for all providers (for health endpoint)."""
        return {
            "tiers": {name: {"provider": cfg.provider, "model": cfg.model}
                      for name, cfg in self._tiers.items()},
            "circuit_breakers": {
                provider: {"state": breaker.state,
                           "failure_count": breaker._failure_count}
                for provider, breaker in self._breakers.items()
            },
            "fallback_provider": self._fallback_provider,
        }

    # -- internal ----------------------------------------------------------

    def _get_or_create_llm(
        self,
        tier_cfg: LLMTierConfig,
        override_provider: Optional[str] = None,
    ):
        """Create or return a cached LLM for the given tier (or override provider)."""
        provider = override_provider or tier_cfg.provider
        cache_key = f"{tier_cfg.name}:{provider}:{tier_cfg.model}"

        with self._lock:
            if cache_key in self._llm_cache:
                return self._llm_cache[cache_key]

        from model_factory import get_chat_model_for_tier

        llm = get_chat_model_for_tier(
            provider=provider,
            model=tier_cfg.model,
            temperature=tier_cfg.temperature,
            timeout=tier_cfg.timeout_seconds,
            max_tokens=tier_cfg.max_tokens,
        )

        # Wrap with circuit-breaker tracking if we have a breaker for the provider
        breaker = self._breakers.get(provider)
        if breaker:
            llm = _CircuitBreakerWrapper(llm, breaker, router=self, tier_cfg=tier_cfg)

        with self._lock:
            self._llm_cache[cache_key] = llm
        return llm


# ---------------------------------------------------------------------------
# Circuit-breaker wrapper around an LLM instance
# ---------------------------------------------------------------------------

class _CircuitBreakerWrapper:
    """Wraps an LLM instance so that every ``invoke`` / ``ainvoke`` call
    records success/failure with the circuit breaker.

    This is a transparent proxy — it delegates all attribute access to the
    wrapped LLM except for ``invoke`` and ``ainvoke``, which are augmented
    with breaker tracking.
    """

    def __init__(self, wrapped_llm, breaker: CircuitBreaker, *, router: TieredLLMRouter, tier_cfg: LLMTierConfig):
        object.__setattr__(self, "_wrapped", wrapped_llm)
        object.__setattr__(self, "_breaker", breaker)
        object.__setattr__(self, "_router", router)
        object.__setattr__(self, "_tier_cfg", tier_cfg)

    # -- invoke / ainvoke with breaker tracking ----------------------------

    def invoke(self, *args, **kwargs):
        try:
            result = self._wrapped.invoke(*args, **kwargs)
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            # If primary provider is now open and we have a fallback, retry
            if self._breaker.state == "open" and self._router._fallback_provider:
                fallback_llm = self._router.get_llm(self._tier_cfg.name)
                # Avoid infinite recursion: if the fallback is also this wrapper,
                # just re-raise
                if fallback_llm is not self:
                    logger.info("Retrying with fallback provider for tier %r", self._tier_cfg.name)
                    return fallback_llm.invoke(*args, **kwargs)
            raise

    async def ainvoke(self, *args, **kwargs):
        try:
            result = await self._wrapped.ainvoke(*args, **kwargs)
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            if self._breaker.state == "open" and self._router._fallback_provider:
                fallback_llm = self._router.get_llm(self._tier_cfg.name)
                if fallback_llm is not self:
                    logger.info("Retrying with fallback provider for tier %r", self._tier_cfg.name)
                    return await fallback_llm.ainvoke(*args, **kwargs)
            raise

    # -- transparent proxy for everything else ------------------------------

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        setattr(self._wrapped, name, value)

    def bind_tools(self, tools, **kwargs):
        """bind_tools must return a wrapper too so that invoke still tracks."""
        bound = self._wrapped.bind_tools(tools, **kwargs)
        return _CircuitBreakerWrapper(
            bound, self._breaker, router=self._router, tier_cfg=self._tier_cfg
        )

    def with_config(self, **kwargs):
        """with_config must return a wrapper too so that invoke still tracks."""
        configured = self._wrapped.with_config(**kwargs)
        return _CircuitBreakerWrapper(
            configured, self._breaker, router=self._router, tier_cfg=self._tier_cfg
        )

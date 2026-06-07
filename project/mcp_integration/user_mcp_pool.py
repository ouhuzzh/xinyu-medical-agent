"""Per-user MCP client pool.

For each authenticated user, lazily builds a MultiServerMCPClient that connects
to all of that user's bound hospitals. Caches the resulting tool list in memory
so repeated turns in the same session don't re-handshake.

Each hospital has its own CircuitBreaker — if hospital A's MCP server is down,
it's marked unhealthy and excluded from the next pool rebuild, but other
hospitals continue to work.

All MCP operations are async, but the rest of the project is sync. We bridge
via asyncio.run() inside short-lived calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import config
from llm_tiered_router import CircuitBreaker

from .hospital_registry import HospitalRegistry
from .user_hospital_store import UserHospitalStore

logger = logging.getLogger(__name__)

_NAMESPACE_SEP = config.MCP_TOOL_NAMESPACE_SEPARATOR


class _UserPool:
    """Holds one user's cached tools + per-hospital health state."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.tools: List[Any] = []
        self.connected_hospitals: List[str] = []
        self.failed_hospitals: Dict[str, str] = {}  # code → error message
        self.built_at: float = 0.0
        self.breakers: Dict[str, CircuitBreaker] = {}


class UserMCPPool:
    """Per-user MCP client pool with health tracking and tool namespacing."""

    def __init__(
        self,
        hospital_registry: HospitalRegistry,
        user_hospital_store: UserHospitalStore,
    ):
        self._registry = hospital_registry
        self._store = user_hospital_store
        self._pools: Dict[str, _UserPool] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tools_for_user(self, user_id: str) -> List[Any]:
        """Return the cached LangChain tools for this user, building if needed."""
        if not config.MCP_ENABLED:
            return []
        pool = self._get_or_build_pool(user_id)
        return list(pool.tools)

    def get_connected_hospitals(self, user_id: str) -> List[str]:
        if not config.MCP_ENABLED:
            return []
        pool = self._get_or_build_pool(user_id)
        return list(pool.connected_hospitals)

    def get_failed_hospitals(self, user_id: str) -> Dict[str, str]:
        if not config.MCP_ENABLED:
            return {}
        pool = self._get_or_build_pool(user_id)
        return dict(pool.failed_hospitals)

    def invalidate(self, user_id: str):
        """Drop this user's pool (called when credentials change)."""
        with self._lock:
            self._pools.pop(user_id, None)

    def get_status_summary(self, user_id: str) -> Dict[str, Any]:
        """Status info for the system status endpoint."""
        if not config.MCP_ENABLED:
            return {"enabled": False, "connected": [], "failed": {}}
        pool = self._get_or_build_pool(user_id)
        return {
            "enabled": True,
            "connected": list(pool.connected_hospitals),
            "failed": dict(pool.failed_hospitals),
            "tool_count": len(pool.tools),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_build_pool(self, user_id: str) -> _UserPool:
        with self._lock:
            pool = self._pools.get(user_id)
            if pool is None:
                pool = _UserPool(user_id)
                self._pools[user_id] = pool
            else:
                # Refresh pool if it's older than the health check interval
                age = time.time() - pool.built_at
                if age < config.MCP_HEALTH_CHECK_INTERVAL_SECONDS:
                    return pool
        # Build outside the lock — could be slow
        self._build_pool(pool)
        return pool

    def _build_pool(self, pool: _UserPool):
        """Connect to all of this user's hospitals and load their tools."""
        creds = self._store.get_all_decrypted(pool.user_id)
        if not creds:
            pool.tools = []
            pool.connected_hospitals = []
            pool.failed_hospitals = {}
            pool.built_at = time.time()
            return

        # Build per-hospital connection configs
        connections = {}
        for code, token in creds.items():
            # Skip hospitals whose breaker is open
            breaker = pool.breakers.setdefault(code, CircuitBreaker(failure_threshold=3, recovery_timeout=120))
            if not breaker.allow_request():
                logger.info("Skipping hospital %s for user %s: circuit breaker open", code, pool.user_id)
                pool.failed_hospitals[code] = "circuit_breaker_open"
                continue

            hospital = self._registry.get_by_code(code)
            if not hospital or not hospital.get("is_active"):
                logger.warning("Hospital %s not in registry or inactive; skipping", code)
                pool.failed_hospitals[code] = "hospital_not_active"
                continue

            connections[code] = {
                "transport": "streamable_http",
                "url": hospital["mcp_url"],
                "headers": {"Authorization": f"Bearer {token}"},
            }

        if not connections:
            pool.tools = []
            pool.connected_hospitals = []
            pool.built_at = time.time()
            return

        # Build the MCP client + load tools (async bridge)
        tools, connected, failed = self._sync_load_tools(connections, pool)

        # Update breakers based on results
        for code in connected:
            pool.breakers[code].record_success()
            self._store.update_health(pool.user_id, code, "healthy")
        for code, err in failed.items():
            pool.breakers[code].record_failure()
            pool.failed_hospitals[code] = err
            self._store.update_health(pool.user_id, code, "failed")

        pool.tools = tools
        pool.connected_hospitals = connected
        pool.failed_hospitals.update(failed)
        pool.built_at = time.time()

    def _sync_load_tools(
        self,
        connections: Dict[str, Dict[str, Any]],
        pool: _UserPool,
    ) -> Tuple[List[Any], List[str], Dict[str, str]]:
        """Bridge async MCP loading to sync code."""
        try:
            return asyncio.run(self._async_load_tools(connections, pool))
        except RuntimeError as e:
            # We're already inside an event loop (e.g., when called from FastAPI)
            if "asyncio.run() cannot be called" in str(e):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self._async_load_tools(connections, pool))
                finally:
                    loop.close()
            raise

    async def _async_load_tools(
        self,
        connections: Dict[str, Dict[str, Any]],
        pool: _UserPool,
    ) -> Tuple[List[Any], List[str], Dict[str, str]]:
        """Connect to each MCP server and load tools, with namespacing."""
        all_tools: List[Any] = []
        connected: List[str] = []
        failed: Dict[str, str] = {}

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            logger.error("langchain-mcp-adapters not installed; pip install langchain-mcp-adapters")
            for code in connections:
                failed[code] = "package_missing"
            return [], [], failed

        # Build client for ALL servers (the adapter will lazy-connect per call)
        client = MultiServerMCPClient(connections)

        # Load tools per-server so a single failure doesn't kill all
        for code in connections.keys():
            try:
                async with asyncio.timeout(config.MCP_DEFAULT_TIMEOUT_SECONDS):
                    server_tools = await self._load_one_server_tools(client, code)
                # Namespace the tool names
                for tool in server_tools:
                    original_name = tool.name
                    tool.name = f"{code}{_NAMESPACE_SEP}{original_name}"
                    # Tag description with hospital name for the LLM
                    hospital = self._registry.get_by_code(code)
                    if hospital:
                        tool.description = f"[{hospital['name']}] {tool.description}"
                all_tools.extend(server_tools)
                connected.append(code)
                logger.info("Loaded %d tools from hospital %s for user %s",
                            len(server_tools), code, pool.user_id)
            except Exception as e:
                err_msg = type(e).__name__ + ": " + str(e)[:200]
                logger.warning("Failed to load tools from hospital %s: %s", code, err_msg)
                failed[code] = err_msg

        return all_tools, connected, failed

    async def _load_one_server_tools(self, client, server_code: str) -> List[Any]:
        """Load tools from a single named server."""
        # MultiServerMCPClient exposes a per-server session context manager
        async with client.session(server_code) as session:
            from langchain_mcp_adapters.tools import load_mcp_tools
            return await load_mcp_tools(session)

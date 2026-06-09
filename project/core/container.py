"""ServiceContainer — holds all subsystem singletons.

Gradually replaces direct attribute access on RAGSystem.  Each subsystem
is registered by name and can be retrieved via container.get(name) or
container.name (attribute access).  New code should prefer injecting
individual services instead of the whole container.
"""

import threading
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Holds initialized subsystem instances.

    Usage::

        container = ServiceContainer()
        container.register("vector_db", VectorDbManager())
        container.register("session_memory", RedisSessionMemory())

        # Access patterns:
        container.get("vector_db")       # explicit
        container.vector_db              # attribute delegation
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, name: str, service: Any) -> None:
        """Register a service by name."""
        with self._lock:
            self._services[name] = service

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a service by name, or None if not registered."""
        return self._services.get(name)

    def has(self, name: str) -> bool:
        """Check whether a service is registered."""
        return name in self._services

    @property
    def service_names(self):
        """Return registered service names."""
        return list(self._services.keys())

    def __getattr__(self, name: str) -> Any:
        # Allow container.vector_db instead of container.get("vector_db")
        svc = self._services.get(name)
        if svc is not None:
            return svc
        raise AttributeError(f"Service {name!r} not registered in ServiceContainer")

"""Abstract base class for pluggable skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class BaseSkill(ABC):
    """Abstract base class for pluggable skills.

    Each skill declares:
    - ``name``: unique identifier (e.g. "greeting")
    - ``priority``: match priority (lower = checked first)
    - ``intent_label``: intent string this skill claims for routing
    - ``match()``: whether this skill should handle the query
    - ``register_nodes()``: add graph nodes
    - ``register_edges()``: add graph edges
    - ``get_route_targets()``: intent → node_name mapping
    - ``get_state_schema()``: default state for ``skill_data[name]``
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill identifier, e.g. 'greeting', 'medical_rag'."""
        ...

    @property
    def priority(self) -> int:
        """Match priority. Lower = higher priority (checked first). Default 100."""
        return 100

    @property
    def intent_label(self) -> str:
        """Intent string this skill claims for routing. Defaults to ``self.name``."""
        return self.name

    @abstractmethod
    def match(self, query: str, *, context: Dict[str, Any]) -> bool:
        """Return True if this skill should handle the query.

        Args:
            query: The user's current query text.
            context: Dict with 'conversation_summary', 'recent_context',
                     'topic_focus', etc.
        """
        ...

    def get_state_schema(self) -> Dict[str, Any]:
        """Return the skill's default state dict for ``skill_data[self.name]``."""
        return {}

    def register_nodes(
        self, graph_builder, *, llm_router=None, tools_list=None, services=None
    ) -> Dict[str, Callable]:
        """Register this skill's nodes into the graph builder.

        Returns:
            Dict mapping node_name → node_function.
        """
        return {}

    def register_edges(self, graph_builder) -> None:
        """Register this skill's edges into the graph builder."""
        pass

    def get_route_targets(self) -> Dict[str, str]:
        """Return intent → node_name mapping for routing.

        E.g. ``{"greeting": "greeting_handler"}`` means when intent is
        "greeting", route to the ``greeting_handler`` node.
        """
        return {self.intent_label: f"{self.name}_handler"}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} priority={self.priority}>"

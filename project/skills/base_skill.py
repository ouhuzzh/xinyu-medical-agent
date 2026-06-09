"""Abstract base class for pluggable skills.

Each skill declares:
  - ``name``: unique identifier
  - ``priority``: match priority (lower = checked first)
  - ``intent_label``: intent string for routing
  - ``keywords``: L1 high-confidence keywords (exact match, <8 chars each)
  - ``utterances``: L2 semantic samples for embedding matching
  - ``llm_hint``: L3 description injected into LLM intent-classification prompts
  - ``match()``: optional override — defaults to keyword / utterance check
  - ``register_nodes()`` / ``register_edges()``: graph wiring
  - ``get_route_targets()``: intent → node_name mapping
  - ``get_state_schema()``: default state for ``skill_data[name]``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple


class BaseSkill(ABC):
    """Abstract base class for pluggable skills."""

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill identifier, e.g. 'greeting', 'medical_rag'."""
        ...

    # ------------------------------------------------------------------
    # Optional properties (override to customise)
    # ------------------------------------------------------------------

    @property
    def priority(self) -> int:
        """Match priority. Lower = higher priority (checked first). Default 100."""
        return 100

    @property
    def intent_label(self) -> str:
        """Intent string for routing. Defaults to ``self.name``."""
        return self.name

    # ------------------------------------------------------------------
    # L1 keywords — extremely high-confidence exact matches
    # ------------------------------------------------------------------

    @property
    def keywords(self) -> Tuple[str, ...]:
        """L1 keywords for exact-match intent classification.

        Each keyword is matched EXACTLY (not substring) against the user query.
        Keep these short (< 8 chars) and unambiguous — e.g. "你好" not
        "帮我".  Default: empty tuple (L1 doesn't match).
        """
        return ()

    @property
    def allow_l1_substring(self) -> bool:
        """If True, keywords match as substrings. If False, exact match only.
        Default False — most skills should use exact matching to avoid
        catching residual intent in compound queries.
        """
        return False

    # ------------------------------------------------------------------
    # L2 utterances — semantic embedding samples
    # ------------------------------------------------------------------

    @property
    def utterances(self) -> List[str]:
        """L2 semantic embedding samples.

        These are encoded into a centroid vector at startup.  User queries
        with high cosine similarity to this centroid are classified as this
        intent.  Default: empty list (L2 doesn't match).
        """
        return []

    # ------------------------------------------------------------------
    # L3 llm_hint — description injected into LLM intent classification prompt
    # ------------------------------------------------------------------

    @property
    def llm_hint(self) -> str:
        """L3 hint describing this skill's intent for the LLM classifier.

        This string is injected into the intent-router and rewrite-query
        prompts so the LLM knows about Skill-registered intents beyond the
        hardcoded core set.  Return empty string to skip L3 injection
        (e.g. for skills that are fully handled by L1/L2).  Default: empty.
        """
        return ""

    # ------------------------------------------------------------------
    # Legacy match() — defaults to L1 keyword check
    # ------------------------------------------------------------------

    def match(self, query: str, *, context: Dict[str, Any]) -> bool:
        """Default match: check L1 keywords.

        Skills can override for custom matching logic, or rely on L1+L2+L3
        pipeline where L1 checks keywords and L2 checks utterances.
        """
        normalized = (query or "").strip()
        if not normalized:
            return False
        if self.allow_l1_substring:
            return any(kw in normalized for kw in self.keywords)
        return normalized in self.keywords

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def get_state_schema(self) -> Dict[str, Any]:
        """Return the skill's default state dict for ``skill_data[self.name]``."""
        return {}

    def register_nodes(
        self, graph_builder, *, llm_router=None, tools_list=None, services=None
    ) -> Dict[str, Callable]:
        """Register this skill's nodes into the graph builder."""
        return {}

    def register_edges(self, graph_builder) -> None:
        """Register this skill's edges into the graph builder."""
        pass

    def get_route_targets(self) -> Dict[str, str]:
        """Return intent → node_name mapping for routing."""
        return {self.intent_label: f"{self.name}_handler"}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} priority={self.priority}>"

    def is_active(self) -> bool:
        """Whether this skill should participate in classification.

        Override to gate on runtime conditions (e.g. MCP availability).
        Default: always active.
        """
        return True

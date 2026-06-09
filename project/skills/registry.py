"""Central registry for skill plugins."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_skill import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for skill plugins.

    Skills are sorted by priority (lower = checked first).  The registry
    provides a unified ``classify_intent()`` method that replaces hardcoded
    if-elif chains in the routing layer.
    """

    def __init__(self) -> None:
        self._skills: List[BaseSkill] = []

    def register(self, skill: BaseSkill) -> None:
        """Register a skill.  Skills are kept sorted by priority."""
        self._skills.append(skill)
        self._skills.sort(key=lambda s: s.priority)
        logger.info("Registered skill %s (priority=%d)", skill.name, skill.priority)

    @property
    def skills(self) -> List[BaseSkill]:
        """Registered skills in priority order."""
        return list(self._skills)

    def classify_intent(
        self, query: str, *, context: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """Try each skill's ``match()`` in priority order.

        Returns:
            ``(intent_label, skill_name)`` for the first matching skill,
            or ``None`` if no skill matches.
        """
        for skill in self._skills:
            try:
                if skill.match(query, context=context):
                    logger.debug(
                        "Skill %r matched query %r (intent=%s)",
                        skill.name, query[:50], skill.intent_label,
                    )
                    return (skill.intent_label, skill.name)
            except Exception:
                logger.exception("Skill %r.match() raised an exception", skill.name)
        return None

    # ------------------------------------------------------------------
    # L1: keyword-based classification
    # ------------------------------------------------------------------

    def classify_by_keywords(self, query: str) -> Optional[Tuple[str, str]]:
        """Run L1 keyword matching across all skills in priority order.

        Skips inactive skills (e.g. MCPSkill when MCP is disabled).

        Returns ``(intent_label, "l1_keyword")`` or None.
        """
        normalized = (query or "").strip()
        if not normalized:
            return None
        for skill in self._skills:
            if not skill.is_active():
                continue
            kw = skill.keywords
            if not kw:
                continue
            try:
                if skill.allow_l1_substring:
                    hit = any(k in normalized for k in kw)
                else:
                    hit = normalized in kw
                if hit:
                    return (skill.intent_label, "l1_keyword")
            except Exception:
                logger.exception("Skill %r keywords raised", skill.name)
        return None

    # ------------------------------------------------------------------
    # L2: utterance collection
    # ------------------------------------------------------------------

    def collect_utterances(self) -> Dict[str, List[str]]:
        """Collect utterances from all ACTIVE skills for L2 embedding centroids.

        Returns ``{intent_label: [utterances]}``.
        """
        result: Dict[str, List[str]] = {}
        for skill in self._skills:
            if not skill.is_active():
                continue
            utterances = skill.utterances
            if utterances:
                existing = result.setdefault(skill.intent_label, [])
                existing.extend(utterances)
        return result

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_route_mapping(self) -> Dict[str, str]:
        """Merge all skill route targets into a single intent → node_name dict."""
        mapping: Dict[str, str] = {}
        for skill in self._skills:
            mapping.update(skill.get_route_targets())
        return mapping

    def register_all_nodes(
        self, graph_builder, *, llm_router=None, tools_list=None, services=None
    ) -> None:
        """Call ``register_nodes()`` on all registered skills."""
        for skill in self._skills:
            nodes = skill.register_nodes(
                graph_builder,
                llm_router=llm_router,
                tools_list=tools_list,
                services=services,
            )
            for node_name, node_func in nodes.items():
                graph_builder.add_node(node_name, node_func)
                logger.debug("Skill %r registered node %r", skill.name, node_name)

    def register_all_edges(self, graph_builder) -> None:
        """Call ``register_edges()`` on all registered skills."""
        for skill in self._skills:
            skill.register_edges(graph_builder)

    def get_all_state_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{skill_name: state_schema}`` for all skills."""
        return {skill.name: skill.get_state_schema() for skill in self._skills}

    # ------------------------------------------------------------------
    # L3: LLM intent classification support
    # ------------------------------------------------------------------

    def collect_llm_hints(self) -> List[Tuple[str, str]]:
        """Collect (intent_label, llm_hint) from all ACTIVE skills.

        Only skills with a non-empty ``llm_hint`` are included.
        Used to dynamically inject skill-specific intent descriptions
        into the L3 LLM classification prompts.
        """
        hints: List[Tuple[str, str]] = []
        for skill in self._skills:
            if not skill.is_active():
                continue
            hint = skill.llm_hint
            if hint:
                hints.append((skill.intent_label, hint))
        return hints

    def build_intent_labels(self) -> List[str]:
        """Build the full list of intent labels for L3 LLM classification.

        Returns the union of core intents and all active skill intent_labels.
        """
        # Core intents that always exist
        labels = {"medical_rag", "triage", "appointment",
                  "cancel_appointment", "greeting", "clarification"}
        for skill in self._skills:
            if skill.is_active():
                labels.add(skill.intent_label)
        return sorted(labels)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry, creating it lazily."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def reset_skill_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None

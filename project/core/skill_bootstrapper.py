"""Bootstrap skill plugins into the shared skill registry."""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import config


logger = logging.getLogger(__name__)


class SkillBootstrapper:
    def __init__(self, *, registry_factory: Callable | None = None, skill_factories: Iterable[Callable] | None = None):
        self._registry_factory = registry_factory
        self._skill_factories = list(skill_factories) if skill_factories is not None else None

    def bootstrap(self) -> int:
        if not getattr(config, "SKILLS_ENABLED", False):
            return 0

        registry = self._get_registry()
        existing_names = {skill.name for skill in registry.skills}
        registered_count = 0

        for factory in self._get_skill_factories():
            skill = factory()
            if skill.name in existing_names:
                logger.debug("Skill %s already registered; skipping", skill.name)
                continue
            registry.register(skill)
            existing_names.add(skill.name)
            registered_count += 1

        logger.info(
            "Skill plugin framework enabled: %d new skills registered, %d total skills",
            registered_count,
            len(registry.skills),
        )
        return len(registry.skills)

    def _get_registry(self):
        if self._registry_factory is not None:
            return self._registry_factory()

        from skills.registry import get_skill_registry

        return get_skill_registry()

    def _get_skill_factories(self) -> list[Callable]:
        if self._skill_factories is not None:
            return list(self._skill_factories)

        from mcp_integration.mcp_skill import MCPSkill
        from skills.booking_skill import AppointmentSkill as BookingIntentSkill
        from skills.cancel_skill import CancelSkill
        from skills.greeting_skill import GreetingSkill
        from skills.medical_rag_skill import MedicalRagSkill
        from skills.triage_skill import TriageSkill

        return [
            GreetingSkill,
            TriageSkill,
            BookingIntentSkill,
            CancelSkill,
            MedicalRagSkill,
            MCPSkill,
        ]

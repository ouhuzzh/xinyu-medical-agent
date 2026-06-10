import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from core.skill_bootstrapper import SkillBootstrapper  # noqa: E402


class FakeSkill:
    priority = 100

    def __init__(self, name):
        self.name = name


class FakeRegistry:
    def __init__(self):
        self._skills = []

    @property
    def skills(self):
        return list(self._skills)

    def register(self, skill):
        self._skills.append(skill)


class SkillBootstrapperTests(unittest.TestCase):
    def test_bootstrap_returns_zero_when_skills_disabled(self):
        registry = FakeRegistry()
        bootstrapper = SkillBootstrapper(
            registry_factory=lambda: registry,
            skill_factories=[lambda: FakeSkill("greeting")],
        )

        with patch("core.skill_bootstrapper.config.SKILLS_ENABLED", False):
            self.assertEqual(bootstrapper.bootstrap(), 0)

        self.assertEqual(registry.skills, [])

    def test_bootstrap_registers_configured_skills(self):
        registry = FakeRegistry()
        bootstrapper = SkillBootstrapper(
            registry_factory=lambda: registry,
            skill_factories=[
                lambda: FakeSkill("greeting"),
                lambda: FakeSkill("triage"),
            ],
        )

        with patch("core.skill_bootstrapper.config.SKILLS_ENABLED", True):
            total = bootstrapper.bootstrap()

        self.assertEqual(total, 2)
        self.assertEqual([skill.name for skill in registry.skills], ["greeting", "triage"])

    def test_bootstrap_is_idempotent_by_skill_name(self):
        registry = FakeRegistry()
        registry.register(FakeSkill("greeting"))
        bootstrapper = SkillBootstrapper(
            registry_factory=lambda: registry,
            skill_factories=[
                lambda: FakeSkill("greeting"),
                lambda: FakeSkill("triage"),
            ],
        )

        with patch("core.skill_bootstrapper.config.SKILLS_ENABLED", True):
            total = bootstrapper.bootstrap()

        self.assertEqual(total, 2)
        self.assertEqual([skill.name for skill in registry.skills], ["greeting", "triage"])


if __name__ == "__main__":
    unittest.main()

"""Tests for Skill plugin framework."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from skills.base_skill import BaseSkill
from skills.registry import SkillRegistry, get_skill_registry, reset_skill_registry


# -- Test skill implementations ------------------------------------------

class _FakeHighPrioritySkill(BaseSkill):
    @property
    def name(self) -> str: return "high_priority"
    @property
    def priority(self) -> int: return 10
    @property
    def intent_label(self) -> str: return "high_priority"
    def match(self, query, *, context=None): return query.startswith("紧急")
    def get_route_targets(self): return {"high_priority": "high_priority_handler"}
    def register_nodes(self, graph_builder, **kwargs):
        return {"high_priority_handler": lambda state: {"messages": []}}


class _FakeLowPrioritySkill(BaseSkill):
    @property
    def name(self) -> str: return "low_priority"
    @property
    def priority(self) -> int: return 90
    @property
    def intent_label(self) -> str: return "low_priority"
    def match(self, query, *, context=None): return True  # always matches
    def get_route_targets(self): return {"low_priority": "low_priority_handler"}
    def register_nodes(self, graph_builder, **kwargs):
        return {"low_priority_handler": lambda state: {"messages": []}}


class _FakeNoMatchSkill(BaseSkill):
    @property
    def name(self) -> str: return "never_matches"
    @property
    def priority(self) -> int: return 50
    @property
    def intent_label(self) -> str: return "never_matches"
    def match(self, query, *, context=None): return False


# -- Test cases -----------------------------------------------------------

class TestBaseSkill(unittest.TestCase):

    def test_default_priority(self):
        class MinimalSkill(BaseSkill):
            @property
            def name(self): return "test"
            def match(self, query, *, context=None): return False
        s = MinimalSkill()
        self.assertEqual(s.priority, 100)
        self.assertEqual(s.intent_label, "test")

    def test_default_route_targets(self):
        class MinimalSkill(BaseSkill):
            @property
            def name(self): return "my_skill"
            def match(self, query, *, context=None): return False
        s = MinimalSkill()
        self.assertEqual(s.get_route_targets(), {"my_skill": "my_skill_handler"})

    def test_repr(self):
        s = _FakeHighPrioritySkill()
        self.assertIn("high_priority", repr(s))


class TestSkillRegistry(unittest.TestCase):

    def setUp(self):
        reset_skill_registry()

    def tearDown(self):
        reset_skill_registry()

    def test_register_and_skills_sorted_by_priority(self):
        reg = SkillRegistry()
        reg.register(_FakeLowPrioritySkill())   # priority 90
        reg.register(_FakeHighPrioritySkill())   # priority 10
        self.assertEqual([s.name for s in reg.skills], ["high_priority", "low_priority"])

    def test_classify_intent_returns_first_match(self):
        reg = SkillRegistry()
        reg.register(_FakeLowPrioritySkill())   # priority 90, always matches
        reg.register(_FakeHighPrioritySkill())   # priority 10, matches "紧急..."
        # "紧急求助" should match high_priority first (lower priority number)
        result = reg.classify_intent("紧急求助", context={})
        self.assertEqual(result, ("high_priority", "high_priority"))

    def test_classify_intent_falls_through(self):
        reg = SkillRegistry()
        reg.register(_FakeHighPrioritySkill())   # priority 10, only "紧急..."
        # "普通问题" should NOT match high_priority
        result = reg.classify_intent("普通问题", context={})
        self.assertIsNone(result)

    def test_classify_intent_no_skills(self):
        reg = SkillRegistry()
        result = reg.classify_intent("任何查询", context={})
        self.assertIsNone(result)

    def test_get_route_mapping(self):
        reg = SkillRegistry()
        reg.register(_FakeHighPrioritySkill())
        reg.register(_FakeLowPrioritySkill())
        mapping = reg.get_route_mapping()
        self.assertEqual(mapping["high_priority"], "high_priority_handler")
        self.assertEqual(mapping["low_priority"], "low_priority_handler")

    def test_get_all_state_schemas(self):
        reg = SkillRegistry()
        reg.register(_FakeHighPrioritySkill())
        schemas = reg.get_all_state_schemas()
        self.assertIn("high_priority", schemas)

    def test_register_nodes(self):
        reg = SkillRegistry()
        reg.register(_FakeHighPrioritySkill())
        mock_builder = MagicMock()
        reg.register_all_nodes(mock_builder)
        mock_builder.add_node.assert_called_once_with(
            "high_priority_handler", unittest.mock.ANY
        )


class TestGreetingSkill(unittest.TestCase):

    def test_greeting_skill_matches_greeting(self):
        from skills.greeting_skill import GreetingSkill
        skill = GreetingSkill()
        self.assertTrue(skill.match("你好", context={}))
        self.assertTrue(skill.match("hello", context={}))

    def test_greeting_skill_no_match_medical(self):
        from skills.greeting_skill import GreetingSkill
        skill = GreetingSkill()
        self.assertFalse(skill.match("高血压怎么办", context={}))

    def test_greeting_skill_priority(self):
        from skills.greeting_skill import GreetingSkill
        skill = GreetingSkill()
        self.assertEqual(skill.priority, 10)

    def test_greeting_skill_route_targets(self):
        from skills.greeting_skill import GreetingSkill
        skill = GreetingSkill()
        targets = skill.get_route_targets()
        self.assertEqual(targets, {"greeting": "greeting_handler"})


class TestMedicalRagSkill(unittest.TestCase):

    def test_medical_rag_skill_matches_medical(self):
        from skills.medical_rag_skill import MedicalRagSkill
        skill = MedicalRagSkill()
        # This depends on the _looks_like_medical_knowledge_question implementation
        # A simple medical question should match
        self.assertTrue(skill.match("高血压的症状是什么", context={}))

    def test_medical_rag_skill_no_match_greeting(self):
        from skills.medical_rag_skill import MedicalRagSkill
        skill = MedicalRagSkill()
        self.assertFalse(skill.match("你好", context={}))

    def test_medical_rag_skill_route_targets(self):
        from skills.medical_rag_skill import MedicalRagSkill
        skill = MedicalRagSkill()
        targets = skill.get_route_targets()
        self.assertEqual(targets, {"medical_rag": "rewrite_query"})

    def test_medical_rag_skill_no_custom_nodes(self):
        from skills.medical_rag_skill import MedicalRagSkill
        skill = MedicalRagSkill()
        nodes = skill.register_nodes(MagicMock())
        self.assertEqual(nodes, {})


class TestSkillRegistrySingleton(unittest.TestCase):

    def setUp(self):
        reset_skill_registry()

    def tearDown(self):
        reset_skill_registry()

    def test_get_skill_registry_creates_singleton(self):
        reg1 = get_skill_registry()
        reg2 = get_skill_registry()
        self.assertIs(reg1, reg2)

    def test_reset_clears_registry(self):
        reg1 = get_skill_registry()
        reg1.register(_FakeHighPrioritySkill())
        self.assertEqual(len(reg1.skills), 1)
        reset_skill_registry()
        reg2 = get_skill_registry()
        self.assertEqual(len(reg2.skills), 0)


if __name__ == "__main__":
    unittest.main()

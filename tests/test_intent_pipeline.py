"""Unit tests for the Hybrid 3-tier intent routing pipeline.

Covers L1 strict rules, L2 embedding semantic matching, and regression
for the 5 original boundary bugs.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from rag_agent.node_helpers import (  # noqa: E402
    _looks_like_greeting,
    _looks_like_explicit_appointment_intent,
    _looks_like_explicit_cancel_intent,
    _looks_like_department_question,
    _starts_with_polite_decline,
)

# L1 wrapper (same logic as in routing_nodes)
def _l1_strict_rules(user_query: str) -> str | None:
    if _looks_like_greeting(user_query):
        return "greeting"
    if _looks_like_explicit_cancel_intent(user_query):
        return "cancel_appointment"
    if _looks_like_explicit_appointment_intent(user_query):
        return "appointment"
    if _looks_like_department_question(user_query):
        return "triage"
    return None


class TestL1StrictRules(unittest.TestCase):
    """L1 should only intercept extremely high-confidence, unambiguous cases."""

    # --- Pure greetings (should match) ---
    def test_pure_greetings_hello(self):
        self.assertEqual(_l1_strict_rules("你好"), "greeting")

    def test_pure_greetings_thanks(self):
        self.assertEqual(_l1_strict_rules("谢谢"), "greeting")

    def test_pure_greetings_hi(self):
        self.assertEqual(_l1_strict_rules("hi"), "greeting")

    def test_pure_greetings_bye(self):
        self.assertEqual(_l1_strict_rules("拜拜"), "greeting")

    # --- Compound queries with greeting prefix (should NOT match greeting) ---
    def test_greeting_plus_appointment_not_l1(self):
        """Bug 1: '你好我要挂号' has residual intent → NOT a pure greeting."""
        result = _l1_strict_rules("你好我要挂号")
        self.assertIsNone(result, f"Expected None (not pure greeting), got {result}")

    def test_greeting_plus_long_not_l1(self):
        """Query > 5 chars shouldn't match greeting."""
        result = _l1_strict_rules("你好请问")
        self.assertIsNone(result, f"Expected None (too long for L1 greeting), got {result}")

    # --- Explicit appointment (must have action + entity) ---
    def test_explicit_appointment_with_dept(self):
        self.assertEqual(_l1_strict_rules("帮我预约心内科"), "appointment")

    def test_explicit_appointment_with_date(self):
        self.assertEqual(_l1_strict_rules("帮我预约明天"), "appointment")

    def test_explicit_appointment_with_dept_and_time(self):
        self.assertEqual(_l1_strict_rules("帮我预约心内科明天"), "appointment")

    def test_vague_appointment_not_l1(self):
        """'我要挂号' without entity shouldn't match L1."""
        result = _l1_strict_rules("我要挂号")
        self.assertIsNone(result, f"Expected None (no entity), got {result}")

    def test_pre_appointment_not_l1_appointment(self):
        """Bug 3: '预约前注意什么' is a medical question, NOT appointment."""
        result = _l1_strict_rules("预约前注意什么")
        self.assertIsNone(result, f"Expected None (medical question), got {result}")

    def test_pre_register_not_l1_appointment(self):
        """'挂号前要准备什么' is a medical question."""
        result = _l1_strict_rules("挂号前要准备什么")
        self.assertIsNone(result, f"Expected None (medical question), got {result}")

    # --- Explicit cancel (must have action + reference) ---
    def test_explicit_cancel_with_ref(self):
        self.assertEqual(_l1_strict_rules("取消预约 APT123"), "cancel_appointment")

    def test_explicit_cancel_last_appointment(self):
        self.assertEqual(_l1_strict_rules("取消刚才的挂号"), "cancel_appointment")

    def test_medical_cancel_not_l1(self):
        """Bug 4: '取消对药物的依赖' is a medical question, NOT cancel."""
        result = _l1_strict_rules("取消对药物的依赖")
        self.assertIsNone(result, f"Expected None (medical question), got {result}")

    def test_cancel_without_reference_not_l1(self):
        """'取消' alone without appointment reference shouldn't match L1."""
        result = _l1_strict_rules("取消")
        self.assertIsNone(result, f"Expected None (no reference), got {result}")

    # --- Department question ---
    def test_department_question(self):
        self.assertEqual(_l1_strict_rules("挂什么科"), "triage")

    def test_department_question_with_symptom(self):
        """Bug 5: '胸痛挂什么科' should be triage."""
        self.assertEqual(_l1_strict_rules("胸痛挂什么科"), "triage")

    def test_department_question_variant(self):
        self.assertEqual(_l1_strict_rules("看哪个科"), "triage")


class TestPoliteDecline(unittest.TestCase):
    """Polite decline detection for Bug 2."""

    def test_polite_decline_thanks_no_need(self):
        self.assertTrue(_starts_with_polite_decline("谢谢我不用了"))

    def test_polite_decline_thanks_forget_it(self):
        self.assertTrue(_starts_with_polite_decline("谢谢算了"))

    def test_polite_decline_hello_no_need(self):
        self.assertTrue(_starts_with_polite_decline("你好不用了"))

    def test_not_polite_decline_pure_thanks(self):
        self.assertFalse(_starts_with_polite_decline("谢谢"))

    def test_not_polite_decline_cancel(self):
        self.assertFalse(_starts_with_polite_decline("取消预约"))


class TestL2Embedding(unittest.TestCase):
    """L2 embedding semantic matching — requires embedding model."""

    @classmethod
    def setUpClass(cls):
        """Lazy-init the embedder once for all tests."""
        # Ensure registry is populated so embedder loads utterances
        from skills.registry import get_skill_registry
        registry = get_skill_registry()
        if not registry.skills:
            from skills.greeting_skill import GreetingSkill
            from skills.medical_rag_skill import MedicalRagSkill
            from skills.booking_skill import AppointmentSkill as BookingIntentSkill
            from skills.cancel_skill import CancelSkill
            from skills.triage_skill import TriageSkill
            registry.register(GreetingSkill())
            registry.register(TriageSkill())
            registry.register(BookingIntentSkill())
            registry.register(CancelSkill())
            registry.register(MedicalRagSkill())
        try:
            from rag_agent.intent_embedder import get_intent_embedder  # noqa: E402
            cls.embedder = get_intent_embedder()
            # Force init
            cls.embedder._lazy_init()
        except Exception as e:
            raise unittest.SkipTest(f"Embedding model not available: {e}")
        # _lazy_init swallows model-load failures (e.g. missing embedding API
        # key) and leaves centroids empty - detect that and skip rather than
        # fail every assertion with empty/None classifications.
        if not getattr(cls.embedder, "_centroids", None):
            raise unittest.SkipTest(
                "Embedding model not available: no intent centroids built "
                "(likely missing embedding API key in the test env)"
            )

    def _classify_l2(self, query: str) -> tuple:
        """Run L1+L2 pipeline on a query."""
        l1 = _l1_strict_rules(query)
        if l1:
            return (l1, 1.0, "l1_rule")
        result = self.embedder.classify(query)
        if result:
            return (*result, "l2_embedding")
        return ("", 0.0, "need_llm")

    # --- Semantic matching tests ---
    def test_l2_semantic_greeting(self):
        intent, conf, source = self._classify_l2("早上好")
        self.assertIn(intent, ["greeting"], f"Expected greeting, got {intent} (conf={conf:.2f})")

    def test_l2_semantic_appointment(self):
        intent, conf, source = self._classify_l2("我想挂号看心内科")
        self.assertEqual(intent, "appointment", f"Expected appointment, got {intent} (conf={conf:.2f})")

    def test_l2_semantic_cancel(self):
        intent, conf, source = self._classify_l2("我要退号")
        self.assertEqual(intent, "cancel_appointment", f"Expected cancel_appointment, got {intent} (conf={conf:.2f})")

    # --- Boundary cases (should NOT be misclassified) ---
    def test_l2_boundary_pre_appointment_is_medical(self):
        """Bug 3: embedding should match medical_rag, not appointment."""
        intent, conf, source = self._classify_l2("预约前注意什么")
        self.assertEqual(intent, "medical_rag",
                         f"Expected medical_rag, got {intent} (conf={conf:.2f}, source={source})")

    def test_l2_boundary_cancel_drug_is_medical(self):
        """Bug 4: embedding should match medical_rag, not cancel_appointment."""
        intent, conf, source = self._classify_l2("取消对药物的依赖")
        self.assertEqual(intent, "medical_rag",
                         f"Expected medical_rag, got {intent} (conf={conf:.2f}, source={source})")

    def test_l2_boundary_greeting_plus_booking(self):
        """Bug 1: L2 should NOT classify as greeting."""
        intent, conf, source = self._classify_l2("你好我要挂号")
        self.assertNotEqual(intent, "greeting",
                            f"Should not be greeting (has real intent), got {intent} (conf={conf:.2f})")

    def test_l2_boundary_polite_decline_is_greeting(self):
        """Bug 2: polite decline should go to greeting/medical_rag, not cancel."""
        intent, conf, source = self._classify_l2("谢谢我不用了")
        self.assertNotEqual(intent, "cancel_appointment",
                            f"Should not be cancel_appointment, got {intent} (conf={conf:.2f})")

    # --- Non-regression: correct cases should still work ---
    def test_l2_normal_appointment(self):
        intent, conf, source = self._classify_l2("我要预约挂号")
        self.assertEqual(intent, "appointment",
                         f"Expected appointment, got {intent} (conf={conf:.2f})")

    def test_l2_normal_medical(self):
        intent, conf, source = self._classify_l2("高血压怎么控制")
        self.assertEqual(intent, "medical_rag",
                         f"Expected medical_rag, got {intent} (conf={conf:.2f})")

    def test_l2_triage(self):
        intent, conf, source = self._classify_l2("应该挂哪个科室")
        self.assertEqual(intent, "triage",
                         f"Expected triage, got {intent} (conf={conf:.2f})")

    # --- Threshold behavior ---
    def test_l2_low_confidence_returns_none(self):
        """Ambiguous medical/booking boundary query should have lower confidence."""
        # Test with a genuinely confusing query that mixes intents
        result = self.embedder.classify("帮我看看这个预约还能不能取消")
        if result is not None:
            intent, conf = result
            # Either appointment or cancel_appointment is plausible,
            # but confidence should not be sky-high for this ambiguous query
            self.assertIn(intent, ["appointment", "cancel_appointment"],
                          f"Ambiguous query should match appointment or cancel, got {intent}")
            self.assertLess(conf, 0.80,
                            f"Ambiguous query should have moderate confidence, got {conf:.2f}")


if __name__ == "__main__":
    unittest.main()

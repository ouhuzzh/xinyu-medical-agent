"""Skill plugin framework for extensible intent routing."""

from .base_skill import BaseSkill
from .registry import SkillRegistry, get_skill_registry

__all__ = ["BaseSkill", "SkillRegistry", "get_skill_registry"]

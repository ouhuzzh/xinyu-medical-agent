"""Core memory — L5 of the six-layer cognitive memory architecture.

A structured text block that is ALWAYS injected into the LLM context at the
start of every conversation turn.  It acts as the agent's "working knowledge"
about the user — a compact summary that is faster and more reliable than
retrieving from the semantic store every time.

Structure:
    [用户画像]
    姓名/称呼: ...
    年龄/性别: ...

    [已知病史]
    - 高血压 (8)
    - 青霉素过敏 (10)

    [偏好]
    - 偏好简洁回答 (4)

The core memory is updated by the LLM after memory extraction, consolidating
the most important user facts into a compact representation that fits within
CORE_MEMORY_MAX_LENGTH characters.

Stored in PostgreSQL (user_core_memory table, one row per user).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import config
import psycopg

logger = logging.getLogger(__name__)

# Default core memory template for a new user
_DEFAULT_CORE_MEMORY = """[用户画像]
(暂无信息)

[已知病史]
(暂无)

[用药/过敏]
(暂无)

[偏好]
(暂无)"""


class CoreMemoryStore:
    """PostgreSQL-backed core memory — the always-in-context user knowledge block."""

    def __init__(self):
        self._conninfo = (
            f"host={config.POSTGRES_HOST} "
            f"port={config.POSTGRES_PORT} "
            f"dbname={config.POSTGRES_DB} "
            f"user={config.POSTGRES_USER} "
            f"password={config.POSTGRES_PASSWORD}"
        )

    def _connect(self):
        return psycopg.connect(self._conninfo)

    def get_core_memory(self, user_id: str) -> str:
        """Get the core memory block for a user.  Returns default if none exists."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content FROM user_core_memory
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        return row[0] if row else _DEFAULT_CORE_MEMORY

    def save_core_memory(self, user_id: str, content: str):
        """Save or update the core memory block for a user."""
        # Truncate to max length
        if len(content) > config.CORE_MEMORY_MAX_LENGTH:
            content = content[:config.CORE_MEMORY_MAX_LENGTH - 3] + "..."
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_core_memory (user_id, content)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET content = EXCLUDED.content, updated_at = NOW()
                    """,
                    (user_id, content),
                )
            conn.commit()

    def update_core_memory_with_new_facts(
        self,
        user_id: str,
        new_memories: list[dict],
    ) -> str:
        """Update core memory by incorporating newly extracted user memories.

        This is a simple merge strategy: reads the existing core memory,
        appends new facts under the appropriate sections, and saves.

        Returns the updated core memory text.
        """
        current = self.get_core_memory(user_id)

        # Ensure the table exists (lazy creation for migration 013)
        self._ensure_table()

        # Categorize new memories
        medical_facts = []
        allergy_facts = []
        preference_facts = []
        general_facts = []

        for mem in new_memories:
            content = mem.get("content", "").strip()
            importance = mem.get("importance", 5)
            memory_type = mem.get("memory_type", "fact")
            entry = f"- {content} ({importance})"

            if memory_type == "medical":
                # Check if it's an allergy
                if any(kw in content.lower() for kw in ("过敏", "allergy", "变态反应")):
                    allergy_facts.append(entry)
                else:
                    medical_facts.append(entry)
            elif memory_type == "preference":
                preference_facts.append(entry)
            else:
                general_facts.append(entry)

        # Insert new facts into existing sections
        sections = {
            "已知病史": medical_facts,
            "用药/过敏": allergy_facts,
            "偏好": preference_facts,
        }

        lines = current.split("\n")
        updated_lines = list(lines)

        for section_name, facts in sections.items():
            if not facts:
                continue
            # Find the section header
            section_idx = None
            for i, line in enumerate(updated_lines):
                if line.strip().startswith(f"[{section_name}]"):
                    section_idx = i
                    break

            if section_idx is not None:
                # Find the next section or end of list
                next_section_idx = len(updated_lines)
                for i in range(section_idx + 1, len(updated_lines)):
                    if updated_lines[i].strip().startswith("["):
                        next_section_idx = i
                        break

                # Remove "(暂无)" placeholder if present
                for i in range(section_idx + 1, next_section_idx):
                    if "(暂无)" in updated_lines[i]:
                        updated_lines[i] = ""

                # Check for duplicates before inserting
                existing_text = "\n".join(updated_lines[section_idx:next_section_idx])
                new_facts = [f for f in facts if f.split(" (")[0].replace("- ", "") not in existing_text]

                # Insert new facts before the next section
                for j, fact in enumerate(new_facts):
                    updated_lines.insert(next_section_idx + j, fact)

        # Handle general facts (add to 用户画像 section if there's useful info)
        if general_facts:
            profile_idx = None
            for i, line in enumerate(updated_lines):
                if line.strip().startswith("[用户画像]"):
                    profile_idx = i
                    break
            if profile_idx is not None:
                # Remove "(暂无信息)" placeholder
                for i in range(profile_idx + 1, len(updated_lines)):
                    if "(暂无信息)" in updated_lines[i]:
                        existing_text = "\n".join(updated_lines)
                        new_general = [f for f in general_facts if f.split(" (")[0].replace("- ", "") not in existing_text]
                        if new_general:
                            updated_lines[i] = "\n".join(new_general)
                        break

        result = "\n".join(updated_lines)
        self.save_core_memory(user_id, result)
        return result

    def clear_core_memory(self, user_id: str):
        """Reset core memory for a user (for testing)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_core_memory WHERE user_id = %s", (user_id,))
            conn.commit()

    def _ensure_table(self):
        """Create the user_core_memory table if it doesn't exist."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_core_memory (
                        user_id     VARCHAR(128) PRIMARY KEY,
                        content     TEXT NOT NULL,
                        created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
                        updated_at  TIMESTAMP NOT NULL DEFAULT NOW()
                    )
                """)
            conn.commit()

    def status_info(self) -> Dict[str, Any]:
        if not config.CORE_MEMORY_ENABLED:
            return {"component": "core_memory", "mode": "disabled", "degraded": False,
                    "message": "Core memory is disabled by configuration."}
        return {"component": "core_memory", "mode": "active", "degraded": False,
                "message": "Core memory is available."}

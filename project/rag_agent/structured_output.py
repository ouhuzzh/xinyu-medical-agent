"""Structured output parsing — LLM JSON extraction with safe defaults.

Extracted from node_helpers for focused reusability.
"""

import logging
import re as _re
import json as _json

logger = logging.getLogger(__name__)


def _structured_output_llm(llm, schema, *, temperature: float = 0.1, max_tokens: int | None = None):
    """Return an object with .invoke() that calls LLM and parses JSON.

    Avoids LangChain's with_structured_output (uses response_format: json_object
    which SiliconFlow/Qwen doesn't support — long retry loop).
    Instead: get raw text, extract JSON via regex.
    """

    base = llm.with_config(temperature=temperature)
    base = base.bind(max_tokens=max_tokens or 256)

    class _StructureParser:
        """Thin wrapper: __call__ + .invoke()."""

        def __call__(self, messages: list):
            try:
                raw = str(base.invoke(messages).content or "").strip()
            except Exception:
                logger.warning("_StructureParser: LLM invoke failed for schema %s, returning default", schema.__name__)
                return _default()
            if not raw:
                logger.debug("_StructureParser: empty LLM response for schema %s", schema.__name__)
                return _default()
            # Try JSON patterns
            for pattern in [r"```(?:json)?\s*\n?(.*?)```", r"(\{.*\})"]:
                m = _re.search(pattern, raw, _re.DOTALL)
                if m:
                    try:
                        return schema(**_json.loads(m.group(1)))
                    except Exception:
                        logger.debug("_StructureParser: JSON pattern %s matched but schema parse failed for %s", pattern[:20], schema.__name__)
            try:
                return schema(**_json.loads(raw))
            except Exception:
                logger.debug("_StructureParser: raw JSON parse failed for schema %s", schema.__name__)
            logger.warning("_StructureParser: all parse attempts failed for schema %s, returning default", schema.__name__)
            return _default()

        def invoke(self, messages: list):
            return self(messages)

    def _default():
        """Return a safe default based on the schema."""
        vals = {}
        for fn, fi in schema.model_fields.items():
            t = str(fi.annotation)
            if "str" in t:
                vals[fn] = ""
            elif "bool" in t:
                vals[fn] = False
            elif "int" in t:
                vals[fn] = 0
            else:
                vals[fn] = ""
        return schema(**vals)

    return _StructureParser()

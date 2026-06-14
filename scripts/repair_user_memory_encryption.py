"""Inspect and repair legacy or corrupted encrypted user memory rows.

Usage:
    python scripts/repair_user_memory_encryption.py
    python scripts/repair_user_memory_encryption.py --apply --reencrypt-plaintext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from memory.user_memory_store import UserMemoryStore  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write repairs back to the database.")
    parser.add_argument(
        "--reencrypt-plaintext",
        action="store_true",
        help="Re-encrypt legacy plaintext rows in place when USER_MEMORY_ENCRYPT_PII=true.",
    )
    parser.add_argument(
        "--rewrite-unreadable",
        action="store_true",
        help="Rewrite unreadable encrypted rows to the sentinel marker. Use carefully.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="How many example row ids to include in the JSON report.",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    store = UserMemoryStore()
    report = store.repair_encryption_records(
        apply=args.apply,
        reencrypt_plaintext=args.reencrypt_plaintext,
        rewrite_invalid_format=True,
        rewrite_unreadable=args.rewrite_unreadable,
        sample_limit=max(1, args.sample_limit),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

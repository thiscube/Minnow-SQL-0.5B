"""Cut Spider's test split into a validation half and a final half, by database.

Choices such as how candidates are voted on must not be tuned on the questions that are
reported. The validation half is for making choices; the final half is scored once, at the end.
Whole databases go to one half, so no database is seen in both.

    python -m minnow.splits data/spider

writes test_val.json and test_final.json next to test.json.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

SOURCE = "test.json"
VALIDATION = "test_val.json"
FINAL = "test_final.json"


def in_validation(db_id: str) -> bool:
    """Put a database in the validation half or the final half, the same way every time."""
    return hashlib.sha256(db_id.encode()).digest()[0] % 2 == 0


def split(items: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Return the validation items and the final items."""
    validation = [item for item in items if in_validation(item["db_id"])]
    final = [item for item in items if not in_validation(item["db_id"])]
    return validation, final


def main(argv: Sequence[str] | None = None) -> int:
    """Write the two halves and print their sizes."""
    parser = argparse.ArgumentParser(description="Split Spider test into validation and final.")
    parser.add_argument("spider", type=Path, help="unpacked Spider folder")
    args = parser.parse_args(argv)
    items = json.loads((args.spider / SOURCE).read_text(encoding="utf-8"))
    for name, part in zip((VALIDATION, FINAL), split(items), strict=True):
        text = json.dumps(part, indent=1) + "\n"
        (args.spider / name).write_text(text, encoding="utf-8", newline="\n")
        databases = len({item["db_id"] for item in part})
        print(f"{name}: {len(part)} questions on {databases} databases")
    return 0


if __name__ == "__main__":
    sys.exit(main())

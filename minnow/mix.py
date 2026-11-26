"""Select the SynSQL questions a training run should see: plain SQL, one turn, with joins.

The model fails mostly on questions that join tables. This keeps the questions whose query
has JOINs and reads like Spider's SQL: no CTE, window, CASE, outer or cross join, and none of
the date, NULL and text functions that Spider never uses. A question on a database the
adapter was already trained on has to bring more joins than one on a new database.

    python -m minnow.mix data/spider --from train_synsql_new.json train_synsql_more.json
        --seen train_synsql.json --out-file train_synsql_joins.json
"""

import argparse
import re
import sys
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from minnow.spider import SpiderExample, load_split
from minnow.synsql import write_questions

MIN_JOINS_UNSEEN = 1
MIN_JOINS_SEEN = 2
_FUNCTIONS = (
    "strftime|julianday|date|datetime|time|coalesce|ifnull|nullif|cast|substr|substring|instr|iif"
)
_NOT_PLAIN = re.compile(
    r"\b(?:with|over|case|current_date|current_time|current_timestamp)\b"
    r"|\b(?:left|right|full|outer|cross)\s+(?:outer\s+)?join\b"
    rf"|\b(?:{_FUNCTIONS})\s*\(",
    re.IGNORECASE,
)
_JOIN = re.compile(r"\bjoin\b", re.IGNORECASE)
_QUOTED = re.compile(r"""'(?:[^']|'')*'|"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\]""")
_DIALOGUE = re.compile(r"\*\*(?:User|Assistant)\*\*|\b(?:User|Assistant):|[\r\n]")
_JOIN_GROUPS = {0: "no join", 1: "1 join", 2: "2 joins", 3: "3 or more joins"}


def _unquoted(sql: str) -> str:
    """Empty every quoted name and value, so their text is never read as a keyword."""
    return _QUOTED.sub("''", sql)


def joins(sql: str) -> int:
    """Count the JOIN keywords of a query."""
    return len(_JOIN.findall(_unquoted(sql)))


def is_plain(sql: str) -> bool:
    """Tell whether a query uses only the kind of SQL that Spider's queries use."""
    return not _NOT_PLAIN.search(_unquoted(sql))


def is_single_turn(question: str) -> bool:
    """Tell whether a question is one request on one line, not a dialogue."""
    return not _DIALOGUE.search(question)


def select(
    examples: Iterable[SpiderExample],
    seen: Collection[str],
    min_joins_unseen: int = MIN_JOINS_UNSEEN,
    min_joins_seen: int = MIN_JOINS_SEEN,
) -> list[SpiderExample]:
    """Keep the plain single-turn questions that have enough joins, in the order they came.

    A question on a database in seen needs min_joins_seen joins, any other min_joins_unseen.
    The same question on the same database is kept once.
    """
    chosen: list[SpiderExample] = []
    asked: set[tuple[str, str]] = set()
    for example in examples:
        needed = min_joins_seen if example.db_id in seen else min_joins_unseen
        wanted = (
            (example.db_id, example.question) not in asked
            and is_single_turn(example.question)
            and is_plain(example.query)
            and joins(example.query) >= needed
        )
        if wanted:
            asked.add((example.db_id, example.question))
            chosen.append(example)
    return chosen


def _size(examples: Sequence[SpiderExample]) -> str:
    databases = len({example.db_id for example in examples})
    return f"{len(examples)} questions, {databases} databases"


def summary(chosen: Sequence[SpiderExample], seen: Collection[str]) -> str:
    """Describe a selection: how much is on unseen and on seen databases, and by joins."""
    old = [example for example in chosen if example.db_id in seen]
    new = [example for example in chosen if example.db_id not in seen]
    groups = Counter(min(joins(example.query), 3) for example in chosen)
    shown = [f"{name}: {groups[key]}" for key, name in _JOIN_GROUPS.items() if key or groups[key]]
    return "\n".join(
        [f"on unseen databases: {_size(new)}", f"on seen databases: {_size(old)}", ", ".join(shown)]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the selection and print how many questions it holds."""
    parser = argparse.ArgumentParser(description="Select SynSQL questions with joins.")
    parser.add_argument("spider", type=Path, help="Spider folder that holds the question files")
    parser.add_argument("--from", dest="files", nargs="+", required=True, help="question files")
    parser.add_argument("--seen", nargs="*", default=[], help="files with databases trained on")
    parser.add_argument("--out-file", required=True, help="question file to write")
    parser.add_argument("--min-joins-unseen", type=int, default=MIN_JOINS_UNSEEN)
    parser.add_argument("--min-joins-seen", type=int, default=MIN_JOINS_SEEN)
    args = parser.parse_args(argv)
    if args.out_file in [*args.files, *args.seen]:
        parser.error("--out-file would overwrite a file that is read")
    seen = frozenset(e.db_id for name in args.seen for e in load_split(args.spider, name))
    examples = [e for name in args.files for e in load_split(args.spider, name)]
    chosen = select(examples, seen, args.min_joins_unseen, args.min_joins_seen)
    write_questions(args.spider / args.out_file, chosen)
    print(f"{len(chosen)} questions written to {args.spider / args.out_file}")
    print(summary(chosen, seen))
    return 0


if __name__ == "__main__":
    sys.exit(main())

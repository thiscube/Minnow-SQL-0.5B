"""Add a slice of SynSQL-2.5M to a Spider folder as extra training questions.

SynSQL is one 9 GB JSON list, so it is read as a stream. Only questions that look like
Spider's are kept: no extra knowledge needed, not the hardest level, and a query that runs.
A few questions are taken from each of many databases, picked in a fixed order, so the same
command always gives the same slice.

    python -m minnow.synsql data/synsql data/spider --databases 6000 --per-database 2

The questions go to train_synsql.json and each database to database/synsql_<name>/.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from pathlib import Path

from minnow.execution import QueryError, run_query
from minnow.spider import SpiderExample, database_path

OUT_FILE = "train_synsql.json"
PREFIX = "synsql_"
LEVELS = ("Simple", "Moderate", "Complex")
MAX_SQL_CHARACTERS = 400
_CHUNK = 1 << 20
_BETWEEN_ITEMS = re.compile(r"[\s,\[]*")
_COMMENT = re.compile(r"\s*/\*.*?\*/", re.DOTALL)
_TABLES = (
    "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL "
    "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
)


def read_items(path: Path) -> Iterator[dict]:
    """Yield the objects of a JSON list one at a time, without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer, position = "", 0
    with path.open(encoding="utf-8") as file:
        while True:
            position = _BETWEEN_ITEMS.match(buffer, position).end()
            try:
                item, position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = file.read(_CHUNK)
                if not more:
                    if buffer[position:].strip() not in ("", "]"):
                        raise ValueError(f"{path} is cut off or is not a JSON list.") from None
                    return
                buffer, position = buffer[position:] + more, 0
                continue
            yield item


def one_line(sql: str) -> str:
    """Write a query the way Spider does: on one line, without a closing semicolon."""
    return " ".join(sql.split()).rstrip("; ")


def looks_like_spider(item: dict) -> bool:
    """Tell whether a SynSQL item is the kind of question Spider asks."""
    sql = item["sql"]
    return (
        not item["external_knowledge"].strip()
        and item["sql_complexity"] in LEVELS
        and "--" not in sql
        and len(sql) <= MAX_SQL_CHARACTERS
    )


def candidates(items: Iterable[dict], per_database: int) -> dict[str, list[SpiderExample]]:
    """Keep up to per_database Spider-like questions per database, each of a different level."""
    kept: dict[str, list[SpiderExample]] = {}
    levels: dict[str, set[str]] = {}
    for item in items:
        db_id, level = item["db_id"], item["sql_complexity"]
        seen = levels.setdefault(db_id, set())
        if len(seen) < per_database and level not in seen and looks_like_spider(item):
            seen.add(level)
            example = SpiderExample(item["question"], one_line(item["sql"]), PREFIX + db_id)
            kept.setdefault(db_id, []).append(example)
    return kept


def fixed_order(db_ids: Iterable[str]) -> list[str]:
    """Sort database names in an order that is stable but unrelated to their names."""
    return sorted(db_ids, key=lambda db_id: hashlib.sha256(db_id.encode()).hexdigest())


def _runs(database: Path, sql: str) -> bool:
    try:
        run_query(database, sql)
    except QueryError:
        return False
    return True


def copy_without_comments(raw: bytes, target: Path) -> bool:
    """Write a database to target with the comments taken out of its table definitions.

    SynSQL describes every column in a comment, which Spider never does and which makes the
    schema about twice as long. Returns False when the database cannot be copied.
    """
    source = target.with_suffix(".raw")
    source.write_bytes(raw)
    target.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(source)) as old, closing(sqlite3.connect(target)) as new:
            for name, statement in old.execute(_TABLES).fetchall():
                new.execute(_COMMENT.sub("", statement))
                rows = old.execute(f'SELECT * FROM "{name}"')
                marks = ", ".join("?" * len(rows.description))
                new.executemany(f'INSERT INTO "{name}" VALUES ({marks})', rows)
            new.commit()
    except sqlite3.Error:
        return False
    finally:
        source.unlink()
    return True


def write_slice(
    kept: dict[str, list[SpiderExample]], archive: Path, spider: Path, databases: int
) -> list[SpiderExample]:
    """Unpack up to `databases` databases into the Spider folder with their runnable questions."""
    chosen: list[SpiderExample] = []
    used = 0
    with zipfile.ZipFile(archive) as packed:
        names = set(packed.namelist())
        for db_id in fixed_order(kept):
            member = f"databases/{db_id}/{db_id}.sqlite"
            if used == databases or member not in names:
                continue
            target = database_path(spider, PREFIX + db_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = copy_without_comments(packed.read(member), target)
            runnable = [e for e in kept[db_id] if copied and _runs(target, e.query)]
            if not runnable:
                target.unlink()
                target.parent.rmdir()
                continue
            chosen.extend(runnable)
            used += 1
    rows = [{"question": e.question, "query": e.query, "db_id": e.db_id} for e in chosen]
    text = json.dumps(rows, indent=1) + "\n"
    (spider / OUT_FILE).write_text(text, encoding="utf-8", newline="\n")
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    """Build the slice and print how many questions and databases it has."""
    parser = argparse.ArgumentParser(description="Add a SynSQL slice to a Spider folder.")
    parser.add_argument("synsql", type=Path, help="folder with data.json and databases.zip")
    parser.add_argument("spider", type=Path, help="unpacked Spider folder to add to")
    parser.add_argument("--databases", type=int, default=6000)
    parser.add_argument("--per-database", type=int, default=2)
    args = parser.parse_args(argv)
    kept = candidates(read_items(args.synsql / "data.json"), args.per_database)
    chosen = write_slice(kept, args.synsql / "databases.zip", args.spider, args.databases)
    used = len({example.db_id for example in chosen})
    print(f"{len(chosen)} questions on {used} databases written to {args.spider / OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

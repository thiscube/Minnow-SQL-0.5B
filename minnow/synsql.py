"""Add a slice of SynSQL-2.5M to a Spider folder as extra training questions.

SynSQL is one 9 GB JSON list, so it is read as a stream. Only questions that look like
Spider's are kept: no extra knowledge needed, not the hardest level, and a query that runs.
A few questions are taken from each of many databases, picked in a fixed order, so the same
command always gives the same slice.

    python -m minnow.synsql data/synsql data/spider --databases 6000 --per-database 2

The questions go to train_synsql.json and each database to database/synsql_<name>/.

More questions on the databases of slices that exist already, with nothing unpacked again:

    python -m minnow.synsql data/synsql data/spider --more-from train_synsql.json
        --per-level 3 --out-file train_synsql_more.json
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from pathlib import Path

from minnow.execution import QueryError, run_query
from minnow.spider import SpiderExample, database_path, load_split

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
    kept: dict[str, list[SpiderExample]],
    archive: Path,
    spider: Path,
    databases: int,
    out_file: str = OUT_FILE,
    exclude: frozenset[str] = frozenset(),
) -> list[SpiderExample]:
    """Unpack up to `databases` databases into the Spider folder with their runnable questions.

    Databases named in exclude (as their Spider-folder id) are passed over, so a second slice
    can be made from databases the first one did not use.
    """
    chosen: list[SpiderExample] = []
    used = 0
    with zipfile.ZipFile(archive) as packed:
        names = set(packed.namelist())
        for db_id in fixed_order(kept):
            member = f"databases/{db_id}/{db_id}.sqlite"
            if used == databases:
                break
            if member not in names or PREFIX + db_id in exclude:
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
    write_questions(spider / out_file, chosen)
    return chosen


def write_questions(path: Path, examples: Sequence[SpiderExample]) -> None:
    """Write questions as a question file in Spider's form."""
    rows = [{"question": e.question, "query": e.query, "db_id": e.db_id} for e in examples]
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8", newline="\n")


def more_questions(
    items: Iterable[dict], slices: Sequence[SpiderExample], spider: Path, per_level: int
) -> list[SpiderExample]:
    """Find up to per_level more runnable questions per level on each database of the slices.

    Nothing is unpacked: only databases the slices already use are looked at. A question the
    slices already ask on a database is passed over, and so is one whose query does not run.
    The result is grouped by database in a fixed order, so it is the same on every call.
    """
    asked = {(example.db_id, example.question) for example in slices}
    kept: dict[str, list[SpiderExample]] = {example.db_id: [] for example in slices}
    count: Counter[tuple[str, str]] = Counter()
    for item in items:
        db_id, level = PREFIX + item["db_id"], item["sql_complexity"]
        if db_id not in kept or count[db_id, level] == per_level or not looks_like_spider(item):
            continue
        example = SpiderExample(item["question"], one_line(item["sql"]), db_id)
        if (db_id, example.question) in asked:
            continue
        asked.add((db_id, example.question))
        if _runs(database_path(spider, db_id), example.query):
            count[db_id, level] += 1
            kept[db_id].append(example)
    return [example for db_id in fixed_order(kept) for example in kept[db_id]]


def _options(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a SynSQL slice to a Spider folder.")
    parser.add_argument("synsql", type=Path, help="folder with data.json and databases.zip")
    parser.add_argument("spider", type=Path, help="unpacked Spider folder to add to")
    parser.add_argument("--databases", type=int, default=6000)
    parser.add_argument("--per-database", type=int, default=2)
    parser.add_argument("--out-file", default=OUT_FILE, help="question file to write")
    parser.add_argument("--exclude", nargs="*", default=[], help="slices whose databases to skip")
    parser.add_argument("--more-from", nargs="*", default=[], help="slices to add questions to")
    parser.add_argument("--per-level", type=int, default=3, help="most per level with --more-from")
    args = parser.parse_args(argv)
    if args.out_file in args.more_from:
        parser.error("--out-file would overwrite a slice given to --more-from")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Build the slice and print how many questions and databases it has."""
    args = _options(argv)
    items = read_items(args.synsql / "data.json")
    if args.more_from:
        slices = [e for name in args.more_from for e in load_split(args.spider, name)]
        chosen = more_questions(items, slices, args.spider, args.per_level)
        write_questions(args.spider / args.out_file, chosen)
    else:
        exclude = frozenset(e.db_id for name in args.exclude for e in load_split(args.spider, name))
        kept = candidates(items, args.per_database)
        archive = args.synsql / "databases.zip"
        chosen = write_slice(kept, archive, args.spider, args.databases, args.out_file, exclude)
    used = len({example.db_id for example in chosen})
    print(f"{len(chosen)} questions on {used} databases written to {args.spider / args.out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

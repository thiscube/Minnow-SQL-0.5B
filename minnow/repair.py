"""Repair the picks of a run without a model: empty results last, then missing columns.

Two steps that use only the database, never the reference query:

1. A run that saved its candidates is voted again with empty results last (minnow.select), the
   rule the vote across runs already uses. A run without saved candidates keeps its picks.
2. A pick that fails with "no such column" gets the column reached another way: through the
   alias of the table in the query that has it, else by natural-joining the table that holds
   it, else by the closest column name. The repair is kept only when the query then runs, and
   a pick that already runs is never touched.

    python -m minnow.repair outputs/repaired outputs/run --spider data/spider

Several runs of one model on one split can be named. Their candidates are pooled per question,
all of the first run's and then the next run's, and voted once, so a tie goes to the run named
first and its greedy answer. Runs of different models are voted by python -m minnow.combine.

The new folder gets predictions.jsonl, outcomes.jsonl and summary.json and no candidates.jsonl,
so python -m minnow.combine votes over the repaired pick of each run.
"""

import argparse
import difflib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from minnow.combine import combine
from minnow.evaluate import CANDIDATES_FILE, append_prediction, load_candidates, load_predictions
from minnow.execution import QueryError, run_query
from minnow.outcomes import (
    OUTCOMES_FILE,
    PREDICTIONS_FILE,
    Breakdown,
    breakdown,
    format_breakdown,
    judge_all,
    save_outcomes,
)
from minnow.runfiles import SUMMARY_FILE, accuracy, read_text
from minnow.schema import connect_read_only
from minnow.spider import SpiderExample, database_path, load_split

Schema = dict[str, list[str]]
Sources = list[tuple[str, str]]

_KEYWORDS = frozenset({
    "where", "join", "on", "group", "order", "limit", "inner", "left", "right", "outer",
    "cross", "natural", "union", "intersect", "except", "having", "as", "using", "full",
    "select", "from", "and", "or", "not", "in", "like", "between", "is", "null",
})  # fmt: skip
_SOURCE = re.compile(r"\b(?:from|join)\s+[\"`\[]?(\w+)[\"`\]]?(?:\s+(?:as\s+)?(\w+))?", re.I)
_TABLES = "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY rowid"
_MISSING = "no such column"
_MAX_ROUNDS = 4
_CLOSE_ENOUGH = 0.6
_VOTED = "picks voted again with empty results last"
_KEPT = "picks kept (no saved candidates)"
_POOLED = "candidates of the {} runs pooled and voted once with empty results last"


def load_schema(database: Path) -> Schema:
    """Return the column names of every table, in the order the tables were created."""
    connection = connect_read_only(database)
    try:
        tables = [row[0] for row in connection.execute(_TABLES)]
        return {
            table: [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            for table in tables
        }
    finally:
        connection.close()


def sources(sql: str, schema: Schema) -> Sources:
    """Return (alias, table) for every table the query reads, in order of appearance.

    A table without an alias that is directly followed by JOIN hides the table joined next,
    because the keyword is read in the alias's place. The measured repair works this way.
    """
    by_lower = {table.lower(): table for table in schema}
    found = []
    for match in _SOURCE.finditer(sql):
        table = by_lower.get(match.group(1).lower())
        if table is None:
            continue
        alias = match.group(2)
        if alias is None or alias.lower() in _KEYWORDS:
            alias = match.group(1)
        found.append((alias, table))
    return found


def _column(schema: Schema, table: str, name: str) -> str | None:
    """Return a table's column of that name, whatever its case."""
    return next((column for column in schema[table] if column.lower() == name.lower()), None)


def _reference(qualifier: str, column: str) -> re.Pattern[str]:
    """Return the pattern that finds the failing column reference in the query."""
    if qualifier:
        return re.compile(rf"\b{re.escape(qualifier)}\.{re.escape(column)}\b", re.I)
    return re.compile(rf"(?<![\w.]){re.escape(column)}\b", re.I)


def _other_alias(
    sql: str, qualifier: str, column: str, schema: Schema, read: Sources
) -> str | None:
    """Point an alias mix-up at the first other table in the query that has the column."""
    if not qualifier:
        return None
    owners = [
        alias
        for alias, table in read
        if alias.lower() != qualifier.lower() and _column(schema, table, column)
    ]
    if not owners:
        return None
    return _reference(qualifier, column).sub(lambda _: f"{owners[0]}.{column}", sql)


def _joined_table(
    sql: str, qualifier: str, column: str, schema: Schema, read: Sources
) -> str | None:
    """Reach a column of a table the query does not read: natural-join that table to a source."""
    read_tables = {table for _, table in read}
    for alias, table in read:
        if qualifier and alias.lower() != qualifier.lower():
            continue
        mine = {name.lower() for name in schema[table]}
        for other in schema:
            if other in read_tables or _column(schema, other, column) is None:
                continue
            if not mine & {name.lower() for name in schema[other]}:
                continue
            source = re.compile(
                rf"\b(from|join)\s+[\"`\[]?{re.escape(table)}[\"`\]]?"
                rf"(?:\s+(?:as\s+)?{re.escape(alias)}\b)?",
                re.I,
            )
            joined = f'(SELECT * FROM "{table}" NATURAL JOIN "{other}") AS {alias}'
            return source.sub(lambda m, j=joined: f"{m.group(1)} {j}", sql, count=1)
    return None


def _closest_name(
    sql: str, qualifier: str, column: str, schema: Schema, read: Sources
) -> str | None:
    """Use the closest column name of the table the alias names, or of every table read."""
    own = [table for alias, table in read if qualifier and alias.lower() == qualifier.lower()]
    tables = own or [table for _, table in read]
    names = {name.lower(): name for table in tables for name in schema[table]}
    close = difflib.get_close_matches(column.lower(), sorted(names), n=1, cutoff=_CLOSE_ENOUGH)
    if not close:
        return None
    new = f"{qualifier}.{names[close[0]]}" if qualifier else names[close[0]]
    return _reference(qualifier, column).sub(lambda _: new, sql)


def _fix_missing(sql: str, message: str, schema: Schema) -> str | None:
    """Return the query with the missing column of the error reached another way, or None."""
    qualifier, _, column = message.split(":", 1)[1].strip().rpartition(".")
    read = sources(sql, schema)
    for fix in (_other_alias, _joined_table, _closest_name):
        fixed = fix(sql, qualifier, column, schema, read)
        if fixed is not None:
            return fixed
    return None


def _error(database: Path, sql: str) -> str | None:
    """Return why a query does not run, or None when it runs."""
    try:
        run_query(database, sql)
    except QueryError as error:
        return str(error)
    return None


def repair_columns(database: Path, sql: str) -> str:
    """Return a repaired query that runs, or the query as it was.

    Only a query that fails with "no such column" is changed, one missing column per round,
    and the original comes back unless the repaired query runs.
    """
    schema: Schema = {}
    current = sql
    for _ in range(_MAX_ROUNDS):
        error = _error(database, current)
        if error is None:
            return current
        if not error.startswith(_MISSING):
            return sql
        schema = schema or load_schema(database)
        fixed = _fix_missing(current, error, schema)
        if fixed is None or fixed == current:
            return sql
        current = fixed
    return sql


def repair_picks(
    examples: Sequence[SpiderExample], picks: Mapping[int, str], spider: Path
) -> dict[int, str]:
    """Return every pick, repaired when it fails on a missing column."""
    return {
        index: repair_columns(database_path(spider, examples[index].db_id), sql)
        for index, sql in picks.items()
    }


def pool_candidates(runs: Sequence[Mapping[int, Sequence[str]]]) -> dict[int, list[str]]:
    """Return each question's candidates of every run: all of the first run's, then the next."""
    pooled: dict[int, list[str]] = {}
    for run in runs:
        for index, candidates in run.items():
            pooled.setdefault(index, []).extend(candidates)
    return pooled


def _summary(
    sources: Sequence[Mapping[str, object]], voted: bool, counts: Breakdown
) -> dict[str, object]:
    """Return the first run's summary with the new counts and what was done to the picks."""
    several = len(sources) > 1
    done = _POOLED.format(len(sources)) if several else _VOTED if voted else _KEPT
    decodings = " + ".join(str(source["decoding"]) for source in sources)
    summary = {
        **sources[0],
        "correct": counts.correct,
        "total": counts.total,
        "execution_accuracy": accuracy(counts.correct, counts.total),
        "decoding": f"{decodings}; {done}, then repaired where a column was missing",
    }
    if several:
        seconds = [source.get("generation_seconds") for source in sources]
        summary["generation_seconds"] = None if None in seconds else sum(seconds)
    return summary


def _sources(
    parser: argparse.ArgumentParser, runs: Sequence[Path], split: str
) -> list[dict[str, object]]:
    """Return every run's summary, refusing a folder that is no run of that split and model."""
    sources: list[dict[str, object]] = []
    for run in runs:
        for name in (PREDICTIONS_FILE, SUMMARY_FILE):
            if not (run / name).is_file():
                parser.error(f"{run / name} does not exist, so there is no run to repair.")
        source = json.loads(read_text(run / SUMMARY_FILE))
        if source["split"] != split:
            parser.error(f"{run} answered {source['split']}, not {split}. Pass its --split.")
        sources.append(source)
        if source.get("model") != sources[0].get("model"):
            parser.error(f"{run} is by another model than {runs[0]}. Use minnow.combine.")
    return sources


def main(argv: Sequence[str] | None = None) -> int:
    """Write a new run folder with the picks voted again and repaired, and print its counts."""
    parser = argparse.ArgumentParser(
        description="Vote a run again with empty results last and repair picks that do not run."
    )
    parser.add_argument("out", type=Path, help="new run folder to write")
    parser.add_argument(
        "runs", type=Path, nargs="+", metavar="run", help="run folder to repair; more are pooled"
    )
    parser.add_argument("--spider", type=Path, required=True, help="unpacked Spider folder")
    parser.add_argument("--split", default="dev.json", help="question file the runs answered")
    args = parser.parse_args(argv)
    if (args.out / PREDICTIONS_FILE).exists():
        parser.error(f"{args.out} already holds a run.")
    sources = _sources(parser, args.runs, args.split.removesuffix(".json"))
    saved = [load_candidates(run / CANDIDATES_FILE) for run in args.runs]
    if len(saved) > 1 and not all(saved):
        parser.error("Runs can only be pooled when each of them saved its candidates.")
    examples = load_split(args.spider, args.split)
    voted = combine(examples, [pool_candidates(saved)], args.spider, prefer_rows=True)
    picks = {**load_predictions(args.runs[0] / PREDICTIONS_FILE), **voted}
    repaired = repair_picks(examples, picks, args.spider)
    args.out.mkdir(parents=True, exist_ok=True)
    for index in sorted(repaired):
        append_prediction(args.out / PREDICTIONS_FILE, index, repaired[index])
    outcomes = judge_all(examples, repaired, args.spider)
    save_outcomes(args.out / OUTCOMES_FILE, outcomes)
    counts = breakdown(outcomes)
    text = json.dumps(_summary(sources, bool(voted), counts), indent=2) + "\n"
    (args.out / SUMMARY_FILE).write_text(text, encoding="utf-8", newline="\n")
    print(format_breakdown(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

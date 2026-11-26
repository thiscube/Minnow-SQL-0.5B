"""Show each question the tables of its database in a shuffled order.

One model's candidates agree less by accident when some of them were written with the tables
in another order (multiple-prompt self-consistency, as in MCS-SQL). The order depends only on
the seed and on the question's position in its split, so the same command builds the same
prompts every time.

The plain prompt does not change. minnow.schema and minnow.prompts are left as they are,
because the training cache is named after their code (minnow.sft). This module reads the
tables once more, one text per table, and refuses them unless they are the plain prompt's.
"""

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from minnow.prompts import PromptBook
from minnow.schema import _TABLES, _example_rows, connect_read_only
from minnow.spider import SpiderExample, database_path, schemas

_BETWEEN = "\n\n"


def table_texts(path: Path, sample_rows: int = 0) -> list[str]:
    """Return the text that minnow.schema.create_statements joins, as one piece per table."""
    connection = connect_read_only(path)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        texts = []
        for table, statement in connection.execute(_TABLES).fetchall():
            texts.append(statement.strip())
            if sample_rows:
                texts[-1] += "\n" + _example_rows(connection, table, sample_rows)
    finally:
        connection.close()
    return texts


def shuffled(items: Sequence[str], seed: int, index: int) -> list[str]:
    """Return the items in an order decided by the seed and the question's position alone.

    Each item is placed by a hash, not by Python's random numbers, so the order is the same on
    every machine and version. Two or more items never keep the order they came in, because a
    shuffled prompt that equals the plain one would add nothing.
    """
    order = sorted(
        range(len(items)),
        key=lambda place: hashlib.sha256(f"{seed}:{index}:{place}".encode()).digest(),
    )
    if len(order) > 1 and order == sorted(order):
        order = order[1:] + order[:1]
    return [items[place] for place in order]


class ShuffledBook:
    """A PromptBook that, given a seed, lists the tables in another order for each question."""

    def __init__(
        self,
        spider: Path,
        examples: Iterable[SpiderExample],
        sample_rows: int = 0,
        value_hints: bool = False,
        seed: int | None = None,
    ) -> None:
        """Read what PromptBook reads and, with a seed, every database's tables one by one."""
        examples = list(examples)
        self._book = PromptBook(spider, examples, sample_rows, value_hints)
        self._seed = seed
        self._tables: dict[str, list[str]] = {}
        if seed is None:
            return
        for db_id, text in schemas(spider, examples, sample_rows).items():
            tables = table_texts(database_path(spider, db_id), sample_rows)
            if _BETWEEN.join(tables) != text:
                raise ValueError(f"The tables read for {db_id} are not the ones in its prompt.")
            self._tables[db_id] = tables

    def schema(self, example: SpiderExample, index: int) -> str:
        """Return the schema text for the question at that position of its split.

        Without a seed this is PromptBook's text. With one, only the order of the tables
        changes: each keeps its example rows, and the value hints stay at the end as they are.
        """
        plain = self._book.schema(example)
        if self._seed is None:
            return plain
        tables = self._tables[example.db_id]
        hints = plain[len(_BETWEEN.join(tables)) :]
        return _BETWEEN.join(shuffled(tables, self._seed, index)) + hints

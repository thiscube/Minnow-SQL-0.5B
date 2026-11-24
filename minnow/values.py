"""Find the database values a question mentions, so the prompt can say where they live.

A small model often filters the wrong column for a name it read in the question, such as
Code = 'Angola' when 'Angola' is in Name. Looking the words up in the database's text columns
and listing the hits under the schema tells the model which column holds them.
"""

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from minnow.schema import connect_read_only

MAX_VALUES_PER_COLUMN = 5000
MIN_VALUE_LENGTH = 3
MAX_VALUE_LENGTH = 60
MAX_HINTS = 8
_WORD = re.compile(r"\w+")
_TEXT_TYPES = ("CHAR", "TEXT", "CLOB")
_TABLES = (
    "SELECT name FROM sqlite_master WHERE type = 'table' "
    "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY rowid"
)
_COMMON_WORDS = (
    "the and for with from that this what which who how many much all any are was were has "
    "have had not yes one two name names number type list show find give each"
)
_COMMON = frozenset(_COMMON_WORDS.split())


@dataclass(frozen=True)
class Value:
    """One text value and the column it was found in."""

    table: str
    column: str
    text: str


def _text_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    """Return the columns declared as text, or with no type at all."""
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [row[1] for row in rows if not row[2] or any(t in row[2].upper() for t in _TEXT_TYPES)]


def value_index(path: Path) -> list[Value]:
    """Return the distinct short text values of every text column of a database."""
    connection = connect_read_only(path)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    values = []
    try:
        for (table,) in connection.execute(_TABLES).fetchall():
            for column in _text_columns(connection, table):
                query = f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT {MAX_VALUES_PER_COLUMN}'
                for (value,) in connection.execute(query):
                    text = value.strip() if isinstance(value, str) else ""
                    if MIN_VALUE_LENGTH <= len(text) <= MAX_VALUE_LENGTH:
                        values.append(Value(table, column, text))
    finally:
        connection.close()
    return values


def _words(text: str) -> tuple[str, ...]:
    """Split text into lower-case words."""
    return tuple(word.lower() for word in _WORD.findall(text))


def mentioned(values: Iterable[Value], question: str) -> list[Value]:
    """Return the values whose words appear together, in order, in the question.

    Values made only of common words or digits are left out. Longer values come first.
    """
    text = " " + " ".join(_words(question)) + " "
    hits = []
    for value in values:
        parts = _words(value.text)
        meaningful = any(part not in _COMMON and not part.isdigit() for part in parts)
        if meaningful and f" {' '.join(parts)} " in text:
            hits.append(value)
    hits.sort(key=lambda value: (-len(value.text), value.table, value.column, value.text))
    return hits[:MAX_HINTS]


def hint(values: Iterable[Value]) -> str:
    """Write the found values as a comment to put under the schema, or nothing if none."""
    lines = [f"{v.table}.{v.column} = '{v.text.replace(chr(39), chr(39) * 2)}'" for v in values]
    if not lines:
        return ""
    return "/*\nValues from the question found in the database:\n" + "\n".join(lines) + "\n*/"

"""Describe a SQLite database's tables as text for a prompt."""

import sqlite3
from pathlib import Path

_TABLES = (
    "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL ORDER BY rowid"
)
_MAX_CELL = 30


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite file so that nothing can change it."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _cell(value: object) -> str:
    text = "NULL" if value is None else " ".join(str(value).split())
    return text if len(text) <= _MAX_CELL else text[: _MAX_CELL - 3] + "..."


def _example_rows(connection: sqlite3.Connection, table: str, count: int) -> str:
    cursor = connection.execute(f'SELECT * FROM "{table}" LIMIT {int(count)}')
    lines = [" | ".join(column[0] for column in cursor.description)]
    lines += [" | ".join(_cell(value) for value in row) for row in cursor]
    return "/*\nExample rows:\n" + "\n".join(lines) + "\n*/"


def create_statements(path: Path, sample_rows: int = 0) -> str:
    """Return every table's CREATE TABLE statement, in the order the tables were created.

    With sample_rows above zero, each statement is followed by a comment that shows that many
    rows of the table, so the reader can see what the columns hold.
    """
    connection = connect_read_only(path)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        parts = []
        for table, statement in connection.execute(_TABLES).fetchall():
            parts.append(statement.strip())
            if sample_rows:
                parts[-1] += "\n" + _example_rows(connection, table, sample_rows)
    finally:
        connection.close()
    return "\n\n".join(parts)

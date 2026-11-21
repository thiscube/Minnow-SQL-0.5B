"""Describe a SQLite database's tables as text for a prompt."""

import sqlite3
from pathlib import Path

_TABLE_SQL = "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL ORDER BY rowid"


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite file so that nothing can change it."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def create_statements(path: Path) -> str:
    """Return the CREATE TABLE statements of every table, in the order they were created."""
    connection = connect_read_only(path)
    try:
        statements = [row[0].strip() for row in connection.execute(_TABLE_SQL)]
    finally:
        connection.close()
    return "\n\n".join(statements)

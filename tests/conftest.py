import sqlite3
from pathlib import Path

import pytest

SCHEMA = """
CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, singer_id INTEGER,
    FOREIGN KEY (singer_id) REFERENCES singer (singer_id));
CREATE VIEW adults AS SELECT * FROM singer WHERE age >= 18;
INSERT INTO singer VALUES (1, 'Ana', 30), (2, 'Ben', 25), (3, 'Cleo', 30);
INSERT INTO concert VALUES (10, 1), (11, 1), (12, 3);
"""


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "concert.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.commit()
    connection.close()
    return path

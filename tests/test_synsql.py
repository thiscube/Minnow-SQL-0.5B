import json
import sqlite3
import zipfile
from contextlib import closing

import pytest

from minnow.execution import run_query
from minnow.schema import create_statements
from minnow.spider import SpiderExample, load_split
from minnow.synsql import (
    candidates,
    copy_without_comments,
    fixed_order,
    looks_like_spider,
    main,
    one_line,
    read_items,
)


def item(db_id, level, sql, question="How many singers?", knowledge=""):
    return {
        "db_id": db_id,
        "sql_complexity": level,
        "question": question,
        "external_knowledge": knowledge,
        "sql": sql,
    }


ITEMS = [
    item("music", "Simple", "SELECT count(*)\nFROM singer;"),
    item("music", "Simple", "SELECT name FROM singer"),
    item("music", "Moderate", "SELECT max(age) FROM singer"),
    item("music", "Complex", "SELECT min(age) FROM singer"),
    item("broken", "Simple", "SELECT nothing FROM singer"),
    item("missing", "Simple", "SELECT 1"),
    item("music", "Highly Complex", "SELECT 2"),
]


@pytest.fixture
def synsql(tmp_path, database):
    folder = tmp_path / "synsql"
    folder.mkdir()
    (folder / "data.json").write_text(json.dumps(ITEMS, indent=2), encoding="utf-8")
    with zipfile.ZipFile(folder / "databases.zip", "w") as packed:
        for name in ("music", "broken"):
            packed.write(database, f"databases/{name}/{name}.sqlite")
    return folder


def test_stream_gives_the_same_items_as_loading(tmp_path, monkeypatch):
    monkeypatch.setattr("minnow.synsql._CHUNK", 7)
    for text in (json.dumps(ITEMS, indent=2), json.dumps(ITEMS), "[]", "[\n]\n"):
        path = tmp_path / "data.json"
        path.write_text(text, encoding="utf-8")
        assert list(read_items(path)) == json.loads(text)


@pytest.mark.parametrize("text", ['[{"a": 1}, {"b": 2', '[{"a": 1}, {"b": tru }, {"c": 3}]'])
def test_cut_off_or_damaged_file_is_an_error_not_a_short_list(tmp_path, text):
    path = tmp_path / "data.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="cut off"):
        list(read_items(path))


def test_query_is_written_on_one_line_without_semicolon():
    assert one_line("SELECT a,\n  b\nFROM t ;\n") == "SELECT a, b FROM t"


def test_only_spider_like_items_are_kept():
    assert looks_like_spider(item("d", "Complex", "SELECT 1"))
    assert not looks_like_spider(item("d", "Highly Complex", "SELECT 1"))
    assert not looks_like_spider(item("d", "Simple", "SELECT 1", knowledge="Big means over 5."))
    assert not looks_like_spider(item("d", "Simple", "SELECT 1 -- note"))
    assert not looks_like_spider(item("d", "Simple", "SELECT " + "1, " * 200 + "1"))


def test_each_database_gives_one_question_per_level_up_to_the_limit():
    kept = candidates(ITEMS, per_database=2)

    assert kept["music"] == [
        SpiderExample("How many singers?", "SELECT count(*) FROM singer", "synsql_music"),
        SpiderExample("How many singers?", "SELECT max(age) FROM singer", "synsql_music"),
    ]


def test_order_is_stable_and_not_alphabetical():
    names = [f"db{number}" for number in range(20)]

    assert fixed_order(names) == fixed_order(reversed(names))
    assert fixed_order(names) != sorted(names)


def test_command_line_adds_runnable_questions_and_their_databases(tmp_path, synsql, capsys):
    spider = tmp_path / "spider"
    spider.mkdir()

    assert main([str(synsql), str(spider), "--databases", "5"]) == 0

    written = load_split(spider, "train_synsql.json")
    assert [example.query for example in written] == [
        "SELECT count(*) FROM singer",
        "SELECT max(age) FROM singer",
    ]
    assert (spider / "database" / "synsql_music" / "synsql_music.sqlite").exists()
    assert not (spider / "database" / "synsql_broken").exists()
    assert not (spider / "database" / "synsql_missing").exists()
    assert "2 questions on 1 databases" in capsys.readouterr().out


def test_copy_drops_column_comments_and_keeps_the_rows(tmp_path):
    source = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute(
            'CREATE TABLE "pet" (\n  "pet_id" INTEGER PRIMARY KEY AUTOINCREMENT /* Unique id */,'
            '\n  "name" TEXT /* Name of\n the pet */\n)'
        )
        connection.execute("INSERT INTO pet (name) VALUES ('Rex'), ('Mia')")
        connection.commit()
    target = tmp_path / "out" / "pet.sqlite"
    target.parent.mkdir()
    target.write_text("left over from an earlier run", encoding="utf-8")

    assert copy_without_comments(source.read_bytes(), target)

    schema = create_statements(target)
    assert "/*" not in schema
    assert '"name" TEXT' in schema
    assert run_query(target, "SELECT name FROM pet ORDER BY pet_id") == [("Rex",), ("Mia",)]
    assert list(target.parent.iterdir()) == [target]


def test_copy_reports_a_file_that_is_not_a_database(tmp_path):
    target = tmp_path / "bad.sqlite"

    assert not copy_without_comments(b"this is not a database", target)
    assert not target.with_suffix(".raw").exists()


def test_database_limit_is_respected(tmp_path, synsql):
    spider = tmp_path / "spider"
    spider.mkdir()

    main([str(synsql), str(spider), "--databases", "0"])

    assert load_split(spider, "train_synsql.json") == []

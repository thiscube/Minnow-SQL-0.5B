import json
from pathlib import Path

from minnow.spider import SpiderExample, database_path, load_split, schemas


def test_loads_question_query_and_database(tmp_path):
    item = {"question": "How many?", "query": "SELECT 1", "db_id": "shop", "sql": {}}
    (tmp_path / "dev.json").write_text(json.dumps([item]), encoding="utf-8")

    assert load_split(tmp_path, "dev.json") == [SpiderExample("How many?", "SELECT 1", "shop")]


def test_schemas_cover_each_database_once(tmp_path, database):
    import shutil

    (tmp_path / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, tmp_path / "database" / "concert" / "concert.sqlite")
    examples = [
        SpiderExample("a", "SELECT 1", "concert"),
        SpiderExample("b", "SELECT 2", "concert"),
    ]

    plain = schemas(tmp_path, examples)
    with_rows = schemas(tmp_path, examples, sample_rows=1)

    assert list(plain) == ["concert"]
    assert "CREATE TABLE singer" in plain["concert"]
    assert "Example rows" not in plain["concert"]
    assert "1 | Ana | 30" in with_rows["concert"]


def test_database_path_follows_the_spider_layout():
    assert database_path(Path("spider"), "shop") == Path("spider/database/shop/shop.sqlite")


def test_test_split_databases_are_found_in_their_own_folder(tmp_path):
    test = tmp_path / "test_database" / "zoo" / "zoo.sqlite"
    test.parent.mkdir(parents=True)
    test.touch()
    both = tmp_path / "database" / "shop" / "shop.sqlite"
    both.parent.mkdir(parents=True)
    both.touch()
    (tmp_path / "test_database" / "shop").mkdir()
    (tmp_path / "test_database" / "shop" / "shop.sqlite").touch()

    assert database_path(tmp_path, "zoo") == test
    assert database_path(tmp_path, "shop") == both

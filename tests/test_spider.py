import json
from pathlib import Path

from minnow.spider import SpiderExample, database_path, load_split


def test_loads_question_query_and_database(tmp_path):
    item = {"question": "How many?", "query": "SELECT 1", "db_id": "shop", "sql": {}}
    (tmp_path / "dev.json").write_text(json.dumps([item]), encoding="utf-8")

    assert load_split(tmp_path, "dev.json") == [SpiderExample("How many?", "SELECT 1", "shop")]


def test_database_path_follows_the_spider_layout():
    assert database_path(Path("spider"), "shop") == Path("spider/database/shop/shop.sqlite")

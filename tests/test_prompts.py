import shutil

import pytest

from minnow.prompts import PromptBook
from minnow.spider import SpiderExample

NAMED = SpiderExample("How old is Ana?", "SELECT age FROM singer WHERE name = 'Ana'", "concert")
PLAIN = SpiderExample("How many singers?", "SELECT count(*) FROM singer", "concert")


@pytest.fixture
def spider(tmp_path, database):
    folder = tmp_path / "spider"
    (folder / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, folder / "database" / "concert" / "concert.sqlite")
    return folder


def test_without_hints_the_schema_is_the_same_for_every_question(spider):
    book = PromptBook(spider, [NAMED, PLAIN])

    assert book.schema(NAMED) == book.schema(PLAIN)
    assert "CREATE TABLE singer" in book.schema(NAMED)


def test_hints_name_the_column_holding_a_mentioned_value(spider):
    book = PromptBook(spider, [NAMED, PLAIN], value_hints=True)

    assert book.schema(NAMED).endswith(
        "\n\n/*\nValues from the question found in the database:\nsinger.name = 'Ana'\n*/"
    )
    assert book.schema(PLAIN) == PromptBook(spider, [PLAIN]).schema(PLAIN)

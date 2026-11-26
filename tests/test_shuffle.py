import sqlite3

import pytest

from minnow import shuffle
from minnow.prompts import PromptBook
from minnow.schema import create_statements
from minnow.shuffle import ShuffledBook, shuffled, table_texts
from minnow.spider import SpiderExample

FESTIVAL = """
CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
CREATE TABLE concert (concert_id INTEGER PRIMARY KEY,

    singer_id INTEGER);
CREATE TABLE stadium (stadium_id INTEGER PRIMARY KEY, city TEXT);
CREATE TABLE ticket (ticket_id INTEGER PRIMARY KEY, concert_id INTEGER, price REAL);
CREATE TABLE band (band_id INTEGER PRIMARY KEY, name TEXT);
CREATE VIEW adults AS SELECT * FROM singer WHERE age >= 18;
INSERT INTO singer VALUES (1, 'Ana', 30), (2, 'Ben', 25), (3, 'Cleo', 30);
INSERT INTO concert VALUES (10, 1), (11, 2);
INSERT INTO stadium VALUES (5, 'Lisbon');
INSERT INTO ticket VALUES (7, 10, 20.5);
"""
SOLO = "CREATE TABLE note (body TEXT); INSERT INTO note VALUES ('hello');"
NAMED = SpiderExample("How old is Ana?", "SELECT age FROM singer WHERE name = 'Ana'", "festival")
PLAIN = SpiderExample("How many singers?", "SELECT count(*) FROM singer", "festival")
NOTE = SpiderExample("How many notes?", "SELECT count(*) FROM note", "solo")
QUESTIONS = [NAMED, PLAIN, NOTE]
HINTS = "\n\n/*\nValues from the question found in the database:\nsinger.name = 'Ana'\n*/"
LETTERS = list("abcdef")


@pytest.fixture
def spider(tmp_path):
    folder = tmp_path / "spider"
    for name, script in (("festival", FESTIVAL), ("solo", SOLO)):
        (folder / "database" / name).mkdir(parents=True)
        connection = sqlite3.connect(folder / "database" / name / f"{name}.sqlite")
        connection.executescript(script)
        connection.commit()
        connection.close()
    return folder


@pytest.fixture
def festival(spider):
    return spider / "database" / "festival" / "festival.sqlite"


def is_reordered(text, plain, tables):
    """True when the text holds every table once and nothing else, in another order."""
    return text != plain and len(text) == len(plain) and all(text.count(t) == 1 for t in tables)


@pytest.mark.parametrize("rows", [0, 2])
def test_table_texts_joined_are_exactly_the_schema_text(festival, rows):
    texts = table_texts(festival, rows)

    assert len(texts) == 5
    assert "\n\n".join(texts) == create_statements(festival, rows)
    assert texts[1].startswith("CREATE TABLE concert (concert_id INTEGER PRIMARY KEY,\n\n")


def test_order_is_decided_by_the_seed_and_the_position_alone():
    assert shuffled(LETTERS, 0, 0) == ["e", "b", "c", "d", "a", "f"]
    assert shuffled(tuple(LETTERS), 0, 1) == ["d", "a", "e", "b", "c", "f"]
    assert shuffled(LETTERS, 7, 0) == ["e", "d", "f", "b", "c", "a"]
    assert len({tuple(shuffled(LETTERS, 0, index)) for index in range(30)}) > 20


def test_every_item_is_kept_exactly_once():
    for seed in range(3):
        for index in range(50):
            assert sorted(shuffled(LETTERS, seed, index)) == LETTERS


def test_two_or_more_items_never_keep_their_order():
    pairs = [shuffled(["a", "b"], seed, index) for seed in range(5) for index in range(50)]
    triples = [shuffled(["a", "b", "c"], 0, index) for index in range(200)]

    assert pairs == [["b", "a"]] * 250
    assert ["a", "b", "c"] not in triples
    assert len({tuple(triple) for triple in triples}) == 5
    assert shuffled(["a"], 0, 0) == ["a"]
    assert shuffled([], 0, 0) == []


@pytest.mark.parametrize(("rows", "hints"), [(0, False), (2, False), (0, True), (1, True)])
def test_without_a_seed_every_prompt_is_the_plain_one(spider, rows, hints):
    plain = PromptBook(spider, QUESTIONS, rows, hints)
    book = ShuffledBook(spider, QUESTIONS, rows, hints)

    for index, question in enumerate(QUESTIONS):
        assert book.schema(question, index) == plain.schema(question)


def test_with_a_seed_every_table_appears_once_in_another_order(spider, festival):
    plain = PromptBook(spider, QUESTIONS)
    book = ShuffledBook(spider, QUESTIONS, seed=0)

    for index in range(40):
        assert is_reordered(book.schema(PLAIN, index), plain.schema(PLAIN), table_texts(festival))
    assert book.schema(NOTE, 2) == plain.schema(NOTE) == "CREATE TABLE note (body TEXT)"


def test_example_rows_stay_under_their_own_table(spider, festival):
    plain = PromptBook(spider, QUESTIONS, sample_rows=2).schema(PLAIN)
    text = ShuffledBook(spider, QUESTIONS, sample_rows=2, seed=0).schema(PLAIN, 1)

    assert is_reordered(text, plain, table_texts(festival, 2))
    assert "age INTEGER)\n/*\nExample rows:\nsinger_id | name | age\n1 | Ana | 30\n2 | Ben" in text
    assert "REAL)\n/*\nExample rows:\nticket_id | concert_id | price\n7 | 10 | 20.5\n*/" in text
    assert "name TEXT)\n/*\nExample rows:\nband_id | name\n*/" in text


def test_value_hints_stay_at_the_end_as_they_are(spider, festival):
    plain = PromptBook(spider, QUESTIONS, value_hints=True)
    book = ShuffledBook(spider, QUESTIONS, value_hints=True, seed=3)
    named, unnamed = book.schema(NAMED, 0), book.schema(PLAIN, 1)

    assert plain.schema(NAMED).endswith(HINTS)
    assert named.endswith(HINTS)
    assert is_reordered(named, plain.schema(NAMED), [*table_texts(festival), HINTS])
    assert unnamed == ShuffledBook(spider, QUESTIONS, seed=3).schema(PLAIN, 1)
    assert "Values from the question" not in unnamed


def test_each_question_gets_its_own_order_and_the_same_one_every_time(spider):
    first = ShuffledBook(spider, QUESTIONS, seed=0)
    again = ShuffledBook(spider, QUESTIONS, seed=0)
    other = ShuffledBook(spider, QUESTIONS, seed=1)
    texts = [first.schema(PLAIN, index) for index in range(20)]

    assert len(set(texts)) > 10
    assert [again.schema(PLAIN, index) for index in range(20)] == texts
    assert [other.schema(PLAIN, index) for index in range(20)] != texts


def test_tables_that_are_not_the_plain_prompts_are_refused(spider, monkeypatch):
    monkeypatch.setattr(shuffle, "table_texts", lambda *_: ["CREATE TABLE other (x)"])

    with pytest.raises(ValueError, match="festival are not the ones in its prompt"):
        ShuffledBook(spider, QUESTIONS, seed=0)
    assert ShuffledBook(spider, QUESTIONS).schema(NOTE, 2) == "CREATE TABLE note (body TEXT)"

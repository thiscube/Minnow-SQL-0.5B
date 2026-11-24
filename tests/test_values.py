import sqlite3
from contextlib import closing

from minnow.values import MAX_HINTS, Value, hint, mentioned, value_index


def make_database(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE country (Code CHAR(3), Name TEXT, Population INTEGER, Note);
            INSERT INTO country VALUES ('AGO', 'Angola', 100, 'The'), ('USA', 'United States',
                300, NULL), ('NZL', 'New Zealand', 5, '  New Zealand  '), ('X', 'ok', 1, 7);
            """
        )
        connection.commit()
    return path


def test_index_keeps_short_text_values_of_text_columns(tmp_path):
    values = value_index(make_database(tmp_path / "w.sqlite"))

    assert Value("country", "Name", "Angola") in values
    assert Value("country", "Code", "AGO") in values
    assert Value("country", "Note", "New Zealand") in values
    assert not [v for v in values if v.column == "Population"]
    assert not [v for v in values if v.text in ("X", "ok", "7")]


def test_question_finds_whole_word_values_longest_first(tmp_path):
    values = value_index(make_database(tmp_path / "w.sqlite"))

    found = mentioned(values, "What is the population of New Zealand and of Angola?")

    assert [v.text for v in found] == ["New Zealand", "New Zealand", "Angola"]
    assert mentioned(values, "Which countries are Angolan?") == []
    assert mentioned(values, "Name the countries that are bigger than the USA.")[0].text == "USA"


def test_common_words_and_numbers_are_not_hints():
    values = [Value("t", "c", "The"), Value("t", "c", "2010"), Value("t", "c", "show all")]

    assert mentioned(values, "Show all the rows from 2010") == []


def test_hints_are_capped():
    values = [Value("t", "c", f"Bob{n}") for n in range(20)]

    assert len(mentioned(values, " ".join(f"Bob{n}" for n in range(20)))) == MAX_HINTS


def test_hint_text_quotes_values_safely():
    text = hint([Value("person", "name", "O'Neil")])

    assert (
        text == "/*\nValues from the question found in the database:\nperson.name = 'O''Neil'\n*/"
    )
    assert hint([]) == ""

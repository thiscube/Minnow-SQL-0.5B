import pytest

from minnow.schema import connect_read_only, create_statements


def test_lists_tables_in_creation_order_without_views(database):
    text = create_statements(database)

    assert text.index("CREATE TABLE singer") < text.index("CREATE TABLE concert")
    assert "FOREIGN KEY (singer_id)" in text
    assert "VIEW" not in text


def test_example_rows_follow_each_table(database):
    text = create_statements(database, sample_rows=2)

    assert "/*\nExample rows:\nsinger_id | name | age\n1 | Ana | 30\n2 | Ben | 25\n*/" in text
    assert "3 | Cleo | 30" not in text
    assert text.index("1 | Ana | 30") < text.index("CREATE TABLE concert")
    assert "concert_id | singer_id\n10 | 1\n11 | 1" in text


def test_long_and_empty_values_are_shortened(tmp_path):
    import sqlite3

    path = tmp_path / "notes.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE note (body TEXT, extra TEXT)")
    connection.execute("INSERT INTO note VALUES (?, NULL)", ("word " * 20 + "\nend",))
    connection.commit()
    connection.close()

    text = create_statements(path, sample_rows=1)

    assert "word word word word word wo... | NULL" in text


def test_connection_cannot_change_the_database(database):
    connection = connect_read_only(database)
    try:
        with pytest.raises(Exception, match="readonly"):
            connection.execute("DELETE FROM singer")
    finally:
        connection.close()


def test_missing_file_is_an_error_not_a_new_database(tmp_path):
    missing = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError):
        create_statements(missing)
    assert not missing.exists()

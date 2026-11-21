import pytest

from minnow.schema import connect_read_only, create_statements


def test_lists_tables_in_creation_order_without_views(database):
    text = create_statements(database)

    assert text.index("CREATE TABLE singer") < text.index("CREATE TABLE concert")
    assert "FOREIGN KEY (singer_id)" in text
    assert "VIEW" not in text


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

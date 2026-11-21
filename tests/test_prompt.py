import pytest

from minnow.prompt import SYSTEM, build_messages, extract_sql


def test_messages_hold_the_schema_and_the_question():
    messages = build_messages("CREATE TABLE t (a INT)", "How many rows?")

    assert messages[0] == {"role": "system", "content": SYSTEM}
    assert messages[1]["role"] == "user"
    assert "CREATE TABLE t (a INT)" in messages[1]["content"]
    assert messages[1]["content"].endswith("Question: How many rows?")


@pytest.mark.parametrize(
    ("answer", "sql"),
    [
        ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer"),
        ("SELECT count(*) FROM singer;", "SELECT count(*) FROM singer"),
        ("```sql\nSELECT name\nFROM singer;\n```", "SELECT name FROM singer"),
        ("```SQL\nSELECT 1\n```\nThis counts rows.", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("```sql\nSELECT 1", "SELECT 1"),
        ("SELECT 1; SELECT 2;", "SELECT 1"),
        ("  SELECT   name \n FROM  singer  ", "SELECT name FROM singer"),
    ],
)
def test_extracts_one_clean_query(answer, sql):
    assert extract_sql(answer) == sql

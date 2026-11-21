import pytest

from minnow.prompt import SYSTEM, build_messages, chat_prompt, extract_sql


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert (tokenize, add_generation_prompt) == (False, True)
        return "|".join(message["content"] for message in messages) + "|answer:"


def test_chat_prompt_ends_where_the_answer_begins():
    prompt = chat_prompt(FakeTokenizer(), "CREATE TABLE t (a INT)", "How many rows?")

    assert prompt.startswith(SYSTEM)
    assert prompt.endswith("Question: How many rows?|answer:")


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
        ("```sqlite\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("```SELECT 1```", "SELECT 1"),
        ("```sql SELECT 1```", "SELECT 1"),
        ("```sql\nSELECT 1", "SELECT 1"),
        ("SELECT 1; SELECT 2;", "SELECT 1"),
        ("  SELECT   name \n FROM  singer  ", "SELECT name FROM singer"),
        ("Here is the query:\nSELECT 1", "SELECT 1"),
        ("Sure.\n```sql\n-- count them\nSELECT 1\n```", "SELECT 1"),
        ("SELECT 'a;b' FROM t; SELECT 2", "SELECT 'a;b' FROM t"),
        ('SELECT "a;b", `c;d` FROM t;', 'SELECT "a;b", `c;d` FROM t'),
        ("SELECT 'it''s; fine' FROM t;", "SELECT 'it''s; fine' FROM t"),
        ("The answer: (SELECT 1) UNION (SELECT 2)", "(SELECT 1) UNION (SELECT 2)"),
        (
            "WITH old AS (SELECT * FROM singer WHERE age > 29) SELECT count(*) FROM old",
            "WITH old AS (SELECT * FROM singer WHERE age > 29) SELECT count(*) FROM old",
        ),
        ("I am not sure.", "I am not sure."),
    ],
)
def test_extracts_one_clean_query(answer, sql):
    assert extract_sql(answer) == sql

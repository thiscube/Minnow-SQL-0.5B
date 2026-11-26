import json

import pytest

from minnow.mix import is_plain, is_single_turn, joins, main, select, summary
from minnow.spider import SpiderExample, load_split

ONE = "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id"
TWO = ONE + " JOIN stadium AS T3 ON T2.stadium_id = T3.stadium_id"
THREE = TWO + " INNER JOIN city AS T4 ON T3.city_id = T4.city_id"
FOUR = THREE + " join country as T5 on T4.country_id = T5.country_id"


def example(query, db_id="new", question="Who sings where?"):
    return SpiderExample(question, query, db_id)


def write(folder, name, examples):
    rows = [{"question": e.question, "query": e.query, "db_id": e.db_id} for e in examples]
    (folder / name).write_text(json.dumps(rows), encoding="utf-8")


@pytest.mark.parametrize(
    "sql",
    [
        "WITH old AS (SELECT * FROM singer) SELECT * FROM old",
        "SELECT name, rank() OVER (ORDER BY age) FROM singer",
        "SELECT CASE WHEN age > 30 THEN 'old' ELSE 'young' END FROM singer",
        "SELECT * FROM singer LEFT JOIN concert ON singer.singer_id = concert.singer_id",
        "SELECT * FROM singer LEFT OUTER JOIN concert ON singer.singer_id = concert.singer_id",
        "SELECT * FROM singer RIGHT JOIN concert ON singer.singer_id = concert.singer_id",
        "SELECT * FROM singer FULL OUTER JOIN concert ON singer.singer_id = concert.singer_id",
        "SELECT * FROM singer OUTER JOIN concert ON singer.singer_id = concert.singer_id",
        "SELECT * FROM singer CROSS JOIN concert",
        "select * from singer natural left\njoin concert",
        "SELECT strftime('%Y', born) FROM singer",
        "SELECT julianday(born) FROM singer",
        "SELECT date(born) FROM singer",
        "SELECT datetime(born) FROM singer",
        "SELECT time (born) FROM singer",
        "SELECT name FROM singer WHERE born < CURRENT_DATE",
        "SELECT current_time, CURRENT_TIMESTAMP FROM singer",
        "SELECT COALESCE(age, 0) FROM singer",
        "SELECT ifnull(age, 0) FROM singer",
        "SELECT NULLIF(age, 0) FROM singer",
        "SELECT CAST(age AS REAL) FROM singer",
        "SELECT SUBSTR(name, 1, 2) FROM singer",
        "SELECT substring(name, 1, 2) FROM singer",
        "SELECT INSTR(name, 'a') FROM singer",
        "SELECT IIF(age > 30, 1, 0) FROM singer",
    ],
)
def test_sql_that_spider_never_uses_is_not_plain(sql):
    assert not is_plain(sql)


@pytest.mark.parametrize(
    "sql",
    [
        THREE,
        "SELECT count(*), max(age) FROM singer WHERE age IN (SELECT age FROM singer) GROUP BY age",
        "SELECT date, time, cast FROM show WHERE case_id = 1 AND overtime = 0 ORDER BY date",
        "SELECT name FROM singer WHERE note = 'a CASE with it''s date(' AND city = \"LEFT JOIN\"",
        'SELECT "with", `over`, [case], "cast"(x) FROM singer',
    ],
)
def test_keywords_inside_names_and_values_do_not_count(sql):
    assert is_plain(sql)


def test_joins_are_counted_by_keyword_outside_quotes():
    assert joins("SELECT name FROM singer") == 0
    assert joins(ONE) == 1
    assert joins(FOUR) == 4
    assert joins("select * from a join b on a.x = b.x where note = 'join us' or \"join\" = 1") == 1
    assert joins("SELECT * FROM a WHERE x IN (SELECT x FROM b JOIN c ON b.y = c.y)") == 1


def test_dialogues_and_questions_over_several_lines_are_not_single_turn():
    assert is_single_turn("Which users joined in 2020, and who assisted them?")
    assert not is_single_turn("**User**: I want singers.\n**Assistant**: Which ones?")
    assert not is_single_turn("**User**: I want singers. **Assistant**: Which ones?")
    assert not is_single_turn("User: I want singers. Assistant: Which ones?")
    assert not is_single_turn("Sure. Assistant: which singers do you want?")
    assert not is_single_turn("How many singers?\nNote: count each singer once.")
    assert not is_single_turn("How many singers?\r")


def test_seen_databases_need_more_joins_than_unseen_ones():
    examples = [
        example("SELECT name FROM singer", question="none new"),
        example(ONE, question="one new"),
        example(ONE, "old", "one old"),
        example(TWO, "old", "two old"),
        example(THREE, question="three new"),
    ]

    chosen = select(examples, seen={"old"})

    assert [e.question for e in chosen] == ["one new", "two old", "three new"]


def test_minimum_joins_can_be_set_for_each_kind_of_database():
    examples = [
        example("SELECT name FROM singer", question="none new"),
        example(ONE, "old", "one old"),
        example(TWO, question="two new"),
    ]

    assert select(examples, {"old"}, min_joins_unseen=2, min_joins_seen=1) == examples[1:]
    assert select(examples, {"old"}, min_joins_unseen=0, min_joins_seen=2) == examples[::2]


def test_fancy_sql_dialogues_and_repeated_questions_are_left_out():
    examples = [
        example(ONE.replace("JOIN", "LEFT JOIN"), question="fancy"),
        example(ONE, question="**User**: Who sings?\n**Assistant**: Where?"),
        example(TWO),
        example(ONE),
        example(ONE, db_id="other"),
    ]

    assert select(examples, seen=set()) == [examples[2], examples[4]]


def test_question_left_out_once_can_come_in_with_a_query_that_passes():
    examples = [example("SELECT name FROM singer"), example(ONE), example(TWO)]

    assert select(examples, seen=set()) == [examples[1]]


def test_summary_counts_by_kind_of_database_and_by_joins():
    chosen = [example(ONE), example(TWO, "old"), example(THREE, "old"), example(FOUR, "older")]

    assert summary(chosen, seen={"old", "older", "unused"}).splitlines() == [
        "on unseen databases: 1 questions, 1 databases",
        "on seen databases: 3 questions, 2 databases",
        "1 join: 1, 2 joins: 1, 3 or more joins: 2",
    ]


def test_summary_shows_questions_without_a_join_only_when_there_are_some():
    assert summary([example("SELECT 1"), example(ONE)], seen=set()).splitlines()[-1] == (
        "no join: 1, 1 join: 1, 2 joins: 0, 3 or more joins: 0"
    )


def test_command_line_writes_the_selection_in_the_order_of_its_input(tmp_path, capsys):
    write(tmp_path, "a.json", [example(ONE, "old", "one old"), example(TWO, "old", "two old")])
    write(tmp_path, "b.json", [example(ONE, question="one new"), example("SELECT 1")])
    write(tmp_path, "seen.json", [example("SELECT 1", "old", "trained on")])
    options = ["--from", "b.json", "a.json", "--seen", "seen.json", "--out-file", "mix.json"]

    assert main([str(tmp_path), *options]) == 0

    assert load_split(tmp_path, "mix.json") == [
        example(ONE, question="one new"),
        example(TWO, "old", "two old"),
    ]
    printed = capsys.readouterr().out
    assert "2 questions written to" in printed
    assert "on unseen databases: 1 questions, 1 databases" in printed
    assert "on seen databases: 1 questions, 1 databases" in printed
    assert "1 join: 1, 2 joins: 1, 3 or more joins: 0" in printed


def test_command_line_takes_the_minimum_joins(tmp_path):
    write(tmp_path, "a.json", [example(ONE, "old", "one old"), example("SELECT 1")])
    options = ["--from", "a.json", "--seen", "a.json", "--out-file", "mix.json"]

    main([str(tmp_path), *options, "--min-joins-seen", "0"])
    assert len(load_split(tmp_path, "mix.json")) == 2

    main([str(tmp_path), *options, "--min-joins-seen", "1"])
    assert load_split(tmp_path, "mix.json") == [example(ONE, "old", "one old")]


@pytest.mark.parametrize("kind", ["--from", "--seen"])
def test_command_line_refuses_to_overwrite_a_file_it_reads(tmp_path, kind):
    write(tmp_path, "a.json", [example(ONE)])
    write(tmp_path, "b.json", [example(TWO)])
    before = (tmp_path / "b.json").read_text(encoding="utf-8")
    options = ["--from", "a.json", "--seen", "a.json", kind, "b.json", "--out-file", "b.json"]

    with pytest.raises(SystemExit):
        main([str(tmp_path), *options])

    assert (tmp_path / "b.json").read_text(encoding="utf-8") == before

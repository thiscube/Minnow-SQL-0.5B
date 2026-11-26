import json
import sqlite3

import pytest

from minnow.evaluate import append_candidates, append_prediction, load_predictions
from minnow.execution import QueryError, run_query
from minnow.outcomes import load_outcomes
from minnow.repair import (
    load_schema,
    main,
    pool_candidates,
    repair_columns,
    repair_picks,
    sources,
)
from minnow.runfiles import judged_run
from minnow.spider import SpiderExample

SCHEMA = """
CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL,
    placed_on TEXT);
CREATE TABLE city (city TEXT, country TEXT);
CREATE TABLE product (product_id INTEGER PRIMARY KEY, weight_grams INTEGER);
CREATE VIEW big_orders AS SELECT * FROM orders WHERE total > 20;
INSERT INTO customer VALUES (1, 'Ana', 'Lisbon'), (2, 'Ben', 'Oslo');
INSERT INTO orders VALUES (10, 1, 25.0, '2026-01-02'), (11, 2, 5.0, '2026-01-03');
INSERT INTO city VALUES ('Lisbon', 'Portugal'), ('Oslo', 'Norway');
INSERT INTO product VALUES (100, 250);
"""
JOIN = "FROM orders AS T1 JOIN customer AS T2 ON T1.customer_id = T2.customer_id"
MIXED_UP = f"SELECT T1.name {JOIN}"
EMPTY = "SELECT name FROM customer WHERE city = 'Rome'"
NAMES = "SELECT name FROM customer"
CITIES = "SELECT city FROM customer"
COUNT = "SELECT count(*) FROM orders"
BROKEN = "SELEC nonsense"
QUESTIONS = [
    {"question": "Names?", "query": NAMES, "db_id": "shop"},
    {"question": "Who ordered?", "query": f"SELECT T2.name {JOIN}", "db_id": "shop"},
    {"question": "How many orders?", "query": "SELECT count(*) FROM orders", "db_id": "shop"},
]
EXAMPLES = [SpiderExample(q["question"], q["query"], q["db_id"]) for q in QUESTIONS]
SUMMARY = {
    "model": "Some/Model + adapter",
    "split": "dev",
    "correct": 0,
    "total": 3,
    "execution_accuracy": 0.0,
    "generation_seconds": 12,
    "decoding": "vote over 3 candidates",
}


@pytest.fixture
def shop(tmp_path):
    path = tmp_path / "spider" / "database" / "shop" / "shop.sqlite"
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def spider(shop):
    folder = shop.parent.parent.parent
    (folder / "dev.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
    return folder


def write_run(folder, predictions, candidates=None, **summary):
    folder.mkdir(parents=True)
    for index, sql in predictions.items():
        append_prediction(folder / "predictions.jsonl", index, sql)
    for index, sqls in (candidates or {}).items():
        append_candidates(folder / "candidates.jsonl", index, sqls)
    text = json.dumps({**SUMMARY, **summary})
    (folder / "summary.json").write_text(text, encoding="utf-8")
    return folder


def test_schema_lists_the_columns_of_every_table_but_not_of_views(shop):
    assert load_schema(shop) == {
        "customer": ["customer_id", "name", "city"],
        "orders": ["order_id", "customer_id", "total", "placed_on"],
        "city": ["city", "country"],
        "product": ["product_id", "weight_grams"],
    }


def test_sources_are_the_tables_a_query_reads_with_their_aliases(shop):
    sql = (
        'SELECT * FROM "Orders" o JOIN [customer] AS c ON o.customer_id = c.customer_id '
        "JOIN (SELECT 1) AS x JOIN nowhere AS n JOIN city WHERE 1"
    )

    assert sources(sql, load_schema(shop)) == [
        ("o", "orders"),
        ("c", "customer"),
        ("city", "city"),
    ]


def test_table_without_alias_hides_the_table_joined_right_after_it(shop):
    """The measured repair reads JOIN as the alias; changing this needs a new measurement."""
    schema = load_schema(shop)

    assert sources("SELECT * FROM orders JOIN customer", schema) == [("orders", "orders")]
    assert sources("SELECT * FROM orders AS o JOIN customer", schema) == [
        ("o", "orders"),
        ("customer", "customer"),
    ]


@pytest.mark.parametrize(
    "sql",
    [
        NAMES,
        EMPTY,
        "SELECT * FROM nowhere",
        BROKEN,
        "SELECT zzz FROM orders",
        "SELECT weight_grams FROM orders",
        f"SELECT T1.nam {JOIN}",
        "SELECT T1 . totl FROM orders AS T1",
    ],
)
def test_query_that_runs_or_cannot_be_repaired_is_returned_unchanged(shop, sql):
    assert repair_columns(shop, sql) == sql


@pytest.mark.parametrize(
    ("sql", "repaired"),
    [
        (MIXED_UP, f"SELECT T2.name {JOIN}"),
        (f"SELECT t1.NAME, T1.total {JOIN}", f"SELECT T2.NAME, T1.total {JOIN}"),
        (
            "SELECT name FROM orders WHERE total > 20",
            'SELECT name FROM (SELECT * FROM "orders" NATURAL JOIN "customer") AS orders '
            "WHERE total > 20",
        ),
        (
            "SELECT T1.name, T1.total FROM orders AS T1",
            'SELECT T1.name, T1.total FROM (SELECT * FROM "orders" NATURAL JOIN "customer") AS T1',
        ),
        (
            f"SELECT T2.country {JOIN}",
            "SELECT T2.country FROM orders AS T1 JOIN "
            '(SELECT * FROM "customer" NATURAL JOIN "city") AS T2 '
            "ON T1.customer_id = T2.customer_id",
        ),
        (
            f"SELECT country {JOIN}",
            "SELECT country FROM orders AS T1 JOIN "
            '(SELECT * FROM "customer" NATURAL JOIN "city") AS T2 '
            "ON T1.customer_id = T2.customer_id",
        ),
        ("SELECT totl FROM orders", "SELECT total FROM orders"),
        ("SELECT T1.Totl FROM orders AS T1", "SELECT T1.total FROM orders AS T1"),
        ("SELECT totl, placed FROM orders", "SELECT total, placed_on FROM orders"),
        (
            "SELECT order_i, totl, placed FROM orders",
            "SELECT order_id, total, placed_on FROM orders",
        ),
    ],
)
def test_missing_column_is_reached_another_way(shop, sql, repaired):
    with pytest.raises(QueryError, match="no such column"):
        run_query(shop, sql)

    assert repair_columns(shop, sql) == repaired
    assert run_query(shop, repaired)


def test_join_repair_skips_a_longer_table_name_that_starts_the_same(tmp_path):
    path = tmp_path / "pets.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE pet_owner (owner_id INTEGER, pet_id INTEGER);"
        "CREATE TABLE pet (pet_id INTEGER, kind_id INTEGER);"
        "CREATE TABLE kind (kind_id INTEGER, kind_name TEXT);"
        "INSERT INTO pet_owner VALUES (1, 7);"
        "INSERT INTO pet VALUES (7, 3);"
        "INSERT INTO kind VALUES (3, 'cat');"
    )
    connection.commit()
    connection.close()
    sql = "SELECT T2.kind_name FROM pet_owner AS T1 JOIN pet AS T2 ON T1.pet_id = T2.pet_id"

    repaired = repair_columns(path, sql)

    assert repaired == (
        "SELECT T2.kind_name FROM pet_owner AS T1 JOIN "
        '(SELECT * FROM "pet" NATURAL JOIN "kind") AS T2 ON T1.pet_id = T2.pet_id'
    )
    assert run_query(path, repaired)


def test_repair_that_still_does_not_run_gives_the_query_back(shop):
    ambiguous = "SELECT T1.name FROM orders AS T1 JOIN customer AS T2 ON customer_id = 1"
    four_names = "SELECT order_i, customer_i, totl, placed FROM orders"

    assert repair_columns(shop, ambiguous) == ambiguous
    assert repair_columns(shop, four_names) == four_names


def test_picks_are_repaired_on_the_database_of_their_question(spider):
    picks = {1: MIXED_UP, 2: BROKEN}

    assert repair_picks(EXAMPLES, picks, spider) == {1: f"SELECT T2.name {JOIN}", 2: BROKEN}


def test_command_line_votes_again_repairs_and_writes_a_complete_run(tmp_path, spider, capsys):
    candidates = {0: [EMPTY, EMPTY, NAMES], 2: [BROKEN, BROKEN]}
    run = write_run(tmp_path / "run", {0: EMPTY, 1: MIXED_UP, 2: BROKEN}, candidates)
    out = tmp_path / "repaired"

    assert main([str(out), str(run), "--spider", str(spider)]) == 0

    assert load_predictions(out / "predictions.jsonl") == {
        0: NAMES,
        1: f"SELECT T2.name {JOIN}",
        2: BROKEN,
    }
    assert [outcome.verdict for outcome in load_outcomes(out / "outcomes.jsonl")] == [
        "correct",
        "correct",
        "error",
    ]
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == {
        **SUMMARY,
        "correct": 2,
        "execution_accuracy": 66.67,
        "decoding": "vote over 3 candidates; picks voted again with empty results last, "
        "then repaired where a column was missing",
    }
    assert not (out / "candidates.jsonl").exists()
    assert judged_run(out, "repaired", "2026-11-26").correct == 2
    assert "correct" in capsys.readouterr().out


def test_command_line_keeps_the_picks_of_a_run_without_saved_candidates(tmp_path, spider, capsys):
    run = write_run(tmp_path / "run", {0: EMPTY, 1: MIXED_UP})
    out = tmp_path / "repaired"

    assert main([str(out), str(run), "--spider", str(spider)]) == 0

    assert load_predictions(out / "predictions.jsonl") == {0: EMPTY, 1: f"SELECT T2.name {JOIN}"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["correct"], summary["total"]) == (1, 3)
    assert summary["decoding"] == (
        "vote over 3 candidates; picks kept (no saved candidates), "
        "then repaired where a column was missing"
    )
    assert "missing prediction" in capsys.readouterr().out


def test_command_line_refuses_to_overwrite_a_run(tmp_path, spider):
    run = write_run(tmp_path / "run", {0: NAMES})

    with pytest.raises(SystemExit):
        main([str(run), str(run), "--spider", str(spider)])


@pytest.mark.parametrize("missing", ["predictions.jsonl", "summary.json"])
def test_command_line_refuses_a_folder_that_is_not_a_run(tmp_path, spider, capsys, missing):
    run = write_run(tmp_path / "run", {0: NAMES})
    (run / missing).unlink()

    with pytest.raises(SystemExit):
        main([str(tmp_path / "out"), str(run), "--spider", str(spider)])

    assert f"{missing} does not exist" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_command_line_refuses_a_run_that_answered_another_split(tmp_path, spider, capsys):
    run = write_run(tmp_path / "run", {0: NAMES}, split="test_val")

    with pytest.raises(SystemExit):
        main([str(tmp_path / "out"), str(run), "--spider", str(spider)])

    assert "answered test_val, not dev" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def two_runs(tmp_path, **second_summary):
    """Two runs of one model. Question 2 is a tie that only the order of the runs decides."""
    first = write_run(
        tmp_path / "first",
        {0: EMPTY, 1: MIXED_UP, 2: CITIES},
        {0: [EMPTY, NAMES], 1: [MIXED_UP], 2: [CITIES, COUNT, COUNT]},
    )
    second = write_run(
        tmp_path / "second",
        {0: EMPTY, 1: BROKEN, 2: NAMES},
        {0: [EMPTY, EMPTY], 1: [BROKEN, BROKEN], 2: [NAMES, NAMES]},
        **second_summary,
    )
    return str(first), str(second)


def test_candidates_are_pooled_run_after_run_and_not_changed():
    first, second = {0: ["a", "b"], 1: ["c"]}, {0: ["d"], 2: ["e"]}

    assert pool_candidates([first]) == first
    assert pool_candidates([first, second]) == {0: ["a", "b", "d"], 1: ["c"], 2: ["e"]}
    assert first == {0: ["a", "b"], 1: ["c"]}


def test_command_line_pools_several_runs_and_votes_once(tmp_path, spider):
    first, second = two_runs(tmp_path, generation_seconds=30)
    out = tmp_path / "pooled"

    assert main([str(out), first, second, "--spider", str(spider)]) == 0

    # 0: one answer with rows beats three empty ones. 1: nothing runs, so the first run's
    # greedy answer is kept and repaired. 2: two against two, the first run's pair comes first.
    assert load_predictions(out / "predictions.jsonl") == {
        0: NAMES,
        1: f"SELECT T2.name {JOIN}",
        2: COUNT,
    }
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == {
        **SUMMARY,
        "correct": 3,
        "execution_accuracy": 100.0,
        "generation_seconds": 42,
        "decoding": "vote over 3 candidates + vote over 3 candidates; candidates of the 2 runs "
        "pooled and voted once with empty results last, then repaired where a column was missing",
    }
    assert judged_run(out, "pooled", "2026-11-26").correct == 3


def test_tie_goes_to_the_run_named_first_and_unknown_time_stays_unknown(tmp_path, spider):
    first, second = two_runs(tmp_path, generation_seconds=None)
    out = tmp_path / "pooled"

    assert main([str(out), second, first, "--spider", str(spider)]) == 0

    assert load_predictions(out / "predictions.jsonl") == {0: NAMES, 1: BROKEN, 2: NAMES}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["correct"], summary["generation_seconds"]) == (1, None)


@pytest.mark.parametrize(
    ("other", "reason"),
    [
        ({"summary": {"split": "test_val"}}, "second answered test_val, not dev"),
        ({"summary": {"model": "Other/Model"}}, "second is by another model than"),
        ({"candidates": None}, "each of them saved its candidates"),
    ],
)
def test_command_line_refuses_runs_that_cannot_be_pooled(tmp_path, spider, capsys, other, reason):
    first = write_run(tmp_path / "first", {0: NAMES}, {0: [NAMES]})
    candidates = other.get("candidates", {0: [NAMES]})
    second = write_run(tmp_path / "second", {0: NAMES}, candidates, **other.get("summary", {}))

    with pytest.raises(SystemExit):
        main([str(tmp_path / "out"), str(first), str(second), "--spider", str(spider)])

    assert reason in capsys.readouterr().err
    assert not (tmp_path / "out").exists()

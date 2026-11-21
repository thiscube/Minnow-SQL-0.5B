import pytest

from minnow.execution import QueryError, execution_match, results_match, run_query


def test_runs_a_query(database):
    assert run_query(database, "SELECT name FROM singer ORDER BY name") == [
        ("Ana",),
        ("Ben",),
        ("Cleo",),
    ]


def test_broken_query_is_a_query_error(database):
    with pytest.raises(QueryError, match="no such table"):
        run_query(database, "SELECT * FROM nowhere")


def test_slow_query_is_stopped(database):
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT max(i) FROM n"

    with pytest.raises(QueryError, match="interrupted"):
        run_query(database, endless, timeout=0.2)


def test_query_with_too_many_rows_is_a_query_error(database):
    assert len(run_query(database, "SELECT * FROM singer", max_rows=3)) == 3
    with pytest.raises(QueryError, match="more than 2 rows"):
        run_query(database, "SELECT * FROM singer", max_rows=2)


def test_query_cannot_change_the_database(database):
    with pytest.raises(QueryError):
        run_query(database, "DELETE FROM singer")
    assert len(run_query(database, "SELECT * FROM singer")) == 3


def test_same_rows_in_another_order_match_when_order_does_not_matter():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)


def test_duplicates_count():
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)], ordered=False)


def test_columns_in_another_order_match():
    assert results_match([("Ana", 30), ("Ben", 25)], [(30, "Ana"), (25, "Ben")], ordered=True)


def test_different_shapes_do_not_match():
    assert not results_match([(1,)], [(1,), (2,)], ordered=False)
    assert not results_match([(1, 2)], [(1,)], ordered=False)
    assert results_match([], [], ordered=False)


def test_wide_results_are_compared_without_reordering_columns():
    row = tuple(range(7))

    assert results_match([row], [row], ordered=False)
    assert not results_match([tuple(reversed(row))], [row], ordered=False)


def test_correct_prediction_matches(database):
    gold = "SELECT count(*) FROM singer WHERE age = 30"

    assert execution_match(database, "SELECT COUNT(singer_id) FROM singer WHERE age > 29", gold)


def test_wrong_or_broken_prediction_does_not_match(database):
    gold = "SELECT count(*) FROM singer WHERE age = 30"

    assert not execution_match(database, "SELECT count(*) FROM singer", gold)
    assert not execution_match(database, "SELEC nonsense", gold)


def test_order_only_matters_when_the_reference_sorts(database):
    sorted_gold = "SELECT name FROM singer ORDER BY age, name"

    assert not execution_match(database, "SELECT name FROM singer ORDER BY name DESC", sorted_gold)
    assert execution_match(
        database, "SELECT name FROM singer ORDER BY name DESC", "SELECT name FROM singer"
    )

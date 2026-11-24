from minnow.execution import QueryError
from minnow.select import pick

RESULTS = {
    "a": [(1,), (2,)],
    "a again": [(2,), (1,)],
    "b": [(3,)],
    "b again": [(3,)],
    "empty": [],
    "empty again": [],
    "nulls": [(None,)],
    "nulls again": [(None,)],
}


def test_prefer_rows_puts_empty_and_null_results_last():
    many_empty = ["empty", "empty again", "nulls", "nulls again", "b"]

    assert pick(many_empty, run) == "empty"
    assert pick(many_empty, run, prefer_rows=True) == "b"
    assert pick(["broken", "nulls", "empty", "empty again"], run, prefer_rows=True) == "empty"


def run(sql):
    if sql not in RESULTS:
        raise QueryError("no such column")
    return RESULTS[sql]


def test_keeps_the_result_most_candidates_agree_on():
    assert pick(["b", "a", "a again"], run) == "a"


def test_candidates_that_do_not_run_are_ignored():
    assert pick(["broken", "also broken", "b"], run) == "b"


def test_tie_goes_to_the_candidate_generated_first():
    assert pick(["b", "a", "a again", "b again"], run) == "b"
    assert pick(["a", "b"], run) == "a"


def test_nothing_runs_returns_the_first_candidate():
    assert pick(["broken", "also broken"], run) == "broken"


def test_empty_result_counts_as_a_result():
    assert pick(["broken", "empty"], run) == "empty"

import json
import shutil

import pytest

from minnow.evaluate import append_prediction
from minnow.execution import execution_match
from minnow.outcomes import (
    Breakdown,
    Outcome,
    breakdown,
    error_class,
    format_breakdown,
    judge,
    judge_all,
    load_outcomes,
    main,
    save_outcomes,
)
from minnow.spider import SpiderExample

QUESTIONS = [
    {"question": "How many singers?", "query": "SELECT count(*) FROM singer", "db_id": "concert"},
    {"question": "Oldest age?", "query": "SELECT max(age) FROM singer", "db_id": "concert"},
    {"question": "Names?", "query": "SELECT name FROM singer", "db_id": "concert"},
    {"question": "Minors?", "query": "SELECT name FROM singer WHERE age < 18", "db_id": "concert"},
    {"question": "Concerts?", "query": "SELECT count(*) FROM concert", "db_id": "concert"},
]
EXAMPLES = [SpiderExample(item["question"], item["query"], item["db_id"]) for item in QUESTIONS]
PREDICTIONS = {
    0: "SELECT COUNT(name) FROM singer",
    1: "SELECT min(age) FROM singer",
    2: "SELECT nme FROM singer",
    3: "SELECT name FROM singer WHERE age < 10",
}
OUTCOMES = [
    Outcome(0, "correct", "", 1),
    Outcome(1, "wrong_rows", "", 1),
    Outcome(2, "error", "no such column", 3),
    Outcome(3, "correct", "", 0),
    Outcome(4, "error", "missing prediction", 1),
]
# (prediction, reference): right and wrong answers, sorted and unsorted references, columns in
# another order, duplicates, empty results and queries that do not run
PAIRS = [
    ("SELECT COUNT(singer_id) FROM singer WHERE age > 29", "SELECT count(*) FROM singer"),
    (
        "SELECT COUNT(singer_id) FROM singer WHERE age > 29",
        "SELECT count(*) FROM singer WHERE age = 30",
    ),
    ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer WHERE age = 30"),
    ("SELECT name FROM singer ORDER BY name DESC", "SELECT name FROM singer"),
    ("SELECT name FROM singer ORDER BY name DESC", "SELECT name FROM singer ORDER BY age, name"),
    ("SELECT name FROM singer ORDER BY name", "SELECT name FROM singer order  by name"),
    ("SELECT age, name FROM singer", "SELECT name, age FROM singer"),
    ("SELECT name, age FROM singer", "SELECT name FROM singer"),
    ("SELECT DISTINCT age FROM singer", "SELECT age FROM singer"),
    ("SELECT name FROM singer WHERE age > 99", "SELECT name FROM singer WHERE age < 18"),
    ("", "SELECT name FROM singer WHERE age < 18"),
    ("SELECT name FROM singer", "SELECT name FROM singer WHERE age < 18"),
    ("SELECT nme FROM singer", "SELECT name FROM singer"),
    ("SELECT * FROM nowhere", "SELECT name FROM singer"),
    ("SELEC nonsense", "SELECT name FROM singer"),
]


@pytest.fixture
def spider(tmp_path, database):
    folder = tmp_path / "spider"
    (folder / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, folder / "database" / "concert" / "concert.sqlite")
    (folder / "dev.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
    return folder


@pytest.fixture
def run_folder(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    for index, sql in PREDICTIONS.items():
        append_prediction(folder / "predictions.jsonl", index, sql)
    return folder


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("no such column: T1.nme", "no such column"),
        ("no such table: nowhere", "no such table"),
        ("misuse of aggregate: count()", "misuse of aggregate"),
        ("misuse of aggregate function COUNT()", "misuse of aggregate"),
        (
            "Incorrect number of bindings supplied. The current statement uses 1, and there "
            "are 0 supplied.",
            "unfilled placeholder",
        ),
        ("more than 1000000 rows", "too many rows"),
        ('near "SELEC": syntax error', "syntax error"),
        ("interrupted", "timeout"),
        ("no such column: interrupted", "no such column"),
        ("incomplete input", "incomplete input"),
    ],
)
def test_error_message_is_shortened_to_its_kind(message, expected):
    assert error_class(message) == expected


def test_matching_rows_are_correct(database):
    outcome = judge(database, "SELECT COUNT(name) FROM singer", "SELECT count(*) FROM singer")

    assert outcome == Outcome(0, "correct", "", 1)


def test_other_rows_are_wrong_rows_not_an_error(database):
    outcome = judge(database, "SELECT name FROM singer WHERE age = 30", "SELECT name FROM singer")

    assert outcome == Outcome(0, "wrong_rows", "", 3)


def test_row_order_only_matters_when_the_reference_sorts(database):
    backwards = "SELECT name FROM singer ORDER BY name DESC"

    assert judge(database, backwards, "SELECT name FROM singer").verdict == "correct"
    assert judge(database, backwards, "SELECT name FROM singer ORDER BY name").verdict == (
        "wrong_rows"
    )


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("SELECT nme FROM singer", "no such column"),
        ("SELECT * FROM nowhere", "no such table"),
        ("SELEC nonsense", "syntax error"),
        (None, "missing prediction"),
    ],
)
def test_prediction_that_does_not_run_is_an_error_with_its_kind(database, sql, kind):
    outcome = judge(database, sql, "SELECT name FROM singer", index=7)

    assert outcome == Outcome(7, "error", kind, 3)


@pytest.mark.parametrize(("predicted", "gold"), PAIRS)
def test_judge_and_execution_match_agree_on_what_is_correct(database, predicted, gold):
    verdict = judge(database, predicted, gold).verdict

    assert (verdict == "correct") == execution_match(database, predicted, gold)


def test_the_agreement_cases_cover_every_verdict(database):
    verdicts = {judge(database, predicted, gold).verdict for predicted, gold in PAIRS}

    assert verdicts == {"correct", "wrong_rows", "error"}


def test_judges_a_whole_split(spider):
    outcomes = judge_all(EXAMPLES, PREDICTIONS, spider)

    assert outcomes == OUTCOMES
    assert breakdown(outcomes).correct == 2


def test_outcomes_are_saved_sorted_with_unix_line_endings(tmp_path):
    path = tmp_path / "outcomes.jsonl"

    save_outcomes(path, reversed(OUTCOMES))

    lines = path.read_bytes().split(b"\n")
    assert json.loads(lines[0]) == {"index": 0, "verdict": "correct", "error": "", "gold_rows": 1}
    assert [json.loads(line)["index"] for line in lines[:-1]] == [0, 1, 2, 3, 4]
    assert lines[-1] == b""
    assert b"\r" not in path.read_bytes()
    assert load_outcomes(path) == OUTCOMES


def test_outcomes_load_from_a_file_with_windows_line_endings(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    save_outcomes(path, OUTCOMES)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")

    assert load_outcomes(path) == OUTCOMES


def test_breakdown_counts_verdicts_error_kinds_and_empty_references():
    outcomes = [
        *OUTCOMES,
        Outcome(5, "error", "no such table", 2),
        Outcome(6, "error", "no such column", 0),
        Outcome(7, "error", "ambiguous column name", 2),
        Outcome(8, "wrong_rows", "", 0),
    ]

    assert breakdown(outcomes) == Breakdown(
        total=9,
        correct=2,
        wrong_rows=2,
        did_not_run=5,
        error_classes=(
            ("no such column", 2),
            ("ambiguous column name", 1),
            ("missing prediction", 1),
            ("no such table", 1),
        ),
        correct_on_empty=1,
    )


def test_breakdown_of_nothing_is_all_zero():
    assert breakdown([]) == Breakdown(0, 0, 0, 0, (), 0)


def test_breakdown_is_laid_out_as_a_table():
    assert format_breakdown(breakdown(OUTCOMES)).splitlines() == [
        "correct                                     2",
        "ran with different rows                     1",
        "did not run                                 2",
        "  missing prediction                        1",
        "  no such column                            1",
        "total                                       5",
        "correct with an empty reference result      1",
    ]


def test_command_line_saves_the_outcomes_and_prints_the_counts(spider, run_folder, capsys):
    exit_code = main([str(run_folder), "--spider", str(spider)])

    assert exit_code == 0
    assert load_outcomes(run_folder / "outcomes.jsonl") == OUTCOMES
    assert capsys.readouterr().out == format_breakdown(breakdown(OUTCOMES)) + "\n"


def test_command_line_reads_the_split_it_is_given(spider, run_folder, capsys):
    (spider / "small.json").write_text(json.dumps(QUESTIONS[:2]), encoding="utf-8")

    assert main([str(run_folder), "--spider", str(spider), "--split", "small.json"]) == 0
    assert load_outcomes(run_folder / "outcomes.jsonl") == OUTCOMES[:2]
    assert "total" in capsys.readouterr().out


def test_command_line_refuses_a_folder_without_predictions(spider, tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path), "--spider", str(spider)])

    assert raised.value.code == 2
    assert "predictions.jsonl does not exist" in capsys.readouterr().err
    assert not (tmp_path / "outcomes.jsonl").exists()


def test_outcomes_file_with_a_missing_question_is_refused(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    save_outcomes(path, [OUTCOMES[0], OUTCOMES[2]])

    with pytest.raises(ValueError, match="exactly one outcome per question"):
        load_outcomes(path)


def test_outcomes_file_with_a_question_twice_is_refused(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    save_outcomes(path, [OUTCOMES[0], OUTCOMES[0], OUTCOMES[1]])

    with pytest.raises(ValueError, match="exactly one outcome per question"):
        load_outcomes(path)


def test_empty_outcomes_file_is_refused(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one outcome per question"):
        load_outcomes(path)


def test_outcomes_file_with_an_unknown_verdict_is_refused(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    save_outcomes(path, [Outcome(0, "probably fine", "", 1)])

    with pytest.raises(ValueError, match="unknown verdict: probably fine"):
        load_outcomes(path)

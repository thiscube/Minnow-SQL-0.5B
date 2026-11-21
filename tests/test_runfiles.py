import json

import pytest

from minnow.evaluate import append_prediction
from minnow.outcomes import Outcome, save_outcomes
from minnow.runfiles import (
    RUN_FILES,
    RunLogError,
    accuracy,
    fingerprint,
    judged_run,
    read_text,
    verified_counts,
)

OUTCOMES = [
    Outcome(0, "correct", "", 1),
    Outcome(1, "wrong_rows", "", 2),
    Outcome(2, "error", "no such column", 1),
    Outcome(3, "error", "missing prediction", 1),
]


@pytest.fixture
def run_folder(tmp_path):
    """A run with four questions: one correct, one never predicted."""
    folder = tmp_path / "sft"
    folder.mkdir()
    save_outcomes(folder / "outcomes.jsonl", OUTCOMES)
    for index in range(3):
        append_prediction(folder / "predictions.jsonl", index, "SELECT 1")
    summary = {
        "model": "Qwen/Tiny-Instruct",
        "correct": 1,
        "total": 4,
        "execution_accuracy": 25.0,
        "generation_seconds": 7,
    }
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder


def edit_summary(folder, **changes):
    path = folder / "summary.json"
    path.write_text(json.dumps({**json.loads(path.read_text("utf-8")), **changes}), "utf-8")


def test_accuracy_is_a_percentage_with_two_decimals():
    assert accuracy(469, 1034) == 45.36
    assert accuracy(1, 3) == 33.33


def test_missing_file_reads_as_nothing(tmp_path):
    assert read_text(tmp_path / "missing.txt") == ""


def test_fingerprint_ignores_line_endings_but_not_content(run_folder):
    before = fingerprint(run_folder)
    for name in RUN_FILES:
        path = run_folder / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert fingerprint(run_folder) == before

    append_prediction(run_folder / "predictions.jsonl", 3, "SELECT 2")

    assert fingerprint(run_folder) != before


def test_record_is_built_from_the_run_files(run_folder):
    run = judged_run(run_folder, "Fine-tuned  on\nSpider", "2026-12-01")

    assert (run.name, run.date, run.model) == ("sft", "2026-12-01", "Qwen/Tiny-Instruct")
    assert run.change == "Fine-tuned on Spider"
    assert (run.correct, run.total, run.execution_accuracy) == (1, 4, 25.0)
    assert run.generation_seconds == 7
    assert run.fingerprint == fingerprint(run_folder)


def test_unknown_generation_time_is_kept_as_unknown(run_folder):
    edit_summary(run_folder, generation_seconds=None)

    assert judged_run(run_folder, "Fine-tuned", "2026-12-01").generation_seconds is None


def test_predictions_and_outcomes_must_cover_the_same_questions(run_folder):
    append_prediction(run_folder / "predictions.jsonl", 3, "SELECT 1")

    with pytest.raises(RunLogError, match="do not cover the same questions"):
        judged_run(run_folder, "Fine-tuned", "2026-12-01")


def test_outcomes_file_with_a_gap_is_refused(run_folder):
    save_outcomes(run_folder / "outcomes.jsonl", [OUTCOMES[0], OUTCOMES[2]])

    with pytest.raises(RunLogError, match="exactly one outcome per question"):
        judged_run(run_folder, "Fine-tuned", "2026-12-01")


def test_logged_run_with_untouched_files_is_counted(run_folder):
    run = judged_run(run_folder, "Fine-tuned", "2026-12-01")

    counts = verified_counts(run_folder, run)

    assert (counts.correct, counts.wrong_rows, counts.did_not_run, counts.total) == (1, 1, 2, 4)


def test_summary_edited_after_logging_is_refused(run_folder):
    run = judged_run(run_folder, "Fine-tuned", "2026-12-01")
    edit_summary(run_folder, execution_accuracy=55.0)

    with pytest.raises(RunLogError, match=r"The log and summary\.json disagree about sft\."):
        verified_counts(run_folder, run)


def test_predictions_changed_after_logging_are_refused(run_folder):
    run = judged_run(run_folder, "Fine-tuned", "2026-12-01")
    append_prediction(run_folder / "predictions.jsonl", 0, "SELECT 2")

    with pytest.raises(RunLogError, match="The files of sft changed after it was logged"):
        verified_counts(run_folder, run)


def test_outcomes_changed_after_logging_are_refused(run_folder):
    run = judged_run(run_folder, "Fine-tuned", "2026-12-01")
    save_outcomes(run_folder / "outcomes.jsonl", [Outcome(0, "correct", "", 1)] * 0 + OUTCOMES[:3])

    with pytest.raises(RunLogError, match="The log gives sft 1 of 4 correct"):
        verified_counts(run_folder, run)

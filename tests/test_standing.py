import json

import pytest

from minnow.evaluate import append_prediction
from minnow.outcomes import Outcome, save_outcomes
from minnow.runfiles import RunLogError
from minnow.standing import STANDING_FILE, standing_table

TABLE = {
    "columns": [
        {"title": "Spider dev", "folder": "results"},
        {"title": "Scored once", "folder": "heldout/final"},
    ],
    "rows": [
        {"label": "Before training", "runs": ["baseline", None]},
        {"label": "One model", "runs": ["tuned", "tuned"]},
        {"label": "Three models", "runs": ["vote", "vote"], "bold": True},
    ],
}


def make_run(folder, correct, total):
    """A judged run folder with the given number of correct answers."""
    folder.mkdir(parents=True)
    outcomes = [
        Outcome(index, "correct" if index < correct else "wrong_rows", "", 1)
        for index in range(total)
    ]
    save_outcomes(folder / "outcomes.jsonl", outcomes)
    for outcome in outcomes:
        append_prediction(folder / "predictions.jsonl", outcome.index, "SELECT 1")
    summary = {
        "model": "Qwen/Tiny-Instruct",
        "correct": correct,
        "total": total,
        "execution_accuracy": round(100 * correct / total, 2),
        "generation_seconds": None,
    }
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "heldout").mkdir()
    (tmp_path / STANDING_FILE).write_text(json.dumps(TABLE), encoding="utf-8")
    make_run(tmp_path / "results/baseline", 1, 4)
    make_run(tmp_path / "results/tuned", 2, 4)
    make_run(tmp_path / "results/vote", 3, 4)
    make_run(tmp_path / "heldout/final/tuned", 1, 3)
    make_run(tmp_path / "heldout/final/vote", 2, 3)
    return tmp_path


def test_table_reads_every_score_from_the_run_files(repo):
    assert standing_table(repo) == (
        "|  | Spider dev | Scored once |\n"
        "| --- | ---: | ---: |\n"
        "| Before training | 25.00% |  |\n"
        "| One model | 50.00% | 33.33% |\n"
        "| Three models | **75.00%** | **66.67%** |\n"
    )


def test_headline_list_comes_first_when_a_published_score_is_given(repo):
    table = dict(TABLE, published={"name": "Other-0.5B", "score": 60.5})
    (repo / STANDING_FILE).write_text(json.dumps(table), encoding="utf-8")

    assert standing_table(repo).splitlines()[:4] == [
        "- **1.5 times fewer mistakes.** It now gets 50 of every 100 questions wrong. Before "
        "training it got 75 wrong, more than half.",
        "- **2.0 times as many right.** It answers 2 of the 4 questions correctly, up from 1.",
        "- **33% on questions it had never seen.** Other-0.5B, a published model of the same "
        "size, reports 60.5%.",
        "",
    ]


def test_halved_mistakes_are_called_half(repo):
    table = {
        "published": {"name": "Other", "score": 1},
        "columns": [{"title": "Five", "folder": "five"}],
        "rows": [{"label": "a", "runs": ["base"]}, {"label": "b", "runs": ["good"]}],
    }
    (repo / STANDING_FILE).write_text(json.dumps(table), encoding="utf-8")
    make_run(repo / "five/base", 1, 5)
    make_run(repo / "five/good", 3, 5)

    assert standing_table(repo).startswith(
        "- **Half the mistakes.** It now gets 40 of every 100 questions wrong. Before training "
        "it got 80 wrong, more than half."
    )


def test_runs_of_one_column_have_to_cover_the_same_questions(repo):
    make_run(repo / "heldout/final/other", 1, 5)
    table = json.loads((repo / STANDING_FILE).read_text(encoding="utf-8"))
    table["rows"][0]["runs"][1] = "other"
    (repo / STANDING_FILE).write_text(json.dumps(table), encoding="utf-8")

    with pytest.raises(RunLogError, match=r"heldout/final named in heldout/standing\.json"):
        standing_table(repo)


def test_run_that_is_named_but_missing_is_refused(repo):
    (repo / "heldout/final/vote/summary.json").unlink()

    with pytest.raises(RunLogError, match=r"heldout/final/vote/summary\.json does not exist"):
        standing_table(repo)


def test_summary_that_disagrees_with_the_verdicts_is_refused(repo):
    path = repo / "heldout/final/vote/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**summary, "correct": 3}), encoding="utf-8")

    with pytest.raises(RunLogError, match=r"summary\.json says 3 of 3 correct"):
        standing_table(repo)

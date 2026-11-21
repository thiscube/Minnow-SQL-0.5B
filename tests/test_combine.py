import json
import shutil

import pytest

from minnow.combine import combine, main
from minnow.evaluate import append_prediction, load_predictions
from minnow.spider import SpiderExample

QUESTIONS = [
    {"question": "How many singers?", "query": "SELECT count(*) FROM singer", "db_id": "concert"},
    {"question": "Oldest age?", "query": "SELECT max(age) FROM singer", "db_id": "concert"},
    {"question": "Names?", "query": "SELECT name FROM singer", "db_id": "concert"},
]
EXAMPLES = [SpiderExample(q["question"], q["query"], q["db_id"]) for q in QUESTIONS]
RIGHT = {0: "SELECT count(*) FROM singer", 1: "SELECT max(age) FROM singer"}
ALSO_RIGHT = {0: "SELECT COUNT(name) FROM singer", 1: "SELECT 30"}
WRONG = {0: "SELECT 99", 1: "SELECT nothing FROM singer", 2: "SELECT 1"}


@pytest.fixture
def spider(tmp_path, database):
    folder = tmp_path / "spider"
    (folder / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, folder / "database" / "concert" / "concert.sqlite")
    (folder / "dev.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
    return folder


def write_run(folder, predictions):
    folder.mkdir(parents=True)
    for index, sql in predictions.items():
        append_prediction(folder / "predictions.jsonl", index, sql)
    return folder


def test_keeps_what_most_runs_agree_on(spider):
    combined = combine(EXAMPLES, [WRONG, RIGHT, ALSO_RIGHT], spider)

    assert combined[0] == RIGHT[0]
    assert combined[1] == RIGHT[1]


def test_tie_goes_to_the_run_named_first_and_gaps_are_skipped(spider):
    assert combine(EXAMPLES, [WRONG, RIGHT], spider) == {0: "SELECT 99", 1: RIGHT[1], 2: "SELECT 1"}
    assert combine(EXAMPLES, [RIGHT], spider) == RIGHT


def test_command_line_writes_a_complete_run(tmp_path, spider, capsys):
    runs = [
        write_run(tmp_path / name, p) for name, p in (("a", RIGHT), ("b", ALSO_RIGHT), ("c", WRONG))
    ]
    out = tmp_path / "combined"

    assert main([str(out), *map(str, runs), "--spider", str(spider)]) == 0

    assert load_predictions(out / "predictions.jsonl") == {**RIGHT, 2: "SELECT 1"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["model"], summary["correct"], summary["total"]) == ("vote of a, b, c", 2, 3)
    assert summary["execution_accuracy"] == 66.67
    assert summary["generation_seconds"] is None
    assert (out / "outcomes.jsonl").exists()
    assert "correct" in capsys.readouterr().out


def test_command_line_refuses_to_overwrite_a_run(tmp_path, spider):
    run = write_run(tmp_path / "a", RIGHT)

    with pytest.raises(SystemExit):
        main([str(run), str(run), "--spider", str(spider)])

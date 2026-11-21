import shutil

import pytest

from minnow.evaluate import (
    Score,
    append_prediction,
    claim_run_folder,
    drop_unfinished_line,
    load_predictions,
    score,
)
from minnow.spider import SpiderExample


def test_predictions_survive_a_restart(tmp_path):
    path = tmp_path / "predictions.jsonl"

    assert load_predictions(path) == {}
    append_prediction(path, 0, "SELECT 1")
    append_prediction(path, 1, 'SELECT "a;b"')

    assert load_predictions(path) == {0: "SELECT 1", 1: 'SELECT "a;b"'}


def test_line_cut_off_by_a_stopped_run_is_dropped_and_redone(tmp_path):
    path = tmp_path / "predictions.jsonl"
    drop_unfinished_line(path)
    append_prediction(path, 0, "SELECT 1")
    with path.open("a", encoding="utf-8") as file:
        file.write('{"index": 1, "sq')

    drop_unfinished_line(path)
    append_prediction(path, 1, "SELECT 2")

    assert load_predictions(path) == {0: "SELECT 1", 1: "SELECT 2"}


def test_complete_file_is_left_alone(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, 0, "SELECT 1")
    before = path.read_bytes()

    drop_unfinished_line(path)

    assert path.read_bytes() == before


def test_run_folder_accepts_the_same_settings_again(tmp_path):
    folder = tmp_path / "run"
    settings = {"model": "a", "max_new_tokens": 256}

    claim_run_folder(folder, settings)
    claim_run_folder(folder, settings)

    assert (folder / "settings.json").exists()


def test_run_folder_refuses_other_settings(tmp_path):
    folder = tmp_path / "run"
    claim_run_folder(folder, {"model": "a", "max_new_tokens": 256})

    with pytest.raises(ValueError, match="other settings"):
        claim_run_folder(folder, {"model": "b", "max_new_tokens": 256})


def test_scores_by_execution_and_counts_missing_predictions_as_wrong(tmp_path, database):
    spider = tmp_path / "spider"
    (spider / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, spider / "database" / "concert" / "concert.sqlite")
    examples = [
        SpiderExample("How many singers?", "SELECT count(*) FROM singer", "concert"),
        SpiderExample("Oldest age?", "SELECT max(age) FROM singer", "concert"),
        SpiderExample("Names?", "SELECT name FROM singer", "concert"),
        SpiderExample("Concerts?", "SELECT count(*) FROM concert", "concert"),
    ]
    predictions = {
        0: "SELECT COUNT(name) FROM singer",
        1: "SELECT min(age) FROM singer",
        2: "SELECT nme FROM singer",
    }

    result = score(examples, predictions, spider)

    assert result == Score(correct=1, total=4)
    assert result.accuracy == 25.0

import pytest

from minnow.evaluate import (
    add_generation_seconds,
    append_prediction,
    claim_run_folder,
    drop_unfinished_line,
    load_predictions,
)


def test_predictions_survive_a_restart(tmp_path):
    path = tmp_path / "predictions.jsonl"

    assert load_predictions(path) == {}
    append_prediction(path, 0, "SELECT 1")
    append_prediction(path, 1, 'SELECT "a;b"')

    assert load_predictions(path) == {0: "SELECT 1", 1: 'SELECT "a;b"'}
    assert b"\r" not in path.read_bytes()


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


def test_generation_time_adds_up_across_a_stopped_and_resumed_run(tmp_path):
    assert add_generation_seconds(tmp_path, 100.4) == 100
    assert add_generation_seconds(tmp_path, 190.3) == 291
    assert add_generation_seconds(tmp_path, 0.0) == 291


def test_generation_time_is_unknown_when_nothing_was_ever_timed(tmp_path):
    assert add_generation_seconds(tmp_path, 0.0) is None
    assert not (tmp_path / "timing.json").exists()


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

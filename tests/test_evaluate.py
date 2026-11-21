import shutil

from minnow.evaluate import Score, append_prediction, load_predictions, score
from minnow.spider import SpiderExample


def test_predictions_survive_a_restart(tmp_path):
    path = tmp_path / "predictions.jsonl"

    assert load_predictions(path) == {}
    append_prediction(path, 0, "SELECT 1")
    append_prediction(path, 1, 'SELECT "a;b"')

    assert load_predictions(path) == {0: "SELECT 1", 1: 'SELECT "a;b"'}


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

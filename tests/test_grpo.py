import shutil

import pytest

from minnow.grpo import Progress, advantages, answer_length, group_rewards, training_order
from minnow.spider import SpiderExample

EXAMPLE = SpiderExample("How many singers?", "SELECT count(*) FROM singer", "concert")


@pytest.fixture
def spider(tmp_path, database):
    folder = tmp_path / "spider"
    (folder / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, folder / "database" / "concert" / "concert.sqlite")
    return folder


def test_reward_prefers_right_over_running_over_broken(spider):
    sqls = ["SELECT COUNT(name) FROM singer", "SELECT 99", "SELECT nothing FROM singer"]

    assert group_rewards(spider, EXAMPLE, sqls) == [1.0, 0.1, 0.0]


def test_ordered_reference_needs_the_same_order(spider):
    example = SpiderExample("Ages?", "SELECT age FROM singer ORDER BY age", "concert")
    sqls = ["SELECT age FROM singer ORDER BY age", "SELECT age FROM singer ORDER BY age DESC"]

    assert group_rewards(spider, example, sqls) == [1.0, 0.1]


def test_broken_reference_gives_no_rewards(spider):
    broken = SpiderExample("?", "SELECT x FROM missing_table", "concert")

    assert group_rewards(spider, broken, ["SELECT 1"]) is None


def test_advantages_center_the_group_and_skip_equal_groups():
    assert advantages([1.0, 1.0, 1.0]) is None
    values = advantages([1.0, 0.0, 0.0, 0.0])

    assert values[0] > 0 > values[1]
    assert sum(values) == pytest.approx(0)
    assert values[1] == values[2] == values[3]


def test_answer_ends_at_its_first_stop_token():
    assert answer_length([5, 6, 9, 9, 9], {9}) == 3
    assert answer_length([5, 6, 7], {9}) == 3


def test_order_is_a_fixed_shuffle():
    assert training_order(50, seed=0) == training_order(50, seed=0)
    assert sorted(training_order(50, seed=0)) == list(range(50))
    assert training_order(50, seed=0) != list(range(50))


def test_progress_survives_a_restart(tmp_path):
    assert Progress.load(tmp_path) == Progress()
    progress = Progress()
    progress.record(0.5, learned=True)
    progress.record(None, learned=False)
    progress.record(1.0, learned=False)
    progress.updates = 1
    progress.save(tmp_path)

    loaded = Progress.load(tmp_path)

    assert loaded == progress
    assert loaded.position == 3
    assert loaded.recent_reward() == 0.75
    assert loaded.report(10, 3.25, 5.0) == (
        "3/10 reward 0.750 updates 1 3.2 s/question peak 5.00 GB"
    )
    assert Progress().recent_reward() == 0.0

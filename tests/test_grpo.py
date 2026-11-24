import shutil

import pytest

from minnow.grpo import Progress, advantages, reward, training_order
from minnow.spider import SpiderExample

EXAMPLE = SpiderExample("How many singers?", "SELECT count(*) FROM singer", "concert")


@pytest.fixture
def spider(tmp_path, database):
    folder = tmp_path / "spider"
    (folder / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, folder / "database" / "concert" / "concert.sqlite")
    return folder


def test_reward_prefers_right_over_running_over_broken(spider):
    assert reward(spider, EXAMPLE, "SELECT COUNT(name) FROM singer") == 1.0
    assert reward(spider, EXAMPLE, "SELECT 99") == 0.1
    assert reward(spider, EXAMPLE, "SELECT nothing FROM singer") == 0.0


def test_advantages_center_the_group_and_skip_equal_groups():
    assert advantages([1.0, 1.0, 1.0]) is None
    values = advantages([1.0, 0.0, 0.0, 0.0])

    assert values[0] > 0 > values[1]
    assert sum(values) == pytest.approx(0)
    assert values[1] == values[2] == values[3]


def test_order_is_a_fixed_shuffle():
    assert training_order(50, seed=0) == training_order(50, seed=0)
    assert sorted(training_order(50, seed=0)) == list(range(50))
    assert training_order(50, seed=0) != list(range(50))


def test_progress_survives_a_restart(tmp_path):
    assert Progress.load(tmp_path) == Progress()
    progress = Progress()
    progress.record(0.5, learned=True)
    progress.record(1.0, learned=False)
    progress.updates = 1
    progress.save(tmp_path)

    loaded = Progress.load(tmp_path)

    assert loaded == progress
    assert loaded.position == 2
    assert loaded.recent_reward() == 0.75
    assert Progress().recent_reward() == 0.0

"""The parts of reinforcement learning (GRPO) that need no GPU.

For each training question the model writes a group of candidate queries. Each one is run and
scored against the reference query, and a candidate's advantage is how much better it did than
the group's average. The model is then pushed towards the candidates with a positive advantage.
A group where every candidate scored the same teaches nothing and is skipped.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, pstdev

from minnow.outcomes import CORRECT, WRONG_ROWS, judge
from minnow.spider import SpiderExample, database_path

REWARDS = {CORRECT: 1.0, WRONG_ROWS: 0.1}
NO_RUN_REWARD = 0.0
PROGRESS_FILE = "progress.json"


def reward(spider: Path, example: SpiderExample, sql: str) -> float:
    """Score one candidate: 1 when it returns the reference rows, 0.1 when it runs, else 0."""
    outcome = judge(database_path(spider, example.db_id), sql, example.query)
    return REWARDS.get(outcome.verdict, NO_RUN_REWARD)


def advantages(rewards: list[float]) -> list[float] | None:
    """Return each candidate's reward relative to its group, or None when all are equal."""
    spread = pstdev(rewards)
    if spread == 0:
        return None
    mean = fmean(rewards)
    return [(value - mean) / spread for value in rewards]


def training_order(count: int, seed: int) -> list[int]:
    """Return the question positions in a shuffled order that is the same on every run."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


@dataclass
class Progress:
    """How far a run got, so a stopped run continues with the next question."""

    position: int = 0
    updates: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def record(self, mean_reward: float, learned: bool) -> None:
        """Count one question as done and remember its group's mean reward."""
        self.position += 1
        self.history.append({"position": self.position, "reward": mean_reward, "learned": learned})

    def recent_reward(self, last: int = 200) -> float:
        """Return the mean group reward over the last questions, a sign of learning."""
        window = self.history[-last:]
        return fmean(item["reward"] for item in window) if window else 0.0

    def save(self, folder: Path) -> None:
        """Write the progress next to the adapter it belongs to."""
        text = json.dumps(asdict(self), indent=1) + "\n"
        (folder / PROGRESS_FILE).write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, folder: Path) -> "Progress":
        """Read saved progress, or start fresh when there is none."""
        path = folder / PROGRESS_FILE
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

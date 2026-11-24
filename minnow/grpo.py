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

from minnow.execution import QueryError, Row, reference_is_ordered, results_match, run_query
from minnow.spider import SpiderExample, database_path

CORRECT_REWARD = 1.0
RUNS_REWARD = 0.1
NO_RUN_REWARD = 0.0
SCORING_TIMEOUT = 3.0
PROGRESS_FILE = "progress.json"
MAX_PROMPT_TOKENS = 900
MAX_NEW_TOKENS = 200
SAVE_EVERY = 25
# Plain sampling at temperature 1, overriding the model's saved top_p, top_k and repetition
# penalty, so the answers come from the same distribution the log-probabilities are taken from.
SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": MAX_NEW_TOKENS,
}


def answer_length(tokens: list[int], stops: set[int]) -> int:
    """Return how many generated tokens belong to the answer: up to and including its end."""
    for position, token in enumerate(tokens):
        if token in stops:
            return position + 1
    return len(tokens)


def _score(database: Path, sql: str, gold: list[Row], ordered: bool) -> float:
    """Score one candidate against reference rows that were already fetched."""
    try:
        rows = run_query(database, sql, SCORING_TIMEOUT)
    except QueryError:
        return NO_RUN_REWARD
    return CORRECT_REWARD if results_match(rows, gold, ordered) else RUNS_REWARD


def group_rewards(spider: Path, example: SpiderExample, sqls: list[str]) -> list[float] | None:
    """Score each candidate: 1 when it returns the reference rows, 0.1 when it runs, else 0.

    The reference query is run once. A few training references are broken and do not run;
    for those None is returned and the question teaches nothing.
    """
    database = database_path(spider, example.db_id)
    try:
        gold = run_query(database, example.query)
    except QueryError:
        return None
    ordered = reference_is_ordered(example.query)
    return [_score(database, sql, gold, ordered) for sql in sqls]


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

    def record(self, mean_reward: float | None, learned: bool) -> None:
        """Count one question as done and remember its group's mean reward (None if skipped)."""
        self.position += 1
        self.history.append({"position": self.position, "reward": mean_reward, "learned": learned})

    def recent_reward(self, last: int = 200) -> float:
        """Return the mean group reward over the last scored questions, a sign of learning."""
        window = [item["reward"] for item in self.history[-last:] if item["reward"] is not None]
        return fmean(window) if window else 0.0

    def report(self, total: int, seconds_each: float, peak_gb: float) -> str:
        """Return one progress line: where the run is, the recent reward, speed and memory."""
        return (
            f"{self.position}/{total} reward {self.recent_reward():.3f} updates {self.updates} "
            f"{seconds_each:.1f} s/question peak {peak_gb:.2f} GB"
        )

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

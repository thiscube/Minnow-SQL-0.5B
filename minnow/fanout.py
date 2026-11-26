"""Plan how one encoded prompt serves several sampled answers without overflowing memory.

Encoding the prompt is most of the cost of an answer. With the prompt encoded once, its cache is
copied for every sampled answer, and the copies are decoded in as few passes as the budget allows.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

# The model takes 1 GB and choosing the next tokens of 224 answers up to 1.5 GB more, so the
# fullest block stays under 5 GB of an 8 GB card.
CACHE_GIGABYTES = 2.0


@dataclass(frozen=True)
class Fanout:
    """How many answers to write per prompt from one encoding, and how to write them."""

    copies: int
    max_new_tokens: int
    temperature: float | None
    seed: int
    budget_bytes: int = int(CACHE_GIGABYTES * 1e9)

    def passes(self, prompts: int, prompt_tokens: int, token_bytes: int) -> list[tuple[int, int]]:
        """Return (seed, copies of each prompt) for every decoding pass, sized evenly.

        A pass caches prompts x copies sequences, each as long as the prompt plus the longest
        allowed answer, and that has to fit the budget. One copy always goes through, however
        long the prompt is. The seed of a pass is the plan's seed plus the number of its first
        copy, counted from 1.
        """
        if self.copies < 1:
            return []
        one_copy = prompts * (prompt_tokens + self.max_new_tokens) * token_bytes
        count = math.ceil(self.copies / max(1, self.budget_bytes // one_copy))
        sizes = [self.copies // count + (index < self.copies % count) for index in range(count)]
        return [(self.seed + 1 + sum(sizes[:index]), size) for index, size in enumerate(sizes)]


def by_round(answers: Sequence[str], copies: int) -> list[list[str]]:
    """Regroup answers listed prompt by prompt into rounds with one answer per prompt each."""
    return [list(answers[offset::copies]) for offset in range(copies)]

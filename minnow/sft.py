"""Prepare Spider training questions for supervised fine-tuning.

Nothing here needs a GPU. The tokenizer is passed in, so the logic can be tested with a tiny
stand-in.
"""

from pathlib import Path
from typing import Any

from minnow.guard import Example, assert_no_leak, normalize
from minnow.prompt import build_messages
from minnow.spider import SpiderExample, load_split

IGNORED = -100
TRAIN_FILE = "train_spider.json"
DEV_FILE = "dev.json"


def for_guard(examples: list[SpiderExample]) -> list[Example]:
    """Return the questions and database ids the leak guard checks."""
    return [Example(example.question, example.db_id) for example in examples]


def checked_training_examples(
    spider: Path, files: tuple[str, ...] = (TRAIN_FILE,)
) -> list[SpiderExample]:
    """Load Spider's training questions without the ones that are worded like a dev question.

    A few generic questions, such as "How many employees are there?", appear in both sets on
    different databases. They are left out. A training example on a dev database is never
    acceptable and still raises LeakError.
    """
    dev = for_guard(load_split(spider, DEV_FILE))
    dev_questions = {normalize(example.question) for example in dev}
    train = [example for name in files for example in load_split(spider, name)]
    kept = [example for example in train if normalize(example.question) not in dev_questions]
    assert_no_leak(for_guard(kept), dev)
    return kept


def encode(
    tokenizer: Any,  # noqa: ANN401
    schema: str,
    example: SpiderExample,
    max_tokens: int,
) -> dict[str, list[int]] | None:
    """Turn one example into token ids, with the loss counted on the SQL answer only.

    Returns None when the example is longer than max_tokens, because cutting it would cut
    the question or the answer.
    """
    messages = build_messages(schema, example.question)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answered = [*messages, {"role": "assistant", "content": example.query}]
    whole = tokenizer.apply_chat_template(answered, tokenize=False)
    if not whole.startswith(prompt):
        raise ValueError("The chat template does not put the answer right after the prompt.")
    prompt_ids = tokenizer(prompt)["input_ids"]
    ids = tokenizer(whole)["input_ids"]
    if ids[: len(prompt_ids)] != prompt_ids or len(ids) == len(prompt_ids):
        raise ValueError("The answer's tokens do not follow the prompt's tokens.")
    if len(ids) > max_tokens:
        return None
    labels = [IGNORED] * len(prompt_ids) + ids[len(prompt_ids) :]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

import json

import pytest

from minnow.guard import LeakError
from minnow.sft import IGNORED, checked_training_examples, encode
from minnow.spider import SpiderExample

EXAMPLE = SpiderExample("How many singers?", "SELECT count(*) FROM singer", "concert")


class CharTokenizer:
    """A stand-in tokenizer: one token per character, with a simple chat layout."""

    def __init__(self, answer_first=False):
        self.answer_first = answer_first

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
        assert tokenize is False
        turns = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if self.answer_first:
            turns.reverse()
        return "".join(turns) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text):
        return {"input_ids": [ord(char) for char in text]}


def decode(ids):
    return "".join(chr(i) for i in ids)


def write_split(folder, name, items):
    rows = [{"question": q, "query": "SELECT 1", "db_id": db} for q, db in items]
    (folder / name).write_text(json.dumps(rows), encoding="utf-8")


def test_loss_is_counted_on_the_answer_only():
    encoded = encode(CharTokenizer(), "CREATE TABLE singer (id INT)", EXAMPLE, max_tokens=10_000)

    ids, labels = encoded["input_ids"], encoded["labels"]
    assert decode(ids).endswith("<assistant>SELECT count(*) FROM singer</assistant>")
    assert "CREATE TABLE singer (id INT)" in decode(ids)
    assert len(ids) == len(labels) == len(encoded["attention_mask"])
    learned = [label for label in labels if label != IGNORED]
    assert decode(learned) == "SELECT count(*) FROM singer</assistant>"
    assert labels[: len(labels) - len(learned)] == [IGNORED] * (len(labels) - len(learned))


def test_example_that_is_too_long_is_left_out_not_cut():
    whole = encode(CharTokenizer(), "schema", EXAMPLE, max_tokens=10_000)

    assert encode(CharTokenizer(), "schema", EXAMPLE, max_tokens=len(whole["input_ids"])) == whole
    assert (
        encode(CharTokenizer(), "schema", EXAMPLE, max_tokens=len(whole["input_ids"]) - 1) is None
    )


def test_template_that_does_not_end_with_the_answer_is_refused():
    with pytest.raises(ValueError, match="right after the prompt"):
        encode(CharTokenizer(answer_first=True), "schema", EXAMPLE, max_tokens=10_000)


def test_training_questions_load_when_nothing_leaks(tmp_path):
    write_split(tmp_path, "train_spider.json", [("How many trees?", "orchard")])
    write_split(tmp_path, "dev.json", [("How many loaves?", "bakery")])

    assert checked_training_examples(tmp_path) == [
        SpiderExample("How many trees?", "SELECT 1", "orchard")
    ]


def test_training_question_worded_like_a_dev_question_is_left_out(tmp_path):
    train = [("how many LOAVES", "orchard"), ("How many trees?", "orchard")]
    write_split(tmp_path, "train_spider.json", train)
    write_split(tmp_path, "dev.json", [("How many loaves?", "bakery")])

    assert checked_training_examples(tmp_path) == [
        SpiderExample("How many trees?", "SELECT 1", "orchard")
    ]


def test_training_on_a_dev_database_is_refused(tmp_path):
    write_split(tmp_path, "train_spider.json", [("Who bakes the most?", "bakery")])
    write_split(tmp_path, "dev.json", [("How many loaves?", "bakery")])

    with pytest.raises(LeakError):
        checked_training_examples(tmp_path)


def test_answer_that_adds_no_tokens_is_refused():
    empty = SpiderExample("How many singers?", "", "concert")
    tokenizer = CharTokenizer()
    tokenizer.apply_chat_template = lambda *_args, **_options: "same"

    with pytest.raises(ValueError, match="do not follow the prompt"):
        encode(tokenizer, "schema", empty, max_tokens=10_000)

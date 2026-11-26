import json
import shutil

import pytest

from minnow import sft
from minnow.guard import LeakError
from minnow.prompts import PromptBook
from minnow.sft import (
    IGNORED,
    checked_training_examples,
    encode,
    pack,
    packed_update,
    padded,
    run_settings,
    training_set,
    update_shape,
    write_training_log,
)
from minnow.spider import SpiderExample

EXAMPLE = SpiderExample("How many singers?", "SELECT count(*) FROM singer", "concert")
NAMED = SpiderExample("How old is Ana?", "SELECT age FROM singer WHERE name = 'Ana'", "concert")


class CharTokenizer:
    """A stand-in tokenizer: one token per character, with a simple chat layout."""

    name_or_path = "chars"

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


@pytest.fixture
def spider(tmp_path, database):
    folder = tmp_path / "spider"
    (folder / "database" / "concert").mkdir(parents=True)
    shutil.copy(database, folder / "database" / "concert" / "concert.sqlite")
    return folder


def item(length):
    ids = list(range(1, length + 1))
    return {"input_ids": ids, "attention_mask": [1] * length, "labels": [IGNORED, *ids[1:]]}


def test_training_set_encodes_every_example_that_fits_in_order(spider):
    tokenizer = CharTokenizer()
    book = PromptBook(spider, [NAMED, EXAMPLE])

    whole = training_set(tokenizer, spider, [NAMED, EXAMPLE], max_tokens=10_000)

    assert whole == [encode(tokenizer, book.schema(e), e, 10_000) for e in (NAMED, EXAMPLE)]
    shortest = min(len(encoded["input_ids"]) for encoded in whole)
    assert training_set(tokenizer, spider, [NAMED, EXAMPLE], max_tokens=shortest) == whole[1:]


def test_training_set_with_hints_names_the_column_of_a_mentioned_value(spider):
    plain = training_set(CharTokenizer(), spider, [NAMED], 10_000)
    hinted = training_set(CharTokenizer(), spider, [NAMED], 10_000, value_hints=True)

    assert "singer.name = 'Ana'" in decode(hinted[0]["input_ids"])
    assert "singer.name = 'Ana'" not in decode(plain[0]["input_ids"])


def test_saved_training_set_is_read_back_without_opening_the_databases(spider, tmp_path):
    cache = tmp_path / "cache"
    first = training_set(CharTokenizer(), spider, [NAMED, EXAMPLE], 10_000, cache=cache)
    shutil.rmtree(spider)

    again = training_set(CharTokenizer(), spider, [NAMED, EXAMPLE], 10_000, cache=cache)

    assert again == first
    assert len(first) == 2
    assert [path.suffix for path in cache.iterdir()] == [".json"]


def test_saved_training_set_is_not_reused_for_other_examples_or_settings(spider, tmp_path):
    cache = tmp_path / "cache"
    other_model = CharTokenizer()
    other_model.name_or_path = "other"

    training_set(CharTokenizer(), spider, [EXAMPLE], 10_000, cache=cache)
    training_set(CharTokenizer(), spider, [EXAMPLE, NAMED], 10_000, cache=cache)
    training_set(CharTokenizer(), spider, [EXAMPLE], 9_000, cache=cache)
    training_set(CharTokenizer(), spider, [EXAMPLE], 10_000, sample_rows=1, cache=cache)
    training_set(CharTokenizer(), spider, [EXAMPLE], 10_000, value_hints=True, cache=cache)
    training_set(other_model, spider, [EXAMPLE], 10_000, cache=cache)
    shutil.copytree(spider, tmp_path / "other")
    training_set(CharTokenizer(), tmp_path / "other", [EXAMPLE], 10_000, cache=cache)

    assert len(list(cache.iterdir())) == 7


def test_saved_training_set_is_not_reused_after_the_prompt_code_changes(
    spider, tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    training_set(CharTokenizer(), spider, [EXAMPLE], 10_000, cache=cache)

    monkeypatch.setattr(sft, "_PROMPT_CODE", ("prompt.py",))
    training_set(CharTokenizer(), spider, [EXAMPLE], 10_000, cache=cache)

    assert len(list(cache.iterdir())) == 2


def test_pack_starts_with_the_longest_and_fills_each_batch_up_to_the_budget():
    assert pack([100, 400, 200, 300, 50], budget=600) == [[1], [3, 2], [0, 4]]


def test_pack_gives_an_example_over_the_budget_a_batch_of_its_own():
    assert pack([900, 10], budget=500) == [[0], [1]]


def test_pack_with_a_tiny_budget_keeps_one_example_per_batch():
    assert pack([3, 9, 5], budget=1) == [[1], [2], [0]]


def test_padding_is_on_the_right_masked_and_never_learned():
    assert padded([item(3), item(1)], pad_id=0, wrap=list) == {
        "input_ids": [[1, 2, 3], [1, 0, 0]],
        "attention_mask": [[1, 1, 1], [1, 0, 0]],
        "labels": [[IGNORED, 2, 3], [IGNORED, IGNORED, IGNORED]],
    }


def test_packed_update_holds_every_example_once_in_batches_that_fit():
    batches = packed_update([item(2), item(5), item(1), item(2)], pad_id=0, budget=6, wrap=tuple)

    assert [batch["input_ids"] for batch in batches] == [
        ([1, 2, 3, 4, 5],),
        ([1, 2], [1, 2], [1, 0]),
    ]
    assert batches[1]["attention_mask"] == ([1, 1], [1, 1], [1, 0])


def test_run_settings_of_an_old_command_are_what_old_runs_saved():
    options = {"out": "outputs/run", "epochs": 2.0, "value_hints": True, "max_steps": -1}
    unused = {"init_adapter": None, "pack": 0, "autocast": False, "gradient_checkpointing": False}

    assert run_settings({**options, **unused}) == {
        "epochs": "2.0",
        "value_hints": "True",
        "max_steps": "-1",
    }


def test_run_settings_record_the_later_options_once_they_are_used():
    used = {"sample_rows": 0, "pack": 4096, "autocast": True, "gradient_checkpointing": False}

    assert run_settings(used) == {"sample_rows": "0", "pack": "4096", "autocast": "True"}


def test_plain_update_keeps_its_batch_size_and_accumulates():
    assert update_shape(1, 16, 0) == {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
    }


def test_packed_update_is_handed_over_whole():
    assert update_shape(1, 16, 4096) == {
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 1,
    }


def test_training_log_is_saved_with_sizes_time_and_peak_memory(tmp_path):
    steps = [{"loss": 0.5, "step": 10}]

    log = write_training_log(tmp_path, 10, 8, 91, (2_010_000_000, 3_094_000_000), steps)

    assert log == {
        "examples": 8,
        "left_out_too_long": 2,
        "training_seconds": 91,
        "peak_gpu_gb": 2.01,
        "peak_reserved_gb": 3.09,
        "steps": steps,
    }
    assert json.loads((tmp_path / "training_log.json").read_text(encoding="utf-8")) == log

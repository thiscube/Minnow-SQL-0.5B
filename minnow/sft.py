"""Prepare Spider training questions for supervised fine-tuning.

Nothing here needs a GPU. The tokenizer is passed in, so the logic can be tested with a tiny
stand-in.
"""

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from minnow.guard import Example, assert_no_leak, normalize
from minnow.prompt import SYSTEM, build_messages
from minnow.prompts import PromptBook
from minnow.spider import SpiderExample, load_split

IGNORED = -100
TRAIN_FILE = "train_spider.json"
DEV_FILE = "dev.json"
# the files whose code, with encode() below, decides what an encoded example looks like
_PROMPT_CODE = ("prompt.py", "prompts.py", "schema.py", "spider.py", "values.py")
Encoded = dict[str, list[int]]
# the Trainer settings that every run shares
TRAINING = {
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 2,
    "bf16": True,
    "seed": 0,
    "report_to": "none",
}
# options that runs before 2026-11-26 did not have
_LATER_OPTIONS = ("init_adapter", "pack", "autocast", "gradient_checkpointing")


def run_settings(options: Mapping[str, object]) -> dict[str, str]:
    """Return the settings that tie a run folder to one setup.

    Options that were added later are left out while they are unused, so a run started before
    they existed still matches its folder and can be resumed.
    """
    return {
        key: str(value)
        for key, value in options.items()
        if key != "out" and (value or key not in _LATER_OPTIONS)
    }


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


def _cache_name(settings: tuple[object, ...], examples: Sequence[SpiderExample]) -> str:
    """Name a cache file after everything the encoded examples depend on, the code included."""
    digest = hashlib.sha256(repr((settings, list(examples))).encode("utf-8"))
    digest.update(inspect.getsource(encode).encode("utf-8"))
    for name in _PROMPT_CODE:
        digest.update((Path(__file__).parent / name).read_bytes())
    return f"sft-{digest.hexdigest()[:16]}.json"


def training_set(  # noqa: PLR0913
    tokenizer: Any,  # noqa: ANN401
    spider: Path,
    examples: Sequence[SpiderExample],
    max_tokens: int,
    sample_rows: int = 0,
    value_hints: bool = False,
    cache: Path | None = None,
) -> list[Encoded]:
    """Encode every example that fits in max_tokens, in order.

    Building the prompts reads every database, which takes minutes. With a cache folder the
    result is saved under a name made from the examples, the settings and the prompt code, so
    the same run starts in seconds the next time. The databases are assumed not to change.
    """
    settings = (tokenizer.name_or_path, str(spider), max_tokens, sample_rows, value_hints, SYSTEM)
    path = cache / _cache_name(settings, examples) if cache else None
    if path and path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        book = PromptBook(spider, examples, sample_rows, value_hints)
        encoded = [encode(tokenizer, book.schema(e), e, max_tokens) for e in examples]
        rows = [[item["labels"].count(IGNORED), item["input_ids"]] for item in encoded if item]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            unfinished = path.with_suffix(".part")
            unfinished.write_text(json.dumps(rows), encoding="utf-8")
            unfinished.replace(path)
    return [
        {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [IGNORED] * prompt + ids[prompt:],
        }
        for prompt, ids in rows
    ]


def pack(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Group the positions of examples into batches, longest example first.

    A batch is padded to its longest example, so it holds that length times its number of
    examples in tokens. Each batch takes as many examples as fit in the budget, at least one.
    """
    batches: list[list[int]] = []
    for position in sorted(range(len(lengths)), key=lambda position: -lengths[position]):
        if batches and (len(batches[-1]) + 1) * lengths[batches[-1][0]] <= budget:
            batches[-1].append(position)
        else:
            batches.append([position])
    return batches


def padded(
    items: Sequence[Mapping[str, list[int]]], pad_id: int, wrap: Callable[[list[list[int]]], Any]
) -> dict[str, Any]:
    """Pad examples on the right to the longest one. Padding is masked and never learned.

    wrap turns each padded table of numbers into what the trainer wants, such as torch.tensor.
    """
    longest = max(len(item["input_ids"]) for item in items)
    fill = {"input_ids": pad_id, "attention_mask": 0, "labels": IGNORED}
    return {
        key: wrap([item[key] + [value] * (longest - len(item[key])) for item in items])
        for key, value in fill.items()
    }


def packed_update(
    items: Sequence[Mapping[str, list[int]]],
    pad_id: int,
    budget: int,
    wrap: Callable[[list[list[int]]], Any],
) -> list[dict[str, Any]]:
    """Split the examples of one weight update into padded batches that fit the budget."""
    groups = pack([len(item["input_ids"]) for item in items], budget)
    return [padded([items[position] for position in group], pad_id, wrap) for group in groups]


def update_shape(batch_size: int, accumulate: int, pack_budget: int) -> dict[str, int]:
    """Return the Trainer's batch settings. A packed update arrives whole and is split later."""
    if pack_budget:
        whole = batch_size * accumulate
        return {"per_device_train_batch_size": whole, "gradient_accumulation_steps": 1}
    return {"per_device_train_batch_size": batch_size, "gradient_accumulation_steps": accumulate}


def write_training_log(
    out: Path,
    examples: int,
    kept: int,
    seconds: int | None,
    peak_bytes: tuple[int, int],
    steps: list[dict[str, float]],
) -> dict[str, object]:
    """Save the loss curve with the run's size, time and peak GPU memory, and return it.

    peak_bytes is the most memory in use and the most held back from other programs.
    """
    log = {
        "examples": kept,
        "left_out_too_long": examples - kept,
        "training_seconds": seconds,
        "peak_gpu_gb": round(peak_bytes[0] / 1e9, 2),
        "peak_reserved_gb": round(peak_bytes[1] / 1e9, 2),
        "steps": steps,
    }
    text = json.dumps(log, indent=2) + "\n"
    (out / "training_log.json").write_text(text, encoding="utf-8", newline="\n")
    return log

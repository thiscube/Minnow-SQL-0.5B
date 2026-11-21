"""Score an untouched model on Spider dev by execution accuracy.

    python scripts/baseline.py

Predictions are saved after every batch, so a stopped run continues where it left off.
Needs the model extra (pip install -e ".[model]") and Spider unpacked in data/spider.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minnow.evaluate import append_prediction, load_predictions, score
from minnow.prompt import build_messages, extract_sql
from minnow.schema import create_statements
from minnow.spider import database_path, load_split

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--spider", type=Path, default=Path("data/spider"))
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def generate(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list[str], max_new_tokens: int
) -> list[str]:
    """Answer a batch of prompts with greedy decoding."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    """Generate any missing predictions, then score them all."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    predictions_file = args.out / "predictions.jsonl"
    examples = load_split(args.spider, "dev.json")
    done = load_predictions(predictions_file)
    todo = [index for index in range(len(examples)) if index not in done]
    print(f"{len(done)} of {len(examples)} already predicted, {len(todo)} to go.")

    started = time.monotonic()
    if todo:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
        schemas = {
            db_id: create_statements(database_path(args.spider, db_id))
            for db_id in {e.db_id for e in examples}
        }
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(schemas[examples[i].db_id], examples[i].question),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for i in batch
            ]
            for index, answer in zip(
                batch, generate(model, tokenizer, prompts, args.max_new_tokens), strict=True
            ):
                append_prediction(predictions_file, index, extract_sql(answer))
            print(f"{start + len(batch)} of {len(todo)} predicted")
    generation_seconds = time.monotonic() - started

    result = score(examples, load_predictions(predictions_file), args.spider)
    summary = {
        "model": args.model,
        "split": "dev",
        "correct": result.correct,
        "total": result.total,
        "execution_accuracy": round(result.accuracy, 2),
        "generation_seconds_this_run": round(generation_seconds),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "torch": torch.__version__,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Execution accuracy: {result.accuracy:.2f}% ({result.correct} of {result.total})")


if __name__ == "__main__":
    main()

"""Score a model on Spider dev by execution accuracy: python scripts/baseline.py

Saved after every batch, so a stopped run continues. The folder gets predictions.jsonl,
outcomes.jsonl and summary.json for minnow.runlog, and candidates.jsonl with every candidate.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from generation import generate, load, peak_memory, sample_shared

from minnow.evaluate import (
    CANDIDATES_FILE,
    add_generation_seconds,
    append_candidates,
    append_prediction,
    claim_run_folder,
    drop_unfinished_line,
    load_predictions,
)
from minnow.execution import run_query
from minnow.fanout import CACHE_GIGABYTES, Fanout
from minnow.outcomes import OUTCOMES_FILE, breakdown, format_breakdown, judge_all, save_outcomes
from minnow.prompt import SYSTEM, chat_prompt, extract_sql
from minnow.select import pick
from minnow.shuffle import ShuffledBook
from minnow.spider import database_path, load_split

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter", type=Path, help="folder of a trained LoRA adapter to load")
    parser.add_argument("--spider", type=Path, default=Path("data/spider"))
    parser.add_argument("--split", default="dev.json", help="question file in the Spider folder")
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample-rows", type=int, default=0, help="example rows per table")
    parser.add_argument("--value-hints", action="store_true", help="list mentioned values")
    parser.add_argument("--shuffle-tables", type=int, metavar="SEED", help="shuffled table order")
    parser.add_argument("--samples", type=int, default=1, help="candidate queries per question")
    parser.add_argument("--temperature", type=float, default=0.8, help="for the extra candidates")
    parser.add_argument("--share-prompt", action="store_true", help="encode once for all samples")
    parser.add_argument("--cache-gb", type=float, default=CACHE_GIGABYTES, help="max GB of cache")
    parser.add_argument("--limit", type=int, help="score only the first questions, as a check")
    parser.add_argument("--device", default="cuda", help="cpu runs in full precision, slowly")
    return parser.parse_args()


def main() -> None:
    """Generate any missing predictions, then score them all."""
    args = parse_args()
    name = f"{args.model} + {args.adapter.parent.name}" if args.adapter else args.model
    settings = {
        "model": name,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "samples": args.samples,
        "split": args.split,
        "sample_rows": args.sample_rows,
        "value_hints": args.value_hints,
        "temperature": args.temperature,
        "system_prompt": SYSTEM,
    }
    # Only written when used, so folders of earlier runs keep matching their settings.
    extras = {"share_prompt_gb": args.cache_gb if args.share_prompt else None, "limit": args.limit}
    extras |= {"device": args.device, "shuffle_tables": args.shuffle_tables}
    settings |= {key: value for key, value in extras.items() if value not in (None, "cuda")}
    claim_run_folder(args.out, settings)
    predictions_file = args.out / "predictions.jsonl"
    drop_unfinished_line(predictions_file)
    drop_unfinished_line(args.out / CANDIDATES_FILE)
    examples = load_split(args.spider, args.split)[: args.limit]
    done = load_predictions(predictions_file)
    todo = [index for index in range(len(examples)) if index not in done]
    print(f"{len(done)} of {len(examples)} already predicted, {len(todo)} to go.")

    started = time.monotonic()
    if todo:
        model, tokenizer = load(args.model, args.adapter, args.device)
        book = ShuffledBook(
            args.spider, examples, args.sample_rows, args.value_hints, args.shuffle_tables
        )
        # Answers depend a little on batch mates, so batches are fixed blocks of the split.
        for start in range(0, len(examples), args.batch_size):
            batch = range(start, min(start + args.batch_size, len(examples)))
            if all(index in done for index in batch):
                continue
            prompts = [
                chat_prompt(tokenizer, book.schema(examples[i], i), examples[i].question)
                for i in batch
            ]
            # One greedy answer, then sampled ones; the vote never sees the reference query.
            tokens, heat = args.max_new_tokens, args.temperature
            rounds = [generate(model, tokenizer, prompts, tokens)]
            if args.share_prompt and args.samples > 1:
                plan = Fanout(
                    args.samples - 1, tokens, heat, 1000 * start, int(args.cache_gb * 1e9)
                )
                rounds += sample_shared(model, tokenizer, prompts, plan)
            else:
                for sample in range(1, args.samples):
                    torch.manual_seed(1000 * start + sample)
                    rounds.append(generate(model, tokenizer, prompts, tokens, heat))
            for position, index in enumerate(batch):
                if index not in done:
                    database = database_path(args.spider, examples[index].db_id)
                    options = [extract_sql(answers[position]) for answers in rounds]
                    chosen = pick(options, lambda sql, db=database: run_query(db, sql))
                    append_candidates(args.out / CANDIDATES_FILE, index, options)
                    append_prediction(predictions_file, index, chosen)
            print(f"{batch.stop} of {len(examples)} predicted")
        print(peak_memory(args.device))
    spent = time.monotonic() - started if todo else 0.0
    generation_seconds = add_generation_seconds(args.out, spent)

    outcomes = judge_all(examples, load_predictions(predictions_file), args.spider)
    save_outcomes(args.out / OUTCOMES_FILE, outcomes)
    counts = breakdown(outcomes)
    accuracy = 100 * counts.correct / counts.total
    summary = {
        "model": name,
        "split": args.split.removesuffix(".json"),
        "correct": counts.correct,
        "total": counts.total,
        "execution_accuracy": round(accuracy, 2),
        "generation_seconds": generation_seconds,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy" if args.samples == 1 else f"vote over {args.samples} candidates",
        "torch": torch.__version__,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (args.out / "summary.json").write_text(text, encoding="utf-8", newline="\n")
    print(format_breakdown(counts))
    print(f"Execution accuracy: {accuracy:.2f}% ({counts.correct} of {counts.total})")


if __name__ == "__main__":
    main()

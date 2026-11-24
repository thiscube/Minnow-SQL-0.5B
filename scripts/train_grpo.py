"""Improve a fine-tuned adapter with reinforcement learning (GRPO) on execution results.

python scripts/train_grpo.py --adapter outputs/sft-synsql/adapter --out outputs/grpo. Saves
every few questions; the same command continues a stopped run. The method is in minnow.grpo.
"""

import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from minnow import grpo
from minnow.evaluate import add_generation_seconds, claim_run_folder
from minnow.guard import assert_no_leak
from minnow.prompt import SYSTEM, chat_prompt, extract_sql
from minnow.prompts import PromptBook
from minnow.sft import checked_training_examples, for_guard
from minnow.spider import load_split

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, required=True, help="fine-tuned adapter to start")
    parser.add_argument("--out", type=Path, default=Path("outputs/grpo"))
    parser.add_argument("--spider", type=Path, default=Path("data/spider"))
    parser.add_argument("--train-files", nargs="+", default=["train_spider.json"])
    parser.add_argument("--sample-rows", type=int, default=0, help="example rows per table")
    parser.add_argument("--value-hints", action="store_true", help="list mentioned values")
    parser.add_argument("--group", type=int, default=8, help="queries written per question")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-update", type=int, default=4, help="questions with a signal")
    parser.add_argument("--max-questions", type=int, default=-1, help="stop early, for a test")
    return parser.parse_args()


def log_prob(model: PeftModel, prompt: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
    """Return the answer tokens' summed log-probability over a fixed length (not its own)."""
    ids = torch.cat([prompt, answer]).unsqueeze(0)
    base = model.get_base_model()
    hidden = base.model(input_ids=ids).last_hidden_state[0, len(prompt) - 1 : -1]
    logits = base.lm_head(hidden).float()
    picked = torch.log_softmax(logits, -1).gather(1, answer.unsqueeze(1))
    return picked.sum() / grpo.MAX_NEW_TOKENS


def sample(model: PeftModel, prompt: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Write a group of answers to one prompt and return their new tokens."""
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt.repeat(args.group, 1),
            attention_mask=torch.ones(args.group, len(prompt), device=model.device),
            **grpo.SAMPLING,
        )
    return output[:, len(prompt) :]


def learn(model: PeftModel, prompt: torch.Tensor, answers: list, weights: list[float]) -> None:
    """Add the gradient that makes better-than-average answers more likely to the model."""
    for answer, weight in zip(answers, weights, strict=True):
        if weight:
            (-weight * log_prob(model, prompt, answer.clone()) / len(answers)).backward()


def load(args: argparse.Namespace) -> tuple[PeftModel, torch.optim.Optimizer]:
    """Load the adapter (the latest save of this run if there is one) and its optimizer."""
    start = args.out / "latest" if (args.out / "latest").exists() else args.adapter
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, start, is_trainable=True)
    model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    if (args.out / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(args.out / "optimizer.pt"))
    return model, optimizer


def step(model: PeftModel, optimizer: torch.optim.Optimizer, progress: grpo.Progress) -> None:
    """Apply the gradients gathered so far."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    progress.updates += 1


def save(
    model: PeftModel, optimizer: torch.optim.Optimizer, progress: grpo.Progress, out: Path
) -> None:
    """Save the adapter and optimizer first and the progress last, so a crash only repeats."""
    model.save_pretrained(out / "latest")
    torch.save(optimizer.state_dict(), out / "optimizer.pt")
    progress.save(out)


def main() -> None:
    """Train until every question was seen once, saving as it goes."""
    args = parse_args()
    settings = {k: str(v) for k, v in vars(args).items() if k not in ("out", "max_questions")}
    claim_run_folder(args.out, {**settings, "system_prompt": SYSTEM})
    examples = checked_training_examples(args.spider, tuple(args.train_files))
    dev = load_split(args.spider, "dev.json")
    assert_no_leak(for_guard(examples), for_guard(dev))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    book = PromptBook(args.spider, examples, args.sample_rows, args.value_hints)
    stops = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    progress = grpo.Progress.load(args.out)
    model, optimizer = load(args)
    order = grpo.training_order(len(examples), seed=0)
    first = progress.position
    stop = len(order) if args.max_questions < 0 else first + args.max_questions
    started, signals = time.monotonic(), 0
    for position in range(first, min(stop, len(order))):
        example = examples[order[position]]
        text = chat_prompt(tokenizer, book.schema(example), example.question)
        prompt = tokenizer(text, return_tensors="pt")["input_ids"][0].to(model.device)
        weights, rewards = None, None
        if len(prompt) <= grpo.MAX_PROMPT_TOKENS:
            torch.manual_seed(position)
            rows = sample(model, prompt, args)
            answers = [row[: grpo.answer_length(row.tolist(), stops)] for row in rows]
            sqls = [extract_sql(t) for t in tokenizer.batch_decode(answers, True)]
            rewards = grpo.group_rewards(args.spider, example, sqls)
            weights = grpo.advantages(rewards) if rewards else None
        if weights is not None:
            learn(model, prompt, answers, [w / args.per_update for w in weights])
            signals += 1
            if signals % args.per_update == 0:
                step(model, optimizer, progress)
        progress.record(sum(rewards) / len(rewards) if rewards else None, weights is not None)
        if progress.position % grpo.SAVE_EVERY == 0 or progress.position == stop:
            if signals % args.per_update:
                signals = 0
                step(model, optimizer, progress)
            save(model, optimizer, progress, args.out)
        if progress.position % 10 == 0:
            each = (time.monotonic() - started) / (progress.position - first)
            print(progress.report(len(order), each, torch.cuda.max_memory_allocated() / 1e9))
    add_generation_seconds(args.out, time.monotonic() - started)
    if progress.position >= len(order):
        model.save_pretrained(args.out / "adapter")
    print(f"Stopped at question {progress.position} of {len(order)}.")


if __name__ == "__main__":
    main()

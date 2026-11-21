"""Fine-tune the base model on Spider's training questions with LoRA.

    python scripts/train_sft.py

Checkpoints are saved every 100 steps. Stop the run with Ctrl+C at any time and start the same
command again: it continues from the latest checkpoint. The finished adapter is written to
<out>/adapter, and the loss curve to <out>/training_log.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from minnow.evaluate import add_generation_seconds, claim_run_folder
from minnow.guard import Example, assert_no_leak
from minnow.prompt import SYSTEM
from minnow.sft import IGNORED, checked_training_examples, encode
from minnow.spider import load_split, schemas

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class AnswerOnlyTrainer(Trainer):
    """Compute the loss from the answer's positions only.

    The stock loss scores every position against the whole vocabulary and then ignores the
    prompt. For a long schema that costs gigabytes for nothing. This gives the same loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **_):  # noqa: ANN001, ANN201, ANN003
        """Return the cross-entropy of the answer tokens."""
        labels = inputs["labels"][:, 1:]
        base = model.get_base_model()
        hidden = base.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state[:, :-1]
        learned = labels != IGNORED
        logits = base.lm_head(hidden[learned]).float()
        loss = torch.nn.functional.cross_entropy(logits, labels[learned])
        return (loss, None) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--spider", type=Path, default=Path("data/spider"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sft-spider"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    # Memory grows with the square of the length, so 8 GB holds one long example at a time.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=16, help="batches per weight update")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--sample-rows", type=int, default=0, help="example rows per table")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early, for a test run")
    parser.add_argument("--train-files", nargs="+", default=["train_spider.json"])
    return parser.parse_args()


def main() -> None:
    """Train, resuming from the latest checkpoint when there is one."""
    args = parse_args()
    settings = {key: str(value) for key, value in vars(args).items() if key != "out"}
    claim_run_folder(args.out, {**settings, "system_prompt": SYSTEM})

    examples = checked_training_examples(args.spider, tuple(args.train_files))
    dev = load_split(args.spider, "dev.json")
    assert_no_leak(
        [Example(e.question, e.db_id) for e in examples],
        [Example(e.question, e.db_id) for e in dev],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    schema_of = schemas(args.spider, examples, args.sample_rows)
    encoded = [encode(tokenizer, schema_of[e.db_id], e, args.max_tokens) for e in examples]
    dataset = [item for item in encoded if item is not None]
    print(f"{len(dataset)} of {len(examples)} examples fit in {args.max_tokens} tokens.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainer = AnswerOnlyTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORED),
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulate,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            bf16=True,
            seed=0,
            report_to="none",
        ),
    )
    # compute_loss returns a mean, so the trainer has to divide it by the batches per update
    trainer.model_accepts_loss_kwargs = False
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=any(args.out.glob("checkpoint-*")))
    seconds = add_generation_seconds(args.out, time.monotonic() - started)

    model.save_pretrained(args.out / "adapter")
    log = {
        "examples": len(dataset),
        "left_out_too_long": len(examples) - len(dataset),
        "training_seconds": seconds,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "steps": trainer.state.log_history,
    }
    text = json.dumps(log, indent=2) + "\n"
    (args.out / "training_log.json").write_text(text, encoding="utf-8", newline="\n")
    print(f"Done in {seconds} s, peak GPU memory {log['peak_gpu_gb']} GB.")


if __name__ == "__main__":
    main()

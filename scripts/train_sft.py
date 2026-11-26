"""Fine-tune the base model on Spider's training questions with LoRA.

    python scripts/train_sft.py

Checkpoints are saved every 100 steps. Stop the run with Ctrl+C at any time and start the same
command again: it continues from the latest checkpoint. The finished adapter is written to
<out>/adapter, and the loss curve to <out>/training_log.json.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model

from minnow import sft
from minnow.evaluate import add_generation_seconds, claim_run_folder
from minnow.guard import assert_no_leak
from minnow.prompt import SYSTEM
from minnow.spider import load_split

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = {"lora_dropout": 0.05, "target_modules": LORA_TARGETS, "task_type": "CAUSAL_LM"}
CACHE = Path("outputs/cache")


class AnswerOnlyTrainer(transformers.Trainer):
    """Compute the loss from the answer's positions only.

    The stock loss scores every position against the whole vocabulary and then ignores the
    prompt. For a long schema that costs gigabytes for nothing. This gives the same loss.
    """

    autocast = False

    def compute_loss(self, model, inputs, return_outputs=False, **_):  # noqa: ANN001, ANN201, ANN003
        """Return the cross-entropy of the answer tokens, every example counting the same."""
        labels = inputs["labels"][:, 1:]
        base = model.get_base_model()
        learned = labels != sft.IGNORED
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            hidden = base.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).last_hidden_state[:, :-1]
            logits = base.lm_head(hidden[learned]).float()
        losses = torch.nn.functional.cross_entropy(logits, labels[learned], reduction="none")
        share = (1 / learned.sum(1, keepdim=True)).expand_as(learned)[learned]
        loss = (losses * share).sum() / len(labels)
        return (loss, None) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):  # noqa: ANN001, ANN201
        """Run a packed update one batch after another, so memory only holds one of them."""
        if not isinstance(inputs, list):
            return super().training_step(model, inputs, num_items_in_batch)
        model.train()
        torch.cuda.empty_cache()  # the longest batch comes first and sets what gets reserved
        examples = sum(len(batch["labels"]) for batch in inputs)
        total = 0.0
        for batch in inputs:
            loss = self.compute_loss(model, batch) * len(batch["labels"]) / examples
            self.accelerator.backward(loss)
            total += loss.detach()
        return total


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--spider", type=Path, default=Path("data/spider"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sft-spider"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=16, help="batches per weight update")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--sample-rows", type=int, default=0, help="example rows per table")
    parser.add_argument("--value-hints", action="store_true", help="list mentioned values")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early, for a test run")
    parser.add_argument("--train-files", nargs="+", default=["train_spider.json"])
    parser.add_argument("--init-adapter", type=Path, help="start from this adapter's weights")
    parser.add_argument("--pack", type=int, default=0, help="padded tokens per batch (sft.pack)")
    parser.add_argument("--autocast", action="store_true", help="bf16 in the LoRA branch too")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="much less memory")
    return parser.parse_args()


def main() -> None:
    """Train, resuming from the latest checkpoint when there is one."""
    args = parse_args()
    claim_run_folder(args.out, {**sft.run_settings(vars(args)), "system_prompt": SYSTEM})

    examples = sft.checked_training_examples(args.spider, tuple(args.train_files))
    dev = load_split(args.spider, "dev.json")
    assert_no_leak(sft.for_guard(examples), sft.for_guard(dev))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    dataset = sft.training_set(
        tokenizer, args.spider, examples, args.max_tokens, args.sample_rows, args.value_hints, CACHE
    )
    print(f"{len(dataset)} of {len(examples)} examples fit in {args.max_tokens} tokens.")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    if args.gradient_checkpointing:
        model.activation_checkpointing_enable()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, **LORA)
        model = get_peft_model(model, lora)
    pad = tokenizer.pad_token_id
    packed = partial(sft.packed_update, pad_id=pad, budget=args.pack, wrap=torch.tensor)
    padding = transformers.DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=sft.IGNORED)
    trainer = AnswerOnlyTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=packed if args.pack else padding,
        args=transformers.TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            **sft.update_shape(args.batch_size, args.accumulate, args.pack),
            **sft.TRAINING,
        ),
    )
    # compute_loss returns a mean, so the trainer has to divide it by the batches per update
    trainer.model_accepts_loss_kwargs = False
    trainer.autocast = args.autocast
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=any(args.out.glob("checkpoint-*")))
    seconds = add_generation_seconds(args.out, time.monotonic() - started)

    model.save_pretrained(args.out / "adapter")
    peaks = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    steps = trainer.state.log_history
    log = sft.write_training_log(args.out, len(examples), len(dataset), seconds, peaks, steps)
    print(f"Done in {seconds} s, peak GPU memory {log['peak_gpu_gb']} GB allocated.")
    print(f"Reserved at most {log['peak_reserved_gb']} GB.")


if __name__ == "__main__":
    main()

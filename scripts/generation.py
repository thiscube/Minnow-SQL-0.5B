"""Turn prompts into answers: one answer per call, or many sampled answers from one encoding.

Used by scripts/baseline.py. The planning (how many copies fit in memory) is in minnow.fanout.
"""

import copy
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from minnow.fanout import Fanout, by_round


def load(
    name: str, adapter: Path | None, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and its tokenizer, with a trained adapter on top when one is given."""
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer


def peak_memory(device: str) -> str:
    """Say how much GPU memory the run needed at its fullest."""
    if device == "cpu":
        return "Peak GPU memory: none, this ran on the CPU."
    used, held = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9
    return f"Peak GPU memory: {used:.2f} GB in use, {held:.2f} GB held by PyTorch."


def decoding(temperature: float | None) -> dict[str, object]:
    """Return the generate() options: sampled when a temperature is given, greedy otherwise."""
    return {"do_sample": True, "temperature": temperature} if temperature else {"do_sample": False}


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float | None = None,
) -> list[str]:
    """Answer a batch of prompts: greedy, or sampled when a temperature is given."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, **decoding(temperature))
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def encode(model: AutoModelForCausalLM, ids: torch.Tensor, mask: torch.Tensor) -> DynamicCache:
    """Run every prompt but its last token through the model once and return the cache.

    Positions count real tokens only, the way generate() numbers left-padded prompts. Word
    scores are worked out for one token only, like generate() does: for every token of every
    prompt they would take more memory than the model itself.
    """
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
    output = model(
        input_ids=ids[:, :-1],
        attention_mask=mask[:, :-1],
        position_ids=positions[:, :-1],
        use_cache=True,
        logits_to_keep=1,
    )
    return output.past_key_values


def sample_shared(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list[str], plan: Fanout
) -> list[list[str]]:
    """Answer each prompt plan.copies times from one encoding: a list of answers per round."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    rounds: list[list[str]] = []
    with torch.inference_mode():
        prefix = encode(model, ids, mask)
        cached = sum(layer.keys.nbytes + layer.values.nbytes for layer in prefix.layers)
        token_bytes = cached // (ids.shape[0] * (ids.shape[1] - 1))
        for seed, copies in plan.passes(ids.shape[0], ids.shape[1], token_bytes):
            cache = copy.deepcopy(prefix)
            cache.batch_repeat_interleave(copies)
            torch.manual_seed(seed)
            # generate() gets the whole prompt, sees the cache and feeds only the last token
            output = model.generate(
                input_ids=ids.repeat_interleave(copies, dim=0),
                attention_mask=mask.repeat_interleave(copies, dim=0),
                past_key_values=cache,
                max_new_tokens=plan.max_new_tokens,
                **decoding(plan.temperature),
            )
            answers = tokenizer.batch_decode(output[:, ids.shape[1] :], skip_special_tokens=True)
            rounds += by_round(answers, copies)
            del cache, output
    return rounds

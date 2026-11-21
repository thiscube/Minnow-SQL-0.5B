# Minnow-SQL-0.5B

[![checks](https://github.com/thiscube/minnow-sql/actions/workflows/checks.yml/badge.svg)](https://github.com/thiscube/minnow-sql/actions/workflows/checks.yml)

A small text-to-SQL model: ask a question in English, get a SQL query back.

It is a fine-tune of [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) (about 500M parameters), trained almost entirely on one consumer GPU (RTX 3060 Ti, 8 GB).

> **Status: work in progress.** Nothing is trained yet and there are no results to show. The first step is measuring the untouched base model. This page will be updated with real numbers as they come in.

## Goal

Score as high as possible on the [Spider](https://yale-lily.github.io/spider) dev set (1,034 questions), measured by execution accuracy: the generated query has to run and return the correct rows.

| Model | Spider dev, execution accuracy |
| --- | --- |
| Untouched base model (reported by [another fine-tune's model card](https://huggingface.co/alialialialaiali/qwen2.5-coder-spider-sql)) | about 34.5% |
| That fine-tune | 39.2% |
| Best published 0.5B result ([SLM-SQL, arXiv 2507.22478](https://arxiv.org/abs/2507.22478)) | 73.5% |
| **Minnow-SQL-0.5B** | not measured yet (target: 70%+) |

## Plan

1. **Baseline.** Run the untouched model on Spider dev and record the score.
2. **First fine-tune.** Train on Spider's training set (about 7,000 question/SQL pairs) and measure again.
3. **Improve.** Add synthetic pairs from SynSQL-2.5M, try LoRA settings and different ways of showing the table schema, and try generating several queries and keeping one that runs. Measure after every change.
4. **Show it.** Live demo, results chart and write-up.

## Ground rules

- Spider dev questions are never used for training, only for scoring.
- Every measured change is logged in `results.md` with the date, what changed and the score.
- Rented GPU spending is capped at $15 for the whole project. Everything else trains locally.

## Code quality

A wrong score is worse than no score, so every change has to pass the same checks before it is merged:

- **Format and lint** with [ruff](https://docs.astral.sh/ruff/): no unused or commented-out code, type hints and docstrings on public functions, and limits on how large or tangled a function may get.
- **Tests** with pytest. At least 90% of `minnow/` has to be covered, so new logic can't arrive without tests.
- **A data-leak guard** ([`minnow/guard.py`](minnow/guard.py)). Training data may not contain a Spider dev question or use a Spider dev database. The guard is run against a planted leak on every check, so it can't quietly stop working.
- **Structure tests.** Any file that starts training has to call the leak guard, and no source file may grow past 300 lines.

Run them all on Windows with:

```
powershell -File scripts/verify.ps1
```

The same steps run again on GitHub after each merge (see [`checks.yml`](.github/workflows/checks.yml)).

## Download

The model weights will be published on Hugging Face as `Minnow-SQL-0.5B` once there is a trained version worth sharing. This repository holds the code and results.

## License

Code in this repository is under the [MIT License](LICENSE). The base model and the datasets have their own licenses.

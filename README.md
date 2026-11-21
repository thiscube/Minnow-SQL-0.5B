# Minnow-SQL-0.5B

[![checks](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml/badge.svg)](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Minnow-SQL-0.5B is a text-to-SQL model in the 0.5B-parameter class, fine-tuned from [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct). Given a natural-language question and a database schema, it produces a SQL query.

> [!NOTE]
> This project is in progress. The baseline is measured (45.36% on Spider dev), training has not started, and no model has been released. See the [roadmap](#roadmap).

## Goal

Reach at least 70% execution accuracy on the [Spider][spider] development set (1,034 questions) with a 0.5B-parameter model trained on a single consumer GPU. Execution accuracy counts a prediction as correct when the generated query runs and returns the same rows as the reference query.

## Results

Measured in this repository. Every run is logged in [results.md](results.md), with its predictions under [results/](results).

| Model | Training | Spider dev EX (%) | Correct |
| --- | --- | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct | none (baseline) | 45.36 | 469 / 1034 |
| Minnow-SQL-0.5B | | not yet trained | |

Of the 565 baseline errors, 389 are queries that do not run, and 356 of those name a column that does not exist. Of the 469 correct predictions, 30 match a reference query that returns no rows; with one database per question, a wrong query that also returns nothing is counted as correct, so the figure is slightly optimistic.

### Published results for comparison

| Model | Training | Spider dev EX (%) | Spider test EX (%) | Source |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B | none | 34.5 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B | fine-tuned | 39.2 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B-Instruct | none | 44.07 | 44.31 | [SLM-SQL][slm-sql] |
| SLM-SQL-0.5B | SFT and RL on SynSQL-derived data | 72.08 | 73.50 | [SLM-SQL][slm-sql] |

None of the published rows has been reproduced here, and they are not directly comparable with each other or with this repository's numbers. The SLM-SQL rows use that paper's corrective self-consistency decoding. The model-card rows use the non-instruct base model. This repository uses greedy decoding and its own implementation of execution accuracy on the original Spider databases, which is simpler than the official test-suite evaluation.

## Roadmap

- [x] Tooling: linting, tests, a train/dev leakage guard, continuous integration
- [x] Baseline: the unmodified base model scores 45.36% on Spider dev
- [ ] Supervised fine-tuning on the Spider training set (about 7,000 examples)
- [ ] Improvements: a subset of [SynSQL-2.5M][omnisql], LoRA configuration, schema serialization, and execution-guided selection among sampled queries
- [ ] Release: model weights on Hugging Face, a demo, and a write-up of the results

## Constraints

- Hardware: one NVIDIA RTX 3060 Ti (8 GB VRAM) and 32 GB of RAM.
- Cloud compute: at most $15 in total.
- Spider dev is used for evaluation only and never for training.

## Repository layout

```
minnow/       library code; imports without a GPU or deep-learning dependencies
tests/        unit tests and structure tests
scripts/      entry points and the local check runner
results/      predictions and score summary of each logged run
results.md    the log of measured runs
```

## Reproducing the baseline

Requires a CUDA GPU with about 3 GB of free memory. The run takes about 5 minutes on an RTX 3060 Ti.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support, then the model dependencies:

   ```bash
   pip install -e ".[model]"
   ```

2. Download [Spider][spider] and unpack it so that `data/spider/dev.json` and `data/spider/database/` exist.

3. Run the evaluation:

   ```bash
   python scripts/baseline.py
   ```

Predictions are written to `outputs/baseline/` after every batch, so an interrupted run continues where it stopped.

## Development

Requires Python 3.12 or later.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the checks:

```bash
ruff format --check .
ruff check .
pytest
```

On Windows, `powershell -File scripts/verify.ps1` runs all of them in one step. The same checks run in CI on every push to `main`. They cover:

- formatting and linting with [ruff](https://docs.astral.sh/ruff/), including limits on function length and complexity
- unit tests with a 90% coverage floor on `minnow/`
- structure tests: every module has a test file, source files stay under a size limit, and any file that starts training must call the leakage guard

### Data leakage guard

`minnow.guard` verifies that a training set shares no question and no database with the Spider dev set. Questions are compared after normalizing case, punctuation, and whitespace.

```bash
python -m minnow.guard train.json dev.json
```

Both files are JSON lists of objects with `question` and `db_id` fields, the format Spider uses. The command exits with 0 when the data is clean and with 3 when it finds a leak.

## License

The code in this repository is released under the [MIT License](LICENSE). The base model and the datasets are distributed under their own licenses.

## References

- Yu et al. [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task][spider-paper]. EMNLP 2018.
- Hui et al. [Qwen2.5-Coder Technical Report][qwen-paper]. 2024.
- Li et al. [OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale][omnisql]. 2025. Source of SynSQL-2.5M.
- Sheng and Xu. [SLM-SQL: An Exploration of Small Language Models for Text-to-SQL][slm-sql]. 2025.

[spider]: https://yale-lily.github.io/spider
[spider-paper]: https://arxiv.org/abs/1809.08887
[qwen-paper]: https://arxiv.org/abs/2409.12186
[omnisql]: https://arxiv.org/abs/2503.02240
[slm-sql]: https://arxiv.org/abs/2507.22478
[thesis]: https://huggingface.co/alialialialaiali/qwen2.5-coder-spider-sql

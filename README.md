# Minnow-SQL-0.5B

[![checks](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml/badge.svg)](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Minnow-SQL-0.5B is a text-to-SQL model in the 0.5B-parameter class, fine-tuned from [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct). Given a natural-language question and a database schema, it produces a SQL query.

> [!NOTE]
> This project is in progress. The baseline is measured (see [results](#results)), training has not started, and no model has been released. See the [roadmap](#roadmap).

## Goal

Reach at least 70% execution accuracy on the [Spider][spider] development set (1,034 questions) with a 0.5B-parameter model trained on a single consumer GPU. Execution accuracy counts a prediction as correct when the generated query runs and returns the same rows as the reference query.

## Results

Measured in this repository. The table and chart below are generated from the runs' own files, and [results.md](results.md) breaks each run down by kind of error.

<!-- results:start -->

| Model | Change | Spider dev EX (%) | Correct |
| --- | --- | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct | Baseline: no training, greedy decoding, CREATE TABLE statements as the schema | 45.36 | 469 / 1034 |

![Execution accuracy by run](results/chart.svg)

<!-- results:end -->

### How results are measured

- A prediction is correct when it runs and returns the same rows as the reference query, in the same order when the reference query sorts, with columns in any order.
- Each question is scored on its one original Spider database. A wrong query can therefore be counted as correct when both it and the reference return no rows; results.md reports how many correct predictions are of this kind.
- Decoding is greedy, in fixed batches of 16. The batch size affects a few answers: redoing 200 baseline questions one at a time changed the verdict for 3 of them. Differences of about one point between runs are within this noise.

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
- [x] Baseline: the unmodified base model, evaluated on Spider dev
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
results/      predictions, verdicts and score summary of each logged run
results.md    the log of measured runs, generated from results/
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

## Results log

No score in this repository is typed by hand. A run writes its predictions, a verdict for every question and a summary. Three small tools turn those files into everything that is reported:

```bash
python -m minnow.outcomes outputs/run --spider data/spider    # judge a run again and count its errors
python -m minnow.compare results/baseline outputs/run         # which questions a change fixed or broke
python -m minnow.runlog add outputs/run --change "..."        # log the run and regenerate the reports
```

`runlog add` copies the run into `results/`, records it in `results/runs.json`, and regenerates `results.md`, the chart and the results table in this README. `python -m minnow.runlog check` fails when any of them is out of date or a run under `results/` is not logged, and it is part of the checks below.

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
python -m minnow.runlog check
```

On Windows, `powershell -File scripts/verify.ps1` runs all of them in one step. The same checks run in CI on every push to `main`. They cover:

- formatting and linting with [ruff](https://docs.astral.sh/ruff/), including limits on function length and complexity
- unit tests with a 90% coverage floor on `minnow/`
- structure tests: every module has a test file, source files stay under a size limit, and any file that starts training must call the leakage guard
- the results log: every reported number matches the runs' files

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

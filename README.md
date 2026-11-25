# Minnow-SQL-0.5B

[![checks](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml/badge.svg)](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Minnow-SQL-0.5B is a text-to-SQL model in the 0.5B-parameter class, fine-tuned from [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct). Given a natural-language question and a database schema, it produces a SQL query.

> [!NOTE]
> Training is done and the 70% goal is reached on Spider dev by a vote of three fine-tuned adapters (71.18%). The best single model scores 66.92%. The final score on held-out Spider test databases is still to be measured, and no weights have been released yet. See the [roadmap](#roadmap).

| | Spider dev EX |
| --- | ---: |
| Base model, untouched | 45.36% |
| Best single model (SFT, then reinforcement learning) | 66.92% |
| Vote of three fine-tuned adapters | **71.18%** |

## Goal

Reach at least 70% execution accuracy on the [Spider][spider] development set (1,034 questions) with a 0.5B-parameter model trained on a single consumer GPU. Execution accuracy counts a prediction as correct when the generated query runs and returns the same rows as the reference query.

## Results

Measured in this repository. The table and chart below are generated from the runs' own files, and [results.md](results.md) breaks each run down by kind of error.

<!-- results:start -->

| Model | Change | Spider dev EX (%) | Correct |
| --- | --- | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct | Baseline: no training, greedy decoding, CREATE TABLE statements as the schema | 45.36 | 469 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-spider | LoRA fine-tuning on Spider train: 6,142 examples of at most 1,024 tokens, 3 epochs | 60.64 | 627 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-spider | Same model as sft-spider; five candidate queries per question, keeping the result most candidates agree on | 64.31 | 665 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-spider-all | LoRA fine-tuning on Spider train plus train_others: 7,800 examples, 3 epochs, one greedy query | 59.77 | 618 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-spider-all | Same model as sft-spider-all; vote over five candidate queries | 65.09 | 673 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | LoRA fine-tuning with one example row per table in the schema: 5,259 examples, one greedy query | 61.51 | 636 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | Same model as sft-rows1; vote over five candidate queries | 65.76 | 680 / 1034 |
| vote of sft-rows1-vote5, sft-spider-all-vote5, sft-spider-vote5 | Vote across the three fine-tuned models' voted answers (15 candidate queries per question in total) | 68.67 | 710 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | Same model as sft-rows1, voting over 15 candidate queries instead of 5 | 67.02 | 693 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-synsql | Fresh LoRA on Spider train plus a SynSQL-2.5M slice (12,215 examples that fit 1,024 tokens, 6,000 synthetic databases), 2 epochs, one greedy query | 61.80 | 639 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-synsql | Same model as sft-synsql; the prompt also lists question words found in the database's text values (value hints), one greedy query | 64.70 | 669 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + sft-rows1 | Same model as sft-rows1 (one example row per table); value hints added to the prompt, one greedy query | 62.28 | 644 / 1034 |
| vote of sft-synsql-hints, sft-rows1-vote15, sft-spider-all-vote5 | Several models: vote across three fine-tuned adapters (sft-synsql with value hints, sft-rows1 with 15 candidates, sft-spider-all with 5), same majority rule as before | 71.18 | 736 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + grpo-synsql | RL pilot: GRPO from sft-synsql on 600 fresh SynSQL questions (only 252 fit 900 prompt tokens), lr 2e-5, value hints, one greedy query | 63.64 | 658 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + grpo-spider | RL pilot: GRPO from sft-synsql on 600 Spider train questions, lr 2e-5, value hints, one greedy query | 64.89 | 671 / 1034 |
| Qwen2.5-Coder-0.5B-Instruct + grpo-spider-lr5 | GRPO from sft-synsql on 2,400 Spider train questions, lr 5e-6, value hints, one greedy query (65.38% at 800, 65.76% at 1,600) | 66.92 | 692 / 1034 |

![Execution accuracy by run](results/chart.svg)

<!-- results:end -->

### How results are measured

- A prediction is correct when it runs and returns the same rows as the reference query, in the same order when the reference query sorts, with columns in any order.
- Each question is scored on its one original Spider database. A wrong query can therefore be counted as correct when both it and the reference return no rows; results.md reports how many correct predictions are of this kind.
- Each row says how its answer was chosen: one greedy query, or a vote. A vote samples several candidate queries, runs them, and keeps the result most candidates agree on. Rows marked "several models" pool the candidates of more than one adapter.
- "Value hints" means the prompt also lists question words that appear in the database's text values, with their table and column. They are found by plain text matching, never from the reference query.
- Generation runs in fixed batches of 16. The batch size affects a few answers: redoing 200 baseline questions one at a time changed the verdict for 3 of them. Differences of about one point between runs are within this noise.
- Choices such as the voting rule were made on Spider dev, so the dev numbers are optimistic. Spider test is split by database into a validation half (1,230 questions, 21 databases) for choosing settings and a final half (917 questions, 19 databases) that is scored once at the end.

### Published results for comparison

| Model | Training | Spider dev EX (%) | Spider test EX (%) | Source |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B | none | 34.5 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B | fine-tuned | 39.2 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B-Instruct | none | 44.07 | 44.31 | [SLM-SQL][slm-sql] |
| SLM-SQL-0.5B | SFT and RL on SynSQL-derived data | 72.08 | 73.50 | [SLM-SQL][slm-sql] |

None of the published rows has been reproduced here, and they are not directly comparable with each other or with this repository's numbers. The SLM-SQL rows use that paper's corrective self-consistency decoding. The model-card rows use the non-instruct base model. This repository uses its own implementation of execution accuracy on the original Spider databases, which is simpler than the official test-suite evaluation.

## Roadmap

- [x] Tooling: linting, tests, a train/dev leakage guard, continuous integration
- [x] Baseline: the unmodified base model, evaluated on Spider dev
- [x] Supervised fine-tuning on the Spider training set: 45.36% to 60.64%
- [x] Improvements: a slice of [SynSQL-2.5M][omnisql], example rows and value hints in the prompt, voting among sampled queries, and reinforcement learning on execution results
- [ ] Final evaluation: validate the voting rules on the Spider test validation half, then score the final setup once on the held-out half
- [ ] Release: model weights on Hugging Face, a demo, and a write-up of the results

## Constraints

- Hardware: one NVIDIA RTX 3060 Ti (8 GB VRAM) and 32 GB of RAM.
- Cloud compute: at most $15 in total.
- Spider dev is used for evaluation only and never for training.

## Repository layout

```
minnow/       library code; imports without a GPU or deep-learning dependencies
tests/        unit tests and structure tests
scripts/      training, evaluation, the dashboard and the local check runner
results/      predictions, verdicts and score summary of each logged run
dashboard/    the local results page
results.md    the log of measured runs, generated from results/
FINDINGS.md   what each experiment showed and the decisions it led to
```

## Training and evaluation

Everything runs on one 8 GB GPU. Every script saves its progress often, so a stopped run continues where it left off when started again with the same command.

### Setup

1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support, then the model dependencies:

   ```bash
   pip install -e ".[model]"
   ```

2. Download [Spider][spider] and unpack it so that `data/spider/dev.json` and `data/spider/database/` exist.

3. Optional: download [SynSQL-2.5M][omnisql] to `data/synsql` and add a slice of it to the Spider folder. Column comments are removed so the schemas look like Spider's.

   ```bash
   python -m minnow.synsql data/synsql data/spider --databases 6000 --per-database 2
   ```

### Evaluate

```bash
python scripts/baseline.py                                    # the untouched base model
python scripts/baseline.py --adapter outputs/sft/adapter      # a trained adapter
```

Useful options: `--value-hints` adds value hints to the prompt, `--samples 15` samples 15 candidates per question and votes, `--sample-rows 1` shows one example row per table, and `--split` scores another question file in the Spider folder. The untouched model takes about 5 minutes on an RTX 3060 Ti and about 3 GB of GPU memory.

### Train

Supervised fine-tuning with LoRA (rank 16, learning rate 2e-4, 16 examples per update, examples up to 1,024 tokens):

```bash
python scripts/train_sft.py --train-files train_spider.json train_synsql.json --epochs 2 --out outputs/sft
```

Reinforcement learning on execution results, starting from a supervised adapter. Each question gets 8 sampled queries; a query that returns the right rows earns 1, one that only runs earns 0.1, and the model is pushed toward the better ones in its group:

```bash
python scripts/train_grpo.py --adapter outputs/sft/adapter --out outputs/grpo --value-hints --learning-rate 5e-6
```

Both check the training data against Spider dev with the [leakage guard](#data-leakage-guard) before they start.

### Vote across runs

```bash
python -m minnow.combine outputs/vote outputs/run-a outputs/run-b outputs/run-c --spider data/spider
python -m minnow.splits data/spider    # writes test_val.json and test_final.json
```

`combine` pools every saved candidate of the given runs and keeps the result most of them agree on. `splits` cuts Spider test into the validation and final halves described [above](#how-results-are-measured); score either with `python scripts/baseline.py --split test_val.json`.

## Results log

No score in this repository is typed by hand. A run writes its predictions, a verdict for every question and a summary. Three small tools turn those files into everything that is reported:

```bash
python -m minnow.outcomes outputs/run --spider data/spider    # judge a run again and count its errors
python -m minnow.compare results/baseline outputs/run         # which questions a change fixed or broke
python -m minnow.runlog add outputs/run --change "..."        # log the run and regenerate the reports
```

`runlog add` copies the run into `results/`, records it in `results/runs.json`, and regenerates `results.md`, the chart and the results table in this README. `python -m minnow.runlog check` fails when any of them is out of date or a run under `results/` is not logged, and it is part of the checks below.

## Dashboard

A local page shows every logged run: scores, kinds of error, accuracy per database, and each question with the model's query next to the reference query.

```bash
python scripts/dashboard.py
```

It serves the repository on `127.0.0.1` only and reads the files under `results/` directly. Question text and reference queries appear when Spider is unpacked in `data/spider`.

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

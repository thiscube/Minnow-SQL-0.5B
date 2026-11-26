# Minnow-SQL-0.5B

[![checks](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml/badge.svg)](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Minnow-SQL-0.5B is a 0.5-billion-parameter model that turns plain English questions into SQLite queries, trained on a single 8 GB gaming GPU.

Big models already write SQL well. I wanted to know how close a tiny one could get, trained at home on my own PC. Minnow starts from [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), which has about 500 million parameters.

My PC:
- **Graphics card:** NVIDIA RTX 3060 Ti (8 GB)
- **Processor:** Intel Core i5-12600KF
- **RAM:** 32 GB
- **System:** Windows 10

You can download the model on [Hugging Face](https://huggingface.co/cubehugging/Minnow-SQL-0.5B).

## Results

Tested on [Spider][spider], a standard SQL benchmark. An answer only counts if the query runs and returns the right rows.

<!-- standing:start -->

- **~2x fewer mistakes.** It now gets 29 of every 100 questions wrong. Before training it got 55 wrong, more than half.
- **1.6x as many right.** It answers 736 of the 1,034 questions correctly, up from 469.
- **77% on questions it had never seen.** SLM-SQL-0.5B, a published model of the same size, reports 73.5%.

|  | Dev | Test A | Test B |
| --- | ---: | ---: | ---: |
| Before training | 45.36% | N/A | N/A |
| My best model | 71.18% | 72.93% | 76.66% |
| Three models voting | **73.79%** | **74.72%** | **78.19%** |

<!-- standing:end -->

Test B is the fairest score. I set those questions aside, chose nothing on them, and scored them once at the very end.

## The goal

I wanted at least 70% on Spider's dev set, with a model this small, on one normal graphics card. The score is execution accuracy: the share of questions where the model's query returns the same rows as the correct one.

I got there.

## How I got there

I measured every step on the same 1,034 questions, so the steps can be compared fairly.

1. **Starting point: 45%.** The model as it comes. Its most common mistake was asking for a column that doesn't exist.
2. **Training on examples: 61%.** I showed it about 6,000 questions from Spider's training set, each with the correct query. I used LoRA, which trains a small add-on and leaves the original model untouched.
3. **Letting it try several times: 67%.** The model writes 15 different queries for each question. I run all of them, and the result most of them agree on wins. Wrong queries tend to be wrong in different ways, while right ones agree.
4. **Telling it what's in the database: 69%.** If the question says "France" and a column holds the value "France", the prompt now points that out. I also added 12,000 machine-written practice questions from [SynSQL-2.5M][omnisql].
5. **Letting three models vote: 73%.** I trained three models in slightly different ways, and the answer most of them give wins. An answer that returns nothing only wins if there's nothing else left.
6. **Fixing queries that fail: 74%.** Sometimes the winning query asks a table for a column that lives in a different table. A small program looks the column up in the database and points the query at the right table. It never sees the correct answer. With this, one model on its own gets 71%, and the three-model vote gets 74%.

Not everything worked. Adding more training data on its own barely helped. A second round of training on questions that need several tables didn't help either, and neither did reinforcement learning (rewarding the model whenever its query returns the right rows). I wrote all of it up, with numbers, in [FINDINGS.md](FINDINGS.md).

## Every run

Here's every run I measured, in order. A script builds the chart and the table from each run's own files, so no score is typed by hand. [results.md](results.md) also breaks down what kinds of mistakes each run made.

<!-- results:start -->

![Execution accuracy by run](results/chart.svg)

<details>
<summary>What each of the 25 runs changed</summary>

| Run | Change | Spider dev EX (%) | Correct |
| --- | --- | ---: | ---: |
| baseline | Baseline: no training, greedy decoding, CREATE TABLE statements as the schema | 45.36 | 469 / 1034 |
| sft-spider | LoRA fine-tuning on Spider train: 6,142 examples of at most 1,024 tokens, 3 epochs | 60.64 | 627 / 1034 |
| sft-spider-vote5 | Same model as sft-spider; five candidate queries per question, keeping the result most candidates agree on | 64.31 | 665 / 1034 |
| sft-spider-all | LoRA fine-tuning on Spider train plus train_others: 7,800 examples, 3 epochs, one greedy query | 59.77 | 618 / 1034 |
| sft-spider-all-vote5 | Same model as sft-spider-all; vote over five candidate queries | 65.09 | 673 / 1034 |
| sft-rows1 | LoRA fine-tuning with one example row per table in the schema: 5,259 examples, one greedy query | 61.51 | 636 / 1034 |
| sft-rows1-vote5 | Same model as sft-rows1; vote over five candidate queries | 65.76 | 680 / 1034 |
| vote-of-three | Vote across the three fine-tuned models' voted answers (15 candidate queries per question in total) | 68.67 | 710 / 1034 |
| sft-rows1-vote15 | Same model as sft-rows1, voting over 15 candidate queries instead of 5 | 67.02 | 693 / 1034 |
| sft-synsql | Fresh LoRA on Spider train plus a SynSQL-2.5M slice (12,215 examples that fit 1,024 tokens, 6,000 synthetic databases), 2 epochs, one greedy query | 61.80 | 639 / 1034 |
| sft-synsql-hints | Same model as sft-synsql; the prompt also lists question words found in the database's text values (value hints), one greedy query | 64.70 | 669 / 1034 |
| sft-rows1-hints | Same model as sft-rows1 (one example row per table); value hints added to the prompt, one greedy query | 62.28 | 644 / 1034 |
| vote-hints-three | Several models: vote across three fine-tuned adapters (sft-synsql with value hints, sft-rows1 with 15 candidates, sft-spider-all with 5), same majority rule as before | 71.18 | 736 / 1034 |
| grpo600-hints | RL pilot: GRPO from sft-synsql on 600 fresh SynSQL questions (only 252 fit 900 prompt tokens), lr 2e-5, value hints, one greedy query | 63.64 | 658 / 1034 |
| grpo-spider600-hints | RL pilot: GRPO from sft-synsql on 600 Spider train questions, lr 2e-5, value hints, one greedy query | 64.89 | 671 / 1034 |
| grpo-lr5-2400-hints | GRPO from sft-synsql on 2,400 Spider train questions, lr 5e-6, value hints, one greedy query (65.38% at 800, 65.76% at 1,600) | 66.92 | 692 / 1034 |
| vote-hints-three-rows | Several models: same three adapters as vote-hints-three, vote over each model's pick with empty and all-NULL results last; rule confirmed on 1,230 unseen Spider test questions (71.14% to 72.20%) | 72.63 | 751 / 1034 |
| sft-hintstrained | Trained with value hints in the prompt (Spider + SynSQL slice, 2 epochs), greedy | 62.48 | 646 / 1034 |
| sft-hintstrained-vote15 | Same hints-trained model, vote over 15 candidates | 68.57 | 709 / 1034 |
| vote-hintstrained-three | Three models vote (hints-trained with 15 candidates, one-row model with 15, all-Spider model with 5), empty results last | 73.31 | 758 / 1034 |
| sft-rows1-shared15 | Same model as sft-rows1-vote15, sampled again with the shared-prompt sampler so that its 15 candidates are saved | 66.73 | 690 / 1034 |
| sft-spider-all-shared5 | Same model as sft-spider-all-vote5, sampled again with the shared-prompt sampler so that its 5 candidates are saved | 63.54 | 657 / 1034 |
| sft-hintstrained-greedy-repaired | One greedy query of the hints-trained model; a query that fails on a missing column is repaired from the database (minnow.repair) | 67.02 | 693 / 1034 |
| sft-hintstrained-repaired | Same 15 candidates as sft-hintstrained-vote15, voted again with empty results last, then repaired where a column was missing | 71.18 | 736 / 1034 |
| vote-three-repaired | Several models: hints-trained (15 candidates), sft-rows1-shared15 and sft-spider-all-shared5, each voted again with empty results last and repaired, then a vote over their picks; 72.82% on the same candidates without these two steps | 73.79 | 763 / 1034 |

</details>

<!-- results:end -->

### How to read the numbers

- **Correct means the query runs and returns the same rows as the correct query.** Row order only matters when the question asks for a sorted answer.
- **A wrong query can get lucky.** Each question is checked on one database. If the right answer is "nothing" and a wrong query also returns nothing, it passes. results.md counts how often that could have happened.
- **One point up or down doesn't mean much.** When I scored 200 questions again with a different batch size, 3 answers flipped. So every run uses the same batch size of 16, and I only call something better when it wins by clearly more than a point.
- **The hints aren't cheating.** They come from matching words in the question against text stored in the database. The correct query is never used.
- **The dev score is a little flattering.** I picked models and settings while looking at dev scores, so dev isn't a fair exam anymore. To fix that, I split Spider's separate test set in two, by database. Test A (1,230 questions) is for choosing settings like the voting rule. Test B (917 questions) was scored once, at the very end, when there was nothing left to choose. The answers behind both are in [heldout/](heldout).

### How it compares to published work

| Model | Training | Spider dev (%) | Spider test (%) | Source |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B | none | 34.5 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B | fine-tuned | 39.2 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B-Instruct | none | 44.07 | 44.31 | [SLM-SQL][slm-sql] |
| SLM-SQL-0.5B | SFT and RL on SynSQL-derived data | 72.08 | 73.50 | [SLM-SQL][slm-sql] |

SLM-SQL is the strongest published result I know of for this exact base model. My numbers land in the same range, but it isn't a clean head-to-head:

- I didn't rerun their models. Their numbers are copied from their papers.
- My best numbers come from three models voting. Theirs come from one model with its own way of picking among several tries. My single model is the "My best model" row in the table at the top.
- Their test score covers all 2,147 Spider test questions. Mine is split into Test A (1,230) and Test B (917).
- I check answers with my own, simpler checker on the original Spider databases. The official test suite is stricter.
- The two model-card rows start from a slightly different base model.

## What's done

- [x] Project checks: tests, code style, and a guard against training on test questions
- [x] Measure the model as it comes: 45.36%
- [x] First fine-tune on Spider's training set: 60.64%
- [x] Improvements: more data, value hints, voting, reinforcement learning
- [x] Final score on Test B, the 917 held-back questions
- [x] Release the model on [Hugging Face](https://huggingface.co/cubehugging/Minnow-SQL-0.5B)
- [ ] A demo and a write-up

## Limits I set myself

- One NVIDIA RTX 3060 Ti with 8 GB of memory, and 32 GB of RAM.
- At most $15 on rented graphics cards. So far I've spent $0.
- Spider dev is only ever used for scoring. The model never trains on it.

## Run it yourself

Everything fits on one 8 GB graphics card. If you stop a training run, running the same command again picks it up where it left off.

### Setup

1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support, then the rest:

   ```bash
   pip install -e ".[model]"
   ```

2. Download [Spider][spider] and unpack it so that `data/spider/dev.json` and `data/spider/database/` exist.

3. Optional: download [SynSQL-2.5M][omnisql] to `data/synsql` and add a slice of it to the Spider folder:

   ```bash
   python -m minnow.synsql data/synsql data/spider --databases 6000 --per-database 2
   ```

### Score a model

```bash
python scripts/baseline.py                                    # the model as it comes
python scripts/baseline.py --adapter outputs/sft/adapter      # a model you trained
```

| Option | What it does |
| --- | --- |
| `--value-hints` | tells the model which words from the question appear in the database |
| `--samples 15` | writes 15 queries per question and lets them vote |
| `--share-prompt` | reads each prompt once for all 15 queries, about 5 times faster |
| `--sample-rows 1` | shows the model one example row of every table |
| `--split test_val.json` | scores a different question file from the Spider folder |

Scoring the base model takes about 5 minutes and 3 GB of GPU memory on an RTX 3060 Ti.

### Train a model

Training on examples (supervised fine-tuning with LoRA):

```bash
python scripts/train_sft.py --train-files train_spider.json train_synsql.json --value-hints --epochs 2 --pack 4096 --gradient-checkpointing --autocast --out outputs/sft
```

The last three options make it about 3 times faster and keep it at 3 GB of GPU memory. Add `--init-adapter outputs/sft/adapter` to keep training a model you already have.

Training by trial and error (reinforcement learning), starting from a model trained as above:

```bash
python scripts/train_grpo.py --adapter outputs/sft/adapter --out outputs/grpo --value-hints --learning-rate 5e-6
```

Here the model writes 8 queries per question. A query that returns the right rows scores 1, one that only runs scores 0.1, and the model gets nudged toward the better ones.

### Let several models vote

```bash
python -m minnow.combine outputs/vote outputs/run-a outputs/run-b outputs/run-c --spider data/spider --prefer-rows
```

This takes the answers of the runs you list and keeps the result most of them agree on. `--prefer-rows` makes an empty result the last choice.

### Fix queries that fail

```bash
python -m minnow.repair outputs/run-repaired outputs/run --spider data/spider
```

This votes again over the run's saved queries, with empty results as the last choice. If the winner still fails because it names a column that isn't there, it tries to reach that column another way: the right table alias, a join to the table that has it, or the closest column name. It only ever looks at the database, never at the correct answer.

### Split the Spider test set

```bash
python -m minnow.splits data/spider
```

This writes `test_val.json` (Test A, for choosing settings) and `test_final.json` (Test B, scored once at the end).

## How the numbers stay honest

It's easy to get a score wrong by accident, so a few things are automatic.

**No score is typed by hand.** Three small tools turn a run's files into everything reported here:

```bash
python -m minnow.outcomes outputs/run --spider data/spider    # judge a run again and count its mistakes
python -m minnow.compare results/baseline outputs/run         # which questions a change fixed or broke
python -m minnow.runlog add outputs/run --change "..."        # log the run and rebuild the reports
```

`runlog add` rebuilds `results.md`, the chart and the score tables in this README. `python -m minnow.runlog check` fails if any of them no longer matches the runs.

**The model can't train on the exam.** `minnow.guard` checks that the training data shares no question and no database with Spider dev. Every training script has to call it before it starts, and a test fails if one doesn't.

```bash
python -m minnow.guard train.json dev.json
```

Both files are JSON lists of objects with `question` and `db_id` fields, the format Spider uses. It exits with 0 when the data is clean and with 3 when it finds an overlap.

## Dashboard

A local page for browsing every logged run: scores, kinds of mistakes, accuracy per database, and each question with the model's query next to the correct one.

```bash
python scripts/dashboard.py
```

It only runs on your own machine (`127.0.0.1`) and reads the files under `results/` directly. Questions and correct queries show up when Spider is unpacked in `data/spider`.

## Working on the code

You need Python 3.12 or later.

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

On Windows, `powershell -File scripts/verify.ps1` runs all of them in one go. The same checks run on GitHub for every push to `main`. They cover:

- code style and linting with [ruff](https://docs.astral.sh/ruff/), with limits on how long and tangled a function can get
- unit tests, which have to cover at least 90% of `minnow/`
- structure tests: every module has a test file, files stay small, and training scripts call the guard
- the results log: every reported number matches the runs' files

### Folders

```
minnow/       the logic; runs without a GPU
tests/        tests
scripts/      training, scoring, the dashboard and the check runner
results/      answers, verdicts and score of every logged run
heldout/      the same for the runs on Spider's test questions
dashboard/    the local results page
results.md    the log of measured runs, built from results/
FINDINGS.md   what each experiment showed and what I decided because of it
```

## License

The code is under the [MIT License](LICENSE). The base model and the datasets come with their own licenses.

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

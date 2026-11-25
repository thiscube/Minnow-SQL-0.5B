# Minnow-SQL-0.5B

[![checks](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml/badge.svg)](https://github.com/thiscube/Minnow-SQL-0.5B/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

A very small language model that turns a plain English question into a database query.

You ask:

> What is the average, minimum, and maximum age of all singers from France?

It looks at the tables in the database and writes:

```sql
SELECT avg(age), min(age), max(age) FROM singer WHERE country = 'France'
```

Big models are good at this. I wanted to see how far a tiny one can get, trained at home on one gaming graphics card. Minnow starts from [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), a model with about 500 million parameters. That is a tiny fraction of the size of the models behind the well-known chat assistants.

## Where it stands

The test is [Spider][spider], a standard benchmark. An answer only counts when the query runs and returns the right rows.

| | Spider dev (1,034 questions) | Questions never used for training or picking models (1,230) |
| --- | ---: | ---: |
| The model before any training | 45.36% | |
| My best single model | 68.57% | 71.14% |
| Three of my models voting on the answer | **73.31%** | **73.25%** |

So training took it from getting less than half right to getting almost three out of four right.

> [!NOTE]
> Still open: one last score on a set of questions I have not looked at, and putting the model on Hugging Face. See [what is left](#what-is-left).

## The goal

Reach at least 70% execution accuracy on the Spider development set, with a model this small, on one consumer graphics card. "Execution accuracy" means the share of questions where the model's query returns the same rows as the correct query.

That goal is reached.

## How I got there

Each step was measured on the same 1,034 questions, so the gains can be compared.

1. **Start: 45%.** The untouched model. Its most common mistake was asking for a column that does not exist.
2. **Teach it with examples: 61%.** I showed it about 6,000 questions with their correct queries from Spider's training set. This is ordinary fine-tuning, done with LoRA, which trains a small add-on and leaves the original model as it is.
3. **Let it try several times: 67%.** The model writes 15 different queries for a question. All of them are run, and the result that most of them agree on wins. Wrong queries tend to be wrong in different ways, and right ones agree.
4. **Tell it what is in the database: 69%.** If the question says "France" and a column contains the value "France", the prompt now says so. I also added 12,000 machine-written practice questions from [SynSQL-2.5M][omnisql].
5. **Let three models vote: 73%.** Three models trained in slightly different ways each give their answer, and the majority wins. An answer that returns nothing only wins when no other answer is left.

Not everything worked. More training data alone did not help much, and neither did reinforcement learning (rewarding the model when its query returns the right rows). Both are written up, with numbers, in [FINDINGS.md](FINDINGS.md).

## Results

Every run I measured, in order. The chart and the table below are made by a script from the runs' own files, so nobody types a score by hand. [results.md](results.md) also shows what kinds of mistakes each run made.

<!-- results:start -->

![Execution accuracy by run](results/chart.svg)

<details>
<summary>What each of the 20 runs changed</summary>

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

</details>

<!-- results:end -->

### How to read the numbers

- **Correct** means the query runs and returns the same rows as the reference query. The order of rows only matters when the question asks for a sorted answer.
- **A wrong query can get lucky.** Each question is checked on one database. If the right answer is "nothing" and a wrong query also returns nothing, it passes. results.md counts how often that could have happened.
- **One point up or down means little.** Scoring 200 questions again with a different batch size changed 3 verdicts. Every run uses the same batch size of 16, and I only call something better when it wins by clearly more than a point.
- **The hints are not cheating.** They come from matching words of the question against the text in the database. The correct query is never used.
- **The dev score flatters a little.** I picked models and settings while looking at dev scores, so dev is no longer a fair exam. To fix that, I split Spider's separate test set in two by database. One half (1,230 questions) is for choosing settings such as the voting rule. The other half (917 questions) is scored once, at the very end.

### How it compares to published work

| Model | Training | Spider dev EX (%) | Spider test EX (%) | Source |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B | none | 34.5 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B | fine-tuned | 39.2 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B-Instruct | none | 44.07 | 44.31 | [SLM-SQL][slm-sql] |
| SLM-SQL-0.5B | SFT and RL on SynSQL-derived data | 72.08 | 73.50 | [SLM-SQL][slm-sql] |

EX is short for execution accuracy. SLM-SQL is the strongest published result I know of for this exact base model. My numbers land in the same range, but this is not a clean head-to-head:

- I did not rerun their models. Their rows are copied from their papers.
- My 73% comes from three models voting. Theirs is one model with its own way of picking among several tries.
- I check answers with my own, simpler checker on the original Spider databases. The official test suite is stricter.
- The two model-card rows start from a slightly different base model.

## What is left

- [x] Project checks: tests, code style, and a guard against training on test questions
- [x] Measure the untouched model: 45.36%
- [x] First fine-tune on Spider's training set: 60.64%
- [x] Improvements: more data, value hints, voting, reinforcement learning
- [ ] Final score on the 917 held-back questions
- [ ] Release: the model on Hugging Face, a demo and a write-up

## Limits I set myself

- One NVIDIA RTX 3060 Ti with 8 GB of memory, and 32 GB of RAM.
- At most $15 for rented graphics cards. Spent so far: $0.
- Spider dev is only ever used for scoring. The model never trains on it.

## Run it yourself

Everything fits on one 8 GB graphics card. If you stop a training run, the same command picks it up where it stopped.

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
python scripts/baseline.py                                    # the untouched base model
python scripts/baseline.py --adapter outputs/sft/adapter      # a model you trained
```

| Option | What it does |
| --- | --- |
| `--value-hints` | tells the model which question words appear in the database |
| `--samples 15` | writes 15 queries per question and lets them vote |
| `--sample-rows 1` | shows the model one example row of every table |
| `--split test_val.json` | scores a different question file from the Spider folder |

Scoring the untouched model takes about 5 minutes and 3 GB of GPU memory on an RTX 3060 Ti.

### Train a model

Learning from examples (supervised fine-tuning with LoRA):

```bash
python scripts/train_sft.py --train-files train_spider.json train_synsql.json --value-hints --epochs 2 --out outputs/sft
```

Learning from trial and error (reinforcement learning), starting from a model trained as above:

```bash
python scripts/train_grpo.py --adapter outputs/sft/adapter --out outputs/grpo --value-hints --learning-rate 5e-6
```

Here the model writes 8 queries per question. A query that returns the right rows scores 1, one that only runs scores 0.1, and the model is nudged toward the better ones.

### Let several models vote

```bash
python -m minnow.combine outputs/vote outputs/run-a outputs/run-b outputs/run-c --spider data/spider --prefer-rows
```

Takes the answers of the runs you list and keeps the result most of them agree on. `--prefer-rows` makes an empty result the last choice.

### Split the Spider test set

```bash
python -m minnow.splits data/spider
```

Writes `test_val.json` (for choosing settings) and `test_final.json` (scored once at the end).

## How the numbers stay honest

A score is easy to get wrong by accident, so a few things are automatic.

**No score is typed by hand.** Three small tools turn a run's files into everything reported here:

```bash
python -m minnow.outcomes outputs/run --spider data/spider    # judge a run again and count its mistakes
python -m minnow.compare results/baseline outputs/run         # which questions a change fixed or broke
python -m minnow.runlog add outputs/run --change "..."        # log the run and rebuild the reports
```

`runlog add` rebuilds `results.md`, the chart and the results table in this README. `python -m minnow.runlog check` fails if any of them no longer matches the runs.

**The model cannot train on the exam.** `minnow.guard` checks that the training data shares no question and no database with Spider dev. Every training script has to call it before it starts, and a test fails if one does not.

```bash
python -m minnow.guard train.json dev.json
```

Both files are JSON lists of objects with `question` and `db_id` fields, the format Spider uses. It exits with 0 when the data is clean and with 3 when it finds an overlap.

## Dashboard

A local page to browse every logged run: scores, kinds of mistakes, accuracy per database, and each question with the model's query next to the correct one.

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

- code style and linting with [ruff](https://docs.astral.sh/ruff/), with limits on how long and tangled a function may get
- unit tests, which have to cover at least 90% of `minnow/`
- structure tests: every module has a test file, files stay small, and training scripts call the guard
- the results log: every reported number matches the runs' files

### Folders

```
minnow/       the logic; runs without a GPU
tests/        tests
scripts/      training, scoring, the dashboard and the check runner
results/      answers, verdicts and score of every logged run
dashboard/    the local results page
results.md    the log of measured runs, built from results/
FINDINGS.md   what each experiment showed and what I decided because of it
```

## License

The code is released under the [MIT License](LICENSE). The base model and the datasets come with their own licenses.

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

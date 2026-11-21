# Minnow-SQL-0.5B

[![checks](https://github.com/thiscube/minnow-sql/actions/workflows/checks.yml/badge.svg)](https://github.com/thiscube/minnow-sql/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Minnow-SQL-0.5B is a text-to-SQL model in the 0.5B-parameter class, fine-tuned from [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct). Given a natural-language question and a database schema, it produces a SQL query.

> [!NOTE]
> This project is in progress. Training has not started and no model has been released. The tooling for evaluation and data integrity is in place; see the [roadmap](#roadmap).

## Goal

Reach at least 70% execution accuracy on the [Spider][spider] development set (1,034 questions) with a 0.5B-parameter model trained on a single consumer GPU. Execution accuracy counts a prediction as correct when the generated query runs and returns the same rows as the reference query.

## Reference results

Published results for models of this size. None of them has been reproduced here yet.

| Model | Training | Spider dev EX (%) | Spider test EX (%) | Source |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B | none | 34.5 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B | fine-tuned | 39.2 | | [model card][thesis] |
| Qwen2.5-Coder-0.5B-Instruct | none | 44.07 | 44.31 | [SLM-SQL][slm-sql] |
| SLM-SQL-0.5B | SFT and RL on SynSQL-derived data | 72.08 | 73.50 | [SLM-SQL][slm-sql] |
| Minnow-SQL-0.5B | | not yet measured | | this repository |

The SLM-SQL rows use that paper's corrective self-consistency decoding. The first two rows come from a different evaluation setup and the non-instruct base model, so the rows are not directly comparable. This project will measure its own baseline with its own evaluation code before any training.

## Roadmap

- [x] Tooling: linting, tests, a train/dev leakage guard, continuous integration
- [ ] Baseline: evaluate the unmodified base model on Spider dev
- [ ] Supervised fine-tuning on the Spider training set (about 7,000 examples)
- [ ] Improvements: a subset of [SynSQL-2.5M][omnisql], LoRA configuration, schema serialization, and execution-guided selection among sampled queries
- [ ] Release: model weights on Hugging Face, a demo, and a write-up of the results

Each measured change will be recorded in `results.md` with its date, what changed, and the score.

## Constraints

- Hardware: one NVIDIA RTX 3060 Ti (8 GB VRAM) and 32 GB of RAM.
- Cloud compute: at most $15 in total.
- Spider dev is used for evaluation only and never for training.

## Repository layout

```
minnow/     library code; imports without a GPU or deep-learning dependencies
tests/      unit tests and structure tests
scripts/    entry points and the local check runner
```

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

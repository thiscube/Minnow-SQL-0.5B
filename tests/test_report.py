from minnow.outcomes import Breakdown
from minnow.report import Run, readme_block, results_markdown

BASELINE = Run(
    name="baseline",
    date="2026-11-21",
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    change="Baseline: no training",
    correct=469,
    total=1034,
    execution_accuracy=45.36,
    generation_seconds=291,
)
TUNED = Run(
    name="sft",
    date="2026-12-01",
    model="minnow-sft",
    change="Fine-tuned | one epoch",
    correct=700,
    total=1034,
    execution_accuracy=67.7,
    generation_seconds=1,
)
COUNTS = {
    "baseline": Breakdown(
        total=1034,
        correct=469,
        wrong_rows=176,
        did_not_run=389,
        error_classes=(
            ("no such column", 356),
            ("no such table", 15),
            ("misuse of aggregate", 5),
            ("no such function", 4),
        ),
        correct_on_empty=30,
    ),
    "sft": Breakdown(
        total=1034,
        correct=700,
        wrong_rows=333,
        did_not_run=1,
        error_classes=(("syntax error", 1),),
        correct_on_empty=0,
    ),
}
RESULTS = """\
# Results

Execution accuracy (EX) on the Spider development set, 1,034 questions. A prediction is correct \
when it runs and returns the same rows as the reference query. Scoring is done by \
[`minnow/execution.py`](minnow/execution.py) on the original Spider databases.

| Date | Model | Change | Dev EX (%) | Correct | Files |
| --- | --- | --- | ---: | ---: | --- |
| 2026-11-21 | Qwen2.5-Coder-0.5B-Instruct | Baseline: no training | 45.36 | 469 / 1034 | \
[results/baseline](results/baseline) |
| 2026-12-01 | minnow-sft | Fine-tuned \\| one epoch | 67.70 | 700 / 1034 | \
[results/sft](results/sft) |

## Notes

### 2026-11-21, baseline

- 389 predictions did not run. The most common errors: `no such column` (356), \
`no such table` (15), `misuse of aggregate` (5).
- 176 predictions ran and returned different rows.
- 30 of the 469 correct predictions matched a reference query that returns no rows. With a \
single database per question, such a match cannot be told apart from a wrong query that also \
returns nothing.
- Generating all 1,034 predictions took 291 seconds.

### 2026-12-01, sft

- 1 prediction did not run. The most common errors: `syntax error` (1).
- 333 predictions ran and returned different rows.
- Generating all 1,034 predictions took 1 second.
"""
README_BLOCK = """\
![Execution accuracy by run](results/chart.svg)

<details>
<summary>What each of the 2 runs changed</summary>

| Run | Change | Spider dev EX (%) | Correct |
| --- | --- | ---: | ---: |
| baseline | Baseline: no training | 45.36 | 469 / 1034 |
| sft | Fine-tuned \\| one epoch | 67.70 | 700 / 1034 |

</details>
"""


def test_results_log_has_a_row_and_notes_for_every_run_in_order():
    assert results_markdown([BASELINE, TUNED], COUNTS) == RESULTS


def test_run_in_which_everything_ran_has_no_error_list():
    counts = {"sft": Breakdown(1034, 700, 334, 0, (), 0)}

    text = results_markdown([TUNED], counts)

    assert "- 0 predictions did not run.\n" in text
    assert "- 334 predictions ran and returned different rows.\n" in text
    assert "most common errors" not in text


def test_results_log_without_runs_is_only_the_frame():
    text = results_markdown([], {})

    assert text.startswith("# Results\n\nExecution accuracy (EX)")
    assert text.endswith("| --- | --- | --- | ---: | ---: | --- |\n\n## Notes\n")


def test_readme_block_is_the_chart_and_a_folded_table_of_the_runs():
    assert readme_block([BASELINE, TUNED]) == README_BLOCK


def test_text_only_depends_on_the_runs():
    first = results_markdown([BASELINE, TUNED], COUNTS) + readme_block([BASELINE, TUNED])
    second = results_markdown([BASELINE, TUNED], COUNTS) + readme_block([BASELINE, TUNED])

    assert first == second
    assert "\r" not in first


def test_unknown_generation_time_is_left_out_of_the_notes():
    untimed = Run("baseline", "2026-11-21", "Qwen/Tiny", "Baseline", 469, 1034, 45.36, None)

    text = results_markdown([untimed], COUNTS)

    assert "took" not in text
    assert "176 predictions ran and returned different rows." in text


def test_intro_gives_the_number_of_questions_of_the_runs():
    assert "development set, 1,034 questions." in results_markdown([BASELINE], COUNTS)
    assert "development set. A prediction" in results_markdown([], {})

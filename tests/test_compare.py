import pytest

from minnow.compare import Comparison, compare, format_comparison, main
from minnow.outcomes import Outcome, save_outcomes


def outcomes(verdicts):
    return [Outcome(index, verdict, "", 1) for index, verdict in enumerate(verdicts)]


BEFORE = outcomes(["correct", "correct", "wrong_rows", "error", "error", "correct"])
AFTER = outcomes(["correct", "wrong_rows", "correct", "correct", "wrong_rows", "error"])


@pytest.fixture
def run_folders(tmp_path):
    folders = (tmp_path / "before", tmp_path / "after")
    for folder, judged in zip(folders, (BEFORE, AFTER), strict=True):
        folder.mkdir()
        save_outcomes(folder / "outcomes.jsonl", judged)
    return folders


def test_counts_what_stayed_what_was_fixed_and_what_was_broken():
    result = compare(BEFORE, AFTER)

    assert result == Comparison(both_correct=1, both_wrong=1, fixed=(2, 3), broken=(1, 5))
    assert result.net_change == 0


def test_fixed_and_broken_swap_when_the_runs_swap():
    result = compare(AFTER, BEFORE)

    assert result.fixed == (1, 5)
    assert result.broken == (2, 3)


def test_net_change_is_fixed_minus_broken():
    assert (
        compare(outcomes(["error", "error", "correct"]), outcomes(["correct"] * 3)).net_change == 2
    )
    assert compare(outcomes(["correct"] * 3), outcomes(["error"] * 3)).net_change == -3


def test_order_of_the_outcomes_does_not_matter():
    assert compare(reversed(BEFORE), AFTER) == compare(BEFORE, AFTER)


def test_runs_over_different_questions_cannot_be_compared():
    with pytest.raises(ValueError, match="do not cover the same questions: 1 differ"):
        compare(BEFORE, AFTER[:-1])
    with pytest.raises(ValueError, match="do not cover the same questions: 2 differ"):
        compare(BEFORE[1:], AFTER[:-1])


def test_comparison_is_laid_out_as_a_table_with_the_indexes():
    assert format_comparison(compare(BEFORE, AFTER)).splitlines() == [
        "both correct      1",
        "both wrong        1",
        "fixed             2",
        "broken            2",
        "net change       +0",
        "fixed: 2, 3",
        "broken: 1, 5",
    ]


def test_only_the_first_20_indexes_are_listed():
    result = compare(outcomes(["error"] * 25), outcomes(["correct"] * 25))

    lines = format_comparison(result).splitlines()

    assert lines[4] == "net change      +25"
    assert lines[5] == "fixed: " + ", ".join(str(index) for index in range(20)) + " and 5 more"
    assert lines[6] == "broken: none"


def test_command_line_prints_the_comparison(run_folders, capsys):
    before, after = run_folders

    assert main([str(before), str(after)]) == 0
    assert capsys.readouterr().out == format_comparison(compare(BEFORE, AFTER)) + "\n"


def test_command_line_refuses_a_run_without_outcomes(run_folders, capsys):
    before, after = run_folders
    (after / "outcomes.jsonl").unlink()

    with pytest.raises(SystemExit) as raised:
        main([str(before), str(after)])

    assert raised.value.code == 2
    assert "outcomes.jsonl does not exist: run minnow.outcomes first." in capsys.readouterr().err


def test_command_line_refuses_runs_over_different_questions(run_folders, capsys):
    before, after = run_folders
    save_outcomes(after / "outcomes.jsonl", AFTER[:-1])

    with pytest.raises(SystemExit) as raised:
        main([str(before), str(after)])

    assert raised.value.code == 2
    assert "do not cover the same questions" in capsys.readouterr().err

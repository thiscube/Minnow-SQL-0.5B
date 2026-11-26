import json
from dataclasses import replace
from datetime import date

import pytest

from minnow.evaluate import append_prediction
from minnow.outcomes import Outcome, breakdown, load_outcomes, save_outcomes
from minnow.report import Run, readme_block, results_markdown
from minnow.runlog import (
    PUBLISHED_ACCURACY,
    TARGET_ACCURACY,
    RunLogError,
    add,
    chart_svg,
    check,
    load_runs,
    main,
    render,
    replace_block,
)

README = "# Demo\n\nIntro.\n\n<!-- results:start -->\nold table\n<!-- results:end -->\n\n## More\n"
GENERATED = ["results.md", "results/chart.svg", "README.md"]
BASELINE = Run("baseline", "2026-11-21", "Qwen/Tiny-Instruct", "Baseline", 2, 4, 50.0, 12)


def make_run(folder, verdicts=("correct", "correct", "wrong_rows", "error"), **summary):
    """A run folder as the baseline script and minnow.outcomes leave it."""
    folder.mkdir(parents=True)
    outcomes = [
        Outcome(index, verdict, "no such column" if verdict == "error" else "", 1)
        for index, verdict in enumerate(verdicts)
    ]
    save_outcomes(folder / "outcomes.jsonl", outcomes)
    for outcome in outcomes:
        append_prediction(folder / "predictions.jsonl", outcome.index, "SELECT 1")
    written = {
        "model": "Qwen/Tiny-Instruct",
        "correct": verdicts.count("correct"),
        "total": len(verdicts),
        "execution_accuracy": round(100 * verdicts.count("correct") / len(verdicts), 2),
        "generation_seconds": 12,
        **summary,
    }
    (folder / "summary.json").write_text(json.dumps(written), encoding="utf-8")
    return folder


def unsigned(run):
    """The run without its fingerprint, to compare with a record written by hand."""
    return replace(run, fingerprint="")


def edit_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "README.md").write_text(README, encoding="utf-8", newline="\n")
    return tmp_path


@pytest.fixture
def logged(repo):
    """A repository with one logged run and an up-to-date results log."""
    add(repo, make_run(repo / "results" / "baseline"), "Baseline", "2026-11-21")
    return repo


def test_reference_values_are_the_target_and_the_published_result():
    chart = chart_svg([BASELINE])

    assert (TARGET_ACCURACY, PUBLISHED_ACCURACY) == (70.0, 72.08)
    assert ">target 70%<" in chart
    assert ">SLM-SQL-0.5B, published 72.08%<" in chart
    assert ">50.00%<" in chart
    assert ">baseline<" in chart


def test_no_runs_file_means_no_runs(repo):
    assert load_runs(repo) == []


def test_add_records_the_run_and_writes_the_whole_log(repo):
    run = add(repo, make_run(repo / "results" / "baseline"), "Baseline", "2026-11-21")

    counts = {"baseline": breakdown(load_outcomes(repo / "results/baseline/outcomes.jsonl"))}
    assert unsigned(run) == BASELINE
    assert len(run.fingerprint) == 64
    assert load_runs(repo) == [run]
    assert json.loads((repo / "results/runs.json").read_text(encoding="utf-8")) == [
        {
            "name": "baseline",
            "date": "2026-11-21",
            "model": "Qwen/Tiny-Instruct",
            "change": "Baseline",
            "correct": 2,
            "total": 4,
            "execution_accuracy": 50.0,
            "generation_seconds": 12,
            "fingerprint": run.fingerprint,
        }
    ]
    assert (repo / "results.md").read_text(encoding="utf-8") == results_markdown([run], counts)
    assert (repo / "results/chart.svg").read_text(encoding="utf-8") == chart_svg([run])
    assert (repo / "README.md").read_text(encoding="utf-8") == (
        "# Demo\n\nIntro.\n\n<!-- results:start -->\n\n"
        + readme_block([run])
        + "\n<!-- results:end -->\n\n## More\n"
    )
    assert check(repo) == []


def test_add_copies_a_run_from_outside_the_results_folder(repo):
    outside = make_run(repo / "outputs" / "sft")

    add(repo, outside, "Fine-tuned", "2026-12-01")

    for name in ("predictions.jsonl", "summary.json", "outcomes.jsonl"):
        assert (repo / "results/sft" / name).read_bytes() == (outside / name).read_bytes()
    assert [run.name for run in load_runs(repo)] == ["sft"]
    assert check(repo) == []


def test_a_second_run_with_a_logged_name_is_refused_and_changes_nothing(logged):
    before = (logged / "results/baseline/summary.json").read_bytes()
    other = make_run(logged / "outputs" / "baseline", verdicts=("correct",) * 4)

    with pytest.raises(RunLogError, match=r"A run named baseline is already logged\."):
        add(logged, other, "Another experiment", "2026-12-01")

    assert (logged / "results/baseline/summary.json").read_bytes() == before
    assert check(logged) == []


def test_adding_a_run_again_with_replace_updates_its_record_in_place(logged):
    add(logged, make_run(logged / "outputs" / "sft"), "Fine-tuned", "2026-12-01")
    again = logged / "results" / "baseline"

    add(logged, again, "Baseline,\n  measured again", "2026-12-02", replace=True)

    runs = load_runs(logged)
    assert [(run.name, run.date) for run in runs] == [
        ("baseline", "2026-12-02"),
        ("sft", "2026-12-01"),
    ]
    assert runs[0].change == "Baseline, measured again"
    assert "### 2026-12-02, baseline" in (logged / "results.md").read_text(encoding="utf-8")
    assert check(logged) == []


def test_add_refuses_a_run_that_has_not_been_judged(repo):
    folder = make_run(repo / "results" / "baseline")
    (folder / "outcomes.jsonl").unlink()

    with pytest.raises(RunLogError, match=r"outcomes\.jsonl does not exist\. Run python -m minnow"):
        add(repo, folder, "Baseline", "2026-11-21")

    assert not (repo / "results/runs.json").exists()
    assert not (repo / "results.md").exists()


@pytest.mark.parametrize("missing", ["predictions.jsonl", "summary.json"])
def test_add_refuses_a_run_folder_that_is_incomplete(repo, missing):
    folder = make_run(repo / "outputs" / "sft")
    (folder / missing).unlink()

    with pytest.raises(RunLogError, match=rf"{missing} does not exist\.$"):
        add(repo, folder, "Fine-tuned", "2026-12-01")

    assert not (repo / "results").exists()


@pytest.mark.parametrize("claimed", [{"correct": 3}, {"total": 5}])
def test_add_refuses_a_summary_that_disagrees_with_the_outcomes(repo, claimed):
    folder = make_run(repo / "outputs" / "sft", **claimed)
    says = {"correct": 2, "total": 4, **claimed}

    with pytest.raises(RunLogError) as raised:
        add(repo, folder, "Fine-tuned", "2026-12-01")

    assert f"summary.json says {says['correct']} of {says['total']} correct" in str(raised.value)
    assert "outcomes.jsonl has 2 of 4" in str(raised.value)
    assert not (repo / "results").exists()


def test_add_changes_no_log_file_when_the_readme_has_no_markers(repo):
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")

    with pytest.raises(RunLogError, match=r"README\.md needs the lines <!-- results:start -->"):
        add(repo, make_run(repo / "results" / "baseline"), "Baseline", "2026-11-21")

    assert not (repo / "results/runs.json").exists()
    assert not (repo / "results.md").exists()
    assert (repo / "README.md").read_text(encoding="utf-8") == "# Demo\n"


@pytest.mark.parametrize(
    "readme",
    ["no markers", "<!-- results:start --> only", "<!-- results:end --> x <!-- results:start -->"],
)
def test_block_needs_both_markers_in_order(readme):
    with pytest.raises(RunLogError, match=r"README\.md needs the lines"):
        replace_block(readme, "table\n")


def test_block_replaces_only_what_is_between_the_markers():
    assert replace_block(README, "table\n") == (
        "# Demo\n\nIntro.\n\n<!-- results:start -->\n\ntable\n\n<!-- results:end -->\n\n## More\n"
    )


def describe_score_table(root):
    """Add a heldout/standing.json that names the logged baseline run."""
    table = {
        "columns": [{"title": "Spider dev", "folder": "results"}],
        "rows": [{"label": "Before training", "runs": ["baseline"]}],
    }
    (root / "heldout").mkdir()
    (root / "heldout/standing.json").write_text(json.dumps(table), encoding="utf-8")


def test_render_fills_the_score_table_when_the_repository_describes_one(logged):
    describe_score_table(logged)
    path = logged / "README.md"
    markers = "\n<!-- standing:start -->\nold\n<!-- standing:end -->\n"
    path.write_text(path.read_text(encoding="utf-8") + markers, encoding="utf-8", newline="\n")

    stale = check(logged)
    render(logged)

    assert stale == ["README.md is out of date: run python -m minnow.runlog render"]
    assert path.read_text(encoding="utf-8").endswith(
        "<!-- standing:start -->\n\n"
        "|  | Spider dev |\n| --- | ---: |\n| Before training | 50.00% |\n"
        "\n<!-- standing:end -->\n"
    )
    assert check(logged) == []


def test_score_table_needs_its_own_markers_in_the_readme(logged):
    describe_score_table(logged)

    (problem,) = check(logged)

    assert problem.startswith("README.md needs the lines <!-- standing:start --> and")


def test_render_restores_every_generated_file_byte_for_byte(logged):
    before = {name: (logged / name).read_bytes() for name in GENERATED}
    (logged / "results.md").unlink()
    (logged / "results/chart.svg").write_text("<svg/>", encoding="utf-8")
    (logged / "README.md").write_bytes(before["README.md"].replace(b"50.00", b"99.99"))

    render(logged)

    assert {name: (logged / name).read_bytes() for name in GENERATED} == before
    assert all(b"\r" not in content for content in before.values())


def test_render_writes_nothing_when_the_readme_has_no_markers(logged):
    (logged / "README.md").write_text("# Demo\n", encoding="utf-8")
    (logged / "results.md").unlink()

    with pytest.raises(RunLogError, match=r"README\.md needs the lines"):
        render(logged)

    assert not (logged / "results.md").exists()


def test_fresh_log_is_up_to_date(logged):
    assert check(logged) == []


@pytest.mark.parametrize("name", GENERATED)
def test_check_names_the_file_that_is_out_of_date(logged, name):
    path = logged / name
    path.write_text(path.read_text(encoding="utf-8").replace("50.00", "99.99"), encoding="utf-8")

    assert check(logged) == [f"{name} is out of date: run python -m minnow.runlog render"]


def test_check_notices_a_missing_generated_file(logged):
    (logged / "results/chart.svg").unlink()

    assert check(logged) == ["results/chart.svg is out of date: run python -m minnow.runlog render"]


def test_check_ignores_readme_changes_outside_the_block(logged):
    path = logged / "README.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nNew section.\n", encoding="utf-8")

    assert check(logged) == []


def test_check_accepts_files_checked_out_with_windows_line_endings(logged):
    for name in GENERATED:
        path = logged / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert check(logged) == []


def test_check_notices_a_run_folder_that_was_never_logged(logged):
    make_run(logged / "results" / "sft")

    assert check(logged) == [
        "results/sft has a summary.json but is not in results/runs.json: "
        "log it with python -m minnow.runlog add"
    ]


def test_check_notices_a_logged_run_without_its_folder(logged):
    (logged / "results/baseline/summary.json").unlink()

    assert check(logged) == [
        "results/runs.json lists baseline, but results/baseline/summary.json does not exist"
    ]


def test_check_notices_a_logged_run_without_its_outcomes(logged):
    (logged / "results/baseline/outcomes.jsonl").unlink()

    (problem,) = check(logged)

    assert problem.endswith(
        "results/baseline/outcomes.jsonl does not exist. Run python -m minnow.outcomes on it first."
    )


@pytest.mark.parametrize(
    ("field", "value"), [("correct", 3), ("total", 5), ("execution_accuracy", 99.9)]
)
def test_check_notices_numbers_in_the_log_that_the_outcomes_do_not_back(logged, field, value):
    edit_json(logged / "results/runs.json", lambda records: records[0].update({field: value}))

    (problem,) = check(logged)

    assert problem.startswith("The log gives baseline ")
    assert problem.endswith("but its outcomes.jsonl has 2 of 4 (50.0%).")
    with pytest.raises(RunLogError, match=r"The log gives baseline"):
        render(logged)


@pytest.mark.parametrize(("field", "value"), [("model", "Other/Model"), ("generation_seconds", 1)])
def test_check_notices_a_log_entry_edited_by_hand(logged, field, value):
    edit_json(logged / "results/runs.json", lambda records: records[0].update({field: value}))

    assert check(logged) == ["The log and summary.json disagree about baseline."]


def test_check_notices_a_score_edited_by_hand_in_the_summary(logged):
    summary = logged / "results/baseline/summary.json"
    edit_json(summary, lambda data: data.update({"execution_accuracy": 55.0}))

    assert check(logged) == ["The log and summary.json disagree about baseline."]


def test_check_notices_predictions_replaced_after_logging(logged):
    append_prediction(logged / "results/baseline/predictions.jsonl", 0, "SELECT 2")

    (problem,) = check(logged)

    assert problem.startswith("The files of baseline changed after it was logged.")


def test_check_notices_the_same_run_name_twice(logged):
    edit_json(logged / "results/runs.json", lambda records: records.append(dict(records[0])))

    assert check(logged) == ["results/runs.json lists the same run name twice."]


def test_runs_must_cover_the_same_number_of_questions(logged):
    longer = make_run(logged / "outputs" / "sft", verdicts=("correct",) * 5)

    with pytest.raises(RunLogError, match="do not all cover the same questions"):
        add(logged, longer, "Fine-tuned", "2026-12-01")

    assert [run.name for run in load_runs(logged)] == ["baseline"]


def test_check_notices_outcomes_that_were_emptied(logged):
    (logged / "results/baseline/outcomes.jsonl").write_text("", encoding="utf-8")

    (problem,) = check(logged)

    assert problem == "outcomes.jsonl does not hold exactly one outcome per question, in order."


def test_check_reports_a_readme_without_markers(logged):
    (logged / "README.md").write_text("# Demo\n", encoding="utf-8")

    (problem,) = check(logged)

    assert problem.startswith("README.md needs the lines <!-- results:start -->")


def test_command_line_adds_checks_and_renders(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    make_run(repo / "results" / "baseline")
    arguments = ["add", "results/baseline", "--change", "Baseline", "--date", "2026-11-21"]

    assert main(arguments) == 0
    assert capsys.readouterr().out == "Logged baseline: 50.00% (2 of 4).\n"
    assert [unsigned(run) for run in load_runs(repo)] == [BASELINE]
    assert main([*arguments, "--replace"]) == 0
    capsys.readouterr()
    assert main(["check"]) == 0
    assert capsys.readouterr().out == "The results log is up to date.\n"

    (repo / "results.md").write_text("edited by hand\n", encoding="utf-8")

    assert main(["check"]) == 1
    assert capsys.readouterr().out == (
        "results.md is out of date: run python -m minnow.runlog render\n"
    )
    assert main(["render"]) == 0
    assert "Rendered results.md" in capsys.readouterr().out
    assert main(["check"]) == 0


def test_command_line_dates_a_run_today_unless_told_otherwise(repo, monkeypatch):
    monkeypatch.chdir(repo)
    make_run(repo / "results" / "baseline")
    earliest = date.today().isoformat()

    assert main(["add", "results/baseline", "--change", "Baseline"]) == 0

    assert earliest <= load_runs(repo)[0].date <= date.today().isoformat()


def test_command_line_exits_with_2_when_it_cannot_do_its_work(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")

    assert main(["render"]) == 2
    assert "README.md needs the lines <!-- results:start -->" in capsys.readouterr().err
    assert main(["add", "outputs/missing", "--change", "Nothing"]) == 2
    assert "predictions.jsonl does not exist." in capsys.readouterr().err


def test_command_line_refuses_a_date_that_is_not_a_date(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit) as raised:
        main(["add", "results/baseline", "--change", "Baseline", "--date", "yesterday"])

    assert raised.value.code == 2
    assert "--date" in capsys.readouterr().err

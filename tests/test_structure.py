"""Rules about the shape of the codebase, checked automatically."""

import ast
import os
from pathlib import Path

from minnow.runlog import PUBLISHED_ACCURACY, TARGET_ACCURACY

ROOT = Path(__file__).resolve().parent.parent
# folders that hold tests, downloads or output, not our source code
SKIPPED_FOLDERS = {
    "tests", "build", "dist", "venv", "__pycache__",
    "data", "checkpoints", "outputs", "models", "wandb", "runs",
}  # fmt: skip
MAX_LINES = 300
MAX_SCRIPT_LINES = 150
TRAINING_CALLS = {"train", "backward", "fit"}


def is_skipped(folder: str) -> bool:
    return folder in SKIPPED_FOLDERS or folder.startswith(".") or folder.endswith(".egg-info")


def source_files() -> list[Path]:
    """Every Python file in the repo that is our own source code."""
    files = []
    for folder, subfolders, names in os.walk(ROOT):
        subfolders[:] = [name for name in subfolders if not is_skipped(name)]
        files += [Path(folder) / name for name in names if name.endswith(".py")]
    return sorted(files)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def called_names(source: str) -> set[str]:
    """Names of everything the source calls: `f()` gives "f", `a.b()` gives "b"."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return names


def starts_training(filename: str, source: str) -> bool:
    return "train" in filename.lower() or bool(called_names(source) & TRAINING_CALLS)


def trains_without_guard(filename: str, source: str) -> bool:
    return starts_training(filename, source) and "assert_no_leak" not in called_names(source)


def test_file_with_train_in_its_name_needs_the_guard():
    assert trains_without_guard("train_sft.py", "print('hello')")
    assert trains_without_guard("sft_Train.py", "print('hello')")


def test_trainer_call_needs_the_guard_whatever_the_file_is_called():
    assert trains_without_guard("finetune.py", "trainer.train()")
    assert trains_without_guard("finetune.py", "model.fit(data)")


def test_manual_training_loop_needs_the_guard():
    assert trains_without_guard("loop.py", "loss.backward()")


def test_importing_the_guard_without_calling_it_is_not_enough():
    source = "from minnow.guard import assert_no_leak\ntrainer.train()"
    assert trains_without_guard("finetune.py", source)


def test_training_file_that_calls_the_guard_is_fine():
    source = "assert_no_leak(train, dev)\ntrainer.train()"
    assert not trains_without_guard("train_sft.py", source)


def test_file_that_does_not_train_needs_no_guard():
    assert not trains_without_guard("evaluate.py", "model.eval()\nmodel.generate()")


def test_source_files_cover_the_package_and_skip_everything_else():
    found = {path.relative_to(ROOT).as_posix() for path in source_files()}

    assert "minnow/guard.py" in found
    assert not any(name.startswith((".venv/", "tests/")) for name in found)


def test_every_file_that_trains_calls_the_leak_guard():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in source_files()
        if trains_without_guard(path.name, path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"These files train without calling assert_no_leak: {offenders}"


def test_no_source_file_is_too_long():
    offenders = [
        path.relative_to(ROOT).as_posix() for path in source_files() if line_count(path) > MAX_LINES
    ]
    assert not offenders, f"These files are over {MAX_LINES} lines, split them up: {offenders}"


def test_scripts_stay_thin():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in source_files()
        if path.parent == ROOT / "scripts" and line_count(path) > MAX_SCRIPT_LINES
    ]
    assert not offenders, (
        f"These scripts are over {MAX_SCRIPT_LINES} lines. Scripts are not covered by tests, "
        f"so move their logic into minnow/: {offenders}"
    )


def test_every_module_has_its_own_test_file():
    modules = [path for path in (ROOT / "minnow").rglob("*.py") if path.name != "__init__.py"]
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in modules
        if not (ROOT / "tests" / f"test_{path.stem}.py").exists()
    ]
    assert not offenders, f"These modules have no tests/test_<name>.py: {offenders}"


def test_chart_reference_lines_match_the_numbers_in_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        f"| SLM-SQL-0.5B | SFT and RL on SynSQL-derived data | {PUBLISHED_ACCURACY:.2f} |" in readme
    )
    assert f"Reach at least {TARGET_ACCURACY:g}% execution accuracy" in readme

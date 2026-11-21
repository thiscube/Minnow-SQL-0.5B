"""Rules about the shape of the codebase, checked automatically."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("minnow", "scripts")
MAX_LINES = 300
TRAINING_CALLS = {"train", "backward"}


def source_files() -> list[Path]:
    return sorted(path for name in SOURCE_DIRS for path in (ROOT / name).rglob("*.py"))


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
    return filename.startswith("train") or bool(called_names(source) & TRAINING_CALLS)


def trains_without_guard(filename: str, source: str) -> bool:
    return starts_training(filename, source) and "assert_no_leak" not in called_names(source)


def test_file_named_train_needs_the_guard():
    assert trains_without_guard("train_sft.py", "print('hello')")


def test_trainer_call_needs_the_guard_whatever_the_file_is_called():
    assert trains_without_guard("finetune.py", "trainer.train()")


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


def test_every_file_that_trains_calls_the_leak_guard():
    offenders = [
        str(path.relative_to(ROOT))
        for path in source_files()
        if trains_without_guard(path.name, path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"These files train without calling assert_no_leak: {offenders}"


def test_no_source_file_is_too_long():
    offenders = [
        str(path.relative_to(ROOT))
        for path in source_files()
        if len(path.read_text(encoding="utf-8").splitlines()) > MAX_LINES
    ]
    assert not offenders, f"These files are over {MAX_LINES} lines, split them up: {offenders}"

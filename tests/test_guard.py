"""Tests for the dev-set leak guard."""

from pathlib import Path

import pytest

from minnow.guard import (
    Example,
    LeakError,
    assert_no_leak,
    find_leaks,
    load_examples,
    main,
    normalize,
)

FIXTURES = Path(__file__).parent / "fixtures"

DEV = [
    Example("How many loaves were sold on Monday?", "bakery"),
    Example("Which platform has the most departures per day?", "train_station"),
]
TRAIN_CLEAN = [
    Example("How many apple trees are in each orchard?", "orchard"),
    Example("List the titles of films released before 1990.", "film_archive"),
]


def test_normalize_ignores_case_spacing_and_punctuation():
    assert normalize("  How MANY loaves,   were sold? ") == "how many loaves were sold"


def test_clean_training_data_passes():
    assert find_leaks(TRAIN_CLEAN, DEV).clean
    assert_no_leak(TRAIN_CLEAN, DEV)


@pytest.mark.parametrize(
    "question",
    [
        "How many loaves were sold on Monday?",
        "HOW MANY LOAVES WERE SOLD ON MONDAY?",
        "how many loaves   were sold on monday",
        "How many loaves were sold, on Monday?!",
    ],
)
def test_dev_question_in_training_data_is_caught(question):
    train = [*TRAIN_CLEAN, Example(question, "orchard")]

    report = find_leaks(train, DEV)

    assert report.questions == (question,)
    assert report.databases == ()
    assert not report.clean


def test_dev_database_in_training_data_is_caught():
    train = [*TRAIN_CLEAN, Example("Which baker works the most shifts?", "Bakery")]

    report = find_leaks(train, DEV)

    assert report.questions == ()
    assert report.databases == ("Bakery",)


def test_assert_no_leak_says_what_leaked():
    train = [*TRAIN_CLEAN, Example("How many loaves were sold on Monday?", "bakery")]

    with pytest.raises(LeakError) as raised:
        assert_no_leak(train, DEV)

    message = str(raised.value)
    assert "1 question(s), 1 database(s)" in message
    assert "question: How many loaves were sold on Monday?" in message
    assert "database: bakery" in message


def test_empty_dev_set_is_an_error_not_a_pass():
    with pytest.raises(ValueError, match="empty"):
        assert_no_leak(TRAIN_CLEAN, [])


def test_load_examples_reads_question_and_database():
    examples = load_examples(FIXTURES / "dev.json")

    assert len(examples) == 3
    assert examples[0] == Example("How many loaves were sold on Monday?", "bakery")


def test_command_line_reports_clean_data(capsys):
    exit_code = main([str(FIXTURES / "train_clean.json"), str(FIXTURES / "dev.json")])

    assert exit_code == 0
    assert "No leak: 3 training examples checked against 3 dev examples." in capsys.readouterr().out


def test_command_line_fails_on_leaky_data(capsys):
    exit_code = main([str(FIXTURES / "train_leaky.json"), str(FIXTURES / "dev.json")])

    assert exit_code == 1
    assert "1 question(s), 1 database(s)" in capsys.readouterr().out

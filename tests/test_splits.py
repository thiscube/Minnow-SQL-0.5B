import json

from minnow.splits import in_validation, main, split

ITEMS = [{"question": f"q{n}", "query": "SELECT 1", "db_id": f"db{n % 10}"} for n in range(40)]


def test_each_database_lands_in_exactly_one_half():
    validation, final = split(ITEMS)

    assert len(validation) + len(final) == len(ITEMS)
    assert {i["db_id"] for i in validation}.isdisjoint({i["db_id"] for i in final})
    assert validation
    assert final


def test_the_choice_is_fixed():
    assert [in_validation(f"db{n}") for n in range(10)] == [
        in_validation(f"db{n}") for n in range(10)
    ]


def test_command_line_writes_both_halves(tmp_path, capsys):
    (tmp_path / "test.json").write_text(json.dumps(ITEMS), encoding="utf-8")

    assert main([str(tmp_path)]) == 0

    validation = json.loads((tmp_path / "test_val.json").read_text(encoding="utf-8"))
    final = json.loads((tmp_path / "test_final.json").read_text(encoding="utf-8"))
    assert (validation, final) == split(ITEMS)
    assert "test_val.json" in capsys.readouterr().out

import re
import xml.etree.ElementTree as ET
from itertools import pairwise

import pytest

from minnow.chart import LABEL_FONT_SIZE, MAX_NAME_LENGTH, ROW_HEIGHT, WIDTH, bar_chart, height

SVG = "{http://www.w3.org/2000/svg}"
BARS = [("baseline", 45.36), ("sft", 61.9)]
MARKERS = [("target 70%", 70.0), ("SLM-SQL-0.5B, published 72.08%", 72.08)]
CHARACTER_WIDTH = 6.6
VALUE_FONT_SIZE = 12
# coordinates are written with two decimals
ROUNDING = 0.01


def draw(bars=BARS, markers=MARKERS):
    return ET.fromstring(bar_chart("Accuracy (%)", bars, markers))


def texts(chart):
    return {element.text: element for element in chart.iter(f"{SVG}text")}


def bar_edges(chart):
    """The left, top, right and bottom edge of every bar, read back from its outline."""
    edges = []
    for path in chart.iter(f"{SVG}path"):
        numbers = [float(number) for number in re.findall(r"-?\d+(?:\.\d+)?", path.get("d"))]
        edges.append((numbers[0], numbers[1], numbers[5], numbers[9]))
    return edges


def tick_x(chart, tick):
    return float(texts(chart)[str(tick)].get("x"))


def test_chart_is_an_svg_that_grows_with_the_number_of_bars():
    chart = draw()

    assert chart.tag == f"{SVG}svg"
    assert (chart.get("width"), chart.get("height")) == (str(WIDTH), str(height(2)))
    assert chart.get("viewBox") == f"0 0 {WIDTH} {height(2)}"
    assert height(16) - height(2) == 14 * ROW_HEIGHT


def test_background_is_light_and_opaque_and_covers_the_whole_chart():
    text = bar_chart("Accuracy (%)", BARS, MARKERS)
    background = draw().find(f"{SVG}rect")

    assert background.get("fill") == "#fcfcfb"
    assert float(background.get("width")) >= WIDTH - 1
    assert float(background.get("height")) >= height(2) - 1
    assert "opacity" not in text
    assert "transparent" not in text


def test_one_row_per_run_in_order_with_its_name_and_accuracy():
    chart = draw()
    labels = texts(chart)
    (left, first_top, first_end, first_bottom), (_, second_top, second_end, _) = bar_edges(chart)
    scale = (tick_x(chart, 100) - tick_x(chart, 0)) / 100

    assert left == pytest.approx(tick_x(chart, 0))
    assert first_end == pytest.approx(left + scale * 45.36, abs=ROUNDING)
    assert second_end == pytest.approx(left + scale * 61.9, abs=ROUNDING)
    assert first_bottom <= second_top
    assert first_top < float(labels["45.36%"].get("y")) < first_bottom + 2
    assert labels["45.36%"].get("y") == labels["baseline"].get("y")
    assert float(labels["45.36%"].get("x")) > first_end
    assert float(labels["baseline"].get("x")) < left
    assert "61.90%" in labels


def test_no_label_can_run_into_another():
    names = [f"run-with-a-long-name-{number}" for number in range(40)]
    chart = draw([(name, 50.0 + number) for number, name in enumerate(names)])
    labels = texts(chart)
    rows = [float(labels[name].get("y")) for name in names]

    assert all(later - earlier >= VALUE_FONT_SIZE for earlier, later in pairwise(rows))
    for name in names:
        assert float(labels[name].get("x")) - CHARACTER_WIDTH * len(name) >= 0
    value = labels["50.00%"]
    assert float(value.get("x")) <= WIDTH
    assert float(value.get("x")) - CHARACTER_WIDTH * len("100.00%") >= tick_x(chart, 100)


def test_a_very_long_name_is_cut_short():
    labels = texts(draw([("x" * 80, 10.0)]))

    assert "x" * (MAX_NAME_LENGTH - 3) + "..." in labels
    assert tick_x(draw([("x" * 80, 10.0)]), 0) < WIDTH / 2


def test_axis_runs_from_0_to_100():
    chart = draw([("none", 0.0), ("all", 100.0)])
    (left, _, none_end, _), (_, _, all_end, _) = bar_edges(chart)

    assert [tick_x(chart, tick) for tick in (0, 20, 40, 60, 80, 100)] == sorted(
        tick_x(chart, tick) for tick in (0, 20, 40, 60, 80, 100)
    )
    assert none_end == left
    assert all_end == pytest.approx(tick_x(chart, 100))


def test_markers_are_dashed_lines_at_their_values_with_labels():
    chart = draw()
    dashed = [group for group in chart.iter(f"{SVG}g") if group.get("stroke-dasharray")]
    positions = [float(line.get("x1")) for line in dashed[0].iter(f"{SVG}line")]
    scale = (tick_x(chart, 100) - tick_x(chart, 0)) / 100

    assert len(dashed) == 1
    assert positions == pytest.approx(
        [tick_x(chart, 0) + scale * 70, tick_x(chart, 0) + scale * 72.08], abs=ROUNDING
    )
    assert {"target 70%", "SLM-SQL-0.5B, published 72.08%"} <= texts(chart).keys()


def test_labels_of_close_markers_do_not_overlap_and_stay_inside():
    labels = texts(draw([("grpo-lr5-2400-hints", 66.92)]))
    target = labels["target 70%"]
    published = labels["SLM-SQL-0.5B, published 72.08%"]

    assert (target.get("text-anchor"), published.get("text-anchor")) == ("end", "start")
    assert float(target.get("x")) < float(published.get("x"))
    assert float(published.get("x")) + CHARACTER_WIDTH * len(published.text) <= WIDTH
    assert float(target.get("y")) + LABEL_FONT_SIZE < bar_edges(draw())[0][1]


def test_chart_without_runs_still_draws():
    chart = draw(bars=[])

    assert bar_edges(chart) == []
    assert "target 70%" in texts(chart)


def test_special_characters_in_labels_are_escaped():
    chart = ET.fromstring(bar_chart('A & B <"%">', [("a<b & 'c'", 10.0)], [("x > y", 50.0)]))

    assert chart.find(f"{SVG}title").text == 'A & B <"%">'
    assert {"a<b & 'c'", "x > y"} <= texts(chart).keys()


def test_same_input_gives_the_same_text_with_unix_line_endings():
    text = bar_chart("Accuracy (%)", BARS, MARKERS)

    assert text == bar_chart("Accuracy (%)", list(BARS), list(MARKERS))
    assert "\r" not in text
    assert text.endswith("</svg>\n")
    assert text.isascii()

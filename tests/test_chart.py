import re
import xml.etree.ElementTree as ET

import pytest

from minnow.chart import HEIGHT, WIDTH, bar_chart

SVG = "{http://www.w3.org/2000/svg}"
BARS = [("baseline", 45.36), ("sft", 61.9)]
MARKERS = [("target 70%", 70.0), ("SLM-SQL-0.5B, published 72.08%", 72.08)]
BASELINE_Y = 310
PIXELS_PER_PERCENT = 2.5
LABEL_FONT_SIZE = 11


def draw(bars=BARS, markers=MARKERS):
    return ET.fromstring(bar_chart("Accuracy (%)", bars, markers))


def texts(chart):
    return {element.text: element for element in chart.iter(f"{SVG}text")}


def bar_edges(chart):
    """The left, top and right edge of every bar, read back from its outline."""
    edges = []
    for path in chart.iter(f"{SVG}path"):
        numbers = [float(number) for number in re.findall(r"-?\d+(?:\.\d+)?", path.get("d"))]
        xs = [numbers[0], numbers[3], numbers[5], numbers[7], numbers[8], numbers[10]]
        edges.append((min(xs), numbers[4], max(xs)))
    return edges


def test_chart_is_an_svg_of_the_expected_size():
    chart = draw()

    assert chart.tag == f"{SVG}svg"
    assert (chart.get("width"), chart.get("height")) == ("720", "360")
    assert chart.get("viewBox") == f"0 0 {WIDTH} {HEIGHT}"


def test_background_is_light_and_opaque_and_covers_the_whole_chart():
    text = bar_chart("Accuracy (%)", BARS, MARKERS)
    background = draw().find(f"{SVG}rect")

    assert background.get("fill") == "#fcfcfb"
    assert float(background.get("width")) >= WIDTH - 1
    assert float(background.get("height")) >= HEIGHT - 1
    assert "opacity" not in text
    assert "transparent" not in text


def test_one_bar_per_run_in_order_with_its_accuracy_printed():
    chart = draw()
    labels = texts(chart)
    (first_left, first_top, first_right), (second_left, second_top, _) = bar_edges(chart)

    assert first_right < second_left
    assert first_top == pytest.approx(BASELINE_Y - PIXELS_PER_PERCENT * 45.36)
    assert second_top == pytest.approx(BASELINE_Y - PIXELS_PER_PERCENT * 61.9)
    assert first_left < float(labels["45.36%"].get("x")) < first_right
    assert float(labels["45.36%"].get("x")) == float(labels["baseline"].get("x"))
    assert float(labels["45.36%"].get("y")) < first_top
    assert float(labels["baseline"].get("y")) > BASELINE_Y
    assert float(labels["baseline"].get("x")) < float(labels["sft"].get("x"))
    assert "61.90%" in labels


def test_axis_runs_from_0_to_100():
    labels = texts(draw())

    assert float(labels["0"].get("y")) > float(labels["100"].get("y"))
    assert {"0", "20", "40", "60", "80", "100"} <= labels.keys()
    assert bar_edges(draw([("none", 0.0), ("all", 100.0)]))[0][1] == BASELINE_Y
    assert bar_edges(draw([("none", 0.0), ("all", 100.0)]))[1][1] == BASELINE_Y - 250


def test_markers_are_dashed_lines_at_their_values_with_labels():
    chart = draw()
    dashed = [group for group in chart.iter(f"{SVG}g") if group.get("stroke-dasharray")]
    heights = [float(line.get("y1")) for line in dashed[0].iter(f"{SVG}line")]

    assert len(dashed) == 1
    assert heights == pytest.approx(
        [BASELINE_Y - PIXELS_PER_PERCENT * 70, BASELINE_Y - PIXELS_PER_PERCENT * 72.08]
    )
    assert {"target 70%", "SLM-SQL-0.5B, published 72.08%"} <= texts(chart).keys()


def test_labels_of_close_markers_do_not_overlap():
    labels = texts(draw())
    target = float(labels["target 70%"].get("y"))
    published = float(labels["SLM-SQL-0.5B, published 72.08%"].get("y"))

    assert target - published >= 2 * LABEL_FONT_SIZE


def test_many_bars_stay_left_of_the_marker_labels_and_inside_the_plot():
    edges = bar_edges(draw([(f"run {number}", 50.0) for number in range(40)]))

    assert len(edges) == 40
    assert edges[0][0] >= 52
    assert edges[-1][2] <= WIDTH - 16 - 224
    assert all(left < right for left, _, right in edges)


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

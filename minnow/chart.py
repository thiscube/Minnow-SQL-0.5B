"""Draw a bar chart of percentages as SVG text, without a plotting library.

The chart paints its own light background, so it reads the same on a light and on a dark page.
The same input always gives the same text.
"""

from collections.abc import Sequence
from html import escape

Labelled = tuple[str, float]
WIDTH = 720
HEIGHT = 360
_LEFT = 52
_RIGHT = WIDTH - 16
_TOP = 60
_BOTTOM = 310
# the right end of the plot stays free of bars, so the marker labels never sit on one
_MARKER_LABEL_SPACE = 224
_MAX_BAND = 96
_MAX_BAR_WIDTH = 24
_BAR_CORNER = 4
_TICKS = (0, 20, 40, 60, 80, 100)
_SURFACE = "#fcfcfb"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_BAR_COLOR = "#2a78d6"
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
# a rim in the background colour keeps a label readable where a line runs behind it
_RIM = f'stroke="{_SURFACE}" stroke-width="3" paint-order="stroke"'


def _number(value: float) -> str:
    return f"{value:.2f}"


def _y(percent: float) -> float:
    return _BOTTOM - (_BOTTOM - _TOP) * percent / 100


def _band(count: int) -> float:
    return min(_MAX_BAND, (_RIGHT - _MARKER_LABEL_SPACE - _LEFT) / max(count, 1))


def _centres(count: int) -> list[float]:
    return [_LEFT + _band(count) * (position + 0.5) for position in range(count)]


def _frame(title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="chart-title chart-desc" '
        f'font-family="{_FONT}">',
        f'<title id="chart-title">{escape(title)}</title>',
        f'<desc id="chart-desc">{escape(description)}</desc>',
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{HEIGHT - 1}" rx="6" '
        f'fill="{_SURFACE}" stroke="{_GRID}"/>',
        f'<text x="16" y="32" font-size="14" font-weight="600" fill="{_INK}">'
        f"{escape(title)}</text>",
    ]


def _grid() -> list[str]:
    lines = [
        f'<g font-size="11" fill="{_MUTED_INK}" text-anchor="end" shape-rendering="crispEdges">'
    ]
    for tick in _TICKS:
        y = _y(tick)
        colour = _AXIS if tick == 0 else _GRID
        lines.append(
            f'<line x1="{_LEFT}" y1="{_number(y)}" x2="{_RIGHT}" y2="{_number(y)}" '
            f'stroke="{colour}"/>'
        )
        lines.append(f'<text x="{_LEFT - 8}" y="{_number(y + 4)}">{tick}</text>')
    return [*lines, "</g>"]


def _bar_shape(centre: float, percent: float, width: float) -> str:
    top = _y(percent)
    left = centre - width / 2
    right = centre + width / 2
    corner = min(_BAR_CORNER, width / 2, _BOTTOM - top)
    return (
        f'<path d="M{_number(left)} {_BOTTOM} V{_number(top + corner)} '
        f"Q{_number(left)} {_number(top)} {_number(left + corner)} {_number(top)} "
        f"H{_number(right - corner)} "
        f"Q{_number(right)} {_number(top)} {_number(right)} {_number(top + corner)} "
        f'V{_BOTTOM} Z"/>'
    )


def _bar_shapes(bars: Sequence[Labelled]) -> list[str]:
    width = min(_MAX_BAR_WIDTH, _band(len(bars)) / 2)
    shapes = [
        _bar_shape(centre, percent, width)
        for centre, (_, percent) in zip(_centres(len(bars)), bars, strict=True)
    ]
    return [f'<g fill="{_BAR_COLOR}">', *shapes, "</g>"]


def _bar_labels(bars: Sequence[Labelled]) -> list[str]:
    values = [f'<g font-size="12" font-weight="600" fill="{_INK}" text-anchor="middle" {_RIM}>']
    names = [f'<g font-size="11" fill="{_SECONDARY_INK}" text-anchor="middle">']
    for centre, (label, percent) in zip(_centres(len(bars)), bars, strict=True):
        x = _number(centre)
        values.append(f'<text x="{x}" y="{_number(_y(percent) - 7)}">{percent:.2f}%</text>')
        names.append(f'<text x="{x}" y="{_BOTTOM + 20}">{escape(label)}</text>')
    return [*values, "</g>", *names, "</g>"]


def _markers(markers: Sequence[Labelled]) -> list[str]:
    highest = max((percent for _, percent in markers), default=0)
    lines = [f'<g stroke="{_SECONDARY_INK}" stroke-dasharray="5 4">']
    labels = [f'<g font-size="11" fill="{_SECONDARY_INK}" text-anchor="end" {_RIM}>']
    for label, percent in markers:
        y = _y(percent)
        # the highest label sits above its line and the others below, so close lines stay apart
        label_y = y - 6 if percent == highest else y + 14
        lines.append(f'<line x1="{_LEFT}" y1="{_number(y)}" x2="{_RIGHT}" y2="{_number(y)}"/>')
        labels.append(f'<text x="{_RIGHT}" y="{_number(label_y)}">{escape(label)}</text>')
    return [*lines, "</g>", *labels, "</g>"]


def bar_chart(title: str, bars: Sequence[Labelled], markers: Sequence[Labelled]) -> str:
    """Return an SVG chart with one bar per (label, percent) and a dashed line per marker."""
    values = [f"{label}: {percent:.2f}%" for label, percent in bars]
    described = "; ".join([*values, *(label for label, _ in markers)])
    parts = [
        *_frame(title, described),
        *_grid(),
        *_bar_shapes(bars),
        *_markers(markers),
        *_bar_labels(bars),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"

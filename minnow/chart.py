"""Draw a horizontal bar chart of percentages as SVG text, without a plotting library.

Each bar gets its own row, with its name to the left and its value in a column right, so labels
cannot run into each other however many bars there are. The chart grows taller with every bar.
It paints its own light background, so it reads the same on a light and on a dark page.
The same input always gives the same text.
"""

from collections.abc import Sequence
from html import escape

Labelled = tuple[str, float]
WIDTH = 720
ROW_HEIGHT = 24
LABEL_FONT_SIZE = 11
MAX_NAME_LENGTH = 32
# widest a character of the label font gets, for reserving room without measuring text
_CHARACTER_WIDTH = 6.6
_MARGIN = 16
_GAP = 8
# room right of the plot for the column of printed values
_RIGHT = WIDTH - 72
_TOP = 76
_BELOW = 36
_BAR_HEIGHT = 14
_BAR_CORNER = 3
_TICKS = (0, 20, 40, 60, 80, 100)
_SURFACE = "#fcfcfb"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_BAR_COLOR = "#2a78d6"
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _number(value: float) -> str:
    return f"{value:.2f}"


def _name(label: str) -> str:
    if len(label) <= MAX_NAME_LENGTH:
        return label
    return label[: MAX_NAME_LENGTH - 3] + "..."


def _left(bars: Sequence[Labelled]) -> float:
    longest = max((len(_name(label)) for label, _ in bars), default=0)
    return _MARGIN + _CHARACTER_WIDTH * longest + _GAP


def _x(left: float, percent: float) -> float:
    return left + (_RIGHT - left) * percent / 100


def _row(position: int) -> float:
    return _TOP + ROW_HEIGHT * (position + 0.5)


def height(count: int) -> int:
    """Return the height in pixels of a chart with that many bars."""
    return _TOP + ROW_HEIGHT * count + _BELOW


def _frame(title: str, description: str, tall: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{tall}" '
        f'viewBox="0 0 {WIDTH} {tall}" role="img" aria-labelledby="chart-title chart-desc" '
        f'font-family="{_FONT}">',
        f'<title id="chart-title">{escape(title)}</title>',
        f'<desc id="chart-desc">{escape(description)}</desc>',
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{tall - 1}" rx="6" '
        f'fill="{_SURFACE}" stroke="{_GRID}"/>',
        f'<text x="{_MARGIN}" y="32" font-size="14" font-weight="600" fill="{_INK}">'
        f"{escape(title)}</text>",
    ]


def _grid(left: float, bottom: float) -> list[str]:
    lines = [
        f'<g font-size="{LABEL_FONT_SIZE}" fill="{_MUTED_INK}" text-anchor="middle" '
        f'shape-rendering="crispEdges">'
    ]
    for tick in _TICKS:
        x = _number(_x(left, tick))
        colour = _AXIS if tick == 0 else _GRID
        lines.append(f'<line x1="{x}" y1="{_TOP}" x2="{x}" y2="{bottom}" stroke="{colour}"/>')
        lines.append(f'<text x="{x}" y="{bottom + 18}">{tick}</text>')
    return [*lines, "</g>"]


def _bar_shape(left: float, middle: float, percent: float) -> str:
    end = _x(left, percent)
    top = middle - _BAR_HEIGHT / 2
    bottom = middle + _BAR_HEIGHT / 2
    corner = min(_BAR_CORNER, end - left)
    return (
        f'<path d="M{_number(left)} {_number(top)} H{_number(end - corner)} '
        f"Q{_number(end)} {_number(top)} {_number(end)} {_number(top + corner)} "
        f"V{_number(bottom - corner)} "
        f"Q{_number(end)} {_number(bottom)} {_number(end - corner)} {_number(bottom)} "
        f'H{_number(left)} Z"/>'
    )


def _bars(left: float, bars: Sequence[Labelled]) -> list[str]:
    shapes = [f'<g fill="{_BAR_COLOR}">']
    # the values line up in their own column right of the plot, clear of bars and marker lines
    values = [f'<g font-size="12" font-weight="600" fill="{_INK}" text-anchor="end">']
    names = [f'<g font-size="{LABEL_FONT_SIZE}" fill="{_SECONDARY_INK}" text-anchor="end">']
    for position, (label, percent) in enumerate(bars):
        middle = _row(position)
        text_y = _number(middle + 4)
        shapes.append(_bar_shape(left, middle, percent))
        values.append(f'<text x="{WIDTH - _MARGIN}" y="{text_y}">{percent:.2f}%</text>')
        names.append(f'<text x="{_number(left - _GAP)}" y="{text_y}">{escape(_name(label))}</text>')
    return [*shapes, "</g>", *names, "</g>", *values, "</g>"]


def _markers(left: float, bottom: float, markers: Sequence[Labelled]) -> list[str]:
    highest = max((percent for _, percent in markers), default=0)
    lines = [f'<g stroke="{_SECONDARY_INK}" stroke-dasharray="5 4">']
    labels = [f'<g font-size="{LABEL_FONT_SIZE}" fill="{_SECONDARY_INK}">']
    for label, percent in markers:
        x = _x(left, percent)
        # the highest label reads to the right of its line and the others to the left,
        # so the labels of close lines stay apart
        anchor, label_x = ("start", x + 5) if percent == highest else ("end", x - 5)
        lines.append(f'<line x1="{_number(x)}" y1="{_TOP - 22}" x2="{_number(x)}" y2="{bottom}"/>')
        labels.append(
            f'<text x="{_number(label_x)}" y="{_TOP - 12}" text-anchor="{anchor}">'
            f"{escape(label)}</text>"
        )
    return [*lines, "</g>", *labels, "</g>"]


def bar_chart(title: str, bars: Sequence[Labelled], markers: Sequence[Labelled]) -> str:
    """Return an SVG chart with one row per (label, percent) and a dashed line per marker."""
    described = "; ".join(
        [*(f"{label}: {percent:.2f}%" for label, percent in bars), *(m for m, _ in markers)]
    )
    left = _left(bars)
    bottom = _TOP + ROW_HEIGHT * len(bars)
    parts = [
        *_frame(title, described, height(len(bars))),
        *_grid(left, bottom),
        *_markers(left, bottom, markers),
        *_bars(left, bars),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"

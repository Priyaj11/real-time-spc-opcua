"""Plotly figures for the dashboard.

Kept separate from app.py so the chart construction can be exercised without
starting Streamlit, and so the colour decisions live in one place.

Colour rules followed here
--------------------------
One data series per chart, so no legend is needed and no categorical palette is
in play. Everything else on the plot is chrome: control limits are dashed and
muted, the centre line is a thin neutral, and only out-of-control points are
coloured.

Out-of-control points are marked by colour AND by a different symbol, never by
colour alone. Alarm severity is always shown with its word next to the colour,
for the same reason.

Specification limits are deliberately absent from the X-bar and R charts. Those
plot subgroup statistics, which are less spread out than individual parts, so
drawing a part-level tolerance across them would be misleading. They belong on
the individuals chart, and that is the only place they appear.
"""

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go

from spc_opcua.config import TagSpec
from spc_opcua.spc.control_charts import ChartLimits

# Chart chrome, from the reference palette's light surface.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# Marks. Validated as a pair against a white surface: CVD delta E 23.8.
SERIES = "#2a78d6"
OUT_OF_CONTROL = "#d03b3b"

# Status palette, fixed. Never used without its word beside it.
STATUS_COLOURS = {
    "IN CONTROL": "#0ca30c",
    "BASELINING": "#2a78d6",
    "WARNING": "#fab219",
    "CRITICAL": "#d03b3b",
    "DISCONNECTED": "#898781",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(title: str, y_title: str, height: int) -> dict[str, Any]:
    """Shared layout: recessive axes, no legend, room for limit labels."""
    return dict(
        title=dict(text=title, font=dict(size=14, color=INK), x=0, xanchor="left"),
        height=height,
        margin=dict(l=8, r=96, t=36, b=28),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            title=dict(text="subgroup", font=dict(size=11, color=MUTED)),
            gridcolor=GRID,
            zeroline=False,
            linecolor=GRID,
            tickfont=dict(size=11, color=MUTED),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=11, color=MUTED)),
            gridcolor=GRID,
            zeroline=False,
            linecolor=GRID,
            tickfont=dict(size=11, color=MUTED),
        ),
    )


def _limit_line(
    figure: go.Figure, value: float, label: str, dash: str, colour: str
) -> None:
    """Draw one horizontal reference line with a label in the right margin."""
    figure.add_hline(
        y=value,
        line=dict(color=colour, width=1, dash=dash),
        annotation_text=label,
        annotation_position="right",
        annotation_font=dict(size=10, color=colour, family=FONT),
    )


def _series(
    figure: go.Figure,
    x: Sequence[int],
    y: Sequence[float],
    flags: Sequence[bool],
    hover_label: str,
    decimals: int,
    flag_symbol: str = "x-thin",
    flag_label: str = "out of control",
) -> None:
    """The data line, plus a separate trace for the out-of-control points."""
    figure.add_trace(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines+markers",
            line=dict(color=SERIES, width=2),
            marker=dict(size=8, color=SERIES),
            name=hover_label,
            hovertemplate=f"subgroup %{{x}}<br>{hover_label} %{{y:.{decimals}f}}"
            "<extra></extra>",
        )
    )
    bad_x = [xi for xi, flag in zip(x, flags) if flag]
    bad_y = [yi for yi, flag in zip(y, flags) if flag]
    if bad_x:
        figure.add_trace(
            go.Scatter(
                x=bad_x,
                y=bad_y,
                mode="markers",
                # A different SYMBOL as well as a different colour, so the
                # signal is never carried by colour alone.
                marker=dict(
                    size=13 if flag_symbol == "x-thin" else 9,
                    color=OUT_OF_CONTROL,
                    symbol=flag_symbol,
                    line=dict(
                        width=3 if flag_symbol == "x-thin" else 0,
                        color=OUT_OF_CONTROL,
                    ),
                ),
                name=flag_label,
                hovertemplate=f"%{{x}}<br>{flag_label.upper()}"
                f"<br>%{{y:.{decimals}f}}<extra></extra>",
            )
        )


def empty_figure(message: str, height: int = 260) -> go.Figure:
    """A placeholder for before there is anything to plot."""
    figure = go.Figure()
    figure.update_layout(**_base_layout("", "", height))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=13, color=MUTED, family=FONT),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    return figure


def xbar_chart(
    rows: Sequence[dict[str, Any]], limits: ChartLimits, spec: TagSpec
) -> go.Figure:
    """Subgroup means against the frozen X-bar control limits."""
    if not rows:
        return empty_figure("waiting for subgroups")

    x = [row["subgroup"] for row in rows]
    y = [row["mean"] for row in rows]
    flags = [bool(row["mean_ooc"]) for row in rows]

    figure = go.Figure()
    _series(figure, x, y, flags, "mean", spec.decimals)
    _limit_line(figure, limits.xbar.upper, "UCL", "dash", MUTED)
    _limit_line(figure, limits.xbar.center, "centre", "solid", INK_SECONDARY)
    _limit_line(figure, limits.xbar.lower, "LCL", "dash", MUTED)
    figure.update_layout(
        **_base_layout(
            f"X-bar chart  ·  {spec.name} subgroup means [{spec.units}]",
            spec.units,
            300,
        )
    )
    return figure


def r_chart(
    rows: Sequence[dict[str, Any]], limits: ChartLimits, spec: TagSpec
) -> go.Figure:
    """Subgroup ranges against the frozen R control limits."""
    if not rows:
        return empty_figure("waiting for subgroups")

    x = [row["subgroup"] for row in rows]
    y = [row["range"] for row in rows]
    flags = [bool(row["range_ooc"]) for row in rows]

    figure = go.Figure()
    _series(figure, x, y, flags, "range", spec.decimals)
    _limit_line(figure, limits.r.upper, "UCL", "dash", MUTED)
    _limit_line(figure, limits.r.center, "R-bar", "solid", INK_SECONDARY)
    if not limits.r.lower_is_floored:
        _limit_line(figure, limits.r.lower, "LCL", "dash", MUTED)
    figure.update_layout(
        **_base_layout(
            f"R chart  ·  {spec.name} subgroup ranges [{spec.units}]",
            spec.units,
            260,
        )
    )
    return figure


def cpk_chart(
    rows: Sequence[dict[str, Any]],
    minimum_acceptable: float = 1.33,
    window: int = 0,
    monitored: int = 0,
) -> go.Figure:
    """The rolling Cpk trend, with the usual industry floor marked."""
    if not rows:
        remaining = max(0, window - monitored)
        message = (
            f"Cpk needs a full window of {window} subgroups"
            f"\n{remaining} more to go"
            if window
            else "waiting for a capability window"
        )
        return empty_figure(message, 240)

    x = [row["subgroup"] for row in rows]
    y = [row["cpk"] for row in rows]
    flags = [value < minimum_acceptable for value in y]

    figure = go.Figure()
    # A plain dot rather than a cross here: with a long spell below the floor
    # every marker would be flagged, and a line of crosses is unreadable. The
    # position relative to the labelled 1.33 line already carries the meaning
    # without relying on colour.
    _series(
        figure,
        x,
        y,
        flags,
        "Cpk",
        2,
        flag_symbol="circle",
        flag_label="below floor",
    )
    _limit_line(figure, minimum_acceptable, "1.33 floor", "dash", OUT_OF_CONTROL)
    figure.update_layout(**_base_layout("Cpk trend  ·  rolling window", "Cpk", 240))
    figure.update_yaxes(rangemode="tozero")
    return figure


def individuals_chart(
    values: Sequence[float], spec: TagSpec, limit: int = 120
) -> go.Figure:
    """Recent individual parts, with the customer's specification limits.

    This is the ONLY chart that carries specification limits, because it is the
    only one plotting individual parts. Drawing them across the X-bar chart
    would compare a part-level tolerance against subgroup means, which are
    narrower by the square root of the subgroup size.
    """
    if not values:
        return empty_figure("waiting for parts", 240)

    recent = list(values)[-limit:]
    start = max(0, len(values) - len(recent))
    x = list(range(start, start + len(recent)))
    flags = [
        (spec.lsl is not None and v < spec.lsl)
        or (spec.usl is not None and v > spec.usl)
        for v in recent
    ]

    figure = go.Figure()
    _series(figure, x, recent, flags, spec.name, spec.decimals)
    if spec.usl is not None:
        _limit_line(figure, spec.usl, "USL", "dot", OUT_OF_CONTROL)
    _limit_line(figure, spec.nominal, "nominal", "solid", INK_SECONDARY)
    if spec.lsl is not None:
        _limit_line(figure, spec.lsl, "LSL", "dot", OUT_OF_CONTROL)
    figure.update_layout(
        **_base_layout(
            f"Individual parts  ·  {spec.name} against specification [{spec.units}]",
            spec.units,
            240,
        )
    )
    figure.update_xaxes(title=dict(text="part", font=dict(size=11, color=MUTED)))
    return figure
"""The BORE-01 operator screen.

Run it with:

    streamlit run spc_opcua/dashboard/app.py

This is meant to read like the panel on a machine, not like a notebook. Big
status, big current values, charts underneath, alarms with words as well as
colours. An operator should be able to tell whether anything needs attention
from across the room, and only then walk over and read the detail.

Everything real happens in LiveSource, which owns the OPC UA connection and the
SPC engine in a background thread. This file only draws what LiveSource hands
it, which is why Streamlit re-running the whole script several times a second
costs nothing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# `streamlit run` puts THIS file's folder on the import path, not the project
# root, so `import spc_opcua` fails unless the project is pip-installed. Adding
# the repo root here makes the command work from a plain clone. This has to
# happen before the spc_opcua imports below, which is why it is not at the top.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from spc_opcua.config import load_config
from spc_opcua.dashboard import charts
from spc_opcua.dashboard.live_source import LiveSource, Snapshot
from spc_opcua.logging_setup import configure_logging
from spc_opcua.opcua_server import FAULT_PRESETS
from spc_opcua.spc.alarms import CRITICAL, describe_rule
from spc_opcua.spc.capability import MINIMUM_ACCEPTABLE_CPK
from spc_opcua.spc.nelson_rules import ALL_RULES, COMMON_RULES, RULES

REFRESH_SECONDS = 1.0

# The tags shown as big numbers along the top, in the order an operator reads
# them: the charted characteristic first, then the things that explain it.
TILE_TAGS = ("BoreDiameter", "Torque", "Temperature", "Vibration", "CycleTime")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


def configure_page() -> None:
    """Wide layout and a little CSS to make the status strip read like a panel."""
    st.set_page_config(
        page_title="BORE-01 SPC Monitor",
        page_icon="M",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
          .status-strip {
              display: flex; align-items: center; gap: 1.25rem;
              padding: 0.9rem 1.2rem; border-radius: 6px;
              border: 1px solid rgba(11,11,11,0.10);
              margin-bottom: 1.1rem;
          }
          .status-word {
              font-size: 1.55rem; font-weight: 700; letter-spacing: 0.04em;
          }
          .status-meta { font-size: 0.86rem; color: #52514e; line-height: 1.5; }
          .tile {
              border: 1px solid rgba(11,11,11,0.10); border-radius: 6px;
              padding: 0.7rem 0.9rem; background: #ffffff; height: 100%;
          }
          .tile-label {
              font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase;
              color: #898781; margin-bottom: 0.25rem;
          }
          .tile-value {
              font-size: 1.5rem; font-weight: 650; color: #0b0b0b;
              font-variant-numeric: tabular-nums; line-height: 1.15;
          }
          .tile-unit { font-size: 0.8rem; color: #898781; font-weight: 400; }
          .tile-note { font-size: 0.72rem; color: #52514e; margin-top: 0.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# The live source lives in session state, so it survives a rerun
# ---------------------------------------------------------------------------


def source_settings() -> dict[str, object]:
    """Read the sidebar controls that decide how the source is built."""
    st.sidebar.title("BORE-01")
    st.sidebar.caption("Real-time SPC monitoring over OPC UA")

    scenario = st.sidebar.selectbox(
        "Fault scenario",
        sorted(FAULT_PRESETS),
        index=sorted(FAULT_PRESETS).index("tool-wear"),
        help="What is wrong with the machine. Injected into the simulator.",
    )
    speed = st.sidebar.slider(
        "Speed factor",
        min_value=5,
        max_value=100,
        value=60,
        step=5,
        help=(
            "Simulated seconds per real second. Above about 120 the publishing "
            "loop starves the event loop and clients cannot connect."
        ),
    )
    baseline = st.sidebar.slider(
        "Baseline subgroups",
        min_value=10,
        max_value=40,
        value=25,
        step=5,
        help="Collected before the control limits are frozen. 20 is the usual minimum.",
    )
    window = st.sidebar.slider(
        "Capability window",
        min_value=10,
        max_value=30,
        value=15,
        step=5,
        help=(
            "Subgroups behind each rolling Cpk. Fewer starts the trend sooner "
            "and makes every point noisier. A Cp from 200 parts already carries "
            "a standard deviation near 0.1."
        ),
    )
    rule_set = st.sidebar.radio(
        "Nelson Rules",
        ["Common five", "All eight", "Rule 1 only"],
        help=(
            "More rules detect faster and produce more false alarms. Measured "
            "on this process: 0.5 percent with rule 1 alone, about 5 percent "
            "with the common five."
        ),
    )
    rules = {
        "All eight": ALL_RULES,
        "Common five": COMMON_RULES,
        "Rule 1 only": (1,),
    }[rule_set]

    return {
        "scenario": scenario,
        "speed": float(speed),
        "baseline_subgroups": int(baseline),
        "capability_window": int(window),
        "rules": rules,
    }


def get_source(settings: dict[str, object]) -> LiveSource:
    """Build the source once, and rebuild it only when the settings change."""
    signature = (
        settings["scenario"],
        settings["speed"],
        settings["baseline_subgroups"],
        settings["capability_window"],
        settings["rules"],
    )
    if st.session_state.get("signature") != signature:
        old: LiveSource | None = st.session_state.get("source")
        if old is not None:
            old.stop(timeout=2.0)
        source = LiveSource(
            scenario=str(settings["scenario"]),
            speed=float(settings["speed"]),
            baseline_subgroups=int(settings["baseline_subgroups"]),
            capability_window=int(settings["capability_window"]),
            rules=settings["rules"],  # type: ignore[arg-type]
        )
        source.start()
        st.session_state["source"] = source
        st.session_state["signature"] = signature
    return st.session_state["source"]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def status_strip(snapshot: Snapshot) -> None:
    """The one thing readable from across the room."""
    if not snapshot.connected:
        word = "DISCONNECTED"
    else:
        word = snapshot.spc_status
    colour = charts.STATUS_COLOURS.get(word, charts.MUTED)

    detail = snapshot.machine_status if snapshot.connected else "no OPC UA connection"
    if snapshot.error:
        detail = snapshot.error

    if snapshot.is_baselining:
        progress = int(snapshot.baseline_progress * 100)
        phase_text = (
            f"Collecting baseline, {progress}% of {snapshot.baseline_target} subgroups"
        )
    else:
        phase_text = f"Monitoring, {snapshot.subgroups} subgroups plotted"

    st.markdown(
        f"""
        <div class="status-strip" style="background:{colour}14;border-left:6px solid {colour};">
          <div class="status-word" style="color:{colour};">{word}</div>
          <div class="status-meta">
            <b>{detail}</b><br>
            {phase_text} &nbsp;·&nbsp; scenario <b>{snapshot.scenario}</b>
            &nbsp;·&nbsp; speed x{snapshot.speed:g}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tile(label: str, value: str, unit: str = "", note: str = "") -> str:
    """One big-number panel."""
    unit_html = f' <span class="tile-unit">{unit}</span>' if unit else ""
    note_html = f'<div class="tile-note">{note}</div>' if note else ""
    return (
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}{unit_html}</div>{note_html}</div>'
    )


def value_tiles(snapshot: Snapshot) -> None:
    """Current value of every process tag, plus the counters."""
    config = load_config()
    columns = st.columns(len(TILE_TAGS) + 3)

    for column, name in zip(columns, TILE_TAGS, strict=False):
        spec = config.tag(name)
        raw = snapshot.latest.get(name)
        shown = "--" if raw is None else f"{float(raw):.{spec.decimals}f}"
        age = snapshot.ages.get(name)
        note = ""
        if age is not None and age > 10.0:
            note = f"last change {age:.0f} s ago"
        column.markdown(tile(name, shown, spec.units, note), unsafe_allow_html=True)

    columns[-3].markdown(
        tile("Parts", f"{snapshot.parts}", "", f"{snapshot.updates} tag updates"),
        unsafe_allow_html=True,
    )
    columns[-2].markdown(
        tile("Scrap", f"{snapshot.scrap}", "parts"), unsafe_allow_html=True
    )

    capability = snapshot.capability
    if capability is None or capability.cpk is None:
        remaining = max(0, snapshot.capability_window - snapshot.subgroups)
        columns[-1].markdown(
            tile("Cpk", "--", "", f"{remaining} more subgroups"),
            unsafe_allow_html=True,
        )
    else:
        columns[-1].markdown(
            tile(
                "Cpk",
                f"{capability.cpk:.2f}",
                "",
                f"{capability.verdict} · Ppk {capability.ppk:.2f}",
            ),
            unsafe_allow_html=True,
        )


def chart_panels(snapshot: Snapshot) -> None:
    """The four charts, in the order an SPC engineer reads them."""
    config = load_config()
    spec = config.tag("BoreDiameter")

    if not snapshot.has_chart:
        st.info(
            "Control limits are calculated from a stable baseline, not from the "
            "first point that arrives. Charts appear once the baseline is complete."
        )
        st.progress(min(1.0, snapshot.baseline_progress))
        st.plotly_chart(
            charts.individuals_chart(snapshot.recent_parts, spec),
            use_container_width=True,
        )
        return

    assert snapshot.limits is not None
    left, right = st.columns([3, 2])

    with left:
        st.plotly_chart(
            charts.xbar_chart(snapshot.chart_rows, snapshot.limits, spec),
            use_container_width=True,
        )
        st.plotly_chart(
            charts.r_chart(snapshot.chart_rows, snapshot.limits, spec),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            charts.cpk_chart(
                snapshot.capability_rows,
                MINIMUM_ACCEPTABLE_CPK,
                window=snapshot.capability_window,
                monitored=snapshot.subgroups,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            charts.individuals_chart(snapshot.recent_parts, spec),
            use_container_width=True,
        )
        st.caption(
            "Specification limits appear only on the individuals chart. Subgroup "
            "means are narrower than single parts by the square root of the "
            "subgroup size, so a part tolerance drawn across an X-bar chart "
            "would mislead."
        )


def alarm_panels(snapshot: Snapshot, source: LiveSource) -> None:
    """Active alarms first, cleared ones tucked away."""
    header, button = st.columns([5, 1])
    header.subheader(f"Active alarms ({len(snapshot.active_alarms)})")
    if button.button("Acknowledge", use_container_width=True):
        source.acknowledge_alarms()

    if not snapshot.active_alarms:
        st.success("No active alarms.")
    else:
        for alarm in snapshot.active_alarms:
            colour = charts.STATUS_COLOURS[
                CRITICAL if alarm.is_critical else "WARNING"
            ]
            acknowledged = " · acknowledged" if alarm.acknowledged else ""
            st.markdown(
                f"""
                <div class="tile" style="border-left:5px solid {colour};margin-bottom:0.45rem;">
                  <b style="color:{colour};">{alarm.severity}</b>
                  &nbsp; {alarm.chart} chart, rule {alarm.rule}
                  &mdash; <b>{alarm.name}</b>{acknowledged}
                  <div class="tile-note">
                    subgroups {alarm.first_index} to {alarm.last_index},
                    {alarm.occurrences} firings &nbsp;·&nbsp; {alarm.detail}<br>
                    {describe_rule(alarm.rule)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander(f"Alarm history ({len(snapshot.alarm_history)} cleared)"):
        if not snapshot.alarm_history:
            st.caption("Nothing has cleared yet.")
        else:
            st.dataframe(
                [alarm.as_row() for alarm in snapshot.alarm_history],
                use_container_width=True,
                hide_index=True,
            )


def footer(snapshot: Snapshot) -> None:
    """Data integrity and the control limits, for whoever wants the numbers."""
    left, right = st.columns(2)

    with left:
        st.subheader("Data integrity")
        st.write(
            {
                "connected": snapshot.connected,
                "endpoint": snapshot.endpoint or "--",
                "tag updates received": snapshot.updates,
                "parts recorded": snapshot.parts,
                "parts missed": snapshot.missed,
            }
        )
        if snapshot.missed:
            st.error(
                f"{snapshot.missed} part measurements never arrived. Statistics "
                "on this page are computed from incomplete data."
            )

    with right:
        st.subheader("Control limits")
        if snapshot.limits is None:
            st.caption("Not calculated yet.")
        else:
            st.code(snapshot.limits.describe(), language="text")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Draw one frame of the dashboard, then schedule the next."""
    configure_logging(level="WARNING")
    configure_page()

    settings = source_settings()
    source = get_source(settings)
    snapshot = source.snapshot()

    st.sidebar.divider()
    st.sidebar.caption(
        "Rules enabled: " + ", ".join(str(n) for n in settings["rules"])  # type: ignore[arg-type]
    )
    for number in settings["rules"]:  # type: ignore[union-attr]
        st.sidebar.caption(f"{number}. {RULES[number].detects}")

    status_strip(snapshot)
    value_tiles(snapshot)
    st.divider()
    chart_panels(snapshot)
    st.divider()
    alarm_panels(snapshot, source)
    st.divider()
    footer(snapshot)

    time.sleep(REFRESH_SECONDS)
    st.rerun()


main()
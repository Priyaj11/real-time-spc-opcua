"""Run the whole evaluation and write the results out.

    python -m spc_opcua.evaluation

Produces two files under data/ and prints the table it wrote:

    evaluation_runs.csv       one row per individual production run
    evaluation_summary.csv    one row per scenario

The per-run file is kept because a summary is a claim and the runs are the
evidence. Anyone who doubts a number in the README can open the raw file, find
the scenario and the seed, and re-run exactly that case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from spc_opcua.config import PROJECT_ROOT
from spc_opcua.evaluation.runner import (
    DEFAULT_REPLICATES,
    headline,
    run_all,
    summarise_all,
)
from spc_opcua.evaluation.scenarios import (
    BASELINE_SUBGROUPS,
    MONITOR_SUBGROUPS,
    SCENARIOS,
)
from spc_opcua.logging_setup import configure_logging
from spc_opcua.spc.nelson_rules import ALL_RULES, COMMON_RULES

RULE_SETS = {
    "common": COMMON_RULES,
    "all": ALL_RULES,
    "rule1": (1,),
}


def build_parser() -> argparse.ArgumentParser:
    """Command line options for the evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m spc_opcua.evaluation",
        description=(
            "Run every fault scenario many times and measure detection rate, "
            "detection latency and false alarm rate."
        ),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help=f"Runs per scenario (default {DEFAULT_REPLICATES}).",
    )
    parser.add_argument(
        "--rules",
        choices=sorted(RULE_SETS),
        default="common",
        help="Which Nelson Rules to apply (default common).",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=BASELINE_SUBGROUPS,
        help=f"Subgroups before limits freeze (default {BASELINE_SUBGROUPS}).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=MONITOR_SUBGROUPS,
        help=f"Subgroups monitored after the fault (default {MONITOR_SUBGROUPS}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Where to write the CSV files.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run the whole evaluation once per rule set and print the "
            "detection-versus-false-alarm trade-off. Takes three times as long."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Log warnings only.")
    return parser


def compare_rule_sets(args: argparse.Namespace) -> str:
    """Run the evaluation under each rule set and tabulate the trade-off.

    This is the central question of the project stated as one table: more rules
    catch more faults sooner, and cry wolf more often on a healthy machine.
    Neither column is a bug. Which one matters more is a plant decision, not a
    software one, and the honest thing a tool can do is measure both.
    """
    rows = []
    for label in ("rule1", "common", "all"):
        results = run_all(
            replicates=args.replicates,
            rules=RULE_SETS[label],
            baseline=args.baseline,
            monitor=args.monitor,
        )
        numbers = headline(summarise_all(results))
        rows.append((label, numbers))

    header = (
        f"{'rules':<8}{'detected':>10}{'rate':>7}{'median sg':>11}"
        f"{'FA/subgroup':>13}{'FA/window':>11}{'ARL0':>8}"
    )
    lines = [header, "-" * len(header)]
    for label, n in rows:
        lines.append(
            f"{label:<8}"
            f"{n['faulted_detected']}/{n['faulted_runs']:<6}"
            f"{n['detection_rate']:>6.0%}"
            f"{n['median_detection_subgroups']:>11g}"
            f"{n['false_alarm_rate_per_subgroup']:>12.2%}"
            f"{n['false_alarm_rate_per_window']:>11.0%}"
            f"{n['subgroups_between_false_alarms']:>8.0f}"
        )
    lines.append("")
    lines.append(
        "ARL0 is the average run length in control: subgroups between one "
        "false alarm and the next."
    )
    return "\n".join(lines)


def format_table(summary: pd.DataFrame) -> str:
    """The summary as a fixed-width table, for a terminal or a README."""
    columns = [
        ("scenario", "scenario", "{:<18}"),
        ("kind", "kind", "{:<8}"),
        ("detection_rate", "detect", "{:>7}"),
        ("median_subgroups", "med sg", "{:>7}"),
        ("median_parts", "parts", "{:>6}"),
        ("worst_subgroups", "worst", "{:>6}"),
        ("median_warning_parts", "warning", "{:>8}"),
        ("scrap_parts", "scrap", "{:>6}"),
        ("scrap_avoidable", "avoid", "{:>6}"),
        ("alarm_rate_per_subgroup", "per sg", "{:>7}"),
    ]
    header = "".join(fmt.format(title) for _, title, fmt in columns)
    lines = [header, "-" * len(header)]

    for _, row in summary.iterrows():
        cells = []
        for key, _, fmt in columns:
            value = row[key]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                text = "-"
            elif key in ("detection_rate", "scrap_avoidable", "alarm_rate_per_subgroup"):
                text = f"{value:.0%}"
            elif isinstance(value, float):
                text = f"{value:g}"
            else:
                text = str(value)
            cells.append(fmt.format(text))
        lines.append("".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation, write the CSVs, print the table."""
    args = build_parser().parse_args(argv)
    configure_logging(level="WARNING" if args.quiet else "INFO")

    if args.compare:
        print()
        print(
            f"{len(SCENARIOS)} scenarios x {args.replicates} replicates, "
            f"per rule set, monitored window={args.monitor} subgroups"
        )
        print()
        print(compare_rule_sets(args))
        return 0

    results = run_all(
        replicates=args.replicates,
        rules=RULE_SETS[args.rules],
        baseline=args.baseline,
        monitor=args.monitor,
    )
    summaries = summarise_all(results)

    runs_frame = pd.DataFrame([r.as_row() for r in results])
    summary_frame = pd.DataFrame([s.as_row() for s in summaries])

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "evaluation_runs.csv"
    summary_path = args.out / "evaluation_summary.csv"
    runs_frame.to_csv(runs_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    print()
    print(
        f"{len(SCENARIOS)} scenarios x {args.replicates} replicates, "
        f"rules={args.rules}, baseline={args.baseline}, "
        f"monitored window={args.monitor} subgroups"
    )
    print()
    print(format_table(summary_frame))
    print()
    print("Headline")
    print(json.dumps(headline(summaries), indent=2, default=str))
    print()
    print(f"Wrote {runs_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
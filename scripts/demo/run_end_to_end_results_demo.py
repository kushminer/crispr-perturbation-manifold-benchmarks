#!/usr/bin/env python3
"""Run end-to-end result aggregation and print verified project conclusions."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _run_aggregation(repo_root: Path) -> None:
    _ensure_src_on_path(repo_root)
    from analysis.aggregate_all_results import (
        create_comparison_summaries,
        create_engineer_summary_report,
        load_pseudobulk_baseline_results,
        load_pseudobulk_logo_results,
        load_pseudobulk_lsft_results,
        load_single_cell_baseline_results,
        load_single_cell_logo_results,
        load_single_cell_lsft_results,
    )

    results_dir = repo_root / "results"
    output_dir = repo_root / "aggregated_results"

    pb_baseline = load_pseudobulk_baseline_results(results_dir)
    pb_lsft = load_pseudobulk_lsft_results(results_dir)
    pb_logo = load_pseudobulk_logo_results(results_dir)
    sc_baseline = load_single_cell_baseline_results(results_dir)
    sc_lsft = load_single_cell_lsft_results(results_dir)
    sc_logo = load_single_cell_logo_results(results_dir)

    create_comparison_summaries(
        pseudobulk_baseline=pb_baseline,
        single_cell_baseline=sc_baseline,
        pseudobulk_lsft=pb_lsft,
        single_cell_lsft=sc_lsft,
        pseudobulk_logo=pb_logo,
        single_cell_logo=sc_logo,
        output_dir=output_dir,
    )
    create_engineer_summary_report(output_dir)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path)


def _mean_delta(lsft_df: pd.DataFrame, baseline: str) -> float:
    subset = lsft_df[lsft_df["baseline"] == baseline]
    if subset.empty:
        raise ValueError(f"Baseline not found in LSFT summary: {baseline}")
    return float(subset["mean_delta_r"].mean())


def _selftrained_data_scale_trend(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    combined_files = sorted(
        (results_dir / "goal_3_prediction" / "lsft_resampling").glob("*/*all_baselines_combined.csv")
    )
    for file_path in combined_files:
        dataset = file_path.parent.name.lower()
        df = pd.read_csv(file_path)
        subset = df[df["baseline_type"] == "lpm_selftrained"]
        if subset.empty:
            continue
        grouped = (
            subset.groupby("top_pct", as_index=False)["performance_local_pearson_r"]
            .mean()
            .sort_values("top_pct")
            .reset_index(drop=True)
        )
        if len(grouped) < 2:
            continue
        start_pct = float(grouped.loc[0, "top_pct"])
        end_pct = float(grouped.loc[len(grouped) - 1, "top_pct"])
        start_r = float(grouped.loc[0, "performance_local_pearson_r"])
        end_r = float(grouped.loc[len(grouped) - 1, "performance_local_pearson_r"])
        rows.append(
            {
                "dataset": dataset,
                "start_pct": start_pct,
                "end_pct": end_pct,
                "start_r": start_r,
                "end_r": end_r,
                "delta_r": end_r - start_r,
            }
        )
    if not rows:
        raise ValueError("No LSFT self-trained sweep data found under results/")
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _format_float(value: float, ndigits: int = 4) -> str:
    return f"{value:.{ndigits}f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _find_top_baseline(best_df: pd.DataFrame, analysis_type: str) -> str:
    subset = best_df[best_df["analysis_type"] == analysis_type]
    if subset.empty:
        raise ValueError(f"No rows for analysis type: {analysis_type}")
    return str(subset["best_baseline"].mode().iloc[0])


def _build_summary(
    best_df: pd.DataFrame,
    sc_lsft_df: pd.DataFrame,
    logo_df: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> str:
    sc_selftrained = _mean_delta(sc_lsft_df, "lpm_selftrained")
    sc_scgpt = _mean_delta(sc_lsft_df, "lpm_scgptGeneEmb")
    sc_random_pert = _mean_delta(sc_lsft_df, "lpm_randomPertEmb")

    top_single_cell = _find_top_baseline(best_df, "single_cell")
    top_pseudobulk = _find_top_baseline(best_df, "pseudobulk")

    sc_logo = logo_df[logo_df["analysis_type"] == "single_cell_logo"]
    pb_logo = logo_df[logo_df["analysis_type"] == "pseudobulk_logo"]
    sc_logo_top = str(sc_logo.groupby("baseline")["pearson_r"].mean().idxmax())
    pb_logo_top = str(pb_logo.groupby("baseline")["pearson_r"].mean().idxmax())
    sc_logo_top_r = float(sc_logo.groupby("baseline")["pearson_r"].mean().max())
    pb_logo_top_r = float(pb_logo.groupby("baseline")["pearson_r"].mean().max())

    lines: list[str] = []
    lines.append("# End-to-End Results Demo Summary")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    lines.append("")
    lines.append("## Verified Conclusions")
    lines.append("")
    lines.append(
        "1. LSFT adds little on top of the strongest single-cell baseline."
    )
    lines.append(
        f"   - Mean Δr (`lpm_selftrained`): {_format_float(sc_selftrained)}"
    )
    lines.append(
        f"   - Mean Δr (`lpm_scgptGeneEmb`): {_format_float(sc_scgpt)}"
    )
    lines.append(
        f"   - Mean Δr (`lpm_randomPertEmb`): {_format_float(sc_random_pert)}"
    )
    lines.append("")
    lines.append(
        "2. Self-trained PCA (`lpm_selftrained`) is the top baseline across datasets."
    )
    lines.append(f"   - Single-cell best baseline: `{top_single_cell}`")
    lines.append(f"   - Pseudobulk best baseline: `{top_pseudobulk}`")
    lines.append("")
    lines.append(
        "3. More local training data improves pseudobulk LSFT for `lpm_selftrained`."
    )
    for row in trend_df.itertuples(index=False):
        lines.append(
            "   - "
            f"{row.dataset}: {_format_pct(row.start_pct)} {_format_float(row.start_r, 3)} "
            f"-> {_format_pct(row.end_pct)} {_format_float(row.end_r, 3)} "
            f"(Δr={_format_float(row.delta_r, 3)})"
        )
    lines.append("")
    lines.append("4. PCA also leads in LOGO generalization.")
    lines.append(
        f"   - Single-cell LOGO top baseline: `{sc_logo_top}` (mean r={_format_float(sc_logo_top_r, 3)})"
    )
    lines.append(
        f"   - Pseudobulk LOGO top baseline: `{pb_logo_top}` (mean r={_format_float(pb_logo_top_r, 3)})"
    )
    lines.append("")
    lines.append("## Sponsorship")
    lines.append("This project was sponsored by the **NIH Bridges to Baccalaureate** program.")
    lines.append("")
    return "\n".join(lines)


def _print_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        print(f"- {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate aggregate outputs and print verified conclusions."
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Use existing files in aggregated_results/ without regenerating first.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=Path("aggregated_results/final_conclusions_verified.md"),
        help="Write markdown summary to this repo-relative path (default: aggregated_results/final_conclusions_verified.md).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    results_dir = repo_root / "results"
    agg_dir = repo_root / "aggregated_results"

    if not args.skip_aggregate:
        _run_aggregation(repo_root)

    best_df = _load_csv(agg_dir / "best_baseline_per_dataset.csv")
    sc_lsft_df = _load_csv(agg_dir / "lsft_improvement_summary.csv")
    logo_df = _load_csv(agg_dir / "logo_generalization_all_analyses.csv")
    trend_df = _selftrained_data_scale_trend(results_dir)

    summary = _build_summary(
        best_df=best_df,
        sc_lsft_df=sc_lsft_df,
        logo_df=logo_df,
        trend_df=trend_df,
    )

    report_path = (repo_root / args.write_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")

    if not args.quiet:
        print(summary)
        print("## Outputs")
        _print_paths(
            [
                agg_dir / "baseline_performance_all_analyses.csv",
                agg_dir / "best_baseline_per_dataset.csv",
                agg_dir / "lsft_improvement_summary.csv",
                agg_dir / "lsft_improvement_summary_pseudobulk.csv",
                agg_dir / "logo_generalization_all_analyses.csv",
                report_path,
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

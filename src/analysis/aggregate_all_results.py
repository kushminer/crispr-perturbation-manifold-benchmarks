#!/usr/bin/env python3
"""
Aggregate research outputs from canonical result artifacts.

This module intentionally avoids hardcoded metrics. All summary tables are
derived from files under `results/`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_ALIASES = {
    "adamson": "adamson",
    "k562": "k562",
    "replogle_k562": "k562",
    "replogle_k562_essential": "k562",
    "rpe1": "rpe1",
    "replogle_rpe1": "rpe1",
    "replogle_rpe1_essential": "rpe1",
}


def _normalize_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name.lower(), name.lower())


def load_single_cell_baseline_results(results_dir: Path) -> pd.DataFrame:
    """Load single-cell baseline summary from baseline_results_all.csv."""
    path = results_dir / "single_cell_analysis" / "comparison" / "baseline_results_all.csv"
    if not path.exists():
        logger.warning("Single-cell baseline file not found: %s", path)
        return pd.DataFrame(columns=["dataset", "baseline", "pearson_r", "l2", "analysis_type"])

    df = pd.read_csv(path)
    required = {"dataset", "baseline", "pert_mean_pearson_r", "pert_mean_l2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    out = df[["dataset", "baseline", "pert_mean_pearson_r", "pert_mean_l2"]].copy()
    out.loc[:, "dataset"] = out["dataset"].map(_normalize_dataset_name)
    out = out.rename(columns={"pert_mean_pearson_r": "pearson_r", "pert_mean_l2": "l2"})
    out["analysis_type"] = "single_cell"
    return out


def _find_lsft_combined_files(results_dir: Path) -> list[Path]:
    lsft_root = results_dir / "goal_3_prediction" / "lsft_resampling"
    if not lsft_root.exists():
        return []
    return sorted(lsft_root.glob("*/*all_baselines_combined.csv"))


def load_pseudobulk_baseline_results(results_dir: Path) -> pd.DataFrame:
    """Load pseudobulk baseline means from LSFT combined result files."""
    rows = []
    for combined_file in _find_lsft_combined_files(results_dir):
        dataset = _normalize_dataset_name(combined_file.parent.name)
        df = pd.read_csv(combined_file)
        required = {
            "baseline_type",
            "test_perturbation",
            "performance_baseline_pearson_r",
            "performance_baseline_l2",
        }
        missing = required.difference(df.columns)
        if missing:
            logger.warning("Skipping %s (missing columns: %s)", combined_file, sorted(missing))
            continue

        dedup = (
            df.sort_values(["baseline_type", "test_perturbation", "top_pct"])
            .drop_duplicates(subset=["baseline_type", "test_perturbation"], keep="first")
        )
        grouped = (
            dedup.groupby("baseline_type", as_index=False)
            .agg(
                pearson_r=("performance_baseline_pearson_r", "mean"),
                l2=("performance_baseline_l2", "mean"),
            )
            .rename(columns={"baseline_type": "baseline"})
        )
        grouped["dataset"] = dataset
        grouped["analysis_type"] = "pseudobulk"
        rows.append(grouped[["dataset", "baseline", "pearson_r", "l2", "analysis_type"]])

    if not rows:
        return pd.DataFrame(columns=["dataset", "baseline", "pearson_r", "l2", "analysis_type"])
    return pd.concat(rows, ignore_index=True)


def load_single_cell_lsft_results(results_dir: Path) -> pd.DataFrame:
    """Load single-cell LSFT summaries from per-baseline summary CSV files."""
    files = sorted(results_dir.glob("single_cell_analysis/*/lsft/lsft_single_cell_summary_*_lpm_*.csv"))
    rows = []
    for file_path in files:
        dataset = _normalize_dataset_name(file_path.parent.parent.name)
        df = pd.read_csv(file_path)
        required = {
            "baseline_type",
            "top_pct",
            "pert_mean_baseline_r",
            "pert_mean_lsft_r",
            "pert_mean_delta_r",
        }
        missing = required.difference(df.columns)
        if missing:
            logger.warning("Skipping %s (missing columns: %s)", file_path, sorted(missing))
            continue

        out = df[
            [
                "baseline_type",
                "top_pct",
                "pert_mean_baseline_r",
                "pert_mean_lsft_r",
                "pert_mean_delta_r",
            ]
        ].copy()
        out = out.rename(
            columns={
                "baseline_type": "baseline",
                "pert_mean_baseline_r": "baseline_r",
                "pert_mean_lsft_r": "lsft_r",
                "pert_mean_delta_r": "delta_r",
            }
        )
        out["dataset"] = dataset
        out["analysis_type"] = "single_cell"
        rows.append(out)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "baseline",
                "top_pct",
                "baseline_r",
                "lsft_r",
                "delta_r",
                "analysis_type",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_pseudobulk_lsft_results(results_dir: Path) -> pd.DataFrame:
    """Load pseudobulk LSFT means from LSFT combined result files."""
    rows = []
    for combined_file in _find_lsft_combined_files(results_dir):
        dataset = _normalize_dataset_name(combined_file.parent.name)
        df = pd.read_csv(combined_file)
        required = {
            "baseline_type",
            "top_pct",
            "performance_baseline_pearson_r",
            "performance_local_pearson_r",
            "improvement_pearson_r",
        }
        missing = required.difference(df.columns)
        if missing:
            logger.warning("Skipping %s (missing columns: %s)", combined_file, sorted(missing))
            continue

        grouped = (
            df.groupby(["baseline_type", "top_pct"], as_index=False)
            .agg(
                baseline_r=("performance_baseline_pearson_r", "mean"),
                lsft_r=("performance_local_pearson_r", "mean"),
                delta_r=("improvement_pearson_r", "mean"),
            )
            .rename(columns={"baseline_type": "baseline"})
        )
        grouped["dataset"] = dataset
        grouped["analysis_type"] = "pseudobulk"
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "baseline",
                "top_pct",
                "baseline_r",
                "lsft_r",
                "delta_r",
                "analysis_type",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_single_cell_logo_results(results_dir: Path) -> pd.DataFrame:
    """Load single-cell LOGO summaries (uses `logo_fixed` when available)."""
    rows = []
    for dataset in ("adamson", "k562", "rpe1"):
        fixed = (
            results_dir
            / "single_cell_analysis"
            / dataset
            / "logo_fixed"
            / f"logo_single_cell_summary_{dataset}_Transcription.csv"
        )
        fallback = (
            results_dir
            / "single_cell_analysis"
            / dataset
            / "logo"
            / f"logo_single_cell_summary_{dataset}_Transcription.csv"
        )
        source = fixed if fixed.exists() else fallback
        if not source.exists():
            logger.warning("Single-cell LOGO summary not found for %s", dataset)
            continue

        df = pd.read_csv(source)
        required = {"baseline_type", "pert_mean_pearson_r", "pert_mean_l2"}
        missing = required.difference(df.columns)
        if missing:
            logger.warning("Skipping %s (missing columns: %s)", source, sorted(missing))
            continue

        out = df[["baseline_type", "pert_mean_pearson_r", "pert_mean_l2"]].copy()
        out = out.rename(
            columns={
                "baseline_type": "baseline",
                "pert_mean_pearson_r": "pearson_r",
                "pert_mean_l2": "l2",
            }
        )
        out["dataset"] = dataset
        out["analysis_type"] = "single_cell_logo"
        rows.append(out[["dataset", "baseline", "pearson_r", "l2", "analysis_type"]])

    if not rows:
        return pd.DataFrame(columns=["dataset", "baseline", "pearson_r", "l2", "analysis_type"])
    return pd.concat(rows, ignore_index=True)


def load_pseudobulk_logo_results(results_dir: Path) -> pd.DataFrame:
    """Load pseudobulk LOGO summary JSONs from functional class holdout runs."""
    files = sorted(
        results_dir.glob("goal_3_prediction/functional_class_holdout_resampling/*/*_summary.json")
    )
    rows = []
    for file_path in files:
        dataset = _normalize_dataset_name(file_path.parent.name)
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for baseline, baseline_payload in payload.items():
            if not baseline.startswith("lpm_"):
                continue
            pearson_stats = baseline_payload.get("pearson_r", {})
            l2_stats = baseline_payload.get("l2", {})
            pearson_mean = pearson_stats.get("mean")
            l2_mean = l2_stats.get("mean")
            if pearson_mean is None or l2_mean is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "pearson_r": pearson_mean,
                    "l2": l2_mean,
                    "analysis_type": "pseudobulk_logo",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["dataset", "baseline", "pearson_r", "l2", "analysis_type"])
    return pd.DataFrame(rows)


def create_comparison_summaries(
    pseudobulk_baseline: pd.DataFrame,
    single_cell_baseline: pd.DataFrame,
    pseudobulk_lsft: pd.DataFrame,
    single_cell_lsft: pd.DataFrame,
    pseudobulk_logo: pd.DataFrame,
    single_cell_logo: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create consolidated output tables in `aggregated_results/`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_baselines = pd.concat([pseudobulk_baseline, single_cell_baseline], ignore_index=True)
    combined_baselines = combined_baselines.sort_values(["analysis_type", "dataset", "baseline"]).reset_index(
        drop=True
    )
    combined_baselines.to_csv(output_dir / "baseline_performance_all_analyses.csv", index=False)
    logger.info("Wrote baseline_performance_all_analyses.csv (%d rows)", len(combined_baselines))

    best = combined_baselines.loc[
        combined_baselines.groupby(["dataset", "analysis_type"])["pearson_r"].idxmax()
    ][["dataset", "analysis_type", "baseline", "pearson_r"]].copy()
    best = best.rename(columns={"baseline": "best_baseline"})
    best = best.sort_values(["analysis_type", "dataset"]).reset_index(drop=True)
    best.to_csv(output_dir / "best_baseline_per_dataset.csv", index=False)
    logger.info("Wrote best_baseline_per_dataset.csv (%d rows)", len(best))

    if not pseudobulk_baseline.empty and not single_cell_baseline.empty:
        pb = pseudobulk_baseline[["dataset", "baseline", "pearson_r"]].rename(
            columns={"pearson_r": "pseudobulk_r"}
        )
        sc = single_cell_baseline[["dataset", "baseline", "pearson_r"]].rename(
            columns={"pearson_r": "single_cell_r"}
        )
        comp = pd.merge(pb, sc, on=["dataset", "baseline"], how="outer")
        comp.loc[:, "delta"] = comp["single_cell_r"] - comp["pseudobulk_r"]
        comp = comp.sort_values(["dataset", "baseline"]).reset_index(drop=True)
        comp.to_csv(output_dir / "baseline_comparison_pseudobulk_vs_single_cell.csv", index=False)
        logger.info("Wrote baseline_comparison_pseudobulk_vs_single_cell.csv (%d rows)", len(comp))

    # Keep legacy compatibility: this file is intentionally single-cell LSFT only.
    if not single_cell_lsft.empty:
        sc_lsft_summary = (
            single_cell_lsft.groupby(["dataset", "baseline"], as_index=False)
            .agg(
                mean_delta_r=("delta_r", "mean"),
                max_delta_r=("delta_r", "max"),
                mean_baseline_r=("baseline_r", "mean"),
                mean_lsft_r=("lsft_r", "mean"),
            )
            .sort_values(["dataset", "baseline"])
            .reset_index(drop=True)
        )
        sc_lsft_summary.to_csv(output_dir / "lsft_improvement_summary.csv", index=False)
        logger.info("Wrote lsft_improvement_summary.csv (%d rows)", len(sc_lsft_summary))

    if not pseudobulk_lsft.empty:
        pb_lsft_summary = (
            pseudobulk_lsft.groupby(["dataset", "baseline"], as_index=False)
            .agg(
                mean_delta_r=("delta_r", "mean"),
                max_delta_r=("delta_r", "max"),
                mean_baseline_r=("baseline_r", "mean"),
                mean_lsft_r=("lsft_r", "mean"),
            )
            .sort_values(["dataset", "baseline"])
            .reset_index(drop=True)
        )
        pb_lsft_summary.to_csv(output_dir / "lsft_improvement_summary_pseudobulk.csv", index=False)
        logger.info("Wrote lsft_improvement_summary_pseudobulk.csv (%d rows)", len(pb_lsft_summary))

    combined_logo = pd.concat([single_cell_logo, pseudobulk_logo], ignore_index=True)
    combined_logo = combined_logo.sort_values(["analysis_type", "dataset", "baseline"]).reset_index(drop=True)
    combined_logo.to_csv(output_dir / "logo_generalization_all_analyses.csv", index=False)
    logger.info("Wrote logo_generalization_all_analyses.csv (%d rows)", len(combined_logo))


def create_engineer_summary_report(output_dir: Path) -> None:
    """Write a compact guide for aggregated outputs."""
    report_path = output_dir / "engineer_analysis_guide.md"
    report = f"""# Aggregated Research Results

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files

- `baseline_performance_all_analyses.csv`: Baseline performance for single-cell and pseudobulk.
- `best_baseline_per_dataset.csv`: Best baseline per dataset and analysis type.
- `baseline_comparison_pseudobulk_vs_single_cell.csv`: Direct baseline comparison across resolutions.
- `lsft_improvement_summary.csv`: Single-cell LSFT lift summary (legacy compatibility).
- `lsft_improvement_summary_pseudobulk.csv`: Pseudobulk LSFT lift summary.
- `logo_generalization_all_analyses.csv`: LOGO extrapolation results (single-cell + pseudobulk).

## Notes

- All tables are generated from files under `results/`; no hardcoded metric values.
- Dataset aliases are normalized to `adamson`, `k562`, `rpe1`.
- For Adamson single-cell LOGO, `logo_fixed` is preferred when available.
"""
    report_path.write_text(report, encoding="utf-8")
    logger.info("Wrote engineer_analysis_guide.md")


def main() -> None:
    results_dir = Path("results")
    output_dir = Path("aggregated_results")

    logger.info("=" * 60)
    logger.info("Aggregating research results from canonical artifacts")
    logger.info("=" * 60)

    pb_baseline = load_pseudobulk_baseline_results(results_dir)
    pb_lsft = load_pseudobulk_lsft_results(results_dir)
    pb_logo = load_pseudobulk_logo_results(results_dir)

    sc_baseline = load_single_cell_baseline_results(results_dir)
    sc_lsft = load_single_cell_lsft_results(results_dir)
    sc_logo = load_single_cell_logo_results(results_dir)

    logger.info("Loaded pseudobulk baseline rows: %d", len(pb_baseline))
    logger.info("Loaded pseudobulk LSFT rows: %d", len(pb_lsft))
    logger.info("Loaded pseudobulk LOGO rows: %d", len(pb_logo))
    logger.info("Loaded single-cell baseline rows: %d", len(sc_baseline))
    logger.info("Loaded single-cell LSFT rows: %d", len(sc_lsft))
    logger.info("Loaded single-cell LOGO rows: %d", len(sc_logo))

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

    logger.info("=" * 60)
    logger.info("Aggregation complete")
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

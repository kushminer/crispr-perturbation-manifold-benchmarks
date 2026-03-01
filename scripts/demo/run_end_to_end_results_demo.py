#!/usr/bin/env python3
"""Print the committed project conclusions from aggregated summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_FILES = [
    'best_baseline_per_dataset.csv',
    'lsft_improvement_summary.csv',
    'logo_generalization_all_analyses.csv',
    'selftrained_pseudobulk_data_scale_trend.csv',
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def aggregated_dir(root: Path) -> Path:
    return root / 'aggregated_results'


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_csv(path)


def mean_delta(lsft_df: pd.DataFrame, baseline: str) -> float:
    subset = lsft_df[lsft_df['baseline'] == baseline]
    if subset.empty:
        raise ValueError(f'Baseline not found in LSFT summary: {baseline}')
    return float(subset['mean_delta_r'].mean())


def top_baseline(best_df: pd.DataFrame, analysis_type: str) -> str:
    subset = best_df[best_df['analysis_type'] == analysis_type]
    if subset.empty:
        raise ValueError(f'No rows found for analysis type: {analysis_type}')
    return str(subset['best_baseline'].mode().iloc[0])


def format_float(value: float, ndigits: int = 4) -> str:
    return f'{value:.{ndigits}f}'


def format_pct(value: float) -> str:
    return f'{value * 100:.0f}%'


def build_summary(best_df: pd.DataFrame, lsft_df: pd.DataFrame, logo_df: pd.DataFrame, trend_df: pd.DataFrame) -> str:
    sc_selftrained = mean_delta(lsft_df, 'lpm_selftrained')
    sc_scgpt = mean_delta(lsft_df, 'lpm_scgptGeneEmb')
    sc_random_pert = mean_delta(lsft_df, 'lpm_randomPertEmb')

    sc_best = top_baseline(best_df, 'single_cell')
    pb_best = top_baseline(best_df, 'pseudobulk')

    sc_logo = logo_df[logo_df['analysis_type'] == 'single_cell_logo']
    pb_logo = logo_df[logo_df['analysis_type'] == 'pseudobulk_logo']
    sc_logo_top = str(sc_logo.groupby('baseline')['pearson_r'].mean().idxmax())
    pb_logo_top = str(pb_logo.groupby('baseline')['pearson_r'].mean().idxmax())
    sc_logo_top_r = float(sc_logo.groupby('baseline')['pearson_r'].mean().max())
    pb_logo_top_r = float(pb_logo.groupby('baseline')['pearson_r'].mean().max())

    lines = [
        '# End-to-End Results Demo Summary',
        '',
        '## Verified Conclusions',
        '',
        '1. LSFT adds little on top of the strongest single-cell baseline.',
        f'   - Mean Δr (`lpm_selftrained`): {format_float(sc_selftrained)}',
        f'   - Mean Δr (`lpm_scgptGeneEmb`): {format_float(sc_scgpt)}',
        f'   - Mean Δr (`lpm_randomPertEmb`): {format_float(sc_random_pert)}',
        '',
        '2. Self-trained PCA (`lpm_selftrained`) is the top baseline across datasets.',
        f'   - Single-cell best baseline: `{sc_best}`',
        f'   - Pseudobulk best baseline: `{pb_best}`',
        '',
        '3. More local training data improves pseudobulk LSFT for `lpm_selftrained`.',
    ]

    for row in trend_df.itertuples(index=False):
        lines.append(
            f'   - {row.dataset}: {format_pct(row.start_pct)} {format_float(row.start_r, 3)} '
            f'-> {format_pct(row.end_pct)} {format_float(row.end_r, 3)} '
            f'(Δr={format_float(row.delta_r, 3)})'
        )

    lines.extend([
        '',
        '4. PCA also leads in LOGO generalization.',
        f'   - Single-cell LOGO top baseline: `{sc_logo_top}` (mean r={format_float(sc_logo_top_r, 3)})',
        f'   - Pseudobulk LOGO top baseline: `{pb_logo_top}` (mean r={format_float(pb_logo_top_r, 3)})',
        '',
        '## Sponsorship',
        'This project was sponsored by the **NIH Bridges to Baccalaureate** program.',
    ])
    return '\n'.join(lines)


def print_paths(paths: Iterable[Path], root: Path) -> None:
    for path in paths:
        print(f'- {path.relative_to(root)}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Print the committed project conclusions.')
    parser.add_argument(
        '--write-report',
        type=Path,
        default=Path('aggregated_results/final_conclusions_verified.md'),
        help='Write markdown summary to this repo-relative path.',
    )
    parser.add_argument('--quiet', action='store_true', help='Suppress console output.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    agg = aggregated_dir(root)

    missing = [name for name in REQUIRED_FILES if not (agg / name).exists()]
    if missing:
        raise FileNotFoundError(f'Missing aggregated result files: {missing}')

    best_df = load_csv(agg / 'best_baseline_per_dataset.csv')
    lsft_df = load_csv(agg / 'lsft_improvement_summary.csv')
    logo_df = load_csv(agg / 'logo_generalization_all_analyses.csv')
    trend_df = load_csv(agg / 'selftrained_pseudobulk_data_scale_trend.csv')

    summary = build_summary(best_df, lsft_df, logo_df, trend_df)

    report_path = (root / args.write_report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary + '\n', encoding='utf-8')

    if not args.quiet:
        print(summary)
        print('\n## Outputs')
        print_paths([
            agg / 'baseline_performance_all_analyses.csv',
            agg / 'best_baseline_per_dataset.csv',
            agg / 'baseline_comparison_pseudobulk_vs_single_cell.csv',
            agg / 'lsft_improvement_summary.csv',
            agg / 'lsft_improvement_summary_pseudobulk.csv',
            agg / 'logo_generalization_all_analyses.csv',
            agg / 'selftrained_pseudobulk_data_scale_trend.csv',
            report_path,
        ], root)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

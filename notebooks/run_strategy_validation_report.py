from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DISPLAY_NAMES = {
    "fast_ma_churn": "Fast MA crossover churn (stocks)",
    "micro_momentum_flip": "Micro-momentum threshold flip (stocks)",
    "crypto_fast_ema_long_only": "Fast EMA long-only trend (crypto)",
    "crypto_regime_trend": "Regime-aware EMA trend (crypto)",
    "cross_sectional_aggressive": "Aggressive cross-sectional intraday reversal",
    "cross_sectional_vol_targeted": "Vol-targeted cross-sectional reversal",
}


def run_single_asset(output_dir: Path, refresh_data: bool, cost_bps: float) -> None:
    cmd = [
        sys.executable,
        "notebooks/validate_single_asset_strategies.py",
        "--output-dir",
        str(output_dir),
        "--stock-cost-bps",
        str(cost_bps),
        "--crypto-cost-bps",
        str(cost_bps),
    ]
    if refresh_data:
        cmd.append("--refresh-data")
    subprocess.run(cmd, check=True)


def run_cross_sectional(output_dir: Path, refresh_data: bool, cost_bps: float) -> None:
    cmd = [
        sys.executable,
        "notebooks/validate_cross_sectional_strategies.py",
        "--output-dir",
        str(output_dir),
        "--cost-bps",
        str(cost_bps),
    ]
    if refresh_data:
        cmd.append("--refresh-data")
    subprocess.run(cmd, check=True)


def render_markdown(ranked: pd.DataFrame) -> str:
    lines = [
        "# Strategy Validation Report",
        "",
        "Vectorized pre-backtest check across proposed paper-trading strategy ideas.",
        "",
        "## Ranked Results (higher net_sharpe first)",
        "",
        "| Rank | Strategy | Net Sharpe | Net Ann Return | Net Max Drawdown | Avg Turnover | Break-even Cost (bps) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for idx, row in ranked.reset_index(drop=True).iterrows():
        name = str(row["display_name"])
        lines.append(
            f"| {idx + 1} | {name} | {row['net_sharpe']:.3f} | {row['net_ann_return']:.3f} | "
            f"{row['net_max_drawdown']:.3f} | {row['avg_turnover']:.4f} | {row['break_even_cost_bps']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a vectorized sanity check, not a broker-accurate simulation.",
            "- Metrics are sensitive to data quality/window and omit broker lifecycle rejects.",
            "- Use this report to down-select candidates before running full `run_backtest.py` parity runs.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all notebook-style vectorized validations and generate a report."
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/results",
        help="Directory for all intermediate and report outputs.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh market data via download scripts before validation.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Linear transaction cost in bps for all validations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_single_asset(
        output_dir=output_dir,
        refresh_data=args.refresh_data,
        cost_bps=args.cost_bps,
    )
    run_cross_sectional(
        output_dir=output_dir,
        refresh_data=args.refresh_data,
        cost_bps=args.cost_bps,
    )

    single_summary_path = output_dir / "single_asset_portfolio_summary.csv"
    cross_summary_path = output_dir / "cross_sectional_summary.csv"

    single = pd.read_csv(single_summary_path)
    cross = pd.read_csv(cross_summary_path)
    merged = pd.concat([single, cross], ignore_index=True, sort=False)
    merged["display_name"] = merged["strategy"].map(DISPLAY_NAMES).fillna(merged["strategy"])

    ranked = merged.sort_values(["net_sharpe", "net_ann_return"], ascending=False)
    ranked_path = output_dir / "strategy_validation_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    md_text = render_markdown(ranked)
    md_path = output_dir / "strategy_validation_report.md"
    md_path.write_text(md_text)

    print("Strategy validation report complete")
    print(f"Ranked CSV: {ranked_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()

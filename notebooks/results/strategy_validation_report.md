# Strategy Validation Report

Vectorized pre-backtest check across proposed paper-trading strategy ideas.

## Ranked Results (higher net_sharpe first)

| Rank | Strategy | Net Sharpe | Net Ann Return | Net Max Drawdown | Avg Turnover | Break-even Cost (bps) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Fast EMA long-only trend (crypto) | 14.748 | 5.100 | -0.031 | 0.0527 | 1.84 |
| 2 | Vol-targeted cross-sectional reversal | 7.472 | 9.307 | -0.224 | 4.3687 | 0.22 |
| 3 | Aggressive cross-sectional intraday reversal | 6.304 | 0.751 | -0.017 | 0.4277 | 0.18 |
| 4 | Fast MA crossover churn (stocks) | -2.971 | -0.762 | -0.059 | 0.1322 | -0.59 |
| 5 | Micro-momentum threshold flip (stocks) | -5.821 | -1.424 | -0.078 | 0.2864 | -0.51 |

## Notes

- This is a vectorized sanity check, not a broker-accurate simulation.
- Metrics are sensitive to data quality/window and omit broker lifecycle rejects.
- Use this report to down-select candidates before running full `run_backtest.py` parity runs.

# CryptoCompetitionStrategy Backtest Performance

## Run Metadata
- Date: 2026-02-20
- Command:
  - `uv run python run_backtest.py --strategy crypto_competition --csv data/legacy/clean_data_crypto/BTC-USD_2m_ohlcv.csv`
- Data file: `data/legacy/clean_data_crypto/BTC-USD_2m_ohlcv.csv`
- Data bars: 21,868 (2-minute BTC/USD)
- Asset mode: `crypto` (auto-resolved)
- Paper parity: enabled (default)
- Trade log verbosity: off (default)
- Wall-clock runtime: 270.93 seconds

## Summary Metrics
- Equity data points: 21,868
- Trades executed: 112
- Final portfolio value: 49,999.65
- PnL: -0.35
- Sharpe: -1.64
- Max drawdown: -0.0000
- Win rate: 42.11%

## Interpretation
- This run is approximately flat-to-negative on PnL with negative Sharpe.
- Drawdown is tiny because position sizes are small after converting strategy notional targets into BTC units.
- Strategy edge is not yet strong in this backtest configuration.

## Crypto Backtester Fixes Applied
- Single-asset backtester now supports `asset_class='crypto'`.
- Crypto `target_qty` is interpreted as notional and converted to units (`qty = target_notional / price`) with 1e-6 precision.
- Fractional crypto quantities are supported end-to-end in order/position accounting.
- Offline backtester now blocks opening crypto shorts to align with live behavior.
- `--paper-parity` is now enabled by default (`--no-paper-parity` to disable).

## Notes
- The matching engine uses stochastic partial-fill simulation, so exact metrics can vary run-to-run.
- For deterministic comparisons, add a fixed RNG seed path in the matching engine/backtest runner.

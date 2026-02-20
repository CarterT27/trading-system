# Crypto Competition Strategy Integration Report

## Scope
This report documents the production integration of four optimized components into a single tradeable crypto strategy:
- `meta_model`
- `quality_filter`
- `dynamic_sizing`
- `exit_analysis`

Implemented strategy class: `CryptoCompetitionStrategy` in `strategies/strategy_base.py`.

## Source of Parameter Choices
Parameter choices were taken from walk-forward experiments in:
- `notebooks/results/crypto_experiment_suite_20260220_optimized_v2/fold_results.csv`
- `notebooks/results/crypto_experiment_suite_20260220_optimized_v2/summary.csv`
- `notebooks/results/crypto_experiment_suite_20260220_optimized_v2/recommendations.csv`

Decision rule used for deployable defaults:
- Prefer robust/majority settings across folds.
- Keep settings near best-performing fold-level candidates.
- Avoid fragile one-off values when fold disagreement is high.

## OOS Evidence Snapshot (from summary.csv)
Compared to `base_regime_tuned`:
- `meta_model`: strong improvement (`pooled_total_return=0.1753`, `pooled_net_sharpe=36.11`)
- `quality_filter`: strong improvement (`pooled_total_return=0.0904`, `pooled_net_sharpe=16.36`)
- `dynamic_sizing`: positive uplift (`pooled_total_return=0.0241`, `pooled_net_sharpe=3.75`)
- `exit_analysis`: positive uplift (`pooled_total_return=0.0184`, `pooled_net_sharpe=3.04`)

Recommendation file labels:
- `meta_model`: `add_now`
- `quality_filter`: `add_now`
- `dynamic_sizing`: `add_now`
- `exit_analysis`: `paper_test`

Even though `exit_analysis` was tagged `paper_test`, it was still included because the request was to integrate all four components with best parameters.

## Final Deployed Parameters
### Base regime (existing framework baseline for this integrated class)
- `short_window=20`
- `long_window=120`
- `slope_lookback=120`
- `volatility_window=1440`
- `volatility_quantile=0.70`

### Quality filter
- `spread_z_threshold=0.0`
- `trend_tstat_threshold=0.0`
- `vol_ratio_max=1.2`
- `require_ret5_positive=True`
- `require_ret30_positive=False`

Rationale:
- `vol_ratio_max=1.2` and `require_ret5_positive=True` were consistent winners.
- Stricter spread/tstat thresholds were not consistently superior across folds.

### Meta model (RandomForest)
- `meta_n_estimators=350`
- `meta_max_depth=4`
- `meta_min_samples_leaf=15`
- `meta_max_features='sqrt'`
- `meta_prob_threshold=0.50`

Rationale:
- Majority setting from fold winners, balancing capacity and stability.

### Dynamic sizing
- `sizing_scale=1.5`
- `sizing_offset=0.3`
- `sizing_vol_exponent=0.75`
- `sizing_smooth_span=1`
- `sizing_min_size=0.05`

Rationale:
- Captures strongest repeated values while avoiding over-smoothed behavior.

### Exit analysis
- `exit_mode='combo'`
- `exit_time_bars=180`
- `exit_trail_stop=0.008`
- `exit_profit_take=None`
- `exit_min_hold_bars=0`

Rationale:
- Most frequent high-performing exit pattern in the optimized fold set.

## Lookahead-Bias Controls
### Causal feature construction
- Volatility threshold is shifted by one bar: `vol_gate_threshold = rolling_quantile(...).shift(1)`.
- All rolling features use only past/current bars.

### Causal meta-model training
- Forward return label: `fwd_ret = Close.pct_change().shift(-1)` and terminal unknown label remains `NaN` (not force-filled).
- At timestamp index `i`, model training only uses samples with index `< i`.
- Periodic refit schedule (`meta_refit_interval`) reduces compute while preserving causality.
- If data is insufficient or class distribution collapses, model falls back to constant probability.

### Causal exit logic
- Exit state machine uses only current and prior state (`time`, `trail`, `profit_take`, `min_hold`).
- No future bars referenced.

## Combined Decision Flow (implemented)
1. Build regime features and `base_long`.
2. Apply quality filter gate.
3. Compute causal meta probability and apply threshold gate.
4. Compute dynamic size signal from trend strength and vol normalization.
5. Apply exit state machine with lockout behavior for forced exits.
6. Convert desired position-notional changes into explicit buy/sell order signals.

## Practical Execution Choice
Dynamic size is held per trade once entered (entry-time freeze) and fully unwound on exit signal.

Reason:
- Keeps order flow compatible with current execution path while still using dynamic sizing to scale trade risk.
- Avoids continuous intratrade resizing churn in the current live runner logic.

## Code Changes
- Added `CryptoCompetitionStrategy` to `strategies/strategy_base.py`.
- Registered aliases in `strategies/__init__.py`:
  - `crypto_competition`
  - `crypto_meta`
  - `crypto_comp`
- Wired constructor handling in:
  - `run_live.py`
  - `run_backtest.py`

## Validation Run Results
### Build/syntax
- `python -m compileall strategies run_live.py run_backtest.py` passed.
- `uv run pytest -q` passed (`35 passed`).

### Smoke test (real crypto CSV tail)
- Strategy executed on 3,500 bars from `BTC-USD_2m_data.csv`.
- Emitted valid signals and quantities without runtime errors.

### Causality audit (truncation invariance)
- Recomputed strategy on truncated histories and compared with full-history outputs at identical timestamps.
- Result: `causal_check PASS`.

## How To Use
Backtest:
- `uv run python run_backtest.py --strategy crypto_competition --csv <path_to_ohlcv_csv>`

Live/paper loop:
- `uv run python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_competition --live`

List aliases:
- `uv run python -c "from strategies import list_strategies; print('\n'.join([s for s in list_strategies() if 'crypto' in s]))"`

## Risks / Known Constraints
- Meta model retraining increases compute cost versus simple EMA-only logic.
- Exit settings are optimized for competition assumptions (`cost_bps=0`) and may need retuning with real frictions.
- Cross-asset overlay was not integrated in this class by design (single-symbol strategy class).

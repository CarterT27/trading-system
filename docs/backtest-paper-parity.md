# Backtest vs Alpaca Paper (Free/Basic) Parity Map

Last verified: **February 20, 2026**.

This document maps the current behavior of this repository against Alpaca paper trading on the free/basic plan, with emphasis on:
- what can be shorted,
- how much can be shorted,
- which equities are tradable,
- and what to control so backtests behave closer to paper trading.

## Scope
- Backtest paths: `run_backtest.py`, `core/backtester.py`, `core/multi_asset_backtester.py`, `core/order_manager.py`, `core/matching_engine.py`
- Paper paths: `run_live.py`, `core/alpaca_trader.py`, `core/multi_asset_trader.py`, `pipeline/alpaca.py`

## Direct answers

### What can we short?
- **Single-symbol backtest**:
  - default mode: any symbol present in your CSV.
  - parity mode (`--paper-parity` + optional `--asset-flags-csv`): enforces `tradable` and opening-short checks (`shortable`, optional `easy_to_borrow`).
- **Multi-asset backtest**:
  - default mode: any symbol in panel data.
  - parity mode: same eligibility checks as single-symbol mode.
- **Alpaca paper (equities)**: shorting is allowed only when account/asset constraints pass (margin-enabled account, asset shortable/borrowable rules).
- **Alpaca paper (crypto)**: no shorting.

### How much can we short?
- **Single-symbol backtest**: capped by local engine max short position (default 1,000 shares in `run_backtest.py`).
- **Multi-asset backtest**:
  - supports `--max-short-notional` and `--max-notional-per-order`.
  - parity mode also applies short-open buying-power checks using the 1.03 valuation rule.
- **Alpaca paper**: constrained by buying power and margin rules. Alpaca checks opening short order value against buying power and applies margin/PDT protections.

#### Alpaca short sizing constraints to model
- Opening short order valuation uses: `max(limit_price, ask_price * 1.03) * quantity` for buying-power checks.
- Margin account buying power is tied to equity level:
  - `< $2,000`: no margin (limited buying power)
  - `>= $2,000`: typically up to `2x` overnight buying power (Reg-T)
  - `>= $25,000` (PDT flagged): typically up to `4x` intraday day-trading buying power

### Which equities can we trade?
- **Backtests**: any symbol in your historical files.
- **Alpaca paper**: tradable assets returned by `/v2/assets` and subject to account/data entitlements and broker-side protections.

## Parity matrix

| Topic | Current backtest behavior | Alpaca paper (free/basic) behavior | Parity risk |
|---|---|---|---|
| Tradable universe | Default mode trades whatever is in CSV/panel. Parity mode can enforce asset flags from `--asset-flags-csv`. | Asset must be active/tradable in Alpaca assets model. | Residual risk if flags CSV is stale or omitted. |
| Short eligibility | Parity mode enforces opening-short checks (`shortable`, optional `easy_to_borrow`); close/reduce still allowed. | New shorts depend on short/margin eligibility and borrow status. | Lower than before; still depends on input flags freshness. |
| Short sizing | Single-symbol max short shares + parity short BP checks; multi-asset also supports `--max-short-notional` and per-order notional cap. | Short opens consume buying power and are margin constrained. | Improved; still not full broker-level parity. |
| Buying power checks | Parity mode applies short-open valuation rule `max(limit_price, ref_price * 1.03) * qty`; reservation semantics are modeled for accepted orders. | Broker performs buying-power checks for long/short orders, including open orders. | Improved reject parity; PDT/DTMC still out of scope. |
| PDT/DTMC protections | Not simulated. | Enforced in paper, including PDT checks. | Strategies can pass backtest and fail in paper. |
| Order lifecycle | Single-symbol backtest uses synthetic order book + random fill outcomes; multi-asset backtest fills immediately when signaled. | Orders follow broker lifecycle/statuses; non-marketable limits wait. | Fill timing/quantity mismatch. |
| Fill assumptions | Single-symbol: 70% full, 20% partial, 10% cancel (engine random). Multi-asset: immediate fills at close/limit. | Paper fills against simulated NBBO assumptions and can partial-fill. | PnL path and turnover mismatch. |
| Liquidity constraints | No NBBO size/liquidity model. | Paper also ignores NBBO quantity sizing for fills. | Closer than live, but still model-dependent. |
| Order types/TIF actually used by this repo | Stocks: market/limit with `day`. Crypto (single-symbol): market/limit with `gtc`. Multi-asset paper path is stocks only. | Alpaca supports wider combinations, but with restrictions by asset/order type. | Strategy relying on unsupported combos will diverge. |
| Fractional equities | Not used in this code path (stock qty cast to int). | Alpaca supports fractional equities with constraints. | Fractional live behavior not represented in backtest. |
| Extended hours | No explicit Alpaca `extended_hours` handling in order submission. | Extended-hours execution requires specific order settings. | After-hours behavior can diverge. |
| Data feed | Repo defaults stock data feed to `iex` in live and download helpers. | Basic/free has limited real-time coverage (IEX for equities) and stricter historical limits. | Backtest data source mismatch can create signal drift. |
| Latest historical window | Not enforced by backtester. | Basic plan documents a latest historical-window limitation for equities. | Near-real-time backtests may see bars unavailable in free-plan workflows. |
| Corporate actions/dividends/fees | No explicit realistic brokerage event simulation. | Paper is a simulation and excludes/approximates some live effects. | Live deployment gap remains. |

## Biggest repo-specific limitations

1. Backtests do not enforce Alpaca PDT/DTMC protections.
2. Single-symbol backtest fill simulation is synthetic/random and not tied to real quote state.
3. Multi-asset backtest fills are immediate and optimistic relative to real order lifecycle.
4. Asset-flag gating quality depends on the freshness/completeness of your `--asset-flags-csv`.
5. Fractional stock behavior and extended-hours specifics are still not fully modeled.

## Implemented parity controls

### CLI flags

- `--paper-parity`
- `--asset-flags-csv`
- `--account-equity`
- `--buying-power-mode {disabled,multiplier,tiered}`
- `--buying-power-multiplier`
- `--reserve-open-orders`
- `--max-notional-per-order` (multi-asset)
- `--max-short-notional` (multi-asset)

### Reject semantics used in parity mode

- `reject_not_tradable`
- `reject_not_shortable`
- `reject_not_easy_to_borrow`
- `reject_short_open_buying_power`
- `reject_reserved_buying_power`

### Notes on behavior

- Opening-short valuation uses `max(limit_price, reference_price * 1.03) * qty`.
- Close/reduce orders are allowed even when opening-short flags would reject new shorts.
- Reservation lifecycle is tracked:
  - single-symbol via order logger `reservation` events (`reserve` and `release`),
  - multi-asset via `MultiAssetBacktester.reservation_events`.

## Parity-safe operating rules (recommended)

1. **Use stock-only strategies for parity with current multi-asset paper runner**.
2. **Use integer share sizing only** (already how stock paper path submits orders).
3. **Refresh your universe from Alpaca daily** and keep only names that are short-ready:

```bash
python download_panel_data.py \
  --alpaca-universe \
  --only-shortable \
  --only-easy-to-borrow \
  --simple-symbols-only \
  --symbols-only \
  --symbols-out data/universe_long_short.csv
```

4. **Backtest and paper with the same symbol universe and feed assumptions** (`iex` for free/basic parity).
5. **Constrain strategy sizing conservatively** so expected gross exposure is below realistic buying power (especially for shorts).
6. **Assume paper rejects can still happen** due to account protections (PDT/margin/buying power) even if backtest accepts.
7. **Avoid relying on extended-hours behavior** unless you explicitly implement/order for it.
8. **Treat backtest-to-paper as a staging step, not proof of live equivalence**.

## Suggested “same-strategy” workflow

1. Build/refresh universe with Alpaca filters (`shortable` + `easy_to_borrow`).
2. Download panel bars with the same feed assumptions used in paper.
3. Run backtest on that exact universe/data.
4. Run paper with the same symbols and strategy parameters.
5. Compare rejects and fills from `logs/trades.csv` and tighten strategy constraints where paper rejects exceed tolerance.

## Source references (Alpaca docs)
- Paper trading behavior and assumptions: https://docs.alpaca.markets/docs/paper-trading
- Market Data API plans (Basic vs Algo Trader Plus): https://docs.alpaca.markets/docs/about-market-data-api
- Margin and short selling rules: https://docs.alpaca.markets/docs/margin-and-short-selling
- User protection (PDT / DTMC): https://docs.alpaca.markets/docs/user-protection
- Working with assets (`/v2/assets`): https://docs.alpaca.markets/docs/working-with-assets
- Order behavior/time-in-force matrix: https://docs.alpaca.markets/v1.3/docs/orders-at-alpaca

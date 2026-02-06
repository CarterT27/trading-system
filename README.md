# Trading System

![](https://media1.tenor.com/m/JAJEdIBuT_gAAAAC/diegopatricio-patricio.gif)

## What this includes
- Alpaca paper-data download + cleaning helpers
- A paper-trading runner that submits orders through Alpaca
- A lightweight backtester with market data gateway, order book, order manager, and matching engine
- Automatic trade logging to `logs/trades.csv` and `logs/system.log`
- Built-in strategies:
  - **MovingAverageStrategy** (`ma`) - Simple moving average crossover
  - **TemplateStrategy** (`template`) - Momentum-based starter template
  - **CryptoTrendStrategy** (`crypto_trend`) - EMA trend follower for crypto (long-only)
  - **CrossSectionalPaperReversalStrategy** (`cross_sectional_reversal`) - Multi-symbol long/short intraday reversal portfolio strategy

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` file (only 2 fields required!)
```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

### 3. Run live trading
```bash
python run_live.py --symbol AAPL --strategy ma --live
```

Press `Ctrl+C` to stop and see performance summary (P&L, Sharpe ratio, win rate, etc.)

---

## Available Timeframes

| Timeframe | Description | Best For |
|-----------|-------------|----------|
| `1Min` | 1-minute bars | High-frequency, intraday trading |
| `5Min` | 5-minute bars | Short-term intraday |
| `15Min` | 15-minute bars | Intraday swing |
| `30Min` | 30-minute bars | Intraday swing |
| `1Hour` | 1-hour bars | Intraday/multi-day |
| `1Day` | Daily bars | Swing/position trading |

**Example usage:**
```bash
# Fast trading with 1-minute bars, checking every 5 seconds
python run_live.py --symbol AAPL --strategy ma --timeframe 1Min --sleep 5 --live

# Daily strategy
python run_live.py --symbol AAPL --strategy ma --timeframe 1Day --live
```

**Note:** Alpaca's minimum bar resolution is 1 minute. The `--sleep` parameter controls how often we check for new signals (can be as low as 1 second).

---

## CLI Reference

### List available strategies
```bash
python run_live.py --list-strategies
```

### Live trading
```bash
python run_live.py --symbol AAPL --strategy ma --live
```

### Live trading with debug logs in console
```bash
python run_live.py --symbol AAPL --strategy ma --live --debug
```

### Cross-sectional live portfolio strategy
```bash
python run_live.py --symbols-file data/universe.txt --strategy cross_sectional_reversal --timeframe 1Min --live --batch-size 100 --max-api-requests-per-minute 190
```

### Dry run (no real orders)
```bash
python run_live.py --symbol AAPL --strategy ma --dry-run --iterations 10
```

### Crypto trading
```bash
python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live
```

### Fast demo mode (check every 5 seconds)
```bash
python run_live.py --symbol AAPL --strategy template --timeframe 1Min --sleep 5 --live
```

### All options
| Option | Default | Description |
|--------|---------|-------------|
| `--symbol` | AAPL | Stock ticker or crypto symbol |
| `--asset-class` | stock | `stock` or `crypto` |
| `--strategy` | ma | Strategy name |
| `--timeframe` | 1Min | Bar timeframe |
| `--sleep` | 60 | Seconds between iterations |
| `--debug` | false | Show DEBUG logs in console during live/dry runs |
| `--position-size` | 10.0 | Shares (stocks) or USD (crypto) per trade |
| `--live` | false | Run continuously until Ctrl+C |
| `--dry-run` | false | Print decisions without placing orders |
| `--save-data` | false | Save market data to CSV |

---

## Backtest Mode

```bash
python run_backtest.py --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy ma --plot
```

Multi-asset backtest from panel data:
```bash
python run_backtest.py --panel-csv data/panels/us_equities_1min_panel.csv --symbols-file data/universe.txt --strategy cross_sectional_reversal
```

Cross-sectional portfolio backtest from panel data:
```bash
python run_backtest.py --panel-csv data/panels/us_equities_1min_panel.csv --symbols-file data/universe.txt --strategy cross_sectional_reversal
```

---

## Build Your Own Strategy

Edit `strategies/strategy_base.py` and add your class:

```python
class MyStrategy(Strategy):
    def __init__(self, lookback=20, position_size=10.0):
        self.lookback = lookback
        self.position_size = position_size

    def add_indicators(self, df):
        df['sma'] = df['Close'].rolling(self.lookback).mean()
        return df

    def generate_signals(self, df):
        df['signal'] = 0
        df.loc[df['Close'] > df['sma'], 'signal'] = 1
        df.loc[df['Close'] < df['sma'], 'signal'] = -1
        df['position'] = df['signal']
        df['target_qty'] = self.position_size
        return df
```

Then run:
```bash
python run_live.py --symbol AAPL --strategy mystrategy --live
```

---

## Logs and Output

All trades are logged to:
- `logs/trades.csv` - Trade records (timestamp, symbol, side, qty, price, P&L)
- `logs/system.log` - Full system log with debug info

Use `--debug` with `run_live.py` if you also want DEBUG lines printed to the terminal in real time.

When you stop with `Ctrl+C`, you'll see a performance summary:
```
SESSION SUMMARY
  Trades: 15 | Wins: 9 (60.0%)
  Net P&L: $+127.50
  Sharpe Ratio: 1.23
  Volatility: 0.82%
```

---

## Project Structure
```
trading-system/
  core/           # Trading engine, logger
  pipeline/       # Alpaca data helpers
  strategies/     # Strategy implementations
  logs/           # Trade logs (auto-created)
  data/           # Market data CSVs
  run_live.py     # Live trading CLI
  run_backtest.py # Backtesting CLI
```

## Fetch data with Alpaca
Use the notebooks in `notebooks/`:
- `notebooks/fetch_data_stock.ipynb`
- `notebooks/fetch_data_crypto.ipynb`

They download bars from Alpaca and save raw/clean CSVs to `data/`.

### Build a multi-symbol panel CSV
For cross-sectional research, use:
```bash
python download_panel_data.py --symbols AAPL,MSFT,NVDA,SPY --timeframe 1Min --limit 5000
```

To resolve tradable symbols directly from Alpaca and save a universe file:
```bash
python download_panel_data.py --alpaca-universe --symbols-only --symbols-out data/universe.txt
```

To generate a long/short-ready universe file (shortable + easy-to-borrow):
```bash
python download_panel_data.py --alpaca-universe --only-shortable --only-easy-to-borrow --symbols-only --symbols-out data/universe_long_short.csv
```

Optional filters for the Alpaca universe include:
- `--universe-exchanges NYSE,NASDAQ,ARCA`
- `--only-shortable`
- `--only-easy-to-borrow`
- `--fractionable-only`

Or from a symbols file:
```bash
python download_panel_data.py --symbols-file data/universe.txt --timeframe 1Min --start 2025-01-01 --end 2025-01-31
```

For large universes, set `--start`/`--end` and use batched requests (much faster than one request per symbol):
```bash
python download_panel_data.py --symbols-file data/universe.txt --start 2025-01-01 --end 2025-01-31 --batch-size 100
```

In ranged mode, the downloader paginates automatically until the full time window is fetched.

By default, the downloader enforces a conservative rate limiter (`190 req/min`) to stay below common Alpaca limits (`200 req/min`).

## Build your own strategy
Open `strategies/strategy_base.py` and edit `TemplateStrategy` (recommended), or add your own class.
The backtester expects:
- `signal`: 1 for buy, -1 for sell, 0 for no action.
- `target_qty`: the quantity to trade when a signal triggers.
- Optionally, `limit_price` if you want a limit price different from `Close`.

The Alpaca runner uses these same fields to submit paper orders.
To run your custom class from the CLI, give it a no-arg constructor and call (case-insensitive):
```
python run_backtest.py --csv data/AAPL_1Min_stock_alpaca_clean.csv --strategy MyStrategy
```
or
```
python run_live.py --symbol AAPL --asset-class stock --strategy MyStrategy --timeframe 1Min --live
```

Example signal logic inside `generate_signals`:
```python
df["signal"] = 0
buy = df["momentum"] > 0.01
sell = df["momentum"] < -0.01
df.loc[buy, "signal"] = 1
df.loc[sell, "signal"] = -1
df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
df["target_qty"] = df["position"].abs() * self.position_size
```

## Trade output format (live)
Each fill prints a line like:
```
2024-01-01 09:31:00 | BUY 10 AAPL @ 101.23 | order_id=1234 | net_pnl=+12.50
```

## Project structure
```
core/
pipeline/
strategies/
data/
notebooks/
.env
run_backtest.py
run_live.py
test_system.py
```

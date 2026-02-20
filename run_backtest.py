"""
Offline backtest runner for a CSV file.

Usage:
    python run_backtest.py --csv data\\AAPL_1Min_stock_alpaca_clean.csv --strategy ma

Replace "AAPL_1Min_stock_alpaca_clean.csv" with your desired CSV file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.asset_eligibility import AssetEligibilityFlags, parse_asset_flag_bool
from core.backtester import Backtester, PerformanceAnalyzer, plot_equity
from core.gateway import MarketDataGateway
from core.matching_engine import MatchingEngine
from core.multi_asset_backtester import MultiAssetBacktester
from core.order_book import OrderBook
from core.order_manager import OrderLoggingGateway, OrderManager
from core.paper_parity import BuyingPowerConfig, PaperParityConfig
from core.symbols import resolve_symbols
from strategies import (
    CrossSectionalPaperReversalStrategy,
    CryptoCompetitionStrategy,
    MovingAverageStrategy,
    TemplateStrategy,
    get_strategy_class,
)


DATA_DIR = Path("data")


def _parse_bool(value: object, *, default: bool = True) -> bool:
    return parse_asset_flag_bool(value, default=default)


def load_asset_flags_by_symbol(path: Path) -> dict[str, AssetEligibilityFlags]:
    df = pd.read_csv(path)
    if df.empty:
        return {}

    columns = {str(col).strip().lower(): str(col) for col in df.columns}
    symbol_col = columns.get("symbol") or columns.get("ticker")
    if symbol_col is None:
        raise ValueError("Asset flags CSV must include a symbol or ticker column.")

    tradable_col = columns.get("tradable")
    shortable_col = columns.get("shortable")
    etb_col = columns.get("easy_to_borrow")

    out: dict[str, AssetEligibilityFlags] = {}
    for _, row in df.iterrows():
        symbol = str(row[symbol_col]).strip().upper()
        if not symbol:
            continue
        out[symbol] = AssetEligibilityFlags(
            tradable=_parse_bool(
                row[tradable_col] if tradable_col is not None else True,
                default=True,
            ),
            shortable=_parse_bool(
                row[shortable_col] if shortable_col is not None else True,
                default=True,
            ),
            easy_to_borrow=_parse_bool(
                row[etb_col] if etb_col is not None else True,
                default=True,
            ),
        )
    return out


def _apply_min_symbol_bars_filter(
    panel_df: pd.DataFrame,
    symbol_col: str,
    min_symbol_bars: int,
) -> pd.DataFrame:
    if min_symbol_bars <= 0:
        return panel_df
    counts = panel_df.groupby(symbol_col).size()
    keep = counts[counts >= int(min_symbol_bars)].index
    keep_list = [str(x).upper() for x in keep.tolist()]
    symbols = panel_df[symbol_col].astype(str).str.upper()
    return panel_df[symbols.isin(keep_list)].copy()


def _build_auto_liquidity_universe(
    panel_df: pd.DataFrame,
    symbol_col: str,
    top_n: int,
    min_days: int,
    max_symbols: int,
) -> list[str]:
    if top_n <= 0:
        return []

    required = {"Datetime", "Close", "Volume", symbol_col}
    missing = [c for c in required if c not in panel_df.columns]
    if missing:
        raise ValueError(
            f"Liquidity universe filter requires columns {sorted(required)}; missing={missing}"
        )

    work = panel_df[["Datetime", "Close", "Volume", symbol_col]].copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], utc=True, errors="coerce")
    work = work.dropna(subset=["Datetime", "Close", "Volume", symbol_col]).copy()
    if work.empty:
        return []

    work["session_date"] = work["Datetime"].dt.date
    work["dollar_volume"] = work["Close"].astype(float) * work["Volume"].astype(float)

    daily = (
        work.groupby(["session_date", symbol_col], as_index=False)["dollar_volume"]
        .median()
        .rename(columns={"dollar_volume": "daily_dv_median"})
    )
    if daily.empty:
        return []

    daily["rank"] = daily.groupby("session_date")["daily_dv_median"].rank(
        ascending=False,
        method="first",
    )
    selected = daily[daily["rank"] <= float(top_n)].copy()
    if selected.empty:
        return []

    selected_counts = selected.groupby(symbol_col).size()
    min_days = max(1, int(min_days))
    keep_symbols = selected_counts[selected_counts >= min_days].index
    if len(keep_symbols) == 0:
        return []

    keep_list = [str(x).upper() for x in keep_symbols.tolist()]
    scores = (
        daily[daily[symbol_col].astype(str).str.upper().isin(keep_list)]
        .groupby(symbol_col)["daily_dv_median"]
        .median()
        .sort_values(ascending=False)
    )

    if max_symbols > 0:
        scores = scores.head(int(max_symbols))

    return [str(s).upper() for s in scores.index.tolist()]


def create_sample_data(path: Path, periods: int = 200) -> None:
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range(
                start="2024-01-01 09:30", periods=periods, freq="T"
            ),
            "Open": np.random.uniform(100, 105, periods),
            "High": np.random.uniform(105, 110, periods),
            "Low": np.random.uniform(95, 100, periods),
            "Close": np.random.uniform(100, 110, periods),
            "Volume": np.random.randint(1_000, 5_000, periods),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline CSV backtest.")
    parser.add_argument(
        "--csv", type=str, default="", help="Path to a CSV with OHLCV data."
    )
    parser.add_argument(
        "--panel-csv",
        type=str,
        default="",
        help="Path to a long-form multi-symbol panel CSV with Datetime,symbol,OHLCV.",
    )
    parser.add_argument(
        "--asset-class",
        choices=("auto", "stock", "crypto"),
        default="auto",
        help="Asset class for single-asset backtests (default: auto).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Default symbol when not using --symbols/--symbols-file (single-mode).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Optional comma-separated symbol filter for --panel-csv.",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default="",
        help="Optional .txt/.csv symbol filter file for --panel-csv.",
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=0,
        help="Optional cap on symbol count in panel mode (0 means all).",
    )
    parser.add_argument(
        "--min-symbol-bars",
        type=int,
        default=0,
        help="Drop symbols with fewer than this many bars in panel mode (0 disables).",
    )
    parser.add_argument(
        "--auto-universe-top-n",
        type=int,
        default=0,
        help="Build dynamic universe from top-N daily median dollar volume (0 disables).",
    )
    parser.add_argument(
        "--auto-universe-min-days",
        type=int,
        default=5,
        help="For auto universe, symbol must appear in top-N for at least this many days.",
    )
    parser.add_argument(
        "--auto-universe-max-symbols",
        type=int,
        default=0,
        help="Optional cap on auto-selected universe size (0 means all passing symbols).",
    )
    parser.add_argument(
        "--strategy",
        default="ma",
        help="Strategy name (ma, template, or a class name).",
    )
    parser.add_argument(
        "--short-window", type=int, default=20, help="Short MA window (MA strategy)."
    )
    parser.add_argument(
        "--long-window", type=int, default=60, help="Long MA window (MA strategy)."
    )
    parser.add_argument(
        "--position-size", type=float, default=10.0, help="Per-trade position size."
    )
    parser.add_argument(
        "--momentum-lookback",
        type=int,
        default=14,
        help="Momentum lookback (template).",
    )
    parser.add_argument(
        "--buy-threshold", type=float, default=0.01, help="Buy threshold (template)."
    )
    parser.add_argument(
        "--sell-threshold", type=float, default=-0.01, help="Sell threshold (template)."
    )
    parser.add_argument(
        "--cs-lookback",
        type=int,
        default=15,
        help="Lookback minutes for cross-sectional reversal strategy.",
    )
    parser.add_argument(
        "--cs-hold",
        type=int,
        default=45,
        help="Hold window in minutes for cross-sectional reversal strategy.",
    )
    parser.add_argument(
        "--cs-tail-quantile",
        type=float,
        default=0.016,
        help="Tail quantile for long/short selection in cross-sectional strategy.",
    )
    parser.add_argument(
        "--cs-top-n",
        type=int,
        default=600,
        help="Top liquid symbols eligible per rebalance in cross-sectional strategy.",
    )
    parser.add_argument(
        "--cs-liquidity-lookback",
        type=int,
        default=30,
        help="Rolling minute window for liquidity ranking in cross-sectional strategy.",
    )
    parser.add_argument(
        "--cs-min-universe",
        type=int,
        default=40,
        help="Minimum eligible symbols before cross-sectional strategy trades.",
    )
    parser.add_argument(
        "--cs-base-notional",
        type=float,
        default=500.0,
        help="Base dollar notional per symbol before leverage in cross-sectional strategy.",
    )
    parser.add_argument(
        "--cs-target-annual-vol",
        type=float,
        default=0.30,
        help="Target annual vol for cross-sectional leverage sizing; 0 disables.",
    )
    parser.add_argument(
        "--cs-vol-window",
        type=int,
        default=60,
        help="Rolling window for cross-sectional volatility proxy.",
    )
    parser.add_argument(
        "--cs-max-leverage",
        type=float,
        default=2.0,
        help="Max leverage for cross-sectional strategy.",
    )
    parser.add_argument(
        "--cs-min-annual-vol",
        type=float,
        default=0.01,
        help="Vol floor for cross-sectional leverage calculation.",
    )
    parser.add_argument(
        "--cs-no-flips",
        dest="cs_no_flips",
        action="store_true",
        help="Disable immediate side flips in cross-sectional strategy (default).",
    )
    parser.add_argument(
        "--cs-allow-flips",
        dest="cs_no_flips",
        action="store_false",
        help="Allow immediate side flips in cross-sectional strategy.",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000, help="Initial capital."
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot equity curve at the end."
    )
    parser.add_argument(
        "--verbose-trades",
        dest="verbose_trades",
        action="store_true",
        help="Print each simulated fill line in single-asset mode (default).",
    )
    parser.add_argument(
        "--no-verbose-trades",
        dest="verbose_trades",
        action="store_false",
        help="Disable per-fill trade prints in single-asset mode.",
    )
    parser.add_argument(
        "--fill-seed",
        type=int,
        default=42,
        help="Deterministic random seed for single-asset fill simulation (default: 42).",
    )
    parser.add_argument(
        "--paper-parity",
        dest="paper_parity",
        action="store_true",
        help="Enable paper-parity controls for backtests (default).",
    )
    parser.add_argument(
        "--no-paper-parity",
        dest="paper_parity",
        action="store_false",
        help="Disable paper-parity controls for backtests.",
    )
    parser.add_argument(
        "--asset-flags-csv",
        type=str,
        default="",
        help=(
            "Optional symbol flags CSV with symbol/ticker and tradable/shortable/"
            "easy_to_borrow columns for parity eligibility gating."
        ),
    )
    parser.add_argument(
        "--account-equity",
        type=float,
        default=None,
        help="Account equity for parity buying-power tier logic.",
    )
    parser.add_argument(
        "--buying-power-mode",
        choices=("disabled", "multiplier", "tiered"),
        default="tiered",
        help=(
            "Buying power policy when --paper-parity is enabled: "
            "disabled, multiplier, or tiered."
        ),
    )
    parser.add_argument(
        "--buying-power-multiplier",
        type=float,
        default=None,
        help="Explicit buying-power multiplier used when --buying-power-mode=multiplier.",
    )
    parser.add_argument(
        "--reserve-open-orders",
        dest="reserve_open_orders",
        action="store_true",
        help="Reserve buying power for open orders in parity mode (default).",
    )
    parser.add_argument(
        "--no-reserve-open-orders",
        dest="reserve_open_orders",
        action="store_false",
        help="Disable buying-power reservation for open orders in parity mode.",
    )
    parser.add_argument(
        "--max-notional-per-order",
        type=float,
        default=2_000.0,
        help="Optional per-order notional cap for multi-asset backtests.",
    )
    parser.add_argument(
        "--max-short-notional",
        type=float,
        default=20_000.0,
        help="Optional total short notional cap for multi-asset backtests.",
    )
    parser.set_defaults(
        cs_no_flips=True,
        reserve_open_orders=True,
        paper_parity=True,
        verbose_trades=True,
    )
    return parser.parse_args()


def build_paper_parity_config(args: argparse.Namespace) -> PaperParityConfig:
    if not args.paper_parity:
        return PaperParityConfig.disabled()

    buying_power = BuyingPowerConfig(
        mode=str(args.buying_power_mode),
        explicit_multiplier=args.buying_power_multiplier,
    )
    account_equity = args.account_equity
    if account_equity is None and buying_power.mode == "tiered":
        account_equity = buying_power.tier_day_trader_equity

    return PaperParityConfig(
        enabled=True,
        require_tradable=True,
        require_shortable=True,
        require_easy_to_borrow=False,
        allow_crypto_shorts=False,
        buying_power=buying_power,
        account_equity=account_equity,
        reserve_open_orders=bool(args.reserve_open_orders),
    )


def build_strategy(strategy_cls, args: argparse.Namespace):
    if strategy_cls is MovingAverageStrategy:
        return MovingAverageStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            position_size=args.position_size,
        )
    if strategy_cls is TemplateStrategy:
        return TemplateStrategy(
            lookback=args.momentum_lookback,
            position_size=args.position_size,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
    if strategy_cls is CryptoCompetitionStrategy:
        return CryptoCompetitionStrategy(
            position_size=args.position_size,
        )
    if strategy_cls is CrossSectionalPaperReversalStrategy:
        return CrossSectionalPaperReversalStrategy(
            lookback_minutes=args.cs_lookback,
            hold_minutes=args.cs_hold,
            tail_quantile=args.cs_tail_quantile,
            top_n=args.cs_top_n,
            liquidity_lookback=args.cs_liquidity_lookback,
            min_universe_size=args.cs_min_universe,
            base_notional_per_name=args.cs_base_notional,
            target_annual_vol=args.cs_target_annual_vol,
            volatility_window=args.cs_vol_window,
            max_leverage=args.cs_max_leverage,
            min_annual_vol=args.cs_min_annual_vol,
            allow_flips=not args.cs_no_flips,
        )
    try:
        return strategy_cls()
    except TypeError as exc:
        raise SystemExit(
            f"{strategy_cls.__name__} must support a no-arg constructor or use --strategy template."
        ) from exc


def resolve_single_asset_class(
    args: argparse.Namespace, strategy, csv_path: Path
) -> str:
    requested = str(getattr(args, "asset_class", "auto")).strip().lower()
    if requested in {"stock", "crypto"}:
        return requested

    strategy_name = strategy.__class__.__name__.lower()
    if "crypto" in strategy_name:
        return "crypto"

    stem = csv_path.stem.upper()
    if "-USD" in stem or "_USD" in stem or "BTC" in stem or "ETH" in stem or "SOL" in stem:
        return "crypto"
    return "stock"


def main() -> None:
    args = parse_args()
    strategy_cls = get_strategy_class(args.strategy)

    asset_flags_by_symbol: dict[str, AssetEligibilityFlags] = {}
    if args.asset_flags_csv:
        asset_flags_path = Path(args.asset_flags_csv)
        if not asset_flags_path.exists():
            raise FileNotFoundError(f"Asset flags CSV not found: {asset_flags_path}")
        asset_flags_by_symbol = load_asset_flags_by_symbol(asset_flags_path)
    paper_parity = build_paper_parity_config(args)
    strategy_probe = build_strategy(strategy_cls, args)
    is_cross_sectional = hasattr(strategy_probe, "run_panel")

    if is_cross_sectional and not args.panel_csv:
        raise SystemExit(
            "Cross-sectional strategies require --panel-csv (multi-symbol long-form data)."
        )

    if args.panel_csv:
        panel_path = Path(args.panel_csv)
        if not panel_path.exists():
            raise FileNotFoundError(f"Panel CSV not found: {panel_path}")

        panel_df = pd.read_csv(panel_path)
        sym_col = "symbol"
        user_specified_symbols = bool(
            args.symbols or args.symbols_file or args.symbol_limit > 0
        )
        symbols = []
        if user_specified_symbols:
            symbols = resolve_symbols(
                default_symbol=args.symbol,
                symbols_arg=args.symbols,
                symbols_file_arg=args.symbols_file,
                symbol_limit=args.symbol_limit,
            )

        if symbols:
            panel_df.columns = [str(c) for c in panel_df.columns]
            found_sym_col = None
            for col in panel_df.columns:
                if str(col).strip().lower() in {"symbol", "ticker"}:
                    found_sym_col = col
                    break
            if found_sym_col is None:
                raise ValueError("Panel CSV must contain a symbol column.")
            sym_col = str(found_sym_col)
            panel_df[sym_col] = panel_df[sym_col].astype(str).str.upper()
            panel_df = panel_df[panel_df[sym_col].isin(list(symbols))].copy()
        else:
            panel_df.columns = [str(c) for c in panel_df.columns]
            for col in panel_df.columns:
                if str(col).strip().lower() in {"symbol", "ticker"}:
                    sym_col = str(col)
                    break
        if panel_df.empty:
            raise RuntimeError("Filtered panel data is empty.")

        before_symbols = int(panel_df[sym_col].nunique())
        before_rows = len(panel_df)
        panel_df = _apply_min_symbol_bars_filter(
            panel_df,
            symbol_col=sym_col,
            min_symbol_bars=args.min_symbol_bars,
        )

        if args.auto_universe_top_n > 0 and not panel_df.empty:
            auto_symbols = _build_auto_liquidity_universe(
                panel_df,
                symbol_col=sym_col,
                top_n=args.auto_universe_top_n,
                min_days=args.auto_universe_min_days,
                max_symbols=args.auto_universe_max_symbols,
            )
            if not auto_symbols:
                raise RuntimeError(
                    "Auto liquidity universe filter returned zero symbols. "
                    "Try lowering --auto-universe-top-n or --auto-universe-min-days."
                )
            panel_df[sym_col] = panel_df[sym_col].astype(str).str.upper()
            panel_df = panel_df[panel_df[sym_col].isin(list(auto_symbols))].copy()

        panel_df = pd.DataFrame(panel_df)
        after_symbols = int(panel_df[sym_col].nunique())
        after_rows = len(panel_df)

        if panel_df.empty:
            raise RuntimeError("Panel data is empty after quality/universe filters.")

        if before_symbols != after_symbols or before_rows != after_rows:
            print(
                "Panel filter summary: "
                f"symbols {before_symbols} -> {after_symbols}, "
                f"rows {before_rows} -> {after_rows}"
            )

        def strategy_factory():
            return build_strategy(strategy_cls, args)

        multi_bt = MultiAssetBacktester(
            panel_df=panel_df,
            strategy_factory=strategy_factory,
            initial_capital=args.capital,
            max_notional_per_order=args.max_notional_per_order,
            max_short_notional=args.max_short_notional,
            paper_parity=paper_parity,
            asset_flags_by_symbol=asset_flags_by_symbol,
        )
        interrupted = False
        try:
            equity_df = multi_bt.run()
        except KeyboardInterrupt:
            interrupted = True
            equity_df = multi_bt.equity_frame()
            print("\nBacktest interrupted. Showing partial results...")

        if equity_df.empty:
            raise RuntimeError("No equity points produced in multi-asset backtest.")

        returns = equity_df["equity"].pct_change().fillna(0.0)
        sharpe = 0.0
        if returns.std() > 0:
            sharpe = float(np.sqrt(252 * 6.5 * 60) * returns.mean() / returns.std())
        drawdown = (equity_df["equity"] / equity_df["equity"].cummax() - 1.0).min()

        title = "=== Multi-Asset Backtest Summary ==="
        if interrupted:
            title = "=== Multi-Asset Backtest Summary (Partial) ==="
        print(f"\n{title}")
        symbols_traded = int(pd.Series(panel_df[sym_col]).nunique())
        print(f"Symbols traded: {symbols_traded}")
        print(f"Equity data points: {len(equity_df)}")
        print(f"Trades executed: {len(multi_bt.trades)}")
        print(f"Final portfolio value: {equity_df.iloc[-1]['equity']:.2f}")
        print(f"PnL: {equity_df.iloc[-1]['equity'] - equity_df.iloc[0]['equity']:.2f}")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"Max Drawdown: {float(drawdown):.4f}")
        print(f"Paper parity enabled: {multi_bt.paper_parity.enabled}")
        if multi_bt.paper_parity.enabled:
            bp = multi_bt.paper_parity.buying_power
            print(f"Parity buying-power mode: {bp.mode}")
            print(f"Parity reserve-open-orders: {multi_bt.paper_parity.reserve_open_orders}")
        print(
            "Caps: "
            f"max_notional_per_order={multi_bt.max_notional_per_order}, "
            f"max_short_notional={multi_bt.max_short_notional}"
        )

        if args.plot:
            plot_equity(equity_df)
        return

    csv_path = Path(args.csv) if args.csv else DATA_DIR / "sample_system_test_data.csv"
    if not csv_path.exists():
        if args.csv:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        create_sample_data(csv_path)
        print(f"Sample data generated at {csv_path}.")

    strategy = strategy_probe
    single_asset_class = resolve_single_asset_class(args, strategy, csv_path)
    gateway = MarketDataGateway(csv_path)
    order_book = OrderBook()
    order_manager = OrderManager(
        capital=args.capital, max_long_position=1_000, max_short_position=1_000
    )
    matching_engine = MatchingEngine(seed=args.fill_seed)
    logger = OrderLoggingGateway()

    backtester = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=logger,
        default_position_size=int(max(1, args.position_size)),
        verbose=bool(args.verbose_trades),
        show_progress=True,
        asset_class=single_asset_class,
        paper_parity=paper_parity,
        asset_flags_by_symbol=asset_flags_by_symbol,
    )

    interrupted = False
    try:
        equity_df = backtester.run()
    except KeyboardInterrupt:
        interrupted = True
        points = min(len(backtester.market_history), len(backtester.equity_curve))
        timestamps = [
            pd.Timestamp(backtester.market_history[i]["Datetime"])
            for i in range(points)
        ]
        equity_df = pd.DataFrame(
            {
                "Datetime": timestamps,
                "equity": backtester.equity_curve[:points],
                "cash": backtester.cash_history[:points],
                "position": backtester.position_history[:points],
            }
        )
        print("\nBacktest interrupted. Showing partial results...")

    if equity_df.empty:
        raise RuntimeError("No equity points produced in backtest.")

    analyzer = PerformanceAnalyzer(equity_df["equity"].tolist(), backtester.trades)

    title = "=== Backtest Summary ==="
    if interrupted:
        title = "=== Backtest Summary (Partial) ==="
    print(f"\n{title}")
    print(f"Equity data points: {len(equity_df)}")
    print(f"Trades executed: {sum(1 for t in backtester.trades if t.qty > 0)}")
    print(f"Final portfolio value: {equity_df.iloc[-1]['equity']:.2f}")
    print(f"PnL: {analyzer.pnl():.2f}")
    print(f"Sharpe: {analyzer.sharpe():.2f}")
    print(f"Max Drawdown: {analyzer.max_drawdown():.4f}")
    print(f"Win Rate: {analyzer.win_rate():.2%}")

    if args.plot:
        plot_equity(equity_df)


if __name__ == "__main__":
    main()

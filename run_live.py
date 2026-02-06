"""
Alpaca paper-trading runner.

Requires .env file with:
    ALPACA_API_KEY      (required)
    ALPACA_API_SECRET   (required)

Usage:
    # Single iteration
    python run_live.py --symbol AAPL --strategy ma

    # Continuous live trading
    python run_live.py --symbol AAPL --strategy ma --live

    # Continuous live trading with debug console logs
    python run_live.py --symbol AAPL --strategy ma --live --debug

    # Dry run (no real orders)
    python run_live.py --symbol AAPL --strategy ma --dry-run

    # Crypto trading
    python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live

Logs are saved to: logs/trades.csv, logs/signals.csv, logs/system.log
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

from core.alpaca_trader import AlpacaTrader
from core.logger import get_logger, get_trade_logger, set_console_log_level
from core.multi_asset_trader import MultiAssetAlpacaTrader
from core.symbols import resolve_symbols
from pipeline.alpaca import clean_market_data, save_bars
from strategies import (
    CrossSectionalPaperReversalStrategy,
    CryptoTrendStrategy,
    DemoStrategy,
    MovingAverageStrategy,
    TemplateStrategy,
    get_strategy_class,
    list_strategies,
)

logger = get_logger("run_live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-trading loop with Alpaca.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies: {", ".join(list_strategies())}

Examples:
  python run_live.py --symbol AAPL --strategy ma --live
  python run_live.py --symbol AAPL --strategy ma --live --debug
  python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live
  python run_live.py --symbol AAPL --strategy ma --dry-run --iterations 5
        """,
    )
    parser.add_argument(
        "--symbol", default="AAPL", help="Ticker or crypto symbol (default: AAPL)"
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols for multi-asset mode (stocks only).",
    )
    parser.add_argument(
        "--symbols-file",
        default="",
        help="Path to .txt/.csv symbols file for multi-asset mode (cross-sectional defaults to data/universe_long_short.csv if omitted).",
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=0,
        help="Optional cap on multi-asset symbol count (0 means all).",
    )
    parser.add_argument(
        "--asset-class",
        choices=["stock", "crypto"],
        default="stock",
        help="Asset class (default: stock)",
    )
    parser.add_argument(
        "--timeframe",
        default="1Min",
        help="Alpaca timeframe: 1Min, 5Min, 15Min, 1H, 1D (default: 1Min)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=200,
        help="Bars to fetch each iteration (default: 200)",
    )
    parser.add_argument("--strategy", default="ma", help="Strategy name (default: ma)")
    parser.add_argument(
        "--short-window", type=int, default=20, help="Short MA window (default: 20)"
    )
    parser.add_argument(
        "--long-window", type=int, default=60, help="Long MA window (default: 60)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=10.0,
        help="Per-trade position size (default: 10.0)",
    )
    parser.add_argument(
        "--max-order-notional",
        type=float,
        default=None,
        help="Max notional per order (crypto only)",
    )
    parser.add_argument(
        "--momentum-lookback",
        type=int,
        default=14,
        help="Momentum lookback for template strategy (default: 14)",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.01,
        help="Buy threshold for template strategy (default: 0.01)",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.01,
        help="Sell threshold for template strategy (default: -0.01)",
    )
    parser.add_argument(
        "--cs-lookback",
        type=int,
        default=15,
        help="Lookback minutes for cross-sectional reversal strategy (default: 15).",
    )
    parser.add_argument(
        "--cs-hold",
        type=int,
        default=30,
        help="Holding window in minutes for cross-sectional reversal strategy (default: 30).",
    )
    parser.add_argument(
        "--cs-tail-quantile",
        type=float,
        default=0.016,
        help="Tail quantile for cross-sectional long/short selection (default: 0.016).",
    )
    parser.add_argument(
        "--cs-top-n",
        type=int,
        default=600,
        help="Max liquid symbols eligible each rebalance in cross-sectional mode (default: 600).",
    )
    parser.add_argument(
        "--cs-liquidity-lookback",
        type=int,
        default=30,
        help="Rolling minutes for ADV/liquidity ranking in cross-sectional mode (default: 30).",
    )
    parser.add_argument(
        "--cs-min-universe",
        type=int,
        default=40,
        help="Minimum eligible symbols before cross-sectional orders are emitted (default: 40).",
    )
    parser.add_argument(
        "--cs-base-notional",
        type=float,
        default=1000.0,
        help="Base dollar notional per name before leverage in cross-sectional mode (default: 1000).",
    )
    parser.add_argument(
        "--cs-target-annual-vol",
        type=float,
        default=0.60,
        help="Vol-target for cross-sectional leverage sizing; 0 disables (default: 0.60).",
    )
    parser.add_argument(
        "--cs-vol-window",
        type=int,
        default=60,
        help="Rolling window for cross-sectional volatility proxy (default: 60).",
    )
    parser.add_argument(
        "--cs-max-leverage",
        type=float,
        default=10.0,
        help="Max leverage for cross-sectional sizing (default: 10.0).",
    )
    parser.add_argument(
        "--cs-min-annual-vol",
        type=float,
        default=0.01,
        help="Vol floor for cross-sectional leverage calculation (default: 0.01).",
    )
    parser.add_argument(
        "--cs-no-flips",
        action="store_true",
        help="Disable immediate side flips in cross-sectional mode.",
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Number of loops to run (default: 1)"
    )
    parser.add_argument(
        "--sleep", type=int, default=60, help="Seconds between loops (default: 60)"
    )
    parser.add_argument(
        "--live", action="store_true", help="Run continuously until Ctrl+C"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show DEBUG logs in console (system.log always includes DEBUG).",
    )
    parser.add_argument(
        "--save-data", action="store_true", help="Save raw+clean CSVs to data/"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print decisions without placing orders"
    )
    parser.add_argument(
        "--feed",
        default=None,
        help="Data feed (iex or sip for stocks; defaults to iex for stocks).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Symbols per market-data request in multi-asset mode (default: 100).",
    )
    parser.add_argument(
        "--max-api-requests-per-minute",
        type=int,
        default=190,
        help="Global API request cap for multi-asset mode (default: 190).",
    )
    parser.add_argument(
        "--max-orders-per-cycle",
        type=int,
        default=25,
        help="Maximum orders to submit each loop in multi-asset mode (default: 25).",
    )
    parser.add_argument(
        "--data-fetch-retries",
        type=int,
        default=3,
        help="Retries per failed market-data request in multi-asset mode (default: 3).",
    )
    parser.add_argument(
        "--data-fetch-backoff",
        type=float,
        default=0.75,
        help="Base backoff seconds between market-data retries in multi-asset mode (default: 0.75).",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies and exit",
    )
    return parser.parse_args()


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
    if strategy_cls is CryptoTrendStrategy:
        return CryptoTrendStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            position_size=args.position_size,
        )
    if strategy_cls is DemoStrategy:
        return DemoStrategy(
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


def main() -> None:
    args = parse_args()

    if args.debug:
        set_console_log_level("DEBUG")
        logger.info("Console logging set to DEBUG.")

    # Handle --list-strategies
    if args.list_strategies:
        print("Available strategies:")
        for name in list_strategies():
            print(f"  - {name}")
        sys.exit(0)

    strategy_cls = get_strategy_class(args.strategy)

    if args.asset_class == "stock" and not args.feed:
        args.feed = "iex"
        logger.info("No stock feed specified; defaulting to iex.")

    if (
        strategy_cls is CrossSectionalPaperReversalStrategy
        and not args.symbols
        and not args.symbols_file
    ):
        default_symbols_file = Path("data/universe_long_short.csv")
        if default_symbols_file.exists():
            args.symbols_file = str(default_symbols_file)
            logger.info(
                "No symbols provided for cross-sectional strategy; using %s.",
                args.symbols_file,
            )

    resolved_symbols = resolve_symbols(
        default_symbol=args.symbol,
        symbols_arg=args.symbols,
        symbols_file_arg=args.symbols_file,
        symbol_limit=args.symbol_limit,
    )
    multi_asset_mode = len(resolved_symbols) > 1

    strategy_probe = build_strategy(strategy_cls, args)
    if hasattr(strategy_probe, "run_panel") and not multi_asset_mode:
        raise SystemExit(
            "Cross-sectional strategies require multiple symbols. Use --symbols or --symbols-file."
        )
    data_lookback = max(
        args.lookback, int(getattr(strategy_probe, "required_lookback", args.lookback))
    )
    if data_lookback != args.lookback:
        logger.info(
            "Adjusted lookback from %s to %s for %s history requirements.",
            args.lookback,
            data_lookback,
            strategy_probe.__class__.__name__,
        )

    mode = "DRY RUN" if args.dry_run else "LIVE"
    if multi_asset_mode:
        if args.asset_class != "stock":
            raise SystemExit("Multi-asset mode currently supports stock symbols only.")
        logger.info(
            "Starting %s multi-asset trading: symbols=%s | strategy=%s | timeframe=%s",
            mode,
            len(resolved_symbols),
            args.strategy,
            args.timeframe,
        )

        def strategy_factory():
            return build_strategy(strategy_cls, args)

        trader = MultiAssetAlpacaTrader(
            symbols=resolved_symbols,
            timeframe=args.timeframe,
            lookback=data_lookback,
            strategy_factory=strategy_factory,
            feed=args.feed,
            dry_run=args.dry_run,
            max_order_notional=args.max_order_notional,
            max_api_requests_per_minute=args.max_api_requests_per_minute,
            batch_size=args.batch_size,
            max_orders_per_cycle=args.max_orders_per_cycle,
            data_fetch_retries=args.data_fetch_retries,
            data_fetch_backoff_seconds=args.data_fetch_backoff,
        )
    else:
        logger.info(
            "Starting %s trading: %s | strategy=%s | timeframe=%s",
            mode,
            resolved_symbols[0],
            args.strategy,
            args.timeframe,
        )
        strategy = strategy_probe
        trader = AlpacaTrader(
            symbol=resolved_symbols[0],
            asset_class=args.asset_class,
            timeframe=args.timeframe,
            lookback=data_lookback,
            strategy=strategy,
            feed=args.feed,
            dry_run=args.dry_run,
            max_order_notional=args.max_order_notional,
        )

    trade_logger = get_trade_logger()
    start_equity = trader.starting_equity
    iteration_count = 0

    def handle_iteration() -> None:
        nonlocal iteration_count
        iteration_count += 1
        if multi_asset_mode:
            logger.debug(
                "Iteration %s: fetching data for %s symbols",
                iteration_count,
                len(resolved_symbols),
            )
        else:
            logger.debug(
                "Iteration %s: fetching data for %s",
                iteration_count,
                resolved_symbols[0],
            )
        df = trader.run_once()
        if args.save_data and df is not None:
            if multi_asset_mode:
                out_dir = Path("data")
                out_dir.mkdir(parents=True, exist_ok=True)
                raw_path = out_dir / f"MULTI_{args.timeframe}_stock_live_raw.csv"
                frame = df.copy()
                if "Datetime" in frame.columns:
                    frame["Datetime"] = pd.to_datetime(
                        frame["Datetime"], utc=True, errors="coerce"
                    )
                write_header = not raw_path.exists()
                frame.to_csv(raw_path, mode="a", index=False, header=write_header)
            else:
                raw_path = save_bars(
                    df, resolved_symbols[0], args.timeframe, args.asset_class
                )
                clean_market_data(raw_path)

    def print_summary() -> None:
        summary = trade_logger.get_session_summary(start_equity)
        logger.info("")
        logger.info("=" * 60)
        logger.info("                    SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Iterations:      {iteration_count}")
        logger.info(f"  Total Trades:    {summary['total_trades']}")
        logger.info(f"  Buys / Sells:    {summary['buys']} / {summary['sells']}")
        logger.info("-" * 60)
        logger.info(f"  Wins / Losses:   {summary['wins']} / {summary['losses']}")
        logger.info(f"  Win Rate:        {summary['win_rate']:.1f}%")
        logger.info(f"  Avg Trade P&L:   ${summary['avg_trade_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Start Equity:    ${summary['start_equity']:,.2f}")
        logger.info(f"  End Equity:      ${summary['end_equity']:,.2f}")
        logger.info(f"  Net P&L:         ${summary['net_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Sharpe Ratio:    {summary['sharpe_ratio']:.2f}")
        logger.info(f"  Volatility:      {summary['volatility']:.2f}%")
        logger.info(f"  Max Drawdown:    {summary['max_drawdown']:.2f}%")
        logger.info("=" * 60)
        logger.info("Logs: logs/trades.csv, logs/system.log")

    if args.live:
        logger.info(
            f"Running continuously (Ctrl+C to stop). Sleep: {args.sleep}s between iterations."
        )
        try:
            while True:
                loop_started_at = time.monotonic()
                handle_iteration()
                elapsed = time.monotonic() - loop_started_at
                sleep_for = max(0.0, args.sleep - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    logger.warning(
                        "Iteration took %.1fs which exceeded sleep interval (%ss).",
                        elapsed,
                        args.sleep,
                    )
        except KeyboardInterrupt:
            logger.info("Received stop signal.")
            print_summary()
    else:
        logger.info(f"Running {args.iterations} iteration(s)...")
        for i in range(args.iterations):
            handle_iteration()
            if i < args.iterations - 1:
                time.sleep(args.sleep)
        print_summary()


if __name__ == "__main__":
    main()

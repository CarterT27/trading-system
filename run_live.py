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
    CryptoCompetitionStrategy,
    CryptoCompetitionPortfolioStrategy,
    CryptoRegimeTrendStrategy,
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
        help="Comma-separated symbols for multi-asset mode.",
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
        "--meta-no-trade-band",
        type=float,
        default=0.03,
        help="Hysteresis half-band around meta threshold for crypto competition strategy.",
    )
    parser.add_argument(
        "--meta-use-xgboost",
        action="store_true",
        help="Include optional XGBoost leg in crypto competition strategy meta ensemble.",
    )
    parser.add_argument(
        "--meta-prob-threshold",
        type=float,
        default=0.50,
        help="Meta probability threshold for crypto competition strategies.",
    )
    parser.add_argument(
        "--meta-refit-interval",
        type=int,
        default=120,
        help="Bars between meta-model refits for crypto competition strategies.",
    )
    parser.add_argument(
        "--meta-train-window",
        type=int,
        default=3000,
        help="Rolling training window for meta model (<=0 disables full-history mode).",
    )
    parser.add_argument(
        "--sizing-scale",
        type=float,
        default=1.5,
        help="Dynamic sizing scale for crypto competition strategies.",
    )
    parser.add_argument(
        "--sizing-offset",
        type=float,
        default=0.3,
        help="Dynamic sizing offset for crypto competition strategies.",
    )
    parser.add_argument(
        "--sizing-min-size",
        type=float,
        default=0.05,
        help="Minimum dynamic position fraction for crypto competition strategies.",
    )
    parser.add_argument(
        "--exit-time-bars",
        type=int,
        default=180,
        help="Time-stop bars for crypto competition strategies (<=0 disables).",
    )
    parser.add_argument(
        "--exit-trail-stop",
        type=float,
        default=0.008,
        help="Trailing-stop drawdown for crypto competition strategies (<=0 disables).",
    )
    parser.add_argument(
        "--exit-min-hold-bars",
        type=int,
        default=0,
        help="Minimum hold bars before exit rules can trigger in crypto competition strategies.",
    )
    parser.add_argument(
        "--portfolio-notional",
        type=float,
        default=100_000.0,
        help="Portfolio gross notional budget for crypto competition portfolio strategy.",
    )
    parser.add_argument(
        "--portfolio-min-order-notional",
        type=float,
        default=5.0,
        help="Minimum notional delta required for portfolio rebalance orders.",
    )
    parser.add_argument(
        "--portfolio-fractional-qty",
        dest="portfolio_fractional_qty",
        action="store_true",
        help="Allow fractional target_qty in portfolio strategy output (default).",
    )
    parser.add_argument(
        "--portfolio-integer-qty",
        dest="portfolio_fractional_qty",
        action="store_false",
        help="Force integer target_qty in portfolio strategy output.",
    )
    parser.add_argument(
        "--max-order-notional",
        type=float,
        default=2_000.0,
        help="Max notional per order",
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
        default=45,
        help="Holding window in minutes for cross-sectional reversal strategy (default: 45).",
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
        default=500.0,
        help="Base dollar notional per name before leverage in cross-sectional mode (default: 500).",
    )
    parser.add_argument(
        "--cs-target-annual-vol",
        type=float,
        default=0.30,
        help="Vol-target for cross-sectional leverage sizing; 0 disables (default: 0.30).",
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
        default=2.0,
        help="Max leverage for cross-sectional sizing (default: 2.0).",
    )
    parser.add_argument(
        "--cs-min-annual-vol",
        type=float,
        default=0.01,
        help="Vol floor for cross-sectional leverage calculation (default: 0.01).",
    )
    parser.add_argument(
        "--cs-no-flips",
        dest="cs_no_flips",
        action="store_true",
        help="Disable immediate side flips in cross-sectional mode (default).",
    )
    parser.add_argument(
        "--cs-allow-flips",
        dest="cs_no_flips",
        action="store_false",
        help="Allow immediate side flips in cross-sectional mode.",
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
        default=10,
        help="Maximum orders to submit each loop in multi-asset mode (default: 10).",
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
        "--buying-power-buffer",
        type=float,
        default=0.20,
        help="Fraction of buying power to reserve in multi-asset mode (default: 0.20).",
    )
    parser.add_argument(
        "--buying-power-cooldown-cycles",
        type=int,
        default=5,
        help="Cooldown cycles after insufficient buying-power rejects in multi-asset mode (default: 5).",
    )
    parser.add_argument(
        "--skip-preload",
        action="store_true",
        help="Skip startup history preload in multi-asset mode.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies and exit",
    )
    parser.set_defaults(cs_no_flips=True, portfolio_fractional_qty=True)
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
    if strategy_cls is CryptoCompetitionStrategy:
        meta_train_window = (
            None if int(args.meta_train_window) <= 0 else int(args.meta_train_window)
        )
        exit_time_bars = (
            None if int(args.exit_time_bars) <= 0 else int(args.exit_time_bars)
        )
        exit_trail_stop = (
            None if float(args.exit_trail_stop) <= 0 else float(args.exit_trail_stop)
        )
        return CryptoCompetitionStrategy(
            position_size=args.position_size,
            meta_no_trade_band=args.meta_no_trade_band,
            meta_use_xgboost=bool(args.meta_use_xgboost),
            meta_prob_threshold=args.meta_prob_threshold,
            meta_refit_interval=args.meta_refit_interval,
            meta_train_window=meta_train_window,
            sizing_scale=args.sizing_scale,
            sizing_offset=args.sizing_offset,
            sizing_min_size=args.sizing_min_size,
            exit_time_bars=exit_time_bars,
            exit_trail_stop=exit_trail_stop,
            exit_min_hold_bars=args.exit_min_hold_bars,
        )
    if strategy_cls is CryptoCompetitionPortfolioStrategy:
        meta_train_window = (
            None if int(args.meta_train_window) <= 0 else int(args.meta_train_window)
        )
        exit_time_bars = (
            None if int(args.exit_time_bars) <= 0 else int(args.exit_time_bars)
        )
        exit_trail_stop = (
            None if float(args.exit_trail_stop) <= 0 else float(args.exit_trail_stop)
        )
        return CryptoCompetitionPortfolioStrategy(
            portfolio_notional=args.portfolio_notional,
            allow_fractional_qty=bool(args.portfolio_fractional_qty),
            min_order_notional=args.portfolio_min_order_notional,
            meta_no_trade_band=args.meta_no_trade_band,
            meta_use_xgboost=bool(args.meta_use_xgboost),
            meta_prob_threshold=args.meta_prob_threshold,
            meta_refit_interval=args.meta_refit_interval,
            meta_train_window=meta_train_window,
            sizing_scale=args.sizing_scale,
            sizing_offset=args.sizing_offset,
            sizing_min_size=args.sizing_min_size,
            exit_time_bars=exit_time_bars,
            exit_trail_stop=exit_trail_stop,
            exit_min_hold_bars=args.exit_min_hold_bars,
        )
    if strategy_cls is CryptoRegimeTrendStrategy:
        return CryptoRegimeTrendStrategy(
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
            asset_class=args.asset_class,
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
            buying_power_buffer=args.buying_power_buffer,
            buying_power_cooldown_cycles=args.buying_power_cooldown_cycles,
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

    if multi_asset_mode and not args.skip_preload:
        try:
            trader.preload_history()
        except Exception as exc:
            logger.warning("Startup preload failed: %s", exc)

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
                raw_path = out_dir / f"MULTI_{args.timeframe}_{args.asset_class}_live_raw.csv"
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

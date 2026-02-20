from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import random
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List


SUMMARY_KEYS = (
    "Equity data points",
    "Trades executed",
    "Final portfolio value",
    "PnL",
    "Sharpe",
    "Max Drawdown",
    "Win Rate",
)


@dataclass
class RunResult:
    run_id: int
    command: List[str]
    params: Dict[str, object]
    return_code: int
    duration_sec: float
    summary: Dict[str, object]
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic grid runner for run_backtest.py. "
            "Supports arbitrary strategy args and writes ranked CSV outputs."
        )
    )
    parser.add_argument("--strategy", required=True, help="Strategy name for run_backtest.py")
    parser.add_argument("--csv", required=True, help="CSV path for run_backtest.py")
    parser.add_argument(
        "--runner",
        default="run_backtest.py",
        help="Path to backtest runner script (default: run_backtest.py)",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable token used under uv run (default: python)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Fixed argument across all runs. "
            "Example: --set position-size=100000 --set tail-bars=10000"
        ),
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=v1,v2,...",
        help=(
            "Grid-search argument. "
            "Example: --grid meta-prob-threshold=0.48,0.5,0.52"
        ),
    )
    parser.add_argument(
        "--flag",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "Constant boolean flag included in all runs. "
            "Example: --flag no-verbose-trades"
        ),
    )
    parser.add_argument(
        "--sort-by",
        default="Sharpe",
        help="Metric used for leaderboard sorting (default: Sharpe)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort leaderboard ascending (default descending)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on number of runs after grid expansion (0 means all).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for independent runs (default: 1).",
    )
    parser.add_argument(
        "--search-mode",
        choices=("full", "one-at-a-time"),
        default="full",
        help=(
            "Grid construction mode. "
            "'full' = cartesian product, "
            "'one-at-a-time' = baseline + vary one key at a time."
        ),
    )
    parser.add_argument(
        "--max-values-per-grid",
        type=int,
        default=0,
        help=(
            "Downsample each --grid dimension to at most this many values "
            "before expansion (0 keeps all)."
        ),
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=0,
        help=(
            "Sample this many combos uniformly after expansion (0 keeps all). "
            "Useful to cap large grids."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for --random-sample (default: 42).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Per-run timeout in seconds (0 means no timeout).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the grid immediately on first non-zero return code.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands only; do not execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=".context/grid_runs",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag for output folder name.",
    )
    return parser.parse_args()


def _parse_scalar(text: str) -> object:
    token = text.strip()
    low = token.lower()
    if low in {"none", "null"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in token or "e" in low:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _parse_key_value(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise ValueError(f"Expected KEY=VALUE format, got: {item}")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Missing key in: {item}")
    return key, _parse_scalar(value)


def _parse_grid_item(item: str) -> tuple[str, List[object]]:
    if "=" not in item:
        raise ValueError(f"Expected KEY=v1,v2,... format, got: {item}")
    key, values = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Missing key in: {item}")
    raw_vals = [x.strip() for x in values.split(",") if x.strip()]
    if not raw_vals:
        raise ValueError(f"No values provided for key {key}")
    return key, [_parse_scalar(v) for v in raw_vals]


def _flag_name(key: str) -> str:
    k = key.strip()
    if not k:
        raise ValueError("Empty argument key")
    if k.startswith("--"):
        return k
    return "--" + k.replace("_", "-")


def _kv_to_args(key: str, value: object) -> List[str]:
    flag = _flag_name(key)
    if value is None:
        return []
    if isinstance(value, bool):
        return [flag] if value else []
    return [flag, str(value)]


def _parse_summary(stdout: str) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in SUMMARY_KEYS:
            continue

        if key == "Win Rate":
            value = value.replace("%", "").strip()
            try:
                summary[key] = float(value)
            except ValueError:
                summary[key] = math.nan
            continue

        try:
            summary[key] = float(value)
        except ValueError:
            summary[key] = value
    return summary


def _safe_sort_value(v: object) -> float:
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isnan(x):
            return -math.inf
        return x
    return -math.inf


def _build_grid_rows(
    fixed_params: Dict[str, object],
    grid_params: Dict[str, List[object]],
    search_mode: str,
) -> List[Dict[str, object]]:
    if not grid_params:
        return [dict(fixed_params)]

    if search_mode == "one-at-a-time":
        baseline: Dict[str, object] = dict(fixed_params)
        for key, values in grid_params.items():
            if key not in baseline and values:
                baseline[key] = values[0]

        rows: List[Dict[str, object]] = [dict(baseline)]
        for key, values in grid_params.items():
            base_val = baseline.get(key)
            for value in values:
                if value == base_val:
                    continue
                row = dict(baseline)
                row[key] = value
                rows.append(row)
        return rows

    if search_mode != "full":
        raise ValueError(f"Unsupported search mode: {search_mode}")

    keys = list(grid_params.keys())
    product_values = itertools.product(*(grid_params[k] for k in keys))
    rows: List[Dict[str, object]] = []
    for combo in product_values:
        row = dict(fixed_params)
        for k, v in zip(keys, combo):
            row[k] = v
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _downsample_evenly(values: List[object], max_count: int) -> List[object]:
    if max_count <= 0 or len(values) <= max_count:
        return values
    if max_count == 1:
        return [values[0]]

    n = len(values)
    step = (n - 1) / (max_count - 1)
    out: List[object] = []
    last_idx = -1
    for i in range(max_count):
        idx = int(round(i * step))
        idx = max(0, min(n - 1, idx))
        if idx <= last_idx:
            idx = min(n - 1, last_idx + 1)
        out.append(values[idx])
        last_idx = idx
    return out


def _execute_single_run(
    run_id: int,
    cmd: List[str],
    combo: Dict[str, object],
    logs_dir: Path,
    timeout_sec: int,
) -> RunResult:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None if timeout_sec <= 0 else int(timeout_sec),
            check=False,
        )
        rc = int(proc.returncode)
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.stdout or ""
        err = (exc.stderr or "") + "\nTIMEOUT"

    dt = time.perf_counter() - t0
    summary = _parse_summary(out)
    log_path = logs_dir / f"run_{run_id:04d}.log"
    log_path.write_text(
        "\n".join(
            [
                f"COMMAND: {shlex.join(cmd)}",
                f"RETURN_CODE: {rc}",
                f"DURATION_SEC: {dt:.4f}",
                "",
                "--- STDOUT ---",
                out,
                "",
                "--- STDERR ---",
                err,
            ]
        ),
        encoding="utf-8",
    )

    return RunResult(
        run_id=run_id,
        command=cmd,
        params=combo,
        return_code=rc,
        duration_sec=dt,
        summary=summary,
        stdout=out,
        stderr=err,
    )


def _render_report(
    args: argparse.Namespace,
    run_dir: Path,
    leaderboard_rows: List[Dict[str, object]],
    total_runs: int,
    failures: int,
) -> str:
    lines: List[str] = []
    lines.append("# Backtest Grid Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Strategy: `{args.strategy}`")
    lines.append(f"- CSV: `{args.csv}`")
    lines.append(f"- Runner: `{args.runner}`")
    lines.append(f"- Total runs: {total_runs}")
    lines.append(f"- Failed runs: {failures}")
    lines.append(f"- Parallel jobs: {args.jobs}")
    lines.append(f"- Search mode: `{args.search_mode}`")
    if args.max_values_per_grid > 0:
        lines.append(f"- Max values per grid key: {args.max_values_per_grid}")
    if args.random_sample > 0:
        lines.append(f"- Random sample size: {args.random_sample} (seed={args.seed})")
    lines.append(f"- Sort metric: `{args.sort_by}` ({'asc' if args.ascending else 'desc'})")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- All runs: `{run_dir / 'all_runs.csv'}`")
    lines.append(f"- Leaderboard: `{run_dir / 'leaderboard.csv'}`")
    lines.append(f"- Logs dir: `{run_dir / 'logs'}`")
    lines.append("")
    lines.append("## Top Results")
    lines.append("")
    lines.append(
        "| Rank | Run | Return Code | PnL | Sharpe | Max Drawdown | Win Rate % | Trades | Duration (s) | Params |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for i, row in enumerate(leaderboard_rows[:20], start=1):
        lines.append(
            f"| {i} | {row.get('run_id')} | {row.get('return_code')} | "
            f"{row.get('PnL')} | {row.get('Sharpe')} | {row.get('Max Drawdown')} | "
            f"{row.get('Win Rate')} | {row.get('Trades executed')} | {row.get('duration_sec')} | "
            f"{row.get('params_json')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be >= 1")
    if args.stop_on_error and args.jobs > 1:
        raise ValueError("--stop-on-error requires --jobs 1")

    fixed_params: Dict[str, object] = {}
    for item in args.set:
        k, v = _parse_key_value(item)
        fixed_params[k] = v

    grid_params: Dict[str, List[object]] = {}
    for item in args.grid:
        k, vals = _parse_grid_item(item)
        if args.max_values_per_grid > 0:
            vals = _downsample_evenly(vals, args.max_values_per_grid)
        grid_params[k] = vals

    combos = _build_grid_rows(
        fixed_params=fixed_params,
        grid_params=grid_params,
        search_mode=args.search_mode,
    )
    if args.random_sample > 0 and args.random_sample < len(combos):
        rng = random.Random(args.seed)
        combos = rng.sample(combos, k=int(args.random_sample))
    if args.max_runs > 0:
        combos = combos[: int(args.max_runs)]

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    label = args.tag.strip() if args.tag.strip() else f"{args.strategy}_{ts}"
    run_dir = Path(args.output_dir) / label
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "uv",
        "run",
        args.python,
        args.runner,
        "--strategy",
        args.strategy,
        "--csv",
        args.csv,
    ]
    for flag in args.flag:
        base_cmd.append(_flag_name(flag))

    results: List[RunResult] = []

    run_specs: List[tuple[int, Dict[str, object], List[str]]] = []
    for i, combo in enumerate(combos, start=1):
        cmd = list(base_cmd)
        for k, v in combo.items():
            cmd.extend(_kv_to_args(k, v))
        run_specs.append((i, combo, cmd))

    if args.dry_run:
        for i, _, cmd in run_specs:
            print(f"[{i}/{len(run_specs)}] {shlex.join(cmd)}")
        return

    if args.jobs == 1:
        for i, combo, cmd in run_specs:
            print(f"[{i}/{len(run_specs)}] running: {shlex.join(cmd)}")
            run_result = _execute_single_run(
                run_id=i,
                cmd=cmd,
                combo=combo,
                logs_dir=logs_dir,
                timeout_sec=args.timeout_sec,
            )
            results.append(run_result)
            if run_result.return_code != 0 and args.stop_on_error:
                print("Stopping on first error as requested.")
                break
    else:
        total = len(run_specs)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_run = {
                executor.submit(
                    _execute_single_run,
                    run_id=i,
                    cmd=cmd,
                    combo=combo,
                    logs_dir=logs_dir,
                    timeout_sec=args.timeout_sec,
                ): i
                for i, combo, cmd in run_specs
            }
            for future in concurrent.futures.as_completed(future_to_run):
                run_result = future.result()
                results.append(run_result)
                done += 1
                print(
                    f"[{done}/{total}] finished run {run_result.run_id} "
                    f"(rc={run_result.return_code}, {run_result.duration_sec:.1f}s)"
                )

    results.sort(key=lambda r: r.run_id)

    row_dicts: List[Dict[str, object]] = []
    for r in results:
        row: Dict[str, object] = {
            "run_id": r.run_id,
            "return_code": r.return_code,
            "duration_sec": round(r.duration_sec, 4),
            "command": shlex.join(r.command),
            "params_json": json.dumps(r.params, sort_keys=True),
            "log_path": str((logs_dir / f"run_{r.run_id:04d}.log")),
        }
        for k in SUMMARY_KEYS:
            row[k] = r.summary.get(k, math.nan)
        row_dicts.append(row)

    all_fields = [
        "run_id",
        "return_code",
        "duration_sec",
        "Equity data points",
        "Trades executed",
        "Final portfolio value",
        "PnL",
        "Sharpe",
        "Max Drawdown",
        "Win Rate",
        "params_json",
        "command",
        "log_path",
    ]
    _write_csv(run_dir / "all_runs.csv", row_dicts, all_fields)

    sort_key = args.sort_by
    ranked = sorted(
        row_dicts,
        key=lambda x: _safe_sort_value(x.get(sort_key, math.nan)),
        reverse=not args.ascending,
    )
    _write_csv(run_dir / "leaderboard.csv", ranked, all_fields)

    failed = sum(1 for r in row_dicts if int(r.get("return_code", 1)) != 0)
    report = _render_report(
        args=args,
        run_dir=run_dir,
        leaderboard_rows=ranked,
        total_runs=len(row_dicts),
        failures=failed,
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print("Grid run complete")
    print(f"Output dir: {run_dir}")
    print(f"All runs: {run_dir / 'all_runs.csv'}")
    print(f"Leaderboard: {run_dir / 'leaderboard.csv'}")
    print(f"Report: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()

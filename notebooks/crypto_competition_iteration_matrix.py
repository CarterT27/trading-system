from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency guard
    ExtraTreesClassifier = None  # type: ignore[assignment]
    HistGradientBoostingClassifier = None  # type: ignore[assignment]
    RandomForestClassifier = None  # type: ignore[assignment]
    SGDClassifier = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - optional dependency guard
    LogisticRegression = None  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None  # type: ignore[assignment]

from crypto_experiment_suite import (
    BARS_PER_YEAR_2M,
    RegimeParams,
    build_feature_universe,
    load_universe_from_glob,
    score_key,
    simulate_equal_weight_positions,
    slice_universe,
    split_inner_train_validation,
    summarize_returns,
)

BASE_FEATURE_COLS = [
    "spread_z",
    "slow_slope",
    "vol_ratio",
    "ret1",
    "ret5",
    "ret30",
    "trend_tstat",
]

CONTEXT_FEATURE_COLS = [
    "mkt_ret1_mean",
    "mkt_ret1_std",
    "mkt_vol_mean",
    "mkt_ret1_mean_roll",
    "mkt_vol_mean_roll",
    "ret1_vs_mkt",
]

AUX_HORIZONS = (1, 4, 20)
AUX_WEIGHTS = (0.60, 0.25, 0.15)


@dataclass
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class ExperimentSpec:
    step: str
    name: str
    description: str
    mode: str
    use_context_features: bool
    use_aux_targets: bool
    use_ensemble: bool
    tune_train_window: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ordered crypto strategy iteration matrix: gap-aware CV, online updates, "
            "auxiliary horizons, context features, ensembling, and train-window selection."
        )
    )
    parser.add_argument(
        "--crypto-glob",
        default="data/legacy/raw_data_crypto/*_2m_data.csv",
        help="Glob pattern for crypto CSV files.",
    )
    parser.add_argument("--train-days", type=int, default=10, help="Train window in days.")
    parser.add_argument("--gap-days", type=int, default=2, help="Gap between train and test in days.")
    parser.add_argument("--test-days", type=int, default=5, help="Test window in days.")
    parser.add_argument("--step-days", type=int, default=5, help="Fold step in days.")
    parser.add_argument(
        "--min-train-bars",
        type=int,
        default=5000,
        help="Minimum bars required per symbol in train.",
    )
    parser.add_argument(
        "--min-test-bars",
        type=int,
        default=2000,
        help="Minimum bars required per symbol in test.",
    )
    parser.add_argument(
        "--min-symbols",
        type=int,
        default=2,
        help="Minimum overlapping symbols required per fold.",
    )
    parser.add_argument("--short-window", type=int, default=20)
    parser.add_argument("--long-window", type=int, default=120)
    parser.add_argument("--slope-lookback", type=int, default=120)
    parser.add_argument("--volatility-window", type=int, default=1440)
    parser.add_argument("--volatility-quantile", type=float, default=0.70)
    parser.add_argument(
        "--meta-threshold",
        type=float,
        default=0.50,
        help="Probability threshold for turning meta probabilities into long entries.",
    )
    parser.add_argument(
        "--meta-band",
        type=float,
        default=0.03,
        help="No-trade hysteresis half-band around threshold.",
    )
    parser.add_argument(
        "--use-xgboost",
        action="store_true",
        help="Include XGBoost in diverse ensemble when installed.",
    )
    parser.add_argument(
        "--train-window-grid",
        default="1000,2000,4000,none",
        help="Candidate train-window lengths for experiment 06 (use 'none' for full history).",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/results/crypto_iteration_matrix",
        help="Output directory.",
    )
    parser.add_argument(
        "--position-mode",
        choices=("binary", "sized"),
        default="binary",
        help="Binary gate position or dynamic sized position from spread/vol features.",
    )
    parser.add_argument(
        "--sizing-scale",
        type=float,
        default=1.5,
        help="Sizing scale when --position-mode sized.",
    )
    parser.add_argument(
        "--sizing-offset",
        type=float,
        default=0.3,
        help="Spread-z offset for sized mode.",
    )
    parser.add_argument(
        "--sizing-vol-exponent",
        type=float,
        default=0.75,
        help="Vol-ratio exponent for sized mode.",
    )
    parser.add_argument(
        "--sizing-min-size",
        type=float,
        default=0.05,
        help="Minimum size threshold for sized mode.",
    )
    parser.add_argument(
        "--exec-model",
        choices=("ideal", "full_fill", "stochastic"),
        default="ideal",
        help="Vectorized execution model used in simulation.",
    )
    parser.add_argument(
        "--exec-seed",
        type=int,
        default=42,
        help="Random seed for stochastic execution model.",
    )
    parser.add_argument(
        "--exec-full-prob",
        type=float,
        default=0.70,
        help="Probability of full fill in stochastic execution model.",
    )
    parser.add_argument(
        "--exec-partial-prob",
        type=float,
        default=0.20,
        help="Probability of partial fill in stochastic execution model.",
    )
    parser.add_argument(
        "--exec-partial-min",
        type=float,
        default=0.10,
        help="Minimum partial fill ratio in stochastic execution model.",
    )
    parser.add_argument(
        "--exec-partial-max",
        type=float,
        default=0.90,
        help="Maximum partial fill ratio in stochastic execution model.",
    )
    parser.add_argument(
        "--steps",
        default="01,02,03,04,05,06",
        help="Comma-separated step ids to run (e.g., 01,02,03).",
    )
    return parser.parse_args()


def parse_train_window_grid(text: str) -> List[Optional[int]]:
    vals: List[Optional[int]] = []
    for token in [x.strip().lower() for x in text.split(",") if x.strip()]:
        if token in {"none", "full", "all"}:
            vals.append(None)
        else:
            v = int(token)
            if v <= 0:
                raise ValueError("Train-window values must be positive integers or 'none'.")
            vals.append(v)
    if not vals:
        raise ValueError("Train-window grid cannot be empty.")
    dedup: List[Optional[int]] = []
    seen: set[Optional[int]] = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    return dedup


def parse_step_selection(text: str) -> List[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Step selection cannot be empty.")
    out: List[str] = []
    seen: set[str] = set()
    for raw in vals:
        token = raw.replace("step", "").replace("_", "").replace("-", "")
        if token.isdigit():
            token = f"{int(token):02d}"
        if token not in {"01", "02", "03", "04", "05", "06"}:
            raise ValueError(f"Unknown step id: {raw}")
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def make_folds_with_gap(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_days: int,
    gap_days: int,
    test_days: int,
    step_days: int,
) -> List[FoldWindow]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days/test_days/step_days must be positive.")
    if gap_days < 0:
        raise ValueError("gap_days must be non-negative.")

    train_td = pd.Timedelta(days=train_days)
    gap_td = pd.Timedelta(days=gap_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)

    folds: List[FoldWindow] = []
    cursor = start
    fold_id = 1
    while True:
        train_start = cursor
        train_end = train_start + train_td
        test_start = train_end + gap_td
        test_end = test_start + test_td
        if test_end > end:
            break
        folds.append(
            FoldWindow(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        cursor += step_td
    return folds


def add_context_features(
    train_features: Dict[str, pd.DataFrame],
    test_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    symbols = sorted(set(train_features.keys()) & set(test_features.keys()))
    if not symbols:
        return train_features, test_features

    all_close: Dict[str, pd.Series] = {}
    for symbol in symbols:
        tr = train_features[symbol]["Close"].astype(float)
        te = test_features[symbol]["Close"].astype(float)
        all_close[symbol] = pd.concat([tr, te]).sort_index()

    close_wide = pd.concat(all_close, axis=1, sort=True).sort_index().ffill()
    ret1_wide = close_wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_wide = ret1_wide.rolling(720, min_periods=50).std()

    mkt_ret1_mean = ret1_wide.mean(axis=1).fillna(0.0)
    mkt_ret1_std = ret1_wide.std(axis=1).fillna(0.0)
    mkt_vol_mean = vol_wide.mean(axis=1).fillna(0.0)
    mkt_ret1_mean_roll = mkt_ret1_mean.rolling(720, min_periods=50).mean().fillna(0.0)
    mkt_vol_mean_roll = mkt_vol_mean.rolling(720, min_periods=50).mean().fillna(0.0)

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        idx = out.index
        out["mkt_ret1_mean"] = mkt_ret1_mean.reindex(idx).fillna(0.0)
        out["mkt_ret1_std"] = mkt_ret1_std.reindex(idx).fillna(0.0)
        out["mkt_vol_mean"] = mkt_vol_mean.reindex(idx).fillna(0.0)
        out["mkt_ret1_mean_roll"] = mkt_ret1_mean_roll.reindex(idx).fillna(0.0)
        out["mkt_vol_mean_roll"] = mkt_vol_mean_roll.reindex(idx).fillna(0.0)
        out["ret1_vs_mkt"] = out["ret1"].fillna(0.0) - out["mkt_ret1_mean"]
        return out

    train_out = {symbol: _enrich(train_features[symbol]) for symbol in symbols}
    test_out = {symbol: _enrich(test_features[symbol]) for symbol in symbols}
    return train_out, test_out


def add_horizon_labels(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"fwd_ret_h{int(h)}"] = (
            out["Close"].pct_change(int(h)).shift(-int(h)).replace([np.inf, -np.inf], np.nan)
        )
    return out


def _matrix_from_df(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    X = (
        df[list(feature_cols)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values.astype(float)
    )
    return np.clip(X, -50.0, 50.0)


def _extract_train_sample(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    train_window: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    valid = (
        df["trend_pre"].astype(bool)
        & df[label_col].notna()
        & df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    )
    sample = df.loc[valid]
    if train_window is not None and len(sample) > int(train_window):
        sample = sample.iloc[-int(train_window) :]

    if sample.empty:
        return np.empty((0, len(feature_cols))), np.empty((0,), dtype=int), 0.5

    X = _matrix_from_df(sample, feature_cols)
    y = (sample[label_col].values > 0.0).astype(int)
    p = float(y.mean()) if len(y) else 0.5
    return X, y, p


def fit_tree_head(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    seed: int,
    model_kind: str,
    train_window: Optional[int],
    min_samples: int = 300,
) -> Tuple[Optional[object], float]:
    if (
        RandomForestClassifier is None
        or ExtraTreesClassifier is None
        or HistGradientBoostingClassifier is None
    ):
        raise RuntimeError(
            "scikit-learn is required for tree-based experiments. "
            "Install dependencies from requirements.txt."
        )
    X, y, pos_rate = _extract_train_sample(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        train_window=train_window,
    )
    if len(y) < min_samples:
        return None, pos_rate
    if pos_rate <= 0.01 or pos_rate >= 0.99:
        return None, pos_rate

    if model_kind == "rf":
        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=4,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=int(seed),
        )
    elif model_kind == "et":
        model = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=4,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=int(seed),
        )
    elif model_kind == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=350,
            max_depth=4,
            min_samples_leaf=15,
            learning_rate=0.05,
            random_state=int(seed),
        )
    elif model_kind == "xgb":
        if XGBClassifier is None:
            return None, pos_rate
        model = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(seed),
            n_jobs=1,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    try:
        model.fit(X, y)
    except Exception:
        return None, pos_rate
    return model, pos_rate


def predict_tree_head(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    model: Optional[object],
    const_prob: float,
) -> pd.Series:
    if model is None:
        return pd.Series(const_prob, index=df.index, dtype=float)
    X = _matrix_from_df(df, feature_cols)
    p = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return pd.Series(p, index=df.index, dtype=float)


def _split_fit_cal(
    train_df: pd.DataFrame,
    calibration_frac: float = 0.2,
    min_cal_rows: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train_df)
    if n <= min_cal_rows * 2:
        return train_df, train_df
    split = int(n * (1.0 - calibration_frac))
    split = max(min_cal_rows, min(split, n - min_cal_rows))
    if split <= 0 or split >= n:
        return train_df, train_df
    return train_df.iloc[:split].copy(), train_df.iloc[split:].copy()


def _tree_model_specs(use_ensemble: bool, use_xgboost: bool) -> List[Tuple[str, int]]:
    if use_ensemble:
        specs: List[Tuple[str, int]] = [
            ("rf", 11),
            ("rf", 29),
            ("rf", 47),
            ("et", 11),
            ("et", 29),
            ("et", 47),
            ("hgb", 11),
            ("hgb", 29),
            ("hgb", 47),
        ]
    else:
        specs = [("rf", 42), ("et", 42), ("hgb", 42)]
    if use_xgboost and XGBClassifier is not None:
        specs.extend([("xgb", 11), ("xgb", 29)] if use_ensemble else [("xgb", 42)])
    return specs


def _weighted_and_calibrated_probs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    use_ensemble: bool,
    train_window: Optional[int],
    use_xgboost: bool,
) -> pd.Series:
    fit_df, cal_df = _split_fit_cal(train_df, calibration_frac=0.2, min_cal_rows=60)
    model_specs = _tree_model_specs(use_ensemble=use_ensemble, use_xgboost=use_xgboost)

    cal_preds: List[pd.Series] = []
    test_preds: List[pd.Series] = []
    losses: List[float] = []

    y_cal = (cal_df[label_col].values > 0.0).astype(int) if label_col in cal_df.columns else np.array([])
    for kind, seed in model_specs:
        model, const_prob = fit_tree_head(
            df=fit_df,
            feature_cols=feature_cols,
            label_col=label_col,
            seed=seed,
            model_kind=kind,
            train_window=train_window,
        )
        p_cal = predict_tree_head(
            df=cal_df,
            feature_cols=feature_cols,
            model=model,
            const_prob=const_prob,
        )
        p_test = predict_tree_head(
            df=test_df,
            feature_cols=feature_cols,
            model=model,
            const_prob=const_prob,
        )
        cal_preds.append(p_cal)
        test_preds.append(p_test)

        if len(y_cal) == len(p_cal):
            brier = float(np.mean((p_cal.values.astype(float) - y_cal.astype(float)) ** 2))
        else:
            brier = 0.25
        losses.append(max(1e-6, brier))

    if not test_preds:
        return pd.Series(0.5, index=test_df.index, dtype=float)

    weights = np.asarray([1.0 / x for x in losses], dtype=float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(len(test_preds), dtype=float)
    weights = weights / float(weights.sum())

    test_mat = pd.concat(test_preds, axis=1).fillna(0.5).values.astype(float)
    p_weighted = np.dot(test_mat, weights)
    out = pd.Series(np.clip(p_weighted, 1e-6, 1.0 - 1e-6), index=test_df.index, dtype=float)

    if (
        LogisticRegression is not None
        and len(cal_preds) >= 2
        and len(y_cal) == len(cal_preds[0])
        and len(y_cal) >= 80
        and len(np.unique(y_cal)) >= 2
    ):
        Z_cal = pd.concat(cal_preds, axis=1).fillna(0.5).values.astype(float)
        Z_test = test_mat
        try:
            stacker = LogisticRegression(max_iter=200, solver="lbfgs")
            stacker.fit(Z_cal, y_cal)
            p_stack = stacker.predict_proba(Z_test)[:, 1].astype(float)
            out = pd.Series(
                np.clip(0.6 * out.values + 0.4 * p_stack, 1e-6, 1.0 - 1e-6),
                index=test_df.index,
                dtype=float,
            )
        except Exception:
            pass
    return out


def hysteresis_gate(prob: pd.Series, threshold: float, band: float) -> pd.Series:
    lower = max(1e-6, float(threshold) - float(band))
    upper = min(1.0 - 1e-6, float(threshold) + float(band))
    state = False
    out = np.zeros(len(prob), dtype=float)
    vals = prob.fillna(0.5).astype(float).values
    for i, p in enumerate(vals):
        if p >= upper:
            state = True
        elif p <= lower:
            state = False
        out[i] = 1.0 if state else 0.0
    return pd.Series(out, index=prob.index, dtype=float)


def fit_online_head(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    train_window: Optional[int],
) -> Tuple[Optional[StandardScaler], Optional[SGDClassifier], float]:
    if SGDClassifier is None or StandardScaler is None:
        raise RuntimeError(
            "scikit-learn is required for online-update experiments. "
            "Install dependencies from requirements.txt."
        )
    X, y, pos_rate = _extract_train_sample(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        train_window=train_window,
    )
    if len(y) < 200 or pos_rate <= 0.01 or pos_rate >= 0.99:
        return None, None, pos_rate

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = SGDClassifier(
        loss="log_loss",
        alpha=0.0003,
        random_state=42,
        max_iter=1000,
        tol=1e-3,
    )
    try:
        model.fit(Xs, y)
    except Exception:
        return None, None, pos_rate
    return scaler, model, pos_rate


def predict_online_with_updates(
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    scaler: Optional[StandardScaler],
    model: Optional[SGDClassifier],
    const_prob: float,
) -> pd.Series:
    probs = pd.Series(const_prob, index=test_df.index, dtype=float)
    if scaler is None or model is None or test_df.empty:
        return probs

    day_key = test_df.index.floor("D")
    for day in sorted(day_key.unique()):
        day_mask = day_key == day
        day_df = test_df.loc[day_mask]
        if day_df.empty:
            continue

        X_day = _matrix_from_df(day_df, feature_cols)
        X_day_scaled = scaler.transform(X_day)
        p = np.asarray(model.predict_proba(X_day_scaled)[:, 1], dtype=float)
        probs.loc[day_df.index] = np.clip(p, 1e-6, 1.0 - 1e-6)

        upd_mask = day_df[label_col].notna()
        if not bool(upd_mask.any()):
            continue
        upd_df = day_df.loc[upd_mask]
        y_upd = (upd_df[label_col].values > 0.0).astype(int)
        if len(y_upd) == 0:
            continue
        X_upd = _matrix_from_df(upd_df, feature_cols)
        X_upd_scaled = scaler.transform(X_upd)
        try:
            model.partial_fit(X_upd_scaled, y_upd, classes=np.array([0, 1], dtype=int))
        except Exception:
            continue

    return probs


def probs_from_aux_tree_stack(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    use_ensemble: bool,
    train_window: Optional[int],
    use_xgboost: bool,
) -> pd.Series:
    horizon_probs: List[pd.Series] = []
    for h in AUX_HORIZONS:
        label_col = f"fwd_ret_h{int(h)}"
        horizon_probs.append(
            _weighted_and_calibrated_probs(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                label_col=label_col,
                use_ensemble=use_ensemble,
                train_window=train_window,
                use_xgboost=use_xgboost,
            )
        )

    combo = pd.Series(0.0, index=test_df.index, dtype=float)
    for w, p in zip(AUX_WEIGHTS, horizon_probs):
        combo = combo + float(w) * p
    return combo.clip(lower=1e-6, upper=1.0 - 1e-6)


def build_positions_for_experiment(
    spec: ExperimentSpec,
    train_close: Dict[str, pd.Series],
    test_close: Dict[str, pd.Series],
    regime_params: RegimeParams,
    meta_threshold: float,
    meta_band: float,
    position_mode: str,
    sizing_scale: float,
    sizing_offset: float,
    sizing_vol_exponent: float,
    sizing_min_size: float,
    train_window: Optional[int],
    use_xgboost: bool,
) -> Dict[str, pd.Series]:
    train_features = build_feature_universe(train_close, regime_params)
    test_features = build_feature_universe(test_close, regime_params)

    symbols = sorted(set(train_features.keys()) & set(test_features.keys()))
    train_features = {s: train_features[s] for s in symbols}
    test_features = {s: test_features[s] for s in symbols}

    if spec.use_context_features:
        train_features, test_features = add_context_features(train_features, test_features)
        feature_cols = BASE_FEATURE_COLS + CONTEXT_FEATURE_COLS
    else:
        feature_cols = list(BASE_FEATURE_COLS)

    if spec.mode == "baseline":
        return {
            symbol: (test_features[symbol]["base_long"] > 0.0).astype(float)
            for symbol in symbols
        }

    pos_by_symbol: Dict[str, pd.Series] = {}
    for symbol in symbols:
        train_df = train_features[symbol].copy()
        test_df = test_features[symbol].copy()

        if spec.use_aux_targets:
            train_df = add_horizon_labels(train_df, AUX_HORIZONS)
            test_df = add_horizon_labels(test_df, AUX_HORIZONS)
        else:
            train_df = add_horizon_labels(train_df, (1,))
            test_df = add_horizon_labels(test_df, (1,))

        if spec.mode == "online":
            scaler, model, const_prob = fit_online_head(
                df=train_df,
                feature_cols=feature_cols,
                label_col="fwd_ret_h1",
                train_window=train_window,
            )
            prob = predict_online_with_updates(
                test_df=test_df,
                feature_cols=feature_cols,
                label_col="fwd_ret_h1",
                scaler=scaler,
                model=model,
                const_prob=const_prob,
            )
        elif spec.mode == "tree_aux":
            prob = probs_from_aux_tree_stack(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                use_ensemble=spec.use_ensemble,
                train_window=train_window,
                use_xgboost=use_xgboost,
            )
        else:
            raise ValueError(f"Unsupported experiment mode: {spec.mode}")

        base_long = (test_df["base_long"] > 0.0)
        gated = hysteresis_gate(prob, threshold=meta_threshold, band=meta_band)
        base_pos = (base_long & (gated > 0.0)).astype(float)
        if position_mode == "sized":
            strength = (test_df["spread_z"] - float(sizing_offset)).clip(lower=0.0).fillna(0.0)
            vol_ratio = test_df["vol_ratio"].clip(lower=0.5, upper=5.0).fillna(1.0)
            size = (float(sizing_scale) * strength / (vol_ratio ** float(sizing_vol_exponent))).clip(
                lower=0.0, upper=1.0
            )
            size = size.where(size >= float(sizing_min_size), 0.0).astype(float)
            pos_by_symbol[symbol] = (base_pos * size).astype(float)
        else:
            pos_by_symbol[symbol] = base_pos.astype(float)

    return pos_by_symbol


def choose_train_window(
    spec: ExperimentSpec,
    train_close: Dict[str, pd.Series],
    regime_params: RegimeParams,
    meta_threshold: float,
    meta_band: float,
    position_mode: str,
    sizing_scale: float,
    sizing_offset: float,
    sizing_vol_exponent: float,
    sizing_min_size: float,
    exec_model: str,
    exec_seed: int,
    exec_full_prob: float,
    exec_partial_prob: float,
    exec_partial_min: float,
    exec_partial_max: float,
    candidates: Sequence[Optional[int]],
    use_xgboost: bool,
) -> Optional[int]:
    train_features = build_feature_universe(train_close, regime_params)
    core_close, val_close, core_feat, val_feat = split_inner_train_validation(
        train_close,
        train_features,
    )
    use_validation = len(core_close) >= 2 and len(val_close) >= 2
    if not use_validation:
        return None

    best_window: Optional[int] = None
    best_key = (-1e9, -1e9, -1e9)

    for window in candidates:
        if spec.use_context_features:
            core_feat_use, val_feat_use = add_context_features(core_feat, val_feat)
        else:
            core_feat_use, val_feat_use = core_feat, val_feat

        feature_cols = BASE_FEATURE_COLS + (CONTEXT_FEATURE_COLS if spec.use_context_features else [])
        pos_by_symbol: Dict[str, pd.Series] = {}
        symbols = sorted(set(core_feat_use.keys()) & set(val_feat_use.keys()))

        for symbol in symbols:
            tr = core_feat_use[symbol].copy()
            va = val_feat_use[symbol].copy()
            if spec.use_aux_targets:
                tr = add_horizon_labels(tr, AUX_HORIZONS)
                va = add_horizon_labels(va, AUX_HORIZONS)
                prob = probs_from_aux_tree_stack(
                    train_df=tr,
                    test_df=va,
                    feature_cols=feature_cols,
                    use_ensemble=spec.use_ensemble,
                    train_window=window,
                    use_xgboost=use_xgboost,
                )
            else:
                tr = add_horizon_labels(tr, (1,))
                va = add_horizon_labels(va, (1,))
                scaler, model, const_prob = fit_online_head(
                    df=tr,
                    feature_cols=feature_cols,
                    label_col="fwd_ret_h1",
                    train_window=window,
                )
                prob = predict_online_with_updates(
                    test_df=va,
                    feature_cols=feature_cols,
                    label_col="fwd_ret_h1",
                    scaler=scaler,
                    model=model,
                    const_prob=const_prob,
                )

            gated = hysteresis_gate(prob, threshold=meta_threshold, band=meta_band)
            base_pos = ((va["base_long"] > 0.0) & (gated > 0.0)).astype(float)
            if position_mode == "sized":
                strength = (va["spread_z"] - float(sizing_offset)).clip(lower=0.0).fillna(0.0)
                vol_ratio = va["vol_ratio"].clip(lower=0.5, upper=5.0).fillna(1.0)
                size = (
                    float(sizing_scale) * strength / (vol_ratio ** float(sizing_vol_exponent))
                ).clip(lower=0.0, upper=1.0)
                size = size.where(size >= float(sizing_min_size), 0.0).astype(float)
                pos_by_symbol[symbol] = (base_pos * size).astype(float)
            else:
                pos_by_symbol[symbol] = base_pos.astype(float)

        ret, turn, exp, _ = simulate_equal_weight_positions(
            val_close,
            pos_by_symbol,
            execution_model=exec_model,
            execution_seed=exec_seed,
            full_fill_prob=exec_full_prob,
            partial_fill_prob=exec_partial_prob,
            partial_fill_min=exec_partial_min,
            partial_fill_max=exec_partial_max,
        )
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_window = window

    return best_window


def pooled_metrics(
    returns_chunks: List[pd.Series],
    turnover_chunks: List[pd.Series],
    exposure_chunks: List[pd.Series],
) -> Dict[str, float]:
    if not returns_chunks:
        zero = pd.Series(dtype=float)
        return summarize_returns(zero, zero, zero, BARS_PER_YEAR_2M)

    ret = pd.concat(returns_chunks).sort_index()
    ret = ret[~ret.index.duplicated(keep="last")]

    if turnover_chunks:
        turn = pd.concat(turnover_chunks).sort_index()
        turn = turn[~turn.index.duplicated(keep="last")]
        turn = turn.reindex(ret.index).fillna(0.0)
    else:
        turn = pd.Series(0.0, index=ret.index)

    if exposure_chunks:
        exp = pd.concat(exposure_chunks).sort_index()
        exp = exp[~exp.index.duplicated(keep="last")]
        exp = exp.reindex(ret.index).fillna(0.0)
    else:
        exp = pd.Series(0.0, index=ret.index)

    return summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)


def render_report(
    args: argparse.Namespace,
    folds: List[FoldWindow],
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Crypto Iteration Matrix Report")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "Ordered ablation of six ideas: gap-aware CV, online updates, auxiliary horizons, "
        "context features, ensemble averaging, and train-window selection."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Data glob: `{args.crypto_glob}`")
    lines.append(
        f"- Train/Gap/Test/Step days: {args.train_days}/{args.gap_days}/{args.test_days}/{args.step_days}"
    )
    lines.append(f"- Folds attempted: {len(folds)}")
    lines.append(
        f"- Folds evaluated: {fold_df['fold_id'].nunique() if not fold_df.empty else 0}"
    )
    lines.append(f"- Bars/year annualization: {BARS_PER_YEAR_2M:,}")
    lines.append(f"- Position mode: `{args.position_mode}`")
    lines.append(f"- Execution model: `{args.exec_model}`")
    lines.append("")
    lines.append("## Ablation Summary")
    lines.append("")
    lines.append(
        "| Step | Experiment | Folds | Mean Fold Return | Mean Fold Sharpe | Pooled Return | Pooled Sharpe | Pooled Max DD | Uplift vs Step 01 Return | Uplift vs Step 01 Sharpe |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    baseline = summary_df[summary_df["step"] == "01"]
    base_ret = float(baseline["pooled_total_return"].iloc[0]) if not baseline.empty else 0.0
    base_sh = float(baseline["pooled_net_sharpe"].iloc[0]) if not baseline.empty else 0.0

    for _, row in summary_df.sort_values("step").iterrows():
        uplift_ret = float(row["pooled_total_return"]) - base_ret
        uplift_sh = float(row["pooled_net_sharpe"]) - base_sh
        lines.append(
            f"| {row['step']} | {row['experiment']} | {int(row['folds'])} | "
            f"{row['mean_fold_total_return']:.4f} | {row['mean_fold_net_sharpe']:.3f} | "
            f"{row['pooled_total_return']:.4f} | {row['pooled_net_sharpe']:.3f} | "
            f"{row['pooled_net_max_drawdown']:.4f} | {uplift_ret:+.4f} | {uplift_sh:+.3f} |"
        )

    lines.append("")
    lines.append("## Fold Results")
    lines.append("")
    lines.append(
        "| Fold | Step | Experiment | Symbols | Test Return | Test Sharpe | Test Ann Return | Test Max DD | Avg Turnover | Avg Exposure | Params |"
    )
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for _, row in fold_df.sort_values(["fold_id", "step"]).iterrows():
        lines.append(
            f"| {int(row['fold_id'])} | {row['step']} | {row['experiment']} | {int(row['n_symbols'])} | "
            f"{row['test_total_return']:.4f} | {row['test_net_sharpe']:.3f} | {row['test_net_ann_return']:.3f} | "
            f"{row['test_net_max_drawdown']:.4f} | {row['test_avg_turnover']:.4f} | {row['test_avg_exposure']:.4f} | {row['params']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- These are research estimates with `cost_bps=0` inside the simulation primitive.")
    lines.append(
        f"- Execution model: `{args.exec_model}` (seed={args.exec_seed}, full={args.exec_full_prob}, partial={args.exec_partial_prob}, partial_range={args.exec_partial_min}-{args.exec_partial_max})."
    )
    lines.append(
        f"- Position mode: `{args.position_mode}` (sizing scale/offset/vol_exp/min = {args.sizing_scale}/{args.sizing_offset}/{args.sizing_vol_exponent}/{args.sizing_min_size})."
    )
    lines.append("- Step 02 uses day-by-day online updates with the 1-bar target only.")
    lines.append("- Step 03/04/05/06 use auxiliary horizon blending (1, 4, 20 bars).")
    lines.append("- Tree stacks use diverse model families (RF/ET/HGB and optional XGBoost).")
    lines.append("- Ensemble weights are inverse-Brier on a time-ordered calibration split.")
    lines.append("- Probabilities are calibrated with a logistic stacker when calibration data is sufficient.")
    lines.append("- Entry gate uses hysteresis around threshold (`meta_threshold ± meta_band`).")
    lines.append("- Step 06 selects train-window length on an inner time-ordered validation split.")
    lines.append("")
    return "\n".join(lines)


def experiment_specs() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            step="01",
            name="01_gap_cv_baseline",
            description="Gap-aware CV baseline (base regime only)",
            mode="baseline",
            use_context_features=False,
            use_aux_targets=False,
            use_ensemble=False,
            tune_train_window=False,
        ),
        ExperimentSpec(
            step="02",
            name="02_online_updates",
            description="Meta gate with daily online updates",
            mode="online",
            use_context_features=False,
            use_aux_targets=False,
            use_ensemble=False,
            tune_train_window=False,
        ),
        ExperimentSpec(
            step="03",
            name="03_aux_horizons",
            description="Auxiliary horizons (1/4/20) blended",
            mode="tree_aux",
            use_context_features=False,
            use_aux_targets=True,
            use_ensemble=False,
            tune_train_window=False,
        ),
        ExperimentSpec(
            step="04",
            name="04_context_features",
            description="Aux horizons + cross-asset context features",
            mode="tree_aux",
            use_context_features=True,
            use_aux_targets=True,
            use_ensemble=False,
            tune_train_window=False,
        ),
        ExperimentSpec(
            step="05",
            name="05_seed_ensemble",
            description="Context + aux with RF/ET seed ensemble",
            mode="tree_aux",
            use_context_features=True,
            use_aux_targets=True,
            use_ensemble=True,
            tune_train_window=False,
        ),
        ExperimentSpec(
            step="06",
            name="06_train_window_selection",
            description="Ensemble stack + train-window selection",
            mode="tree_aux",
            use_context_features=True,
            use_aux_targets=True,
            use_ensemble=True,
            tune_train_window=True,
        ),
    ]


def main() -> None:
    args = parse_args()
    if args.meta_band < 0.0 or args.meta_band >= 0.5:
        raise ValueError("--meta-band must be in [0, 0.5).")
    if args.sizing_scale <= 0.0:
        raise ValueError("--sizing-scale must be positive.")
    if args.sizing_vol_exponent <= 0.0:
        raise ValueError("--sizing-vol-exponent must be positive.")
    if args.sizing_min_size < 0.0:
        raise ValueError("--sizing-min-size must be non-negative.")
    if args.exec_full_prob < 0.0 or args.exec_partial_prob < 0.0:
        raise ValueError("--exec-full-prob/--exec-partial-prob must be non-negative.")
    if (args.exec_full_prob + args.exec_partial_prob) > 1.0:
        raise ValueError("--exec-full-prob + --exec-partial-prob must be <= 1.")
    if args.exec_partial_min < 0.0 or args.exec_partial_max > 1.0:
        raise ValueError("--exec-partial-min/max must be within [0, 1].")
    if args.exec_partial_min > args.exec_partial_max:
        raise ValueError("--exec-partial-min must be <= --exec-partial-max.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_params = RegimeParams(
        short_window=int(args.short_window),
        long_window=int(args.long_window),
        slope_lookback=int(args.slope_lookback),
        volatility_window=int(args.volatility_window),
        volatility_quantile=float(args.volatility_quantile),
    )
    window_candidates = parse_train_window_grid(args.train_window_grid)

    close_by_symbol = load_universe_from_glob(args.crypto_glob)
    if not close_by_symbol:
        raise RuntimeError("No symbols loaded.")

    common_start = max(series.index.min() for series in close_by_symbol.values())
    common_end = min(series.index.max() for series in close_by_symbol.values())
    folds = make_folds_with_gap(
        start=common_start,
        end=common_end,
        train_days=int(args.train_days),
        gap_days=int(args.gap_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
    )
    if not folds:
        raise RuntimeError("No folds produced. Reduce train/gap/test windows.")

    selected_steps = set(parse_step_selection(args.steps))
    specs = [spec for spec in experiment_specs() if spec.step in selected_steps]
    if not specs:
        raise RuntimeError("No experiments selected after --steps filtering.")
    fold_rows: List[Dict[str, object]] = []

    returns_chunks: Dict[str, List[pd.Series]] = {spec.name: [] for spec in specs}
    turnover_chunks: Dict[str, List[pd.Series]] = {spec.name: [] for spec in specs}
    exposure_chunks: Dict[str, List[pd.Series]] = {spec.name: [] for spec in specs}

    for fold in folds:
        train_close = slice_universe(
            close_by_symbol=close_by_symbol,
            start=fold.train_start,
            end=fold.train_end,
            min_bars=int(args.min_train_bars),
        )
        test_close = slice_universe(
            close_by_symbol=close_by_symbol,
            start=fold.test_start,
            end=fold.test_end,
            min_bars=int(args.min_test_bars),
        )
        symbols = sorted(set(train_close.keys()) & set(test_close.keys()))
        if len(symbols) < int(args.min_symbols):
            continue
        train_close = {s: train_close[s] for s in symbols}
        test_close = {s: test_close[s] for s in symbols}

        for spec in specs:
            chosen_window: Optional[int] = None
            if spec.tune_train_window:
                chosen_window = choose_train_window(
                    spec=spec,
                    train_close=train_close,
                    regime_params=regime_params,
                    meta_threshold=float(args.meta_threshold),
                    meta_band=float(args.meta_band),
                    position_mode=str(args.position_mode),
                    sizing_scale=float(args.sizing_scale),
                    sizing_offset=float(args.sizing_offset),
                    sizing_vol_exponent=float(args.sizing_vol_exponent),
                    sizing_min_size=float(args.sizing_min_size),
                    exec_model=str(args.exec_model),
                    exec_seed=int(args.exec_seed),
                    exec_full_prob=float(args.exec_full_prob),
                    exec_partial_prob=float(args.exec_partial_prob),
                    exec_partial_min=float(args.exec_partial_min),
                    exec_partial_max=float(args.exec_partial_max),
                    candidates=window_candidates,
                    use_xgboost=bool(args.use_xgboost),
                )

            pos = build_positions_for_experiment(
                spec=spec,
                train_close=train_close,
                test_close=test_close,
                regime_params=regime_params,
                meta_threshold=float(args.meta_threshold),
                meta_band=float(args.meta_band),
                position_mode=str(args.position_mode),
                sizing_scale=float(args.sizing_scale),
                sizing_offset=float(args.sizing_offset),
                sizing_vol_exponent=float(args.sizing_vol_exponent),
                sizing_min_size=float(args.sizing_min_size),
                train_window=chosen_window,
                use_xgboost=bool(args.use_xgboost),
            )
            ret, turn, exp, _ = simulate_equal_weight_positions(
                test_close,
                pos,
                execution_model=str(args.exec_model),
                execution_seed=int(args.exec_seed),
                full_fill_prob=float(args.exec_full_prob),
                partial_fill_prob=float(args.exec_partial_prob),
                partial_fill_min=float(args.exec_partial_min),
                partial_fill_max=float(args.exec_partial_max),
            )
            metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)

            params_text = (
                f"train_window={chosen_window if chosen_window is not None else 'full'}"
                if spec.tune_train_window
                else "train_window=full"
            )
            fold_rows.append(
                {
                    "fold_id": int(fold.fold_id),
                    "step": spec.step,
                    "experiment": spec.name,
                    "description": spec.description,
                    "n_symbols": int(len(symbols)),
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "test_total_return": float(metrics["total_return"]),
                    "test_net_sharpe": float(metrics["net_sharpe"]),
                    "test_net_ann_return": float(metrics["net_ann_return"]),
                    "test_net_max_drawdown": float(metrics["net_max_drawdown"]),
                    "test_avg_turnover": float(metrics["avg_turnover"]),
                    "test_avg_exposure": float(metrics["avg_exposure"]),
                    "params": params_text,
                }
            )
            returns_chunks[spec.name].append(ret.rename(f"fold{fold.fold_id}_{spec.name}"))
            turnover_chunks[spec.name].append(turn.rename(f"fold{fold.fold_id}_{spec.name}"))
            exposure_chunks[spec.name].append(exp.rename(f"fold{fold.fold_id}_{spec.name}"))

    if not fold_rows:
        raise RuntimeError("No folds evaluated after filters.")

    fold_df = pd.DataFrame(fold_rows).sort_values(["fold_id", "step"]).reset_index(drop=True)

    summary_rows: List[Dict[str, object]] = []
    for spec in specs:
        model_rows = fold_df[fold_df["experiment"] == spec.name]
        pooled = pooled_metrics(
            returns_chunks=returns_chunks[spec.name],
            turnover_chunks=turnover_chunks[spec.name],
            exposure_chunks=exposure_chunks[spec.name],
        )
        summary_rows.append(
            {
                "step": spec.step,
                "experiment": spec.name,
                "description": spec.description,
                "folds": int(len(model_rows)),
                "mean_fold_total_return": float(model_rows["test_total_return"].mean()) if len(model_rows) else 0.0,
                "mean_fold_net_sharpe": float(model_rows["test_net_sharpe"].mean()) if len(model_rows) else 0.0,
                "pooled_total_return": float(pooled["total_return"]),
                "pooled_net_sharpe": float(pooled["net_sharpe"]),
                "pooled_net_ann_return": float(pooled["net_ann_return"]),
                "pooled_net_max_drawdown": float(pooled["net_max_drawdown"]),
                "pooled_avg_turnover": float(pooled["avg_turnover"]),
                "pooled_avg_exposure": float(pooled["avg_exposure"]),
                "pooled_hit_rate": float(pooled["hit_rate"]),
                "pooled_bars": float(pooled["bars"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("step").reset_index(drop=True)

    fold_path = out_dir / "fold_results.csv"
    summary_path = out_dir / "ablation_summary.csv"
    report_path = out_dir / "report.md"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    report_path.write_text(render_report(args, folds, fold_df, summary_df))

    print("Crypto iteration matrix complete")
    print(f"Fold results: {fold_path}")
    print(f"Ablation summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

# Crypto Iteration Matrix Report

## Scope
Ordered ablation of six ideas: gap-aware CV, online updates, auxiliary horizons, context features, ensemble averaging, and train-window selection.

## Setup
- Data glob: `data/legacy/raw_data_crypto/*_2m_data.csv`
- Train/Gap/Test/Step days: 10/2/5/5
- Folds attempted: 3
- Folds evaluated: 3
- Bars/year annualization: 262,800
- Position mode: `binary`
- Execution model: `stochastic`

## Ablation Summary

| Step | Experiment | Folds | Mean Fold Return | Mean Fold Sharpe | Pooled Return | Pooled Sharpe | Pooled Max DD | Uplift vs Step 01 Return | Uplift vs Step 01 Sharpe |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 01_gap_cv_baseline | 3 | 0.0090 | 2.077 | 0.0267 | 5.183 | -0.0301 | +0.0000 | +0.000 |
| 02 | 02_online_updates | 3 | 0.0226 | 23.578 | 0.0688 | 23.103 | -0.0109 | +0.0421 | +17.920 |
| 03 | 03_aux_horizons | 3 | 0.0190 | 13.825 | 0.0573 | 19.022 | -0.0132 | +0.0306 | +13.839 |
| 04 | 04_context_features | 3 | 0.0293 | 38.598 | 0.0897 | 37.364 | -0.0070 | +0.0629 | +32.181 |
| 05 | 05_seed_ensemble | 3 | 0.0288 | 33.218 | 0.0881 | 34.597 | -0.0065 | +0.0614 | +29.414 |
| 06 | 06_train_window_selection | 3 | 0.0274 | 38.961 | 0.0839 | 40.133 | -0.0069 | +0.0571 | +34.949 |

## Fold Results

| Fold | Step | Experiment | Symbols | Test Return | Test Sharpe | Test Ann Return | Test Max DD | Avg Turnover | Avg Exposure | Params |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 01 | 01_gap_cv_baseline | 3 | -0.0098 | -11.180 | -0.720 | -0.0143 | 0.0019 | 0.0739 | train_window=full |
| 1 | 02 | 02_online_updates | 3 | 0.0038 | 9.021 | 0.283 | -0.0077 | 0.0058 | 0.0166 | train_window=full |
| 1 | 03 | 03_aux_horizons | 3 | -0.0005 | -0.969 | -0.039 | -0.0092 | 0.0085 | 0.0400 | train_window=full |
| 1 | 04 | 04_context_features | 3 | 0.0087 | 14.666 | 0.634 | -0.0070 | 0.0084 | 0.0384 | train_window=full |
| 1 | 05 | 05_seed_ensemble | 3 | 0.0088 | 14.675 | 0.642 | -0.0065 | 0.0086 | 0.0405 | train_window=full |
| 1 | 06 | 06_train_window_selection | 3 | 0.0049 | 9.565 | 0.358 | -0.0069 | 0.0103 | 0.0367 | train_window=2000 |
| 2 | 01 | 01_gap_cv_baseline | 3 | 0.0282 | 11.302 | 2.057 | -0.0183 | 0.0030 | 0.2209 | train_window=full |
| 2 | 02 | 02_online_updates | 3 | 0.0443 | 29.076 | 3.195 | -0.0052 | 0.0225 | 0.1041 | train_window=full |
| 2 | 03 | 03_aux_horizons | 3 | 0.0540 | 34.319 | 3.871 | -0.0063 | 0.0204 | 0.1044 | train_window=full |
| 2 | 04 | 04_context_features | 3 | 0.0603 | 52.117 | 4.305 | -0.0042 | 0.0232 | 0.0461 | train_window=full |
| 2 | 05 | 05_seed_ensemble | 3 | 0.0592 | 49.223 | 4.231 | -0.0043 | 0.0244 | 0.0484 | train_window=full |
| 2 | 06 | 06_train_window_selection | 3 | 0.0558 | 57.462 | 3.995 | -0.0033 | 0.0238 | 0.0473 | train_window=2000 |
| 3 | 01 | 01_gap_cv_baseline | 3 | 0.0085 | 6.107 | 0.624 | -0.0145 | 0.0033 | 0.1081 | train_window=full |
| 3 | 02 | 02_online_updates | 3 | 0.0195 | 32.637 | 1.413 | -0.0022 | 0.0088 | 0.0146 | train_window=full |
| 3 | 03 | 03_aux_horizons | 3 | 0.0037 | 8.124 | 0.270 | -0.0036 | 0.0045 | 0.0070 | train_window=full |
| 3 | 04 | 04_context_features | 3 | 0.0189 | 49.010 | 1.369 | -0.0004 | 0.0072 | 0.0123 | train_window=full |
| 3 | 05 | 05_seed_ensemble | 3 | 0.0183 | 35.756 | 1.330 | -0.0029 | 0.0080 | 0.0127 | train_window=full |
| 3 | 06 | 06_train_window_selection | 3 | 0.0216 | 49.856 | 1.563 | -0.0008 | 0.0089 | 0.0141 | train_window=1000 |

## Notes
- These are research estimates with `cost_bps=0` inside the simulation primitive.
- Execution model: `stochastic` (seed=42, full=0.7, partial=0.2, partial_range=0.1-0.9).
- Position mode: `binary` (sizing scale/offset/vol_exp/min = 1.5/0.3/0.75/0.05).
- Step 02 uses day-by-day online updates with the 1-bar target only.
- Step 03/04/05/06 use auxiliary horizon blending (1, 4, 20 bars).
- Tree stacks use diverse model families (RF/ET/HGB and optional XGBoost).
- Ensemble weights are inverse-Brier on a time-ordered calibration split.
- Probabilities are calibrated with a logistic stacker when calibration data is sufficient.
- Entry gate uses hysteresis around threshold (`meta_threshold ± meta_band`).
- Step 06 selects train-window length on an inner time-ordered validation split.

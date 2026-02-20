# Crypto Iteration Matrix Report

## Scope
Ordered ablation of six ideas: gap-aware CV, online updates, auxiliary horizons, context features, ensemble averaging, and train-window selection.

## Setup
- Data glob: `data/legacy/raw_data_crypto/*_2m_data.csv`
- Train/Gap/Test/Step days: 3/1/2/3
- Folds attempted: 9
- Folds evaluated: 9
- Bars/year annualization: 262,800

## Ablation Summary

| Step | Experiment | Folds | Mean Fold Return | Mean Fold Sharpe | Pooled Return | Pooled Sharpe | Pooled Max DD | Uplift vs Step 01 Return | Uplift vs Step 01 Sharpe |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 01_gap_cv_baseline | 9 | -0.0009 | -6.969 | -0.0079 | -2.807 | -0.0192 | +0.0000 | +0.000 |
| 03 | 03_aux_horizons | 9 | 0.0000 | -1.050 | 0.0003 | 0.245 | -0.0057 | +0.0082 | +3.052 |
| 04 | 04_context_features | 9 | -0.0000 | -1.547 | -0.0001 | -0.023 | -0.0057 | +0.0078 | +2.784 |
| 05 | 05_seed_ensemble | 9 | -0.0000 | -1.605 | -0.0001 | -0.070 | -0.0057 | +0.0077 | +2.737 |

## Fold Results

| Fold | Step | Experiment | Symbols | Test Return | Test Sharpe | Test Ann Return | Test Max DD | Avg Turnover | Avg Exposure | Params |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 01 | 01_gap_cv_baseline | 3 | -0.0018 | -6.881 | -0.320 | -0.0053 | 0.0023 | 0.1123 | train_window=full |
| 1 | 03 | 03_aux_horizons | 3 | -0.0001 | -0.711 | -0.015 | -0.0015 | 0.0007 | 0.0484 | train_window=full |
| 1 | 04 | 04_context_features | 3 | -0.0001 | -0.711 | -0.015 | -0.0015 | 0.0007 | 0.0484 | train_window=full |
| 1 | 05 | 05_seed_ensemble | 3 | -0.0001 | -0.711 | -0.015 | -0.0015 | 0.0007 | 0.0484 | train_window=full |
| 2 | 01 | 01_gap_cv_baseline | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 2 | 03 | 03_aux_horizons | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 2 | 04 | 04_context_features | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 2 | 05 | 05_seed_ensemble | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 3 | 01 | 01_gap_cv_baseline | 3 | 0.0095 | 13.165 | 1.740 | -0.0093 | 0.0016 | 0.0961 | train_window=full |
| 3 | 03 | 03_aux_horizons | 3 | 0.0025 | 5.576 | 0.465 | -0.0057 | 0.0014 | 0.0519 | train_window=full |
| 3 | 04 | 04_context_features | 3 | 0.0025 | 5.576 | 0.465 | -0.0057 | 0.0014 | 0.0519 | train_window=full |
| 3 | 05 | 05_seed_ensemble | 3 | 0.0025 | 5.576 | 0.465 | -0.0057 | 0.0014 | 0.0519 | train_window=full |
| 4 | 01 | 01_gap_cv_baseline | 3 | 0.0004 | 13.580 | 0.080 | 0.0000 | 0.0005 | 0.0002 | train_window=full |
| 4 | 03 | 03_aux_horizons | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 4 | 04 | 04_context_features | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 4 | 05 | 05_seed_ensemble | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 5 | 01 | 01_gap_cv_baseline | 3 | -0.0037 | -22.898 | -0.676 | -0.0045 | 0.0009 | 0.0070 | train_window=full |
| 5 | 03 | 03_aux_horizons | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 5 | 04 | 04_context_features | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 5 | 05 | 05_seed_ensemble | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 6 | 01 | 01_gap_cv_baseline | 3 | -0.0050 | -25.053 | -0.913 | -0.0062 | 0.0014 | 0.0526 | train_window=full |
| 6 | 03 | 03_aux_horizons | 3 | -0.0021 | -14.318 | -0.383 | -0.0039 | 0.0019 | 0.0343 | train_window=full |
| 6 | 04 | 04_context_features | 3 | -0.0025 | -18.788 | -0.455 | -0.0033 | 0.0014 | 0.0259 | train_window=full |
| 6 | 05 | 05_seed_ensemble | 3 | -0.0026 | -19.309 | -0.468 | -0.0033 | 0.0019 | 0.0255 | train_window=full |
| 7 | 01 | 01_gap_cv_baseline | 3 | -0.0002 | -0.714 | -0.028 | -0.0021 | 0.0009 | 0.0372 | train_window=full |
| 7 | 03 | 03_aux_horizons | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 7 | 04 | 04_context_features | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 7 | 05 | 05_seed_ensemble | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 8 | 01 | 01_gap_cv_baseline | 3 | -0.0023 | -6.204 | -0.423 | -0.0081 | 0.0012 | 0.0553 | train_window=full |
| 8 | 03 | 03_aux_horizons | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 8 | 04 | 04_context_features | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 8 | 05 | 05_seed_ensemble | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 9 | 01 | 01_gap_cv_baseline | 3 | -0.0048 | -27.716 | -0.881 | -0.0055 | 0.0014 | 0.0044 | train_window=full |
| 9 | 03 | 03_aux_horizons | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 9 | 04 | 04_context_features | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |
| 9 | 05 | 05_seed_ensemble | 3 | 0.0000 | 0.000 | 0.000 | 0.0000 | 0.0000 | 0.0000 | train_window=full |

## Notes
- These are research estimates with `cost_bps=0` inside the simulation primitive.
- Step 02 uses day-by-day online updates with the 1-bar target only.
- Step 03/04/05/06 use auxiliary horizon blending (1, 4, 20 bars).
- Tree stacks use diverse model families (RF/ET/HGB and optional XGBoost).
- Ensemble weights are inverse-Brier on a time-ordered calibration split.
- Probabilities are calibrated with a logistic stacker when calibration data is sufficient.
- Entry gate uses hysteresis around threshold (`meta_threshold ± meta_band`).
- Step 06 selects train-window length on an inner time-ordered validation split.

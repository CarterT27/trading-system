# Crypto Strategy Experiment Report

## Objective
Run walk-forward experiments (`cost_bps=0`) for six strategy ideas and decide which are strong enough to add.

## Data and Setup
- Data glob: `data/legacy/raw_data_crypto/*_2m_data.csv`
- Train/Test/Step days: 10/5/5
- Bars/year used for annualization (2-minute crypto): 262,800
- Fold count attempted: 4 | Fold count evaluated: 4
- Cost model: 0 bps linear transaction cost (paper competition assumption).
- Lookahead audit (truncation invariance): PASS (28/28).

## Experiment Families
- `base_regime_tuned`: baseline EMA regime model with fold-wise train tuning.
- `regime_decomposition`: train-time regime-state expectancy map, then test-time state gating.
- `quality_filter`: spread-z and medium-horizon momentum quality gates.
- `meta_model`: sklearn classifier probability gate on top of baseline entries.
- `dynamic_sizing`: fractional sizing from trend strength and volatility normalization.
- `exit_analysis`: baseline vs time-stop/trailing-stop/combined exits.
- `cross_asset_overlay`: active-equal/strength/inv-vol/top1 portfolio overlays.

## OOS Summary

| Model | Folds | Mean Fold Return | Mean Fold Sharpe | Median Fold Sharpe | Positive Fold % | Pooled Return | Pooled Sharpe | Pooled Ann Return | Pooled Max DD | Mean Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| meta_model | 4 | 0.046 | 36.575 | 38.095 | 100.00% | 0.198 | 34.901 | 3.317 | -0.007 | 0.091 |
| dynamic_sizing | 4 | 0.006 | 3.901 | 5.857 | 50.00% | 0.026 | 3.899 | 0.469 | -0.030 | 0.131 |
| quality_filter | 4 | 0.003 | 2.321 | 3.909 | 50.00% | 0.011 | 1.763 | 0.214 | -0.034 | 0.133 |
| exit_analysis | 4 | 0.002 | 2.796 | 2.596 | 50.00% | 0.006 | 0.825 | 0.112 | -0.048 | 0.179 |
| base_regime_tuned | 4 | 0.000 | 0.778 | 2.365 | 50.00% | 0.001 | 0.157 | 0.023 | -0.046 | 0.199 |
| cross_asset_overlay | 4 | -0.009 | -2.969 | -3.460 | 25.00% | -0.037 | -3.219 | -0.672 | -0.069 | 0.284 |
| regime_decomposition | 4 | -0.012 | -3.261 | 0.023 | 50.00% | -0.048 | -5.168 | -0.890 | -0.106 | 0.222 |

## Recommendations

| Model | Decision | Reason |
| --- | --- | --- |
| meta_model | add_now | Strong pooled-return lift with good stability and no drawdown degradation. |
| dynamic_sizing | paper_test | Moderate improvement; validate with shadow paper runs before promotion. |
| quality_filter | paper_test | Moderate improvement; validate with shadow paper runs before promotion. |
| exit_analysis | paper_test | Moderate improvement; validate with shadow paper runs before promotion. |
| base_regime_tuned | control | Reference control model. |
| cross_asset_overlay | do_not_add | No robust OOS edge relative to control. |
| regime_decomposition | do_not_add | No robust OOS edge relative to control. |

## Fold-Level Notes

| Fold | Model | Train Return | Train Sharpe | Test Return | Test Sharpe | Test Ann Return | Test Max DD | Params |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | base_regime_tuned | 0.052 | 12.437 | -0.021 | -10.811 | -1.549 | -0.032 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] |
| 1 | cross_asset_overlay | 0.108 | 16.482 | -0.023 | -7.689 | -1.680 | -0.045 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | overlay_mode=strength |
| 1 | dynamic_sizing | 0.059 | 19.539 | -0.015 | -9.505 | -1.133 | -0.026 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | offset=0.25,scale=2.0 |
| 1 | exit_analysis | 0.052 | 12.437 | -0.021 | -10.811 | -1.549 | -0.032 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | mode=baseline,time_bars=None,trail_stop=None |
| 1 | meta_model | 0.190 | 58.992 | 0.038 | 30.449 | 2.766 | -0.005 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | prob_threshold=0.5 |
| 1 | quality_filter | 0.064 | 18.935 | -0.016 | -8.443 | -1.144 | -0.026 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | require_ret30_positive=False,spread_z_threshold=0.25 |
| 1 | regime_decomposition | 0.099 | 14.343 | -0.051 | -18.959 | -3.857 | -0.060 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | allowed_states=1,2,3,5,6,7,min_samples=50 |
| 2 | base_regime_tuned | 0.034 | 11.879 | 0.012 | 4.785 | 0.894 | -0.034 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] |
| 2 | cross_asset_overlay | 0.054 | 11.632 | -0.010 | -2.540 | -0.677 | -0.047 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | overlay_mode=strength |
| 2 | dynamic_sizing | 0.031 | 13.167 | 0.032 | 13.393 | 2.309 | -0.021 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | offset=0.0,scale=1.5 |
| 2 | exit_analysis | 0.041 | 14.799 | 0.011 | 5.246 | 0.834 | -0.026 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | mode=combo,time_bars=360,trail_stop=0.008 |
| 2 | meta_model | 0.116 | 55.029 | 0.067 | 39.660 | 4.766 | -0.007 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | prob_threshold=0.5 |
| 2 | quality_filter | 0.038 | 16.444 | 0.021 | 9.907 | 1.576 | -0.016 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | require_ret30_positive=True,spread_z_threshold=0.0 |
| 2 | regime_decomposition | 0.028 | 9.860 | -0.006 | -2.010 | -0.453 | -0.050 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | allowed_states=2,3,7,min_samples=50 |
| 3 | base_regime_tuned | -0.014 | -3.457 | 0.010 | 9.190 | 0.755 | -0.011 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] |
| 3 | cross_asset_overlay | -0.037 | -6.189 | 0.006 | 2.734 | 0.433 | -0.029 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | overlay_mode=top1 |
| 3 | dynamic_sizing | 0.014 | 4.916 | 0.011 | 12.388 | 0.776 | -0.006 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | offset=0.5,scale=1.0 |
| 3 | exit_analysis | -0.002 | -0.459 | 0.016 | 16.803 | 1.171 | -0.006 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | mode=time,time_bars=90,trail_stop=None |
| 3 | meta_model | 0.106 | 47.670 | 0.025 | 39.150 | 1.796 | -0.003 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | prob_threshold=0.5 |
| 3 | quality_filter | 0.029 | 9.007 | 0.008 | 9.534 | 0.607 | -0.010 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | require_ret30_positive=True,spread_z_threshold=0.5 |
| 3 | regime_decomposition | 0.000 | 0.081 | 0.006 | 5.870 | 0.443 | -0.011 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | allowed_states=7,min_samples=50 |
| 4 | base_regime_tuned | 0.044 | 7.975 | -0.000 | -0.054 | -0.008 | -0.020 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] |
| 4 | cross_asset_overlay | 0.035 | 5.093 | -0.011 | -4.380 | -0.769 | -0.024 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | overlay_mode=active_equal |
| 4 | dynamic_sizing | 0.071 | 16.305 | -0.001 | -0.675 | -0.068 | -0.017 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | offset=0.5,scale=2.0 |
| 4 | exit_analysis | 0.044 | 7.975 | -0.000 | -0.054 | -0.008 | -0.020 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | mode=baseline,time_bars=None,trail_stop=None |
| 4 | meta_model | 0.260 | 58.954 | 0.055 | 37.040 | 3.947 | -0.006 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | prob_threshold=0.5 |
| 4 | quality_filter | 0.080 | 17.988 | -0.003 | -1.715 | -0.181 | -0.019 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | require_ret30_positive=False,spread_z_threshold=0.75 |
| 4 | regime_decomposition | 0.041 | 8.980 | 0.004 | 2.056 | 0.296 | -0.018 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | allowed_states=7,min_samples=50 |

## Caveats
- Sample window is limited to available local files; conclusions are provisional.
- `cost_bps=0` matches competition optimization, but high-turnover variants still carry operational risk.
- Meta model uses sklearn `RandomForestClassifier` with per-symbol training and probability gating.
- Lookahead audit checks whether decisions at timestamp t are unchanged when recomputed with data truncated at t.
- Overlay models can increase gross exposure vs baseline equal-average construction; compare with that in mind.

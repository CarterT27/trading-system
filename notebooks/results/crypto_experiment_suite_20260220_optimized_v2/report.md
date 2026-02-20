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
- Optimization protocol for meta/sizing/quality/exit: per-fold inner train/validation split (time-ordered) for parameter selection.

## OOS Summary

| Model | Folds | Mean Fold Return | Mean Fold Sharpe | Median Fold Sharpe | Positive Fold % | Pooled Return | Pooled Sharpe | Pooled Ann Return | Pooled Max DD | Mean Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| meta_model | 4 | 0.041 | 38.110 | 38.351 | 100.00% | 0.175 | 36.112 | 2.964 | -0.007 | 0.079 |
| quality_filter | 4 | 0.022 | 16.911 | 17.881 | 100.00% | 0.090 | 16.356 | 1.590 | -0.012 | 0.088 |
| dynamic_sizing | 4 | 0.006 | 3.828 | 6.564 | 75.00% | 0.024 | 3.753 | 0.443 | -0.036 | 0.132 |
| exit_analysis | 4 | 0.005 | 4.410 | 7.168 | 75.00% | 0.018 | 3.040 | 0.340 | -0.033 | 0.142 |
| base_regime_tuned | 4 | 0.000 | 0.778 | 2.365 | 50.00% | 0.001 | 0.157 | 0.023 | -0.046 | 0.199 |
| cross_asset_overlay | 4 | -0.009 | -2.969 | -3.460 | 25.00% | -0.037 | -3.219 | -0.672 | -0.069 | 0.284 |
| regime_decomposition | 4 | -0.012 | -3.261 | 0.023 | 50.00% | -0.048 | -5.168 | -0.890 | -0.106 | 0.222 |

## Recommendations

| Model | Decision | Reason |
| --- | --- | --- |
| meta_model | add_now | Strong pooled-return lift with good stability and no drawdown degradation. |
| quality_filter | add_now | Strong pooled-return lift with good stability and no drawdown degradation. |
| dynamic_sizing | add_now | Strong pooled-return lift with good stability and no drawdown degradation. |
| exit_analysis | paper_test | Moderate improvement; validate with shadow paper runs before promotion. |
| base_regime_tuned | control | Reference control model. |
| cross_asset_overlay | do_not_add | No robust OOS edge relative to control. |
| regime_decomposition | do_not_add | No robust OOS edge relative to control. |

## Fold-Level Notes

| Fold | Model | Train Return | Train Sharpe | Test Return | Test Sharpe | Test Ann Return | Test Max DD | Params |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | base_regime_tuned | 0.052 | 12.437 | -0.021 | -10.811 | -1.549 | -0.032 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] |
| 1 | cross_asset_overlay | 0.108 | 16.482 | -0.023 | -7.689 | -1.680 | -0.045 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | overlay_mode=strength |
| 1 | dynamic_sizing | 0.058 | 19.640 | -0.018 | -11.850 | -1.346 | -0.029 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | min_size=0.0,offset=0.3,scale=1.5,smooth_span=8,vol_exponent=1.25 |
| 1 | exit_analysis | 0.033 | 8.843 | -0.017 | -9.509 | -1.278 | -0.028 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | min_hold_bars=0,mode=trail,profit_take=None,time_bars=None,trail_stop=0.012 |
| 1 | meta_model | 0.197 | 60.782 | 0.039 | 31.516 | 2.835 | -0.005 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | max_depth=4,max_features=sqrt,min_samples_leaf=15,n_estimators=200,prob_threshold=0.5 |
| 1 | quality_filter | 0.062 | 22.119 | 0.006 | 5.102 | 0.460 | -0.012 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | require_ret30_positive=False,require_ret5_positive=True,spread_z_threshold=0.0,trend_tstat_threshold=0.0,vol_ratio_max=1.2 |
| 1 | regime_decomposition | 0.099 | 14.343 | -0.051 | -18.959 | -3.857 | -0.060 | regime[short=30,long=180,slope=120,vol_w=1440,vol_q=0.80] | allowed_states=1,2,3,5,6,7,min_samples=50 |
| 2 | base_regime_tuned | 0.034 | 11.879 | 0.012 | 4.785 | 0.894 | -0.034 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] |
| 2 | cross_asset_overlay | 0.054 | 11.632 | -0.010 | -2.540 | -0.677 | -0.047 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | overlay_mode=strength |
| 2 | dynamic_sizing | 0.029 | 12.979 | 0.031 | 14.035 | 2.237 | -0.018 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | min_size=0.05,offset=0.3,scale=1.5,smooth_span=1,vol_exponent=0.75 |
| 2 | exit_analysis | 0.020 | 12.667 | 0.005 | 3.485 | 0.386 | -0.015 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | min_hold_bars=10,mode=time,profit_take=0.007,time_bars=90,trail_stop=None |
| 2 | meta_model | 0.112 | 73.056 | 0.044 | 44.222 | 3.175 | -0.005 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | max_depth=6,max_features=sqrt,min_samples_leaf=15,n_estimators=350,prob_threshold=0.55 |
| 2 | quality_filter | 0.042 | 23.133 | 0.048 | 26.780 | 3.446 | -0.006 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | require_ret30_positive=False,require_ret5_positive=True,spread_z_threshold=0.6,trend_tstat_threshold=0.0,vol_ratio_max=1.2 |
| 2 | regime_decomposition | 0.028 | 9.860 | -0.006 | -2.010 | -0.453 | -0.050 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | allowed_states=2,3,7,min_samples=50 |
| 3 | base_regime_tuned | -0.014 | -3.457 | 0.010 | 9.190 | 0.755 | -0.011 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] |
| 3 | cross_asset_overlay | -0.037 | -6.189 | 0.006 | 2.734 | 0.433 | -0.029 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | overlay_mode=top1 |
| 3 | dynamic_sizing | 0.013 | 4.160 | 0.011 | 12.518 | 0.816 | -0.006 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | min_size=0.0,offset=0.6,scale=1.5,smooth_span=1,vol_exponent=0.75 |
| 3 | exit_analysis | -0.001 | -0.309 | 0.013 | 12.814 | 0.946 | -0.008 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | min_hold_bars=0,mode=combo,profit_take=None,time_bars=180,trail_stop=0.008 |
| 3 | meta_model | 0.105 | 47.155 | 0.025 | 39.082 | 1.800 | -0.003 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | max_depth=4,max_features=sqrt,min_samples_leaf=30,n_estimators=350,prob_threshold=0.5 |
| 3 | quality_filter | 0.041 | 16.058 | 0.019 | 24.650 | 1.343 | -0.003 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | require_ret30_positive=False,require_ret5_positive=True,spread_z_threshold=0.6,trend_tstat_threshold=0.3,vol_ratio_max=1.2 |
| 3 | regime_decomposition | 0.000 | 0.081 | 0.006 | 5.870 | 0.443 | -0.011 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.60] | allowed_states=7,min_samples=50 |
| 4 | base_regime_tuned | 0.044 | 7.975 | -0.000 | -0.054 | -0.008 | -0.020 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] |
| 4 | cross_asset_overlay | 0.035 | 5.093 | -0.011 | -4.380 | -0.769 | -0.024 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | overlay_mode=active_equal |
| 4 | dynamic_sizing | 0.067 | 14.969 | 0.001 | 0.611 | 0.070 | -0.016 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | min_size=0.05,offset=0.3,scale=1.5,smooth_span=1,vol_exponent=0.75 |
| 4 | exit_analysis | 0.039 | 9.743 | 0.018 | 10.852 | 1.297 | -0.012 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | min_hold_bars=0,mode=combo,profit_take=None,time_bars=180,trail_stop=0.008 |
| 4 | meta_model | 0.275 | 62.706 | 0.057 | 37.620 | 4.045 | -0.007 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | max_depth=4,max_features=sqrt,min_samples_leaf=15,n_estimators=350,prob_threshold=0.5 |
| 4 | quality_filter | 0.117 | 29.042 | 0.015 | 11.112 | 1.121 | -0.010 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | require_ret30_positive=False,require_ret5_positive=True,spread_z_threshold=0.0,trend_tstat_threshold=0.0,vol_ratio_max=1.2 |
| 4 | regime_decomposition | 0.041 | 8.980 | 0.004 | 2.056 | 0.296 | -0.018 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | allowed_states=7,min_samples=50 |

## Caveats
- Sample window is limited to available local files; conclusions are provisional.
- `cost_bps=0` matches competition optimization, but high-turnover variants still carry operational risk.
- Meta model uses sklearn `RandomForestClassifier` with per-symbol training and probability gating.
- Lookahead audit checks whether decisions at timestamp t are unchanged when recomputed with data truncated at t.
- Overlay models can increase gross exposure vs baseline equal-average construction; compare with that in mind.

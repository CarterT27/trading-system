# Crypto Strategy Experiment Report

## Objective
Run walk-forward experiments (`cost_bps=0`) for six strategy ideas and decide which are strong enough to add.

## Data and Setup
- Data glob: `data/legacy/raw_data_crypto/*_2m_data.csv`
- Train/Test/Step days: 10/5/5
- Bars/year used for annualization (2-minute crypto): 262,800
- Fold count attempted: 4 | Fold count evaluated: 4
- Cost model: 0 bps linear transaction cost (paper competition assumption).

## Experiment Families
- `base_regime_tuned`: baseline EMA regime model with fold-wise train tuning.
- `regime_decomposition`: train-time regime-state expectancy map, then test-time state gating.
- `quality_filter`: spread-z and medium-horizon momentum quality gates.
- `meta_model`: in-script logistic gate on top of baseline entries.
- `dynamic_sizing`: fractional sizing from trend strength and volatility normalization.
- `exit_analysis`: baseline vs time-stop/trailing-stop/combined exits.
- `cross_asset_overlay`: active-equal/strength/inv-vol/top1 portfolio overlays.

## OOS Summary

| Model | Folds | Mean Fold Sharpe | Median Fold Sharpe | Positive Fold % | Pooled Sharpe | Pooled Ann Return | Pooled Max DD | Mean Exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| meta_model | 4 | 29.109 | 32.103 | 100.00% | 26.692 | 1.985 | -0.006 | 0.050 |
| dynamic_sizing | 4 | 0.651 | 6.998 | 50.00% | 5.124 | 0.526 | -0.019 | 0.090 |
| quality_filter | 4 | 0.353 | 4.011 | 50.00% | 2.656 | 0.255 | -0.028 | 0.080 |
| exit_analysis | 4 | 3.268 | 6.444 | 75.00% | 2.601 | 0.310 | -0.037 | 0.130 |
| base_regime_tuned | 4 | -0.799 | 2.365 | 50.00% | -0.812 | -0.116 | -0.047 | 0.186 |
| regime_decomposition | 4 | -0.939 | 0.023 | 50.00% | -2.860 | -0.494 | -0.091 | 0.220 |
| cross_asset_overlay | 4 | -3.646 | -3.460 | 25.00% | -3.951 | -0.815 | -0.062 | 0.268 |

## Recommendations

| Model | Decision | Reason |
| --- | --- | --- |
| meta_model | add_now | Improves pooled Sharpe with acceptable drawdown and fold stability. |
| dynamic_sizing | add_now | Improves pooled Sharpe with acceptable drawdown and fold stability. |
| quality_filter | add_now | Improves pooled Sharpe with acceptable drawdown and fold stability. |
| exit_analysis | add_now | Improves pooled Sharpe with acceptable drawdown and fold stability. |
| base_regime_tuned | control | Reference control model. |
| regime_decomposition | do_not_add | No robust OOS edge relative to control. |
| cross_asset_overlay | do_not_add | No robust OOS edge relative to control. |

## Fold-Level Notes

| Fold | Model | Train Sharpe | Test Sharpe | Test Ann Return | Test Max DD | Params |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | base_regime_tuned | 12.774 | -17.410 | -2.187 | -0.033 | regime[short=20,long=120,slope=240,vol_w=1440,vol_q=0.80] |
| 1 | cross_asset_overlay | 15.289 | -13.506 | -2.774 | -0.044 | regime[short=20,long=120,slope=240,vol_w=1440,vol_q=0.80] | overlay_mode=strength |
| 1 | dynamic_sizing | 19.664 | -26.851 | -1.042 | -0.014 | regime[short=20,long=120,slope=240,vol_w=1440,vol_q=0.80] | offset=0.5,scale=0.5 |
| 1 | exit_analysis | 17.364 | -16.266 | -1.689 | -0.023 | regime[short=20,long=120,slope=240,vol_w=1440,vol_q=0.80] | mode=trail,time_bars=None,trail_stop=0.008 |
| 1 | meta_model | 36.245 | 14.088 | 1.012 | -0.006 | regime[short=20,long=120,slope=240,vol_w=1440,vol_q=0.80] | prob_threshold=0.5 |
| 1 | quality_filter | 18.700 | -19.758 | -1.354 | -0.018 | regime[short=20,long=120,slope=240,vol_w=1440,vol_q=0.80] | require_ret30_positive=True,spread_z_threshold=0.75 |
| 1 | regime_decomposition | 15.692 | -12.098 | -2.471 | -0.045 | regime[short=20,long=120,slope=240,vol_w=1440,vol_q=0.80] | allowed_states=1,2,3,5,7,min_samples=50,state_table=   state      mean  count
0      0 -0.000057    136
1      1  0.000037   2060
2      2  0.000088    117
3      3  0.000049   1387
4      4 -0.000068    589
5      5  0.000014   2186
6      6 -0.000003   1378
7      7  0.000034   3228 |
| 2 | base_regime_tuned | 11.879 | 4.785 | 0.894 | -0.034 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] |
| 2 | cross_asset_overlay | 11.632 | -2.540 | -0.677 | -0.047 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | overlay_mode=strength |
| 2 | dynamic_sizing | 13.235 | 14.671 | 2.457 | -0.019 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | offset=0.25,scale=2.0 |
| 2 | exit_analysis | 14.903 | 7.384 | 1.332 | -0.031 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | mode=time,time_bars=180,trail_stop=None |
| 2 | meta_model | 42.956 | 38.143 | 4.167 | -0.004 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | prob_threshold=0.5 |
| 2 | quality_filter | 16.763 | 13.146 | 1.937 | -0.011 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | require_ret30_positive=True,spread_z_threshold=0.75 |
| 2 | regime_decomposition | 9.860 | -2.010 | -0.453 | -0.050 | regime[short=20,long=120,slope=120,vol_w=1440,vol_q=0.80] | allowed_states=2,3,7,min_samples=50,state_table=   state      mean  count
0      0 -0.000025    225
1      1 -0.000003   1888
2      2  0.000038     92
3      3  0.000007    712
4      4 -0.000037    196
5      5 -0.000035   3375
6      6 -0.000002    919
7      7  0.000025   2967 |
| 3 | base_regime_tuned | -3.408 | 9.485 | 0.835 | -0.012 | regime[short=30,long=120,slope=120,vol_w=1440,vol_q=0.60] |
| 3 | cross_asset_overlay | -5.427 | 5.842 | 0.951 | -0.026 | regime[short=30,long=120,slope=120,vol_w=1440,vol_q=0.60] | overlay_mode=top1 |
| 3 | dynamic_sizing | 5.423 | 15.460 | 0.763 | -0.004 | regime[short=30,long=120,slope=120,vol_w=1440,vol_q=0.60] | offset=0.5,scale=0.5 |
| 3 | exit_analysis | 0.660 | 16.449 | 1.062 | -0.005 | regime[short=30,long=120,slope=120,vol_w=1440,vol_q=0.60] | mode=time,time_bars=90,trail_stop=None |
| 3 | meta_model | 19.669 | 34.588 | 0.775 | -0.001 | regime[short=30,long=120,slope=120,vol_w=1440,vol_q=0.60] | prob_threshold=0.5 |
| 3 | quality_filter | 9.410 | 12.146 | 0.771 | -0.008 | regime[short=30,long=120,slope=120,vol_w=1440,vol_q=0.60] | require_ret30_positive=True,spread_z_threshold=0.5 |
| 3 | regime_decomposition | 0.621 | 8.296 | 0.644 | -0.011 | regime[short=30,long=120,slope=120,vol_w=1440,vol_q=0.60] | allowed_states=2,7,min_samples=50,state_table=   state      mean  count
0      0 -0.000100    109
1      1 -0.000061   1500
2      2  0.000011    261
3      3 -0.000030   1081
4      4 -0.000113     96
5      5 -0.000034   2150
6      6 -0.000093    437
7      7  0.000001   3620 |
| 4 | base_regime_tuned | 7.975 | -0.054 | -0.008 | -0.020 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] |
| 4 | cross_asset_overlay | 5.093 | -4.380 | -0.769 | -0.024 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | overlay_mode=active_equal |
| 4 | dynamic_sizing | 16.305 | -0.675 | -0.068 | -0.017 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | offset=0.5,scale=2.0 |
| 4 | exit_analysis | 10.559 | 5.504 | 0.533 | -0.008 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | mode=time,time_bars=90,trail_stop=None |
| 4 | meta_model | 41.067 | 29.618 | 1.997 | -0.003 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | prob_threshold=0.5 |
| 4 | quality_filter | 18.048 | -4.123 | -0.330 | -0.013 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | require_ret30_positive=True,spread_z_threshold=0.75 |
| 4 | regime_decomposition | 8.980 | 2.056 | 0.296 | -0.018 | regime[short=20,long=180,slope=120,vol_w=1440,vol_q=0.70] | allowed_states=7,min_samples=50,state_table=   state      mean  count
0      0 -0.000020    157
1      1 -0.000013   1067
2      2 -0.000141    375
3      3 -0.000043    801
4      4 -0.000220    106
5      5 -0.000033   2208
6      6 -0.000011    910
7      7  0.000031   3971 |

## Caveats
- Sample window is limited to available local files; conclusions are provisional.
- `cost_bps=0` matches competition optimization, but high-turnover variants still carry operational risk.
- Meta model uses a lightweight custom logistic implementation due no sklearn dependency in this repo.
- Overlay models can increase gross exposure vs baseline equal-average construction; compare with that in mind.

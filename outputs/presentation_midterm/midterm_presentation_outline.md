# Midterm Presentation Outline

## 0. Title / Goal (0:00-0:40)
- Project goal: predict future truck freight demand between U.S. regions using FAF5.
- Business context: identify high-growth OD demand and support warehouse / vehicle pre-positioning.
- Midterm focus: stabilize the official modeling problem before scenario expansion.

## 1. Data And Scope (0:40-1:40)
- Main data: FAF5 2018-2024.
- Mode filter: truck only (`dms_mode = 1`).
- Original task target: next-year freight tons.
- Initial external data candidates: fuel, macro, state GDP / CBP, population.

Suggested visuals:
- `01_yearly_total_tons.png`
- `outputs/preprocessing_truck_only/preprocessing_summary.md`

## 2. Preprocessing And EDA (1:40-3:00)
- Converted wide annual columns into long format (`year`, `tons`, `value`, `current_value`, `tmiles`).
- Cleaned truck-only data and checked missing / negative values.
- Used `dms_orig` / `dms_dest` as the main spatial unit because `fr_orig` / `fr_dest` are structurally sparse.
- Built route and map summaries for later spatial interpretation.

Suggested visuals:
- `01_yearly_total_tons.png`
- `outputs/preprocessing_truck_only/eda/top_routes.png`
- `outputs/maps/top_dms_corridors_2024.html` or a screenshot from it

## 3. Initial Feature Engineering And Baseline (3:00-4:10)
- First modeling approach used route-level lag features.
- Lag features produced strong baseline performance.
- However, this triggered a key issue: the model behaved like a persistence / restoration model rather than a scenario-sensitive model.

Suggested visuals:
- `02_lag_dominance_validation_rmse.png`
- `outputs/models/baseline_suite/feature_group_ablation_validation_rmse.png`

## 4. Why We Changed The Modeling Frame (4:10-5:15)
- Professor feedback: excessive lag dominance is abnormal and should be reconsidered.
- We tested lag-only, no-lag, and delta-based setups.
- Conclusion: route-level lag-heavy modeling is accurate but not appropriate for scenario-oriented research.

Suggested visuals:
- `02_lag_dominance_validation_rmse.png`
- `outputs/models/lag_role_experiments/lag_role_summary.md`

## 5. Official Lag-Free Scenario Model (5:15-6:20)
- Redefined the official unit:
  - `state-state x commodity x dist_band x trade_type x year`
- Kept target as `tons`.
- Removed same-route lag entirely.
- Re-ran `Linear / RF / XGB / LGBM`.

Key message:
- Linear collapsed without lag.
- Tree models became meaningful.

Suggested visuals:
- `03_lag_free_model_comparison.png`
- `outputs/scenario_model/scenario_model_summary.md`

## 6. External Data Selection Journey (6:20-7:30)
- Tested fuel and nationwide macro indicators first.
- They added limited value.
- State-level economy and CBP variables were more useful.
- Population / GDP per capita had mixed value.
- The strongest additional teammate features were:
  - state highway accident count
  - real manufacturing GDP

Suggested visuals:
- `04_population_feature_rmse_delta.png`
- `06_joeun_candidate_avg_rmse.png`

## 7. Compact Feature Set Search (7:30-8:40)
- Reduced the feature space instead of adding more variables.
- Compact sets outperformed the larger full set.
- Current strongest set is based on:
  - internal route structure
  - economy gap
  - population / per-capita GDP gap
  - manufacturing structure
  - highway accident variables

Suggested visuals:
- `05_feature_set_avg_rmse.png`
- `06_joeun_candidate_avg_rmse.png`

## 8. Current Midterm Conclusion (8:40-9:30)
- Current strongest stable candidate:
  - `Random Forest + plus_both`
- Current strongest challenger:
  - `XGBoost + plus_accidents`
- Interpretation:
  - RF is more stable across splits.
  - XGB is sharper on validation / 2023.
- Final model choice is intentionally left open until tuning.

Suggested visuals:
- `07_final_candidate_comparison.png`
- `presentation_key_metrics.csv`

## 9. Next Steps (9:30-10:00)
- Tune XGBoost and possibly Random Forest.
- Finalize model selection.
- Then move to expanded scenarios:
  - hub addition / removal
  - threshold detection
  - resilience / disruption analysis

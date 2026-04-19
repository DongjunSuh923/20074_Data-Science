# Session Handoff - 2026-04-18

## Purpose
- This file summarizes the current project state so the work can be resumed from another PC without relying on this chat thread.

## Project Goal
- Predict future U.S. truck freight demand using FAF5.
- Midterm focus: stabilize the official prediction model.
- Later extension: hub addition/removal, threshold detection, resilience/disruption scenarios.

## Current Official Problem Definition
- Mode: `dms_mode = 1` only (`Truck`).
- Official unit: `state-state x commodity x dist_band x trade_type x year`.
- Target: `tons`.
- Same-route lag: excluded from the **official** model.

## Why The Problem Was Redefined
- Initial route-level lag-based modeling gave strong numerical results.
- But `tons_lag1` / `tons_lag2` dominated too strongly.
- `Persistence`, `Lag Only`, and `Full Linear` performed too similarly.
- This implied the model was acting more like a restoration model than a structure-based predictive model.
- Professor feedback also indicated that such lag dominance was undesirable.

## Key Modeling Transition
- Old idea:
  - Route-level next-year tons prediction with same-route lag features.
- New official idea:
  - Lag-free structure model using broader state-state aggregation.

## Current Main Model Candidates
- Stable candidate:
  - `Random Forest + plus_both`
- Challenger candidate:
  - `XGBoost + plus_accidents`
- Interpretation:
  - RF is currently more stable across splits.
  - XGB is stronger on validation and `test_2023`, and may improve further after tuning.

## Current Recommended Final Feature Set Candidate
- `plus_both`

### Internal features
- `orig_state_fips`
- `dest_state_fips`
- `sctg2`
- `dist_band`
- `trade_type`

### External features kept in the current strongest set
- `route_gdp_total_gap`
- `route_gdp_per_capita_gap`
- `route_population_gap`
- `orig_gdp_per_capita`
- `dest_gdp_per_capita`
- `orig_population`
- `orig_cbp_mfg_emp`
- `dest_cbp_mfg_emp`
- `orig_gdp_wholesale_share`
- `dest_gdp_wholesale_share`
- `orig_real_mfg_gdp`
- `dest_real_mfg_gdp`
- `route_real_mfg_gdp_gap`
- `route_real_mfg_gdp_sum`
- `orig_highway_accidents`
- `dest_highway_accidents`
- `route_highway_accident_gap`
- `route_highway_accident_sum`

## External Data Decisions

### Considered but not kept as core
- `dms_mode = 4` (`Air include truck-air`)
  - not used in the main model
  - kept only as a possible sensitivity-analysis idea
- Nationwide macro variables
  - limited usefulness for route/state differentiation
- PADD fuel price features
  - weak improvement
- DFF effective federal funds rate
  - currently low priority, nationwide and indirect

### Considered and kept
- State-level GDP / GDP gap
- CBP manufacturing / wholesale / warehousing structure
- Population and GDP per capita gap
- State real manufacturing GDP
- State highway accident count

## Key Findings To Remember
- `fr_orig` / `fr_dest` were structurally sparse because they are foreign-region columns.
- For domestic truck network analysis, `dms_orig` / `dms_dest` and later `state-state` are the meaningful geography.
- `tons = 0` rows were not deleted.
  - They were retained.
  - `log1p` was used where needed.
  - ratio-like features handled zero safely to avoid division issues.

## Important Presentation Assets
Folder:
- `C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024\outputs\presentation_midterm`

Most useful files:
- `01_yearly_total_tons.png`
- `02_lag_dominance_validation_rmse.png`
- `03_lag_free_model_comparison.png`
- `04_population_feature_rmse_delta.png`
- `05_feature_set_avg_rmse.png`
- `06_joeun_candidate_avg_rmse.png`
- `07_final_candidate_comparison.png`
- `08_lag_feature_example_route.png`
- `presentation_key_metrics.csv`
- `midterm_presentation_outline.md`

## Useful Result Files
- Lag dominance:
  - `outputs/models/lag_role_experiments/lag_role_results.csv`
- Official lag-free model:
  - `outputs/scenario_model/scenario_model_results.csv`
- Feature importance:
  - `outputs/scenario_model/feature_importance/scenario_group_importance.csv`
- Compact feature-set comparison:
  - `outputs/scenario_model/feature_set_selection/feature_set_results.csv`
- Teammate external data comparison:
  - `outputs/scenario_model/joeun_feature_eval/joeun_candidate_results.csv`

## Repository Status
- Selected scripts and presentation assets were already pushed to GitHub `main`.
- Remote repo:
  - `https://github.com/DongjunSuh923/20074_Data-Science.git`

## Recommended Midterm Presentation Message
- The team first tried a natural route-level time-series approach.
- That approach suffered from lag dominance and restoration-model behavior.
- The official model was redefined into a lag-free, state-state structure model.
- Under that new definition, tree models became meaningful while linear regression weakened sharply.
- Current conclusion:
  - Random Forest is the safest current candidate.
  - XGBoost remains a strong challenger pending tuning.

## Immediate Next Step After Midterm
- Tune `XGBoost` and, if needed, `Random Forest`.
- Then finalize the official model.
- After that, move to scenario extensions such as hub disruption and resilience analysis.

# Domestic Issue Progress Reference

This working note keeps the two domestic-issue tracks separate from natural-disaster scenarios.

## 1. Construction / Infrastructure hybrid

Source outputs:
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_issue_hybrid\construction_infra_hybrid_summary.csv`
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_issue_hybrid\construction_infra_hybrid_next15_excluding_must_have.csv`
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_issue_hybrid\DOMESTIC_ISSUE_HYBRID_REFERENCE.md`

Interpretation:
- This track represents destination-side domestic demand weakening in construction/infrastructure-linked commodities.
- It combines `state blocking` with a `destination-side demand overlay`.
- same-state short-haul remains excluded.

Key result:
- cumulative network loss: `2.33%`
- peak network loss: `1.27%` in `2026`
- cumulative construction subset loss: `8.62%`
- peak construction subset loss: `4.71%` in `2026`

## 2. Domestic bottleneck / congestion shock

Source outputs:
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_bottleneck_shock\domestic_bottleneck_state_scores.csv`
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_bottleneck_shock\domestic_bottleneck_state_blocking_cases.csv`
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_bottleneck_shock\domestic_bottleneck_summary.csv`
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_bottleneck_shock\domestic_bottleneck_next15_excluding_must_have.csv`
- `C:\Users\서동준\Downloads\FAF5.7.1_2018-2024\outputs\scenario_model\domestic_bottleneck_shock\DOMESTIC_BOTTLENECK_SHOCK_REFERENCE.md`

Input logic:
- official FHWA TTTR by State from `Freight Mobility Trends Report 2019`
- official top Interstate freight bottlenecks by State from Appendix B
- state score combines:
  - `tttr_2019`
  - `tttr_change_pct_2017_2019`
  - `bottleneck_count`
  - `bottleneck_dpm`
  - `bottleneck_congestion_cost`

Selected blocking states:
- `NY` severe
- `TX` medium
- `CA` medium
- `VA` mild
- `CO` mild

Key result:
- cumulative network loss: `2.80%`
- peak network loss: `1.56%` in `2026`
- cumulative interstate loss: `13.41%`
- peak interstate loss: `7.44%` in `2026`

Interpretation:
- This track is a broad interstate reliability/congestion stress, not a commodity-specific domestic demand shock.
- Compared with construction/infrastructure hybrid, it hurts interstate structure more directly and more strongly.

## 3. Practical takeaway

- Keep `construction/infrastructure` and `bottleneck/congestion` as separate domestic-issue families.
- Use construction hybrid when the question is about domestic demand weakness in material-heavy sectors.
- Use bottleneck shock when the question is about national freight reliability and corridor congestion stress.
- Do not merge either of these with natural-disaster scenarios in the first analysis layer.

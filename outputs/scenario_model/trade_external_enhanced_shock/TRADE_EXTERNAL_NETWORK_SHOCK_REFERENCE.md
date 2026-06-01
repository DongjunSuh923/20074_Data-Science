# Trade / External Network Shock Reference

## Status

This is the **official reference document** for the `trade / external network shock` family.

It supersedes the earlier first-pass trade shock documents that were built from export exposure alone.

## Why this version is official

The enhanced version materially improves the trade shock definition by adding:
- Census `state imports` (`ISTNAICS`, 2018-2025)
- combined total trade value (`exports + imports`)
- import intensity
- combined vessel and containerized trade exposure
- partner-country dependence for:
  - `China`
  - `Canada`
  - `Mexico`
- trade-war overlay based on:
  - `USTR Section 301` China actions
  - `USTR Section 232` Canada/Mexico steel-aluminum episode

Because of those additions, this version should be treated as the default trade/external scenario for later cross-family comparison.

## External data actually used

- Census `STNAICS` monthly state exports (`2018-2025`)
- Census `ISTNAICS` monthly state imports (`2018-2025`)
- USTR China Section 301 tariff actions (`2018-2020`)
- USTR Canada/Mexico Section 232 steel and aluminum episode (`2018-2019`)

Supporting file list:
- [enhanced_trade_external_data_list.csv](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_trade_external_data_list.csv)

## Final feature logic

The enhanced trade score is built from:
- `total export value`
- `total import value`
- `total trade value`
- `import intensity`
- `combined vessel trade share`
- `combined containerized trade share`
- `China / Canada / Mexico trade-war partner exposure`
- `trade dependency` vs baseline `2031 interstate touch`
- `import dependency` vs baseline `2031 interstate touch`

## Selected shock states

- `CA` severe
- `GA` medium
- `NJ` medium
- `MI` mild
- `IL` mild

State selection details:
- [trade_external_enhanced_state_scores.csv](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_enhanced_shock/trade_external_enhanced_state_scores.csv)
- [trade_external_enhanced_state_blocking_cases.csv](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_enhanced_shock/trade_external_enhanced_state_blocking_cases.csv)
- [trade_external_enhanced_state_year_metrics.csv](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_enhanced_shock/trade_external_enhanced_state_year_metrics.csv)

## Simulation result

- cumulative network loss: `0.32%`
- peak network loss: `1.32%` in `2026`
- cumulative interstate loss: `1.54%`
- peak interstate loss: `6.28%` in `2026`

Simulation result files:
- [trade_external_enhanced_summary.csv](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_enhanced_shock/simulation_results/trade_external_enhanced_summary.csv)
- [trade_external_enhanced_summary.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_enhanced_shock/simulation_results/trade_external_enhanced_summary.md)
- [trade_external_enhanced_next15_excluding_must_have.csv](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_enhanced_shock/simulation_results/trade_external_enhanced_next15_excluding_must_have.csv)

## Interpretation of what this shock means

This is **not** just an export shock.

It should be interpreted as a combined:
- import vulnerability shock
- container gateway shock
- partner-country manufacturing trade shock
- interstate redistribution stress

That is why the enhanced result shifts away from the original
`LA / WA / TX / CA / NC`
selection pattern and toward
`CA / GA / NJ / MI / IL`.

In practical terms:
- `CA` remains a massive Pacific gateway and inbound container state
- `GA` rises as a Southeast container-distribution state
- `NJ` rises as a Northeast port + warehousing + consumer-market state
- `MI` rises because Canada/Mexico-linked manufacturing trade is now visible
- `IL` rises because inland intermodal redistribution is now visible

## NGL comparison

The key comparison summary is:
- [enhanced_scenario_ngl_overlap_summary.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_scenario_overlap_maps/enhanced_scenario_ngl_overlap_summary.md)

For the enhanced trade next-15 comparison:
- overlap: `AL, CA, CO, FL, IL, IN, KS, MI, MN, NE, NV, NY, OH, OR, PA, TN, TX, WA, WI`
- scenario-only: `GA, NC, NJ, SC, UT, VA`
- NGL-only: `AZ, CT, IA, KY, LA, MA, MO, MS, OK, RI, SD`

Important caveat:
- this is a `next-15 excluding must-have` comparison
- so some states that appear on the `NGL-only` side, such as `CA`, `IL`, and `MI`, can still be highly important in the shock
- they appear there only because they are already part of the core or selected-shock structure, not because the scenario says they are unimportant

## Practical strategic takeaway

This enhanced trade scenario most strongly supports three interpretation buckets:

1. `Gateway reinforcement`
- `NJ`
- `GA`

2. `East Coast inland connector reinforcement`
- `NC`
- `VA`

3. `Secondary inland redistribution fallback`
- `UT`
- plus part of the interior support group depending on later cross-family comparison

## Related visuals and interpretation

Maps:
- [01_enhanced_trade_ngl_overlap.png](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_scenario_overlap_maps/01_enhanced_trade_ngl_overlap.png)
- [01_enhanced_trade_ngl_overlap.html](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_scenario_overlap_maps/01_enhanced_trade_ngl_overlap.html)
- [04_enhanced_scenario_ngl_static_comparison.png](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_scenario_overlap_maps/04_enhanced_scenario_ngl_static_comparison.png)

Working interpretation:
- [STRATEGIC_INSIGHTS_ENHANCED_TRADE_AND_DOMESTIC_DEMAND.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/STRATEGIC_INSIGHTS_ENHANCED_TRADE_AND_DOMESTIC_DEMAND.md)

## Official-use note

From this point onward, any reference to the project’s `trade / external network shock`
should use this enhanced version unless a later document explicitly supersedes it.

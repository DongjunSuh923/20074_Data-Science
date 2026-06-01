from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
OUT_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "TIER1_SCENARIO_PROGRESS_REFERENCE.md"
TRADE_REF = PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "TRADE_EXTERNAL_NETWORK_SHOCK_REFERENCE.md"


def load_summary(path: Path) -> pd.Series:
    return pd.read_csv(path).iloc[0]


def main() -> None:
    natural = load_summary(PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "simulation_results" / "natural_disaster_simulation_summary.csv")
    grain = load_summary(PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "simulation_results" / "grain_supply_simulation_summary.csv")
    construction = load_summary(PROJECT_ROOT / "outputs" / "scenario_model" / "construction_infra_shock" / "simulation_results" / "construction_infra_simulation_summary.csv")
    trade = load_summary(PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "simulation_results" / "trade_external_simulation_summary.csv")

    lines = [
        "# Tier 1 Scenario Progress Reference",
        "",
        "## Completed Scenario Families",
        "### 1. Natural Disaster / Capacity Shock",
        "Reference:",
        "- `natural_disaster_capacity/NATURAL_DISASTER_CAPACITY_SHOCK_REFERENCE.md`",
        "",
        "Completed scenarios:",
        "- `snow_heat_state_blocking`",
        "- `tornado_state_blocking`",
        "- `wildfire_smoke_state_blocking`",
        "",
        "Result summary:",
        "- `snow_heat_state_blocking`: cumulative network loss `1.85%`, peak annual loss `7.12%` in 2026",
        "- `tornado_state_blocking`: cumulative network loss `1.64%`, peak annual loss `6.32%` in 2026",
        "- `wildfire_smoke_state_blocking`: cumulative network loss `1.46%`, peak annual loss `5.78%` in 2026",
        "",
        "### 2. Agricultural / Grain Supply Shock",
        "Reference files:",
        "- `grain_supply_shock/grain_supply_state_blocking_summary.md`",
        "- `grain_supply_shock/simulation_results/grain_supply_simulation_summary.md`",
        "",
        "Blocked states:",
        "- `IA` severe",
        "- `TX` medium",
        "- `NE` medium",
        "- `IL` mild",
        "- `MN` mild",
        "",
        "Result summary:",
        f"- cumulative network loss `{grain['cumulative_network_loss_pct_of_annual_sum']:.2%}`",
        f"- peak network loss `{grain['peak_year_loss_pct']:.2%}` in {int(grain['peak_year'])}",
        f"- cereal-only cumulative loss `{grain['cereal_cumulative_loss_pct_of_annual_sum']:.2%}`",
        f"- cereal-only peak loss `{grain['cereal_peak_year_loss_pct']:.2%}`",
        "",
        "Interpretation:",
        "- this is a narrow commodity shock rather than a broad network shock",
        "- statewide network damage is small, but cereal flow damage is material",
        "",
        "### 3. Construction / Infrastructure Shock",
        "Reference files:",
        "- `construction_infra_shock/construction_infra_state_blocking_summary.md`",
        "- `construction_infra_shock/simulation_results/construction_infra_simulation_summary.md`",
        "",
        "Blocked states:",
        "- `TX` severe",
        "- `FL` medium",
        "- `CA` medium",
        "- `PA` mild",
        "- `IL` mild",
        "",
        "Result summary:",
        f"- cumulative network loss `{construction['cumulative_network_loss_pct_of_annual_sum']:.2%}`",
        f"- peak network loss `{construction['peak_year_loss_pct']:.2%}` in {int(construction['peak_year'])}",
        f"- construction-material subset cumulative loss `{construction['subset_cumulative_loss_pct_of_annual_sum']:.2%}`",
        f"- construction-material subset peak loss `{construction['subset_peak_year_loss_pct']:.2%}`",
        "",
        "Interpretation:",
        "- stronger than grain at whole-network level",
        "- still more commodity-focused than nationwide network-wide",
        "- directly relevant to `Gravel / Nonmetal / Natural sands` stress testing",
        "",
        "### 4. Trade / External Network Shock",
        "Reference files:",
        "- `trade_external_shock/trade_external_state_blocking_summary.md`",
        "- `trade_external_shock/simulation_results/trade_external_simulation_summary.md`",
        "- `trade_external_shock/TRADE_EXTERNAL_NETWORK_SHOCK_REFERENCE.md`",
        "",
        "Blocked states:",
        "- `LA` severe",
        "- `WA` medium",
        "- `TX` medium",
        "- `CA` mild",
        "- `NC` mild",
        "",
        "Result summary:",
        f"- cumulative network loss `{trade['cumulative_network_loss_pct_of_annual_sum']:.2%}`",
        f"- peak network loss `{trade['peak_year_loss_pct']:.2%}` in {int(trade['peak_year'])}",
        f"- interstate-only cumulative loss `{trade['interstate_cumulative_loss_pct_of_annual_sum']:.2%}`",
        f"- interstate-only peak loss `{trade['interstate_peak_year_loss_pct']:.2%}`",
        "",
        "Interpretation:",
        "- first-pass trade shock is broader than grain but weaker than natural-disaster capacity shocks",
        "- exposure is driven by export scale, China share, and vessel/container dependence",
        "- same-state short-haul is excluded on purpose, so the trade result is best read as interstate network stress",
        "",
        "## Current Takeaway",
        "Across Tier 1 official scenario groups so far:",
        "- the largest broad statewide capacity shock still comes from `natural disaster` scenarios",
        "- `grain supply shock` remains the most commodity-concentrated stressor",
        "- `construction / infrastructure shock` is a material sector stress for construction-material flows",
        "- `trade / external network shock` now adds an interstate-focused external disruption layer centered on export-heavy and vessel-dependent states",
        "",
        "## Recommended Next Step",
        "Use these four completed Tier 1 scenario families as the official first simulation layer, then move to hub ranking under state blocking with must-have hubs fixed and NGL kept as a benchmark only.",
    ]
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    trade_lines = [
        "# Trade / External Network Shock Reference",
        "",
        "## Scope",
        "- Tier 1 official scenario family",
        "- state-level only",
        "- same-state short-haul excluded from shock application",
        "- first-pass implementation uses state blocking rather than feature perturbation",
        "",
        "## Source Data",
        "- Official U.S. Census `STNAICS` monthly state export ZIP files for `2018-2025`",
        "- Aggregated annual state metrics built from raw country-by-state monthly export records",
        "",
        "## First-pass Exposure Metrics",
        "- total export value by state",
        "- China export share",
        "- vessel export share",
        "- containerized vessel export share",
        "- export dependency relative to 2031 interstate touch volume",
        "",
        "## Selected Blocking States",
        "- `LA` severe",
        "- `WA` medium",
        "- `TX` medium",
        "- `CA` mild",
        "- `NC` mild",
        "",
        "## Simulation Result",
        f"- cumulative network loss `{trade['cumulative_network_loss_pct_of_annual_sum']:.2%}`",
        f"- peak network loss `{trade['peak_year_loss_pct']:.2%}` in {int(trade['peak_year'])}",
        f"- interstate-only cumulative loss `{trade['interstate_cumulative_loss_pct_of_annual_sum']:.2%}`",
        f"- interstate-only peak loss `{trade['interstate_peak_year_loss_pct']:.2%}`",
        "",
        "## Interpretation",
        "- This first pass behaves like an interstate trade disruption rather than a same-state demand shock.",
        "- `LA`, `WA`, `TX`, and `CA` emerge because large export scale overlaps with heavy vessel/container dependence.",
        "- `NC` appears as a lighter exposure case because its container share and China share are high enough to matter even with smaller total exports.",
        "",
        "## Limitations",
        "- Official state import panels are not yet included.",
        "- Port congestion is proxied through vessel/container exposure rather than separate delay data.",
        "- Non-contiguous states are excluded from final scoring for trucking-network relevance.",
    ]
    TRADE_REF.write_text("\n".join(trade_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

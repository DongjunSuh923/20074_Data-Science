# Scenario State Selection Rationale

This document fixes the state-selection logic for the official Tier 1 scenario families so that later simulations use consistent and explainable blocking targets.

## Selection Principle
- Shocked states should differ by scenario family.
- A state is selected because it has high exposure to that scenario's transmission mechanism, not simply because it is a large freight state.
- The same state may appear in multiple scenario families, but the reason must differ across families.

## Official Mapping

### 1. Capacity Shock
These scenarios are selected from hazard exposure, not from commodity demand or trade structure.

#### `snow_heat_state_blocking`
- `TX` severe
- `CA` medium
- `SD` medium
- `MN` mild
- `IA` mild

Why:
- Selected from the first-pass snow/heat composite hazard score.
- Interpreted as weather-driven route disruption or temporary capacity loss.

#### `tornado_state_blocking`
- `TX` severe
- `MS` medium
- `OK` medium
- `IL` mild
- `IA` mild

Why:
- Selected from tornado event and damage exposure.
- Interpreted as state-level transport disruption caused by severe storm events.

#### `wildfire_smoke_state_blocking`
- `CA` severe
- `TX` medium
- `CO` medium
- `OK` mild
- `WA` mild

Why:
- Selected from wildfire / smoke hazard exposure.
- Interpreted as burn-area and smoke-related blocking pressure on statewide transport capacity.

### 2. Commodity-specific Supply Shock
These scenarios are selected from commodity exposure, not from general statewide hazard.

#### `grain_supply_state_blocking`
- `IA` severe
- `TX` medium
- `NE` medium
- `IL` mild
- `MN` mild

Why:
- Selected from grain production scale combined with drought-related stress.
- Interpreted as a cereal supply shock rather than a broad network shock.

### 3. Demand Shock
These scenarios are selected from sector demand exposure.

#### `construction_infra_state_blocking`
- `TX` severe
- `FL` medium
- `CA` medium
- `PA` mild
- `IL` mild

Why:
- Selected from highway spending and construction-material demand indicators.
- Interpreted as an infrastructure / construction downturn or project disruption shock.

### 4. Trade / External Network Shock
These scenarios are selected from trade exposure rather than domestic demand.

#### `trade_external_state_blocking`
- `LA` severe
- `WA` medium
- `TX` medium
- `CA` mild
- `NC` mild

Why:
- Selected from official Census state export data.
- First-pass score uses:
  - total export value
  - China export share
  - vessel export share
  - container share
  - export dependency relative to 2031 interstate touch
- Interpreted as an interstate trade disruption, not a same-state demand shock.

## Interpretation Rule
- `Capacity shock` states are chosen because hazards can block movement directly.
- `Commodity-specific supply shock` states are chosen because production-side exposure is high.
- `Demand shock` states are chosen because sectoral demand concentration is high.
- `Trade/external network shock` states are chosen because trade, port, or export dependency is high.

## Important Note
- `same-state short-haul` remains excluded from official scenario simulation.
- This means all selected states should be interpreted as state-level blocking targets for interstate or broad network stress, not for sub-state routing analysis.

## Machine-readable Reference
- [scenario_state_selection_rationale.csv](/C:/Users/서동준/IdeaProjects/FAF5.7.1_2018-2024/outputs/scenario_model/scenario_state_selection_rationale.csv)

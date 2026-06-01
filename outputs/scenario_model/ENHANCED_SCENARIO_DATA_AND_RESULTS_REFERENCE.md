# Enhanced Scenario Data And Results Reference

This note records the first `Tier 2` enhancement pass for
- broader domestic demand shock
- enhanced trade / external network shock

It is intentionally non-final and can be extended later.

## 1. Broader Domestic Demand

### External data actually used
- BEA `SASUMMARY` state annual summary files (`1998–2025`)
- Variables used in the final score:
  - `Real GDP`
  - `Real PCE`
  - `Total employment`
  - `2021–2024 CAGR` of each series

### Scenario construction
- Shock family: `Demand shock`
- Mechanism: destination-side reduction on interstate inbound flows
- same-state short-haul remains excluded

### Selected shock states
- `FL` severe
- `TX` medium
- `CA` medium
- `NY` mild
- `WA` mild

### Simulation result
- cumulative network loss: `0.15%`
- peak network loss: `0.63%` in `2026`
- cumulative interstate loss: `0.74%`
- peak interstate loss: `3.00%` in `2026`

### Non-must-have next 15
- `WA, NE, LA, AL, OR, MN, WI, KS, CO, OK, SD, VA, NV, NM, NC`

### Interpretation
- This is weaker than the construction/infrastructure hybrid and the domestic bottleneck shock.
- It behaves like a broad consumer-demand slowdown rather than a route-capacity disruption.
- The result still leaves a familiar western / plains support set (`WA, NE, LA, MN, KS, CO`) plus eastern support (`VA, NC`).

## 2. Enhanced Trade / External Network

### External data actually used
- Census `STNAICS` monthly state exports (`2018–2025`) from the existing trade track
- Census `ISTNAICS` monthly state imports (`2018–2025`) newly downloaded for this enhancement
- USTR China Section 301 tariff actions (`2018–2020`) for trade-war partner weighting
- USTR Section 232 Canada/Mexico steel-aluminum episode (`2018–2019`) for additional trade-war partner weighting

### Trade features added relative to the first-pass version
- `total import value`
- `combined total trade value` (`exports + imports`)
- `import intensity`
- `China / Canada / Mexico combined trade-war partner exposure`
- `combined vessel trade share`
- `combined containerized trade share`
- `trade dependency` and `import dependency` vs baseline interstate touch

### Selected shock states
- `CA` severe
- `GA` medium
- `NJ` medium
- `MI` mild
- `IL` mild

### Simulation result
- cumulative network loss: `0.32%`
- peak network loss: `1.32%` in `2026`
- cumulative interstate loss: `1.54%`
- peak interstate loss: `6.28%` in `2026`

### Non-must-have next 15
- `NJ, WA, NE, AL, OR, SC, GA, MN, WI, NV, NC, KS, VA, UT, CO`

### What changed vs the first-pass trade shock
- First-pass trade shock was dominated by export / vessel exposure and selected:
  - `LA, WA, TX, CA, NC`
- Enhanced trade shock is more import- and East/Midwest-oriented and selected:
  - `CA, GA, NJ, MI, IL`

### Interpretation
- Adding imports and non-China trade-war partner exposure moves the scenario away from a pure port-export stress test.
- The enhanced version behaves more like a combined import / manufacturing / container vulnerability test.
- This is why `GA`, `NJ`, `MI`, and `IL` rise while `LA`, `TX`, and `NC` become less dominant at the shock-selection stage.

## 3. Practical takeaway

- `Domestic broad demand` is now better represented, but it is still a milder stress family than construction/material or bottleneck shocks.
- `Trade / external` is materially better than the first pass because it no longer depends on exports alone.
- The enhanced trade track should be treated as the new default trade/external version for later cross-family comparison.

# One-Page Context

## Current Project State
- Goal: predict future U.S. truck freight demand and later extend to hub / resilience scenarios.
- Main data: FAF5 2018-2024.
- Main mode kept: `dms_mode = 1` only (`Truck`).

## Important Modeling Decision
- The initial route-level + lag setup produced strong scores but behaved like a restoration model.
- Same-route past tons dominated too strongly.
- After professor feedback, the official problem was redefined as:
  - `state-state x commodity x dist_band x trade_type x year`
  - target stays `tons`
  - same-route lag removed from the official model

## Current Official Model Direction
- Tree-based lag-free model is now the main track.
- Linear regression is no longer competitive under the lag-free setup.

## Current Best Candidates
- Stable candidate: `Random Forest + plus_both`
- Challenger candidate: `XGBoost + plus_accidents`

## Current Best External Features
- GDP / GDP per capita gap
- manufacturing employment and wholesale structure
- population gap
- real manufacturing GDP
- state highway accident count

## Presentation Message
- We started with a natural time-series route model.
- We discovered lag dominance.
- We redefined the problem to make it scenario-friendly.
- We now have a lag-free structure model where RF is most stable and XGB remains a strong challenger.

## Most Important Files
- `outputs/presentation_midterm/SESSION_HANDOFF_2026-04-18.md`
- `outputs/presentation_midterm/midterm_presentation_outline.md`
- `outputs/presentation_midterm/presentation_key_metrics.csv`
- `outputs/presentation_midterm/07_final_candidate_comparison.png`

## Best Prompt To Resume Later
- “`outputs/presentation_midterm/SESSION_HANDOFF_2026-04-18.md`와 `ONE_PAGE_CONTEXT.md` 기준으로 이어서 보자.”

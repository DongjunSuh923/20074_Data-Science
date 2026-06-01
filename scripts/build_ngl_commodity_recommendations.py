from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
FINAL_DIR = WORK_ROOT / "outputs" / "scenario_model" / "final_reinforcement_candidates"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "ngl_commodity_recommendations"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    states = pd.read_csv(FINAL_DIR / "final_reinforcement_candidate_state_roles.csv")

    rows = [
        {
            "commodity_theme": "Cereal grains and agricultural support",
            "priority_group": "Primary",
            "recommended_states": "NE, MN, KS",
            "supporting_states": "ND, SD",
            "reason": "These states repeatedly survive across families and show strong cereal-grain / agricultural mix in 2031 baseline touch.",
            "candidate_role": "surviving grain core and plains balancing support",
        },
        {
            "commodity_theme": "Construction materials and aggregates",
            "priority_group": "Primary",
            "recommended_states": "NC, VA, CO, WI",
            "supporting_states": "AL, SC, NM, NV, UT",
            "reason": "These states combine repeated cross-family survival with high Gravel / Nonmetal mineral product shares, making them strong material-balancing states.",
            "candidate_role": "East Coast corridor reinforcement plus inland construction/material balancing",
        },
        {
            "commodity_theme": "Logs and wood-linked flows",
            "priority_group": "Primary",
            "recommended_states": "WA, OR, AL, NC",
            "supporting_states": "SC",
            "reason": "These states show recurring Logs presence in baseline commodity mix and survive capacity, domestic, or trade shocks as fallback/reinforcement candidates.",
            "candidate_role": "Pacific gateway support plus southeastern wood/material reinforcement",
        },
        {
            "commodity_theme": "Trade redistribution and gateway support",
            "priority_group": "Primary",
            "recommended_states": "WA, LA",
            "supporting_states": "NC, VA",
            "reason": "WA and LA remain the strongest final-candidate gateway support states; NC and VA repeatedly reinforce the East Coast inland connector layer.",
            "candidate_role": "gateway redistribution and East Coast corridor reinforcement",
        },
        {
            "commodity_theme": "Fuel and energy-related balancing",
            "priority_group": "Secondary",
            "recommended_states": "KS, LA",
            "supporting_states": "UT, NJ",
            "reason": "These states are not pure energy cores, but they retain notable fuel/energy-linked commodity presence while surviving multiple families or key trade scenarios.",
            "candidate_role": "secondary fuel balancing and corridor backup",
        },
        {
            "commodity_theme": "Shock-specific trade gateway reinforcement",
            "priority_group": "Adjunct",
            "recommended_states": "NJ, GA",
            "supporting_states": "SC, UT",
            "reason": "These states are not in the final cross-family core set, but the enhanced trade shock specifically elevates them as import/container/partner-exposure reinforcement candidates.",
            "candidate_role": "trade/external-specific reinforcement only",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ngl_commodity_specific_recommendations.csv", index=False)

    md_lines = [
        "# NGL Commodity-Specific Recommendation Draft",
        "",
        "This is a working proposal based on the current final reinforcement candidate set and the cross-family scenario comparison.",
        "",
        "## 1. Core logic",
        "",
        "- `NC` and `VA` are the clearest non-NGL core expansion candidates.",
        "- `WA, MN, NE, KS, CO` are the strongest internal reinforcement states already inside the current footprint.",
        "- `OR, WI, AL, LA` form the second-line internal reinforcement set.",
        "- `NJ` and `GA` matter mainly as trade-specific adjunct candidates rather than broad cross-family core candidates.",
        "",
        "## 2. Commodity-specific proposal",
    ]
    for row in rows:
        md_lines.extend(
            [
                "",
                f"### {row['commodity_theme']}",
                f"- priority: {row['priority_group']}",
                f"- recommended states: {row['recommended_states']}",
                f"- supporting states: {row['supporting_states']}",
                f"- strategic role: {row['candidate_role']}",
                f"- rationale: {row['reason']}",
            ]
        )
    md_lines.extend(
        [
            "",
            "## 3. Practical interpretation for NGL",
            "",
            "- For network-wide expansion, the strongest new-state recommendation remains `NC` and `VA`.",
            "- For grain-linked portfolio reinforcement, the center of gravity should be `NE`, `MN`, and `KS`.",
            "- For material-heavy resilience, `CO`, `WI`, and the East Coast pair `NC/VA` are the strongest support states.",
            "- For log/wood-related resilience, `WA`, `OR`, `AL`, and `NC` are the most defensible candidates.",
            "- For trade-specific supplementation, `NJ` and `GA` should be treated as targeted gateway add-ons rather than universal expansion priorities.",
        ]
    )
    (OUT_DIR / "NGL_COMMODITY_SPECIFIC_RECOMMENDATIONS.md").write_text("\n".join(md_lines), encoding="utf-8")

    metadata = {
        "csv": str(OUT_DIR / "ngl_commodity_specific_recommendations.csv"),
        "md": str(OUT_DIR / "NGL_COMMODITY_SPECIFIC_RECOMMENDATIONS.md"),
    }
    (OUT_DIR / "ngl_commodity_specific_recommendations_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

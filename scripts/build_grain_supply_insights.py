from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

GRAIN_CASES = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "grain_supply_state_blocking_cases.csv"
GRAIN_SCORES = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "grain_supply_state_scores.csv"
GRAIN_SUMMARY = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "simulation_results" / "grain_supply_simulation_summary.csv"
GRAIN_NEXT15 = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next15_hubs" / "scenario_next15_excluding_must_have.csv"
GRAIN_OVERLAP_SUMMARY = WORK_ROOT / "outputs" / "scenario_model" / "grain_supply_overlap_maps" / "grain_supply_ngl_overlap_summary.md"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "grain_supply_visuals"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(GRAIN_CASES)
    scores = pd.read_csv(GRAIN_SCORES).sort_values("grain_supply_score", ascending=False)
    summary = pd.read_csv(GRAIN_SUMMARY).iloc[0]
    next15 = pd.read_csv(GRAIN_NEXT15)
    next15 = next15.loc[next15["scenario_name"] == "grain_supply_state_blocking", "state_abbr"].astype(str).tolist()

    top5 = scores.head(5)
    bullet_lines = []
    for row in top5.itertuples(index=False):
        bullet_lines.append(
            f"- `{row.state_abbr}` {row.severity if pd.notna(row.severity) else 'candidate'}: grain production `{row.grain_production_index_bu:,.0f}` bu, "
            f"harvested area `{row.grain_harvested_area_index:,.0f}`, drought events `{row.avg_drought_events_2022_2025}`"
        )

    lines = [
        "# STRATEGIC_INSIGHTS_GRAIN_SUPPLY_SHOCK",
        "",
        "이 문서는 grain supply shock 시뮬레이션의 중간 해석 메모다. 풍작/흉작과 공급량 변화가 물류에 미치는 영향을 보기 위한 것으로, 자연재해가 아니라 **commodity-specific supply shock** 관점에서 정리한다.",
        "",
        "## 핵심 결과",
        "",
        f"- cumulative network loss: {summary['cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {summary['peak_year_loss_pct']:.2%} in {int(summary['peak_year'])}",
        f"- cereal-only cumulative loss: {summary['cereal_cumulative_loss_pct_of_annual_sum']:.2%}",
        f"- cereal-only peak loss: {summary['cereal_peak_year_loss_pct']:.2%}",
        "",
        "해석상 이 shock는 전체 네트워크보다 **곡물 계열 물동량 자체를 강하게 흔드는 narrow commodity shock**에 가깝다.",
        "",
        "## 왜 이 주들이 선정됐나",
        "",
        *bullet_lines,
        "",
        "즉 선정 기준은 생산 기반 규모와 최근 기후 스트레스가 결합된 곳이다. 생산량이 크고, 수확면적이 넓고, 최근 drought exposure가 높은 주일수록 공급 shock에 취약한 것으로 보았다.",
        "",
        "## 시나리오 이후 next 15의 의미",
        "",
        "- must-have 제외 next 15: " + ", ".join(next15),
        "",
        "여기서 `MN`, `NE`, `WA`, `KS`, `WI`, `CO`, `SD`가 상위권에 남는 것은, direct grain core가 흔들릴 때 **Midwest support + western redistribution** 조합이 중요하다는 뜻에 가깝다.",
        "",
        "`IA`, `IL`, `NE`, `MN` 같은 직접 생산 핵심 주는 shock 대상과 support 후보가 부분적으로 겹치지만, `IA`와 `IL`은 직접 타격을 받아 오히려 순위에서 밀리고, 주변 support state가 상대적으로 부상하는 구조가 보인다.",
        "",
        "## NGL 포트폴리오와의 관계",
        "",
        "이 트랙은 NGL의 기존 footprint와 상당 부분 겹치지만, 완전히 같지는 않다. 따라서 grain shock는 trade shock처럼 coastal exposure 문제도 아니고, domestic shock처럼 corridor reliability 문제도 아니다. **농업 공급 축이 흔들릴 때 어떤 주가 cereal-support state로 살아남는가**에 더 가깝다.",
        "",
        "## 현재 단계의 중간 판단",
        "",
        "- grain supply shock는 total network loss는 작지만 cereal-only impact는 크다.",
        "- 따라서 이 트랙은 전사적 네트워크 우선순위보다, commodity-specific portfolio resilience를 보는 용도로 더 적합하다.",
        "- 향후 NGL 제안에선 `MN`, `NE`, `KS`, `WA`를 grain-support candidates로 다시 볼 필요가 있다.",
    ]
    (OUT_DIR / "STRATEGIC_INSIGHTS_GRAIN_SUPPLY_SHOCK.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "grain_cases": str(GRAIN_CASES),
        "grain_scores": str(GRAIN_SCORES),
        "grain_summary": str(GRAIN_SUMMARY),
        "grain_overlap_summary": str(GRAIN_OVERLAP_SUMMARY),
        "output_doc": str(OUT_DIR / "STRATEGIC_INSIGHTS_GRAIN_SUPPLY_SHOCK.md"),
    }
    (OUT_DIR / "grain_supply_insights_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

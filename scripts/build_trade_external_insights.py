from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

TRADE_CASES = PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "trade_external_state_blocking_cases.csv"
TRADE_SCORES = PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "trade_external_state_scores.csv"
TRADE_SUMMARY = PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "simulation_results" / "trade_external_simulation_summary.csv"
TRADE_NEXT15 = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next15_hubs" / "scenario_next15_excluding_must_have.csv"
TRADE_OVERLAP_SUMMARY = WORK_ROOT / "outputs" / "scenario_model" / "trade_external_overlap_maps" / "trade_external_ngl_overlap_summary.md"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "trade_external_visuals"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(TRADE_CASES)
    scores = pd.read_csv(TRADE_SCORES).sort_values("trade_external_score", ascending=False)
    summary = pd.read_csv(TRADE_SUMMARY).iloc[0]
    next15 = pd.read_csv(TRADE_NEXT15)
    next15 = next15.loc[next15["scenario_name"] == "trade_external_state_blocking", "state_abbr"].astype(str).tolist()

    top5 = scores.head(5)
    bullet_lines = []
    for row in top5.itertuples(index=False):
        bullet_lines.append(
            f"- `{row.state_abbr}` {row.severity}: 대외 무역 shock에 가장 노출된 주로 평가됐다. "
            f"총수출 `${row.total_export_value_usd:,.0f}`, China share `{row.china_share:.1%}`, "
            f"vessel share `{row.vessel_share:.1%}`, container share `{row.container_share:.1%}`의 조합이 높다."
        )

    lines = [
        "# STRATEGIC_INSIGHTS_TRADE_EXTERNAL_SHOCK",
        "",
        "이 문서는 trade / external network shock 시뮬레이션의 중간 해석 메모다. 완결 결론이 아니라, 이후 다른 시나리오 결과와 함께 계속 보완할 전제로 작성한다.",
        "",
        "## 핵심 결과",
        "",
        f"- cumulative network loss: {summary['cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {summary['peak_year_loss_pct']:.2%} in {int(summary['peak_year'])}",
        f"- interstate cumulative loss: {summary['interstate_cumulative_loss_pct_of_annual_sum']:.2%}",
        f"- interstate peak loss: {summary['interstate_peak_year_loss_pct']:.2%}",
        "",
        "해석상 이 shock는 자연재해처럼 broad statewide capacity를 크게 무너뜨리기보다, **해상·수출·항만 의존도가 높은 주들의 interstate reliability를 흔드는 shock**에 가깝다.",
        "",
        "## 왜 이 주들이 선정됐나",
        "",
        *bullet_lines,
        "",
        "즉 선정 기준은 단순 물동량 크기가 아니라, **외부 네트워크 의존성**이다. 중국 수출 의존, vessel/container reliance, export dependency가 높은 주일수록 외부 무역 충격에 취약한 것으로 보았다.",
        "",
        "## 시나리오 이후 next 15의 의미",
        "",
        "- must-have 제외 next 15: " + ", ".join(next15),
        "",
        "여기서 `WA`, `LA`, `NC`, `OR`, `OK`, `NE`가 상위권에 반복적으로 남는 것은 두 가지를 시사한다.",
        "",
        "1. `WA`, `LA`는 coastal trade stress 하에서 항만·해상 축의 대체/보강 후보로 읽힌다.",
        "2. `NC`, `OR`, `OK`, `NE`는 직접적인 대형 항만 주는 아니지만, shock 이후 inland rerouting 혹은 coastal demand redistribution을 흡수할 수 있는 보조축 후보로 읽힌다.",
        "",
        "## NGL 포트폴리오와의 관계",
        "",
        "이 트랙은 NGL의 기존 footprint와 상당 부분 겹치지만, 완전히 동일하지는 않다. 따라서 trade shock는 `새로운 대형 주를 여는 문제`보다, **기존 coastal / export-adjacent footprint를 어떻게 우선순위 재조정할지**에 더 가깝다.",
        "",
        "특히 `NC`는 domestic shock에서도 반복적으로 남았고 trade shock에서도 mild 선정 주로 포함되므로, 단순 domestic 보강 후보를 넘어 **East Coast growth + trade exposure를 동시에 가진 주**로 읽을 수 있다.",
        "",
        "## 현재 단계의 중간 판단",
        "",
        "- trade/external network shock는 자연재해보다 overall network loss는 작지만, interstate-only stress는 분명히 만든다.",
        "- 따라서 이 트랙은 `전국망 전체 붕괴`보다 `coastal and export-dependent corridor stress test`로 해석하는 편이 맞다.",
        "- 향후 NGL 제안에선 `WA`, `LA`, `NC`를 trade-sensitive support candidates로 다시 볼 필요가 있다.",
    ]
    (OUT_DIR / "STRATEGIC_INSIGHTS_TRADE_EXTERNAL_SHOCK.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "trade_cases": str(TRADE_CASES),
        "trade_scores": str(TRADE_SCORES),
        "trade_summary": str(TRADE_SUMMARY),
        "trade_overlap_summary": str(TRADE_OVERLAP_SUMMARY),
        "output_doc": str(OUT_DIR / "STRATEGIC_INSIGHTS_TRADE_EXTERNAL_SHOCK.md"),
    }
    (OUT_DIR / "trade_external_insights_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

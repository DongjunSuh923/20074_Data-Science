from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


GRAIN_CASES = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "grain_supply_state_blocking_cases.csv"
GRAIN_SCORES = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "grain_supply_state_scores.csv"
GRAIN_SUMMARY = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "simulation_results" / "grain_supply_simulation_summary.csv"
GRAIN_NETWORK_TOTALS = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "simulation_results" / "grain_supply_network_totals.csv"
GRAIN_CEREAL_TOTALS = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "simulation_results" / "grain_supply_cereal_totals.csv"
GRAIN_NEXT15 = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next15_hubs" / "scenario_next15_excluding_must_have.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "grain_supply_visuals"

STATE_CENTROIDS = {
    "AL": (32.8, -86.8), "AZ": (34.2, -111.7), "AR": (34.9, -92.4), "CA": (37.2, -119.7),
    "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5), "DC": (38.9, -77.0),
    "FL": (28.4, -82.4), "GA": (32.7, -83.3), "ID": (44.2, -114.5), "IL": (40.0, -89.2),
    "IN": (39.9, -86.3), "IA": (42.1, -93.5), "KS": (38.5, -98.0), "KY": (37.5, -85.3),
    "LA": (31.2, -92.3), "ME": (45.3, -69.0), "MD": (39.0, -76.7), "MA": (42.2, -71.8),
    "MI": (44.3, -85.6), "MN": (46.0, -94.3), "MS": (32.7, -89.7), "MO": (38.5, -92.5),
    "MT": (46.9, -110.0), "NE": (41.5, -99.7), "NV": (39.3, -116.6), "NH": (43.7, -71.6),
    "NJ": (40.1, -74.5), "NM": (34.4, -106.1), "NY": (42.9, -75.5), "NC": (35.5, -79.0),
    "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5), "OR": (43.9, -120.6),
    "PA": (41.0, -77.5), "RI": (41.7, -71.5), "SC": (33.8, -80.9), "SD": (44.4, -100.2),
    "TN": (35.8, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.7), "WA": (47.4, -120.7), "WV": (38.6, -80.6), "WI": (44.6, -89.6),
    "WY": (43.0, -107.6),
}
ALL_CONTIG = [s for s in STATE_CENTROIDS if s not in {"AK", "HI"}]
ALL_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
    "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
    "UT","VT","VA","WA","WV","WI","WY"
]


def plot_loss_profile(network: pd.DataFrame, cereal: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(network["year"], network["loss_pct"] * 100, marker="o", linewidth=2.2, color="#2563EB", label="Total network loss")
    ax.plot(cereal["year"], cereal["loss_pct"] * 100, marker="s", linewidth=2.2, color="#16A34A", label="Cereal-only loss")
    ax.axhline(0, color="#9CA3AF", linewidth=1)
    ax.set_title("Grain Supply Shock: Annual Loss Profile", fontsize=14, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Loss (%)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_grain_supply_loss_profile.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_state_selection_map(cases: pd.DataFrame) -> None:
    color_map = {"severe": "#B91C1C", "medium": "#F59E0B", "mild": "#FCD34D", "none": "#E5E7EB"}
    severity_by_state = {row.state_abbr: row.severity for row in cases.itertuples(index=False)}
    fig, ax = plt.subplots(figsize=(10, 7))
    for state in ALL_CONTIG:
        lat, lon = STATE_CENTROIDS[state]
        severity = severity_by_state.get(state, "none")
        ax.scatter(lon, lat, s=240, color=color_map[severity], edgecolor="white", linewidth=0.8, zorder=2)
        if severity != "none":
            ax.text(lon, lat, state, ha="center", va="center", fontsize=8, fontweight="bold", color="#111827", zorder=3)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#F8FAFC")
    ax.set_title("Grain Supply Shock: Selected Blocking States", fontsize=14, weight="bold")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Severe", markerfacecolor=color_map["severe"], markeredgecolor="white", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Medium", markerfacecolor=color_map["medium"], markeredgecolor="white", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Mild", markerfacecolor=color_map["mild"], markeredgecolor="white", markersize=10),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_grain_supply_state_selection_map.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_state_selection_html(cases: pd.DataFrame) -> None:
    color_map = {"severe": "#B91C1C", "medium": "#F59E0B", "mild": "#FCD34D", "none": "#E5E7EB"}
    severity_by_state = {row.state_abbr: row.severity for row in cases.itertuples(index=False)}
    df = pd.DataFrame({"state_abbr": ALL_STATES})
    df["severity"] = df["state_abbr"].map(lambda s: severity_by_state.get(s, "none"))

    fig = go.Figure()
    for idx, severity in enumerate(["none", "mild", "medium", "severe"]):
        sub = df.loc[df["severity"] == severity]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[idx] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, color_map[severity]], [1, color_map[severity]]],
                showscale=False,
                showlegend=severity != "none",
                name=severity.capitalize(),
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + severity.capitalize() + "</extra>",
            )
        )
    labeled = sorted(set(cases["state_abbr"]) & set(STATE_CENTROIDS))
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in labeled],
            lon=[STATE_CENTROIDS[s][1] for s in labeled],
            mode="text",
            text=labeled,
            textfont=dict(size=11, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Grain Supply Shock: Selected Blocking States",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
    )
    fig.write_html(str(OUT_DIR / "02_grain_supply_state_selection_map.html"), include_plotlyjs="cdn")


def plot_next15(next15: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(next15["state_abbr"], next15["touch_2031"], color="#7C3AED")
    ax.set_title("Grain Supply Shock: Next 15 Hubs Excluding Must-have (2031 touch)", fontsize=14, weight="bold")
    ax.set_xlabel("State")
    ax.set_ylabel("2031 touch")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_grain_supply_next15_touch.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: pd.Series, cases: pd.DataFrame, scores: pd.DataFrame, next15: pd.DataFrame) -> None:
    severe = cases.loc[cases["severity"] == "severe", "state_abbr"].tolist()
    medium = cases.loc[cases["severity"] == "medium", "state_abbr"].tolist()
    mild = cases.loc[cases["severity"] == "mild", "state_abbr"].tolist()
    top5 = scores.sort_values("grain_supply_score", ascending=False).head(5)
    lines = [
        "# Grain Supply Shock Progress Reference",
        "",
        "## Shock setup",
        "",
        f"- severe states: {', '.join(severe)}",
        f"- medium states: {', '.join(medium)}",
        f"- mild states: {', '.join(mild)}",
        "- state selection logic: grain production + harvested area + planted area + recent drought exposure",
        "",
        "## Aggregate outcome",
        "",
        f"- cumulative network loss: {summary['cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {summary['peak_year_loss_pct']:.2%} in {int(summary['peak_year'])}",
        f"- cereal-only cumulative loss: {summary['cereal_cumulative_loss_pct_of_annual_sum']:.2%}",
        f"- cereal-only peak loss: {summary['cereal_peak_year_loss_pct']:.2%}",
        "",
        "## Why these states were selected",
        "",
    ]
    for row in top5.itertuples(index=False):
        lines.append(
            f"- `{row.state_abbr}` {row.severity if pd.notna(row.severity) else 'candidate'}: production index `{row.grain_production_index_bu:,.0f}`, "
            f"harvested area `{row.grain_harvested_area_index:,.0f}`, drought events `{row.avg_drought_events_2022_2025}`"
        )
    lines.extend(
        [
            "",
            "## Next 15 (excluding must-have)",
            "",
            "- " + ", ".join(next15["state_abbr"].tolist()),
            "",
            "## Interim interpretation",
            "",
            "- This shock is narrow at the total-network level but strong inside cereal flows.",
            "- `MN`, `NE`, `WA`, `KS`, `WI` stay near the top, suggesting Midwest grain support plus western redistribution value.",
            "- `IA` and `IL` are directly shocked and therefore drop out of candidate status, while nearby support states rise.",
        ]
    )
    (OUT_DIR / "GRAIN_SUPPLY_PROGRESS_REFERENCE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(GRAIN_CASES)
    scores = pd.read_csv(GRAIN_SCORES)
    summary = pd.read_csv(GRAIN_SUMMARY).iloc[0]
    network = pd.read_csv(GRAIN_NETWORK_TOTALS)
    cereal = pd.read_csv(GRAIN_CEREAL_TOTALS)
    next15 = pd.read_csv(GRAIN_NEXT15)
    next15 = next15.loc[next15["scenario_name"] == "grain_supply_state_blocking"].copy()
    next15["touch_2031"] = pd.to_numeric(next15["touch_2031"], errors="coerce")

    plot_loss_profile(network, cereal)
    plot_state_selection_map(cases)
    make_state_selection_html(cases)
    plot_next15(next15)
    write_summary(summary, cases, scores, next15)

    metadata = {
        "loss_profile_png": str(OUT_DIR / "01_grain_supply_loss_profile.png"),
        "selection_map_png": str(OUT_DIR / "02_grain_supply_state_selection_map.png"),
        "selection_map_html": str(OUT_DIR / "02_grain_supply_state_selection_map.html"),
        "next15_png": str(OUT_DIR / "03_grain_supply_next15_touch.png"),
        "summary_md": str(OUT_DIR / "GRAIN_SUPPLY_PROGRESS_REFERENCE.md"),
    }
    (OUT_DIR / "grain_supply_visuals_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

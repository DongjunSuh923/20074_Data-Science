from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


READ_ROOT = PROJECT_ROOT
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

CONSTR_SUMMARY = READ_ROOT / "outputs" / "scenario_model" / "domestic_issue_hybrid" / "construction_infra_hybrid_summary.csv"
CONSTR_NEXT15 = READ_ROOT / "outputs" / "scenario_model" / "domestic_issue_hybrid" / "construction_infra_hybrid_next15_excluding_must_have.csv"
CONSTR_CASES = READ_ROOT / "outputs" / "scenario_model" / "construction_infra_shock" / "construction_infra_state_blocking_cases.csv"

BOTTLENECK_SUMMARY = WORK_ROOT / "outputs" / "scenario_model" / "domestic_bottleneck_shock" / "domestic_bottleneck_summary.csv"
BOTTLENECK_NEXT15 = WORK_ROOT / "outputs" / "scenario_model" / "domestic_bottleneck_shock" / "domestic_bottleneck_next15_excluding_must_have.csv"
BOTTLENECK_CASES = WORK_ROOT / "outputs" / "scenario_model" / "domestic_bottleneck_shock" / "domestic_bottleneck_state_blocking_cases.csv"

OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "domestic_issue_visuals"

STATE_ABBR_MAP = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}

TILE_POSITIONS = {
    "WA": (1, 1), "MT": (2, 1), "ND": (3, 1), "MN": (4, 1), "WI": (5, 1), "MI": (6, 1), "VT": (9, 1), "NH": (10, 1), "ME": (11, 1),
    "OR": (1, 2), "ID": (2, 2), "SD": (3, 2), "IA": (4, 2), "IL": (5, 2), "IN": (6, 2), "OH": (7, 2), "PA": (8, 2), "NY": (9, 2), "MA": (10, 2),
    "CA": (1, 3), "NV": (2, 3), "WY": (3, 3), "NE": (4, 3), "MO": (5, 3), "KY": (6, 3), "WV": (7, 3), "VA": (8, 3), "NJ": (9, 3), "CT": (10, 3), "RI": (11, 3),
    "AZ": (1, 4), "UT": (2, 4), "CO": (3, 4), "KS": (4, 4), "AR": (5, 4), "TN": (6, 4), "NC": (8, 4), "MD": (9, 4), "DE": (10, 4),
    "NM": (1, 5), "OK": (3, 5), "LA": (5, 5), "MS": (6, 5), "AL": (7, 5), "SC": (8, 5), "DC": (9, 5),
    "TX": (2, 6), "FL": (9, 6), "GA": (8, 5),
    "AK": (1, 7), "HI": (2, 7),
}
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
ALL_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
    "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
    "UT","VT","VA","WA","WV","WI","WY"
]


def load_frames():
    constr_summary = pd.read_csv(CONSTR_SUMMARY)
    bottleneck_summary = pd.read_csv(BOTTLENECK_SUMMARY)
    constr_next15 = pd.read_csv(CONSTR_NEXT15)
    bottleneck_next15 = pd.read_csv(BOTTLENECK_NEXT15)
    constr_cases = pd.read_csv(CONSTR_CASES, dtype={"state_fips": "string"})
    bottleneck_cases = pd.read_csv(BOTTLENECK_CASES, dtype={"state_fips": "string"})
    constr_cases["state_abbr"] = constr_cases["state_fips"].astype(str).str.zfill(2).map(STATE_ABBR_MAP)
    return constr_summary, bottleneck_summary, constr_next15, bottleneck_next15, constr_cases, bottleneck_cases


def make_loss_chart(constr_summary: pd.DataFrame, bottleneck_summary: pd.DataFrame) -> None:
    rows = [
        ("Construction hybrid", constr_summary.loc[0, "cumulative_network_loss_pct_of_annual_sum"] * 100, "Cumulative network loss"),
        ("Construction hybrid", constr_summary.loc[0, "peak_network_loss_pct"] * 100, "Peak network loss"),
        ("Construction hybrid", constr_summary.loc[0, "cumulative_subset_loss_pct_of_annual_sum"] * 100, "Cumulative material-subset loss"),
        ("Construction hybrid", constr_summary.loc[0, "peak_subset_loss_pct"] * 100, "Peak material-subset loss"),
        ("Bottleneck shock", bottleneck_summary.loc[0, "cumulative_network_loss_pct_of_annual_sum"] * 100, "Cumulative network loss"),
        ("Bottleneck shock", bottleneck_summary.loc[0, "peak_network_loss_pct"] * 100, "Peak network loss"),
        ("Bottleneck shock", bottleneck_summary.loc[0, "cumulative_interstate_loss_pct_of_annual_sum"] * 100, "Cumulative interstate loss"),
        ("Bottleneck shock", bottleneck_summary.loc[0, "peak_interstate_loss_pct"] * 100, "Peak interstate loss"),
    ]
    df = pd.DataFrame(rows, columns=["scenario", "loss_pct", "metric"])

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot(index="metric", columns="scenario", values="loss_pct")
    pivot = pivot.loc[
        [
            "Cumulative network loss",
            "Peak network loss",
            "Cumulative material-subset loss",
            "Peak material-subset loss",
            "Cumulative interstate loss",
            "Peak interstate loss",
        ]
    ]
    pivot.plot(kind="bar", ax=ax, color=["#c96a24", "#2c6db2"])
    ax.set_ylabel("Loss (%)")
    ax.set_xlabel("")
    ax.set_title("Domestic Issue Shocks: Loss Structure Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_domestic_issue_loss_comparison.png", dpi=200)
    plt.close(fig)


def make_next15_overlap_chart(constr_next15: pd.DataFrame, bottleneck_next15: pd.DataFrame) -> pd.DataFrame:
    states = sorted(set(constr_next15["state_abbr"]) | set(bottleneck_next15["state_abbr"]))
    rows = []
    for state in states:
        rows.append(
            {
                "state_abbr": state,
                "construction_rank": int(constr_next15.loc[constr_next15["state_abbr"] == state, "next_rank"].iloc[0]) if state in set(constr_next15["state_abbr"]) else None,
                "bottleneck_rank": int(bottleneck_next15.loc[bottleneck_next15["state_abbr"] == state, "next_rank"].iloc[0]) if state in set(bottleneck_next15["state_abbr"]) else None,
                "overlap_count": int(state in set(constr_next15["state_abbr"])) + int(state in set(bottleneck_next15["state_abbr"])),
            }
        )
    overlap = pd.DataFrame(rows)
    overlap["construction_score"] = overlap["construction_rank"].apply(lambda x: 16 - x if pd.notna(x) else 0)
    overlap["bottleneck_score"] = overlap["bottleneck_rank"].apply(lambda x: 16 - x if pd.notna(x) else 0)
    overlap["total_score"] = overlap["construction_score"] + overlap["bottleneck_score"]
    overlap = overlap.sort_values(["overlap_count", "total_score"], ascending=[False, False]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    top = overlap.head(15).copy()
    y = range(len(top))
    ax.barh(y, top["construction_score"], color="#c96a24", label="Construction/infrastructure")
    ax.barh(y, top["bottleneck_score"], left=top["construction_score"], color="#2c6db2", label="Domestic bottleneck")
    ax.set_yticks(list(y))
    ax.set_yticklabels(top["state_abbr"])
    ax.invert_yaxis()
    ax.set_xlabel("Presence-weighted rank score")
    ax.set_title("Domestic Issue Candidates: Overlap and Relative Strength")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_domestic_issue_candidate_overlap.png", dpi=200)
    plt.close(fig)

    overlap.to_csv(OUT_DIR / "domestic_issue_candidate_overlap.csv", index=False)
    return overlap


def make_selection_map(constr_cases: pd.DataFrame, bottleneck_cases: pd.DataFrame) -> None:
    all_states = sorted(set(STATE_ABBR_MAP.values()) - {"AK", "HI", "DC"})
    df = pd.DataFrame({"state_abbr": all_states})
    constr_set = set(constr_cases["state_abbr"])
    bottleneck_set = set(bottleneck_cases["state_abbr"])

    def classify(state: str) -> str:
        if state in constr_set and state in bottleneck_set:
            return "Both"
        if state in constr_set:
            return "Construction only"
        if state in bottleneck_set:
            return "Bottleneck only"
        return "Other"

    df["category"] = df["state_abbr"].apply(classify)
    color_map = {
        "Construction only": "#c96a24",
        "Bottleneck only": "#2c6db2",
        "Both": "#7a3eb1",
        "Other": "#d9d9d9",
    }
    fig, ax = plt.subplots(figsize=(14, 8))
    for _, row in df.iterrows():
        state = row["state_abbr"]
        if state not in TILE_POSITIONS:
            continue
        x, y = TILE_POSITIONS[state]
        rect = plt.Rectangle((x, -y), 0.95, 0.95, facecolor=color_map[row["category"]], edgecolor="white")
        ax.add_patch(rect)
        ax.text(x + 0.475, -y + 0.475, state, ha="center", va="center", fontsize=9, fontweight="bold")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map["Both"], label="Both"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["Construction only"], label="Construction only"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["Bottleneck only"], label="Bottleneck only"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["Other"], label="Other"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-7.8, -0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Domestic Issue Shock State Selection")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_domestic_issue_state_selection_map.png", dpi=200)
    plt.close(fig)


def make_selection_map_html(constr_cases: pd.DataFrame, bottleneck_cases: pd.DataFrame) -> None:
    constr_set = set(constr_cases["state_abbr"])
    bottleneck_set = set(bottleneck_cases["state_abbr"])

    def classify(state: str) -> str:
        if state in constr_set and state in bottleneck_set:
            return "Both"
        if state in constr_set:
            return "Construction only"
        if state in bottleneck_set:
            return "Bottleneck only"
        return "Other"

    df = pd.DataFrame({"state_abbr": ALL_STATES})
    df["category"] = df["state_abbr"].apply(classify)
    color_map = {
        "Both": "#7a3eb1",
        "Construction only": "#c96a24",
        "Bottleneck only": "#2c6db2",
        "Other": "#d9d9d9",
    }

    fig = go.Figure()
    for idx, category in enumerate(["Other", "Construction only", "Bottleneck only", "Both"]):
        sub = df.loc[df["category"] == category]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[idx] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, color_map[category]], [1, color_map[category]]],
                showscale=False,
                showlegend=category != "Other",
                name=category,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + category + "</extra>",
            )
        )
    labeled = sorted((constr_set | bottleneck_set) & set(STATE_CENTROIDS))
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
        title="Domestic Issue Shock State Selection",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
    )
    fig.write_html(str(OUT_DIR / "03_domestic_issue_state_selection_map.html"), include_plotlyjs="cdn")


def write_summary(constr_summary: pd.DataFrame, bottleneck_summary: pd.DataFrame, overlap: pd.DataFrame, constr_cases: pd.DataFrame, bottleneck_cases: pd.DataFrame) -> None:
    overlap_states = overlap.loc[overlap["overlap_count"] == 2, "state_abbr"].tolist()
    construction_only = overlap.loc[overlap["construction_rank"].notna() & overlap["bottleneck_rank"].isna(), "state_abbr"].tolist()
    bottleneck_only = overlap.loc[overlap["construction_rank"].isna() & overlap["bottleneck_rank"].notna(), "state_abbr"].tolist()

    lines = [
        "# Strategic Insights: Domestic Issue Scenarios",
        "",
        "This note keeps domestic issue scenarios separate from natural-disaster scenarios.",
        "",
        "## 1. Construction / infrastructure hybrid",
        "",
        "- This shock combines state blocking with destination-side demand weakening for material-linked commodities.",
        f"- cumulative network loss: {constr_summary.loc[0, 'cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {constr_summary.loc[0, 'peak_network_loss_pct']:.2%} in {int(constr_summary.loc[0, 'peak_network_loss_year'])}",
        f"- cumulative material-subset loss: {constr_summary.loc[0, 'cumulative_subset_loss_pct_of_annual_sum']:.2%}",
        "",
        "Interpretation:",
        "- This is a domestic demand deterioration story, not a corridor reliability story.",
        "- The surviving candidates skew toward steady interior support states and material-distribution support states.",
        f"- selected blocking states: {', '.join(f'{r.state_abbr} ({r.severity})' for r in constr_cases.itertuples(index=False))}",
        "",
        "## 2. Domestic bottleneck / congestion shock",
        "",
        "- This shock uses official FHWA TTTR and top bottleneck corridor exposure.",
        f"- cumulative network loss: {bottleneck_summary.loc[0, 'cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {bottleneck_summary.loc[0, 'peak_network_loss_pct']:.2%} in {int(bottleneck_summary.loc[0, 'peak_network_loss_year'])}",
        f"- cumulative interstate loss: {bottleneck_summary.loc[0, 'cumulative_interstate_loss_pct_of_annual_sum']:.2%}",
        "",
        "Interpretation:",
        "- This is a broad interstate reliability stress.",
        "- It hurts cross-state freight structure more directly than the construction hybrid.",
        f"- selected blocking states: {', '.join(f'{r.state_abbr} ({r.severity})' for r in bottleneck_cases.itertuples(index=False))}",
        "",
        "## 3. Candidate pattern insight",
        "",
        f"- overlap candidates across both domestic tracks: {', '.join(overlap_states)}",
        f"- construction-leaning only candidates: {', '.join(construction_only)}",
        f"- bottleneck-leaning only candidates: {', '.join(bottleneck_only)}",
        "",
        "Reading guide:",
        "- Overlap states are resilient across both kinds of domestic issue.",
        "- Construction-only states are stronger when domestic material demand shifts matter more than corridor reliability.",
        "- Bottleneck-only states are stronger when freight reliability and interstate congestion dominate.",
    ]
    (OUT_DIR / "STRATEGIC_INSIGHTS_DOMESTIC_ISSUES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    constr_summary, bottleneck_summary, constr_next15, bottleneck_next15, constr_cases, bottleneck_cases = load_frames()
    make_loss_chart(constr_summary, bottleneck_summary)
    overlap = make_next15_overlap_chart(constr_next15, bottleneck_next15)
    make_selection_map(constr_cases, bottleneck_cases)
    make_selection_map_html(constr_cases, bottleneck_cases)
    write_summary(constr_summary, bottleneck_summary, overlap, constr_cases, bottleneck_cases)

    metadata = {
        "loss_chart_png": str(OUT_DIR / "01_domestic_issue_loss_comparison.png"),
        "candidate_overlap_png": str(OUT_DIR / "02_domestic_issue_candidate_overlap.png"),
        "selection_map_png": str(OUT_DIR / "03_domestic_issue_state_selection_map.png"),
        "selection_map_html": str(OUT_DIR / "03_domestic_issue_state_selection_map.html"),
        "overlap_csv": str(OUT_DIR / "domestic_issue_candidate_overlap.csv"),
        "summary_md": str(OUT_DIR / "STRATEGIC_INSIGHTS_DOMESTIC_ISSUES.md"),
    }
    (OUT_DIR / "domestic_issue_visuals_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

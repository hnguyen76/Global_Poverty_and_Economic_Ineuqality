from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("global_poverty_economic_inequality.csv")

COLORWAY = [
    "#0E7C86",
    "#D95F59",
    "#F2B84B",
    "#5B5F97",
    "#2A9D8F",
    "#8E6C88",
    "#607D8B",
    "#E76F51",
]

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORWAY


def configure_page() -> None:
    st.set_page_config(
        page_title="Global Poverty & Economic Inequality",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe4ea;
            border-radius: 8px;
            padding: 14px 14px 12px;
            border-top: 4px solid #0E7C86;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        .section-note {
            color: #536675;
            font-size: 0.94rem;
            margin-top: -0.45rem;
            margin-bottom: 0.85rem;
        }
        .insight-card {
            background: #ffffff;
            border: 1px solid #dbe4ea;
            border-left: 5px solid #0E7C86;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 128px;
        }
        .footer {
            color: #647482;
            font-size: 0.86rem;
            padding-top: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_pct(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}%"


def fmt_num(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.{digits}f}"


def fmt_money(value: float) -> str:
    return "n/a" if pd.isna(value) else f"${value:,.0f}"


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    span = series.max() - series.min()
    if pd.isna(span) or span == 0:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = (series - series.min()) / span
    return scaled if higher_is_better else 1 - scaled


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return engineer_features(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["income_concentration_gap"] = (
        data["income_share_top10_pct"] - data["income_share_bottom40_pct"]
    )
    data["access_foundation_index"] = data[
        [
            "electricity_access_pct",
            "clean_water_access_pct",
            "internet_penetration_pct",
            "literacy_rate_pct",
            "social_protection_coverage_pct",
        ]
    ].mean(axis=1)
    data["human_capital_index"] = pd.concat(
        [
            minmax(data["hdi_score"]),
            minmax(data["literacy_rate_pct"]),
            minmax(data["life_expectancy_years"]),
            minmax(data["female_labor_participation_pct"]),
        ],
        axis=1,
    ).mean(axis=1) * 100
    data["poverty_pressure_index"] = pd.concat(
        [
            minmax(data["poverty_rate_pct"]),
            minmax(data["unemployment_rate_pct"]),
            minmax(data["inflation_rate_pct"]),
            minmax(data["child_mortality_per_1000"]),
            minmax(np.log1p(data["gdp_per_capita_usd"]), higher_is_better=False),
            minmax(data["hdi_score"], higher_is_better=False),
            minmax(data["access_foundation_index"], higher_is_better=False),
        ],
        axis=1,
    ).mean(axis=1) * 100
    return data


def aggregate_countries(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    return (
        data.groupby(["country", "region", "income_group"], as_index=False)
        .agg(
            poverty_rate_pct=("poverty_rate_pct", "mean"),
            gini_coefficient=("gini_coefficient", "mean"),
            gdp_per_capita_usd=("gdp_per_capita_usd", "median"),
            hdi_score=("hdi_score", "mean"),
            unemployment_rate_pct=("unemployment_rate_pct", "mean"),
            inflation_rate_pct=("inflation_rate_pct", "mean"),
            literacy_rate_pct=("literacy_rate_pct", "mean"),
            life_expectancy_years=("life_expectancy_years", "mean"),
            child_mortality_per_1000=("child_mortality_per_1000", "mean"),
            electricity_access_pct=("electricity_access_pct", "mean"),
            clean_water_access_pct=("clean_water_access_pct", "mean"),
            internet_penetration_pct=("internet_penetration_pct", "mean"),
            female_labor_participation_pct=("female_labor_participation_pct", "mean"),
            social_protection_coverage_pct=("social_protection_coverage_pct", "mean"),
            income_concentration_gap=("income_concentration_gap", "mean"),
            access_foundation_index=("access_foundation_index", "mean"),
            human_capital_index=("human_capital_index", "mean"),
            poverty_pressure_index=("poverty_pressure_index", "mean"),
        )
        .sort_values("poverty_pressure_index", ascending=False)
    )


def aggregate_regions(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    return (
        data.groupby("region", as_index=False)
        .agg(
            records=("record_id", "count"),
            countries=("country", "nunique"),
            poverty_rate_pct=("poverty_rate_pct", "mean"),
            gini_coefficient=("gini_coefficient", "mean"),
            gdp_per_capita_usd=("gdp_per_capita_usd", "median"),
            hdi_score=("hdi_score", "mean"),
            child_mortality_per_1000=("child_mortality_per_1000", "mean"),
            electricity_access_pct=("electricity_access_pct", "mean"),
            clean_water_access_pct=("clean_water_access_pct", "mean"),
            internet_penetration_pct=("internet_penetration_pct", "mean"),
            social_protection_coverage_pct=("social_protection_coverage_pct", "mean"),
            access_foundation_index=("access_foundation_index", "mean"),
            poverty_pressure_index=("poverty_pressure_index", "mean"),
        )
        .sort_values("poverty_rate_pct", ascending=False)
    )


def build_cluster_data(country_data: pd.DataFrame) -> pd.DataFrame:
    cluster_features = [
        "poverty_rate_pct",
        "gini_coefficient",
        "hdi_score",
        "unemployment_rate_pct",
        "inflation_rate_pct",
        "literacy_rate_pct",
        "life_expectancy_years",
        "child_mortality_per_1000",
        "electricity_access_pct",
        "clean_water_access_pct",
        "internet_penetration_pct",
        "social_protection_coverage_pct",
        "income_concentration_gap",
        "access_foundation_index",
        "poverty_pressure_index",
    ]
    cluster_data = country_data.dropna(subset=cluster_features).copy()
    if len(cluster_data) < 3:
        cluster_data["cluster_label"] = "Limited sample"
        return cluster_data

    cluster_data["log_gdp_per_capita"] = np.log1p(cluster_data["gdp_per_capita_usd"])
    features = cluster_features + ["log_gdp_per_capita"]
    n_clusters = min(4, len(cluster_data))
    scaled = StandardScaler().fit_transform(cluster_data[features])
    cluster_data["cluster"] = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=30,
    ).fit_predict(scaled)

    profile = (
        cluster_data.groupby("cluster", as_index=False)
        .agg(
            poverty_rate_pct=("poverty_rate_pct", "mean"),
            gini_coefficient=("gini_coefficient", "mean"),
            gdp_per_capita_usd=("gdp_per_capita_usd", "median"),
            access_foundation_index=("access_foundation_index", "mean"),
        )
    )

    poverty_cut = profile["poverty_rate_pct"].quantile(0.66)
    gdp_cut = profile["gdp_per_capita_usd"].quantile(0.34)
    gini_cut = profile["gini_coefficient"].quantile(0.66)
    access_cut = profile["access_foundation_index"].quantile(0.34)

    def label(row: pd.Series) -> str:
        if row["poverty_rate_pct"] >= poverty_cut and row["gdp_per_capita_usd"] <= gdp_cut:
            return "High-poverty structural pressure"
        if row["gini_coefficient"] >= gini_cut:
            return "Inequality concentration risk"
        if row["access_foundation_index"] <= access_cut:
            return "Access foundation gap"
        return "Transition and stabilization group"

    label_map = profile.assign(cluster_label=profile.apply(label, axis=1)).set_index("cluster")[
        "cluster_label"
    ]
    cluster_data["cluster_label"] = cluster_data["cluster"].map(label_map)
    return cluster_data


def render_metric_cards(data: pd.DataFrame, country_data: pd.DataFrame, focus_year: int) -> None:
    high_pressure_country = "n/a"
    if not country_data.empty:
        high_pressure_country = country_data.iloc[0]["country"]

    metrics = [
        ("Records", f"{len(data):,}", f"Filtered period to {focus_year}"),
        ("Countries", f"{data['country'].nunique():,}", "Distinct economies"),
        ("Avg poverty", fmt_pct(data["poverty_rate_pct"].mean()), "Filtered mean"),
        ("Avg Gini", fmt_num(data["gini_coefficient"].mean()), "Inequality intensity"),
        ("Median GDP", fmt_money(data["gdp_per_capita_usd"].median()), "GDP per capita"),
        ("Highest pressure", high_pressure_country, "Focus-year country"),
    ]

    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


def render_overview(period_data: pd.DataFrame, focus_country: pd.DataFrame, region_data: pd.DataFrame) -> None:
    trend = (
        period_data.groupby("year", as_index=False)
        .agg(
            poverty_rate_pct=("poverty_rate_pct", "mean"),
            gini_coefficient=("gini_coefficient", "mean"),
            gdp_per_capita_usd=("gdp_per_capita_usd", "median"),
            hdi_score=("hdi_score", "mean"),
        )
        .sort_values("year")
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Poverty Rate",
            "Gini Coefficient",
            "GDP per Capita",
            "HDI Score",
        ),
        vertical_spacing=0.14,
    )
    fig.add_trace(
        go.Scatter(
            x=trend["year"],
            y=trend["poverty_rate_pct"],
            mode="lines+markers",
            line=dict(color="#D95F59", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trend["year"],
            y=trend["gini_coefficient"],
            mode="lines+markers",
            line=dict(color="#5B5F97", width=3),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=trend["year"],
            y=trend["gdp_per_capita_usd"],
            mode="lines+markers",
            line=dict(color="#0E7C86", width=3),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trend["year"],
            y=trend["hdi_score"],
            mode="lines+markers",
            line=dict(color="#2A9D8F", width=3),
        ),
        row=2,
        col=2,
    )
    fig.update_layout(height=560, showlegend=False, margin=dict(l=20, r=20, t=60, b=30))
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(tickprefix="$", row=2, col=1)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if region_data.empty:
            st.info("No regional data for the selected filters.")
        else:
            bar = px.bar(
                region_data.sort_values("poverty_rate_pct"),
                x="poverty_rate_pct",
                y="region",
                color="access_foundation_index",
                orientation="h",
                color_continuous_scale=["#2A9D8F", "#F2B84B", "#D95F59"],
                labels={
                    "poverty_rate_pct": "Average poverty rate (%)",
                    "region": "",
                    "access_foundation_index": "Access index",
                },
                title="Regional Poverty vs Access",
            )
            bar.update_layout(height=560, margin=dict(l=10, r=10, t=55, b=30))
            st.plotly_chart(bar, use_container_width=True)

    if not focus_country.empty:
        scatter = px.scatter(
            focus_country,
            x="gdp_per_capita_usd",
            y="poverty_rate_pct",
            color="region",
            size="child_mortality_per_1000",
            hover_name="country",
            hover_data={
                "income_group": True,
                "gini_coefficient": ":.1f",
                "hdi_score": ":.3f",
                "access_foundation_index": ":.1f",
                "poverty_pressure_index": ":.1f",
                "gdp_per_capita_usd": ":,.0f",
                "poverty_rate_pct": ":.1f",
            },
            labels={
                "gdp_per_capita_usd": "GDP per capita, log scale",
                "poverty_rate_pct": "Poverty rate (%)",
            },
            title="Country Positioning",
        )
        scatter.update_xaxes(type="log", tickprefix="$")
        scatter.update_layout(height=610, margin=dict(l=20, r=20, t=60, b=35))
        st.plotly_chart(scatter, use_container_width=True)


def render_inequality(period_data: pd.DataFrame, focus_country: pd.DataFrame) -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        if focus_country.empty:
            st.info("No country data for the selected filters.")
        else:
            inequality = px.scatter(
                focus_country,
                x="gini_coefficient",
                y="poverty_rate_pct",
                color="income_group",
                size="income_concentration_gap",
                hover_name="country",
                hover_data={"region": True, "access_foundation_index": ":.1f", "hdi_score": ":.3f"},
                labels={
                    "gini_coefficient": "Gini coefficient",
                    "poverty_rate_pct": "Poverty rate (%)",
                    "income_concentration_gap": "Income concentration gap",
                },
                title="Inequality and Poverty",
            )
            inequality.update_layout(height=540, margin=dict(l=20, r=20, t=60, b=35))
            st.plotly_chart(inequality, use_container_width=True)

    with right:
        top_gap = focus_country.sort_values("income_concentration_gap", ascending=False).head(15)
        if top_gap.empty:
            st.info("No income concentration data for the selected filters.")
        else:
            gap_fig = px.bar(
                top_gap.sort_values("income_concentration_gap"),
                x="income_concentration_gap",
                y="country",
                color="region",
                orientation="h",
                labels={"income_concentration_gap": "Top 10% share minus bottom 40%", "country": ""},
                title="Largest Income Concentration Gaps",
            )
            gap_fig.update_layout(height=540, margin=dict(l=10, r=10, t=60, b=35))
            st.plotly_chart(gap_fig, use_container_width=True)

    corr_columns = [
        "gdp_per_capita_usd",
        "poverty_rate_pct",
        "gini_coefficient",
        "hdi_score",
        "unemployment_rate_pct",
        "inflation_rate_pct",
        "literacy_rate_pct",
        "life_expectancy_years",
        "child_mortality_per_1000",
        "electricity_access_pct",
        "clean_water_access_pct",
        "internet_penetration_pct",
        "female_labor_participation_pct",
        "social_protection_coverage_pct",
        "income_concentration_gap",
        "access_foundation_index",
        "poverty_pressure_index",
    ]
    corr_data = period_data[corr_columns].dropna()
    if len(corr_data) >= 3:
        corr = corr_data.corr(numeric_only=True)
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation Structure",
        )
        heatmap.update_layout(height=720, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Select more records to calculate a reliable correlation matrix.")


def render_priorities(focus_country: pd.DataFrame) -> None:
    if focus_country.empty:
        st.info("No focus-year country data for the selected filters.")
        return

    high = focus_country.sort_values("poverty_rate_pct", ascending=False).head(12)
    low = focus_country.sort_values("poverty_rate_pct", ascending=True).head(12)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Highest Poverty", "Lowest Poverty"),
        horizontal_spacing=0.17,
    )
    fig.add_trace(
        go.Bar(
            x=high["poverty_rate_pct"],
            y=high["country"],
            orientation="h",
            marker_color="#D95F59",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=low["poverty_rate_pct"],
            y=low["country"],
            orientation="h",
            marker_color="#0E7C86",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=590, showlegend=False, margin=dict(l=20, r=20, t=60, b=30))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    cluster_data = build_cluster_data(focus_country)
    if not cluster_data.empty:
        cluster_fig = px.scatter(
            cluster_data,
            x="access_foundation_index",
            y="poverty_rate_pct",
            color="cluster_label",
            size="gdp_per_capita_usd",
            hover_name="country",
            hover_data={
                "region": True,
                "income_group": True,
                "gini_coefficient": ":.1f",
                "hdi_score": ":.3f",
                "poverty_pressure_index": ":.1f",
                "gdp_per_capita_usd": ":,.0f",
            },
            labels={
                "access_foundation_index": "Access foundation index",
                "poverty_rate_pct": "Poverty rate (%)",
                "cluster_label": "Archetype",
            },
            title="Strategic Country Archetypes",
        )
        cluster_fig.update_layout(height=610, margin=dict(l=20, r=20, t=60, b=35))
        st.plotly_chart(cluster_fig, use_container_width=True)

    display_columns = [
        "country",
        "region",
        "income_group",
        "poverty_pressure_index",
        "poverty_rate_pct",
        "gini_coefficient",
        "gdp_per_capita_usd",
        "hdi_score",
        "access_foundation_index",
        "social_protection_coverage_pct",
    ]
    st.dataframe(
        focus_country[display_columns]
        .sort_values("poverty_pressure_index", ascending=False)
        .head(20)
        .style.format(
            {
                "poverty_pressure_index": "{:.1f}",
                "poverty_rate_pct": "{:.1f}%",
                "gini_coefficient": "{:.1f}",
                "gdp_per_capita_usd": "${:,.0f}",
                "hdi_score": "{:.3f}",
                "access_foundation_index": "{:.1f}",
                "social_protection_coverage_pct": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_insights(period_data: pd.DataFrame, focus_country: pd.DataFrame, region_data: pd.DataFrame) -> None:
    if period_data.empty:
        st.info("No data available for the selected filters.")
        return

    corr_candidates = [
        "gdp_per_capita_usd",
        "gini_coefficient",
        "hdi_score",
        "unemployment_rate_pct",
        "inflation_rate_pct",
        "literacy_rate_pct",
        "life_expectancy_years",
        "child_mortality_per_1000",
        "electricity_access_pct",
        "clean_water_access_pct",
        "internet_penetration_pct",
        "social_protection_coverage_pct",
        "income_concentration_gap",
        "access_foundation_index",
    ]
    corr = period_data[corr_candidates + ["poverty_rate_pct"]].corr(numeric_only=True)[
        "poverty_rate_pct"
    ].drop("poverty_rate_pct")
    strongest_negative = corr.sort_values().head(1)
    strongest_positive = corr.sort_values(ascending=False).head(1)

    trend = period_data.groupby("year", as_index=False).agg(
        poverty_rate_pct=("poverty_rate_pct", "mean"),
        access_foundation_index=("access_foundation_index", "mean"),
        gini_coefficient=("gini_coefficient", "mean"),
    )
    first, last = trend.iloc[0], trend.iloc[-1]
    poverty_change = last["poverty_rate_pct"] - first["poverty_rate_pct"]
    access_change = last["access_foundation_index"] - first["access_foundation_index"]
    gini_change = last["gini_coefficient"] - first["gini_coefficient"]

    pressure_region = "n/a"
    pressure_value = np.nan
    if not region_data.empty:
        top_region = region_data.sort_values("poverty_pressure_index", ascending=False).iloc[0]
        pressure_region = top_region["region"]
        pressure_value = top_region["poverty_pressure_index"]

    pressure_country = "n/a"
    if not focus_country.empty:
        pressure_country = focus_country.sort_values("poverty_pressure_index", ascending=False).iloc[0][
            "country"
        ]

    cards = [
        (
            "Poverty system signal",
            f"The strongest negative poverty association is {strongest_negative.index[0]} "
            f"({strongest_negative.iloc[0]:+.2f}).",
        ),
        (
            "Compounding pressure",
            f"The strongest positive poverty association is {strongest_positive.index[0]} "
            f"({strongest_positive.iloc[0]:+.2f}).",
        ),
        (
            "Period movement",
            f"Poverty changed {poverty_change:+.1f} pts, access changed {access_change:+.1f}, "
            f"and Gini changed {gini_change:+.1f}.",
        ),
        (
            "Priority focus",
            f"{pressure_region} has the highest regional pressure index "
            f"({fmt_num(pressure_value)}); {pressure_country} leads country pressure.",
        ),
    ]

    columns = st.columns(4)
    for column, (title, body) in zip(columns, cards):
        column.markdown(
            f"""
            <div class="insight-card">
                <strong>{title}</strong>
                <p>{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Recommendations")
    st.markdown(
        """
        - Build an access floor across electricity, clean water, internet, literacy, and social protection.
        - Pair GDP growth with distribution metrics so income concentration does not hide persistent deprivation.
        - Track child mortality as a household-vulnerability signal, not only as a health outcome.
        - Use archetypes to split intervention design between basic service expansion, inclusive labor markets, and resilience.
        """
    )


def main() -> None:
    configure_page()

    if not DATA_PATH.exists():
        st.error(f"Cannot find dataset: {DATA_PATH}")
        st.stop()

    data = load_data(DATA_PATH)
    years = sorted(data["year"].dropna().astype(int).unique().tolist())
    regions = sorted(data["region"].dropna().unique().tolist())
    income_groups = sorted(data["income_group"].dropna().unique().tolist())
    countries = sorted(data["country"].dropna().unique().tolist())

    st.sidebar.title("Filters")
    selected_years = st.sidebar.slider(
        "Year range",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years)),
    )
    available_focus_years = [year for year in years if selected_years[0] <= year <= selected_years[1]]
    focus_year = st.sidebar.selectbox(
        "Focus year",
        available_focus_years,
        index=len(available_focus_years) - 1,
    )
    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    selected_income = st.sidebar.multiselect("Income group", income_groups, default=income_groups)
    selected_countries = st.sidebar.multiselect("Country", countries, placeholder="All countries")

    period_data = data[
        data["year"].between(selected_years[0], selected_years[1])
        & data["region"].isin(selected_regions)
        & data["income_group"].isin(selected_income)
    ].copy()
    if selected_countries:
        period_data = period_data[period_data["country"].isin(selected_countries)].copy()
    focus_data = period_data[period_data["year"] == focus_year].copy()
    focus_country = aggregate_countries(focus_data)
    region_data = aggregate_regions(period_data)

    st.title("Global Poverty & Economic Inequality")
    st.caption("Created by Hieu Nguyen")

    if period_data.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    render_metric_cards(period_data, focus_country, focus_year)

    overview, inequality, priorities, insights, data_table = st.tabs(
        ["Overview", "Inequality & Access", "Country Priorities", "Strategic Insights", "Data"]
    )

    with overview:
        render_overview(period_data, focus_country, region_data)

    with inequality:
        render_inequality(period_data, focus_country)

    with priorities:
        render_priorities(focus_country)

    with insights:
        render_insights(period_data, focus_country, region_data)

    with data_table:
        st.dataframe(period_data, use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered CSV",
            data=period_data.to_csv(index=False).encode("utf-8"),
            file_name="filtered_global_poverty_dashboard.csv",
            mime="text/csv",
        )

    st.markdown('<div class="footer">Created by Hieu Nguyen</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

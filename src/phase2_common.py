from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase2"

COUNTRY_NAME_MAP = {
    "Bahamas": "Bahamas, The",
    "Belarus (Byelorussia)": "Belarus",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Brunei": "Brunei Darussalam",
    "Burkina Faso (Upper Volta)": "Burkina Faso",
    "Cambodia (Kampuchea)": "Cambodia",
    "Cape Verde": "Cabo Verde",
    "Congo": "Congo, Rep.",
    "Congo, Democratic Republic of (Zaire)": "Congo, Dem. Rep.",
    "Cote D'Ivoire": "Cote d'Ivoire",
    "Czech Republic": "Czechia",
    "East Timor": "Timor-Leste",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Iran (Persia)": "Iran, Islamic Rep.",
    "Korea, People's Republic of": "Korea, Dem. People's Rep.",
    "Korea, Republic of": "Korea, Rep.",
    "Laos": "Lao PDR",
    "Macedonia (FYROM/North Macedonia)": "North Macedonia",
    "Madagascar (Malagasy)": "Madagascar",
    "Myanmar (Burma)": "Myanmar",
    "Romania": "Romania",
    "Rumania": "Romania",
    "Russia (Soviet Union)": "Russian Federation",
    "Slovakia": "Slovak Republic",
    "Sri Lanka (Ceylon)": "Sri Lanka",
    "Surinam": "Suriname",
    "Swaziland (Eswatini)": "Eswatini",
    "Syria": "Syrian Arab Republic",
    "Tanzania (Tanganyika)": "Tanzania",
    "Turkey (Ottoman Empire)": "Turkiye",
    "United States of America": "United States",
    "Venezuela": "Venezuela, RB",
    "Yemen (Arab Republic of Yemen)": "Yemen, Rep.",
    "Yemen, People's Republic of": "Yemen, Rep.",
}

SELECTED_WDI_INDICATORS = [
    "GDP per capita (constant 2015 US$)",
    "Life expectancy at birth, total (years)",
    "Military expenditure (% of GDP)",
    "Military expenditure (current USD)",
    "Population, total",
    "Refugees under the mandate of the UNHCR by country or territory of asylum",
    "Urban population (% of total population)",
]

GROWUP_COLUMNS = [
    "year",
    "countryname",
    "onset_ko_flag",
    "incidence_flag",
    "onset_do_flag",
    "warhist",
    "peaceyears",
]

CLUSTER_FEATURES = [
    "log_gdp_per_capita",
    "Life expectancy at birth, total (years)",
    "Urban population (% of total population)",
    "Military expenditure (% of GDP)",
    "refugees_asylum_per_100k",
    "warhist",
    "peaceyears",
    "incidence_flag",
]

KNN_TARGETS = [
    "GDP per capita (constant 2015 US$)",
    "Life expectancy at birth, total (years)",
    "refugees_asylum_per_100k",
]


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)


def prepare_modeling_dataset(
    start_year: int = 1990,
    end_year: int = 2022,
) -> pd.DataFrame:
    wdi_path = PROCESSED_DATA_DIR / "WDIcleaned.csv"
    growup_path = RAW_DATA_DIR / "GROWupRaw.csv"
    country_path = RAW_DATA_DIR / "WDICountry.csv"

    wdi_df = pd.read_csv(wdi_path)
    wdi_long = wdi_df.melt(
        id_vars=["Country Name", "Indicator Name"],
        var_name="Year",
        value_name="Value",
    )
    wdi_long["Year"] = pd.to_numeric(wdi_long["Year"], errors="coerce").astype("Int64")
    wdi_long = wdi_long.dropna(subset=["Year"])
    wdi_long["Year"] = wdi_long["Year"].astype(int)
    wdi_long = wdi_long[
        (wdi_long["Year"] >= start_year)
        & (wdi_long["Year"] <= end_year)
        & (wdi_long["Indicator Name"].isin(SELECTED_WDI_INDICATORS))
    ]

    wdi_pivot = (
        wdi_long.pivot_table(
            index=["Country Name", "Year"],
            columns="Indicator Name",
            values="Value",
            aggfunc="mean",
        )
        .reset_index()
    )

    country_meta = pd.read_csv(
        country_path,
        usecols=["Short Name", "Region", "Income Group"],
    ).rename(columns={"Short Name": "Country Name"})

    growup_df = pd.read_csv(growup_path, usecols=GROWUP_COLUMNS)
    growup_df["Country Name"] = growup_df["countryname"].replace(COUNTRY_NAME_MAP)
    growup_df = growup_df.drop(columns=["countryname"]).rename(columns={"year": "Year"})
    growup_df = (
        growup_df.groupby(["Country Name", "Year"], as_index=False)
        .mean(numeric_only=True)
    )

    modeling_df = (
        wdi_pivot.merge(country_meta, on="Country Name", how="left")
        .merge(growup_df, on=["Country Name", "Year"], how="left")
        .sort_values(["Country Name", "Year"])
        .reset_index(drop=True)
    )

    modeling_df["log_gdp_per_capita"] = np.log1p(
        modeling_df["GDP per capita (constant 2015 US$)"]
    )
    modeling_df["log_population"] = np.log1p(modeling_df["Population, total"])
    modeling_df["log_refugees_asylum"] = np.log1p(
        modeling_df["Refugees under the mandate of the UNHCR by country or territory of asylum"]
    )
    modeling_df["refugees_asylum_per_100k"] = (
        modeling_df["Refugees under the mandate of the UNHCR by country or territory of asylum"]
        / modeling_df["Population, total"]
        * 100000
    )
    modeling_df["military_usd_per_capita"] = (
        modeling_df["Military expenditure (current USD)"]
        / modeling_df["Population, total"]
    )

    output_path = PROCESSED_DATA_DIR / "phase2_modeling_data.csv"
    modeling_df.to_csv(output_path, index=False)
    return modeling_df


def get_latest_country_snapshot(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_non_null_share: float = 0.3,
) -> pd.DataFrame:
    coverage_by_year = df.groupby("Year")[feature_cols].apply(lambda x: x.notna().mean())
    min_feature_coverage = coverage_by_year.min(axis=1)
    eligible_years = min_feature_coverage[min_feature_coverage >= min_non_null_share]
    snapshot_year = int(eligible_years.index.max()) if not eligible_years.empty else int(df["Year"].max())
    return df[df["Year"] == snapshot_year].copy()


def build_knn_feature_list(target: str) -> list[str]:
    candidate_features = [
        "log_gdp_per_capita",
        "Life expectancy at birth, total (years)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "military_usd_per_capita",
        "log_population",
        "refugees_asylum_per_100k",
        "warhist",
        "peaceyears",
        "incidence_flag",
        "onset_ko_flag",
        "onset_do_flag",
    ]

    blocked = {
        "GDP per capita (constant 2015 US$)": {"log_gdp_per_capita"},
        "Life expectancy at birth, total (years)": {
            "Life expectancy at birth, total (years)"
        },
        "refugees_asylum_per_100k": {"refugees_asylum_per_100k"},
    }
    return [feature for feature in candidate_features if feature not in blocked.get(target, set())]

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

for folder in [FIGURES_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

WDI_PATH = PROCESSED_DIR / "WDIcleaned.csv"
GROWUP_PATH = RAW_DIR / "GROWupRaw.csv"

# Helper functions
def year_stack(df: pd.DataFrame, indicator: str, firstyr: int, lastyr: int) -> pd.DataFrame:
    indicator_df = df[df["Indicator Name"] == indicator].copy()
    year_cols = [str(year) for year in range(firstyr, lastyr + 1)]
    year_cols = [col for col in year_cols if col in indicator_df.columns]

    if len(year_cols) == 0:
        raise ValueError(f"No year columns found for indicator: {indicator}")

    melted = pd.melt(
        indicator_df,
        id_vars=["Country Name"],
        value_vars=year_cols,
        var_name="Year",
        value_name=indicator
    )

    melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
    melted = melted.dropna(subset=[indicator, "Year"])
    return melted


def build_wdi_panel(df: pd.DataFrame, indicators: list[str], firstyr: int, lastyr: int) -> pd.DataFrame:
    result_df = year_stack(df, indicators[0], firstyr, lastyr)

    for indicator in indicators[1:]:
        next_df = year_stack(df, indicator, firstyr, lastyr)
        result_df = pd.merge(
            result_df,
            next_df,
            on=["Country Name", "Year"],
            how="inner"
        )

    return result_df


def load_growup_features(path: Path, firstyr: int, lastyr: int) -> pd.DataFrame:
    growup = pd.read_csv(path)

    needed_cols = ["year", "countryname", "warhist", "peaceyears"]
    missing_cols = [col for col in needed_cols if col not in growup.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in GROWupRaw.csv: {missing_cols}")

    growup = growup[needed_cols].copy()
    growup = growup.rename(columns={
        "year": "Year",
        "countryname": "Country Name"
    })

    growup["Year"] = pd.to_numeric(growup["Year"], errors="coerce")
    growup = growup[(growup["Year"] >= firstyr) & (growup["Year"] <= lastyr)]

    growup = growup.groupby(["Country Name", "Year"], as_index=False).mean(numeric_only=True)
    return growup


def prepare_pca_data(firstyr: int = 2000, lastyr: int = 2020, avg_start: int = 2015, avg_end: int = 2020) -> pd.DataFrame:
    wdi = pd.read_csv(WDI_PATH)
    wdi.columns = wdi.columns.astype(str).str.strip()

    indicators = [
        "GDP per capita (constant 2015 US$)",
        "Life expectancy at birth, total (years)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)"
    ]

    wdi_panel = build_wdi_panel(
        df=wdi,
        indicators=indicators,
        firstyr=firstyr,
        lastyr=lastyr
    )

    growup_panel = load_growup_features(GROWUP_PATH, firstyr, lastyr)

    final_df = pd.merge(
        wdi_panel,
        growup_panel,
        on=["Country Name", "Year"],
        how="inner"
    )

    numeric_cols = [
        "GDP per capita (constant 2015 US$)",
        "Life expectancy at birth, total (years)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "warhist",
        "peaceyears"
    ]

    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

    final_df["GDP_log"] = np.log1p(final_df["GDP per capita (constant 2015 US$)"])

    final_df = final_df[
        [
            "Country Name",
            "Year",
            "GDP_log",
            "Life expectancy at birth, total (years)",
            "Urban population (% of total population)",
            "Military expenditure (% of GDP)",
            "warhist",
            "peaceyears"
        ]
    ].dropna()

    final_df = final_df[(final_df["Year"] >= avg_start) & (final_df["Year"] <= avg_end)]
    country_profile = final_df.groupby("Country Name", as_index=False).mean(numeric_only=True)

    return country_profile

# Main PCA
def run_pca():
    df = prepare_pca_data()

    feature_cols = [
        "GDP_log",
        "Life expectancy at birth, total (years)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "warhist",
        "peaceyears"
    ]

    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    explained_df = pd.DataFrame({
        "Principal Component": np.arange(1, len(feature_cols) + 1),
        "Explained Variance Ratio": pca.explained_variance_ratio_,
        "Cumulative Explained Variance": np.cumsum(pca.explained_variance_ratio_)
    })

    explained_df.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i}" for i in range(1, len(feature_cols) + 1)]
    )
    loadings_df.to_csv(TABLES_DIR / "pca_loadings.csv")

    # Scree plot
    plt.figure(figsize=(9, 5))
    plt.plot(
        explained_df["Principal Component"],
        explained_df["Explained Variance Ratio"],
        marker="o"
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_scree_plot.png", dpi=300)
    plt.show()

    # PC1 vs PC2 plot
    plot_df = df.copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(plot_df["PC1"], plot_df["PC2"], alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: Countries Projected onto First Two Principal Components")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_country_projection.png", dpi=300)
    plt.show()

    n_90 = np.argmax(explained_df["Cumulative Explained Variance"].values >= 0.90) + 1

    print("\nNumber of components needed for 90% variance:", n_90)
    print("\nExplained variance table:")
    print(explained_df)

    print("\nSaved files:")
    print(TABLES_DIR / "pca_explained_variance.csv")
    print(TABLES_DIR / "pca_loadings.csv")
    print(FIGURES_DIR / "pca_scree_plot.png")
    print(FIGURES_DIR / "pca_country_projection.png")


if __name__ == "__main__":
    run_pca()
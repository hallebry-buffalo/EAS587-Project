from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
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

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
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


def prepare_clustering_data(firstyr: int = 2000, lastyr: int = 2020, avg_start: int = 2015, avg_end: int = 2020) -> pd.DataFrame:
    wdi = pd.read_csv(WDI_PATH)
    wdi.columns = wdi.columns.astype(str).str.strip()

    required_cols = ["Country Name", "Indicator Name"]
    for col in required_cols:
        if col not in wdi.columns:
            raise ValueError(f"{col} not found in WDIcleaned.csv")

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

    # Average country values over recent years
    final_df = final_df[(final_df["Year"] >= avg_start) & (final_df["Year"] <= avg_end)]

    country_profile = final_df.groupby("Country Name", as_index=False).mean(numeric_only=True)

    return country_profile


# ---------------------------------------------------
# Main clustering
# ---------------------------------------------------
def run_clustering():
    df = prepare_clustering_data()

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

    k_results = []
    best_k = None
    best_score = -1
    best_model = None

    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        k_results.append({
            "k": k,
            "silhouette_score": score
        })

        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans

    results_df = pd.DataFrame(k_results)
    results_df.to_csv(TABLES_DIR / "clustering_model_selection.csv", index=False)

    df["Cluster"] = best_model.predict(X_scaled)

    # Save cluster assignments
    df.sort_values(["Cluster", "Country Name"]).to_csv(
        TABLES_DIR / "country_cluster_assignments.csv",
        index=False
    )

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    plot_df = df.copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    plt.figure(figsize=(10, 7))
    for cluster_id in sorted(plot_df["Cluster"].unique()):
        subset = plot_df[plot_df["Cluster"] == cluster_id]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster_id}", alpha=0.75)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"K-Means Clustering of Countries (best k = {best_k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase2_kmeans_cluster_plot.png", dpi=300)
    plt.show()

    print("\nBest k:", best_k)
    print("Best silhouette score:", round(best_score, 4))

    print("\nSaved files:")
    print(TABLES_DIR / "clustering_model_selection.csv")
    print(TABLES_DIR / "country_cluster_assignments.csv")
    print(FIGURES_DIR / "phase2_kmeans_cluster_plot.png")


if __name__ == "__main__":
    run_clustering()
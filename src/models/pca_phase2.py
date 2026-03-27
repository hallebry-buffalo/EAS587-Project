from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
MODELS_DIR = BASE_DIR / "models"

for folder in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Prefer the corrected merged file directly
MERGED_CANDIDATES = [
    PROCESSED_DIR / "GROWupWDImerged.csv",
    RAW_DIR / "GROWupWDImerged.csv",
    BASE_DIR / "GROWupWDImerged.csv",
]
MERGED_PATH = next((p for p in MERGED_CANDIDATES if p.exists()), None)
if MERGED_PATH is None:
    raise FileNotFoundError("Could not find GROWupWDImerged.csv in data/processed, data/raw, or project root.")

# clean dataset and check whether important columns exist
def load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    required = ["Country Name", "Indicator Name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def looks_like_aggregate(country: str) -> bool:
    if not isinstance(country, str):
        return True
    x = country.strip().lower()
    bad_tokens = [
        "income", "world", "europe", "asia", "africa", "america", "caribbean",
        "middle east", "north america", "south asia", "sub-saharan", "euro area",
        "oecd", "ida", "ibrd", "least developed", "fragile", "arab world",
        "small states", "heavily indebted", "union", "demographic", "dividend",
        "classified", "excluding", "including", "members", "countries", "states",
        "transition", "pre-demographic", "post-demographic", "early-demographic",
        "late-demographic", "low & middle", "high income", "low income",
        "upper middle income", "lower middle income", "north africa", "latin america"
    ]
    return any(tok in x for tok in bad_tokens)

# Convert year and value columns to numeric
def pivot_indicators(df: pd.DataFrame, indicators: list[str], first_year: int = 2000, last_year: int = 2022) -> pd.DataFrame:
    year_cols = [str(y) for y in range(first_year, last_year + 1) if str(y) in df.columns]
    subset = df[df["Indicator Name"].isin(indicators)].copy()
    long_df = subset.melt(
        id_vars=["Country Name", "Indicator Name"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Country Name", "Indicator Name", "Year", "Value"])
    wide = long_df.pivot_table(
        index=["Country Name", "Year"],
        columns="Indicator Name",
        values="Value",
        aggfunc="mean"
    ).reset_index()
    return wide


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    log_cols = [
        "GDP per capita (constant 2015 US$)",
        "Refugees under the mandate of the UNHCR by country or territory of origin",
        "peaceyears",
        "Mortality rate, infant (per 1,000 live births)",
        "Mortality rate, under-5 (per 1,000 live births)",
        "Adolescent fertility rate (births per 1,000 women ages 15-19)",
    ]
    for col in log_cols:
        if col in out.columns:
            min_val = out[col].min()
            shift = 0 if pd.isna(min_val) or min_val > -1 else abs(min_val) + 1
            safe_name = (
                col.replace(" ", "_")
                   .replace("(", "")
                   .replace(")", "")
                   .replace("%", "pct")
                   .replace(",", "")
                   .replace("-", "_")
                   .replace("/", "_")
            )
            out[f"log__{safe_name}"] = np.log1p(out[col] + shift)
    return out

# Indicators selected for PCA analysis
def build_country_profile(first_year: int = 2000, last_year: int = 2022, avg_start: int = 2015, avg_end: int = 2022) -> pd.DataFrame:
    indicators = [
        "GDP per capita (constant 2015 US$)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "warhist",
        "peaceyears",
        "Refugees under the mandate of the UNHCR by country or territory of origin",
        "Mortality rate, infant (per 1,000 live births)",
        "Mortality rate, under-5 (per 1,000 live births)",
        "Fertility rate, total (births per woman)",
        "Adolescent fertility rate (births per 1,000 women ages 15-19)",
        "Population growth (annual %)",
        "Rural population (% of total population)",
        "Life expectancy at birth, total (years)",
    ]
    merged = load_merged(MERGED_PATH)
    panel = pivot_indicators(merged, indicators, first_year=first_year, last_year=last_year)
    panel = panel[~panel["Country Name"].apply(looks_like_aggregate)].copy()
    panel = add_engineered_features(panel)
    recent = panel[(panel["Year"] >= avg_start) & (panel["Year"] <= avg_end)].copy()
    profile = recent.groupby("Country Name", as_index=False).mean(numeric_only=True)
    return panel, profile

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#final country-level dataset
def run_pca():
    _, profile = build_country_profile()

    feature_cols = [
        "log__GDP_per_capita_constant_2015_US$",
        "Life expectancy at birth, total (years)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "warhist",
        "log__peaceyears",
        "log__Refugees_under_the_mandate_of_the_UNHCR_by_country_or_territory_of_origin",
        "log__Mortality_rate_infant_per_1000_live_births",
        "log__Mortality_rate_under_5_per_1000_live_births",
        "Fertility rate, total (births per woman)",
        "log__Adolescent_fertility_rate_births_per_1000_women_ages_15_19",
        "Population growth (annual %)",
    ]
    feature_cols = [c for c in feature_cols if c in profile.columns]
    X = profile[feature_cols].copy()

    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_ready = prep.fit_transform(X)

    pca = PCA(random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_ready)

    explained_df = pd.DataFrame({
        "Principal Component": np.arange(1, len(feature_cols) + 1),
        "Explained Variance Ratio": pca.explained_variance_ratio_,
        "Cumulative Explained Variance": np.cumsum(pca.explained_variance_ratio_)
    })
    explained_df.to_csv(TABLES_DIR / "pca_explained_variance_v2.csv", index=False)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i}" for i in range(1, len(feature_cols) + 1)]
    )
    loadings_df.to_csv(TABLES_DIR / "pca_loadings_v2.csv")

# Scree plot to show variance explained by each component
    plt.figure(figsize=(9, 5))
    plt.plot(explained_df["Principal Component"], explained_df["Explained Variance Ratio"], marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot on Corrected Merged Dataset")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_scree_plot_v2.png", dpi=300)
    plt.show()

# Save PC1 and PC2 values for each country
    plot_df = profile[["Country Name"]].copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]
    plot_df.to_csv(TABLES_DIR / "pca_country_projection_v2.csv", index=False)

# Scatter plot using first two principal components
    plt.figure(figsize=(10, 7))
    plt.scatter(plot_df["PC1"], plot_df["PC2"], alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Countries Projected onto the First Two PCs")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_country_projection_v2.png", dpi=300)
    plt.show()
    
#components needed to explain at least 90% variance
    n_90 = int(np.argmax(explained_df["Cumulative Explained Variance"].values >= 0.90) + 1)
    print("Using merged file:", MERGED_PATH)
    print("Countries used:", len(profile))
    print("Features used:", feature_cols)
    print("Number of components needed for 90% variance:", n_90)
    print(explained_df)

if __name__ == "__main__":
    run_pca()
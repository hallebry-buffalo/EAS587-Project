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

# corrected merged file
MERGED_CANDIDATES = [
    PROCESSED_DIR / "GROWupWDImerged.csv",
    RAW_DIR / "GROWupWDImerged.csv",
    BASE_DIR / "GROWupWDImerged.csv",
]
MERGED_PATH = next((p for p in MERGED_CANDIDATES if p.exists()), None)
if MERGED_PATH is None:
    raise FileNotFoundError("Could not find GROWupWDImerged.csv in data/processed, data/raw, or project root.")

#Load data set and clean column names
def load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    required = ["Country Name", "Indicator Name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

#remove aggregate regions
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

#format and reshape dataset
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
    #reduce skewness using log
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

#final country level dataset
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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def run_clustering():
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
#preprocessing
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90, random_state=RANDOM_STATE))
    ])
    X_ready = prep.fit_transform(X)

#choosing best k values
    scores = []
    best_model = None
    best_score = -1
    best_k = None

    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
        labels = km.fit_predict(X_ready)
        score = silhouette_score(X_ready, labels)
        scores.append({"k": k, "silhouette_score": score})
        if score > best_score:
            best_score = score
            best_k = k
            best_model = km

    score_df = pd.DataFrame(scores)
    score_df.to_csv(TABLES_DIR / "clustering_model_selection_v2.csv", index=False)

    plot_pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = plot_pca.fit_transform(X_ready)

    out = profile[["Country Name"]].copy()
    out["Cluster"] = best_model.predict(X_ready)
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    out.to_csv(TABLES_DIR / "country_cluster_assignments_v2.csv", index=False)

    cluster_summary = profile.copy()
    cluster_summary["Cluster"] = out["Cluster"]
    cluster_summary.groupby("Cluster")[feature_cols].mean().to_csv(TABLES_DIR / "cluster_summary_v2.csv")

#plot clusters
    plt.figure(figsize=(10, 7))
    for cluster_id in sorted(out["Cluster"].unique()):
        s = out[out["Cluster"] == cluster_id]
        plt.scatter(s["PC1"], s["PC2"], label=f"Cluster {cluster_id}", alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"K-Means on Corrected Merged Dataset (best k = {best_k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase2_kmeans_cluster_plot_v2.png", dpi=300)
    plt.show()

    print("Using merged file:", MERGED_PATH)
    print("Countries used:", len(profile))
    print("Features used:", feature_cols)
    print("Best k:", best_k)
    print("Best silhouette score:", round(best_score, 4))

if __name__ == "__main__":
    run_clustering()
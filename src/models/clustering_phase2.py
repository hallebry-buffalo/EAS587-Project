from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
for folder in [FIGURES_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def load_main_data(path):
    df = pd.read_csv(path)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    return df[["Country Name", "Indicator Name"] + year_cols].copy()

def load_country_metadata(path):
    meta = pd.read_csv(path)
    meta = meta[["Table Name", "Region", "Income Group"]].rename(columns={"Table Name": "Country Name"})
    return meta.drop_duplicates()

def melt_indicator(df, indicator, first_year, last_year):
    year_cols = [str(y) for y in range(first_year, last_year + 1)]
    temp = df[df["Indicator Name"] == indicator].copy()
    melted = temp.melt(id_vars=["Country Name"], value_vars=year_cols, var_name="Year", value_name=indicator)
    melted["Year"] = melted["Year"].astype(int)
    return melted

def build_country_year_panel(df, indicators, first_year, last_year):
    panel = melt_indicator(df, indicators[0], first_year, last_year)
    for ind in indicators[1:]:
        panel = panel.merge(melt_indicator(df, ind, first_year, last_year), on=["Country Name", "Year"], how="inner")
    return panel

def attach_country_metadata(panel, meta):
    df = panel.merge(meta, on="Country Name", how="left")
    return df[df["Region"].notna()].copy()

def add_log_features(df, cols_to_log):
    out = df.copy()
    for col in cols_to_log:
        min_val = out[col].min()
        if pd.notna(min_val) and min_val > -1:
            out[f"log_{col}"] = np.log1p(out[col])
        else:
            out[f"log_{col}"] = np.log1p(out[col] + abs(min_val) + 1)
    return out

def aggregate_recent_country_profile(panel, feature_cols, start_year, end_year):
    df = panel[(panel["Year"] >= start_year) & (panel["Year"] <= end_year)].copy()
    agg_map = {col: "mean" for col in feature_cols}
    agg_map["Region"] = "first"
    agg_map["Income Group"] = "first"
    return df.groupby("Country Name", as_index=False).agg(agg_map)

def run_kmeans(country_df, feature_cols, out_prefix="phase2_kmeans", k_values=range(2, 9)):
    model_df = country_df[["Country Name", "Region", "Income Group"] + feature_cols].dropna().reset_index(drop=True)
    X_scaled = StandardScaler().fit_transform(model_df[feature_cols])

    best_model, best_k, best_score = None, None, -1
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        rows.append({"k": k, "silhouette_score": score})
        if score > best_score:
            best_model, best_k, best_score = km, k, score

    pd.DataFrame(rows).to_csv(TABLES_DIR / f"{out_prefix}_silhouette_scores.csv", index=False)
    model_df["Cluster"] = best_model.predict(X_scaled)

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    model_df["PC1"], model_df["PC2"] = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(10, 7))
    for cid in sorted(model_df["Cluster"].unique()):
        s = model_df[model_df["Cluster"] == cid]
        plt.scatter(s["PC1"], s["PC2"], label=f"Cluster {cid}", alpha=0.75)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"K-Means Clusters (best k = {best_k})")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{out_prefix}_cluster_plot.png", dpi=300); plt.close()

    model_df.sort_values(["Cluster", "Country Name"]).to_csv(TABLES_DIR / f"{out_prefix}_cluster_membership.csv", index=False)
    print("Best k:", best_k)
    print("Best silhouette score:", round(best_score, 4))

if __name__ == "__main__":
    merged_path = PROCESSED_DIR / "GROWupWDImerged.csv"
    if not merged_path.exists():
        merged_path = BASE_DIR / "GROWupWDImerged.csv"
    country_meta_path = RAW_DIR / "WDICountry.csv"
    if not country_meta_path.exists():
        country_meta_path = BASE_DIR / "WDICountry.csv"

    data = load_main_data(merged_path)
    country_meta = load_country_metadata(country_meta_path)

    raw_features = [
        "GDP per capita (constant 2015 US$)",
        "Life expectancy at birth, total (years)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "warhist", "peaceyears",
        "Refugees under the mandate of the UNHCR by country or territory of origin",
    ]
    panel = build_country_year_panel(data, raw_features, 2000, 2020)
    panel = attach_country_metadata(panel, country_meta)
    panel = add_log_features(panel, [
        "GDP per capita (constant 2015 US$)",
        "Refugees under the mandate of the UNHCR by country or territory of origin",
    ])
    features = [
        "log_GDP per capita (constant 2015 US$)",
        "Life expectancy at birth, total (years)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "warhist", "peaceyears",
        "log_Refugees under the mandate of the UNHCR by country or territory of origin",
    ]
    profile = aggregate_recent_country_profile(panel, features, 2015, 2020)
    run_kmeans(profile, features)
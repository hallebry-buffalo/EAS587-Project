from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from phase2_common import CLUSTER_FEATURES, OUTPUT_DIR, ensure_output_dirs, get_latest_country_snapshot, prepare_modeling_dataset


def choose_best_k(x_scaled, k_values: range) -> tuple[int, pd.DataFrame]:
    scores = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x_scaled)
        scores.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(x_scaled, labels),
                "inertia": model.inertia_,
            }
        )
    score_df = pd.DataFrame(scores)
    best_k = int(score_df.loc[score_df["silhouette_score"].idxmax(), "k"])
    return best_k, score_df


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    candidate_features = [feature for feature in CLUSTER_FEATURES if df[feature].notna().any()]
    snapshot = get_latest_country_snapshot(df, candidate_features)
    numeric_features = [feature for feature in candidate_features if snapshot[feature].notna().any()]

    cluster_df = snapshot[
        ["Country Name", "Year", "Region", "Income Group"] + numeric_features
    ].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(cluster_df[numeric_features])
    x_scaled = scaler.fit_transform(x_imputed)

    best_k, score_df = choose_best_k(x_scaled, range(2, 9))
    score_df.to_csv(OUTPUT_DIR / "cluster_model_selection.csv", index=False)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_df["cluster"] = kmeans.fit_predict(x_scaled)

    viz_pca = PCA(n_components=2, random_state=42)
    components = viz_pca.fit_transform(x_scaled)
    cluster_df["PC1"] = components[:, 0]
    cluster_df["PC2"] = components[:, 1]

    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=cluster_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        style="Region",
        palette="tab10",
        s=85,
    )
    plt.title(f"Country Clusters for {cluster_df['Year'].max()} (k={best_k})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "country_clusters.png", dpi=300)
    plt.close()

    summary = (
        cluster_df.groupby("cluster")[numeric_features]
        .mean()
        .round(3)
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    cluster_df.to_csv(OUTPUT_DIR / "country_cluster_assignments.csv", index=False)
    return cluster_df


def main() -> None:
    ensure_output_dirs()
    modeling_df = prepare_modeling_dataset()
    cluster_df = run_clustering(modeling_df)

    print(f"Clustering countries used: {cluster_df['Country Name'].nunique()}")
    print(f"Clustering outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

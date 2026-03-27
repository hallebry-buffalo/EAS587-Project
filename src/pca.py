from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from phase2_common import CLUSTER_FEATURES, OUTPUT_DIR, ensure_output_dirs, get_latest_country_snapshot, prepare_modeling_dataset


def run_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_features = [feature for feature in CLUSTER_FEATURES if df[feature].notna().any()]
    snapshot = get_latest_country_snapshot(df, candidate_features)
    numeric_features = [feature for feature in candidate_features if snapshot[feature].notna().any()]

    pca_df = snapshot[
        ["Country Name", "Year", "Region", "Income Group"] + numeric_features
    ].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(pca_df[numeric_features])
    x_scaled = scaler.fit_transform(x_imputed)

    pca = PCA(random_state=42)
    pca.fit(x_scaled)

    explained = pd.DataFrame(
        {
            "principal_component": [f"PC{i}" for i in range(1, len(numeric_features) + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": pca.explained_variance_ratio_.cumsum(),
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric_features,
        columns=[f"PC{i}" for i in range(1, len(numeric_features) + 1)],
    ).reset_index(names="feature")

    explained.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)
    loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=explained, x="principal_component", y="explained_variance_ratio")
    plt.title("Explained Variance by Principal Component")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_explained_variance.png", dpi=300)
    plt.close()

    pca_2 = PCA(n_components=2, random_state=42)
    transformed = pca_2.fit_transform(x_scaled)
    plot_df = pca_df[["Country Name", "Region"]].copy()
    plot_df["PC1"] = transformed[:, 0]
    plot_df["PC2"] = transformed[:, 1]

    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Region",
        s=80,
    )
    plt.title("PCA Projection of Countries")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_country_projection.png", dpi=300)
    plt.close()

    return explained, loadings


def main() -> None:
    ensure_output_dirs()
    modeling_df = prepare_modeling_dataset()
    explained_df, _ = run_pca(modeling_df)

    print("Top PCA explained variance rows:")
    print(explained_df.head().to_string(index=False))
    print(f"PCA outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

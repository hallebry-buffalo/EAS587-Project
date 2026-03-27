from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    snapshot = df[df["Year"] == snapshot_year].copy()
    return snapshot


def choose_best_k(x_scaled: np.ndarray, k_values: range) -> tuple[int, pd.DataFrame]:
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
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
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


def fit_knn_pipeline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    use_pca: bool,
) -> GridSearchCV:
    numeric_features = x_train.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]
    if use_pca:
        steps.append(("pca", PCA(random_state=42)))
    steps.append(("knn", KNeighborsRegressor()))
    pipeline = Pipeline(steps=steps)

    cv_splits = min(5, groups_train.nunique())
    cv = GroupKFold(n_splits=max(3, cv_splits))

    param_grid: dict[str, list[object]] = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    if use_pca:
        max_components = min(len(numeric_features), 8)
        param_grid["pca__n_components"] = list(range(2, max_components + 1))

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups_train)
    return search


def evaluate_knn_models(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for target in KNN_TARGETS:
        feature_cols = build_knn_feature_list(target)
        modeling_cols = ["Country Name", "Year", target] + feature_cols
        target_df = df[modeling_cols].dropna(subset=[target]).copy()

        x = target_df[feature_cols]
        y = target_df[target]
        groups = target_df["Country Name"]

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups))

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        baseline_search = fit_knn_pipeline(x_train, y_train, groups_train, use_pca=False)
        pca_search = fit_knn_pipeline(x_train, y_train, groups_train, use_pca=True)

        for label, search in [("knn_only", baseline_search), ("pca_knn", pca_search)]:
            predictions = search.best_estimator_.predict(x_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            result = {
                "target": target,
                "model": label,
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
                "best_params": str(search.best_params_),
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "train_countries": groups_train.nunique(),
                "test_countries": groups.iloc[test_idx].nunique(),
            }
            results.append(result)

            prediction_df = pd.DataFrame(
                {
                    "actual": y_test.to_numpy(),
                    "predicted": predictions,
                }
            )
            safe_target = (
                target.lower()
                .replace(" ", "_")
                .replace("%", "pct")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("/", "_")
                .replace("$", "usd")
            )
            prediction_df.to_csv(
                OUTPUT_DIR / f"{safe_target}_{label}_predictions.csv",
                index=False,
            )

            plt.figure(figsize=(7, 7))
            sns.scatterplot(data=prediction_df, x="actual", y="predicted", s=60)
            min_val = min(prediction_df["actual"].min(), prediction_df["predicted"].min())
            max_val = max(prediction_df["actual"].max(), prediction_df["predicted"].max())
            plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
            plt.title(f"{label} Predictions for {target}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{safe_target}_{label}_scatter.png", dpi=300)
            plt.close()

    results_df = pd.DataFrame(results).sort_values(["target", "test_rmse"])
    results_df.to_csv(OUTPUT_DIR / "knn_results.csv", index=False)
    return results_df


def main() -> None:
    ensure_output_dirs()
    modeling_df = prepare_modeling_dataset()

    print(f"Saved modeling table to: {PROCESSED_DATA_DIR / 'phase2_modeling_data.csv'}")
    print(f"Modeling table shape: {modeling_df.shape}")

    cluster_df = run_clustering(modeling_df)
    explained_df, loadings_df = run_pca(modeling_df)
    knn_results_df = evaluate_knn_models(modeling_df)

    print(f"Clustering countries used: {cluster_df['Country Name'].nunique()}")
    print("Top PCA explained variance rows:")
    print(explained_df.head().to_string(index=False))
    print("KNN results:")
    print(knn_results_df.to_string(index=False))
    print(f"All Phase 2 outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

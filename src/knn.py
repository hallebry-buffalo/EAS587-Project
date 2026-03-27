from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase2_common import KNN_TARGETS, OUTPUT_DIR, build_knn_feature_list, ensure_output_dirs, prepare_modeling_dataset


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

            results.append(
                {
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
            )

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
    knn_results_df = evaluate_knn_models(modeling_df)

    print("KNN results:")
    print(knn_results_df.to_string(index=False))
    print(f"KNN outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

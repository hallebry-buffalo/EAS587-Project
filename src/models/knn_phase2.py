from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
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
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

for folder in [MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

WDI_PATH = PROCESSED_DIR / "WDIcleaned.csv"
GROWUP_PATH = RAW_DIR / "GROWupRaw.csv"

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def year_stack(df: pd.DataFrame, indicator: str, firstyr: int, lastyr: int) -> pd.DataFrame:
    """
    Convert one wide-format indicator into long format:
    Country Name | Year | indicator_value
    """
    indicator_df = df[df["Indicator Name"] == indicator].copy()

    year_cols = [str(year) for year in range(firstyr, lastyr + 1)]

    # keep only the years that actually exist in the file
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


def build_wdi_panel(df: pd.DataFrame, features: list[str], target: str, firstyr: int, lastyr: int) -> pd.DataFrame:
    """
    Merge multiple WDI indicators together by Country Name + Year
    """
    all_indicators = features + [target]

    result_df = year_stack(df, all_indicators[0], firstyr, lastyr)

    for indicator in all_indicators[1:]:
        next_df = year_stack(df, indicator, firstyr, lastyr)
        result_df = pd.merge(
            result_df,
            next_df,
            on=["Country Name", "Year"],
            how="inner"
        )

    return result_df


def load_growup_features(path: Path, firstyr: int, lastyr: int) -> pd.DataFrame:
    """
    Load only the conflict-related columns needed for KNN
    """
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

    # if duplicates exist for country-year, average them
    growup = growup.groupby(["Country Name", "Year"], as_index=False).mean(numeric_only=True)

    return growup


def prepare_model_data(firstyr: int = 2000, lastyr: int = 2020) -> pd.DataFrame:
    """
    Build the final country-year dataset for KNN
    """
    wdi = pd.read_csv(WDI_PATH)
    wdi.columns = wdi.columns.astype(str).str.strip()

    required_cols = ["Country Name", "Indicator Name"]
    for col in required_cols:
        if col not in wdi.columns:
            raise ValueError(
                f"{col} not found in WDIcleaned.csv. "
                f"Columns found: {wdi.columns.tolist()[:10]} ..."
            )

    feature_indicators = [
        "GDP per capita (constant 2015 US$)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)"
    ]
    target_indicator = "Life expectancy at birth, total (years)"

    wdi_panel = build_wdi_panel(
        df=wdi,
        features=feature_indicators,
        target=target_indicator,
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

    # clean numeric columns
    numeric_cols = [
        "GDP per capita (constant 2015 US$)",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "Life expectancy at birth, total (years)",
        "warhist",
        "peaceyears"
    ]

    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

    # log transform GDP because it is usually very skewed
    final_df["GDP_log"] = np.log1p(final_df["GDP per capita (constant 2015 US$)"])

    # keep only needed columns
    final_df = final_df[
        [
            "Country Name",
            "Year",
            "GDP_log",
            "Urban population (% of total population)",
            "Military expenditure (% of GDP)",
            "warhist",
            "peaceyears",
            "Life expectancy at birth, total (years)"
        ]
    ].dropna()

    return final_df


# ---------------------------------------------------
# Main KNN model
# ---------------------------------------------------
def run_knn():
    df = prepare_model_data(firstyr=2000, lastyr=2020)

    print("Final dataset shape:", df.shape)
    print(df.head().to_string())

    feature_cols = [
        "GDP_log",
        "Urban population (% of total population)",
        "Military expenditure (% of GDP)",
        "warhist",
        "peaceyears"
    ]
    target_col = "Life expectancy at birth, total (years)"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ])

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]   # 1 = Manhattan, 2 = Euclidean
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nTraining Metrics:")
    print("RMSE:", round(train_rmse, 4))
    print("MAE :", round(train_mae, 4))
    print("R²  :", round(train_r2, 4))

    print("\nTest Metrics:")
    print("RMSE:", round(test_rmse, 4))
    print("MAE :", round(test_mae, 4))
    print("R²  :", round(test_r2, 4))

    # Save predictions
    pred_df = X_test.copy()
    pred_df["Actual"] = y_test.values
    pred_df["Predicted"] = y_test_pred
    pred_df["Absolute Error"] = np.abs(pred_df["Actual"] - pred_df["Predicted"])
    pred_df.to_csv(TABLES_DIR / "knn_test_predictions.csv", index=False)

    # Save model
    joblib.dump(best_model, MODELS_DIR / "knn_life_expectancy_model.pkl")

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual Life Expectancy")
    plt.ylabel("Predicted Life Expectancy")
    plt.title("KNN Regression: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "knn_actual_vs_predicted.png", dpi=300)
    plt.show()

    print("\nSaved files:")
    print(TABLES_DIR / "knn_test_predictions.csv")
    print(MODELS_DIR / "knn_life_expectancy_model.pkl")
    print(FIGURES_DIR / "knn_actual_vs_predicted.png")


if __name__ == "__main__":
    run_knn()
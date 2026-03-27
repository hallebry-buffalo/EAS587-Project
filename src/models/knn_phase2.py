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

#corrected merged file
MERGED_CANDIDATES = [
    PROCESSED_DIR / "GROWupWDImerged.csv",
    RAW_DIR / "GROWupWDImerged.csv",
    BASE_DIR / "GROWupWDImerged.csv",
]
MERGED_PATH = next((p for p in MERGED_CANDIDATES if p.exists()), None)
if MERGED_PATH is None:
    raise FileNotFoundError("Could not find GROWupWDImerged.csv in data/processed, data/raw, or project root.")

# Load dataset and clean column names
def load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    required = ["Country Name", "Indicator Name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

#remove non-country aggregates
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

#Convert dataset into country-year format
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

#important indicators
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

import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def run_knn():
    panel, _ = build_country_profile(first_year=2000, last_year=2022, avg_start=2015, avg_end=2022)

    feature_cols = [
        "log__GDP_per_capita_constant_2015_US$",
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
        "Rural population (% of total population)",
    ]
    feature_cols = [c for c in feature_cols if c in panel.columns]
    target_col = "Life expectancy at birth, total (years)"

    df = panel[["Country Name", "Year"] + feature_cols + [target_col]].copy()
    df = df[~df["Country Name"].apply(looks_like_aggregate)].dropna(subset=[target_col]).reset_index(drop=True)

    # Hold out entire countries to prevent leakage from same-country rows across years.
    X = df[feature_cols]
    y = df[target_col]
    groups = df["Country Name"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

#pipeline for knn
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ])

# Hyperparameter tuning
    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    cv = GroupKFold(n_splits=5)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )
#train
    grid.fit(X_train, y_train, groups=train_groups)
    best_model = grid.best_estimator_
#test
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

# Evaluation metrics
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "best_params": grid.best_params_,
        "countries_train": groups.iloc[train_idx].nunique(),
        "countries_test": groups.iloc[test_idx].nunique(),
    }

    pred_df = X_test.copy()
    pred_df["Country Name"] = groups.iloc[test_idx].values
    pred_df["Actual"] = y_test.values
    pred_df["Predicted"] = y_test_pred
    pred_df["Absolute Error"] = np.abs(pred_df["Actual"] - pred_df["Predicted"])
    pred_df.to_csv(TABLES_DIR / "knn_test_predictions_v2.csv", index=False)

    pd.DataFrame([metrics]).to_csv(TABLES_DIR / "knn_metrics_v2.csv", index=False)
    joblib.dump(best_model, MODELS_DIR / "knn_life_expectancy_model_v2.pkl")

# Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    lo = min(y_test.min(), y_test_pred.min())
    hi = max(y_test.max(), y_test_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual Life Expectancy")
    plt.ylabel("Predicted Life Expectancy")
    plt.title("KNN Regression on Corrected Merged Dataset")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "knn_actual_vs_predicted_v2.png", dpi=300)
    plt.show()

    print("Using merged file:", MERGED_PATH)
    print("Rows used:", len(df))
    print("Features used:", feature_cols)
    print("Best Parameters:", grid.best_params_)
    print("\nTraining Metrics:")
    print("RMSE:", round(metrics["train_rmse"], 4))
    print("MAE :", round(metrics["train_mae"], 4))
    print("R²  :", round(metrics["train_r2"], 4))
    print("\nTest Metrics:")
    print("RMSE:", round(metrics["test_rmse"], 4))
    print("MAE :", round(metrics["test_mae"], 4))
    print("R²  :", round(metrics["test_r2"], 4))
    print("\nTrain countries:", metrics["countries_train"])
    print("Test countries :", metrics["countries_test"])

if __name__ == "__main__":
    run_knn()
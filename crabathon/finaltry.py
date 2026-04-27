import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
KAGGLE_INPUT = Path("/kaggle/input")
SEARCH_DIRS = [SCRIPT_DIR, SCRIPT_DIR / "data", Path.cwd(), KAGGLE_INPUT]


def find_data_file(name):
    # Search direct candidate paths
    for directory in SEARCH_DIRS:
        candidate = directory / name
        if candidate.exists():
            return candidate

    # If the file isn't directly in the root, look one level deeper in Kaggle input folders
    for directory in SEARCH_DIRS:
        if directory.exists() and directory.is_dir():
            for child in directory.iterdir():
                candidate = child / name
                if candidate.exists():
                    return candidate

    return None

KAGGLE_TRAIN = Path("/kaggle/input/competitions/crabathon/train.csv")
KAGGLE_TEST = Path("/kaggle/input/competitions/crabathon/test.csv")
TRAIN_PATH = KAGGLE_TRAIN if KAGGLE_TRAIN.exists() else find_data_file("train.csv")
TEST_PATH = KAGGLE_TEST if KAGGLE_TEST.exists() else find_data_file("test.csv")
OUTPUT_PATH = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else SCRIPT_DIR / "submission.csv"

if TRAIN_PATH is None or TEST_PATH is None:
    searched = ", ".join(str(p) for p in SEARCH_DIRS)
    raise FileNotFoundError(
        "Could not find train.csv and/or test.csv. "
        f"Searched: {searched}"
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9

    df["Shell_per_Len"] = df["Shell Weight"] / (df["Length"] + eps)
    df["Shell_sqrt"] = np.sqrt(df["Shell Weight"])
    df["Shell_Height"] = df["Shell Weight"] * df["Height"]
    df["BMI_proxy"] = df["Weight"] / (df["Length"] * df["Diameter"] + eps)
    df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]
    df["Weight_sqrt"] = np.sqrt(df["Weight"])
    df["Len_cubed"] = df["Length"] ** 3
    df["Meat_Ratio"] = df["Shucked Weight"] / (df["Weight"] + eps)
    df["Shucked_Shell"] = df["Shucked Weight"] / (df["Shell Weight"] + eps)
    df["Shell_Diam"] = df["Shell Weight"] / (df["Diameter"] + eps)
    df["Height_ratio"] = df["Height"] / (df["Diameter"] + eps)
    df["All_weight"] = df["Shucked Weight"] + df["Viscera Weight"] + df["Shell Weight"]
    df["Viscera_ratio"] = df["Viscera Weight"] / (df["Weight"] + eps)

    return df


print("\n=== Crab Age Regression (MAE-focused) ===\n")

# ---- STEP 1: Load Training and Test Data ----
print(f"Loading data from: {TRAIN_PATH}, {TEST_PATH}\n")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# ---- STEP 2: Remove Invalid Records (Height = 0) ----
train = train[train["Height"] > 0].reset_index(drop=True)

# ---- STEP 3: Add Engineered Features ----
train = add_features(train)
test = add_features(test)

# ---- STEP 4: Encode Categorical Variables (Sex) ----
train = pd.get_dummies(train, columns=["Sex"])
test = pd.get_dummies(test, columns=["Sex"])

test = test.reindex(columns=[c for c in train.columns if c != "Age"], fill_value=0)

# ---- STEP 5: Prepare Features (X) and Target (y) ----
X = train.drop(columns=["id", "Age"])
y = train["Age"]
X_test = test.drop(columns=["id"])

test_ids = test["id"]
print(f"Training rows: {len(X)}")
print(f"Test rows:     {len(X_test)}")
print(f"Feature count: {X.shape[1]}\n")

# ---- STEP 6: Setup 5-Fold Cross-Validation ----
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

# ---- STEP 7: Train XGBoost on Each Fold ----
for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.75,
        colsample_bylevel=0.8,
        min_child_weight=3,
        gamma=0.03,
        reg_alpha=0.01,
        reg_lambda=0.6,
        eval_metric="mae",
        early_stopping_rounds=120,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    oof_predictions[val_idx] = model.predict(X_val)
    test_predictions += model.predict(X_test) / kf.n_splits

    best_iter = model.best_iteration if model.best_iteration is not None else model.n_estimators
    fold_mae = mean_absolute_error(y_val, np.floor(oof_predictions[val_idx] + 0.25).clip(1, 29))
    print(f"Fold {fold}/5 | best_iter={best_iter} | fold MAE={fold_mae:.4f}")

# ---- STEP 8: Apply Smart Rounding (Threshold = 0.25) ----
print("\nEvaluating out-of-fold performance...")

oof_mae = mean_absolute_error(y, oof_predictions)
rounded_oof = np.floor(oof_predictions + 0.25).clip(1, 29).astype(int)
rounded_mae = mean_absolute_error(y, rounded_oof)

print(f"OOF MAE (raw)      : {oof_mae:.4f}")
print(f"OOF MAE (rounded)  : {rounded_mae:.4f}\n")

# ---- STEP 9: Generate Final Test Predictions ----
final_test = np.floor(test_predictions + 0.25).clip(1, 29).astype(int)

# ---- STEP 10: Create and Save Submission ----
submission = pd.DataFrame({"id": test_ids, "Age": final_test})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"Saved submission to: {OUTPUT_PATH}")
print("=== Done ===\n")

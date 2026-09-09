import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report
)
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_features.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "pd_xgboost.joblib"


# ============================================================
# 1. LOAD FEATURES
# ============================================================

print("=" * 70)
print("HOME CREDIT — PD MODEL TRAINING")
print("=" * 70)

print("\n[1] Loading feature dataset...")

df = pd.read_parquet(FEATURE_FILE)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# 2. CHECK TARGET
# ============================================================

print("\n[2] Target distribution...")

if "TARGET" not in df.columns:
    raise ValueError("TARGET column not found.")

print(df["TARGET"].value_counts())
print("\nDefault rate:")
print(f"{df['TARGET'].mean():.4%}")


# ============================================================
# 3. REMOVE IDENTIFIER
# ============================================================

print("\n[3] Preparing features...")

TARGET = "TARGET"

DROP_COLUMNS = [
    "SK_ID_CURR"
]

X = df.drop(columns=[TARGET] + [
    c for c in DROP_COLUMNS
    if c in df.columns
])

y = df[TARGET].astype(int)

print(f"Model features: {X.shape[1]}")


# ============================================================
# 4. HANDLE DATA TYPES
# ============================================================

print("\n[4] Processing feature types...")

# Convert object/category columns to numeric codes
# so XGBoost can consume the dataset.

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(f"Categorical columns: {len(categorical_columns)}")

for col in categorical_columns:
    X[col] = X[col].astype("category").cat.codes

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# XGBoost handles missing numeric values natively.


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

print("\n[5] Creating train/validation split...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")

print(
    f"Training default rate:   {y_train.mean():.4%}"
)

print(
    f"Validation default rate: {y_valid.mean():.4%}"
)


# ============================================================
# 6. CALCULATE CLASS IMBALANCE
# ============================================================

print("\n[6] Calculating class imbalance...")

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"Non-default: {negative:,}")
print(f"Default:     {positive:,}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# ============================================================
# 7. TRAIN XGBOOST
# ============================================================

print("\n[7] Training XGBoost PD model...")
print("-" * 70)

model = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",
    eval_metric="auc",

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    verbose=50
)


# ============================================================
# 8. PREDICT PROBABILITY OF DEFAULT
# ============================================================

print("\n[8] Generating PD predictions...")

pd_valid = model.predict_proba(X_valid)[:, 1]


# ============================================================
# 9. EVALUATION
# ============================================================

print("\n[9] MODEL EVALUATION")
print("=" * 70)

roc_auc = roc_auc_score(
    y_valid,
    pd_valid
)

pr_auc = average_precision_score(
    y_valid,
    pd_valid
)

brier = brier_score_loss(
    y_valid,
    pd_valid
)

print(f"ROC-AUC : {roc_auc:.6f}")
print(f"PR-AUC  : {pr_auc:.6f}")
print(f"Brier   : {brier:.6f}")


# ============================================================
# 10. BASIC CLASSIFICATION VIEW
# ============================================================

print("\n[10] Classification report @ 0.50 threshold")
print("-" * 70)

predicted_class = (
    pd_valid >= 0.50
).astype(int)

print(
    classification_report(
        y_valid,
        predicted_class,
        digits=4
    )
)


# ============================================================
# 11. PD DISTRIBUTION
# ============================================================

print("\n[11] PD distribution")
print("-" * 70)

print(
    pd.Series(pd_valid).describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

print("\n[12] Top feature importance")
print("-" * 70)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.head(25).to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

print("\n[13] Saving model...")

artifact = {
    "model": model,
    "features": X.columns.tolist(),
    "categorical_columns": categorical_columns,
    "threshold": 0.50,
    "random_state": 42
}

joblib.dump(
    artifact,
    MODEL_PATH
)

print(f"Model saved to:")
print(MODEL_PATH)


# ============================================================
# 14. SAVE VALIDATION PREDICTIONS
# ============================================================

print("\n[14] Saving validation predictions...")

validation_predictions = pd.DataFrame({
    "SK_ID_CURR": df.loc[
        X_valid.index,
        "SK_ID_CURR"
    ].values,
    "TARGET": y_valid.values,
    "PD": pd_valid
})

prediction_path = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "validation_predictions.parquet"
)

validation_predictions.to_parquet(
    prediction_path,
    index=False
)

print(f"Predictions saved to:")
print(prediction_path)


print("\n" + "=" * 70)
print("PD MODEL TRAINING COMPLETE")
print("=" * 70)
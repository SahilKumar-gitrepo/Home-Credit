import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from sklearn.calibration import calibration_curve


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "pd_xgboost.joblib"

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "validation_predictions.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PD MODEL — DETAILED EVALUATION")
print("=" * 70)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]

df = pd.read_parquet(PREDICTION_PATH)

y_true = df["TARGET"].astype(int)
pd_score = df["PD"].astype(float)


# ============================================================
# 1. BASIC METRICS
# ============================================================

print("\n[1] OVERALL METRICS")
print("-" * 70)

roc_auc = roc_auc_score(y_true, pd_score)
pr_auc = average_precision_score(y_true, pd_score)
brier = brier_score_loss(y_true, pd_score)

print(f"ROC-AUC : {roc_auc:.6f}")
print(f"PR-AUC  : {pr_auc:.6f}")
print(f"Brier   : {brier:.6f}")


# ============================================================
# 2. BASELINE PR-AUC
# ============================================================

default_rate = y_true.mean()

print("\n[2] CLASS IMBALANCE")
print("-" * 70)

print(f"Default rate: {default_rate:.4%}")
print(f"Baseline PR-AUC: {default_rate:.6f}")


# ============================================================
# 3. PD BANDS
# ============================================================

print("\n[3] PD RISK BANDS")
print("-" * 70)

bands = pd.cut(
    pd_score,
    bins=[
        -np.inf,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        np.inf
    ],
    labels=[
        "<5%",
        "5-10%",
        "10-20%",
        "20-30%",
        "30-50%",
        ">50%"
    ]
)

band_table = (
    pd.DataFrame({
        "band": bands,
        "target": y_true
    })
    .groupby("band", observed=False)
    .agg(
        applicants=("target", "size"),
        defaults=("target", "sum"),
        actual_default_rate=("target", "mean")
    )
)

band_table["share"] = (
    band_table["applicants"] /
    len(df)
)

print(band_table)


# ============================================================
# 4. TOP RISK APPLICANTS
# ============================================================

print("\n[4] HIGHEST PD APPLICANTS")
print("-" * 70)

top_risk = (
    df[
        [
            "SK_ID_CURR",
            "TARGET",
            "PD"
        ]
    ]
    .sort_values("PD", ascending=False)
    .head(20)
)

print(top_risk.to_string(index=False))


# ============================================================
# 5. LOWEST RISK APPLICANTS
# ============================================================

print("\n[5] LOWEST PD APPLICANTS")
print("-" * 70)

low_risk = (
    df[
        [
            "SK_ID_CURR",
            "TARGET",
            "PD"
        ]
    ]
    .sort_values("PD", ascending=True)
    .head(20)
)

print(low_risk.to_string(index=False))


# ============================================================
# 6. THRESHOLD ANALYSIS
# ============================================================

print("\n[6] THRESHOLD ANALYSIS")
print("-" * 70)

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]

rows = []

for threshold in thresholds:

    predicted = (
        pd_score >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1]
    ).ravel()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    approval_rate = (
        predicted == 0
    ).mean()

    rows.append({
        "threshold": threshold,
        "approval_rate": approval_rate,
        "precision": precision,
        "recall": recall,
        "false_positives": fp,
        "false_negatives": fn
    })

threshold_table = pd.DataFrame(rows)

print(
    threshold_table.to_string(
        index=False,
        formatters={
            "approval_rate": "{:.2%}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format
        }
    )
)


# ============================================================
# 7. CALIBRATION
# ============================================================

print("\n[7] CALIBRATION")
print("-" * 70)

prob_true, prob_pred = calibration_curve(
    y_true,
    pd_score,
    n_bins=10,
    strategy="quantile"
)

calibration_table = pd.DataFrame({
    "mean_predicted_PD": prob_pred,
    "actual_default_rate": prob_true
})

calibration_table["difference"] = (
    calibration_table["actual_default_rate"]
    - calibration_table["mean_predicted_PD"]
)

print(calibration_table.to_string(index=False))


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

print("\n[8] TOP 30 FEATURES")
print("-" * 70)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.head(30).to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "pd_evaluation.csv"
)

combined = pd.DataFrame({
    "SK_ID_CURR": df["SK_ID_CURR"],
    "TARGET": y_true,
    "PD": pd_score
})

combined.to_csv(
    output_path,
    index=False
)

print("\nEvaluation file saved:")
print(output_path)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
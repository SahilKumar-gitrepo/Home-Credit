import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "pd_xgboost.joblib"

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "explanations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HOME CREDIT — SHAP PD EXPLAINABILITY")
print("=" * 70)

print("\n[1] Loading model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_names = artifact["features"]

print(f"Features expected by model: {len(feature_names)}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2] Loading feature data...")

df = pd.read_parquet(FEATURE_FILE)

print(f"Rows: {len(df):,}")


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\n[3] Preparing features...")

X = df.drop(
    columns=[
        c for c in ["TARGET", "SK_ID_CURR"]
        if c in df.columns
    ]
)

# Make sure column order matches training
X = X[feature_names]

categorical_columns = artifact.get(
    "categorical_columns",
    []
)

for col in categorical_columns:
    if col in X.columns:
        X[col] = X[col].astype("category").cat.codes

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SELECT SAMPLE
# ============================================================

print("\n[4] Selecting applicants for explanation...")

# Explain a manageable sample first.
# SHAP on the complete 307k rows is unnecessary at this stage.

SAMPLE_SIZE = min(2000, len(X))

sample_indices = (
    X.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).index
)

X_sample = X.loc[sample_indices]

print(f"Applicants explained: {len(X_sample):,}")


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\n[5] Creating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("\n[6] Calculating SHAP values...")

shap_values = explainer.shap_values(
    X_sample
)

# Binary XGBoost models normally return:
# rows × features

shap_values = np.asarray(shap_values)

print(
    f"SHAP matrix shape: {shap_values.shape}"
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n[7] Calculating global feature importance...")

mean_abs_shap = np.abs(shap_values).mean(axis=0)

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

global_importance = (
    global_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

print("\nTOP 30 FEATURES BY SHAP IMPORTANCE")
print("-" * 70)

print(
    global_importance
    .head(30)
    .to_string(index=False)
)


# ============================================================
# SAVE GLOBAL IMPORTANCE
# ============================================================

global_path = (
    OUTPUT_DIR
    / "global_shap_importance.csv"
)

global_importance.to_csv(
    global_path,
    index=False
)

print(
    f"\nGlobal importance saved: {global_path}"
)


# ============================================================
# APPLICANT-LEVEL EXPLANATIONS
# ============================================================

print("\n[8] Creating applicant-level explanations...")

pd_predictions = model.predict_proba(
    X_sample
)[:, 1]

explanation_rows = []

for row_position, row_index in enumerate(
    X_sample.index
):

    applicant_id = df.loc[
        row_index,
        "SK_ID_CURR"
    ]

    target = (
        df.loc[
            row_index,
            "TARGET"
        ]
        if "TARGET" in df.columns
        else np.nan
    )

    values = shap_values[row_position]

    top_indices = np.argsort(
        np.abs(values)
    )[::-1][:10]

    for rank, feature_index in enumerate(
        top_indices,
        start=1
    ):

        feature = feature_names[
            feature_index
        ]

        feature_value = X_sample.iloc[
            row_position,
            feature_index
        ]

        shap_value = values[
            feature_index
        ]

        explanation_rows.append({
            "SK_ID_CURR": applicant_id,
            "TARGET": target,
            "PD": pd_predictions[row_position],
            "rank": rank,
            "feature": feature,
            "feature_value": feature_value,
            "shap_value": shap_value,
            "impact": (
                "increases PD"
                if shap_value > 0
                else "decreases PD"
            )
        })


applicant_explanations = pd.DataFrame(
    explanation_rows
)


# ============================================================
# SAVE APPLICANT EXPLANATIONS
# ============================================================

applicant_path = (
    OUTPUT_DIR
    / "applicant_shap_explanations.parquet"
)

applicant_explanations.to_parquet(
    applicant_path,
    index=False
)

print(
    f"Applicant explanations saved: "
    f"{applicant_path}"
)


# ============================================================
# SHOW ONE HIGH-RISK APPLICANT
# ============================================================

print("\n[9] Example high-risk applicant")
print("-" * 70)

highest_risk_index = np.argmax(
    pd_predictions
)

high_risk_applicant = df.loc[
    X_sample.index[highest_risk_index]
]

high_risk_id = high_risk_applicant[
    "SK_ID_CURR"
]

high_risk_pd = pd_predictions[
    highest_risk_index
]

print(f"Applicant ID: {high_risk_id}")
print(f"Predicted PD: {high_risk_pd:.4%}")

example = applicant_explanations[
    applicant_explanations[
        "SK_ID_CURR"
    ] == high_risk_id
]

print(
    example[
        [
            "rank",
            "feature",
            "feature_value",
            "shap_value",
            "impact"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY COMPLETE")
print("=" * 70)
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pd_xgboost.joblib"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "decisions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "underwriting_decisions.parquet"
)


# ============================================================
# RESEARCH POLICY
# ============================================================

# IMPORTANT:
# These thresholds are ONLY for the research prototype.
# They are NOT real-world lending policy.

PD_APPROVE_MAX = 0.10
PD_REFER_MAX = 0.20

MIN_INCOME = 50000

MAX_CREDIT_TO_INCOME = 10.0
MAX_CC_UTILIZATION = 1.50


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("AI UNDERWRITING — DECISION ENGINE")
print("=" * 70)

print("\n[1] Loading model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_names = artifact["features"]
categorical_columns = artifact.get(
    "categorical_columns",
    []
)

print(f"Model features: {len(feature_names)}")


print("\n[2] Loading applicant features...")

df = pd.read_parquet(FEATURE_FILE)

print(f"Applicants: {len(df):,}")


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

print("\n[3] Preparing model input...")

X = df.drop(
    columns=[
        c for c in ["TARGET", "SK_ID_CURR"]
        if c in df.columns
    ]
)

X = X[feature_names]

for col in categorical_columns:

    if col in X.columns:

        X[col] = (
            X[col]
            .astype("category")
            .cat.codes
        )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# PREDICT PD
# ============================================================

print("\n[4] Predicting Probability of Default...")

pd_score = model.predict_proba(X)[:, 1]

df["PD"] = pd_score


# ============================================================
# RISK TIER
# ============================================================

print("\n[5] Creating risk tiers...")


def risk_tier(pd_value):

    if pd_value < 0.05:
        return "LOW"

    elif pd_value < 0.10:
        return "MODERATE"

    elif pd_value < 0.20:
        return "HIGH"

    else:
        return "VERY_HIGH"


df["risk_tier"] = df["PD"].apply(
    risk_tier
)


# ============================================================
# DECISION ENGINE
# ============================================================

print("\n[6] Applying underwriting policy...")


def make_decision(row):

    reasons = []

    # --------------------------------------------------------
    # PD RULE
    # --------------------------------------------------------

    pd_value = row["PD"]

    if pd_value >= PD_REFER_MAX:

        reasons.append(
            "Probability of default exceeds "
            "research decline threshold"
        )

        return "DECLINE", reasons

    # --------------------------------------------------------
    # DATA QUALITY / AFFORDABILITY
    # --------------------------------------------------------

    income = row.get(
        "AMT_INCOME_TOTAL",
        np.nan
    )

    credit = row.get(
        "AMT_CREDIT",
        np.nan
    )

    if pd.isna(income):

        reasons.append(
            "Income information unavailable"
        )

        return "REFER", reasons

    if income <= 0:

        reasons.append(
            "Invalid or non-positive income"
        )

        return "REFER", reasons

    if income < MIN_INCOME:

        reasons.append(
            "Income below research minimum"
        )

        return "REFER", reasons

    # --------------------------------------------------------
    # CREDIT / INCOME
    # --------------------------------------------------------

    credit_to_income = row.get(
        "credit_to_income",
        np.nan
    )

    if (
        not pd.isna(credit_to_income)
        and credit_to_income > MAX_CREDIT_TO_INCOME
    ):

        reasons.append(
            "Requested credit is high relative "
            "to reported income"
        )

        return "REFER", reasons

    # --------------------------------------------------------
    # CREDIT CARD UTILIZATION
    # --------------------------------------------------------

    utilization = row.get(
        "cc_utilization",
        np.nan
    )

    if (
        not pd.isna(utilization)
        and utilization > MAX_CC_UTILIZATION
    ):

        reasons.append(
            "Credit utilization is elevated"
        )

        return "REFER", reasons

    # --------------------------------------------------------
    # PD-BASED DECISION
    # --------------------------------------------------------

    if pd_value < PD_APPROVE_MAX:

        reasons.append(
            "Predicted probability of default "
            "is below research approval threshold"
        )

        return "APPROVE", reasons

    else:

        reasons.append(
            "Probability of default falls within "
            "manual-review range"
        )

        return "REFER", reasons


# ============================================================
# APPLY DECISION
# ============================================================

results = df.apply(
    make_decision,
    axis=1
)

df["decision"] = results.apply(
    lambda x: x[0]
)

df["policy_reason"] = results.apply(
    lambda x: x[1][0]
)


# ============================================================
# DECISION SUMMARY
# ============================================================

print("\n[7] DECISION SUMMARY")
print("-" * 70)

summary = (
    df["decision"]
    .value_counts()
)

print(summary)

print("\nDecision percentages:")

print(
    (
        df["decision"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# RISK SUMMARY
# ============================================================

print("\n[8] RISK TIER SUMMARY")
print("-" * 70)

risk_summary = (
    df["risk_tier"]
    .value_counts()
)

print(risk_summary)


# ============================================================
# SAMPLE DECISIONS
# ============================================================

print("\n[9] SAMPLE UNDERWRITING DECISIONS")
print("-" * 70)

sample_columns = [
    "SK_ID_CURR",
    "PD",
    "risk_tier",
    "decision",
    "policy_reason"
]

sample_columns = [
    c for c in sample_columns
    if c in df.columns
]

print(
    df[
        sample_columns
    ]
    .sort_values(
        "PD",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

print("\n[10] Saving decisions...")

output_columns = [
    "SK_ID_CURR",
    "PD",
    "risk_tier",
    "decision",
    "policy_reason"
]

if "TARGET" in df.columns:
    output_columns.insert(
        1,
        "TARGET"
    )

output_columns = [
    c for c in output_columns
    if c in df.columns
]

decision_df = df[
    output_columns
]

decision_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)


print("\n" + "=" * 70)
print("DECISION ENGINE COMPLETE")
print("=" * 70)
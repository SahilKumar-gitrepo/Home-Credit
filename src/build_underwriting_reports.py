import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DECISION_FILE = (
    PROJECT_ROOT
    / "data"
    / "decisions"
    / "underwriting_decisions.parquet"
)

SHAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "explanations"
    / "applicant_shap_explanations.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "underwriting_reports.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("AI UNDERWRITING — REPORT GENERATION")
print("=" * 70)

print("\n[1] Loading decisions...")

decisions = pd.read_parquet(
    DECISION_FILE
)

print(
    f"Decision records: {len(decisions):,}"
)

print("\n[2] Loading SHAP explanations...")

shap_df = pd.read_parquet(
    SHAP_FILE
)

print(
    f"SHAP records: {len(shap_df):,}"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_feature_name(name):
    """
    Convert technical feature names into readable text.
    """

    replacements = {
        "AMT_INCOME_TOTAL": "total income",
        "AMT_CREDIT": "requested credit amount",
        "AMT_ANNUITY": "annuity amount",
        "DAYS_BIRTH": "age",
        "DAYS_EMPLOYED": "employment duration",
        "EXT_SOURCE_1": "external credit indicator 1",
        "EXT_SOURCE_2": "external credit indicator 2",
        "EXT_SOURCE_3": "external credit indicator 3",
        "credit_to_income": "credit-to-income ratio",
        "annuity_to_income": "annuity-to-income ratio",
        "bureau_debt_to_credit_ratio": "bureau debt-to-credit ratio",
        "payment_ratio": "historical payment ratio",
        "underpayment_rate": "underpayment rate",
        "avg_payment_delay": "average payment delay",
        "max_payment_delay": "maximum payment delay",
        "cc_utilization": "credit-card utilization",
    }

    return replacements.get(
        name,
        name.replace("_", " ")
    )


def create_shap_reason(row):
    """
    Convert a SHAP contribution into a human-readable reason.
    """

    feature = format_feature_name(
        row["feature"]
    )

    shap_value = row["shap_value"]

    if shap_value > 0:
        return (
            f"{feature} increased the predicted "
            f"probability of default"
        )

    return (
        f"{feature} decreased the predicted "
        f"probability of default"
    )


# ============================================================
# CREATE REPORTS
# ============================================================

print("\n[3] Building applicant reports...")


reports = []

applicant_ids = decisions[
    "SK_ID_CURR"
].unique()


for applicant_id in applicant_ids:

    decision_row = decisions[
        decisions["SK_ID_CURR"]
        == applicant_id
    ]

    if decision_row.empty:
        continue

    decision_row = decision_row.iloc[0]

    applicant_shap = shap_df[
        shap_df["SK_ID_CURR"]
        == applicant_id
    ].copy()

    applicant_shap = applicant_shap.sort_values(
        "rank"
    )

    # --------------------------------------------------------
    # Separate positive / negative contributors
    # --------------------------------------------------------

    positive = applicant_shap[
        applicant_shap["shap_value"] > 0
    ].head(5)

    negative = applicant_shap[
        applicant_shap["shap_value"] < 0
    ].head(5)

    positive_reasons = [
        create_shap_reason(row)
        for _, row in positive.iterrows()
    ]

    negative_reasons = [
        create_shap_reason(row)
        for _, row in negative.iterrows()
    ]

    # --------------------------------------------------------
    # Top features
    # --------------------------------------------------------

    top_features = applicant_shap.head(10)

    top_feature_names = [
        format_feature_name(x)
        for x in top_features["feature"]
    ]

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = {
        "SK_ID_CURR": applicant_id,

        "TARGET": decision_row.get(
            "TARGET",
            np.nan
        ),

        "PD": float(
            decision_row["PD"]
        ),

        "risk_tier": decision_row[
            "risk_tier"
        ],

        "decision": decision_row[
            "decision"
        ],

        "policy_reason": decision_row[
            "policy_reason"
        ],

        "top_features": " | ".join(
            top_feature_names
        ),

        "risk_increasing_factors":
            " | ".join(
                positive_reasons
            ),

        "risk_reducing_factors":
            " | ".join(
                negative_reasons
            )
    }

    reports.append(report)


# ============================================================
# DATAFRAME
# ============================================================

print("\n[4] Creating report dataset...")

reports_df = pd.DataFrame(
    reports
)

print(
    f"Reports created: {len(reports_df):,}"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n[5] Validating reports...")

print(
    f"Unique applicants: "
    f"{reports_df['SK_ID_CURR'].nunique():,}"
)

duplicate_count = (
    reports_df["SK_ID_CURR"].duplicated().sum()
)

print(
    f"Duplicate reports: {duplicate_count}"
)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate applicant reports detected."
    )


# ============================================================
# SAVE
# ============================================================

print("\n[6] Saving underwriting reports...")

reports_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)


# ============================================================
# SHOW EXAMPLE
# ============================================================

print("\n[7] EXAMPLE UNDERWRITING REPORT")
print("-" * 70)

example = reports_df.sort_values(
    "PD",
    ascending=False
).iloc[0]

print(
    f"Applicant ID : {example['SK_ID_CURR']}"
)

print(
    f"PD           : {example['PD']:.2%}"
)

print(
    f"Risk Tier    : {example['risk_tier']}"
)

print(
    f"Decision     : {example['decision']}"
)

print(
    f"Policy reason: {example['policy_reason']}"
)

print("\nRisk-increasing factors:")

for reason in str(
    example["risk_increasing_factors"]
).split(" | "):

    if reason.strip():
        print(f"  • {reason}")

print("\nRisk-reducing factors:")

for reason in str(
    example["risk_reducing_factors"]
).split(" | "):

    if reason.strip():
        print(f"  • {reason}")


print("\n" + "=" * 70)
print("UNDERWRITING REPORT GENERATION COMPLETE")
print("=" * 70)
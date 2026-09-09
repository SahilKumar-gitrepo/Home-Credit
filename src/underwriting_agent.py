"""
Deterministic Credit Underwriting Workflow

This module:
1. Loads the trained XGBoost PD model
2. Loads applicant features
3. Calculates Probability of Default (PD)
4. Assigns a risk tier
5. Applies deterministic underwriting policy
6. Generates SHAP explanations
7. Caches SHAP explanations
8. Returns a structured underwriting assessment

IMPORTANT:
The deterministic workflow makes the actual decision.
The LLM/agent only interprets and presents the evidence.
"""


import json
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
import shap

from langgraph.graph import StateGraph, END


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_features.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pd_xgboost.joblib"
)

SHAP_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "explanations"
    / "on_demand_shap.parquet"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading credit-risk model...")

model_artifact = joblib.load(
    MODEL_PATH
)


if isinstance(model_artifact, dict):

    model = model_artifact["model"]

    MODEL_FEATURES = model_artifact.get(
        "features",
        None
    )

    CATEGORICAL_COLUMNS = model_artifact.get(
        "categorical_columns",
        []
    )

else:

    model = model_artifact

    MODEL_FEATURES = None

    CATEGORICAL_COLUMNS = []


print(
    f"Model loaded: {type(model).__name__}"
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("Loading model features...")

features_df = pd.read_parquet(
    FEATURE_PATH
)

print(
    f"Features loaded: {len(features_df):,} applicants"
)


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(df):
    """
    Prepare applicant data for the trained model.
    """

    X = df.copy()

    # --------------------------------------------------------
    # Remove target
    # --------------------------------------------------------

    if "TARGET" in X.columns:

        X = X.drop(
            columns=["TARGET"]
        )

    # --------------------------------------------------------
    # Remove applicant ID
    # --------------------------------------------------------

    if "SK_ID_CURR" in X.columns:

        X = X.drop(
            columns=["SK_ID_CURR"]
        )

    # --------------------------------------------------------
    # Remove application date
    # --------------------------------------------------------

    if "application_date" in X.columns:

        X = X.drop(
            columns=["application_date"]
        )

    # --------------------------------------------------------
    # Use exactly the features used during training
    # --------------------------------------------------------

    if MODEL_FEATURES is not None:

        missing_features = [
            feature
            for feature in MODEL_FEATURES
            if feature not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing model features: "
                + str(missing_features)
            )

        X = X[
            MODEL_FEATURES
        ]

    # --------------------------------------------------------
    # Convert known categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        if column in X.columns:

            X[column] = (
                X[column]
                .astype("category")
                .cat.codes
            )

    # --------------------------------------------------------
    # Convert any remaining object/category columns
    # --------------------------------------------------------

    for column in X.columns:

        if (
            X[column].dtype == "object"
            or str(X[column].dtype) == "category"
        ):

            X[column] = (
                X[column]
                .astype("category")
                .cat.codes
            )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X


# ============================================================
# GET APPLICANT
# ============================================================

def get_applicant(applicant_id):
    """
    Retrieve one applicant from the feature dataset.
    """

    applicant = features_df[
        features_df["SK_ID_CURR"]
        == applicant_id
    ]

    if applicant.empty:

        raise ValueError(
            f"Applicant {applicant_id} "
            "was not found."
        )

    return applicant.copy()


# ============================================================
# CALCULATE PD
# ============================================================

def calculate_pd(applicant):
    """
    Calculate Probability of Default.
    """

    X = prepare_model_input(
        applicant
    )

    probability = model.predict_proba(
        X
    )[:, 1][0]

    return float(
        probability
    )


# ============================================================
# RISK TIER
# ============================================================

def get_risk_tier(pd_value):
    """
    Convert PD into a research risk tier.
    """

    if pd_value < 0.05:

        return "LOW"

    elif pd_value < 0.10:

        return "MODERATE"

    elif pd_value < 0.20:

        return "HIGH"

    else:

        return "VERY_HIGH"


# ============================================================
# DETERMINISTIC DECISION POLICY
# ============================================================

def make_decision(
    applicant,
    pd_value
):
    """
    Research underwriting policy.

    Rules:

    PD >= 20%
        -> DECLINE

    Missing income
        -> REFER

    Invalid income
        -> REFER

    Income < 50,000
        -> REFER

    Credit-to-income > 10
        -> REFER

    Maximum credit-card utilization > 1.5
        -> REFER

    PD < 10%
        -> APPROVE

    Otherwise
        -> REFER
    """

    # --------------------------------------------------------
    # Rule 1: High PD
    # --------------------------------------------------------

    if pd_value >= 0.20:

        return (
            "DECLINE",
            "Predicted probability of default "
            "is at or above the research decline threshold"
        )

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    income = applicant[
        "AMT_INCOME_TOTAL"
    ].iloc[0]

    if pd.isna(income):

        return (
            "REFER",
            "Applicant income is missing"
        )

    if income <= 0:

        return (
            "REFER",
            "Applicant income is invalid"
        )

    # --------------------------------------------------------
    # Low income
    # --------------------------------------------------------

    if income < 50000:

        return (
            "REFER",
            "Applicant income is below "
            "the research policy threshold"
        )

    # --------------------------------------------------------
    # Credit-to-income
    # --------------------------------------------------------

    credit_to_income = applicant[
        "credit_to_income"
    ].iloc[0]

    if (
        pd.notna(credit_to_income)
        and credit_to_income > 10
    ):

        return (
            "REFER",
            "Credit-to-income ratio exceeds "
            "the research policy threshold"
        )

    # --------------------------------------------------------
    # Credit-card utilization
    # --------------------------------------------------------

    cc_utilization = applicant[
        "max_cc_utilization"
    ].iloc[0]

    if (
        pd.notna(cc_utilization)
        and cc_utilization > 1.5
    ):

        return (
            "REFER",
            "Credit-card utilization exceeds "
            "the research policy threshold"
        )

    # --------------------------------------------------------
    # Approval
    # --------------------------------------------------------

    if pd_value < 0.10:

        return (
            "APPROVE",
            "Predicted probability of default "
            "is below research approval threshold"
        )

    # --------------------------------------------------------
    # Manual review
    # --------------------------------------------------------

    return (
        "REFER",
        "Predicted probability of default "
        "requires manual review"
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def calculate_shap_explanation(
    applicant,
    top_n=5
):
    """
    Calculate SHAP explanation for one applicant.

    Positive SHAP:
        pushes model output upward.

    Negative SHAP:
        pushes model output downward.

    SHAP values describe model behavior and
    do not establish causality.
    """

    X = prepare_model_input(
        applicant
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    # --------------------------------------------------------
    # Handle SHAP output formats
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        shap_row = np.asarray(
            shap_values[-1]
        )[0]

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            shap_row = shap_array[
                0,
                :,
                -1
            ]

        elif shap_array.ndim == 2:

            shap_row = shap_array[
                0
            ]

        else:

            shap_row = shap_array

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = list(
        X.columns
    )

    feature_values = X.iloc[0]

    explanation = []

    for index, feature in enumerate(
        feature_names
    ):

        shap_value = float(
            shap_row[index]
        )

        # IMPORTANT:
        # Get the ORIGINAL value from the applicant,
        # not the encoded value sent to XGBoost.
        if feature in applicant.columns:

            original_value = applicant[
                feature
            ].iloc[0]

        else:

            original_value = feature_values.iloc[
                index
            ]

        explanation.append({

            "feature":
                feature,

            "value":
                original_value,

            "shap":
                shap_value
        })

    # --------------------------------------------------------
    # Sort by absolute SHAP contribution
    # --------------------------------------------------------

    explanation.sort(
        key=lambda item:
            abs(item["shap"]),
        reverse=True
    )

    # --------------------------------------------------------
    # Positive contributions
    # --------------------------------------------------------

    risk_increasing = [
        item
        for item in explanation
        if item["shap"] > 0
    ][:top_n]

    # --------------------------------------------------------
    # Negative contributions
    # --------------------------------------------------------

    risk_reducing = [
        item
        for item in explanation
        if item["shap"] < 0
    ][:top_n]

    return (
        risk_increasing,
        risk_reducing
    )


# ============================================================
# SAVE SHAP CACHE
# ============================================================

def save_shap_cache(
    applicant_id,
    risk_increasing,
    risk_reducing
):
    """
    Save on-demand SHAP explanations.

    Feature values may be numeric or categorical strings.

    Therefore:
        value -> string
        shap  -> float

    This prevents PyArrow schema conflicts such as:

        Could not convert 'Higher education'
        with type str: tried to convert to double
    """

    SHAP_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    # --------------------------------------------------------
    # Risk-increasing factors
    # --------------------------------------------------------

    for item in risk_increasing:

        rows.append({

            "SK_ID_CURR":
                int(applicant_id),

            "feature":
                str(item["feature"]),

            "value":
                str(item["value"]),

            "shap":
                float(item["shap"]),

            "direction":
                "risk_increasing"
        })

    # --------------------------------------------------------
    # Risk-reducing factors
    # --------------------------------------------------------

    for item in risk_reducing:

        rows.append({

            "SK_ID_CURR":
                int(applicant_id),

            "feature":
                str(item["feature"]),

            "value":
                str(item["value"]),

            "shap":
                float(item["shap"]),

            "direction":
                "risk_reducing"
        })

    new_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Explicit schema
    # --------------------------------------------------------

    new_df["SK_ID_CURR"] = (
        new_df["SK_ID_CURR"]
        .astype("int64")
    )

    new_df["feature"] = (
        new_df["feature"]
        .astype("string")
    )

    new_df["value"] = (
        new_df["value"]
        .astype("string")
    )

    new_df["shap"] = pd.to_numeric(
        new_df["shap"],
        errors="coerce"
    ).astype("float64")

    new_df["direction"] = (
        new_df["direction"]
        .astype("string")
    )

    # --------------------------------------------------------
    # Existing cache
    # --------------------------------------------------------

    if SHAP_CACHE_PATH.exists():

        try:

            old_df = pd.read_parquet(
                SHAP_CACHE_PATH
            )

            # Remove old entries for this applicant.
            old_df = old_df[
                old_df["SK_ID_CURR"]
                != applicant_id
            ].copy()

            # Normalize schema.
            old_df["SK_ID_CURR"] = (
                old_df["SK_ID_CURR"]
                .astype("int64")
            )

            old_df["feature"] = (
                old_df["feature"]
                .astype("string")
            )

            old_df["value"] = (
                old_df["value"]
                .astype("string")
            )

            old_df["shap"] = pd.to_numeric(
                old_df["shap"],
                errors="coerce"
            ).astype("float64")

            old_df["direction"] = (
                old_df["direction"]
                .astype("string")
            )

            new_df = pd.concat(
                [
                    old_df,
                    new_df
                ],
                ignore_index=True
            )

        except Exception as e:

            print(
                "Existing SHAP cache could not be read."
            )

            print(
                f"Rebuilding cache: {e}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    new_df.to_parquet(
        SHAP_CACHE_PATH,
        index=False
    )


# ============================================================
# COMPLETE UNDERWRITING WORKFLOW
# ============================================================

def run_underwriting_workflow(
    applicant_id
):
    """
    Run the complete deterministic underwriting workflow.

    Returns a structured result containing:

        applicant_id
        probability_of_default
        risk_tier
        decision
        policy_reason
        risk_increasing_factors
        risk_reducing_factors
    """

    # --------------------------------------------------------
    # 1. Retrieve applicant
    # --------------------------------------------------------

    applicant = get_applicant(
        applicant_id
    )

    # --------------------------------------------------------
    # 2. Probability of Default
    # --------------------------------------------------------

    pd_value = calculate_pd(
        applicant
    )

    # --------------------------------------------------------
    # 3. Risk tier
    # --------------------------------------------------------

    risk_tier = get_risk_tier(
        pd_value
    )

    # --------------------------------------------------------
    # 4. Deterministic decision
    # --------------------------------------------------------

    (
        decision,
        policy_reason
    ) = make_decision(
        applicant,
        pd_value
    )

    # --------------------------------------------------------
    # 5. SHAP explanation
    # --------------------------------------------------------

    (
        risk_increasing,
        risk_reducing
    ) = calculate_shap_explanation(
        applicant
    )

    # --------------------------------------------------------
    # 6. Save SHAP cache
    # --------------------------------------------------------

    save_shap_cache(
        applicant_id,
        risk_increasing,
        risk_reducing
    )

    # --------------------------------------------------------
    # 7. Build assessment
    # --------------------------------------------------------

    assessment = {

        "applicant_id":
            int(applicant_id),

        "probability_of_default":
            pd_value,

        "risk_tier":
            risk_tier,

        "decision":
            decision,

        "policy_reason":
            policy_reason,

        "risk_increasing_factors":
            risk_increasing,

        "risk_reducing_factors":
            risk_reducing
    }

    return {

        "applicant_id":
            int(applicant_id),

        "final_assessment":
            assessment
    }


# ============================================================
# LANGGRAPH STATE
# ============================================================

class UnderwritingState(
    TypedDict,
    total=False
):

    applicant_id: int

    applicant: dict

    probability_of_default: float

    risk_tier: str

    decision: str

    policy_reason: str

    risk_increasing_factors: list

    risk_reducing_factors: list

    final_assessment: dict


# ============================================================
# LANGGRAPH NODE — LOAD APPLICANT
# ============================================================

def load_applicant_node(
    state
):

    applicant_id = state[
        "applicant_id"
    ]

    applicant = get_applicant(
        applicant_id
    )

    return {

        "applicant":
            applicant.iloc[0].to_dict()
    }


# ============================================================
# LANGGRAPH NODE — PD
# ============================================================

def get_pd_node(
    state
):

    applicant_id = state[
        "applicant_id"
    ]

    applicant = get_applicant(
        applicant_id
    )

    pd_value = calculate_pd(
        applicant
    )

    return {

        "probability_of_default":
            pd_value
    }


# ============================================================
# LANGGRAPH NODE — DECISION
# ============================================================

def get_decision_node(
    state
):

    applicant_id = state[
        "applicant_id"
    ]

    applicant = get_applicant(
        applicant_id
    )

    pd_value = state[
        "probability_of_default"
    ]

    risk_tier = get_risk_tier(
        pd_value
    )

    (
        decision,
        policy_reason
    ) = make_decision(
        applicant,
        pd_value
    )

    return {

        "risk_tier":
            risk_tier,

        "decision":
            decision,

        "policy_reason":
            policy_reason
    }


# ============================================================
# LANGGRAPH NODE — SHAP
# ============================================================

def get_shap_node(
    state
):

    applicant_id = state[
        "applicant_id"
    ]

    applicant = get_applicant(
        applicant_id
    )

    (
        risk_increasing,
        risk_reducing
    ) = calculate_shap_explanation(
        applicant
    )

    save_shap_cache(
        applicant_id,
        risk_increasing,
        risk_reducing
    )

    return {

        "risk_increasing_factors":
            risk_increasing,

        "risk_reducing_factors":
            risk_reducing
    }


# ============================================================
# LANGGRAPH NODE — BUILD FINAL ASSESSMENT
# ============================================================

def build_assessment_node(
    state
):

    assessment = {

        "applicant_id":
            int(state["applicant_id"]),

        "probability_of_default":
            state[
                "probability_of_default"
            ],

        "risk_tier":
            state[
                "risk_tier"
            ],

        "decision":
            state[
                "decision"
            ],

        "policy_reason":
            state[
                "policy_reason"
            ],

        "risk_increasing_factors":
            state.get(
                "risk_increasing_factors",
                []
            ),

        "risk_reducing_factors":
            state.get(
                "risk_reducing_factors",
                []
            )
    }

    return {

        "final_assessment":
            assessment
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow_graph = StateGraph(
    UnderwritingState
)

workflow_graph.add_node(
    "load_applicant",
    load_applicant_node
)

workflow_graph.add_node(
    "get_pd",
    get_pd_node
)

workflow_graph.add_node(
    "get_decision",
    get_decision_node
)

workflow_graph.add_node(
    "get_shap",
    get_shap_node
)

workflow_graph.add_node(
    "build_assessment",
    build_assessment_node
)


workflow_graph.set_entry_point(
    "load_applicant"
)

workflow_graph.add_edge(
    "load_applicant",
    "get_pd"
)

workflow_graph.add_edge(
    "get_pd",
    "get_decision"
)

workflow_graph.add_edge(
    "get_decision",
    "get_shap"
)

workflow_graph.add_edge(
    "get_shap",
    "build_assessment"
)

workflow_graph.add_edge(
    "build_assessment",
    END
)


workflow_app = workflow_graph.compile()


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    applicant_id = 370920

    print()
    print("=" * 70)
    print("DETERMINISTIC CREDIT UNDERWRITING")
    print("=" * 70)

    print()
    print(
        f"Testing applicant: {applicant_id}"
    )

    print()
    print("Running underwriting workflow...")

    result = run_underwriting_workflow(
        applicant_id
    )

    assessment = result[
        "final_assessment"
    ]

    print()
    print("=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    print()

    print(
        f"Applicant: "
        f"{assessment['applicant_id']}"
    )

    print(
        f"Probability of Default: "
        f"{assessment['probability_of_default'] * 100:.2f}%"
    )

    print(
        f"Risk Tier: "
        f"{assessment['risk_tier']}"
    )

    print(
        f"Decision: "
        f"{assessment['decision']}"
    )

    print(
        f"Policy Reason: "
        f"{assessment['policy_reason']}"
    )

    print()
    print(
        "Risk-increasing model factors:"
    )

    for factor in assessment[
        "risk_increasing_factors"
    ]:

        print(
            f"  - {factor['feature']}: "
            f"{factor['value']} "
            f"(SHAP: {factor['shap']:.4f})"
        )

    print()
    print(
        "Risk-reducing model factors:"
    )

    for factor in assessment[
        "risk_reducing_factors"
    ]:

        print(
            f"  - {factor['feature']}: "
            f"{factor['value']} "
            f"(SHAP: {factor['shap']:.4f})"
        )

    print()
    print("=" * 70)
    print("UNDERWRITING WORKFLOW COMPLETE")
    print("=" * 70)
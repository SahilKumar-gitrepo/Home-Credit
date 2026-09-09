"""
predict_test.py
---------------
Run the trained PD model on application_test (no TARGET column).

Outputs
-------
data/features/test_predictions.parquet  -- SK_ID_CURR + PD score + risk band
data/features/submission.csv            -- Kaggle submission format (SK_ID_CURR, TARGET)
"""

from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH    = PROJECT_ROOT / "data" / "home_credit.duckdb"
MODEL_PATH = PROJECT_ROOT / "models" / "pd_xgboost_calibrated.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "models" / "pd_xgboost.joblib"

FEATURE_DIR   = PROJECT_ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH       = FEATURE_DIR / "test_predictions.parquet"
SUBMISSION_PATH = FEATURE_DIR / "submission.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("HOME CREDIT -- TEST SET INFERENCE")
print("=" * 70)

print("\n[1] Loading trained model...")

artifact       = joblib.load(MODEL_PATH)
model          = artifact["model"]
feature_names  = artifact["features"]
cat_cols       = artifact["categorical_columns"]
calibrator     = artifact.get("calibrator")

print(f"    Loaded model : {MODEL_PATH.name}")
print(f"    Calibrator   : {artifact.get('calibrator_type', 'None')}")
print(f"    Model expects {len(feature_names)} features")
print(f"    Categorical columns: {len(cat_cols)}")


# ============================================================
# BUILD TEST FEATURES (mirrors build_features.py)
# ============================================================

print("\n[2] Connecting to DuckDB...")
con = duckdb.connect(str(DB_PATH))

# ---- Application features (no TARGET) ----------------------

print("[3] Building application features for test set...")

con.execute("""
CREATE OR REPLACE TABLE test_application_features AS

SELECT
    SK_ID_CURR,

    -- Basic application information
    NAME_CONTRACT_TYPE,
    CODE_GENDER,
    FLAG_OWN_CAR,
    OWN_CAR_AGE,
    FLAG_OWN_REALTY,
    CNT_CHILDREN,
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    AMT_ANNUITY,
    AMT_GOODS_PRICE,

    -- Family / household
    NAME_FAMILY_STATUS,
    CNT_FAM_MEMBERS,

    -- Education / occupation
    NAME_EDUCATION_TYPE,
    NAME_INCOME_TYPE,
    OCCUPATION_TYPE,
    ORGANIZATION_TYPE,

    -- Housing
    NAME_HOUSING_TYPE,

    -- Employment / age
    DAYS_BIRTH,
    DAYS_EMPLOYED,
    DAYS_REGISTRATION,
    DAYS_ID_PUBLISH,

    -- External credit scores
    EXT_SOURCE_1,
    EXT_SOURCE_2,
    EXT_SOURCE_3,

    -- Contact / identity flags
    FLAG_MOBIL,
    FLAG_EMP_PHONE,
    FLAG_WORK_PHONE,
    FLAG_CONT_MOBILE,
    FLAG_PHONE,
    FLAG_EMAIL,

    -- Document flags
    FLAG_DOCUMENT_2,  FLAG_DOCUMENT_3,  FLAG_DOCUMENT_4,  FLAG_DOCUMENT_5,
    FLAG_DOCUMENT_6,  FLAG_DOCUMENT_7,  FLAG_DOCUMENT_8,  FLAG_DOCUMENT_9,
    FLAG_DOCUMENT_10, FLAG_DOCUMENT_11, FLAG_DOCUMENT_12, FLAG_DOCUMENT_13,
    FLAG_DOCUMENT_14, FLAG_DOCUMENT_15, FLAG_DOCUMENT_16, FLAG_DOCUMENT_17,
    FLAG_DOCUMENT_18, FLAG_DOCUMENT_19, FLAG_DOCUMENT_20, FLAG_DOCUMENT_21,

    -- Derived ratios
    CASE
        WHEN AMT_INCOME_TOTAL > 0
        THEN AMT_CREDIT / AMT_INCOME_TOTAL
        ELSE NULL
    END AS credit_to_income,

    CASE
        WHEN AMT_INCOME_TOTAL > 0
        THEN AMT_ANNUITY / AMT_INCOME_TOTAL
        ELSE NULL
    END AS annuity_to_income,

    CASE
        WHEN AMT_CREDIT > 0
        THEN AMT_ANNUITY / AMT_CREDIT
        ELSE NULL
    END AS annuity_to_credit,

    (
        EXT_SOURCE_1 +
        EXT_SOURCE_2 +
        EXT_SOURCE_3
    ) /
    NULLIF(
        (EXT_SOURCE_1 IS NOT NULL)::INTEGER +
        (EXT_SOURCE_2 IS NOT NULL)::INTEGER +
        (EXT_SOURCE_3 IS NOT NULL)::INTEGER,
        0
    ) AS ext_source_mean

FROM application_test
""")

# ---- Join pre-built supplementary feature tables -----------

print("[4] Joining bureau / installment / POS / CC / bureau-balance features...")

con.execute("""
CREATE OR REPLACE TABLE test_model_features AS

SELECT
    a.*,
    b.*   EXCLUDE (SK_ID_CURR),
    p.*   EXCLUDE (SK_ID_CURR),
    i.*   EXCLUDE (SK_ID_CURR),
    pos.* EXCLUDE (SK_ID_CURR),
    cc.*  EXCLUDE (SK_ID_CURR),
    bb.*  EXCLUDE (SK_ID_CURR)

FROM test_application_features a

LEFT JOIN bureau_features b
    ON a.SK_ID_CURR = b.SK_ID_CURR

LEFT JOIN previous_application_features p
    ON a.SK_ID_CURR = p.SK_ID_CURR

LEFT JOIN installment_features i
    ON a.SK_ID_CURR = i.SK_ID_CURR

LEFT JOIN pos_features pos
    ON a.SK_ID_CURR = pos.SK_ID_CURR

LEFT JOIN credit_card_features cc
    ON a.SK_ID_CURR = cc.SK_ID_CURR

LEFT JOIN bureau_balance_features bb
    ON a.SK_ID_CURR = bb.SK_ID_CURR
""")

# ---- Load into pandas ------------------------------------------

print("[5] Loading feature table into pandas...")

df_test = con.execute("SELECT * FROM test_model_features").df()
con.close()

print(f"    Rows    : {len(df_test):,}")
print(f"    Columns : {len(df_test.columns)}")

sk_ids = df_test["SK_ID_CURR"].copy()


# ============================================================
# ALIGN FEATURES TO TRAINING SCHEMA
# ============================================================

print("\n[6] Aligning features to training schema...")

X_test = pd.DataFrame(index=df_test.index)

missing_cols = []
for feat in feature_names:
    if feat in df_test.columns:
        X_test[feat] = df_test[feat]
    else:
        X_test[feat] = np.nan
        missing_cols.append(feat)

if missing_cols:
    print(f"    WARNING: {len(missing_cols)} features missing -- filled with NaN:")
    print(f"    {sorted(missing_cols)}")
else:
    print(f"    All {len(feature_names)} features found.")

# Re-encode categoricals the same way the model was trained
for col in cat_cols:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype("category").cat.codes

# Replace inf values
X_test = X_test.replace([np.inf, -np.inf], np.nan)

print(f"    Feature matrix shape: {X_test.shape}")


# ============================================================
# PREDICT
# ============================================================

print("\n[7] Generating PD predictions...")

pd_scores_raw = model.predict_proba(X_test)[:, 1]

if calibrator is not None:
    pd_scores = calibrator.predict(pd_scores_raw)
    pd_scores = np.clip(pd_scores, 0.0001, 0.9999)
    print(f"    Calibration applied: {artifact.get('calibrator_type', 'Isotonic')}")
    print(f"    Raw mean PD        : {pd_scores_raw.mean():.6f}")
else:
    pd_scores = pd_scores_raw

print(f"    Final Min PD       : {pd_scores.min():.6f}")
print(f"    Final Max PD       : {pd_scores.max():.6f}")
print(f"    Final Mean PD      : {pd_scores.mean():.6f}")
print(f"    Final Median       : {np.median(pd_scores):.6f}")
print(f"    Final Std Dev      : {pd_scores.std():.6f}")


# ============================================================
# RISK BANDING
# ============================================================

risk_bands = pd.cut(
    pd_scores,
    bins=[-np.inf, 0.05, 0.10, 0.20, 0.30, 0.50, np.inf],
    labels=["<5%", "5-10%", "10-20%", "20-30%", "30-50%", ">50%"]
)

band_counts = (
    pd.Series(risk_bands, name="risk_band")
    .value_counts()
    .reindex(["<5%", "5-10%", "10-20%", "20-30%", "30-50%", ">50%"])
    .fillna(0)
    .astype(int)
    .reset_index()
)
band_counts.columns = ["risk_band", "count"]
band_counts["share"] = band_counts["count"] / len(pd_scores)

print("\n[8] Risk band distribution:")
print("-" * 45)
print(
    band_counts.to_string(
        index=False,
        formatters={"share": "{:.2%}".format}
    )
)


# ============================================================
# SAVE TEST PREDICTIONS (richer parquet)
# ============================================================

print("\n[9] Saving test predictions parquet...")

test_predictions = pd.DataFrame({
    "SK_ID_CURR" : sk_ids.values,
    "PD"         : pd_scores,
    "PD_raw"     : pd_scores_raw,
    "risk_band"  : risk_bands.astype(str)
})

test_predictions.to_parquet(PRED_PATH, index=False)
print(f"    Saved: {PRED_PATH}")


# ============================================================
# SAVE KAGGLE SUBMISSION
# ============================================================

print("\n[10] Saving Kaggle submission CSV...")

submission = pd.DataFrame({
    "SK_ID_CURR" : sk_ids.values.astype(int),
    "TARGET"     : pd_scores      # probability (not binary label)
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"    Saved: {SUBMISSION_PATH}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEST INFERENCE COMPLETE")
print("=" * 70)
print(f"  Applicants scored : {len(pd_scores):,}")
print(f"  Submission CSV    : {SUBMISSION_PATH}")
print(f"  Predictions file  : {PRED_PATH}")
print("=" * 70)

from pathlib import Path
import duckdb

BASE_DIR = Path(r"C:\Users\Sahil Kumar\Downloads\Home_Credit")
DB_PATH = BASE_DIR / "data" / "home_credit.duckdb"
OUTPUT_DIR = BASE_DIR / "data" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

print("=" * 70)
print("BUILDING LEAKAGE-SAFE CREDIT FEATURES")
print("=" * 70)


# ============================================================
# 1. APPLICATION FEATURES
# ============================================================

print("\n[1/8] Application features...")

con.execute("""
CREATE OR REPLACE TABLE application_features AS
SELECT
    SK_ID_CURR,

    TARGET,

    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    AMT_ANNUITY,
    AMT_GOODS_PRICE,

    CNT_CHILDREN,
    CNT_FAM_MEMBERS,

    DAYS_BIRTH,
    DAYS_EMPLOYED,
    DAYS_REGISTRATION,
    DAYS_ID_PUBLISH,

    EXT_SOURCE_1,
    EXT_SOURCE_2,
    EXT_SOURCE_3,

    OWN_CAR_AGE,
    FLAG_OWN_CAR,
    FLAG_OWN_REALTY,

    NAME_CONTRACT_TYPE,
    CODE_GENDER,
    NAME_EDUCATION_TYPE,
    NAME_FAMILY_STATUS,
    NAME_INCOME_TYPE,
    OCCUPATION_TYPE,
    ORGANIZATION_TYPE,

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
        COALESCE(EXT_SOURCE_1, 0) +
        COALESCE(EXT_SOURCE_2, 0) +
        COALESCE(EXT_SOURCE_3, 0)
    )
    /
    NULLIF(
        (CASE WHEN EXT_SOURCE_1 IS NOT NULL THEN 1 ELSE 0 END) +
        (CASE WHEN EXT_SOURCE_2 IS NOT NULL THEN 1 ELSE 0 END) +
        (CASE WHEN EXT_SOURCE_3 IS NOT NULL THEN 1 ELSE 0 END),
        0
    ) AS ext_source_mean

FROM application_train
""")


# ============================================================
# 2. BUREAU FEATURES
# ============================================================

print("[2/8] Bureau features...")

con.execute("""
CREATE OR REPLACE TABLE bureau_features AS
SELECT
    SK_ID_CURR,

    COUNT(*) AS bureau_count,

    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)
        AS active_bureau_count,

    SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)
        AS closed_bureau_count,

    SUM(CASE
        WHEN DAYS_CREDIT >= -365 THEN 1
        ELSE 0
    END) AS bureau_last_year_count,

    MAX(CREDIT_DAY_OVERDUE) AS max_credit_day_overdue,

    SUM(CREDIT_DAY_OVERDUE) AS total_credit_day_overdue,

    SUM(AMT_CREDIT_SUM) AS total_bureau_credit,

    SUM(AMT_CREDIT_SUM_DEBT) AS total_bureau_debt,

    AVG(AMT_CREDIT_SUM) AS avg_bureau_credit,

    AVG(AMT_CREDIT_SUM_DEBT) AS avg_bureau_debt,

    CASE
        WHEN SUM(AMT_CREDIT_SUM) > 0
        THEN SUM(AMT_CREDIT_SUM_DEBT)
             / SUM(AMT_CREDIT_SUM)
        ELSE NULL
    END AS bureau_debt_to_credit_ratio,

    MIN(DAYS_CREDIT) AS oldest_bureau_record_days,

    MAX(DAYS_CREDIT) AS latest_bureau_record_days,

    AVG(DAYS_CREDIT) AS avg_bureau_age

FROM bureau

-- DAYS_CREDIT represents the age of the bureau record
-- relative to the application.
WHERE DAYS_CREDIT <= 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 3. PREVIOUS APPLICATION FEATURES
# ============================================================

print("[3/8] Previous application features...")

con.execute("""
CREATE OR REPLACE TABLE previous_application_features AS
SELECT
    SK_ID_CURR,

    COUNT(*) AS previous_application_count,

    SUM(CASE
        WHEN NAME_CONTRACT_STATUS = 'Approved'
        THEN 1 ELSE 0
    END) AS previous_approved_count,

    SUM(CASE
        WHEN NAME_CONTRACT_STATUS = 'Refused'
        THEN 1 ELSE 0
    END) AS previous_refused_count,

    SUM(CASE
        WHEN NAME_CONTRACT_STATUS = 'Canceled'
        THEN 1 ELSE 0
    END) AS previous_canceled_count,

    SUM(CASE
        WHEN DAYS_DECISION >= -365
        THEN 1 ELSE 0
    END) AS previous_apps_last_year,

    AVG(AMT_APPLICATION) AS avg_previous_application_amount,

    AVG(AMT_CREDIT) AS avg_previous_credit_amount,

    SUM(AMT_APPLICATION) AS total_previous_application_amount,

    SUM(AMT_CREDIT) AS total_previous_credit_amount,

    AVG(AMT_ANNUITY) AS avg_previous_annuity,

    AVG(DAYS_DECISION) AS avg_previous_decision_days,

    MIN(DAYS_DECISION) AS oldest_previous_decision,

    MAX(DAYS_DECISION) AS latest_previous_decision

FROM previous_application

WHERE DAYS_DECISION < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 4. INSTALLMENT FEATURES
# ============================================================

print("[4/8] Installment features...")

con.execute("""
CREATE OR REPLACE TABLE installment_features AS
SELECT
    SK_ID_CURR,

    COUNT(*) AS installment_count,

    SUM(AMT_INSTALMENT) AS total_installment_amount,

    SUM(AMT_PAYMENT) AS total_payment_amount,

    AVG(AMT_INSTALMENT) AS avg_installment_amount,

    AVG(AMT_PAYMENT) AS avg_payment_amount,

    AVG(
        CASE
            WHEN AMT_INSTALMENT > 0
            THEN AMT_PAYMENT / AMT_INSTALMENT
            ELSE NULL
        END
    ) AS avg_payment_ratio,

    SUM(
        CASE
            WHEN AMT_PAYMENT < AMT_INSTALMENT
            THEN 1 ELSE 0
        END
    ) AS underpayment_count,

    AVG(
        CASE
            WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
            THEN DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
            ELSE 0
        END
    ) AS avg_payment_delay,

    MAX(
        CASE
            WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
            THEN DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
            ELSE 0
        END
    ) AS max_payment_delay

FROM installments_payments

WHERE DAYS_INSTALMENT < 0
  AND DAYS_ENTRY_PAYMENT < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 5. POS CASH FEATURES
# ============================================================

print("[5/8] POS Cash features...")

con.execute("""
CREATE OR REPLACE TABLE pos_features AS
SELECT
    SK_ID_CURR,

    COUNT(*) AS pos_record_count,

    MAX(SK_DPD) AS max_pos_dpd,

    AVG(SK_DPD) AS avg_pos_dpd,

    SUM(
        CASE
            WHEN SK_DPD > 0
            THEN 1 ELSE 0
        END
    ) AS pos_dpd_count,

    MAX(SK_DPD_DEF) AS max_pos_dpd_def,

    AVG(SK_DPD_DEF) AS avg_pos_dpd_def,

    AVG(CNT_INSTALMENT) AS avg_pos_installments,

    MAX(CNT_INSTALMENT) AS max_pos_installments

FROM POS_CASH_balance

WHERE MONTHS_BALANCE < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 6. CREDIT CARD FEATURES
# ============================================================

print("[6/8] Credit card features...")

con.execute("""
CREATE OR REPLACE TABLE credit_card_features AS
SELECT
    SK_ID_CURR,

    COUNT(*) AS credit_card_record_count,

    AVG(AMT_BALANCE) AS avg_cc_balance,

    MAX(AMT_BALANCE) AS max_cc_balance,

    AVG(AMT_CREDIT_LIMIT_ACTUAL) AS avg_cc_limit,

    MAX(AMT_CREDIT_LIMIT_ACTUAL) AS max_cc_limit,

    AVG(
        CASE
            WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
            THEN AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
            ELSE NULL
        END
    ) AS avg_cc_utilization,

    MAX(
        CASE
            WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
            THEN AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
            ELSE NULL
        END
    ) AS max_cc_utilization,

    SUM(AMT_DRAWINGS_CURRENT) AS total_cc_drawings,

    AVG(AMT_PAYMENT_CURRENT) AS avg_cc_payment,

    MAX(SK_DPD) AS max_cc_dpd,

    AVG(SK_DPD) AS avg_cc_dpd,

    SUM(
        CASE
            WHEN SK_DPD > 0
            THEN 1 ELSE 0
        END
    ) AS cc_dpd_count

FROM credit_card_balance

WHERE MONTHS_BALANCE < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 7. BUREAU BALANCE FEATURES
# ============================================================

print("[7/8] Bureau balance features...")

con.execute("""
CREATE OR REPLACE TABLE bureau_balance_features AS

SELECT
    b.SK_ID_CURR,

    COUNT(bb.SK_ID_BUREAU) AS bureau_balance_record_count,

    AVG(bb.MONTHS_BALANCE) AS avg_bureau_balance_month,

    MIN(bb.MONTHS_BALANCE) AS oldest_bureau_balance_month,

    SUM(
        CASE
            WHEN bb.STATUS IN ('1','2','3','4','5')
            THEN 1 ELSE 0
        END
    ) AS bureau_overdue_status_count,

    SUM(
        CASE
            WHEN bb.STATUS = 'C'
            THEN 1 ELSE 0
        END
    ) AS bureau_closed_status_count

FROM bureau b

JOIN bureau_balance bb
    ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU

WHERE bb.MONTHS_BALANCE < 0

GROUP BY b.SK_ID_CURR
""")


# ============================================================
# 8. FINAL MODEL TABLE
# ============================================================

print("[8/8] Creating final model features...")

con.execute("""
CREATE OR REPLACE TABLE model_features AS

SELECT
    a.*,

    b.bureau_count,
    b.active_bureau_count,
    b.closed_bureau_count,
    b.bureau_last_year_count,
    b.max_credit_day_overdue,
    b.total_credit_day_overdue,
    b.total_bureau_credit,
    b.total_bureau_debt,
    b.avg_bureau_credit,
    b.avg_bureau_debt,
    b.bureau_debt_to_credit_ratio,
    b.oldest_bureau_record_days,
    b.latest_bureau_record_days,
    b.avg_bureau_age,

    p.previous_application_count,
    p.previous_approved_count,
    p.previous_refused_count,
    p.previous_canceled_count,
    p.previous_apps_last_year,
    p.avg_previous_application_amount,
    p.avg_previous_credit_amount,
    p.total_previous_application_amount,
    p.total_previous_credit_amount,
    p.avg_previous_annuity,
    p.avg_previous_decision_days,
    p.oldest_previous_decision,
    p.latest_previous_decision,

    i.installment_count,
    i.total_installment_amount,
    i.total_payment_amount,
    i.avg_installment_amount,
    i.avg_payment_amount,
    i.avg_payment_ratio,
    i.underpayment_count,
    i.avg_payment_delay,
    i.max_payment_delay,

    pos.pos_record_count,
    pos.max_pos_dpd,
    pos.avg_pos_dpd,
    pos.pos_dpd_count,
    pos.max_pos_dpd_def,
    pos.avg_pos_dpd_def,
    pos.avg_pos_installments,
    pos.max_pos_installments,

    cc.credit_card_record_count,
    cc.avg_cc_balance,
    cc.max_cc_balance,
    cc.avg_cc_limit,
    cc.max_cc_limit,
    cc.avg_cc_utilization,
    cc.max_cc_utilization,
    cc.total_cc_drawings,
    cc.avg_cc_payment,
    cc.max_cc_dpd,
    cc.avg_cc_dpd,
    cc.cc_dpd_count,

    bb.bureau_balance_record_count,
    bb.avg_bureau_balance_month,
    bb.oldest_bureau_balance_month,
    bb.bureau_overdue_status_count,
    bb.bureau_closed_status_count

FROM application_features a

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


# ============================================================
# REMOVE APPLICATION DATE FROM MODEL FEATURES
# ============================================================

con.execute("""
ALTER TABLE model_features
DROP COLUMN IF EXISTS application_date
""")


# ============================================================
# EXPORT
# ============================================================

output_path = OUTPUT_DIR / "model_features.parquet"

con.execute(f"""
COPY model_features
TO '{output_path.as_posix()}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")


# ============================================================
# SUMMARY
# ============================================================

count = con.execute("""
SELECT COUNT(*) FROM model_features
""").fetchone()[0]

columns = con.execute("""
SELECT COUNT(*)
FROM information_schema.columns
WHERE table_name = 'model_features'
""").fetchone()[0]

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Rows:    {count:,}")
print(f"Columns: {columns:,}")
print(f"Output:  {output_path}")

con.close()
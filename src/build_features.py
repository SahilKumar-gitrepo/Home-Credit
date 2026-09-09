from pathlib import Path
import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\Sahil Kumar\Downloads\Home_Credit")

DB_PATH = BASE_DIR / "data" / "home_credit.duckdb"
OUTPUT_DIR = BASE_DIR / "data" / "features"
OUTPUT_FILE = OUTPUT_DIR / "model_features.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONNECT
# ============================================================

print("=" * 70)
print("BUILDING LEAKAGE-SAFE MODEL FEATURES")
print("=" * 70)

con = duckdb.connect(str(DB_PATH))


# ============================================================
# 1. APPLICATION FEATURES
# ============================================================

print("\n[1/8] Building application features...")

con.execute("""
CREATE OR REPLACE TABLE application_features AS

SELECT
    SK_ID_CURR,

    -- Target
    TARGET,

    -- Basic application information
    NAME_CONTRACT_TYPE,
    CODE_GENDER,
    FLAG_OWN_CAR,
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
    FLAG_DOCUMENT_2,
    FLAG_DOCUMENT_3,
    FLAG_DOCUMENT_4,
    FLAG_DOCUMENT_5,
    FLAG_DOCUMENT_6,
    FLAG_DOCUMENT_7,
    FLAG_DOCUMENT_8,
    FLAG_DOCUMENT_9,
    FLAG_DOCUMENT_10,
    FLAG_DOCUMENT_11,
    FLAG_DOCUMENT_12,
    FLAG_DOCUMENT_13,
    FLAG_DOCUMENT_14,
    FLAG_DOCUMENT_15,
    FLAG_DOCUMENT_16,
    FLAG_DOCUMENT_17,
    FLAG_DOCUMENT_18,
    FLAG_DOCUMENT_19,
    FLAG_DOCUMENT_20,
    FLAG_DOCUMENT_21,

    -- Derived application features
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
    ) AS ext_source_mean,

    -- Keep date only for possible future temporal validation.
    -- It is NOT included in final model_features.
    application_date

FROM application_train
""")


# ============================================================
# 2. BUREAU FEATURES
# ============================================================

print("[2/8] Building bureau features...")

con.execute("""
CREATE OR REPLACE TABLE bureau_features AS

SELECT
    SK_ID_CURR,

    COUNT(*) AS bureau_record_count,

    COUNT(DISTINCT SK_ID_BUREAU) AS bureau_account_count,

    SUM(
        CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END
    ) AS active_bureau_accounts,

    SUM(
        CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END
    ) AS closed_bureau_accounts,

    SUM(
        CASE
            WHEN DAYS_CREDIT >= -365 THEN 1
            ELSE 0
        END
    ) AS bureau_recent_1y_count,

    -- Credit exposure
    SUM(COALESCE(AMT_CREDIT_SUM, 0)) AS total_bureau_credit,

    SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0)) AS total_bureau_debt,

    AVG(AMT_CREDIT_SUM) AS avg_bureau_credit,

    AVG(AMT_CREDIT_SUM_DEBT) AS avg_bureau_debt,

    MAX(AMT_CREDIT_SUM) AS max_bureau_credit,

    MAX(AMT_CREDIT_SUM_DEBT) AS max_bureau_debt,

    -- Overdue information
    SUM(
        COALESCE(AMT_CREDIT_SUM_OVERDUE, 0)
    ) AS total_bureau_overdue,

    MAX(
        COALESCE(AMT_CREDIT_SUM_OVERDUE, 0)
    ) AS max_bureau_overdue,

    SUM(
        CASE
            WHEN CREDIT_DAY_OVERDUE > 0 THEN 1
            ELSE 0
        END
    ) AS bureau_overdue_account_count,

    MAX(CREDIT_DAY_OVERDUE) AS max_bureau_days_overdue,

    -- Credit history age
    AVG(DAYS_CREDIT) AS avg_days_since_bureau_credit,

    MIN(DAYS_CREDIT) AS oldest_bureau_credit_days,

    -- Robust aggregate debt / credit ratio.
    -- IMPORTANT:
    -- This uses aggregate sums rather than averaging row-level ratios.
    CASE
        WHEN SUM(
            CASE
                WHEN AMT_CREDIT_SUM > 0
                THEN AMT_CREDIT_SUM
                ELSE 0
            END
        ) > 0
        THEN
            SUM(
                CASE
                    WHEN AMT_CREDIT_SUM_DEBT IS NOT NULL
                    THEN AMT_CREDIT_SUM_DEBT
                    ELSE 0
                END
            )
            /
            SUM(
                CASE
                    WHEN AMT_CREDIT_SUM > 0
                    THEN AMT_CREDIT_SUM
                    ELSE 0
                END
            )
        ELSE NULL
    END AS bureau_debt_to_credit_ratio

FROM bureau

WHERE DAYS_CREDIT <= 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 3. PREVIOUS APPLICATION FEATURES
# ============================================================

print("[3/8] Building previous application features...")

con.execute("""
CREATE OR REPLACE TABLE previous_application_features AS

SELECT
    SK_ID_CURR,

    COUNT(*) AS previous_application_count,

    SUM(
        CASE
            WHEN NAME_CONTRACT_STATUS = 'Approved'
            THEN 1 ELSE 0
        END
    ) AS previous_approved_count,

    SUM(
        CASE
            WHEN NAME_CONTRACT_STATUS = 'Refused'
            THEN 1 ELSE 0
        END
    ) AS previous_refused_count,

    SUM(
        CASE
            WHEN NAME_CONTRACT_STATUS = 'Canceled'
            THEN 1 ELSE 0
        END
    ) AS previous_canceled_count,

    SUM(
        CASE
            WHEN NAME_CONTRACT_STATUS = 'Unused offer'
            THEN 1 ELSE 0
        END
    ) AS previous_unused_offer_count,

    SUM(
        CASE
            WHEN DAYS_DECISION >= -365
            THEN 1 ELSE 0
        END
    ) AS previous_recent_1y_count,

    -- Amounts
    SUM(COALESCE(AMT_APPLICATION, 0))
        AS total_previous_application_amount,

    SUM(COALESCE(AMT_CREDIT, 0))
        AS total_previous_credit_amount,

    AVG(AMT_APPLICATION)
        AS avg_previous_application_amount,

    AVG(AMT_CREDIT)
        AS avg_previous_credit_amount,

    MAX(AMT_CREDIT)
        AS max_previous_credit_amount,

    AVG(AMT_ANNUITY)
        AS avg_previous_annuity,

    -- Timing
    AVG(DAYS_DECISION)
        AS avg_days_since_previous_decision,

    MIN(DAYS_DECISION)
        AS oldest_previous_decision_days

FROM previous_application

WHERE DAYS_DECISION < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 4. INSTALLMENT PAYMENT FEATURES
# ============================================================

print("[4/8] Building installment features...")

con.execute("""
CREATE OR REPLACE TABLE installment_features AS

SELECT
    SK_ID_CURR,

    COUNT(*) AS installment_record_count,

    COUNT(DISTINCT SK_ID_PREV)
        AS installment_previous_loan_count,

    -- Total scheduled amount
    SUM(
        CASE
            WHEN AMT_INSTALMENT > 0
            THEN AMT_INSTALMENT
            ELSE 0
        END
    ) AS total_installment_amount,

    -- Total actual payment
    SUM(
        CASE
            WHEN AMT_PAYMENT > 0
            THEN AMT_PAYMENT
            ELSE 0
        END
    ) AS total_payment_amount,

    AVG(AMT_INSTALMENT)
        AS avg_installment_amount,

    AVG(AMT_PAYMENT)
        AS avg_payment_amount,

    -- Payment difference
    SUM(
        COALESCE(AMT_PAYMENT, 0)
        -
        COALESCE(AMT_INSTALMENT, 0)
    ) AS total_payment_difference,

    AVG(
        COALESCE(AMT_PAYMENT, 0)
        -
        COALESCE(AMT_INSTALMENT, 0)
    ) AS avg_payment_difference,

    -- Number of underpayments
    SUM(
        CASE
            WHEN AMT_PAYMENT < AMT_INSTALMENT
            THEN 1
            ELSE 0
        END
    ) AS underpayment_count,

    -- Number of overpayments
    SUM(
        CASE
            WHEN AMT_PAYMENT > AMT_INSTALMENT
            THEN 1
            ELSE 0
        END
    ) AS overpayment_count,

    -- Robust payment ratio:
    -- SUM(payment) / SUM(installment)
    CASE
        WHEN SUM(
            CASE
                WHEN AMT_INSTALMENT > 0
                THEN AMT_INSTALMENT
                ELSE 0
            END
        ) > 0
        THEN
            SUM(
                CASE
                    WHEN AMT_PAYMENT > 0
                    THEN AMT_PAYMENT
                    ELSE 0
                END
            )
            /
            SUM(
                CASE
                    WHEN AMT_INSTALMENT > 0
                    THEN AMT_INSTALMENT
                    ELSE 0
                END
            )
        ELSE NULL
    END AS payment_ratio,

    -- Underpayment rate
    CASE
        WHEN COUNT(*) > 0
        THEN
            SUM(
                CASE
                    WHEN AMT_PAYMENT < AMT_INSTALMENT
                    THEN 1
                    ELSE 0
                END
            )::DOUBLE
            / COUNT(*)
        ELSE NULL
    END AS underpayment_rate,

    -- Payment delay
    AVG(
        CASE
            WHEN DAYS_ENTRY_PAYMENT IS NOT NULL
                 AND DAYS_INSTALMENT IS NOT NULL
            THEN GREATEST(
                DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
                0
            )
            ELSE NULL
        END
    ) AS avg_payment_delay,

    MAX(
        CASE
            WHEN DAYS_ENTRY_PAYMENT IS NOT NULL
                 AND DAYS_INSTALMENT IS NOT NULL
            THEN GREATEST(
                DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
                0
            )
            ELSE NULL
        END
    ) AS max_payment_delay,

    SUM(
        CASE
            WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
            THEN 1
            ELSE 0
        END
    ) AS late_payment_count,

    CASE
        WHEN COUNT(*) > 0
        THEN
            SUM(
                CASE
                    WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
                    THEN 1
                    ELSE 0
                END
            )::DOUBLE / COUNT(*)
        ELSE NULL
    END AS late_payment_rate

FROM installments_payments

WHERE DAYS_INSTALMENT < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 5. POS CASH FEATURES
# ============================================================

print("[5/8] Building POS cash features...")

con.execute("""
CREATE OR REPLACE TABLE pos_features AS

SELECT
    SK_ID_CURR,

    COUNT(*) AS pos_record_count,

    COUNT(DISTINCT SK_ID_PREV)
        AS pos_previous_loan_count,

    AVG(MONTHS_BALANCE)
        AS avg_pos_months_balance,

    MIN(MONTHS_BALANCE)
        AS oldest_pos_month,

    MAX(
        COALESCE(SK_DPD, 0)
    ) AS max_pos_dpd,

    AVG(
        COALESCE(SK_DPD, 0)
    ) AS avg_pos_dpd,

    SUM(
        CASE
            WHEN SK_DPD > 0
            THEN 1 ELSE 0
        END
    ) AS pos_dpd_record_count,

    MAX(
        COALESCE(SK_DPD_DEF, 0)
    ) AS max_pos_dpd_def,

    AVG(
        COALESCE(SK_DPD_DEF, 0)
    ) AS avg_pos_dpd_def,

    SUM(
        CASE
            WHEN SK_DPD_DEF > 0
            THEN 1 ELSE 0
        END
    ) AS pos_dpd_def_record_count,

    AVG(
        CNT_INSTALMENT
    ) AS avg_pos_installments,

    AVG(
        CNT_INSTALMENT_FUTURE
    ) AS avg_pos_future_installments

FROM POS_CASH_balance

WHERE MONTHS_BALANCE < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 6. CREDIT CARD FEATURES
# ============================================================

print("[6/8] Building credit card features...")

con.execute("""
CREATE OR REPLACE TABLE credit_card_features AS

SELECT
    SK_ID_CURR,

    COUNT(*) AS cc_record_count,

    COUNT(DISTINCT SK_ID_PREV)
        AS cc_previous_loan_count,

    AVG(AMT_BALANCE)
        AS avg_cc_balance,

    MAX(AMT_BALANCE)
        AS max_cc_balance,

    AVG(AMT_CREDIT_LIMIT_ACTUAL)
        AS avg_cc_limit,

    MAX(AMT_CREDIT_LIMIT_ACTUAL)
        AS max_cc_limit,

    -- Robust utilization based on aggregate balance
    -- and aggregate positive credit limit.
    CASE
        WHEN SUM(
            CASE
                WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
                THEN AMT_CREDIT_LIMIT_ACTUAL
                ELSE 0
            END
        ) > 0
        THEN
            SUM(
                CASE
                    WHEN AMT_BALANCE > 0
                    THEN AMT_BALANCE
                    ELSE 0
                END
            )
            /
            SUM(
                CASE
                    WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
                    THEN AMT_CREDIT_LIMIT_ACTUAL
                    ELSE 0
                END
            )
        ELSE NULL
    END AS cc_utilization,

    AVG(
        CASE
            WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
            THEN AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
            ELSE NULL
        END
    ) AS avg_cc_row_utilization,

    MAX(
        CASE
            WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
            THEN AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
            ELSE NULL
        END
    ) AS max_cc_row_utilization,

    -- Drawings
    SUM(
        COALESCE(AMT_DRAWINGS_ATM_CURRENT, 0)
    ) AS total_cc_atm_drawings,

    SUM(
        COALESCE(AMT_DRAWINGS_CURRENT, 0)
    ) AS total_cc_drawings,

    SUM(
        COALESCE(AMT_DRAWINGS_POS_CURRENT, 0)
    ) AS total_cc_pos_drawings,

    -- Payments
    SUM(
        COALESCE(AMT_PAYMENT_CURRENT, 0)
    ) AS total_cc_payments,

    AVG(
        COALESCE(AMT_PAYMENT_CURRENT, 0)
    ) AS avg_cc_payment,

    -- DPD
    MAX(
        COALESCE(SK_DPD, 0)
    ) AS max_cc_dpd,

    AVG(
        COALESCE(SK_DPD, 0)
    ) AS avg_cc_dpd,

    SUM(
        CASE
            WHEN SK_DPD > 0
            THEN 1 ELSE 0
        END
    ) AS cc_dpd_record_count,

    MAX(
        COALESCE(SK_DPD_DEF, 0)
    ) AS max_cc_dpd_def,

    AVG(
        COALESCE(SK_DPD_DEF, 0)
    ) AS avg_cc_dpd_def

FROM credit_card_balance

WHERE MONTHS_BALANCE < 0

GROUP BY SK_ID_CURR
""")


# ============================================================
# 7. BUREAU BALANCE FEATURES
# ============================================================

print("[7/8] Building bureau balance features...")

con.execute("""
CREATE OR REPLACE TABLE bureau_balance_features AS

SELECT
    b.SK_ID_CURR,

    COUNT(*) AS bureau_balance_record_count,

    COUNT(DISTINCT bb.SK_ID_BUREAU)
        AS bureau_balance_account_count,

    AVG(bb.MONTHS_BALANCE)
        AS avg_bureau_balance_month,

    MIN(bb.MONTHS_BALANCE)
        AS oldest_bureau_balance_month,

    -- Status information
    SUM(
        CASE
            WHEN bb.STATUS IN ('1', '2', '3', '4', '5')
            THEN 1 ELSE 0
        END
    ) AS bureau_balance_overdue_count,

    SUM(
        CASE
            WHEN bb.STATUS = 'C'
            THEN 1 ELSE 0
        END
    ) AS bureau_balance_closed_count,

    SUM(
        CASE
            WHEN bb.STATUS = 'X'
            THEN 1 ELSE 0
        END
    ) AS bureau_balance_unknown_count,

    CASE
        WHEN COUNT(*) > 0
        THEN
            SUM(
                CASE
                    WHEN bb.STATUS IN ('1', '2', '3', '4', '5')
                    THEN 1 ELSE 0
                END
            )::DOUBLE / COUNT(*)
        ELSE NULL
    END AS bureau_balance_overdue_rate

FROM bureau b

INNER JOIN bureau_balance bb
    ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU

WHERE
    b.DAYS_CREDIT <= 0
    AND bb.MONTHS_BALANCE < 0

GROUP BY b.SK_ID_CURR
""")


# ============================================================
# 8. FINAL MODEL TABLE
# ============================================================

print("[8/8] Creating final model feature table...")

con.execute("""
CREATE OR REPLACE TABLE model_features AS

SELECT

    -- Application
    a.* EXCLUDE (application_date),

    -- Bureau
    b.* EXCLUDE (SK_ID_CURR),

    -- Previous applications
    p.* EXCLUDE (SK_ID_CURR),

    -- Installments
    i.* EXCLUDE (SK_ID_CURR),

    -- POS
    pos.* EXCLUDE (SK_ID_CURR),

    -- Credit card
    cc.* EXCLUDE (SK_ID_CURR),

    -- Bureau balance
    bb.* EXCLUDE (SK_ID_CURR)

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
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING FINAL FEATURE TABLE")
print("=" * 70)

row_count = con.execute("""
SELECT COUNT(*)
FROM model_features
""").fetchone()[0]

unique_count = con.execute("""
SELECT COUNT(DISTINCT SK_ID_CURR)
FROM model_features
""").fetchone()[0]

column_count = con.execute("""
SELECT COUNT(*)
FROM information_schema.columns
WHERE table_name = 'model_features'
""").fetchone()[0]

print(f"\nRows:             {row_count:,}")
print(f"Unique applicants:{unique_count:,}")
print(f"Columns:          {column_count}")

if row_count == unique_count:
    print("✓ Exactly one row per applicant")
else:
    print("✗ DUPLICATE APPLICANTS FOUND")


# ============================================================
# CHECK NEW RATIOS
# ============================================================

print("\n" + "-" * 70)
print("RATIO SANITY CHECKS")
print("-" * 70)

print("\nBureau debt / credit:")

print(
    con.execute("""
    SELECT
        MIN(bureau_debt_to_credit_ratio),
        MAX(bureau_debt_to_credit_ratio),
        AVG(bureau_debt_to_credit_ratio)
    FROM model_features
    """).fetchone()
)


print("\nPayment ratio:")

print(
    con.execute("""
    SELECT
        MIN(payment_ratio),
        MAX(payment_ratio),
        AVG(payment_ratio)
    FROM model_features
    """).fetchone()
)


print("\nCredit-card utilization:")

print(
    con.execute("""
    SELECT
        MIN(cc_utilization),
        MAX(cc_utilization),
        AVG(cc_utilization)
    FROM model_features
    """).fetchone()
)


# ============================================================
# EXPORT PARQUET
# ============================================================

print("\n" + "-" * 70)
print("EXPORTING FEATURES")
print("-" * 70)

con.execute(f"""
COPY model_features
TO '{OUTPUT_FILE.as_posix()}'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print(f"\n✓ Saved:")
print(OUTPUT_FILE)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("TARGET DISTRIBUTION")
print("-" * 70)

target_rows = con.execute("""
SELECT
    TARGET,
    COUNT(*) AS count,
    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage
FROM model_features
GROUP BY TARGET
ORDER BY TARGET
""").fetchall()

for row in target_rows:
    print(
        f"TARGET={row[0]}  "
        f"count={row[1]:,}  "
        f"percentage={row[2]}%"
    )


# ============================================================
# FINISH
# ============================================================

con.close()

print("\n" + "=" * 70)
print("FEATURE BUILD COMPLETE")
print("=" * 70)
from pathlib import Path
import duckdb

BASE_DIR = Path(r"C:\Users\Sahil Kumar\Downloads\Home_Credit")
DB_PATH = BASE_DIR / "data" / "home_credit.duckdb"

con = duckdb.connect(str(DB_PATH), read_only=True)

print("=" * 70)
print("TEMPORAL RELATIONSHIP ANALYSIS")
print("=" * 70)


# ============================================================
# 1. APPLICATION DATE SAMPLE
# ============================================================

print("\n[1] APPLICATION DATE")

print("-" * 70)

result = con.execute("""
    SELECT
        SK_ID_CURR,
        application_date,
        TARGET
    FROM application_train
    ORDER BY application_date, SK_ID_CURR
    LIMIT 10
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 2. BUREAU RELATIVE DATES
# ============================================================

print("\n[2] BUREAU TEMPORAL FIELDS")

print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS total_records,

        COUNT(DAYS_CREDIT) AS days_credit_non_null,

        MIN(DAYS_CREDIT) AS min_days_credit,
        MAX(DAYS_CREDIT) AS max_days_credit,

        MIN(DAYS_CREDIT_ENDDATE) AS min_days_credit_enddate,
        MAX(DAYS_CREDIT_ENDDATE) AS max_days_credit_enddate,

        MIN(DAYS_ENDDATE_FACT) AS min_days_enddate_fact,
        MAX(DAYS_ENDDATE_FACT) AS max_days_enddate_fact,

        MIN(DAYS_CREDIT_UPDATE) AS min_days_credit_update,
        MAX(DAYS_CREDIT_UPDATE) AS max_days_credit_update

    FROM bureau
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 3. PREVIOUS APPLICATION TEMPORAL FIELDS
# ============================================================

print("\n[3] PREVIOUS APPLICATION TEMPORAL FIELDS")

print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS total_records,

        MIN(DAYS_DECISION) AS min_days_decision,
        MAX(DAYS_DECISION) AS max_days_decision,

        MIN(DAYS_FIRST_DRAWING) AS min_days_first_drawing,
        MAX(DAYS_FIRST_DRAWING) AS max_days_first_drawing,

        MIN(DAYS_FIRST_DUE) AS min_days_first_due,
        MAX(DAYS_FIRST_DUE) AS max_days_first_due,

        MIN(DAYS_LAST_DUE) AS min_days_last_due,
        MAX(DAYS_LAST_DUE) AS max_days_last_due,

        MIN(DAYS_TERMINATION) AS min_days_termination,
        MAX(DAYS_TERMINATION) AS max_days_termination

    FROM previous_application
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 4. INSTALLMENT TEMPORAL FIELDS
# ============================================================

print("\n[4] INSTALLMENT TEMPORAL FIELDS")

print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS total_records,

        MIN(DAYS_INSTALMENT) AS min_days_instalment,
        MAX(DAYS_INSTALMENT) AS max_days_instalment,

        MIN(DAYS_ENTRY_PAYMENT) AS min_days_entry_payment,
        MAX(DAYS_ENTRY_PAYMENT) AS max_days_entry_payment

    FROM installments_payments
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 5. POS / CREDIT CARD MONTHS
# ============================================================

print("\n[5] POS CASH TEMPORAL FIELDS")

print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS total_records,
        MIN(MONTHS_BALANCE) AS min_months_balance,
        MAX(MONTHS_BALANCE) AS max_months_balance
    FROM POS_CASH_balance
""").fetchdf()

print(result.to_string(index=False))


print("\n[6] CREDIT CARD TEMPORAL FIELDS")

print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS total_records,
        MIN(MONTHS_BALANCE) AS min_months_balance,
        MAX(MONTHS_BALANCE) AS max_months_balance
    FROM credit_card_balance
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 7. CHECK DAYS_CREDIT DISTRIBUTION
# ============================================================

print("\n[7] BUREAU DAYS_CREDIT DISTRIBUTION")

print("-" * 70)

result = con.execute("""
    SELECT
        CASE
            WHEN DAYS_CREDIT < 0 THEN 'negative'
            WHEN DAYS_CREDIT = 0 THEN 'zero'
            WHEN DAYS_CREDIT > 0 THEN 'positive'
            ELSE 'NULL'
        END AS category,
        COUNT(*) AS records
    FROM bureau
    GROUP BY category
    ORDER BY category
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 8. CHECK DAYS_DECISION DISTRIBUTION
# ============================================================

print("\n[8] PREVIOUS APPLICATION DAYS_DECISION DISTRIBUTION")

print("-" * 70)

result = con.execute("""
    SELECT
        CASE
            WHEN DAYS_DECISION < 0 THEN 'negative'
            WHEN DAYS_DECISION = 0 THEN 'zero'
            WHEN DAYS_DECISION > 0 THEN 'positive'
            ELSE 'NULL'
        END AS category,
        COUNT(*) AS records
    FROM previous_application
    GROUP BY category
    ORDER BY category
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 9. CHECK MONTHS_BALANCE
# ============================================================

print("\n[9] MONTHS_BALANCE DISTRIBUTION")

print("-" * 70)

result = con.execute("""
    SELECT
        'POS_CASH' AS table_name,
        MIN(MONTHS_BALANCE) AS min_value,
        MAX(MONTHS_BALANCE) AS max_value,
        AVG(MONTHS_BALANCE) AS avg_value
    FROM POS_CASH_balance

    UNION ALL

    SELECT
        'CREDIT_CARD',
        MIN(MONTHS_BALANCE),
        MAX(MONTHS_BALANCE),
        AVG(MONTHS_BALANCE)
    FROM credit_card_balance
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 10. FINAL
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL RELATIONSHIP ANALYSIS COMPLETE")
print("=" * 70)

con.close()
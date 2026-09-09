from pathlib import Path
import duckdb

BASE_DIR = Path(r"C:\Users\Sahil Kumar\Downloads\Home_Credit")
DB_PATH = BASE_DIR / "data" / "home_credit.duckdb"

con = duckdb.connect(str(DB_PATH))

print("=" * 70)
print("FEATURE ANOMALY INVESTIGATION")
print("=" * 70)


# ============================================================
# 1. CHECK DUPLICATES IN FINAL MODEL TABLE
# ============================================================

print("\n[1] FINAL MODEL DUPLICATE CHECK")
print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT SK_ID_CURR) AS unique_ids
    FROM model_features
""").fetchone()

print(f"Total rows:       {result[0]:,}")
print(f"Unique applicants: {result[1]:,}")

if result[0] == result[1]:
    print("✓ Final table has exactly one row per applicant")
else:
    print("✗ DUPLICATES FOUND")


# ============================================================
# 2. CHECK DUPLICATES IN EACH AGGREGATED TABLE
# ============================================================

print("\n[2] AGGREGATED TABLE DUPLICATE CHECK")
print("-" * 70)

tables = [
    "bureau_features",
    "previous_application_features",
    "installment_features",
    "pos_features",
    "credit_card_features",
    "bureau_balance_features",
]

for table in tables:

    total, unique_ids = con.execute(f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT SK_ID_CURR)
        FROM {table}
    """).fetchone()

    duplicates = total - unique_ids

    print(
        f"{table:<35}"
        f"rows={total:>10,}  "
        f"unique={unique_ids:>10,}  "
        f"duplicates={duplicates:>8,}"
    )


# ============================================================
# 3. BUREAU RATIO
# ============================================================

print("\n[3] BUREAU DEBT/CREDIT RATIO")
print("-" * 70)

result = con.execute("""
    SELECT
        SK_ID_CURR,
        total_bureau_credit,
        total_bureau_debt,
        bureau_debt_to_credit_ratio
    FROM model_features
    WHERE bureau_debt_to_credit_ratio < 0
       OR bureau_debt_to_credit_ratio > 1
    ORDER BY bureau_debt_to_credit_ratio
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))

count = con.execute("""
    SELECT COUNT(*)
    FROM model_features
    WHERE bureau_debt_to_credit_ratio < 0
       OR bureau_debt_to_credit_ratio > 1
""").fetchone()[0]

print(f"\nRows outside [0,1]: {count:,}")


# ============================================================
# 4. PAYMENT RATIO
# ============================================================

print("\n[4] PAYMENT RATIO")
print("-" * 70)

result = con.execute("""
    SELECT
        SK_ID_CURR,
        avg_installment_amount,
        avg_payment_amount,
        avg_payment_ratio
    FROM model_features
    WHERE avg_payment_ratio > 2
       OR avg_payment_ratio < 0
    ORDER BY avg_payment_ratio DESC
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))

count = con.execute("""
    SELECT COUNT(*)
    FROM model_features
    WHERE avg_payment_ratio > 2
       OR avg_payment_ratio < 0
""").fetchone()[0]

print(f"\nRows outside [0,2]: {count:,}")


# ============================================================
# 5. CREDIT CARD UTILIZATION
# ============================================================

print("\n[5] CREDIT CARD UTILIZATION")
print("-" * 70)

result = con.execute("""
    SELECT
        SK_ID_CURR,
        avg_cc_balance,
        max_cc_balance,
        avg_cc_limit,
        max_cc_limit,
        avg_cc_utilization,
        max_cc_utilization
    FROM model_features
    WHERE avg_cc_utilization > 1
       OR max_cc_utilization > 1
    ORDER BY max_cc_utilization DESC
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))

count = con.execute("""
    SELECT COUNT(*)
    FROM model_features
    WHERE avg_cc_utilization > 1
       OR max_cc_utilization > 1
""").fetchone()[0]

print(f"\nRows with utilization > 100%: {count:,}")


# ============================================================
# 6. EXTREME CREDIT / INCOME
# ============================================================

print("\n[6] EXTREME CREDIT/INCOME")
print("-" * 70)

result = con.execute("""
    SELECT
        SK_ID_CURR,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        credit_to_income
    FROM model_features
    ORDER BY credit_to_income DESC
    LIMIT 20
""").fetchdf()

print(result.to_string(index=False))


# ============================================================
# 7. EXTREME PAYMENT DELAYS
# ============================================================

print("\n[7] PAYMENT DELAYS")
print("-" * 70)

result = con.execute("""
    SELECT
        MIN(avg_payment_delay),
        MAX(avg_payment_delay),
        MIN(max_payment_delay),
        MAX(max_payment_delay)
    FROM model_features
""").fetchone()

print(f"Average payment delay: {result[0]} → {result[1]}")
print(f"Maximum payment delay: {result[2]} → {result[3]}")


# ============================================================
# 8. FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANOMALY INVESTIGATION COMPLETE")
print("=" * 70)

con.close()
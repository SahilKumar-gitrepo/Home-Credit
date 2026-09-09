from pathlib import Path
import duckdb

BASE_DIR = Path(r"C:\Users\Sahil Kumar\Downloads\Home_Credit")
DB_PATH = BASE_DIR / "data" / "home_credit.duckdb"

con = duckdb.connect(str(DB_PATH), read_only=True)

print("=" * 70)
print("HOME CREDIT FEATURE VALIDATION")
print("=" * 70)


# ============================================================
# 1. DATASET SIZE
# ============================================================

print("\n[1] DATASET SIZE")
print("-" * 70)

rows, columns = con.execute("""
    SELECT
        COUNT(*),
        COUNT(*)
    FROM model_features
""").fetchone()

column_count = con.execute("""
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_name = 'model_features'
""").fetchone()[0]

print(f"Rows:    {rows:,}")
print(f"Columns: {column_count:,}")


# ============================================================
# 2. UNIQUE APPLICANTS
# ============================================================

print("\n[2] UNIQUE APPLICANTS")
print("-" * 70)

total_rows = con.execute("""
    SELECT COUNT(*)
    FROM model_features
""").fetchone()[0]

unique_ids = con.execute("""
    SELECT COUNT(DISTINCT SK_ID_CURR)
    FROM model_features
""").fetchone()[0]

print(f"Total applicant rows: {total_rows:,}")
print(f"Unique applicants:    {unique_ids:,}")
print(f"Duplicate rows:       {total_rows - unique_ids:,}")

if total_rows == unique_ids:
    print("Status: ✓ ONE ROW PER APPLICANT")
else:
    print("Status: ✗ DUPLICATE APPLICANTS FOUND")


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("\n[3] TARGET CHECK")
print("-" * 70)

target = con.execute("""
    SELECT
        TARGET,
        COUNT(*) AS count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
    FROM model_features
    GROUP BY TARGET
    ORDER BY TARGET
""").fetchdf()

print(target.to_string(index=False))


# ============================================================
# 4. NULL VALUE CHECK
# ============================================================

print("\n[4] NULL VALUE CHECK")
print("-" * 70)

columns_info = con.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'model_features'
    ORDER BY ordinal_position
""").fetchall()

null_columns = []

for (column_name,) in columns_info:

    # Don't treat TARGET/ID as feature null checks
    null_count = con.execute(f"""
        SELECT COUNT(*)
        FROM model_features
        WHERE "{column_name}" IS NULL
    """).fetchone()[0]

    if null_count > 0:
        percentage = 100.0 * null_count / rows
        null_columns.append(
            (column_name, null_count, percentage)
        )

print(f"Columns containing NULLs: {len(null_columns)}")

for column_name, count, percentage in null_columns:
    print(
        f"{column_name:<45}"
        f"{count:>10,} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 5. INFINITE VALUE CHECK
# ============================================================

print("\n[5] INFINITE VALUE CHECK")
print("-" * 70)

numeric_columns = con.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'model_features'
      AND data_type IN (
          'DOUBLE',
          'FLOAT',
          'REAL',
          'DECIMAL',
          'BIGINT',
          'INTEGER'
      )
""").fetchall()

infinite_found = False

for (column_name,) in numeric_columns:

    count = con.execute(f"""
        SELECT COUNT(*)
        FROM model_features
        WHERE isinf("{column_name}")
    """).fetchone()[0]

    if count > 0:
        print(f"[WARNING] {column_name}: {count:,} infinite values")
        infinite_found = True

if not infinite_found:
    print("No infinite values found.")


# ============================================================
# 6. IMPORTANT FEATURE SANITY CHECK
# ============================================================

print("\n[6] IMPORTANT FEATURE SANITY CHECK")
print("-" * 70)

features_to_check = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "bureau_count",
    "previous_application_count",
    "installment_count",
    "pos_record_count",
    "credit_card_record_count",
    "bureau_balance_record_count",
]


for feature in features_to_check:

    exists = con.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'model_features'
          AND column_name = '{feature}'
    """).fetchone()[0]

    if exists == 0:
        print(f"[MISSING] {feature}")
        continue

    negative_count = con.execute(f"""
        SELECT COUNT(*)
        FROM model_features
        WHERE "{feature}" < 0
    """).fetchone()[0]

    if negative_count == 0:
        print(f"[OK] {feature}")
    else:
        print(
            f"[CHECK] {feature}: "
            f"{negative_count:,} negative values"
        )


# ============================================================
# 7. RATIO SANITY
# ============================================================

print("\n[7] RATIO SANITY CHECK")
print("-" * 70)

ratio_features = [
    "credit_to_income",
    "annuity_to_income",
    "annuity_to_credit",
    "bureau_debt_to_credit_ratio",
    "avg_payment_ratio",
    "avg_cc_utilization",
    "max_cc_utilization",
]

for feature in ratio_features:

    exists = con.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'model_features'
          AND column_name = '{feature}'
    """).fetchone()[0]

    if exists == 0:
        print(f"[MISSING] {feature}")
        continue

    stats = con.execute(f"""
        SELECT
            MIN("{feature}"),
            MAX("{feature}"),
            AVG("{feature}")
        FROM model_features
        WHERE "{feature}" IS NOT NULL
    """).fetchone()

    print(
        f"{feature:<35}"
        f"min={stats[0]:.4f}  "
        f"max={stats[1]:.4f}  "
        f"mean={stats[2]:.4f}"
    )


# ============================================================
# 8. FEATURE TABLE SUMMARY
# ============================================================

print("\n[8] FEATURE TABLE SUMMARY")
print("-" * 70)

tables = [
    "application_features",
    "bureau_features",
    "previous_application_features",
    "installment_features",
    "pos_features",
    "credit_card_features",
    "bureau_balance_features",
    "model_features",
]

for table in tables:

    count = con.execute(f"""
        SELECT COUNT(*)
        FROM {table}
    """).fetchone()[0]

    print(f"{table:<35} {count:>12,} rows")


# ============================================================
# 9. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION COMPLETE")
print("=" * 70)

if total_rows == unique_ids and not infinite_found:
    print("✓ One row per applicant")
    print("✓ No infinite values")
    print("✓ Feature table is ready for ML preprocessing")
else:
    print("⚠ Review the warnings above before training.")

con.close()
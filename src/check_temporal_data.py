from pathlib import Path
import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Sahil Kumar\Downloads\Home_Credit\data"
)

DB_PATH = BASE_DIR / "home_credit.duckdb"


# ============================================================
# CONNECT
# ============================================================

con = duckdb.connect(str(DB_PATH))

print("=" * 70)
print("HOME CREDIT TEMPORAL / LEAKAGE CHECK")
print("=" * 70)


# ============================================================
# 1. CHECK APPLICATION DATE
# ============================================================

print("\n[1] APPLICATION DATE")
print("-" * 70)

columns = con.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'application_train'
      AND column_name = 'application_date'
""").fetchall()

if not columns:
    print("WARNING: application_date was not found.")
    print("Temporal validation cannot continue.")
    con.close()
    raise SystemExit

print(f"Column found: {columns[0][0]}")
print(f"Data type:    {columns[0][1]}")


# ============================================================
# 2. INSPECT DATE VALUES
# ============================================================

print("\n[2] APPLICATION DATE RANGE")
print("-" * 70)

result = con.execute("""
    SELECT
        MIN(application_date),
        MAX(application_date),
        COUNT(*),
        COUNT(application_date)
    FROM application_train
""").fetchone()

print(f"Minimum date: {result[0]}")
print(f"Maximum date: {result[1]}")
print(f"Total rows:   {result[2]:,}")
print(f"Non-null:     {result[3]:,}")


# ============================================================
# 3. SAMPLE DATES
# ============================================================

print("\n[3] SAMPLE APPLICATION DATES")
print("-" * 70)

sample = con.execute("""
    SELECT
        SK_ID_CURR,
        application_date,
        TARGET
    FROM application_train
    ORDER BY application_date
    LIMIT 10
""").fetchdf()

print(sample.to_string(index=False))


# ============================================================
# 4. CHECK DUPLICATE APPLICATION IDs
# ============================================================

print("\n[4] APPLICATION ID CHECK")
print("-" * 70)

duplicates = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT SK_ID_CURR) AS unique_ids
    FROM application_train
""").fetchone()

print(f"Total rows: {duplicates[0]:,}")
print(f"Unique IDs: {duplicates[1]:,}")

if duplicates[0] == duplicates[1]:
    print("✓ No duplicate application IDs")
else:
    print("✗ Duplicate application IDs detected")


# ============================================================
# 5. CHECK APPLICATION DATE DISTRIBUTION
# ============================================================

print("\n[5] APPLICATIONS BY DATE")
print("-" * 70)

date_distribution = con.execute("""
    SELECT
        application_date,
        COUNT(*) AS applications,
        SUM(TARGET) AS defaults
    FROM application_train
    GROUP BY application_date
    ORDER BY application_date
    LIMIT 20
""").fetchdf()

print(
    date_distribution.to_string(index=False)
)


# ============================================================
# 6. CHECK APPLICATION DATE DUPLICATES
# ============================================================

print("\n[6] DATE DISTRIBUTION SUMMARY")
print("-" * 70)

date_summary = con.execute("""
    SELECT
        COUNT(DISTINCT application_date)
    FROM application_train
""").fetchone()[0]

print(
    f"Distinct application dates: "
    f"{date_summary:,}"
)


# ============================================================
# 7. CHECK AVAILABLE DATE COLUMNS
# ============================================================

print("\n[7] DATE-LIKE COLUMNS IN DATASET")
print("-" * 70)

tables = [
    "application_train",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
]

for table in tables:

    print(f"\n{table}:")

    cols = con.execute(f"""
        SELECT
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_name = '{table}'
          AND (
              LOWER(column_name) LIKE '%date%'
              OR LOWER(column_name) LIKE '%days%'
              OR LOWER(column_name) LIKE '%month%'
          )
        ORDER BY ordinal_position
    """).fetchall()

    if not cols:
        print("  No obvious temporal columns")

    else:

        for column, dtype in cols:
            print(
                f"  {column:35} {dtype}"
            )


# ============================================================
# 8. IMPORTANT HOME CREDIT TIME FIELDS
# ============================================================

print("\n[8] HISTORICAL TIME FIELDS")
print("-" * 70)

print("""
The original Home Credit tables mainly represent historical
information using relative day/month fields.

Important fields include:

bureau:
    DAYS_CREDIT
    DAYS_CREDIT_ENDDATE
    DAYS_ENDDATE_FACT
    DAYS_CREDIT_UPDATE

previous_application:
    DAYS_DECISION
    DAYS_FIRST_DRAWING
    DAYS_FIRST_DUE
    DAYS_LAST_DUE_1ST_VERSION
    DAYS_LAST_DUE
    DAYS_TERMINATION

installments_payments:
    DAYS_INSTALMENT
    DAYS_ENTRY_PAYMENT

POS_CASH_balance:
    MONTHS_BALANCE

credit_card_balance:
    MONTHS_BALANCE

These fields are important for determining whether a record
was available before the current application.
""")


# ============================================================
# 9. CHECK WHETHER application_date IS USABLE
# ============================================================

print("\n[9] APPLICATION DATE USABILITY")
print("-" * 70)

date_type = columns[0][1]

if "DATE" in date_type.upper():

    print("✓ application_date has a DATE-compatible type.")

else:

    print(
        "WARNING: application_date is not stored as a "
        "standard DATE type."
    )


# ============================================================
# 10. FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION STATUS")
print("=" * 70)

print("""
We have confirmed whether application_date exists and
inspected the available historical time fields.

IMPORTANT:

Do NOT automatically use application_date as a model
feature yet.

Do NOT claim that all historical records are valid simply
because they are linked to the applicant.

The next modeling step should use only information that
would have been available at the time of the application.
""")

con.close()

print("=" * 70)
print("TEMPORAL CHECK COMPLETE")
print("=" * 70)
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
# CONNECT TO DUCKDB
# ============================================================

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"DuckDB database not found:\n{DB_PATH}\n\n"
        "Run create_database.py first."
    )

con = duckdb.connect(str(DB_PATH))


print("=" * 70)
print("HOME CREDIT RELATIONSHIP CHECK")
print("=" * 70)


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_relationship(
    number,
    title,
    parent_count,
    child_count,
    parent_label,
    child_label
):
    print(f"\n[{number}] {title}")
    print("-" * 70)

    print(f"{parent_label}: {parent_count:,}")
    print(f"{child_label}:  {child_count:,}")

    if parent_count > 0:
        percentage = (child_count / parent_count) * 100
        print(f"Coverage: {percentage:.2f}%")

        if child_count > 0:
            print("Status:   ✓ RELATIONSHIP EXISTS")
        else:
            print("Status:   ✗ NO MATCHING RECORDS")


# ============================================================
# 1. APPLICATION -> BUREAU
# ============================================================

print("\n[1] APPLICATION -> BUREAU")
print("-" * 70)

result = con.execute("""
    SELECT
        COUNT(DISTINCT a.SK_ID_CURR) AS total_applicants,

        COUNT(DISTINCT b.SK_ID_CURR)
            AS applicants_with_bureau,

        AVG(
            bureau_count
        ) AS avg_bureau_accounts,

        MAX(
            bureau_count
        ) AS max_bureau_accounts

    FROM application_train a

    LEFT JOIN (
        SELECT
            SK_ID_CURR,
            COUNT(*) AS bureau_count
        FROM bureau
        GROUP BY SK_ID_CURR
    ) b
        ON a.SK_ID_CURR = b.SK_ID_CURR

    LEFT JOIN bureau b2
        ON a.SK_ID_CURR = b2.SK_ID_CURR

    GROUP BY a.SK_ID_CURR
""").fetchdf()

total_applicants = len(result)

applicants_with_bureau = con.execute("""
    SELECT COUNT(DISTINCT a.SK_ID_CURR)
    FROM application_train a
    INNER JOIN bureau b
        ON a.SK_ID_CURR = b.SK_ID_CURR
""").fetchone()[0]

avg_bureau = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            SK_ID_CURR,
            COUNT(*) AS cnt
        FROM bureau
        GROUP BY SK_ID_CURR
    )
""").fetchone()[0]

max_bureau = con.execute("""
    SELECT MAX(cnt)
    FROM (
        SELECT
            SK_ID_CURR,
            COUNT(*) AS cnt
        FROM bureau
        GROUP BY SK_ID_CURR
    )
""").fetchone()[0]

print(f"Total applicants:        {total_applicants:,}")
print(f"With bureau history:     {applicants_with_bureau:,}")
print(f"Average bureau records:  {avg_bureau:.2f}")
print(f"Maximum bureau records:  {max_bureau:,}")

if applicants_with_bureau > 0:
    print("Status:                  ✓ RELATIONSHIP EXISTS")


# ============================================================
# 2. BUREAU -> BUREAU BALANCE
# ============================================================

print("\n[2] BUREAU -> BUREAU_BALANCE")
print("-" * 70)

bureau_accounts = con.execute("""
    SELECT COUNT(DISTINCT SK_ID_BUREAU)
    FROM bureau
""").fetchone()[0]

accounts_with_balance = con.execute("""
    SELECT COUNT(DISTINCT b.SK_ID_BUREAU)
    FROM bureau b
    INNER JOIN bureau_balance bb
        ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
""").fetchone()[0]

avg_balance_records = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            SK_ID_BUREAU,
            COUNT(*) AS cnt
        FROM bureau_balance
        GROUP BY SK_ID_BUREAU
    )
""").fetchone()[0]

max_balance_records = con.execute("""
    SELECT MAX(cnt)
    FROM (
        SELECT
            SK_ID_BUREAU,
            COUNT(*) AS cnt
        FROM bureau_balance
        GROUP BY SK_ID_BUREAU
    )
""").fetchone()[0]

print(f"Bureau accounts:          {bureau_accounts:,}")
print(f"With monthly balance:     {accounts_with_balance:,}")
print(f"Average monthly records:  {avg_balance_records:.2f}")
print(f"Maximum monthly records:  {max_balance_records:,}")

if accounts_with_balance > 0:
    print("Status:                   ✓ RELATIONSHIP EXISTS")


# ============================================================
# 3. APPLICATION -> PREVIOUS APPLICATION
# ============================================================

print("\n[3] APPLICATION -> PREVIOUS_APPLICATION")
print("-" * 70)

total_applicants = con.execute("""
    SELECT COUNT(DISTINCT SK_ID_CURR)
    FROM application_train
""").fetchone()[0]

applicants_with_previous = con.execute("""
    SELECT COUNT(DISTINCT a.SK_ID_CURR)
    FROM application_train a
    INNER JOIN previous_application p
        ON a.SK_ID_CURR = p.SK_ID_CURR
""").fetchone()[0]

avg_previous = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            SK_ID_CURR,
            COUNT(*) AS cnt
        FROM previous_application
        GROUP BY SK_ID_CURR
    )
""").fetchone()[0]

max_previous = con.execute("""
    SELECT MAX(cnt)
    FROM (
        SELECT
            SK_ID_CURR,
            COUNT(*) AS cnt
        FROM previous_application
        GROUP BY SK_ID_CURR
    )
""").fetchone()[0]

print(f"Total applicants:         {total_applicants:,}")
print(f"With previous loans:      {applicants_with_previous:,}")
print(f"Average previous loans:   {avg_previous:.2f}")
print(f"Maximum previous loans:   {max_previous:,}")

if applicants_with_previous > 0:
    print("Status:                   ✓ RELATIONSHIP EXISTS")


# ============================================================
# 4. PREVIOUS APPLICATION -> INSTALLMENTS
# ============================================================

print("\n[4] PREVIOUS_APPLICATION -> INSTALLMENTS")
print("-" * 70)

previous_loans = con.execute("""
    SELECT COUNT(DISTINCT SK_ID_PREV)
    FROM previous_application
""").fetchone()[0]

loans_with_installments = con.execute("""
    SELECT COUNT(DISTINCT p.SK_ID_PREV)
    FROM previous_application p
    INNER JOIN installments_payments i
        ON p.SK_ID_PREV = i.SK_ID_PREV
""").fetchone()[0]

avg_installments = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            SK_ID_PREV,
            COUNT(*) AS cnt
        FROM installments_payments
        GROUP BY SK_ID_PREV
    )
""").fetchone()[0]

max_installments = con.execute("""
    SELECT MAX(cnt)
    FROM (
        SELECT
            SK_ID_PREV,
            COUNT(*) AS cnt
        FROM installments_payments
        GROUP BY SK_ID_PREV
    )
""").fetchone()[0]

print(f"Previous loans:           {previous_loans:,}")
print(f"With installments:        {loans_with_installments:,}")
print(f"Average installments:     {avg_installments:.2f}")
print(f"Maximum installments:     {max_installments:,}")

if loans_with_installments > 0:
    print("Status:                   ✓ RELATIONSHIP EXISTS")


# ============================================================
# 5. PREVIOUS APPLICATION -> POS CASH
# ============================================================

print("\n[5] PREVIOUS_APPLICATION -> POS_CASH")
print("-" * 70)

loans_with_pos = con.execute("""
    SELECT COUNT(DISTINCT p.SK_ID_PREV)
    FROM previous_application p
    INNER JOIN POS_CASH_balance pos
        ON p.SK_ID_PREV = pos.SK_ID_PREV
""").fetchone()[0]

avg_pos = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            SK_ID_PREV,
            COUNT(*) AS cnt
        FROM POS_CASH_balance
        GROUP BY SK_ID_PREV
    )
""").fetchone()[0]

max_pos = con.execute("""
    SELECT MAX(cnt)
    FROM (
        SELECT
            SK_ID_PREV,
            COUNT(*) AS cnt
        FROM POS_CASH_balance
        GROUP BY SK_ID_PREV
    )
""").fetchone()[0]

print(f"Previous loans:           {previous_loans:,}")
print(f"With POS history:         {loans_with_pos:,}")
print(f"Average POS records:      {avg_pos:.2f}")
print(f"Maximum POS records:      {max_pos:,}")

if loans_with_pos > 0:
    print("Status:                   ✓ RELATIONSHIP EXISTS")


# ============================================================
# 6. PREVIOUS APPLICATION -> CREDIT CARD
# ============================================================

print("\n[6] PREVIOUS_APPLICATION -> CREDIT_CARD")
print("-" * 70)

loans_with_credit_card = con.execute("""
    SELECT COUNT(DISTINCT p.SK_ID_PREV)
    FROM previous_application p
    INNER JOIN credit_card_balance c
        ON p.SK_ID_PREV = c.SK_ID_PREV
""").fetchone()[0]

avg_credit_card = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            SK_ID_PREV,
            COUNT(*) AS cnt
        FROM credit_card_balance
        GROUP BY SK_ID_PREV
    )
""").fetchone()[0]

max_credit_card = con.execute("""
    SELECT MAX(cnt)
    FROM (
        SELECT
            SK_ID_PREV,
            COUNT(*) AS cnt
        FROM credit_card_balance
        GROUP BY SK_ID_PREV
    )
""").fetchone()[0]

print(f"Previous loans:           {previous_loans:,}")
print(f"With credit card data:    {loans_with_credit_card:,}")
print(f"Average CC records:       {avg_credit_card:.2f}")
print(f"Maximum CC records:       {max_credit_card:,}")

if loans_with_credit_card > 0:
    print("Status:                   ✓ RELATIONSHIP EXISTS")


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\n[7] TARGET DISTRIBUTION")
print("-" * 70)

result = con.execute("""
    SELECT
        TARGET,
        COUNT(*) AS applications,

        ROUND(
            100.0 * COUNT(*)
            / SUM(COUNT(*)) OVER (),
            2
        ) AS percentage

    FROM application_train

    GROUP BY TARGET

    ORDER BY TARGET;
""").fetchall()

for row in result:

    print(
        f"TARGET={row[0]}   "
        f"Applications={row[1]:,}   "
        f"Percentage={row[2]}%"
    )


# ============================================================
# 8. SAMPLE APPLICANT
# ============================================================

print("\n[8] SAMPLE APPLICANT")
print("-" * 70)

applicant = con.execute("""
    SELECT SK_ID_CURR
    FROM application_train
    ORDER BY SK_ID_CURR
    LIMIT 1;
""").fetchone()[0]

print(f"Applicant ID: {applicant}")


# ------------------------------------------------------------
# APPLICATION
# ------------------------------------------------------------

print("\nApplication:")

application = con.execute("""
    SELECT
        SK_ID_CURR,
        TARGET,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        AMT_ANNUITY,
        NAME_INCOME_TYPE,
        NAME_EDUCATION_TYPE

    FROM application_train

    WHERE SK_ID_CURR = ?
""", [applicant]).fetchdf()

print(
    application.to_string(index=False)
)


# ------------------------------------------------------------
# BUREAU
# ------------------------------------------------------------

print("\nBureau records:")

bureau_sample = con.execute("""
    SELECT
        SK_ID_BUREAU,
        CREDIT_ACTIVE,
        DAYS_CREDIT,
        CREDIT_DAY_OVERDUE,
        AMT_CREDIT_SUM,
        AMT_CREDIT_SUM_DEBT

    FROM bureau

    WHERE SK_ID_CURR = ?

    LIMIT 5
""", [applicant]).fetchdf()

if len(bureau_sample) == 0:
    print("No bureau records found.")
else:
    print(
        bureau_sample.to_string(index=False)
    )


# ------------------------------------------------------------
# PREVIOUS APPLICATIONS
# ------------------------------------------------------------

print("\nPrevious applications:")

previous_sample = con.execute("""
    SELECT
        SK_ID_PREV,
        NAME_CONTRACT_STATUS,
        AMT_APPLICATION,
        AMT_CREDIT,
        AMT_ANNUITY,
        DAYS_DECISION

    FROM previous_application

    WHERE SK_ID_CURR = ?

    LIMIT 5
""", [applicant]).fetchdf()

if len(previous_sample) == 0:
    print("No previous applications found.")
else:
    print(
        previous_sample.to_string(index=False)
    )


# ============================================================
# 9. RELATIONSHIP GRAPH
# ============================================================

print("\n[9] RELATIONSHIP GRAPH")
print("-" * 70)

print("""
application_train
       |
       | SK_ID_CURR
       | 1 : N
       v
    bureau
       |
       | SK_ID_BUREAU
       | 1 : N
       v
bureau_balance


application_train
       |
       | SK_ID_CURR
       | 1 : N
       v
previous_application
       |
       +--------------------+
       |                    |
       | SK_ID_PREV         | SK_ID_PREV
       | 1 : N              | 1 : N
       v                    v
installments_payments   POS_CASH_balance
       |
       |
       | SK_ID_PREV
       | 1 : N
       v
credit_card_balance
""")


# ============================================================
# 10. VERIFY RELATIONSHIP METADATA
# ============================================================

print("\n[10] STORED RELATIONSHIP METADATA")
print("-" * 70)

try:

    relationships = con.execute("""
        SELECT
            relationship_id,
            parent_table,
            parent_key,
            child_table,
            child_key,
            relationship_type
        FROM relationships
        ORDER BY relationship_id;
    """).fetchdf()

    print(
        relationships.to_string(index=False)
    )

except Exception:

    print(
        "WARNING: relationships table was not found."
    )

    print(
        "Run create_database.py first to create it."
    )


# ============================================================
# CLOSE
# ============================================================

con.close()


print("\n" + "=" * 70)
print("RELATIONSHIP CHECK COMPLETE")
print("=" * 70)
from pathlib import Path
import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\Sahil Kumar\Downloads\Home_Credit\data")
DB_PATH = BASE_DIR / "home_credit.duckdb"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_parquet(folder_name):
    """
    Find all Parquet files inside a dataset folder.
    """

    folder = BASE_DIR / folder_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    files = sorted(folder.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in:\n{folder}"
        )

    return files


def sql_file_list(files):
    """
    Convert Python file paths into a DuckDB list of strings.
    """

    paths = [
        str(file).replace("\\", "/")
        for file in files
    ]

    return "[" + ", ".join(
        "'" + path.replace("'", "''") + "'"
        for path in paths
    ) + "]"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

datasets = {
    "application_train": "application_train_dated",
    "application_test": "application_test",
    "bureau": "bureau",
    "bureau_balance": "bureau_balance",
    "previous_application": "previous_application",
    "installments_payments": "installments_payments",
    "POS_CASH_balance": "POS_CASH_balance",
    "credit_card_balance": "credit_card_balance",
}


# ============================================================
# FIND ALL PARQUET FILES
# ============================================================

print("=" * 70)
print("SCANNING HOME CREDIT DATASET")
print("=" * 70)

dataset_files = {}

for table_name, folder_name in datasets.items():

    files = find_parquet(folder_name)

    dataset_files[table_name] = files

    print(f"\n{table_name}")
    print("-" * 50)

    for file in files:
        print(file.relative_to(BASE_DIR))


# ============================================================
# CONNECT TO DUCKDB
# ============================================================

print("\n" + "=" * 70)
print("CREATING DUCKDB DATABASE")
print("=" * 70)

con = duckdb.connect(str(DB_PATH))


# ============================================================
# CREATE VIEWS OVER PARQUET FILES
# ============================================================

print("\nCreating data views...\n")

for table_name, files in dataset_files.items():

    file_list = sql_file_list(files)

    query = f"""
        CREATE OR REPLACE VIEW "{table_name}" AS
        SELECT *
        FROM read_parquet({file_list});
    """

    con.execute(query)

    print(f"[OK] {table_name}")


# ============================================================
# CREATE RELATIONSHIP METADATA TABLE
# ============================================================

print("\nCreating relationship metadata...")

con.execute("""
CREATE OR REPLACE TABLE relationships (
    relationship_id INTEGER,
    parent_table VARCHAR,
    parent_key VARCHAR,
    child_table VARCHAR,
    child_key VARCHAR,
    relationship_type VARCHAR,
    description VARCHAR
);
""")


# ============================================================
# INSERT RELATIONSHIPS
# ============================================================

con.execute("""
INSERT INTO relationships VALUES

(
    1,
    'application_train',
    'SK_ID_CURR',
    'bureau',
    'SK_ID_CURR',
    '1:N',
    'One applicant can have multiple bureau credit accounts'
),

(
    2,
    'bureau',
    'SK_ID_BUREAU',
    'bureau_balance',
    'SK_ID_BUREAU',
    '1:N',
    'One bureau credit account can have multiple monthly balance records'
),

(
    3,
    'application_train',
    'SK_ID_CURR',
    'previous_application',
    'SK_ID_CURR',
    '1:N',
    'One applicant can have multiple previous loan applications'
),

(
    4,
    'previous_application',
    'SK_ID_PREV',
    'installments_payments',
    'SK_ID_PREV',
    '1:N',
    'One previous loan can have multiple installment payment records'
),

(
    5,
    'previous_application',
    'SK_ID_PREV',
    'POS_CASH_balance',
    'SK_ID_PREV',
    '1:N',
    'One previous loan can have multiple POS and cash balance records'
),

(
    6,
    'previous_application',
    'SK_ID_PREV',
    'credit_card_balance',
    'SK_ID_PREV',
    '1:N',
    'One previous loan can have multiple credit card balance records'
);
""")


# ============================================================
# DISPLAY RELATIONSHIPS
# ============================================================

print("\n" + "=" * 70)
print("STORED RELATIONSHIPS")
print("=" * 70)

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


# ============================================================
# ROW COUNTS
# ============================================================

print("\n" + "=" * 70)
print("ROW COUNTS")
print("=" * 70)

for table_name in datasets:

    count = con.execute(
        f'SELECT COUNT(*) FROM "{table_name}"'
    ).fetchone()[0]

    print(
        f"{table_name:25} {count:,}"
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_distribution = con.execute("""
    SELECT
        TARGET,
        COUNT(*) AS applications,
        ROUND(
            100.0 * COUNT(*) /
            SUM(COUNT(*)) OVER (),
            2
        ) AS percentage
    FROM application_train
    GROUP BY TARGET
    ORDER BY TARGET;
""").fetchdf()

print(
    target_distribution.to_string(index=False)
)


# ============================================================
# DATABASE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATABASE SUMMARY")
print("=" * 70)

print(f"""
Database:
{DB_PATH}

Data views:
- application_train
- application_test
- bureau
- bureau_balance
- previous_application
- installments_payments
- POS_CASH_balance
- credit_card_balance

Relationship table:
- relationships
""")


# ============================================================
# CLOSE DATABASE
# ============================================================

con.close()


print("=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nHome Credit DuckDB database created successfully.")
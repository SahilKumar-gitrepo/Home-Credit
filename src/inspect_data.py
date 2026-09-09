from pathlib import Path
import duckdb

DATA_DIR = Path(r"C:\Users\Sahil Kumar\Downloads\Home_Credit\data")

con = duckdb.connect()

parquet_files = list(DATA_DIR.rglob("*.parquet"))

print(f"\nFound {len(parquet_files)} parquet files:\n")

for file in parquet_files:
    print("=" * 80)
    print(file.relative_to(DATA_DIR))

    query = f"""
        SELECT COUNT(*) AS row_count
        FROM read_parquet('{file.as_posix()}')
    """

    row_count = con.execute(query).fetchone()[0]

    print(f"Rows: {row_count:,}")

    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{file.as_posix()}')"
    ).fetchdf()

    print("\nColumns:")
    print(schema[["column_name", "column_type"]].to_string(index=False))
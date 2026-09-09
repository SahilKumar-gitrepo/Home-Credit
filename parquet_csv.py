import pandas as pd

# Input Parquet file
input_file = r"C:\Users\Sahil Kumar\Downloads\Home_Credit\data\previous_application\train-00001-of-00002.parquet"

# Output CSV file
output_file = "previous_application_2.csv"

# Read Parquet
df = pd.read_parquet(input_file)

# Save as CSV
df.to_csv(output_file, index=False)

print(f"Successfully converted {input_file} -> {output_file}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")
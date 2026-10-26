from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
print("Hello Ricky")
# Inspect only the yearly files from 2010 to 2018
YEARS = range(2010, 2019)

SAMPLE_ROWS = 1000

for year in YEARS:
    path = DATA_DIR / f"cases_{year}_final.csv.gz"

    print(f"\n{'=' * 70}")
    print(f"YEAR: {year}")
    print(f"FILE: {path.name}")

    if not path.exists():
        print("File not found. Skipping...")
        continue

    try:
        # Read column names only
        columns = pd.read_csv(
            path,
            nrows=0
        ).columns.tolist()

        print(f"\nTotal columns: {len(columns)}")
        print("Column names:")
        print(columns)

        # Read only the first 1,000 rows
        sample = pd.read_csv(
            path,
            nrows=SAMPLE_ROWS,
            low_memory=False
        )

        print("\nSample shape:", sample.shape)

        print("\nData types:")
        print(sample.dtypes)

        print("\nMissing values in sample:")
        print(
            sample.isna()
            .sum()
            .sort_values(ascending=False)
            .head(20)
        )

        print("\nFirst 5 rows:")
        print(sample.head())

        del sample

    except Exception as exc:
        print(f"Could not inspect {path.name}: {exc}")

print("\nInspection completed.")
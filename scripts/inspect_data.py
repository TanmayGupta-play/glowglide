from pathlib import Path
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"

csv_files = sorted(raw_dir.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files\n")

for file in csv_files:
    print("=" * 80)
    print("FILE:", file.name)

    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"SIZE: {size_mb:.2f} MB")

    # Read only a few rows for inspection
    df = pd.read_csv(file, nrows=5)

    print("\nCOLUMNS:")
    for column in df.columns:
        print("-", column)

    print("\nFIRST 5 ROWS:")
    print(df.head())

    print()
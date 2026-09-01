import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "dataset" / "train.csv"

print("Looking for:")
print(file_path)

print("\nFile exists:")
print(file_path.exists())

if file_path.exists():

    df = pd.read_csv(file_path)

    print("\nDataset loaded!")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

else:

    print("ERROR: train.csv not found")
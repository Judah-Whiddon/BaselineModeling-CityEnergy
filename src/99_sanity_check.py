from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

FILES = [
    "denver_2020_core.csv",
    "nyc_2020_core.csv",
    "chicago_2020_core.csv",
]

def sanity(df: pd.DataFrame, name: str):
    print("\n" + "=" * 80)
    print(f"SANITY CHECK: {name}")
    print("=" * 80)

    print("\nSHAPE:")
    print(df.shape)

    print("\nDTYPES:")
    print(df.dtypes)

    print("\nMISSING (%):")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False).head(10))

    print("\nNUMERIC SUMMARY:")
    num = df.select_dtypes("number")
    if not num.empty:
        print(num.describe().round(2))
    else:
        print("No numeric columns detected!")

    print("\nSAMPLE ROWS:")
    print(df.head(3).to_string(index=False))

def main():
    for f in FILES:
        path = OUT / f
        if not path.exists():
            print(f"SKIP (missing): {f}")
            continue

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"SKIP (empty file): {f}")
            continue

        sanity(df, f)



if __name__ == "__main__":
    main()

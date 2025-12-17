from pathlib import Path
import pandas as pd

RAW = Path("../data_raw")

def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suf}")

def main():
    files = [p for p in RAW.iterdir() if p.is_file()]
    print(f"Found {len(files)} file(s)\n")

    for p in files:
        print("=" * 80)
        print(p.name)
        try:
            df = read_any(p)
            print(f"shape: {df.shape}")
            print("\ncolumns:")
            for c in df.columns:
                print(f"  - {c}")
            print("\npreview:")
            print(df.head(5).to_string(index=False))
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()

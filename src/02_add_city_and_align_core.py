from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

CORES = {
    "denver": "denver_2020_core.csv",
    "nyc": "nyc_2020_core.csv",
    "chicago": "chicago_2020_core.csv",
    "philadelphia": "philadelphia_2020_core.csv",
}

def main():
    for city, fname in CORES.items():
        path = OUT / fname
        if not path.exists():
            print(f"SKIP missing: {fname}")
            continue

        df = pd.read_csv(path)

        # add city column
        df.insert(0, "city", city)

        # overwrite (or write to a new file if you prefer)
        df.to_csv(path, index=False)
        print(f"Updated {fname}: added city='{city}' ({df.shape[0]} rows)")

if __name__ == "__main__":
    main()

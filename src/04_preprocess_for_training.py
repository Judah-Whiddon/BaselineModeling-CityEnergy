from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"

INFILE  = OUT / "all_cities_2020_core_standardized.csv"
OUTFILE = OUT / "train_ready_2020_regression.csv"

def main():
    df = pd.read_csv(INFILE, low_memory=False)

    # For the FIRST baseline, exclude Chicago (schema differs; bring back later)
    df = df[df["city"].isin(["denver", "nyc", "philadelphia"])].copy()

    # Keep only columns we’ll use
    use = ["city", "site_eui", "energy_star_score", "gross_floor_area_sqft", "total_ghg_mtco2e"]
    df = df[use].copy()

    # Drop rows missing target
    df = df.dropna(subset=["total_ghg_mtco2e"])

    # Coerce numerics
    for c in ["site_eui", "energy_star_score", "gross_floor_area_sqft", "total_ghg_mtco2e"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing features (baseline simplicity)
    df = df.dropna(subset=["site_eui", "energy_star_score"])

    # Light outlier guard (keeps training stable; we can do better later)
    df = df[df["site_eui"] >= 0]
    df.loc[df["site_eui"] > 1000, "site_eui"] = 1000  # cap extreme EUI
    df.loc[df["total_ghg_mtco2e"] < 0, "total_ghg_mtco2e"] = np.nan
    df = df.dropna(subset=["total_ghg_mtco2e"])

    # Add a log target for easier learning
    df["y_log_ghg"] = np.log1p(df["total_ghg_mtco2e"])

    df.to_csv(OUTFILE, index=False)
    print(f"Wrote {OUTFILE} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(df.groupby("city").size())

if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "outputs"

FILES = {
    "denver": OUT / "denver_2020_core.csv",
    "nyc": OUT / "nyc_2020_core.csv",
    "philadelphia": OUT / "philadelphia_2020_core.csv",
    "chicago": OUT / "chicago_2020_core.csv",  # will be loaded, but not used for training later
}

# Map NYC columns -> standard schema
NYC_MAP = {
    "Property_Id": "property_id",
    "Property_Name": "property_name",
    "Primary_Property_Type___Self_Selected": "property_type",
    "Site_EUI__kBtu_ft__": "site_eui",
    "ENERGY_STAR_Score": "energy_star_score",
    "Total_GHG_Emissions__Metric_Tons_CO2e_": "total_ghg_mtco2e",
    "Property_GFA___Self_Reported__ft__": "gross_floor_area_sqft",
    "Postcode": "zip",
    "city": "city",
}

def standardize_city_df(city: str, df: pd.DataFrame) -> pd.DataFrame:
    # If NYC, rename
    if city == "nyc":
        df = df.rename(columns={k: v for k, v in NYC_MAP.items() if k in df.columns})

    # For other cities, we already exported snake_case, but keep a small fallback:
    fallback = {
        "PROPERTY_NAME": "property_name",
        "PHILADELPHIA_BUILDING_ID": "property_id",
        "MASTER_PROPERTY_TYPE": "property_type",
    }
    df = df.rename(columns={k: v for k, v in fallback.items() if k in df.columns})

    # Ensure city exists
    if "city" not in df.columns:
        df.insert(0, "city", city)

    # Keep a standard set (some may be missing per city)
    cols = [
        "city",
        "property_id",
        "property_name",
        "property_type",
        "gross_floor_area_sqft",
        "site_eui",
        "energy_star_score",
        "total_ghg_mtco2e",
        "zip",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()

    # Coerce numerics (safe)
    for c in ["gross_floor_area_sqft", "site_eui", "energy_star_score", "total_ghg_mtco2e"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def main():
    frames = []
    for city, path in FILES.items():
        if not path.exists():
            print(f"SKIP missing: {path.name}")
            continue
        df = pd.read_csv(path, low_memory=False)
        frames.append(standardize_city_df(city, df))

    all_core = pd.concat(frames, ignore_index=True)

    out_path = OUT / "all_cities_2020_core_standardized.csv"
    all_core.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({all_core.shape[0]} rows, {all_core.shape[1]} cols)")
    print(all_core.groupby("city").size())

if __name__ == "__main__":
    main()

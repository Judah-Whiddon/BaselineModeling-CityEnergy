from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data_raw"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FILE = RAW / "chicago_benchmarking_2020.csv"

def main():
    df = pd.read_csv(FILE, low_memory=False)

    # Keep only what we can confidently use for baseline modeling
    keep = [
        "ID",
        "Property_Name",
        "Data_Year",
        "Primary_Property_Type",
        "Gross_Floor_Area___Buildings__sq_ft_",
        "ENERGY_STAR_Score",
        "Electricity_Use__kBtu_",
        "Natural_Gas_Use__kBtu_",
        "Water_Use__kGal_",
        "Chicago_Energy_Rating",
        "Address",
        "ZIP_Code",
        "Year_Built",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    # Filter to 2020 (safe because we have Data_Year)
    out["Data_Year"] = pd.to_numeric(out["Data_Year"], errors="coerce")
    out = out[out["Data_Year"] == 2020]

    # Rename to align with our “core” schema
    rename = {
        "ID": "property_id",
        "Property_Name": "property_name",
        "Data_Year": "year",
        "Primary_Property_Type": "property_type",
        "Gross_Floor_Area___Buildings__sq_ft_": "gross_floor_area_sqft",
        "ENERGY_STAR_Score": "energy_star_score",
        "Chicago_Energy_Rating": "chicago_energy_rating",
        "Electricity_Use__kBtu_": "electricity_kbtu",
        "Natural_Gas_Use__kBtu_": "natural_gas_kbtu",
        "Water_Use__kGal_": "water_kgal",
        "Address": "address",
        "ZIP_Code": "zip",
        "Year_Built": "year_built",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    # Coerce numerics we care about (prevents strings sneaking in)
    for c in ["gross_floor_area_sqft", "energy_star_score", "chicago_energy_rating",
              "electricity_kbtu", "natural_gas_kbtu", "water_kgal", "year_built"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out_path = OUT / "chicago_2020_core.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out.shape[0]} rows, {out.shape[1]} cols)")

if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data_raw"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FILE = RAW / "denver_energy_2020.csv"

def main():
    df = pd.read_csv(FILE, low_memory=False)

    # Denver columns we KNOW exist from your matched output
    keep = [
        "Denver_Building_Id",
        "Property_Name",
        "Master_Property_Type",
        "Site_EUI__kBtu_sq_ft_",
        "ENERGY_STAR_Score",
        "Total_GHG_Emissions__Metric_Tons_C02e_",
        "Address_Source",
        "Zipcode",
    ]
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()

    # Rename to consistent schema
    rename = {
        "Denver_Building_Id": "property_id",
        "Property_Name": "property_name",
        "Master_Property_Type": "property_type",
        "Site_EUI__kBtu_sq_ft_": "site_eui",
        "ENERGY_STAR_Score": "energy_star_score",
        "Total_GHG_Emissions__Metric_Tons_C02e_": "total_ghg_mtco2e",
        "Address_Source": "address",
        "Zipcode": "zip",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    out_path = OUT / "denver_2020_core.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out.shape[0]} rows, {out.shape[1]} cols)")

if __name__ == "__main__":
    main()

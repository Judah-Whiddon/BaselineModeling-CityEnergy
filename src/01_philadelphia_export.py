from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data_raw"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FILE = RAW / "philadelphia_energy_2020.csv"

def main():
    df = pd.read_csv(FILE, low_memory=False)

    keep = [
        "PHILADELPHIA_BUILDING_ID",
        "PROPERTY_NAME",
        "TOTAL_FLOOR_AREA_BLD_PK_FT2",
        "SITE_EUI_KBTUFT2",
        "ENERGY_STAR_SCORE",
        "TOTAL_GHG_EMISSIONS_MTCO2E",
        "STREET_ADDRESS",
        "POSTAL_CODE",
        "YEAR_BUILT",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    rename = {
        "PHILADELPHIA_BUILDING_ID": "property_id",
        "PROPERTY_NAME": "property_name",
        "TOTAL_FLOOR_AREA_BLD_PK_FT2": "gross_floor_area_sqft",
        "SITE_EUI_KBTUFT2": "site_eui",
        "ENERGY_STAR_SCORE": "energy_star_score",
        "TOTAL_GHG_EMISSIONS_MTCO2E": "total_ghg_mtco2e",
        "STREET_ADDRESS": "address",
        "POSTAL_CODE": "zip",
        "YEAR_BUILT": "year_built",
    }
    out = out.rename(columns=rename)

    for c in ["gross_floor_area_sqft", "site_eui", "energy_star_score", "total_ghg_mtco2e", "year_built"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out_path = OUT / "philadelphia_2020_core.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out.shape[0]} rows, {out.shape[1]} cols)")

if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data_raw"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

NYC_FILE = RAW / "nyc_energy_and_water_2020.csv"   # adjust if your filename differs

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    # make column names easier to work with, but keep your originals in the raw file
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def subset_core(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Property_Id",
        "Property_Name",
        "Year_Ending",
        "Borough",
        "Postcode",
        "Primary_Property_Type___Self_Selected",
        "Largest_Property_Use_Type",
        "Largest_Property_Use_Type___Gross_Floor_Area__ft__",
        "Property_GFA___Self_Reported__ft__",
        "Site_EUI__kBtu_ft__",
        "Source_EUI__kBtu_ft__",
        "ENERGY_STAR_Score",
        "Total_GHG_Emissions__Metric_Tons_CO2e_",
        "Total_GHG_Emissions_Intensity__kgCO2e_ft__",
        "Water_Use__All_Water_Sources___kgal_",
        "Latitude",
        "Longitude",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    num_cols = [
        "Largest_Property_Use_Type___Gross_Floor_Area__ft__",
        "Property_GFA___Self_Reported__ft__",
        "Site_EUI__kBtu_ft__",
        "Source_EUI__kBtu_ft__",
        "ENERGY_STAR_Score",
        "Total_GHG_Emissions__Metric_Tons_CO2e_",
        "Total_GHG_Emissions_Intensity__kgCO2e_ft__",
        "Water_Use__All_Water_Sources___kgal_",
        "Latitude",
        "Longitude",
    ]
    out = to_numeric_safe(out, num_cols)

    # drop rows that aren't actually 2020 ending (optional; comment out if already filtered)
    if "Year_Ending" in out.columns:
        out["Year_Ending"] = pd.to_datetime(out["Year_Ending"], errors="coerce")
        out = out[out["Year_Ending"].dt.year == 2020]

    return out

def subset_emissions(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Property_Id",
        "Year_Ending",
        "Borough",
        "Primary_Property_Type___Self_Selected",
        "Direct_GHG_Emissions__Metric_Tons_CO2e_",
        "Indirect_GHG_Emissions__Metric_Tons_CO2e_",
        "Total_GHG_Emissions__Metric_Tons_CO2e_",
        "Net_Emissions__Metric_Tons_CO2e_",
        "Total_GHG_Emissions_Intensity__kgCO2e_ft__",
        "Direct_GHG_Emissions_Intensity__kgCO2e_ft__",
        "Indirect_GHG_Emissions_Intensity__kgCO2e_ft__",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    num_cols = [c for c in keep if "Emissions" in c or "Intensity" in c]
    out = to_numeric_safe(out, num_cols)

    if "Year_Ending" in out.columns:
        out["Year_Ending"] = pd.to_datetime(out["Year_Ending"], errors="coerce")
        out = out[out["Year_Ending"].dt.year == 2020]

    return out

def subset_counts_by_type(df: pd.DataFrame) -> pd.DataFrame:
    # simple sanity-check table: how many properties per primary type + borough
    cols = ["Primary_Property_Type___Self_Selected", "Borough"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    if "Year_Ending" in df.columns:
        y = pd.to_datetime(df["Year_Ending"], errors="coerce")
        out = out[y.dt.year == 2020]

    grouped = (
        out.groupby(cols, dropna=False)
           .size()
           .reset_index(name="property_count")
           .sort_values("property_count", ascending=False)
    )
    return grouped

def main():
    df = pd.read_csv(NYC_FILE, low_memory=False)
    df = clean_cols(df)

    core = subset_core(df)
    emissions = subset_emissions(df)
    counts = subset_counts_by_type(df)

    core_path = OUT / "nyc_2020_core.csv"
    emissions_path = OUT / "nyc_2020_emissions.csv"
    counts_path = OUT / "nyc_2020_counts_by_type_and_borough.csv"

    core.to_csv(core_path, index=False)
    emissions.to_csv(emissions_path, index=False)
    counts.to_csv(counts_path, index=False)

    print(f"Wrote {core_path}      ({core.shape[0]} rows, {core.shape[1]} cols)")
    print(f"Wrote {emissions_path} ({emissions.shape[0]} rows, {emissions.shape[1]} cols)")
    print(f"Wrote {counts_path}    ({counts.shape[0]} rows, {counts.shape[1]} cols)")

if __name__ == "__main__":
    main()

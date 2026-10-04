"""
Standalone Indian Tier-3 Cities Historical Weather Data Extractor for VAJRA-AI.
Queries Open-Meteo Archive API (2021-2026), computes lightning proxy scores,
and exports hourly and daily rollup Parquet and CSV files into `vajra_ai_data/`.
"""

from __future__ import annotations

import os
import sys
import time
import pandas as pd
import requests
from tqdm import tqdm

# ---------- CONFIG ----------
EXCEL_FILE = "india_tier_3_city_coordinates.xlsx"
START_DATE = "2021-01-01"
END_DATE   = "2026-01-01"
TIMEZONE   = "Asia/Kolkata"
OUTPUT_DIR = "vajra_ai_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOURLY_VARS = [
    "precipitation", "rain", "weathercode",
    "windspeed_10m", "windgusts_10m",
    "temperature_2m", "relativehumidity_2m",
    "cape", "lifted_index", "convective_inhibition"
]

# ---------- LOAD CITIES FROM EXCEL ----------
print(f"Loading cities from {EXCEL_FILE}...")
df_cities = pd.read_excel(EXCEL_FILE, sheet_name="Coordinates", skiprows=4)
df_cities = df_cities.dropna(subset=["City", "Latitude", "Longitude"])
cities = df_cities[["City", "State", "Latitude", "Longitude"]].to_dict("records")
print(f"Loaded {len(cities)} cities from Excel.")


# ---------- FETCH FUNCTION WITH RETRIES ----------
def fetch_location(city: dict, max_retries: int = 5) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city["Latitude"],
        "longitude": city["Longitude"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    
    data = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                wait_time = attempt * 6
                print(f"[RETRY] 429 Rate limited on {city['City']}, waiting {wait_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as err:
            if attempt == max_retries:
                raise err
            time.sleep(3)

    if not data or "hourly" not in data:
        raise RuntimeError(f"Failed to fetch hourly weather data for {city['City']} after {max_retries} retries (Rate Limited).")

    df = pd.DataFrame(data["hourly"])
    df.rename(columns={
        "time": "timestamp",
        "weathercode": "weather_code",
        "windspeed_10m": "wind_speed_10m",
        "windgusts_10m": "wind_gusts_10m",
        "relativehumidity_2m": "humidity_2m",
    }, inplace=True)

    # Metadata
    df["city"] = city["City"]
    df["state"] = city["State"]
    df["latitude"] = city["Latitude"]
    df["longitude"] = city["Longitude"]

    # Thunderstorm flag (WMO code 95+)
    df["is_thunderstorm"] = df["weather_code"].isin([95, 96, 99]).astype(int)

    # Lightning proxy score (0-1)
    cape_norm = (df["cape"].fillna(0).clip(0, 3000) / 3000.0)
    precip_norm = (df["precipitation"].fillna(0).clip(0, 20) / 20.0)
    df["lightning_proxy"] = (0.6 * cape_norm + 0.3 * precip_norm +
                             0.1 * df["is_thunderstorm"]).round(4)

    # Reorder
    cols = ["timestamp", "city", "state", "latitude", "longitude",
            "precipitation", "rain", "weather_code", "is_thunderstorm",
            "wind_speed_10m", "wind_gusts_10m",
            "temperature_2m", "humidity_2m",
            "cape", "lifted_index", "convective_inhibition",
            "lightning_proxy"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def run_pipeline():
    all_dfs = []
    failed = []

    for city in tqdm(cities, desc="Fetching cities"):
        try:
            df = fetch_location(city)
            all_dfs.append(df)
            print(f"[SUCCESS] {city['City']}: {len(df)} rows")
        except Exception as e:
            failed.append(city["City"])
            print(f"[FAILED] {city['City']}: {e}")
        time.sleep(1.2)  # Rate limit safety

    if failed:
        print(f"\n[WARNING] Failed cities: {failed}")

    if not all_dfs:
        print("[ERROR] No data fetched!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # ---------- SAVE HOURLY DATA ----------
    hourly_parquet = f"{OUTPUT_DIR}/vajra_ai_hourly.parquet"
    hourly_csv = f"{OUTPUT_DIR}/vajra_ai_hourly.csv"
    combined.to_parquet(hourly_parquet, index=False)
    combined.to_csv(hourly_csv, index=False)

    # ---------- DAILY ROLLUP ----------
    combined["date"] = pd.to_datetime(combined["timestamp"]).dt.date
    daily = combined.groupby(["city", "state", "latitude", "longitude", "date"]).agg(
        rain_total_mm       = ("rain", "sum"),
        precip_total_mm     = ("precipitation", "sum"),
        wind_speed_max      = ("wind_speed_10m", "max"),
        wind_gust_max       = ("wind_gusts_10m", "max"),
        cape_max            = ("cape", "max"),
        thunderstorm_hours  = ("is_thunderstorm", "sum"),
        lightning_proxy_max = ("lightning_proxy", "max"),
    ).reset_index()

    daily_parquet = f"{OUTPUT_DIR}/vajra_ai_daily.parquet"
    daily_csv = f"{OUTPUT_DIR}/vajra_ai_daily.csv"
    daily.to_parquet(daily_parquet, index=False)
    daily.to_csv(daily_csv, index=False)

    print(f"\n[SUMMARY] Total hourly rows: {len(combined):,}")
    print(f"[SUMMARY] Total daily rows : {len(daily):,}")
    print(f"[SUMMARY] Files saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_pipeline()

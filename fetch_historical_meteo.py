"""
Fetch Historical Atmospheric Data for VAJRA-AI from Open-Meteo Archive API (2021 to 2026).

Saves hourly precipitation, rain, weathercode, wind speed, gusts, and CAPE variables to `vajra_ai_historical.parquet`.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import requests


def fetch_and_save_historical_meteo(
    lat: float = 28.61,
    lon: float = 77.23,
    start_date: str = "2021-01-01",
    end_date: str = "2026-01-01",
    output_filename: str = "vajra_ai_historical.parquet"
) -> str:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,rain,weathercode,windspeed_10m,wind_gusts_10m,cape",
        "timezone": "Asia/Kolkata"
    }

    print(f"Querying Open-Meteo Archive API for ({lat} deg N, {lon} deg E) from {start_date} to {end_date}...")
    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo API returned status code {response.status_code}: {response.text}")

    data = response.json()
    hourly = data.get("hourly", {})

    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "precipitation": hourly["precipitation"],
        "rain": hourly["rain"],
        "weather_code": hourly["weathercode"],
        "wind_speed_10m": hourly["windspeed_10m"],
        "wind_gusts_10m": hourly["wind_gusts_10m"],
        "cape": hourly["cape"]
    })

    # Save to Parquet format
    df.to_parquet(output_filename, index=False)
    
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"Successfully downloaded {len(df):,} hourly weather observation records.")
    print(f"Saved to '{output_filename}' ({file_size_mb:.2f} MB).")
    print("\n--- Dataset Preview ---")
    print(df.head())
    print("\n--- Summary Statistics ---")
    print(df.describe())

    return output_filename


if __name__ == "__main__":
    fetch_and_save_historical_meteo()

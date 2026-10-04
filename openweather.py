"""
OpenWeather API Ingestion & Normalization Adapter for VAJRA-AI Backend.

Features:
- Secure API key handling via OPENWEATHER_API_KEY environment variable.
- Location queries by (lat, lon) or city name (with geocoding).
- Stable normalized weather schema: temperature, humidity, pressure, wind, rain, condition, forecast, location.
- Health state tracking: LIVE, STALE, DEGRADED, UNAVAILABLE, NOT_CONFIGURED.
- In-memory caching with configurable TTL (default 10 min / 600s).
- Independent of 6-channel ML model contract (does not alter model input shapes).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import config as C
from ingestion import BaseIngestionAdapter, IngestionResult, SourceHealthState, SourceHealthStatus


def load_env_file():
    """Loads variables from local .env file if not already present in environment."""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        key = k.strip()
                        val = v.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
        except Exception:
            pass


load_env_file()


class OpenWeatherAdapter(BaseIngestionAdapter):
    """Secure, cached, and normalized adapter for OpenWeather API."""

    def __init__(self, cache_ttl: float = 600.0):
        super().__init__(
            source_id="openweather",
            name="OpenWeather Real-Time Weather & Forecast API",
            channels=["temperature", "humidity", "pressure", "wind", "rain", "condition"],
            env_var_key="OPENWEATHER_API_KEY"
        )
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get_api_key(self) -> str:
        if self.env_var_key not in os.environ:
            load_env_file()
        return os.environ.get(self.env_var_key, "").strip()

    def is_configured(self) -> bool:
        return bool(self.get_api_key())

    def geocode_city(self, city: str, timeout: float = 5.0) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
        """Geocodes a city name to (lat, lon, name, country)."""
        api_key = self.get_api_key()
        if not api_key:
            return None, None, None, None

        encoded_city = urllib.parse.quote(city)
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={encoded_city}&limit=1&appid={api_key}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VAJRA-AI-OpenWeatherAdapter/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        return float(item.get("lat")), float(item.get("lon")), item.get("name"), item.get("country")
        except Exception:
            pass
        return None, None, None, None

    def fetch_weather(
        self,
        lat: float = 18.5204,
        lon: float = 73.8567,
        city: Optional[str] = None,
        units: str = "metric",
        force_refresh: bool = False,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Fetches, normalizes, and caches OpenWeather data.
        Returns stable weather schema dictionary with health status.
        """
        t0 = time.time()
        self.last_attempt_time = t0
        api_key = self.get_api_key()

        if not api_key:
            self.state = SourceHealthState.NOT_CONFIGURED
            self.last_error = "OPENWEATHER_API_KEY environment variable is not set."
            return {
                "status": "error",
                "health_state": SourceHealthState.NOT_CONFIGURED.value,
                "error": self.last_error,
                "timestamp": t0
            }

        # Resolve geocoding if city is provided
        resolved_name = None
        country_code = None
        if city:
            c_lat, c_lon, c_name, c_country = self.geocode_city(city, timeout=timeout)
            if c_lat is not None and c_lon is not None:
                lat, lon = c_lat, c_lon
                resolved_name = c_name
                country_code = c_country

        cache_key = f"{round(lat, 2)}_{round(lon, 2)}_{units}"

        # Check Cache
        if not force_refresh and cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            age = t0 - cached_entry["_cached_at"]
            if age <= self.cache_ttl:
                # Return cached normalized payload
                data = cached_entry["data"].copy()
                data["timestamp"]["cached"] = True
                data["timestamp"]["cache_age_sec"] = round(age, 1)
                self.state = SourceHealthState.LIVE if age < 300.0 else SourceHealthState.STALE
                data["health_state"] = self.state.value
                return data

        # Construct OpenWeather API URLs
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units={units}&appid={api_key}"
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units={units}&appid={api_key}"

        current_data = None
        forecast_data = None
        fetch_error = None

        # 1. Fetch Current Weather
        try:
            req_w = urllib.request.Request(weather_url, headers={"User-Agent": "VAJRA-AI-OpenWeatherAdapter/1.0"})
            with urllib.request.urlopen(req_w, timeout=timeout) as resp:
                self.last_latency_ms = (time.time() - t0) * 1000.0
                if resp.status == 200:
                    current_data = json.loads(resp.read().decode("utf-8"))
                else:
                    fetch_error = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            self.last_latency_ms = (time.time() - t0) * 1000.0
            if e.code == 401:
                fetch_error = "HTTP 401 Unauthorized (Invalid OPENWEATHER_API_KEY)"
            elif e.code == 429:
                fetch_error = "HTTP 429 Rate Limit Exceeded"
            else:
                fetch_error = f"HTTP Error {e.code}: {e.reason}"
        except Exception as e:
            self.last_latency_ms = (time.time() - t0) * 1000.0
            fetch_error = f"Network Exception: {type(e).__name__} - {str(e)}"

        if not current_data or fetch_error:
            self.state = SourceHealthState.UNAVAILABLE
            self.last_error = fetch_error or "Failed to fetch weather data"
            # Return cached data if available on failure
            if cache_key in self.cache:
                fallback_data = self.cache[cache_key]["data"].copy()
                fallback_data["health_state"] = SourceHealthState.STALE.value
                fallback_data["error_warning"] = self.last_error
                return fallback_data
            return {
                "status": "error",
                "health_state": SourceHealthState.UNAVAILABLE.value,
                "error": self.last_error,
                "timestamp": time.time()
            }

        # 2. Fetch Forecast (Optional / Best Effort)
        try:
            req_f = urllib.request.Request(forecast_url, headers={"User-Agent": "VAJRA-AI-OpenWeatherAdapter/1.0"})
            with urllib.request.urlopen(req_f, timeout=timeout) as resp_f:
                if resp_f.status == 200:
                    forecast_data = json.loads(resp_f.read().decode("utf-8"))
        except Exception:
            # Forecast failure does not fail the primary weather endpoint
            pass

        # Normalize into stable weather schema
        normalized = self._normalize_schema(current_data, forecast_data, lat, lon, resolved_name, country_code, units)

        self.last_success_time = time.time()
        self.last_error = None
        self.state = SourceHealthState.LIVE if forecast_data else SourceHealthState.DEGRADED

        normalized["health_state"] = self.state.value

        # Store in Cache
        self.cache[cache_key] = {
            "_cached_at": time.time(),
            "data": normalized
        }

        return normalized

    def _normalize_schema(
        self,
        curr: Dict[str, Any],
        fc: Optional[Dict[str, Any]],
        lat: float,
        lon: float,
        city_override: Optional[str] = None,
        country_override: Optional[str] = None,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """Normalizes OpenWeather API JSON into stable schema."""
        main_info = curr.get("main", {})
        wind_info = curr.get("wind", {})
        rain_info = curr.get("rain", {})
        weather_list = curr.get("weather", [{}])
        cond = weather_list[0] if len(weather_list) > 0 else {}
        sys_info = curr.get("sys", {})

        # Temperature
        temp_c = main_info.get("temp", 0.0)
        feels_like = main_info.get("feels_like", temp_c)
        temp_min = main_info.get("temp_min", temp_c)
        temp_max = main_info.get("temp_max", temp_c)

        # Wind
        wind_ms = float(wind_info.get("speed", 0.0))
        wind_deg = int(wind_info.get("deg", 0))
        wind_gust = float(wind_info.get("gust")) if "gust" in wind_info else None

        # Rain
        rain_1h = float(rain_info.get("1h", 0.0))
        rain_3h = float(rain_info.get("3h", 0.0))

        # Condition
        w_main = cond.get("main", "Clear")
        w_desc = cond.get("description", "clear sky")
        w_icon = cond.get("icon", "01d")

        # Location
        city_name = city_override or curr.get("name") or "Unknown"
        country = country_override or sys_info.get("country") or "IN"

        # Forecast items (up to 8 steps / 24 hours)
        forecast_items = []
        if fc and "list" in fc:
            for item in fc["list"][:8]:
                f_main = item.get("main", {})
                f_w = item.get("weather", [{}])[0]
                f_rain = item.get("rain", {})
                forecast_items.append({
                    "dt": item.get("dt"),
                    "dt_txt": item.get("dt_txt"),
                    "temp_c": float(f_main.get("temp", 0.0)),
                    "feels_like_c": float(f_main.get("feels_like", 0.0)),
                    "humidity": int(f_main.get("humidity", 0)),
                    "condition": f_w.get("main", "Clear"),
                    "description": f_w.get("description", ""),
                    "icon": f_w.get("icon", "01d"),
                    "rain_3h_mm": float(f_rain.get("3h", 0.0))
                })

        return {
            "status": "ok",
            "health_state": SourceHealthState.LIVE.value,
            "provider": "OpenWeather API",
            "location": {
                "city": city_name,
                "country": country,
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4)
            },
            "temperature": {
                "current_c": round(float(temp_c), 1),
                "feels_like_c": round(float(feels_like), 1),
                "temp_min_c": round(float(temp_min), 1),
                "temp_max_c": round(float(temp_max), 1),
                "unit": "°C" if units == "metric" else units
            },
            "humidity": {
                "percentage": int(main_info.get("humidity", 0))
            },
            "pressure": {
                "hpa": float(main_info.get("pressure", 1013.0))
            },
            "wind": {
                "speed_ms": round(wind_ms, 1),
                "speed_kmh": round(wind_ms * 3.6, 1),
                "deg": wind_deg,
                "gust_ms": round(wind_gust, 1) if wind_gust is not None else None
            },
            "rain": {
                "rain_1h_mm": round(rain_1h, 1),
                "rain_3h_mm": round(rain_3h, 1)
            },
            "condition": {
                "main": w_main,
                "description": w_desc,
                "icon": w_icon,
                "icon_url": f"http://openweathermap.org/img/wn/{w_icon}@2x.png"
            },
            "forecast": forecast_items,
            "timestamp": {
                "observation_time": curr.get("dt", int(time.time())),
                "fetch_time": time.time(),
                "cached": False
            }
        }


# Singleton OpenWeather Adapter
OPENWEATHER_ADAPTER = OpenWeatherAdapter(cache_ttl=600.0)

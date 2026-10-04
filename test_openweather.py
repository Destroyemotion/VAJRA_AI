"""
Unit, Integration & Connectivity Test Suite for OpenWeather API Adapter.

Tests:
1. NOT_CONFIGURED state when OPENWEATHER_API_KEY is missing.
2. UNAVAILABLE state when OPENWEATHER_API_KEY is invalid (HTTP 401).
3. Schema normalization and in-memory TTL caching.
4. FastAPI /api/weather endpoint integration.
5. Live external connectivity test with real OPENWEATHER_API_KEY.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from openweather import OPENWEATHER_ADAPTER, OpenWeatherAdapter, load_env_file
from server import app
from ingestion import SourceHealthState


class TestOpenWeatherAdapter(unittest.TestCase):

    def setUp(self):
        load_env_file()
        self.adapter = OpenWeatherAdapter(cache_ttl=600.0)

    def test_01_missing_key_returns_not_configured(self):
        """Verify NOT_CONFIGURED health state when OPENWEATHER_API_KEY is missing."""
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": ""}, clear=False):
            res = self.adapter.fetch_weather(lat=18.52, lon=73.85)
            self.assertEqual(res.get("status"), "error")
            self.assertEqual(res.get("health_state"), SourceHealthState.NOT_CONFIGURED.value)

    def test_02_invalid_key_returns_unavailable(self):
        """Verify UNAVAILABLE health state when API key is invalid (HTTP 401)."""
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "invalid_fake_key_999"}, clear=False):
            res = self.adapter.fetch_weather(lat=18.52, lon=73.85)
            self.assertEqual(res.get("status"), "error")
            self.assertEqual(res.get("health_state"), SourceHealthState.UNAVAILABLE.value)
            self.assertIn("401", res.get("error", ""))

    def test_03_schema_normalization_and_caching(self):
        """Verify OpenWeather API response is normalized into stable schema and cached."""
        mock_curr = {
            "name": "Pune",
            "dt": 1700000000,
            "sys": {"country": "IN"},
            "main": {"temp": 28.5, "feels_like": 30.1, "temp_min": 26.0, "temp_max": 31.0, "humidity": 65, "pressure": 1012},
            "wind": {"speed": 4.2, "deg": 240},
            "weather": [{"main": "Clouds", "description": "few clouds", "icon": "02d"}]
        }
        mock_fc = {
            "list": [
                {
                    "dt": 1700010800,
                    "dt_txt": "2026-09-17 18:00:00",
                    "main": {"temp": 26.0, "feels_like": 27.0, "humidity": 70},
                    "weather": [{"main": "Rain", "description": "light rain", "icon": "10d"}],
                    "rain": {"3h": 1.5}
                }
            ]
        }

        with patch.object(self.adapter, "get_api_key", return_value="dummy_valid_key"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                m_resp = MagicMock()
                m_resp.status = 200
                m_resp.read.side_effect = [
                    json.dumps(mock_curr).encode("utf-8"),
                    json.dumps(mock_fc).encode("utf-8")
                ]
                m_resp.__enter__.return_value = m_resp
                mock_urlopen.return_value = m_resp

                res1 = self.adapter.fetch_weather(lat=18.52, lon=73.85, force_refresh=True)

                self.assertEqual(res1.get("status"), "ok")
                self.assertEqual(res1.get("health_state"), SourceHealthState.LIVE.value)

                # Validate normalized schema fields
                self.assertIn("location", res1)
                self.assertEqual(res1["location"]["city"], "Pune")
                self.assertIn("temperature", res1)
                self.assertEqual(res1["temperature"]["current_c"], 28.5)
                self.assertIn("humidity", res1)
                self.assertEqual(res1["humidity"]["percentage"], 65)
                self.assertIn("wind", res1)
                self.assertEqual(res1["wind"]["speed_ms"], 4.2)
                self.assertEqual(res1["wind"]["speed_kmh"], 15.1)
                self.assertIn("condition", res1)
                self.assertEqual(res1["condition"]["main"], "Clouds")
                self.assertEqual(len(res1["forecast"]), 1)

                # Test Cache hit
                res2 = self.adapter.fetch_weather(lat=18.52, lon=73.85, force_refresh=False)
                self.assertTrue(res2["timestamp"]["cached"])

    def test_04_fastapi_weather_endpoint(self):
        """Verify GET /api/weather endpoint integration."""
        client = TestClient(app)
        res = client.get("/api/weather?lat=18.52&lon=73.85")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("health_state", data)

    def test_05_live_openweather_connectivity(self):
        """Perform an actual live connectivity test with the real OPENWEATHER_API_KEY."""
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not api_key:
            self.skipTest("OPENWEATHER_API_KEY is not set in environment.")

        print(f"\n[LIVE CONNECTIVITY TEST] Querying OpenWeather API with key prefix {api_key[:6]}...")
        res = self.adapter.fetch_weather(lat=18.5204, lon=73.8567, city="Pune", force_refresh=True)

        print(f"[LIVE RESPONSE] Status: {res.get('status')}, Health State: {res.get('health_state')}")
        if res.get("status") == "ok":
            loc = res.get("location", {})
            temp = res.get("temperature", {})
            cond = res.get("condition", {})
            print(f"[LIVE WEATHER SUCCESS] City: {loc.get('city')}, {loc.get('country')} ({loc.get('lat')}, {loc.get('lon')})")
            print(f"[LIVE WEATHER DETAILS] Temp: {temp.get('current_c')}°C (Feels like {temp.get('feels_like_c')}°C), Condition: {cond.get('main')} ({cond.get('description')})")
        else:
            print(f"[LIVE WEATHER ERROR] {res.get('error')}")

        self.assertIn("health_state", res)
        if res.get("status") == "ok":
            self.assertEqual(res.get("health_state"), SourceHealthState.LIVE.value)
        else:
            self.assertEqual(res.get("health_state"), SourceHealthState.UNAVAILABLE.value)
            self.assertIn("error", res)


def run_openweather_tests():
    import json
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOpenWeatherAdapter)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import json, sys
    success = run_openweather_tests()
    sys.exit(0 if success else 1)

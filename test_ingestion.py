"""
Unit and Integration Test Suite for Real-time Multi-Source Data Ingestion Engine.

Tests:
1. Adapter NOT_CONFIGURED health state when env vars are absent.
2. Adapter UNAVAILABLE health state when network fetch fails or times out.
3. Adapter LIVE health state when valid response is fetched and validated.
4. CentralSourceManager concurrent fetch and timeout handling across 5 adapters.
5. Assembly into 6-channel tensor (6, 6, 64, 64) preserving six-channel model contract.
6. FastAPI endpoints /api/sources/health and /api/sources/ingest.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi.testclient import TestClient

from ingestion import (
    CENTRAL_SOURCE_MANAGER,
    CentralSourceManager,
    INSATAdapter,
    LightningAdapter,
    RadarAdapter,
    SourceHealthState,
)
from server import app
import config as C


class TestMultiSourceIngestion(unittest.TestCase):

    def setUp(self):
        # Clear mock env vars
        for key in ["RADAR_API_URL", "INSAT_API_URL", "LIGHTNING_API_URL", "AWS_API_URL", "NWP_API_URL"]:
            os.environ.pop(key, None)

    def test_01_health_state_not_configured(self):
        """Verify adapters report NOT_CONFIGURED when environment endpoint URLs are missing."""
        manager = CentralSourceManager()
        health = manager.get_all_health()

        self.assertIn("radar", health)
        self.assertIn("insat", health)
        self.assertIn("lightning", health)
        self.assertIn("aws", health)
        self.assertIn("nwp", health)

        for source_id, status in health.items():
            self.assertEqual(status["state"], SourceHealthState.NOT_CONFIGURED.value)
            self.assertNotEqual(status["state"], SourceHealthState.LIVE.value)

    def test_02_adapter_unavailable_on_fetch_failure(self):
        """Verify adapter transitions to UNAVAILABLE when network request fails or times out."""
        os.environ["RADAR_API_URL"] = "http://127.0.0.1:59999/invalid_radar_stream"
        adapter = RadarAdapter()

        res = adapter.fetch(timeout=0.5)
        self.assertFalse(res.success)
        self.assertEqual(res.state, SourceHealthState.UNAVAILABLE)
        self.assertTrue(bool(res.error))

        health = adapter.get_health().to_dict()
        self.assertEqual(health["state"], SourceHealthState.UNAVAILABLE.value)

    def test_03_adapter_live_on_valid_payload(self):
        """Verify adapter transitions to LIVE only when endpoint succeeds and payload is validated."""
        os.environ["INSAT_API_URL"] = "http://mock-satellite.local/api/ir_tb"
        adapter = INSATAdapter()

        mock_payload = b'{"status": "ok", "ir_tb": [[280.0]]}'
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = mock_payload
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            res = adapter.fetch(timeout=1.0)
            self.assertTrue(res.success)
            self.assertEqual(res.state, SourceHealthState.LIVE)
            self.assertEqual(adapter.get_health().state, SourceHealthState.LIVE)

    def test_04_central_manager_concurrent_fetch(self):
        """Verify CentralSourceManager executes all adapter fetches concurrently with timeout limit."""
        manager = CentralSourceManager()
        results = manager.fetch_all(timeout=0.5)

        self.assertEqual(len(results), len(manager.adapters))
        self.assertIn("radar", results)
        self.assertIn("insat", results)
        self.assertIn("lightning", results)
        self.assertIn("aws", results)
        self.assertIn("nwp", results)

    def test_05_six_channel_contract_assembly(self):
        """Verify assembly into 6-channel array shape (6, 6, 64, 64) preserving six-channel model contract."""
        manager = CentralSourceManager()
        results = manager.fetch_all(timeout=0.2)

        demo_fallback = np.ones((6, 6, 64, 64), dtype=np.float32) * 5.0
        tensor, sources_mapped = manager.assemble_six_channel_tensor(results, demo_fallback=demo_fallback)

        self.assertEqual(tensor.shape, (6, 6, 64, 64))
        self.assertEqual(len(sources_mapped), 6)
        expected_channels = ["refl_sfc", "refl_m10", "echo_top", "vil", "ir_tb", "light_dens"]
        for ch in expected_channels:
            self.assertIn(ch, sources_mapped)

    def test_06_fastapi_sources_endpoints(self):
        """Verify FastAPI endpoints /api/sources/health and /api/sources/ingest work cleanly."""
        client = TestClient(app)

        # GET /api/sources/health
        r_health = client.get("/api/sources/health")
        self.assertEqual(r_health.status_code, 200)
        h_data = r_health.json()
        self.assertEqual(h_data.get("status"), "ok")
        self.assertIn("sources", h_data)
        self.assertEqual(len(h_data["sources"]), len(CENTRAL_SOURCE_MANAGER.adapters))

        # POST /api/sources/ingest
        r_ingest = client.post("/api/sources/ingest?timeout=1.0")
        self.assertEqual(r_ingest.status_code, 200)
        i_data = r_ingest.json()
        self.assertEqual(i_data.get("status"), "ok")
        self.assertIn("ingest_summary", i_data)
        self.assertEqual(len(i_data["ingest_summary"]), len(CENTRAL_SOURCE_MANAGER.adapters))

    def test_07_nwp_live_fetch_success_and_mock(self):
        """Verify NWPAdapter successfully fetches, parses, and validates Open-Meteo NWP response."""
        from ingestion import NWPAdapter
        adapter = NWPAdapter()

        mock_payload = b'''{
            "latitude": 26.9124,
            "longitude": 74.6399,
            "hourly_units": {
                "time": "iso8601",
                "cape": "J/kg",
                "convective_inhibition": "J/kg",
                "freezing_level_height": "m"
            },
            "hourly": {
                "time": ["2026-09-29T23:00"],
                "cape": [650.0],
                "convective_inhibition": [120.0],
                "freezing_level_height": [4500.0]
            }
        }'''
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = mock_payload
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            res = adapter.fetch_nwp(lat=26.9124, lon=74.6399, timeout=1.0)
            self.assertEqual(res["status"], "LIVE")
            self.assertEqual(res["provider"], "Open-Meteo NWP Forecast API (NOAA GFS / ECMWF IFS)")
            self.assertEqual(res["location"]["lat"], 26.9124)
            self.assertEqual(res["location"]["lon"], 74.6399)
            self.assertEqual(res["values"]["cape_jkg"], 650.0)
            self.assertEqual(res["values"]["cin_jkg"], 120.0)
            self.assertEqual(res["values"]["freezing_level_m"], 4500.0)
            self.assertEqual(res["values"]["freezing_level_km"], 4.5)
            self.assertEqual(res["values"]["charging_lower_km"], 6.0)
            self.assertEqual(res["units"]["cape"], "J/kg")
            self.assertIn("observation_iso", res["timestamp"])

        # Test GET /api/nwp/current endpoint via TestClient
        client = TestClient(app)
        with patch("urllib.request.urlopen", return_value=mock_response):
            r_api = client.get("/api/nwp/current?lat=26.9124&lon=74.6399")
            self.assertEqual(r_api.status_code, 200)
            data = r_api.json()
            self.assertEqual(data["status"], "LIVE")
            self.assertEqual(data["values"]["cape_jkg"], 650.0)
            self.assertEqual(data["values"]["freezing_level_km"], 4.5)

    def test_08_nwp_live_fetch_error_handling(self):
        """Verify NWPAdapter returns explicit UNAVAILABLE status without hardcoded fallbacks on error."""
        from ingestion import NWPAdapter
        adapter = NWPAdapter()

        with patch("urllib.request.urlopen", side_effect=Exception("Connection Refused")):
            res = adapter.fetch_nwp(lat=26.9124, lon=74.6399, timeout=0.5)
            self.assertEqual(res["status"], "UNAVAILABLE")
            self.assertIsNone(res["values"])
            self.assertIn("error", res)
            self.assertIn("Connection Refused", res["error"])

        client = TestClient(app)
        with patch("urllib.request.urlopen", side_effect=Exception("API Timeout")):
            r_api = client.get("/api/nwp/current?lat=26.9124&lon=74.6399")
            self.assertEqual(r_api.status_code, 200)
            data = r_api.json()
            self.assertEqual(data["status"], "UNAVAILABLE")
            self.assertIsNone(data["values"])
            self.assertIn("API Timeout", data["error"])


def run_ingestion_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiSourceIngestion)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_ingestion_tests()
    sys.exit(0 if success else 1)

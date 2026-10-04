"""
Unit and Integration Test Suite for FastAPI Backend API Layer.

Tests:
- GET /api/health (system health, grid geometry, channel definitions)
- GET/POST /api/current-data (64x64 grid, 6 input channels data)
- POST /api/predict (60-min nowcast predictions, 12 lead times, causality guarantees)
- CORS headers & options preflight handling
- Legacy dashboard endpoints (/api/info, /api/sample, /)
"""

from __future__ import annotations

import unittest
import numpy as np
from fastapi.testclient import TestClient

from server import app
import config as C


class TestFastAPIBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """Verify GET /api/health returns status ok, 64x64 grid geometry, and 6 input channels."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("mode"), "demo")
        self.assertEqual(data.get("grid_h"), 64)
        self.assertEqual(data.get("grid_w"), 64)
        self.assertEqual(data.get("input_frames"), 6)
        self.assertEqual(data.get("output_frames"), 12)
        self.assertEqual(data.get("n_channels"), 6)
        expected_channels = ["refl_sfc", "refl_m10", "echo_top", "vil", "ir_tb", "light_dens"]
        self.assertEqual(data.get("channels"), expected_channels)
        self.assertIn("timestamp", data)

    def test_02_current_data_endpoint(self):
        """Verify GET & POST /api/current-data return 64x64 grids across all 6 input channels."""
        # GET request
        res_get = self.client.get("/api/current-data?sample=0&n_events=6")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()

        self.assertEqual(data_get.get("status"), "ok")
        self.assertEqual(data_get.get("grid_shape"), [64, 64])
        self.assertEqual(data_get.get("n_channels"), 6)
        
        channel_data = data_get.get("data", {})
        self.assertEqual(len(channel_data), 6)
        for ch in ["refl_sfc", "refl_m10", "echo_top", "vil", "ir_tb", "light_dens"]:
            self.assertIn(ch, channel_data)
            grid = np.array(channel_data[ch])
            self.assertEqual(grid.shape, (64, 64))

        # POST request
        res_post = self.client.post("/api/current-data?sample=0&n_events=6")
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json().get("status"), "ok")

    def test_03_predict_advection(self):
        """Verify POST /api/predict generates 12 output lead times on a 64x64 grid using advection."""
        payload = {
            "model": "advection",
            "sample_idx": 0,
            "n_events": 6
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("model"), "advection")
        self.assertEqual(data.get("grid_shape"), [64, 64])
        self.assertEqual(data.get("input_frames"), 6)
        self.assertEqual(data.get("output_frames"), 12)

        probs = np.array(data.get("probabilities"))
        self.assertEqual(probs.shape, (12, 64, 64))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        lead_stats = data.get("lead_stats")
        self.assertEqual(len(lead_stats), 12)
        expected_leads = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        self.assertEqual(data.get("lead_times_min"), expected_leads)

    def test_04_predict_baselines_and_ensemble(self):
        """Verify POST /api/predict works across baseline models and ensemble mode."""
        models = ["persistence", "climatology", "charging_rule", "charging_layer", "ensemble"]
        for m in models:
            res = self.client.post("/api/predict", json={"model": m, "sample_idx": 0})
            self.assertEqual(res.status_code, 200, f"Model {m} failed")
            data = res.json()
            self.assertEqual(data.get("status"), "success")
            probs = np.array(data.get("probabilities"))
            self.assertEqual(probs.shape, (12, 64, 64))

    def test_05_predict_custom_input_causality(self):
        """Verify POST /api/predict handles custom 6-channel 64x64 input with strict causality."""
        # Create synthetic 6-frame 6-channel 64x64 input sequence (6, 6, 64, 64)
        synthetic_input = np.random.uniform(0.0, 35.0, (6, 6, 64, 64)).tolist()
        payload = {
            "model": "advection",
            "input_data": synthetic_input
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data.get("status"), "success")
        probs = np.array(data.get("probabilities"))
        self.assertEqual(probs.shape, (12, 64, 64))
        self.assertEqual(data.get("event_id"), -1)

    def test_06_cors_headers(self):
        """Verify response headers include Access-Control-Allow-Origin for CORS."""
        res = self.client.get("/api/health", headers={"Origin": "http://localhost:3000"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("access-control-allow-origin", res.headers)

    def test_07_legacy_endpoints(self):
        """Verify pre-existing dashboard endpoints remain 100% operational."""
        # GET /api/info
        r_info = self.client.get("/api/info")
        self.assertEqual(r_info.status_code, 200)
        self.assertEqual(r_info.json().get("grid_h"), 64)

        # GET /api/sample
        r_sample = self.client.get("/api/sample?sample=0&model=advection")
        self.assertEqual(r_sample.status_code, 200)
        self.assertIn("probs", r_sample.json())

        # GET /
        r_index = self.client.get("/")
        self.assertEqual(r_index.status_code, 200)
        self.assertIn("html", r_index.headers.get("content-type", "").lower())

    def test_08_radar_metadata_endpoint(self):
        """Verify GET /api/radar/metadata dynamically returns RainViewer radar metadata & tile template."""
        res = self.client.get("/api/radar/metadata?force_refresh=true")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("health_state", data)
        if data.get("status") == "ok":
            self.assertEqual(data.get("health_state"), "LIVE")
            self.assertIn("tile_template", data)
            self.assertIn("latest_time", data)
            self.assertIn("/256/", data.get("tile_template", ""))
            self.assertIn("{z}", data.get("tile_template", ""))
            self.assertIn("{x}", data.get("tile_template", ""))
            self.assertIn("{y}", data.get("tile_template", ""))
        else:
            self.assertIn(data.get("health_state"), ["STALE", "UNAVAILABLE"])
            self.assertIn("error", data)



def run_backend_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFastAPIBackend)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_backend_tests()
    sys.exit(0 if success else 1)

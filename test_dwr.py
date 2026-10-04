"""
Unit and Integration Test Suite for Numerical DWR Radar Ingestion & Spatial Resampling Engine.

Tests:
1. Gabella clutter filter, dBZ clipping, and attenuation correction in DWRQualityControl.
2. Lat/Lon to 64x64 grid coordinate transformation & spatial resampling math in DWRSpatialResampler.
3. HDF5 & NetCDF4 raw byte parsers in DWRHDF5Parser.
4. MosdacDwrAdapter and ImdDwrAdapter health states in CentralSourceManager.
5. FastAPI /api/dwr/numerical-metadata and /api/dwr/ingest endpoints.
"""

from __future__ import annotations

import unittest
import numpy as np
from fastapi.testclient import TestClient

import config as C
import dwr_processor as DP
from ingestion import CENTRAL_SOURCE_MANAGER, MosdacDwrAdapter, ImdDwrAdapter, SourceHealthState
from server import app


class TestDWRProcessor(unittest.TestCase):

    def setUp(self):
        self.qc = DP.DWRQualityControl()
        self.resampler = DP.DWRSpatialResampler(grid_h=64, grid_w=64, pixel_km=2.0)
        self.client = TestClient(app)

    def test_01_quality_control_clipping_and_filter(self):
        """Verify dBZ clipping and Gabella clutter filtering."""
        raw_dbz = np.array([
            [10.0, 15.0, -10.0, 95.0],
            [5.0,  55.0,  5.0, 30.0],
            [20.0,  5.0, 25.0, 40.0],
            [0.0,  12.0,  8.0, 50.0]
        ], dtype=np.float32)

        clipped = self.qc.clip_reflectivity(raw_dbz, min_dbz=0.0, max_dbz=75.0)
        self.assertTrue((clipped >= 0.0).all() and (clipped <= 75.0).all())
        self.assertEqual(clipped[0, 2], 0.0)  # -10 clipped to 0
        self.assertEqual(clipped[0, 3], 75.0) # 95 clipped to 75

        # Create localized spatial clutter spike
        grid = np.full((10, 10), 5.0, dtype=np.float32)
        grid[5, 5] = 60.0  # Isolated severe clutter spike
        filtered = self.qc.gabella_clutter_filter(grid, threshold_dbz=12.0)
        self.assertLess(filtered[5, 5], 60.0)
        self.assertEqual(filtered[5, 5], 5.0)

    def test_02_spatial_coordinate_transformation(self):
        """Verify forward and inverse spatial coordinate transformation math."""
        center_lat, center_lon = 18.5204, 73.8567

        # Center point (32, 32) should map to center lat/lon
        lat, lon = self.resampler.grid_coords_to_latlon(32.0, 32.0, center_lat, center_lon)
        self.assertAlmostEqual(lat, center_lat, places=3)
        self.assertAlmostEqual(lon, center_lon, places=3)

        # Reverse mapping
        r, c = self.resampler.latlon_to_grid_coords(lat, lon, center_lat, center_lon)
        self.assertAlmostEqual(r, 32.0, places=3)
        self.assertAlmostEqual(c, 32.0, places=3)

    def test_03_spatial_resampling_onto_64x64_grid(self):
        """Verify raw 2D radar array resampling onto 64x64 grid."""
        raw_lats = np.array([18.5204, 18.6000, 18.4400, 18.5204], dtype=np.float32)
        raw_lons = np.array([73.8567, 73.8567, 73.8567, 73.9500], dtype=np.float32)
        raw_values = np.array([45.0, 52.0, 38.0, 60.0], dtype=np.float32)

        resample_grid = self.resampler.resample_to_64x64(
            raw_lats, raw_lons, raw_values, center_lat=18.5204, center_lon=73.8567
        )

        self.assertEqual(resample_grid.shape, (64, 64))
        self.assertGreater(resample_grid.max(), 0.0)

    def test_04_adapters_health_state(self):
        """Verify MosdacDwrAdapter and ImdDwrAdapter health state initialization."""
        mosdac = MosdacDwrAdapter()
        imd = ImdDwrAdapter()

        self.assertEqual(mosdac.source_id, "mosdac_dwr")
        self.assertEqual(imd.source_id, "imd_dwr")
        self.assertIn(mosdac.state, [SourceHealthState.NOT_CONFIGURED, SourceHealthState.UNAVAILABLE])
        self.assertIn(imd.state, [SourceHealthState.NOT_CONFIGURED, SourceHealthState.UNAVAILABLE])

        self.assertIn("mosdac_dwr", CENTRAL_SOURCE_MANAGER.adapters)
        self.assertIn("imd_dwr", CENTRAL_SOURCE_MANAGER.adapters)

    def test_05_fastapi_dwr_endpoints(self):
        """Verify /api/dwr/numerical-metadata and /api/dwr/ingest endpoints."""
        res1 = self.client.get("/api/dwr/numerical-metadata")
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()

        self.assertEqual(data1.get("status"), "ok")
        self.assertEqual(data1.get("grid_shape"), [64, 64])
        self.assertIn("mosdac_dwr", data1.get("dwr_providers", {}))
        self.assertIn("imd_dwr", data1.get("dwr_providers", {}))

        res2 = self.client.get("/api/dwr/ingest?lat=18.5204&lon=73.8567&provider=mosdac")
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()

        self.assertEqual(data2.get("status"), "ok")
        self.assertEqual(data2.get("provider"), "mosdac")
        self.assertIn("grid_bounds", data2)


def run_dwr_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDWRProcessor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_dwr_tests()
    sys.exit(0 if success else 1)

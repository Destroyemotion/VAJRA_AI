"""
Numerical Doppler Weather Radar (DWR) Processing & Spatial Alignment Module for VAJRA-AI.

Provides:
1. ODIM HDF5 & NetCDF4 raw numerical radar parsing (dBZ, VIL, Echo Tops, Radial Velocity).
2. Quality Control (Gabella spatial clutter removal, attenuation adjustment, physical range bounds).
3. Coordinate transformation & spatial resampling onto VAJRA-AI's 64x64 grid (2.0 km resolution).
4. Preservation of 6-channel ML contract (refl_sfc, refl_m10, echo_top, vil, ir_tb, light_dens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import netCDF4 as nc
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import config as C


@dataclass
class DWRNumericalMetadata:
    source_id: str
    station_name: str
    latitude: float
    longitude: float
    timestamp: float
    format_type: str  # "HDF5", "NETCDF4", "GRIDDED_BINARY"
    channels_available: List[str]
    elevation_angles_deg: List[float] = field(default_factory=list)
    raw_shape: Optional[Tuple[int, ...]] = None
    health_state: str = "LIVE"


class DWRQualityControl:
    """Quality control algorithms for numerical radar data."""

    @staticmethod
    def clip_reflectivity(dbz_array: np.ndarray, min_dbz: float = 0.0, max_dbz: float = 75.0) -> np.ndarray:
        """Clips reflectivity array to valid physical dBZ bounds [0, 75]."""
        arr = np.array(dbz_array, dtype=np.float32)
        arr = np.where(np.isnan(arr) | (arr < min_dbz), 0.0, arr)
        return np.clip(arr, min_dbz, max_dbz)

    @staticmethod
    def gabella_clutter_filter(dbz_array: np.ndarray, threshold_dbz: float = 12.0) -> np.ndarray:
        """
        Identifies isolated transient spikes / ground clutter using Gabella spatial filter heuristic.
        Replaces pixels where the central value exceeds surrounding 8-neighbor median by threshold_dbz.
        """
        grid = np.array(dbz_array, dtype=np.float32)
        if grid.ndim != 2:
            return grid

        h, w = grid.shape
        filtered = grid.copy()

        for r in range(1, h - 1):
            for c in range(1, w - 1):
                val = grid[r, c]
                if val <= 5.0:
                    continue
                neighbors = [
                    grid[r-1, c-1], grid[r-1, c], grid[r-1, c+1],
                    grid[r, c-1],               grid[r, c+1],
                    grid[r+1, c-1], grid[r+1, c], grid[r+1, c+1]
                ]
                med = float(np.median(neighbors))
                if val - med > threshold_dbz and med < 15.0:
                    filtered[r, c] = med

        return filtered

    @staticmethod
    def apply_attenuation_correction(dbz_array: np.ndarray, alpha: float = 0.001) -> np.ndarray:
        """Applies path-integrated attenuation adjustment for high reflectivity cores."""
        arr = np.array(dbz_array, dtype=np.float32)
        cum_dbz = np.cumsum(np.maximum(0.0, arr - 30.0), axis=-1)
        corrected = arr + alpha * cum_dbz
        return np.clip(corrected, 0.0, 75.0)


class DWRSpatialResampler:
    """Resamples continuous latitude/longitude radar observations onto VAJRA-AI's 64x64 spatial grid."""

    def __init__(
        self,
        grid_h: int = C.GRID_H,
        grid_w: int = C.GRID_W,
        pixel_km: float = C.PIXEL_KM
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.pixel_km = pixel_km

    def latlon_to_grid_coords(
        self,
        lat: float,
        lon: float,
        center_lat: float,
        center_lon: float
    ) -> Tuple[float, float]:
        """Maps geographic (lat, lon) coordinates to continuous grid fractional indices (r, c)."""
        km_per_lat = 111.0
        km_per_lon = 111.0 * math.cos(math.radians(center_lat))

        dy_km = (lat - center_lat) * km_per_lat
        dx_km = (lon - center_lon) * km_per_lon

        r = (self.grid_h / 2.0) - (dy_km / self.pixel_km)
        c = (self.grid_w / 2.0) + (dx_km / self.pixel_km)

        return r, c

    def grid_coords_to_latlon(
        self,
        r: float,
        c: float,
        center_lat: float,
        center_lon: float
    ) -> Tuple[float, float]:
        """Maps fractional grid indices (r, c) to geographic (lat, lon)."""
        km_per_lat = 111.0
        km_per_lon = 111.0 * math.cos(math.radians(center_lat))

        dy_km = ( (self.grid_h / 2.0) - r ) * self.pixel_km
        dx_km = ( c - (self.grid_w / 2.0) ) * self.pixel_km

        lat = center_lat + (dy_km / km_per_lat)
        lon = center_lon + (dx_km / km_per_lon)

        return lat, lon

    def resample_to_64x64(
        self,
        raw_lats: np.ndarray,
        raw_lons: np.ndarray,
        raw_values: np.ndarray,
        center_lat: float = 18.5204,
        center_lon: float = 73.8567,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Resamples raw 2D radar array (lats, lons, values) onto the 64x64 target spatial grid.
        Uses SciPy griddata interpolation when available, with fast spatial binning fallback.
        """
        raw_lats = np.array(raw_lats, dtype=np.float32).ravel()
        raw_lons = np.array(raw_lons, dtype=np.float32).ravel()
        raw_values = np.array(raw_values, dtype=np.float32).ravel()

        grid_mesh = np.full((self.grid_h, self.grid_w), fill_value, dtype=np.float32)

        if len(raw_values) == 0:
            return grid_mesh

        # Generate target lat/lon grid points for each 64x64 cell
        target_r, target_c = np.indices((self.grid_h, self.grid_w))
        km_per_lat = 111.0
        km_per_lon = 111.0 * math.cos(math.radians(center_lat))

        target_dy = ( (self.grid_h / 2.0) - target_r ) * self.pixel_km
        target_dx = ( target_c - (self.grid_w / 2.0) ) * self.pixel_km

        target_lats = center_lat + (target_dy / km_per_lat)
        target_lons = center_lon + (target_dx / km_per_lon)

        if HAS_SCIPY and len(raw_values) >= 4:
            try:
                points = np.column_stack((raw_lats, raw_lons))
                target_points = np.column_stack((target_lats.ravel(), target_lons.ravel()))
                interpolated = griddata(points, raw_values, target_points, method='linear', fill_value=fill_value)
                grid_mesh = interpolated.reshape((self.grid_h, self.grid_w)).astype(np.float32)
                grid_mesh = np.nan_to_num(grid_mesh, nan=fill_value)
                return grid_mesh
            except Exception:
                pass

        # Fast nearest-neighbor spatial binning fallback
        r_idx = np.round((self.grid_h / 2.0) - ((raw_lats - center_lat) * km_per_lat / self.pixel_km)).astype(int)
        c_idx = np.round((self.grid_w / 2.0) + ((raw_lons - center_lon) * km_per_lon / self.pixel_km)).astype(int)

        valid_mask = (r_idx >= 0) & (r_idx < self.grid_h) & (c_idx >= 0) & (c_idx < self.grid_w)
        for r, c, val in zip(r_idx[valid_mask], c_idx[valid_mask], raw_values[valid_mask]):
            if val > grid_mesh[r, c]:
                grid_mesh[r, c] = val

        return grid_mesh


class DWRHDF5Parser:
    """Parses WMO ODIM HDF5 & MOSDAC numerical Doppler Weather Radar files."""

    @staticmethod
    def parse_odim_hdf5_bytes(file_bytes: bytes) -> Dict[str, Any]:
        """
        Parses in-memory HDF5 raw bytes and extracts metadata, gain/offset scaling,
        and reflectivity dataset arrays.
        """
        if not HAS_H5PY:
            return {"status": "error", "error": "h5py library not installed"}

        try:
            bio = io.BytesIO(file_bytes)
            with h5py.File(bio, 'r') as h5f:
                # Extract root attributes
                where = h5f.get('where', {})
                lat = float(where.attrs.get('lat', 18.5204)) if hasattr(where, 'attrs') else 18.5204
                lon = float(where.attrs.get('lon', 73.8567)) if hasattr(where, 'attrs') else 73.8567

                what = h5f.get('what', {})
                source = str(what.attrs.get('source', 'MOSDAC_IMD_DWR')) if hasattr(what, 'attrs') else 'MOSDAC_IMD_DWR'

                extracted_channels = {}

                # Search for dataset groups (dataset1, dataset2...)
                for ds_key in h5f.keys():
                    if ds_key.startswith('dataset'):
                        group = h5f[ds_key]
                        for data_key in group.keys():
                            if data_key.startswith('data'):
                                data_group = group[data_key]
                                if 'data' in data_group:
                                    raw_arr = np.array(data_group['data'])
                                    dwhat = data_group.get('what', {})
                                    gain = float(dwhat.attrs.get('gain', 1.0)) if hasattr(dwhat, 'attrs') else 1.0
                                    offset = float(dwhat.attrs.get('offset', 0.0)) if hasattr(dwhat, 'attrs') else 0.0
                                    nodata = float(dwhat.attrs.get('nodata', 255.0)) if hasattr(dwhat, 'attrs') else 255.0
                                    quantity = str(dwhat.attrs.get('quantity', 'DBZH')) if hasattr(dwhat, 'attrs') else 'DBZH'

                                    scaled_arr = np.where(raw_arr == nodata, 0.0, raw_arr * gain + offset)
                                    extracted_channels[quantity] = scaled_arr.astype(np.float32)

                return {
                    "status": "ok",
                    "latitude": lat,
                    "longitude": lon,
                    "source": source,
                    "quantities": list(extracted_channels.keys()),
                    "channel_data": extracted_channels
                }
        except Exception as e:
            return {"status": "error", "error": f"HDF5 parse failure: {str(e)}"}

    @staticmethod
    def parse_netcdf4_bytes(file_bytes: bytes) -> Dict[str, Any]:
        """Parses CF-compliant NetCDF4 radar/satellite bytes."""
        if not HAS_NETCDF4:
            return {"status": "error", "error": "netCDF4 library not installed"}

        try:
            with nc.Dataset("in_memory_dwr.nc", mode="r", memory=file_bytes) as ds:
                lat = float(ds.variables['lat'][0]) if 'lat' in ds.variables else 18.5204
                lon = float(ds.variables['lon'][0]) if 'lon' in ds.variables else 73.8567
                
                extracted = {}
                for var_name in ['DBZH', 'refl_sfc', 'refl_m10', 'echo_top', 'vil']:
                    if var_name in ds.variables:
                        extracted[var_name] = np.array(ds.variables[var_name][:], dtype=np.float32)

                return {
                    "status": "ok",
                    "latitude": lat,
                    "longitude": lon,
                    "variables": list(extracted.keys()),
                    "channel_data": extracted
                }
        except Exception as e:
            return {"status": "error", "error": f"NetCDF4 parse failure: {str(e)}"}

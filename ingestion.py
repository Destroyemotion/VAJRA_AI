"""
Real-time Multi-Source Data Ingestion Engine for VAJRA-AI.

Adapters:
1. RadarAdapter (refl_sfc, refl_m10, echo_top, vil)
2. INSATAdapter (ir_tb - INSAT-3D/3DR TIR1/TIR2)
3. LightningAdapter (light_dens - IITM LLN / NRSC LDS / WWLLN)
4. AWSAdapter (Automatic Weather Stations - temp, humidity, pressure, wind)
5. NWPAdapter (Numerical Weather Prediction - CAPE, CIN, freezing level height)

Health States:
- LIVE: Endpoint active, real data fetched and validated within freshness threshold.
- STALE: Data previously fetched, but timestamp exceeds freshness threshold (> 15-30 min).
- DEGRADED: Partial data, missing sub-channels, or elevated latency.
- UNAVAILABLE: Connection timeout, network failure, or HTTP error.
- NOT_CONFIGURED: API key, endpoint URL, or S3 URI not configured in environment.

Six-Channel Contract Preserved:
1. refl_sfc (dBZ)
2. refl_m10 (dBZ)
3. echo_top (km)
4. vil (kg/m²)
5. ir_tb (K)
6. light_dens (strikes/pixel/5min)
"""

from __future__ import annotations

import concurrent.futures
from enum import Enum
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import config as C


class SourceHealthState(str, Enum):
    LIVE = "LIVE"
    STALE = "STALE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class SourceHealthStatus:
    def __init__(
        self,
        source_id: str,
        name: str,
        state: SourceHealthState,
        channels: List[str],
        endpoint_uri: Optional[str] = None,
        last_success_time: Optional[float] = None,
        last_attempt_time: Optional[float] = None,
        latency_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        freshness_sec: Optional[float] = None,
    ):
        self.source_id = source_id
        self.name = name
        self.state = state
        self.channels = channels
        self.endpoint_uri = endpoint_uri
        self.last_success_time = last_success_time
        self.last_attempt_time = last_attempt_time
        self.latency_ms = latency_ms
        self.error_message = error_message
        self.freshness_sec = freshness_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "state": self.state.value,
            "channels": self.channels,
            "endpoint_uri": self.endpoint_uri or "NOT_SET",
            "last_success_time": self.last_success_time,
            "last_attempt_time": self.last_attempt_time,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms is not None else None,
            "error_message": self.error_message,
            "freshness_sec": round(self.freshness_sec, 1) if self.freshness_sec is not None else None,
        }


class IngestionResult:
    def __init__(
        self,
        source_id: str,
        success: bool,
        state: SourceHealthState,
        timestamp: float,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ):
        self.source_id = source_id
        self.success = success
        self.state = state
        self.timestamp = timestamp
        self.data = data or {}
        self.error = error
        self.latency_ms = latency_ms


class BaseIngestionAdapter:
    """Abstract Base Class for Multi-Source Ingestion Adapters."""

    def __init__(self, source_id: str, name: str, channels: List[str], env_var_key: str):
        self.source_id = source_id
        self.name = name
        self.channels = channels
        self.env_var_key = env_var_key
        self.endpoint_uri = os.environ.get(env_var_key, "").strip()
        self.state = SourceHealthState.NOT_CONFIGURED if not self.endpoint_uri else SourceHealthState.UNAVAILABLE
        self.last_success_time: Optional[float] = None
        self.last_attempt_time: Optional[float] = None
        self.last_latency_ms: Optional[float] = None
        self.last_error: Optional[str] = None
        self.cached_data: Optional[Dict[str, Any]] = None

    def is_configured(self) -> bool:
        self.endpoint_uri = os.environ.get(self.env_var_key, "").strip()
        return bool(self.endpoint_uri)

    def fetch(self, timeout: float = 5.0) -> IngestionResult:
        t0 = time.time()
        self.last_attempt_time = t0

        if not self.is_configured():
            self.state = SourceHealthState.NOT_CONFIGURED
            self.last_error = f"Environment variable '{self.env_var_key}' is not set."
            return IngestionResult(
                source_id=self.source_id,
                success=False,
                state=SourceHealthState.NOT_CONFIGURED,
                timestamp=t0,
                error=self.last_error
            )

        try:
            req = urllib.request.Request(
                self.endpoint_uri,
                headers={"User-Agent": "VAJRA-AI-Ingestion-Engine/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                latency = (time.time() - t0) * 1000.0
                self.last_latency_ms = latency
                if response.status != 200:
                    self.state = SourceHealthState.UNAVAILABLE
                    self.last_error = f"HTTP status {response.status}"
                    return IngestionResult(
                        source_id=self.source_id,
                        success=False,
                        state=SourceHealthState.UNAVAILABLE,
                        timestamp=t0,
                        error=self.last_error,
                        latency_ms=latency
                    )

                raw_bytes = response.read()
                try:
                    payload = json.loads(raw_bytes.decode("utf-8"))
                except Exception:
                    # Raw binary array handling or JSON fallback
                    payload = {"raw_size_bytes": len(raw_bytes)}

                if self.validate(payload):
                    self.last_success_time = time.time()
                    self.cached_data = payload
                    self.last_error = None
                    self.state = SourceHealthState.LIVE
                    return IngestionResult(
                        source_id=self.source_id,
                        success=True,
                        state=SourceHealthState.LIVE,
                        timestamp=self.last_success_time,
                        data=payload,
                        latency_ms=latency
                    )
                else:
                    self.state = SourceHealthState.DEGRADED
                    self.last_error = "Payload validation failed (schema/range out of bounds)"
                    return IngestionResult(
                        source_id=self.source_id,
                        success=False,
                        state=SourceHealthState.DEGRADED,
                        timestamp=t0,
                        error=self.last_error,
                        latency_ms=latency
                    )

        except urllib.error.URLError as e:
            latency = (time.time() - t0) * 1000.0
            self.last_latency_ms = latency
            self.state = SourceHealthState.UNAVAILABLE
            self.last_error = f"Network URL error: {e.reason}"
            return IngestionResult(
                source_id=self.source_id,
                success=False,
                state=SourceHealthState.UNAVAILABLE,
                timestamp=t0,
                error=self.last_error,
                latency_ms=latency
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000.0
            self.last_latency_ms = latency
            self.state = SourceHealthState.UNAVAILABLE
            self.last_error = f"{type(e).__name__}: {str(e)}"
            return IngestionResult(
                source_id=self.source_id,
                success=False,
                state=SourceHealthState.UNAVAILABLE,
                timestamp=t0,
                error=self.last_error,
                latency_ms=latency
            )

    def validate(self, payload: Any) -> bool:
        return payload is not None

    def get_health(self) -> SourceHealthStatus:
        now = time.time()
        freshness = (now - self.last_success_time) if self.last_success_time else None

        # Check STALE threshold (e.g. data older than 1800s)
        current_state = self.state
        if current_state == SourceHealthState.LIVE and freshness and freshness > 1800.0:
            current_state = SourceHealthState.STALE

        return SourceHealthStatus(
            source_id=self.source_id,
            name=self.name,
            state=current_state,
            channels=self.channels,
            endpoint_uri=self.endpoint_uri,
            last_success_time=self.last_success_time,
            last_attempt_time=self.last_attempt_time,
            latency_ms=self.last_latency_ms,
            error_message=self.last_error,
            freshness_sec=freshness
        )


# ---------------------------------------------------------------------------
# Concrete Ingestion Adapters
# ---------------------------------------------------------------------------

class RadarAdapter(BaseIngestionAdapter):
    """Adapter for Doppler Weather Radar (IMD DWR / NOAA NEXRAD Level II)."""

    def __init__(self):
        super().__init__(
            source_id="radar",
            name="Doppler Weather Radar (DWR/NEXRAD)",
            channels=["refl_sfc", "refl_m10", "echo_top", "vil"],
            env_var_key="RADAR_API_URL"
        )

    def validate(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        # Must contain valid radar channel data or grid array metadata
        if "data" in payload or "refl_sfc" in payload or "raw_size_bytes" in payload:
            return True
        return False


class INSATAdapter(BaseIngestionAdapter):
    """Adapter for INSAT-3D/3DR Satellite Thermal IR (TIR1/TIR2 10.8µm)."""

    def __init__(self):
        super().__init__(
            source_id="insat",
            name="INSAT-3D/3DR Satellite Multispectral",
            channels=["ir_tb"],
            env_var_key="INSAT_API_URL"
        )

    def validate(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "ir_tb" in payload or "tir1" in payload or "raw_size_bytes" in payload:
            return True
        return False


class LightningAdapter(BaseIngestionAdapter):
    """Adapter for Real-time Lightning Location Networks (IITM LLN / NRSC LDS)."""

    def __init__(self):
        super().__init__(
            source_id="lightning",
            name="Lightning Location Network (IITM LLN / NRSC LDS)",
            channels=["light_dens"],
            env_var_key="LIGHTNING_API_URL"
        )

    def validate(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "strikes" in payload or "light_dens" in payload or "raw_size_bytes" in payload:
            return True
        return False


class AWSAdapter(BaseIngestionAdapter):
    """Adapter for Automatic Weather Stations (IMD AWS / NOAA MADIS)."""

    def __init__(self):
        super().__init__(
            source_id="aws",
            name="Automatic Weather Stations (IMD/NOAA AWS)",
            channels=["surface_temp", "relative_humidity", "surface_pressure", "wind_u", "wind_v"],
            env_var_key="AWS_API_URL"
        )

    def validate(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "stations" in payload or "observations" in payload or "raw_size_bytes" in payload:
            return True
        return False


class NWPAdapter(BaseIngestionAdapter):
    """Adapter for Numerical Weather Prediction background fields (NOAA GFS / ECMWF IFS via Open-Meteo or IMD GFS/WRF)."""

    DEFAULT_OPEN_METEO_NWP_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        super().__init__(
            source_id="nwp",
            name="Numerical Weather Prediction (Open-Meteo / NOAA GFS / ECMWF IFS)",
            channels=["cape", "cin", "freezing_level_km", "charging_lower_km"],
            env_var_key="NWP_API_URL"
        )

    def is_configured(self) -> bool:
        return True  # Defaults to public Open-Meteo Forecast API if NWP_API_URL is unset

    def get_endpoint(self) -> str:
        url = os.environ.get("NWP_API_URL", "").strip()
        return url if url else NWPAdapter.DEFAULT_OPEN_METEO_NWP_URL

    def fetch_nwp(self, lat: float = 26.9124, lon: float = 74.6399, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Fetches live NWP parameters (CAPE, CIN, Freezing Level) from Open-Meteo Forecast API.
        Default location: Ajmer, Rajasthan (26.9124, 74.6399).
        """
        t0 = time.time()
        self.last_attempt_time = t0
        endpoint = self.get_endpoint()

        url = (
            f"{endpoint}?latitude={lat}&longitude={lon}"
            f"&hourly=cape,convective_inhibition,freezing_level_height"
            f"&timezone=Asia%2FKolkata"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VAJRA-AI-NWPAdapter/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                latency = (time.time() - t0) * 1000.0
                self.last_latency_ms = latency

                if response.status != 200:
                    self.state = SourceHealthState.UNAVAILABLE
                    self.last_error = f"HTTP error {response.status}"
                    return {
                        "status": "UNAVAILABLE",
                        "provider": "Open-Meteo NWP Forecast API",
                        "error": self.last_error,
                        "values": None,
                        "timestamp": {"fetch_time_epoch": t0}
                    }

                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode("utf-8"))

                hourly = data.get("hourly", {})
                hourly_units = data.get("hourly_units", {})
                times = hourly.get("time", [])

                if not times:
                    self.state = SourceHealthState.DEGRADED
                    self.last_error = "Open-Meteo payload missing hourly time series"
                    return {
                        "status": "UNAVAILABLE",
                        "provider": "Open-Meteo NWP Forecast API",
                        "error": self.last_error,
                        "values": None,
                        "timestamp": {"fetch_time_epoch": t0}
                    }

                # Extract first / current hour entry
                idx = 0
                obs_time = times[idx]
                cape_val = hourly.get("cape", [None])[idx]
                cin_val = hourly.get("convective_inhibition", [None])[idx]
                freeze_m = hourly.get("freezing_level_height", [None])[idx]

                cape_jkg = float(cape_val) if cape_val is not None else None
                cin_jkg = float(cin_val) if cin_val is not None else None
                freezing_m = float(freeze_m) if freeze_m is not None else None
                freezing_km = round(freezing_m / 1000.0, 2) if freezing_m is not None else None
                charging_lower_km = round(freezing_km + 1.5, 2) if freezing_km is not None else None

                payload = {
                    "status": "LIVE",
                    "provider": "Open-Meteo NWP Forecast API (NOAA GFS / ECMWF IFS)",
                    "location": {
                        "city": "Ajmer",
                        "state": "Rajasthan",
                        "lat": round(float(lat), 4),
                        "lon": round(float(lon), 4)
                    },
                    "values": {
                        "cape_jkg": cape_jkg,
                        "cin_jkg": cin_jkg,
                        "freezing_level_m": freezing_m,
                        "freezing_level_km": freezing_km,
                        "charging_lower_km": charging_lower_km
                    },
                    "units": {
                        "cape": hourly_units.get("cape", "J/kg"),
                        "cin": hourly_units.get("convective_inhibition", "J/kg"),
                        "freezing_level": hourly_units.get("freezing_level_height", "m")
                    },
                    "timestamp": {
                        "observation_iso": obs_time,
                        "fetch_time_epoch": t0,
                        "freshness_sec": round(time.time() - t0, 1)
                    }
                }

                self.last_success_time = time.time()
                self.cached_data = payload
                self.last_error = None
                self.state = SourceHealthState.LIVE
                return payload

        except Exception as e:
            latency = (time.time() - t0) * 1000.0
            self.last_latency_ms = latency
            self.state = SourceHealthState.UNAVAILABLE
            self.last_error = f"Network exception: {type(e).__name__} - {str(e)}"
            return {
                "status": "UNAVAILABLE",
                "provider": "Open-Meteo NWP Forecast API",
                "error": self.last_error,
                "values": None,
                "timestamp": {"fetch_time_epoch": t0}
            }

    def fetch(self, timeout: float = 5.0) -> IngestionResult:
        t0 = time.time()
        res = self.fetch_nwp(timeout=timeout)
        success = (res.get("status") == "LIVE")
        state = SourceHealthState.LIVE if success else SourceHealthState.UNAVAILABLE
        return IngestionResult(
            source_id=self.source_id,
            success=success,
            state=state,
            timestamp=t0,
            data=res,
            error=res.get("error"),
            latency_ms=self.last_latency_ms
        )

    def validate(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("status") in ("LIVE", "STALE") and "values" in payload:
            return True
        if "hourly" in payload or "cape" in payload or "raw_size_bytes" in payload:
            return True
        return False


class MosdacDwrAdapter(BaseIngestionAdapter):
    """Adapter for MOSDAC / ISRO 3D Volumetric Numerical Doppler Weather Radar (HDF5 format)."""

    def __init__(self):
        super().__init__(
            source_id="mosdac_dwr",
            name="MOSDAC / ISRO 3D Volumetric Numerical DWR (HDF5)",
            channels=["refl_sfc", "refl_m10", "echo_top", "vil"],
            env_var_key="MOSDAC_DWR_API_URL"
        )

    def validate(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "data" in payload or "quantities" in payload or "refl_sfc" in payload or "raw_size_bytes" in payload:
            return True
        return False


class ImdDwrAdapter(BaseIngestionAdapter):
    """Adapter for IMD Doppler Weather Radar Network (ODIM HDF5 / NetCDF4 format)."""

    def __init__(self):
        super().__init__(
            source_id="imd_dwr",
            name="IMD Doppler Weather Radar Network (ODIM HDF5 / NetCDF)",
            channels=["refl_sfc", "refl_m10", "echo_top", "vil"],
            env_var_key="IMD_DWR_API_URL"
        )

    def validate(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if "data" in payload or "variables" in payload or "refl_sfc" in payload or "raw_size_bytes" in payload:
            return True
        return False


# ---------------------------------------------------------------------------
# Central Source Manager
# ---------------------------------------------------------------------------

class CentralSourceManager:
    """
    Central Source Manager managing multi-source adapters independently.
    Handles per-adapter timeouts, state tracking, and 6-channel contract assembly.
    """

    def __init__(self):
        self.adapters: Dict[str, BaseIngestionAdapter] = {
            "radar": RadarAdapter(),
            "mosdac_dwr": MosdacDwrAdapter(),
            "imd_dwr": ImdDwrAdapter(),
            "insat": INSATAdapter(),
            "lightning": LightningAdapter(),
            "aws": AWSAdapter(),
            "nwp": NWPAdapter(),
        }

    def fetch_all(self, timeout: float = 5.0) -> Dict[str, IngestionResult]:
        """Fetch all adapters independently and concurrently with per-adapter timeout."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.adapters)) as executor:
            future_to_id = {
                executor.submit(adapter.fetch, timeout=timeout): source_id
                for source_id, adapter in self.adapters.items()
            }
            for future in concurrent.futures.as_completed(future_to_id):
                source_id = future_to_id[future]
                try:
                    res = future.result()
                    results[source_id] = res
                except Exception as exc:
                    results[source_id] = IngestionResult(
                        source_id=source_id,
                        success=False,
                        state=SourceHealthState.UNAVAILABLE,
                        timestamp=time.time(),
                        error=f"Task exception: {exc}"
                    )
        return results

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Return health status dictionaries for all 5 adapters."""
        return {
            source_id: adapter.get_health().to_dict()
            for source_id, adapter in self.adapters.items()
        }

    def assemble_six_channel_tensor(
        self,
        results: Dict[str, IngestionResult],
        demo_fallback: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Assembles multi-source observations into the exact six-channel array contract
        shape (6, 64, 64) or (T_in, 6, 64, 64) for refl_sfc, refl_m10, echo_top, vil, ir_tb, light_dens.
        Returns tensor and channel source mappings.
        """
        grid_h, grid_w = C.GRID_H, C.GRID_W
        n_frames = C.INPUT_FRAMES

        # Default fallback frame stack shape (T_in, 6, 64, 64)
        if demo_fallback is not None:
            tensor = demo_fallback.copy()
        else:
            tensor = np.zeros((n_frames, C.N_CHANNELS, grid_h, grid_w), dtype=np.float32)

        sources_mapped = {}

        # 1. Radar Channels (refl_sfc, refl_m10, echo_top, vil)
        radar_res = results.get("radar")
        if radar_res and radar_res.success and radar_res.data:
            data = radar_res.data
            for ch in ["refl_sfc", "refl_m10", "echo_top", "vil"]:
                if ch in data and isinstance(data[ch], list):
                    arr = np.array(data[ch], dtype=np.float32)
                    if arr.shape == (grid_h, grid_w):
                        tensor[:, C.CH[ch], :, :] = arr
                    elif arr.shape == (n_frames, grid_h, grid_w):
                        tensor[:, C.CH[ch], :, :] = arr
                    sources_mapped[ch] = "LIVE_RADAR"
                else:
                    sources_mapped[ch] = "FALLBACK_DEMO"
        else:
            for ch in ["refl_sfc", "refl_m10", "echo_top", "vil"]:
                sources_mapped[ch] = f"UNAVAILABLE_OR_NOT_CONFIGURED ({self.adapters['radar'].state.value})"

        # 2. INSAT Channel (ir_tb)
        insat_res = results.get("insat")
        if insat_res and insat_res.success and insat_res.data:
            data = insat_res.data
            if "ir_tb" in data and isinstance(data["ir_tb"], list):
                arr = np.array(data["ir_tb"], dtype=np.float32)
                if arr.shape == (grid_h, grid_w):
                    tensor[:, C.CH["ir_tb"], :, :] = arr
                elif arr.shape == (n_frames, grid_h, grid_w):
                    tensor[:, C.CH["ir_tb"], :, :] = arr
                sources_mapped["ir_tb"] = "LIVE_INSAT"
            else:
                sources_mapped["ir_tb"] = "FALLBACK_DEMO"
        else:
            sources_mapped["ir_tb"] = f"UNAVAILABLE_OR_NOT_CONFIGURED ({self.adapters['insat'].state.value})"

        # 3. Lightning Channel (light_dens)
        light_res = results.get("lightning")
        if light_res and light_res.success and light_res.data:
            data = light_res.data
            if "light_dens" in data and isinstance(data["light_dens"], list):
                arr = np.array(data["light_dens"], dtype=np.float32)
                if arr.shape == (grid_h, grid_w):
                    tensor[:, C.CH["light_dens"], :, :] = arr
                elif arr.shape == (n_frames, grid_h, grid_w):
                    tensor[:, C.CH["light_dens"], :, :] = arr
                sources_mapped["light_dens"] = "LIVE_LIGHTNING"
            else:
                sources_mapped["light_dens"] = "FALLBACK_DEMO"
        else:
            sources_mapped["light_dens"] = f"UNAVAILABLE_OR_NOT_CONFIGURED ({self.adapters['lightning'].state.value})"

        return tensor, sources_mapped


# Singleton Instance for system execution
CENTRAL_SOURCE_MANAGER = CentralSourceManager()

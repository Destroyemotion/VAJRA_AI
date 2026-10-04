"""
Real-data loaders and quality control for the lightning nowcasting system.

WHY THIS FILE EXISTS
--------------------
`synth.py` produces events that are physically reasoned but fabricated. Nothing
in a fabricated dataset can tell you whether the system works. This module is
the bridge to observations: every loader here returns *exactly* the same event
dictionary that `synth.generate_event` returns, so the entire downstream stack
(features -> cv -> model -> metrics) is agnostic to where the numbers came
from, and a real run is a one-line substitution rather than a rewrite.

    {"x": (T, C, H, W) float32,   # C.CHANNELS, FILL_VALUE where missing
     "mask": (T, C, H, W) bool,   # True where the value is a real observation
     "y": (T, H, W) float32,      # strike counts per pixel per TIMESTEP_MIN
     "meta": dict}                # provenance, geometry, honesty flags

THE CONSTRAINT THIS FILE WAS WRITTEN UNDER
------------------------------------------
It was developed in a sandbox with NO network access and with none of h5py,
netCDF4, xarray, boto3, scipy, sklearn, pyart or torch installed. Two
consequences shape the design, and both are deliberate:

  1. EVERY optional dependency is imported lazily, inside the function that
     needs it, and a missing one raises `MissingDependency` naming the exact
     pip package. Importing this module, running `available_sources()`, running
     `validate_event`, `quality_control`, `polar_to_cartesian`,
     `dequantize_colormap`, `save_events` and `load_events` all work with
     nothing but numpy. So the parts that encode the *physics* are testable
     here and now; only the parts that touch bytes on disk need the extras.

  2. NO SCHEMA IN THIS FILE HAS BEEN VERIFIED AGAINST A REAL FILE. Dataset
     layouts (HDF5 dataset names, variable names, column orders, scale factors,
     array axis orders) are recalled from documentation, not read off disk.
     Every such detail carries a `# VERIFY:` comment. Treat them as a starting
     hypothesis to check with `h5ls` / `ncdump -h` on the first real file, not
     as fact. A loader that silently transposes an axis or misses a scale
     factor produces plausible-looking arrays and a quietly worthless model,
     which is the worst possible failure mode; hence the noise.

WHAT QUALITY CONTROL IS FOR HERE
--------------------------------
QC in this project is not cosmetic cleaning. Three of the four rules exist
specifically because they generate FALSE ALARMS in a lightning nowcaster:

  * anomalous propagation / ground clutter -- a 45 dBZ stationary echo with no
    vertical extent. A model keying on surface reflectivity calls it a storm.
  * bright band -- melting aggregates, high dielectric constant, look intense
    on the lowest tilt, contain no graupel and no updraft, produce no lightning.
  * range degradation -- beyond a certain range the radar cannot see the
    charging layer at all, and directly overhead it cannot either. Feeding the
    model a fabricated value there teaches it that the charging-layer channel
    is unreliable, which is the opposite of what we want.

The fourth (physical range check) exists because a silent clamp is a lie: a
-9999 fill masquerading as -32 dBZ is indistinguishable from weak drizzle.
Out-of-range values become FILL_VALUE with mask False, never a clamped number.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

import config as C


# ===========================================================================
# Errors and lazy dependency handling
# ===========================================================================

class MissingDependency(ImportError):
    """Raised when a loader needs an optional package that is not installed."""


class EventContractError(ValueError):
    """Raised when an event dict violates the interface all loaders promise."""


# Module import name -> pip install name. They differ often enough that
# printing the module name alone sends people to a nonexistent package.
PIP_NAME: dict[str, str] = {
    "h5py": "h5py",
    "netCDF4": "netCDF4",
    "boto3": "boto3",
    "s3fs": "s3fs",
    "pyart": "arm_pyart",          # module is `pyart`, package is `arm_pyart`
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "requests": "requests",
    "pandas": "pandas",
    "xarray": "xarray",
    "scipy": "scipy",
}


def _require(module: str, why: str):
    """Import `module` lazily, or raise an actionable MissingDependency.

    Every loader calls this instead of a top-level import. The point is that
    `import data_io` must never fail on a machine that only wants the
    synthetic path or only wants QC, and that a user who does want a real
    loader is told the exact command to run rather than getting a bare
    ModuleNotFoundError three frames deep.
    """
    import importlib
    try:
        return importlib.import_module(module)
    except Exception as exc:  # ImportError, but also broken installs
        pkg = PIP_NAME.get(module, module)
        raise MissingDependency(
            f"'{module}' is required to {why}, and it is not importable "
            f"({type(exc).__name__}: {exc}).\n"
            f"    Install it with:  pip install {pkg}\n"
            f"    Everything else in data_io.py (QC, polar gridding, colour "
            f"dequantisation, save/load) works without it."
        ) from exc


def available_sources() -> dict[str, Any]:
    """Introspect which real-data sources are loadable on THIS machine.

    Returns a dict keyed by source name (matching config.DATA_SOURCES) plus a
    "_env" block. Nothing is imported for real -- we use importlib.find_spec so
    that probing does not pay the cost or the side effects of an import.

    Deliberately does NOT test network reachability. A find_spec probe is
    instant and deterministic; a network probe is neither, and a False from it
    could mean "no credentials", "S3 is down", "corporate proxy" or "the DNS
    lookup timed out", which is not a useful thing to print in a startup
    banner. Credentials that exist as environment variables ARE reported,
    because their absence is a definite, fixable blocker.
    """
    import importlib.util

    def _have(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            # find_spec raises if a parent package is missing or broken.
            return False

    probed = {m: _have(m) for m in sorted(PIP_NAME)}

    out: dict[str, Any] = {}
    for name, spec in C.DATA_SOURCES.items():
        needs = list(spec.get("needs", []))
        missing = [m for m in needs if not probed.get(m, _have(m))]
        out[name] = {
            "desc": spec.get("desc", ""),
            "needs": needs,
            "missing": missing,
            "ready": not missing,
            "auth": spec.get("auth", ""),
            "cadence_min": spec.get("cadence_min"),
            "install": ("pip install "
                        + " ".join(PIP_NAME.get(m, m) for m in missing)
                        ) if missing else "",
        }

    # Credentials. Never read the value, only whether it is set -- printing a
    # token into a log is how tokens leak.
    env_keys = ("MOSDAC_TOKEN", "MOSDAC_USER", "MOSDAC_PASSWORD",
                "IMD_API_KEY", "AWS_ACCESS_KEY_ID", "OPENMETEO_API_KEY")
    out["_env"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "modules": probed,
        "credentials_set": {k: bool(os.environ.get(k)) for k in env_keys},
        "note": ("Module availability only. Network reachability and "
                 "credential validity are NOT tested here."),
    }
    return out


def format_sources_report(rep: dict[str, Any] | None = None) -> str:
    """Human-readable form of `available_sources()`, for `run.py describe`."""
    rep = available_sources() if rep is None else rep
    env = rep.get("_env", {})
    lines = ["REAL DATA SOURCE AVAILABILITY", "=" * 74]
    lines.append(f"  python {env.get('python','?')}   numpy {env.get('numpy','?')}")
    mods = env.get("modules", {})
    have = [m for m, v in mods.items() if v]
    miss = [m for m, v in mods.items() if not v]
    lines.append(f"  importable : {', '.join(have) if have else '(none)'}")
    lines.append(f"  missing    : {', '.join(miss) if miss else '(none)'}")
    lines.append("")
    for name, d in rep.items():
        if name.startswith("_"):
            continue
        status = "READY" if d["ready"] else "BLOCKED"
        lines.append(f"  [{status:7s}] {name:10s} {d['desc']}")
        if not d["ready"]:
            lines.append(f"  {'':10s} {'':10s} missing {d['missing']} -> {d['install']}")
        if d.get("auth") and d["auth"] != "none":
            lines.append(f"  {'':10s} {'':10s} auth: {d['auth']}")
    creds = env.get("credentials_set", {})
    if creds:
        lines.append("")
        lines.append("  credentials in environment:")
        for k, v in creds.items():
            lines.append(f"    {k:22s} {'set' if v else 'not set'}")
    lines.append("")
    lines.append("  " + env.get("note", ""))
    lines.append("  Schemas in data_io.py are UNVERIFIED (see the `# VERIFY:` "
                 "comments); check the first real file with h5ls / ncdump -h.")
    return "\n".join(lines)


# ===========================================================================
# The event contract
# ===========================================================================

def validate_event(ev: Any, *, require_grid: bool = True,
                   strict_fill: bool = True,
                   name: str = "event") -> dict[str, Any]:
    """Check that `ev` satisfies the interface every loader must produce.

    This is the seam between synthetic and real data, and a seam that is not
    checked is a seam that silently drifts. The specific failures it catches:

      * (T, H, W, C) instead of (T, C, H, W). Both are 4-D and both broadcast
        happily through most of the stack; the model just learns nothing.
      * float64 arrays. Not wrong, but they quadruple the memory of the
        sequence tensors relative to what the model layer expects.
      * NaN anywhere. `config.FILL_VALUE` exists precisely so that missing data
        does not propagate through a convolution and turn a whole feature map
        into NaN. A single NaN that reaches the model destroys the run.
      * values outside CHANNEL_RANGES while claiming to be valid -- almost
        always an un-applied scale factor or an unmasked sentinel.
      * mask False but the value is not FILL_VALUE -- means some code path
        masked the flag without blanking the number, so the number is still
        readable by anything that ignores the mask.

    Structural violations raise EventContractError. Softer findings are
    returned in `report["warnings"]` so a caller can decide.

    Args:
        require_grid: enforce H == config.GRID_H and W == config.GRID_W.
            Set False when validating an event on a native sensor grid before
            resampling.
        strict_fill: require x == FILL_VALUE wherever mask is False.
    """
    warnings: list[str] = []

    if not isinstance(ev, dict):
        raise EventContractError(f"{name}: expected a dict, got {type(ev).__name__}")
    for k in ("x", "mask", "y"):
        if k not in ev:
            raise EventContractError(
                f"{name}: missing key {k!r}; an event needs "
                f"'x', 'mask', 'y' and 'meta'")

    x, mask, y = ev["x"], ev["mask"], ev["y"]
    for k, a in (("x", x), ("mask", mask), ("y", y)):
        if not isinstance(a, np.ndarray):
            raise EventContractError(
                f"{name}: {k} must be a numpy array, got {type(a).__name__}")

    if x.ndim != 4:
        raise EventContractError(
            f"{name}: x must be 4-D (T, C, H, W), got shape {x.shape}")
    T, Cn, H, W = x.shape

    if Cn != C.N_CHANNELS:
        # The commonest cause is a channels-last array.
        hint = ""
        if x.shape[-1] == C.N_CHANNELS:
            hint = (" -- shape looks channels-LAST (T, H, W, C); "
                    "use np.moveaxis(x, -1, 1)")
        raise EventContractError(
            f"{name}: x has {Cn} channels, config.CHANNELS defines "
            f"{C.N_CHANNELS} ({', '.join(C.CHANNELS)}){hint}")

    if require_grid and (H != C.GRID_H or W != C.GRID_W):
        raise EventContractError(
            f"{name}: x is {H}x{W}, config grid is {C.GRID_H}x{C.GRID_W}. "
            f"Resample with data_io.resample_grid before returning an event.")

    if mask.shape != x.shape:
        raise EventContractError(
            f"{name}: mask shape {mask.shape} != x shape {x.shape}")
    if mask.dtype != np.bool_:
        raise EventContractError(
            f"{name}: mask must be dtype bool, got {mask.dtype}")

    if y.ndim != 3:
        raise EventContractError(
            f"{name}: y must be 3-D (T, H, W), got shape {y.shape}")
    if y.shape != (T, H, W):
        raise EventContractError(
            f"{name}: y shape {y.shape} != (T, H, W) = {(T, H, W)}")

    if x.dtype != np.float32:
        warnings.append(f"x dtype is {x.dtype}, expected float32")
    if y.dtype != np.float32:
        warnings.append(f"y dtype is {y.dtype}, expected float32")

    if not np.all(np.isfinite(x)):
        n_bad = int((~np.isfinite(x)).sum())
        raise EventContractError(
            f"{name}: x contains {n_bad} non-finite values. Use "
            f"config.FILL_VALUE ({C.FILL_VALUE}) plus mask=False for missing "
            f"data; NaN propagates through convolutions and poisons the run.")
    if not np.all(np.isfinite(y)):
        raise EventContractError(f"{name}: y contains non-finite values")
    if float(y.min()) < 0.0:
        raise EventContractError(
            f"{name}: y has negative strike counts (min {float(y.min())})")

    # Per-channel physical range check, only where the mask claims validity.
    per_channel: dict[str, dict[str, float]] = {}
    for ci, ch in enumerate(C.CHANNELS):
        lo, hi = C.CHANNEL_RANGES[ch]
        v = mask[:, ci]
        vals = x[:, ci][v]
        n_valid = int(v.sum())
        entry = {
            "valid_frac": float(v.mean()),
            "n_valid": n_valid,
            "min": float(vals.min()) if n_valid else float("nan"),
            "max": float(vals.max()) if n_valid else float("nan"),
            "mean": float(vals.mean()) if n_valid else float("nan"),
        }
        if n_valid:
            n_out = int(((vals < lo) | (vals > hi)).sum())
            entry["n_out_of_range"] = n_out
            if n_out:
                warnings.append(
                    f"channel {ch!r}: {n_out} valid-flagged values outside "
                    f"[{lo}, {hi}] (observed {entry['min']:.3g} .. "
                    f"{entry['max']:.3g}) -- suspect a missing scale factor "
                    f"or an unmasked sentinel; run quality_control()")
        else:
            entry["n_out_of_range"] = 0
            warnings.append(f"channel {ch!r} has no valid pixels at all")
        per_channel[ch] = entry

    if strict_fill:
        bad_fill = int((~mask & (x != np.float32(C.FILL_VALUE))).sum())
        if bad_fill:
            warnings.append(
                f"{bad_fill} elements are masked invalid but do not hold "
                f"FILL_VALUE; anything that ignores the mask will read them")

    meta = ev.get("meta")
    if meta is None:
        warnings.append("no 'meta' key: provenance is untracked")
        meta = {}
    elif not isinstance(meta, dict):
        raise EventContractError(f"{name}: meta must be a dict, got {type(meta).__name__}")

    surrogates = list(meta.get("surrogate_channels", []))
    if surrogates:
        warnings.append(
            f"channels {surrogates} are ESTIMATED, not measured "
            f"(source: {meta.get('source', '?')}). Any skill attributed to "
            f"them is skill of the estimator, not of the observation.")

    pos = float((y >= C.LIGHTNING_THRESHOLD).mean())
    if pos == 0.0:
        warnings.append("event contains zero lightning pixels "
                        "(legitimate as a null case, useless alone)")

    return {
        "ok": True,
        "shape": {"T": int(T), "C": int(Cn), "H": int(H), "W": int(W)},
        "dtypes": {"x": str(x.dtype), "mask": str(mask.dtype), "y": str(y.dtype)},
        "valid_frac_overall": float(mask.mean()),
        "per_channel": per_channel,
        "base_rate": pos,
        "total_strikes": float(y.sum()),
        "surrogate_channels": surrogates,
        "warnings": warnings,
    }


# ===========================================================================
# Radar beam geometry -- shared by QC and by the polar gridder
# ===========================================================================

EARTH_RADIUS_KM = 6371.0

# Effective-earth-radius multiplier for standard atmospheric refraction.
#
# Two conventions are in circulation and they are NOT the same number:
#   4/3 = 1.3333  the classical "four-thirds earth" (Doviak & Zrnic)
#   1.21          used in NWS/WDTD operational beam-height calculations
# The difference at 150 km range is about 140 m of beam height, which is small
# against a 1 deg beam that is 2.6 km deep at that range -- so the choice does
# not change any decision here, but it must be stated rather than left implicit.
# VERIFY: confirm which convention your DWR metadata assumes before comparing
# derived echo tops against an operational product.
EARTH_FACTOR = 1.21


def beam_height_km(range_km: np.ndarray | float, elev_deg: np.ndarray | float,
                   radar_alt_km: float = 0.0,
                   earth_factor: float = EARTH_FACTOR) -> np.ndarray:
    """Height of the beam centre above the radar, standard refraction.

    Exact form under the effective-earth approximation:

        h = sqrt(r^2 + (k*Re)^2 + 2*r*k*Re*sin(elev)) - k*Re + radar_alt

    which for r << k*Re reduces to the familiar

        h ~ r*sin(elev) + r^2 / (2*k*Re)

    The second term is the one people forget, and it dominates at range: at
    150 km with a 0.5 deg tilt, r*sin(elev) contributes 1.3 km but the earth
    curvature term contributes another 1.5 km. That is why "the lowest tilt"
    is not "the surface" -- it is a surface only near the radar.
    """
    r = np.asarray(range_km, dtype=np.float64)
    e = np.radians(np.asarray(elev_deg, dtype=np.float64))
    ka = earth_factor * EARTH_RADIUS_KM
    return np.sqrt(r * r + ka * ka + 2.0 * r * ka * np.sin(e)) - ka + radar_alt_km


def beam_width_km(range_km: np.ndarray | float,
                  beamwidth_deg: float = 1.0) -> np.ndarray:
    """Vertical extent of the half-power beam at a given range.

    A 1 deg beam is 2.6 km deep at 150 km and 3.5 km deep at 200 km. Once the
    beam depth exceeds the thickness of the charging layer (6.5-8 km, i.e.
    1.5 km) the "reflectivity at -10C" retrieval is an average over a slab
    that includes things that are not the charging layer -- most damagingly
    the melting layer at 5 km.
    """
    return np.asarray(range_km, dtype=np.float64) * math.radians(beamwidth_deg)


def range_of_altitude_km(alt_km: float, elev_deg: float,
                         earth_factor: float = EARTH_FACTOR,
                         max_km: float = 500.0) -> float:
    """Range at which a beam at `elev_deg` reaches `alt_km`. Monotone bisection."""
    lo, hi = 0.0, float(max_km)
    if float(beam_height_km(hi, elev_deg, earth_factor=earth_factor)) < alt_km:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(beam_height_km(mid, elev_deg, earth_factor=earth_factor)) < alt_km:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def radar_geometry_report(min_elev_deg: float = 0.5,
                          max_elev_deg: float = 20.0,
                          beamwidth_deg: float = 1.0,
                          charging_lower_km: float | None = None,
                          charging_upper_km: float | None = None,
                          freezing_level_km: float | None = None) -> dict:
    """The three ranges that decide where refl_m10 can exist at all.

    THE GEOMETRY, stated once so the QC rules below can just refer to it.

    A radar samples a set of cones, not a volume. To report reflectivity at
    the -10C level (config.CHARGING_LOWER_KM, 6.5 km for the Indian warm-season
    sounding) some tilt must actually pass through 6.5 km at that horizontal
    distance. Three regimes make that impossible:

      NEAR  (cone of silence). Directly over the radar every tilt is low in
            altitude: the highest tilt reaches only r*sin(elev_max). Inside
            roughly charging_lower / tan(elev_max) km there is NO tilt that
            samples the charging layer -- the storm overhead is invisible
            above a couple of kilometres. This is not a data-quality subtlety,
            it is a hole in the observation, and it sits exactly where
            coverage looks best on a map.

      FAR   (overshoot). Beam height grows as r*sin(elev) + r^2/(2*k*Re). Past
            a certain range even the LOWEST tilt is already above the charging
            layer, so the lowest tilt no longer sees the low levels and no
            tilt at all samples 6.5 km from below. Everything reported there
            is extrapolation.

      MID   (beam depth). Well before overshoot, the beam becomes deeper than
            the charging layer is thick. The value labelled "6.5 km" is then a
            weighted average over several kilometres of altitude, and once the
            beam bottom drops below the 0C level it includes the bright band --
            which is exactly the signal we are trying to keep OUT of the
            charging-layer channel.

    Returns the numeric boundaries so QC can use them and so a report can
    print them instead of asserting them.
    """
    cl = C.CHARGING_LOWER_KM if charging_lower_km is None else charging_lower_km
    cu = C.CHARGING_UPPER_KM if charging_upper_km is None else charging_upper_km
    fl = C.FREEZING_LEVEL_KM if freezing_level_km is None else freezing_level_km

    cone = cl / math.tan(math.radians(max_elev_deg))
    overshoot = range_of_altitude_km(cl, min_elev_deg)
    overshoot_upper = range_of_altitude_km(cu, min_elev_deg)
    # Range at which the beam's lower half-power edge, centred on the charging
    # layer, has descended to the melting level -> bright-band contamination.
    half_gap = max(cl - fl, 1e-6)
    bb_contam = 2.0 * half_gap / math.radians(beamwidth_deg)
    # Range at which the beam is thicker than the charging layer itself.
    thick = max(cu - cl, 1e-6) / math.radians(beamwidth_deg)
    return {
        "min_elev_deg": float(min_elev_deg),
        "max_elev_deg": float(max_elev_deg),
        "beamwidth_deg": float(beamwidth_deg),
        "charging_lower_km": float(cl),
        "charging_upper_km": float(cu),
        "freezing_level_km": float(fl),
        "cone_of_silence_km": float(cone),
        "overshoot_range_km": float(overshoot),
        "overshoot_range_upper_km": float(overshoot_upper),
        "beam_thicker_than_layer_km": float(thick),
        "bright_band_contamination_km": float(bb_contam),
    }


def range_azimuth_grid(grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
                       pixel_km: float = C.PIXEL_KM,
                       radar_row: float | None = None,
                       radar_col: float | None = None,
                       north_up: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Slant-ground range (km) and meteorological azimuth (deg) of each pixel.

    Convention, stated explicitly because getting it wrong rotates the whole
    radar volume and nothing downstream complains:
      * row 0 is the NORTHERNMOST row when north_up=True (image convention).
      * azimuth is meteorological: 0 = north, increasing CLOCKWISE toward east.
    """
    rr = (grid_h - 1) / 2.0 if radar_row is None else float(radar_row)
    cc = (grid_w - 1) / 2.0 if radar_col is None else float(radar_col)
    yy, xx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float64)
    east_km = (xx - cc) * pixel_km
    north_km = ((rr - yy) if north_up else (yy - rr)) * pixel_km
    rng = np.hypot(east_km, north_km)
    az = np.degrees(np.arctan2(east_km, north_km)) % 360.0
    return rng, az


# ===========================================================================
# Quality control
# ===========================================================================

def quality_control(x: np.ndarray,
                    mask: np.ndarray | None = None,
                    radar_center: tuple[float, float] | None = None,
                    max_range_km: float | None = None,
                    *,
                    pixel_km: float = C.PIXEL_KM,
                    min_elev_deg: float = 0.5,
                    max_elev_deg: float = 20.0,
                    beamwidth_deg: float = 1.0,
                    ap_sfc_dbz: float = 35.0,
                    ap_top_km: float = 1.5,
                    ap_temporal_std_db: float = 1.5,
                    bb_sfc_dbz: tuple[float, float] = (28.0, 47.0),
                    bb_top_margin_km: float = 2.5,
                    bb_temporal_std_db: float = 3.0,
                    mask_bright_band_sfc: bool = False,
                    north_up: bool = True,
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Flag and mask physically impossible or physically misleading pixels.

    Args:
        x: (T, C, H, W) float array in config.CHANNELS order.
        mask: (T, C, H, W) bool, True where observed. None means all observed.
        radar_center: (row, col) of the radar in grid pixels. None disables
            every range-geometry rule -- and that is a real loss, so pass it.
        max_range_km: maximum range at which the charging-layer retrieval is
            trusted. None uses the physically derived overshoot range for
            `min_elev_deg`; pass the radar's documented quantitative range if
            you know it (IMD S-band DWRs are typically quoted as 250 km
            surveillance / 150 km Doppler -- VERIFY: recalled from memory,
            check the station's own specification).
        mask_bright_band_sfc: see the BRIGHT BAND section below. Default False
            on purpose.

    Returns:
        (x_qc, mask_qc, report). `x_qc` and `mask_qc` are copies; the input is
        never modified in place, because QC is frequently run twice with
        different parameters while tuning and an in-place version silently
        compounds.

    NOTHING IS EVER CLAMPED. An out-of-range value becomes FILL_VALUE with
    mask False. Clamping would convert "this number is broken" into "this
    number is 80 dBZ", which is a hail core, which is the single most
    lightning-predictive value in the whole channel.
    """
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[1] != C.N_CHANNELS:
        raise ValueError(
            f"quality_control expects (T, {C.N_CHANNELS}, H, W), got {x.shape}")
    T, nC, H, W = x.shape
    x_qc = x.astype(np.float32, copy=True)
    if mask is None:
        m_qc = np.ones(x.shape, dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.shape != x.shape:
            raise ValueError(f"mask shape {mask.shape} != x shape {x.shape}")
        m_qc = mask.astype(bool, copy=True)

    fill = np.float32(C.FILL_VALUE)
    valid_before = {ch: float(m_qc[:, i].mean()) for i, ch in enumerate(C.CHANNELS)}
    rules: dict[str, Any] = {}

    def _kill(sel: np.ndarray, ci: int) -> int:
        """Apply a (T,H,W) boolean kill selection to channel ci. Counts newly
        removed elements only, so overlapping rules do not double-count."""
        newly = sel & m_qc[:, ci]
        n = int(newly.sum())
        if n:
            m_qc[:, ci][newly] = False
            x_qc[:, ci][newly] = fill
        return n

    # -- Rule 0: non-finite ------------------------------------------------
    # NaN/Inf reaching a convolution turns an entire feature map into NaN, and
    # the loss becomes NaN several minutes into training with no indication of
    # which channel did it.
    nonfinite = ~np.isfinite(x_qc)
    n_nf = {}
    for ci, ch in enumerate(C.CHANNELS):
        n_nf[ch] = _kill(nonfinite[:, ci], ci)
    rules["nonfinite"] = {"per_channel": n_nf, "masked": int(sum(n_nf.values()))}

    # -- Rule 1: physical range -------------------------------------------
    # Catches un-applied scale factors, sentinel values (-9999, 999, 65535)
    # that survived a loader, and unit confusion (degrees C vs K on IR is the
    # classic: 250 K read as 250 C, or -23 C read as a brightness temperature).
    n_range: dict[str, int] = {}
    for ci, ch in enumerate(C.CHANNELS):
        lo, hi = C.CHANNEL_RANGES[ch]
        bad = (x_qc[:, ci] < lo) | (x_qc[:, ci] > hi)
        n_range[ch] = _kill(bad, ci)
    rules["range_check"] = {
        "per_channel": n_range,
        "masked": int(sum(n_range.values())),
        "policy": "out-of-range -> FILL_VALUE + mask False (never clamped)",
    }

    # Convenience views AFTER the range check, so the derived statistics below
    # are not computed over sentinels.
    i_sfc, i_m10 = C.CH["refl_sfc"], C.CH["refl_m10"]
    i_top, i_vil = C.CH["echo_top"], C.CH["vil"]

    def _temporal_std(ci: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-pixel std over time using only valid frames. Returns (std, ok)."""
        v = m_qc[:, ci]
        n = v.sum(axis=0)
        ok = n >= 3                     # fewer than 3 frames: cannot judge
        num = np.where(v, x_qc[:, ci], 0.0).sum(axis=0)
        mean = np.divide(num, np.maximum(n, 1), dtype=np.float64)
        var = np.divide(
            np.where(v, (x_qc[:, ci] - mean) ** 2, 0.0).sum(axis=0),
            np.maximum(n, 1), dtype=np.float64)
        return np.sqrt(var), ok

    sfc_std, sfc_std_ok = _temporal_std(i_sfc)

    # -- Rule 2: anomalous propagation and ground clutter ------------------
    #
    # WHAT IT IS. Under a super-refractive layer -- a nocturnal radiation
    # inversion, or moist air over irrigated land or a sea breeze, both
    # extremely common over the Indo-Gangetic plain in the pre-monsoon night --
    # the beam bends downward faster than the earth curves and strikes the
    # ground tens of kilometres out. Ground is an enormous scatterer, so the
    # return is strong: 35-55 dBZ is routine.
    #
    # WHY IT MATTERS HERE SPECIFICALLY. It looks exactly like a severe cell on
    # surface reflectivity alone, which is the channel a naive model leans on.
    # It never produces lightning. Every AP pixel a model treats as a storm is
    # a false alarm, and false alarms are what destroy operational trust in a
    # lightning warning.
    #
    # THE THREE-PART SIGNATURE, all required together:
    #   (a) strong low-level return          -> it is bright
    #   (b) near-zero echo top               -> it has no vertical extent, so
    #                                           it is not a convective column
    #   (c) near-zero temporal variability   -> it does not move or evolve.
    #                                           A real cell changes intensity
    #                                           frame to frame and advects;
    #                                           a hill does neither.
    # (c) is the discriminating test. (a)+(b) alone would also flag a shallow
    # warm-rain shower, which is real precipitation we want to keep.
    top_valid = m_qc[:, i_top]
    ap = ((x_qc[:, i_sfc] >= ap_sfc_dbz) & m_qc[:, i_sfc]
          & (x_qc[:, i_top] <= ap_top_km) & top_valid)
    stationary = (sfc_std <= ap_temporal_std_db) & sfc_std_ok
    ap = ap & stationary[None, :, :]
    n_ap: dict[str, int] = {}
    # Kill the radar-derived channels only. IR comes from a satellite and the
    # lightning channel from a ground network; neither is affected by a ground
    # return, and blanking them would throw away good data.
    for ci, ch in ((i_sfc, "refl_sfc"), (i_m10, "refl_m10"),
                   (i_top, "echo_top"), (i_vil, "vil")):
        n_ap[ch] = _kill(ap, ci)
    rules["ap_clutter"] = {
        "criteria": (f"refl_sfc >= {ap_sfc_dbz} dBZ AND echo_top <= "
                     f"{ap_top_km} km AND temporal std(refl_sfc) <= "
                     f"{ap_temporal_std_db} dB over >= 3 valid frames"),
        "n_pixels_flagged": int(ap.any(axis=0).sum()),
        "n_frame_pixels_flagged": int(ap.sum()),
        "per_channel_masked": n_ap,
        "masked": int(sum(n_ap.values())),
    }

    # -- Rule 3: bright band ----------------------------------------------
    #
    # WHAT IT IS. In stratiform rain, snow aggregates fall through the 0C level
    # and acquire a liquid coating before they collapse into raindrops. For a
    # few hundred metres they are large, wet particles: the dielectric factor
    # |K|^2 of water is about 0.93 against roughly 0.20 for ice, so the same
    # particle reflects about 5x more power the instant it is wet -- and it is
    # still aggregate-sized, so the size term amplifies it further. The result
    # is a 5-15 dB horizontal band of enhanced reflectivity sitting right at
    # config.FREEZING_LEVEL_KM.
    #
    # WHY IT IS A FALSE-ALARM GENERATOR, WHICH IS THE ONLY REASON WE CARE.
    # It is bright, it is horizontally extensive, and it is METEOROLOGICALLY
    # REAL -- there is genuine precipitation there. But the microphysics is
    # melting, not riming. There is no graupel, no supercooled liquid water in
    # a strong updraft, and therefore no non-inductive charging. A stratiform
    # rain area with a textbook bright band produces essentially no lightning
    # while displaying 40 dBZ. Any model that maps reflectivity to
    # electrification will warn on it, every time, over a huge area.
    #
    # WHY WE FLAG BUT DO NOT MASK BY DEFAULT (mask_bright_band_sfc=False).
    # Unlike AP, the measurement is not erroneous. Blanking it would delete
    # real precipitation, break the optical-flow advection baseline (which
    # needs the stratiform field to estimate motion), and hide from the model
    # the very population it must learn to reject. The right treatment is to
    # let the model see it and to score it: the flag is returned so that
    # metrics can report the false-alarm rate ON bright-band pixels
    # specifically, which is the number an operational user cares about.
    #
    # WHAT WE DO MASK: refl_m10 where the bright band coincides with a range
    # at which the beam sampling the charging layer is deep enough to reach
    # down to the melting level (see radar_geometry_report). There the
    # "charging layer reflectivity" is partly a measurement of the bright band,
    # which is contamination of the one channel the whole project rests on.
    #
    # DETECTION FROM CHANNELS ALONE IS WEAK. With a full vertical profile you
    # detect a bright band by finding a local reflectivity maximum just below
    # the 0C level with a sharp gradient above it (see polar_to_cartesian,
    # which does have the profile). Working only from the six config channels
    # we can use a proxy: moderate surface echo, echo top not far above the
    # freezing level, little charging-layer signal, and low temporal
    # variability (stratiform evolves slowly). Recall is fair; precision is
    # not. Treated as an advisory flag, not as truth.
    bb_lo, bb_hi = bb_sfc_dbz
    bb = ((x_qc[:, i_sfc] >= bb_lo) & (x_qc[:, i_sfc] <= bb_hi) & m_qc[:, i_sfc]
          & (x_qc[:, i_top] <= C.FREEZING_LEVEL_KM + bb_top_margin_km)
          & (x_qc[:, i_top] >= C.FREEZING_LEVEL_KM - 1.0) & m_qc[:, i_top]
          & (x_qc[:, i_m10] < C.CHARGING_DBZ_THRESHOLD - 10.0) & m_qc[:, i_m10])
    bb = bb & ((sfc_std <= bb_temporal_std_db) & sfc_std_ok)[None, :, :]
    bb_masked = 0
    if mask_bright_band_sfc:
        bb_masked += _kill(bb, i_sfc)

    # -- Rule 4: range degradation ----------------------------------------
    geom = radar_geometry_report(min_elev_deg, max_elev_deg, beamwidth_deg)
    n_range_deg: dict[str, int] = {}
    rng_km = None
    if radar_center is not None:
        rr, cc = float(radar_center[0]), float(radar_center[1])
        rng_km, _az = range_azimuth_grid(H, W, pixel_km, rr, cc, north_up)
        r_max = geom["overshoot_range_km"] if max_range_km is None else float(max_range_km)

        # FAR FIELD. Beyond r_max the lowest tilt is at or above the charging
        # layer (or beyond the quantitative range the operator trusts), so a
        # reported -10C reflectivity is an extrapolation dressed as an
        # observation. Echo top goes too: echo top needs the HIGH tilts, whose
        # beams are far above the storm at long range, so the estimate becomes
        # a lower bound at best.
        far = np.broadcast_to((rng_km > r_max)[None, :, :], (T, H, W))
        n_range_deg["far_refl_m10"] = _kill(far, i_m10)
        n_range_deg["far_echo_top"] = _kill(far, i_top)

        # NEAR FIELD -- the cone of silence. Inside charging_lower/tan(elev_max)
        # no tilt reaches the charging layer. Coverage maps show this region as
        # the best-covered part of the domain, which is precisely backwards for
        # anything that needs data aloft.
        near = np.broadcast_to((rng_km < geom["cone_of_silence_km"])[None, :, :],
                               (T, H, W))
        n_range_deg["cone_of_silence_refl_m10"] = _kill(near, i_m10)
        n_range_deg["cone_of_silence_echo_top"] = _kill(near, i_top)

        # BRIGHT-BAND CONTAMINATION OF THE CHARGING CHANNEL (see Rule 3).
        contam = np.broadcast_to(
            (rng_km > geom["bright_band_contamination_km"])[None, :, :], (T, H, W))
        bb_masked += _kill(bb & contam, i_m10)

    rules["bright_band"] = {
        "criteria": (f"refl_sfc in [{bb_lo}, {bb_hi}] dBZ AND echo_top within "
                     f"[-1.0, +{bb_top_margin_km}] km of the "
                     f"{C.FREEZING_LEVEL_KM} km freezing level AND refl_m10 < "
                     f"{C.CHARGING_DBZ_THRESHOLD - 10.0} dBZ AND temporal "
                     f"std(refl_sfc) <= {bb_temporal_std_db} dB"),
        "n_frame_pixels_flagged": int(bb.sum()),
        "n_pixels_flagged": int(bb.any(axis=0).sum()),
        "masked": int(bb_masked),
        "refl_m10_contamination_range_km": geom["bright_band_contamination_km"],
        "policy": ("FLAG, do not mask, unless mask_bright_band_sfc=True. The "
                   "echo is real; it is misleading, not wrong. refl_m10 IS "
                   "masked where the beam at that range also contains the "
                   "melting level, i.e. beyond "
                   f"{geom['bright_band_contamination_km']:.0f} km -- which a "
                   "domain smaller than that never reaches, so masked=0 with "
                   "a nonzero flag count is the expected result here."),
    }
    rules["range_degradation"] = {
        "enabled": radar_center is not None,
        "radar_center_rowcol": None if radar_center is None else [float(radar_center[0]),
                                                                 float(radar_center[1])],
        "max_range_km_used": (None if radar_center is None else
                              (geom["overshoot_range_km"] if max_range_km is None
                               else float(max_range_km))),
        "per_rule_masked": n_range_deg,
        "masked": int(sum(n_range_deg.values())),
        "note": ("Disabled without radar_center. Passing it is worth doing: "
                 "these are the pixels where refl_m10 is fiction."),
    }

    valid_after = {ch: float(m_qc[:, i].mean()) for i, ch in enumerate(C.CHANNELS)}
    report = {
        "shape": {"T": int(T), "C": int(nC), "H": int(H), "W": int(W)},
        "n_elements": int(x.size),
        "total_masked": int(sum(r.get("masked", 0) for r in rules.values())),
        "rules": rules,
        "valid_frac_before": valid_before,
        "valid_frac_after": valid_after,
        "valid_frac_delta": {ch: valid_after[ch] - valid_before[ch]
                             for ch in C.CHANNELS},
        "geometry": geom,
        "flags": {  # (T, H, W) boolean arrays, for metrics stratification
            "ap_clutter": ap,
            "bright_band": bb,
            "range_km": rng_km,
        },
    }
    return x_qc, m_qc, report


def format_qc_report(rep: dict) -> str:
    """Printable QC summary. Skips the boolean flag arrays."""
    L = ["QUALITY CONTROL REPORT", "=" * 74]
    s = rep["shape"]
    L.append(f"  input {s['T']}x{s['C']}x{s['H']}x{s['W']} = "
             f"{rep['n_elements']:,} elements, {rep['total_masked']:,} newly masked")
    g = rep["geometry"]
    L.append("")
    L.append("  beam geometry (elev {:.1f}-{:.1f} deg, {:.1f} deg beam):".format(
        g["min_elev_deg"], g["max_elev_deg"], g["beamwidth_deg"]))
    L.append(f"    cone of silence           r < {g['cone_of_silence_km']:6.1f} km "
             f"(no tilt reaches {g['charging_lower_km']:.1f} km)")
    L.append(f"    lowest tilt overshoots    r > {g['overshoot_range_km']:6.1f} km "
             f"(lowest beam already above {g['charging_lower_km']:.1f} km)")
    L.append(f"    beam thicker than layer   r > {g['beam_thicker_than_layer_km']:6.1f} km")
    L.append(f"    bright-band contamination r > {g['bright_band_contamination_km']:6.1f} km "
             f"(beam bottom reaches the {g['freezing_level_km']:.1f} km melting level)")
    L.append("")
    for name, r in rep["rules"].items():
        L.append(f"  {name}: masked {r.get('masked', 0):,}")
        if "criteria" in r:
            L.append(f"    criteria: {r['criteria']}")
        for k in ("per_channel", "per_channel_masked", "per_rule_masked"):
            if k in r and any(r[k].values()):
                nz = {a: b for a, b in r[k].items() if b}
                L.append(f"    {k}: {nz}")
        for k in ("n_pixels_flagged", "n_frame_pixels_flagged"):
            if k in r:
                L.append(f"    {k}: {r[k]:,}")
        if "policy" in r:
            L.append(f"    policy: {r['policy']}")
        if "note" in r:
            L.append(f"    note: {r['note']}")
    L.append("")
    L.append(f"  {'channel':12s} {'valid before':>13s} {'valid after':>12s} {'delta':>9s}")
    for ch in C.CHANNELS:
        L.append(f"  {ch:12s} {rep['valid_frac_before'][ch]:13.4f} "
                 f"{rep['valid_frac_after'][ch]:12.4f} "
                 f"{rep['valid_frac_delta'][ch]:+9.4f}")
    return "\n".join(L)


# ===========================================================================
# Regridding helpers (pure numpy; no scipy, no cv2 required)
# ===========================================================================

def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) area-overlap weight matrix, rows summing to 1.

    Exact conservative resampling for downsampling by any (including
    non-integer) ratio. For UPsampling the overlap degenerates to a nearest /
    piecewise-constant assignment rather than a smooth interpolation, which is
    the honest behaviour: upsampling invents no information, and a bilinear
    upsample of an 8 km lightning grid to 2 km would create a smooth field that
    looks like it resolves 2 km structure when it does not.
    """
    s = n_in / float(n_out)
    lo = np.arange(n_out, dtype=np.float64) * s
    hi = lo + s
    j = np.arange(n_in, dtype=np.float64)
    ov = np.clip(np.minimum(hi[:, None], j[None, :] + 1.0)
                 - np.maximum(lo[:, None], j[None, :]), 0.0, None)
    tot = ov.sum(axis=1, keepdims=True)
    tot[tot <= 0] = 1.0
    return ov / tot


def resample_grid(a: np.ndarray, out_h: int, out_w: int,
                  valid: np.ndarray | None = None,
                  fill: float = C.FILL_VALUE,
                  min_valid_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Area-average `a` (..., H, W) onto (..., out_h, out_w), mask-aware.

    Missing pixels are excluded from the average rather than averaged as
    FILL_VALUE, which would drag a -999 into every neighbouring output cell.
    Output cells whose valid fraction falls below `min_valid_frac` are
    themselves marked invalid: a 2 km cell built from one surviving 1 km pixel
    out of four is not a measurement of that cell.
    """
    a = np.asarray(a, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(a) & (a > C.FILL_VALUE / 2.0)
    valid = np.asarray(valid, dtype=bool)
    ih, iw = a.shape[-2:]
    Wr = _area_weights(ih, out_h)
    Wc = _area_weights(iw, out_w)

    av = np.where(valid, a, 0.0)
    vf = valid.astype(np.float64)

    def _apply(z: np.ndarray) -> np.ndarray:
        z = np.tensordot(z, Wc, axes=([-1], [1]))       # (..., ih, out_w)
        z = np.tensordot(z, Wr, axes=([-2], [1]))       # (..., out_w, out_h)
        return np.swapaxes(z, -1, -2)

    num = _apply(av)
    den = _apply(vf)
    ok = den >= min_valid_frac
    out = np.where(ok, np.divide(num, np.maximum(den, 1e-12)), fill)
    return out.astype(np.float32), ok


def crop_centre(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Centre-crop the last two axes. Pads with FILL_VALUE if too small."""
    ih, iw = a.shape[-2:]
    r0 = (ih - out_h) // 2
    c0 = (iw - out_w) // 2
    if r0 >= 0 and c0 >= 0:
        return a[..., r0:r0 + out_h, c0:c0 + out_w]
    pad = [(0, 0)] * (a.ndim - 2)
    pad += [(max(0, -r0), max(0, out_h - ih - max(0, -r0))),
            (max(0, -c0), max(0, out_w - iw - max(0, -c0)))]
    a = np.pad(a, pad, mode="constant", constant_values=C.FILL_VALUE)
    return crop_centre(a, out_h, out_w)


def to_config_grid(a: np.ndarray, native_km: float,
                   valid: np.ndarray | None = None,
                   grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
                   pixel_km: float = C.PIXEL_KM
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Centre-crop a native-resolution field to the config domain, then resample.

    The config domain is 64 x 64 at 2 km = 128 x 128 km. SEVIR's VIL tile is
    384 x 384 at 1 km = 384 x 384 km, so this discards about 89% of the tile's
    area. That is a real cost and worth stating: a storm moving at 60 km/h
    crosses 120 km in the 2 h a SEVIR event spans, so a fast cell will leave a
    128 km box mid-event. If you train on SEVIR at scale, raise GRID_H/GRID_W
    (and accept the compute) rather than silently forecasting storms that have
    already exited the frame.
    """
    need_h = int(round(grid_h * pixel_km / native_km))
    need_w = int(round(grid_w * pixel_km / native_km))
    a_c = crop_centre(a, need_h, need_w)
    v_c = None if valid is None else crop_centre(valid, need_h, need_w)
    return resample_grid(a_c, grid_h, grid_w, valid=v_c)


def latlon_to_rowcol(lat: np.ndarray, lon: np.ndarray,
                     center_lat: float, center_lon: float,
                     grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
                     pixel_km: float = C.PIXEL_KM,
                     north_up: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular projection of lat/lon onto the config grid.

    Valid because the domain is only 128 km across: over that span the error
    of treating the earth as locally flat with a cos(lat) longitude scaling is
    well under one pixel. Do NOT reuse this for a domain of continental size,
    and do not reuse it near the poles.
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
    north_km = (np.asarray(lat, dtype=np.float64) - center_lat) * km_per_deg_lat
    east_km = (np.asarray(lon, dtype=np.float64) - center_lon) * km_per_deg_lon
    cc = (grid_w - 1) / 2.0
    rr = (grid_h - 1) / 2.0
    col = cc + east_km / pixel_km
    row = (rr - north_km / pixel_km) if north_up else (rr + north_km / pixel_km)
    return row, col


def rasterise_points(row: np.ndarray, col: np.ndarray, frame: np.ndarray,
                     n_frames: int, grid_h: int = C.GRID_H,
                     grid_w: int = C.GRID_W,
                     weights: np.ndarray | None = None) -> np.ndarray:
    """Bin scattered point events into a (n_frames, grid_h, grid_w) count grid."""
    r = np.rint(np.asarray(row)).astype(np.int64)
    c = np.rint(np.asarray(col)).astype(np.int64)
    t = np.asarray(frame).astype(np.int64)
    keep = ((r >= 0) & (r < grid_h) & (c >= 0) & (c < grid_w)
            & (t >= 0) & (t < n_frames))
    out = np.zeros((n_frames, grid_h, grid_w), dtype=np.float64)
    w = np.ones(keep.sum()) if weights is None else np.asarray(weights)[keep]
    np.add.at(out, (t[keep], r[keep], c[keep]), w)
    return out.astype(np.float32)


def _empty_event(n_frames: int, grid_h: int = C.GRID_H,
                 grid_w: int = C.GRID_W) -> dict:
    """All-missing event skeleton; loaders fill in what their source provides."""
    x = np.full((n_frames, C.N_CHANNELS, grid_h, grid_w), C.FILL_VALUE, np.float32)
    mask = np.zeros_like(x, dtype=bool)
    y = np.zeros((n_frames, grid_h, grid_w), np.float32)
    return {"x": x, "mask": mask, "y": y, "meta": {}}


# ===========================================================================
# SEVIR
# ===========================================================================
#
# SEVIR is the best public benchmark for this task and it is also the source
# whose limitations must be stated most loudly, because it is the one people
# actually manage to download.
#
# WHAT SEVIR CONTAINS (VERIFY: all of the following is recalled from the SEVIR
# paper and repository documentation, not read off a file):
#   vil    NEXRAD vertically integrated liquid,   384x384 @ 1 km
#   ir069  GOES-16 ABI band 9, 6.9 um water vapour, 192x192 @ 2 km
#   ir107  GOES-16 ABI band 13, 10.7 um IR window, 192x192 @ 2 km
#   vis    GOES-16 ABI band 2, 0.64 um visible,   768x768 @ 0.5 km
#   lght   GOES-16 GLM flashes, as an EVENT LIST, rasterised at 48x48 @ 8 km
# Each event is 49 frames at 5 min = 4 h, centred on the event time.
#
# HOW THE CHANNELS MAP -- AND WHERE THE PROJECT'S PHYSICS IS ONLY APPROXIMATED
#
#   DIRECT (measured):
#     vil        <- SEVIR vil, decoded to kg/m^2
#     ir_tb      <- SEVIR ir107, converted to K
#     light_dens <- SEVIR lght, rasterised
#
#   SURROGATE (estimated, NOT measured):
#     refl_sfc   SEVIR contains NO radar reflectivity at all. It carries VIL,
#                which is a column integral, and `vis`, which is GOES visible
#                imagery and not radar despite occasionally being described as
#                such. Surface dBZ is inverted from VIL.
#     echo_top   Not present. Estimated from IR cloud-top temperature, which
#                conflates the CLOUD top with the 18 dBZ ECHO top. In an anvil
#                these differ by several kilometres over an area far larger
#                than the echo, so the estimate is worst exactly where the
#                distinction between an active updraft and a decaying anvil
#                matters most.
#     refl_m10   THE CENTRAL LIMITATION OF THIS ENTIRE PROJECT ON SEVIR.
#                config.py argues, correctly, that reflectivity at the -10C
#                level is the single most important predictor because it is
#                the direct radar signature of graupel in the non-inductive
#                charging zone. SEVIR does not contain it, cannot contain it
#                (VIL is an integral over the column, so the vertical
#                information has already been destroyed), and no combination
#                of VIL and IR recovers it. What `estimate_refl_m10` produces
#                is a plausible-looking monotone function of VIL and cloud-top
#                temperature. It is not a measurement of the charging layer.
#
#                CONSEQUENCE: a model trained on SEVIR that scores well on
#                refl_m10 has learned the estimator, not the physics. The
#                estimator is built from VIL and IR, so any apparent
#                "charging-layer skill" is skill already available from VIL and
#                IR. Run `mark_surrogates_invalid=True` to train without the
#                surrogate channels and compare: if the score is unchanged, the
#                surrogate was adding nothing but the appearance of physics.
#                Only NEXRAD Level II (or IMD DWR) volumes give a real -10C
#                reflectivity, which is why load_imd_dwr / polar_to_cartesian
#                exist.
#
#   AIRMASS MISMATCH. SEVIR is CONUS. The US warm-season freezing level is
#   around 3.5 km; config.FREEZING_LEVEL_KM is 5.0 km for the Indian
#   warm-season sounding, and CHARGING_LOWER/UPPER_KM follow from it. Applying
#   the Indian isotherm heights to US storms mislabels the charging layer by
#   1.5-3 km. Pass `freezing_level_km` when loading SEVIR.

SEVIR_NATIVE_KM = {           # VERIFY: recalled from the SEVIR paper
    "vil": 1.0,
    "ir069": 2.0,
    "ir107": 2.0,
    "vis": 0.5,
    "lght": 8.0,
}
SEVIR_NATIVE_PIX = {          # VERIFY: recalled from the SEVIR paper
    "vil": 384, "ir069": 192, "ir107": 192, "vis": 768, "lght": 48,
}
SEVIR_FRAMES = 49             # VERIFY: 49 frames, 5 min, -120..+120 min
# VERIFY: SEVIR utils define FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60,
# i.e. seconds relative to the event reference time. The lightning event list's
# time column is measured against the same origin.
SEVIR_FRAME_TIMES_S = np.arange(-120.0, 125.0, 5.0) * 60.0


def _sevir_decode_ir(raw: np.ndarray) -> np.ndarray:
    """SEVIR ir069/ir107 counts -> brightness temperature in kelvin.

    VERIFY: recalled as int16 stored in units of 0.01 degrees CELSIUS, so
    BT[K] = raw * 0.01 + 273.15. Check with a real file: a deep convective top
    must land near 190-210 K and a clear tropical land surface near 290-310 K.
    If your decoded values come out near 250 *degrees* rather than 250 K, the
    scale factor or the offset is wrong, and config.CHANNEL_RANGES["ir_tb"]
    (180-320) will catch it in quality_control.
    """
    return np.asarray(raw, dtype=np.float64) * 0.01 + 273.15


def _sevir_decode_vil(raw: np.ndarray) -> np.ndarray:
    """SEVIR VIL uint8 counts -> kg/m^2.

    VERIFY: recalled piecewise encoding from the SEVIR repository:
        raw <= 5            -> 0
        5 < raw <= 18       -> (raw - 2) / 90.66
        raw > 18            -> exp((raw - 83.9) / 38.9)
    Sanity check on a real file: raw 255 should give roughly 80 kg/m^2 and the
    field maximum over a severe event should be tens, not thousands. If the
    decoded maximum is around 255, the decoding was not applied.
    """
    r = np.asarray(raw, dtype=np.float64)
    out = np.zeros_like(r)
    m2 = (r > 5) & (r <= 18)
    m3 = r > 18
    out[m2] = (r[m2] - 2.0) / 90.66
    out[m3] = np.exp((r[m3] - 83.9) / 38.9)
    return out


def estimate_refl_sfc_from_vil(vil: np.ndarray,
                               depth_km: float = 5.0) -> np.ndarray:
    """SURROGATE: invert VIL to an equivalent surface reflectivity in dBZ.

    VIL is defined as the column integral of liquid water content estimated
    from reflectivity, conventionally

        VIL = 3.44e-6 * integral( Z^(4/7) ) dz          [kg/m^2, z in metres]

    Assuming a uniform Z over a column of depth `depth_km` inverts to

        Z = (VIL / (3.44e-6 * depth_m)) ^ (7/4)

    WHY THIS IS A SURROGATE AND NOT A MEASUREMENT. The assumption of a uniform
    profile is false for every interesting storm: a convective core has most of
    its Z concentrated in a few kilometres, so the true peak reflectivity is
    higher than this returns, while stratiform rain is closer to uniform and
    comes out about right. The inversion is therefore biased in a
    storm-type-dependent way -- it systematically underestimates exactly the
    deep cores that matter. It also cannot distinguish a tall thin core from a
    short fat one, which is the distinction the whole project rests on.
    """
    depth_m = max(depth_km, 0.1) * 1000.0
    v = np.clip(np.asarray(vil, dtype=np.float64), 0.0, None)
    z = np.power(np.maximum(v, 1e-6) / (3.44e-6 * depth_m), 7.0 / 4.0)
    dbz = 10.0 * np.log10(np.maximum(z, 1e-3))
    return np.clip(dbz, C.CHANNEL_RANGES["refl_sfc"][0],
                   C.CHANNEL_RANGES["refl_sfc"][1])


def estimate_echo_top_from_ir(ir_tb_k: np.ndarray, vil: np.ndarray,
                              surface_temp_k: float = 300.0,
                              lapse_k_per_km: float = 6.5,
                              anvil_offset_km: float = 1.5,
                              vil_gate: float = 0.5) -> np.ndarray:
    """SURROGATE: cloud-top height from IR, gated by VIL, called an echo top.

    Height from a constant lapse rate: h = (T_sfc - Tb) / lapse. Then subtract
    `anvil_offset_km` because the 18 dBZ echo top is systematically below the
    radiative cloud top -- the uppermost cloud is small ice crystals that are
    radiatively opaque and radar-invisible.

    THE FAILURE THIS CANNOT AVOID. A mature anvil is cold over an area many
    times the size of the precipitating core, so IR alone reports a high
    "echo top" across the entire anvil. Gating on VIL removes the worst of it
    (no condensate, no echo) but not the part that matters: the decaying
    stratiform region behind an MCS keeps both a cold top and moderate VIL
    while its updraft, and its lightning, are gone. That is a mimic population
    this surrogate cannot separate, and echo_top is supposed to be one of the
    channels that separates mimics.

    Also note the lapse rate is a single constant standing in for a sounding.
    Real tropopause-penetrating tops appear WARMER than their surroundings,
    so this inverts the height of the most intense storms of all.
    """
    tb = np.asarray(ir_tb_k, dtype=np.float64)
    h = (surface_temp_k - tb) / max(lapse_k_per_km, 0.1) - anvil_offset_km
    h = np.where(np.asarray(vil, dtype=np.float64) >= vil_gate, h, 0.0)
    return np.clip(h, C.CHANNEL_RANGES["echo_top"][0],
                   C.CHANNEL_RANGES["echo_top"][1])


def estimate_refl_m10(vil: np.ndarray, ir_tb_k: np.ndarray,
                      echo_top_km: np.ndarray,
                      freezing_level_km: float | None = None,
                      charging_lower_km: float | None = None) -> np.ndarray:
    """SURROGATE for the charging-layer reflectivity. Read the block comment above.

    Construction: take the VIL-inverted surface reflectivity and reduce it by
    an amount that depends on how much of the column lies above the charging
    level. If the estimated echo top barely exceeds the charging level, almost
    none of the condensate is up there and the value collapses; if the top is
    far above it, a larger fraction is. IR adds a weak second constraint --
    colder tops mean deeper glaciation.

    WHAT IT CANNOT DO, and this is the whole problem: a warm-rain-dominated
    tropical cell and an electrified cell can have the SAME VIL and the SAME
    cloud-top temperature and differ entirely in whether their condensate
    arrives in the charging layer as graupel or falls out below the freezing
    level as rain. synth.py models exactly that overlap deliberately, because
    it is the real discrimination task. This estimator, being a monotone
    function of VIL and IR, assigns them the same value. It therefore cannot
    represent the physics the project claims to use; on SEVIR the project's
    central claim is untested.
    """
    fl = C.FREEZING_LEVEL_KM if freezing_level_km is None else float(freezing_level_km)
    cl = C.CHARGING_LOWER_KM if charging_lower_km is None else float(charging_lower_km)
    # If the isotherm heights were overridden for a different airmass, keep the
    # charging level the same distance above the freezing level as in config.
    if freezing_level_km is not None and charging_lower_km is None:
        cl = fl + (C.CHARGING_LOWER_KM - C.FREEZING_LEVEL_KM)

    sfc = estimate_refl_sfc_from_vil(vil)
    top = np.asarray(echo_top_km, dtype=np.float64)
    # Fraction of the echo column lying above the charging level, 0..1 over a
    # 4 km ramp. Arbitrary shape; not calibrated against anything.
    frac = np.clip((top - cl) / 4.0, 0.0, 1.0)
    # Glaciation proxy: 235 K is the conventional "deep convection" threshold.
    glac = np.clip((250.0 - np.asarray(ir_tb_k, dtype=np.float64)) / 40.0, 0.0, 1.0)
    # dBZ decrement from the surface to the charging layer. 25 dB when nothing
    # reaches the layer, ~5 dB when the column is deep and glaciated. These
    # constants are a guess shaped to give a plausible dynamic range.
    decrement = 25.0 - 20.0 * (0.65 * frac + 0.35 * glac)
    out = sfc - decrement
    out = np.where(top <= cl, C.CHANNEL_RANGES["refl_m10"][0], out)
    return np.clip(out, C.CHANNEL_RANGES["refl_m10"][0],
                   C.CHANNEL_RANGES["refl_m10"][1])


def read_sevir_catalog(catalog_path: str) -> list[dict[str, str]]:
    """Read CATALOG.csv into a list of row dicts using the stdlib csv module.

    pandas is deliberately not required: the catalog is a few hundred thousand
    rows of plain CSV and the stdlib handles the quoted `proj` field correctly.

    VERIFY: columns recalled as
        id, file_name, file_index, img_type, time_utc, minute_offsets,
        episode_id, event_id, event_type, llcrnrlat, llcrnrlon, urcrnrlat,
        urcrnrlon, proj, size_x, size_y, height_m, width_m, data_min,
        data_max, pct_missing
    Check the header line of your CATALOG.csv before relying on any of these.
    """
    with open(catalog_path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{catalog_path} is empty")
    for need in ("id", "img_type", "file_name", "file_index"):
        if need not in rows[0]:
            raise ValueError(
                f"CATALOG.csv is missing the {need!r} column. Found: "
                f"{list(rows[0])}. The schema assumed by data_io.py is "
                f"UNVERIFIED -- adjust read_sevir_catalog to match the real file.")
    return rows


def _sevir_read_raster(h5py_mod, path: str, img_type: str,
                       file_index: int) -> np.ndarray:
    """Read one event's frames for a raster image type.

    VERIFY: recalled layout is one dataset per file NAMED AFTER THE IMAGE TYPE
    with shape (n_events, H, W, 49) -- time LAST. If your file has time first,
    every frame in this project will be a spatial slice of a single timestep
    and nothing downstream will complain. Check with:
        h5ls -r file.h5      # or  python -c "import h5py; print(h5py.File(p).items())"
    and assert the last axis is 49.
    """
    with h5py_mod.File(path, "r") as hf:
        if img_type not in hf:
            raise KeyError(
                f"dataset {img_type!r} not in {path}. Datasets present: "
                f"{list(hf.keys())}. VERIFY the SEVIR HDF5 layout.")
        arr = np.asarray(hf[img_type][int(file_index)])
    if arr.ndim != 3:
        raise ValueError(f"{path}[{img_type}][{file_index}] has shape {arr.shape}, "
                         f"expected (H, W, n_frames)")
    if arr.shape[-1] != SEVIR_FRAMES:
        raise ValueError(
            f"{path}[{img_type}][{file_index}] last axis is {arr.shape[-1]}, "
            f"expected {SEVIR_FRAMES} frames. If the FIRST axis is "
            f"{SEVIR_FRAMES} the array is time-first and this loader's "
            f"assumed axis order is wrong -- fix it here, not downstream.")
    return np.moveaxis(arr, -1, 0)     # -> (n_frames, H, W)


def _sevir_read_lght(h5py_mod, path: str, event_id: str) -> np.ndarray:
    """Read one event's GLM flash list.

    VERIFY: recalled as one dataset PER EVENT ID (not indexed by file_index
    like the rasters), shape (n_flashes, 5), columns
        [time_offset_seconds, lat, lon, x, y]
    where x, y are already pixel indices into the 48x48 8 km lightning grid and
    time_offset is measured against the same origin as SEVIR_FRAME_TIMES_S.
    Check the column meaning on a real file: lat should be in [20, 50] and lon
    in [-130, -60] for CONUS, and x, y should be integers in [0, 48).
    """
    with h5py_mod.File(path, "r") as hf:
        if event_id not in hf:
            return np.zeros((0, 5), dtype=np.float64)
        return np.asarray(hf[event_id][:], dtype=np.float64)


def sevir_lght_to_grid(flashes: np.ndarray,
                       native_pix: int = 48,
                       n_frames: int = SEVIR_FRAMES,
                       frame_times_s: np.ndarray | None = None,
                       corner_latlon: tuple[float, float, float, float] | None = None
                       ) -> np.ndarray:
    """Accumulate a SEVIR flash event list into (n_frames, native_pix, native_pix).

    Uses the x, y pixel columns when they look like pixel indices, and falls
    back to lat/lon binning against the catalog corner coordinates otherwise.
    That fallback is not cosmetic: if the columns turn out to be projection
    metres rather than pixels (VERIFY), the pixel path would silently pile
    every flash into the corner of the grid and the target would be garbage
    while remaining a perfectly plausible-looking sparse array.
    """
    ft = SEVIR_FRAME_TIMES_S if frame_times_s is None else np.asarray(frame_times_s)
    if flashes.size == 0:
        return np.zeros((n_frames, native_pix, native_pix), dtype=np.float32)

    t = flashes[:, 0]
    # np.digitize with the frame start times: frame k covers [ft[k], ft[k+1]).
    k = np.digitize(t, ft) - 1

    x = flashes[:, 3]
    y = flashes[:, 4]
    looks_like_pixels = (np.nanmax(np.abs(x)) < native_pix * 2
                         and np.nanmax(np.abs(y)) < native_pix * 2)
    if looks_like_pixels:
        col, row = x, y
    elif corner_latlon is not None:
        lat, lon = flashes[:, 1], flashes[:, 2]
        llat, llon, ulat, ulon = corner_latlon
        row = (ulat - lat) / max(ulat - llat, 1e-9) * native_pix
        col = (lon - llon) / max(ulon - llon, 1e-9) * native_pix
    else:
        raise ValueError(
            "SEVIR lght x/y columns do not look like pixel indices "
            f"(max |x| = {np.nanmax(np.abs(x)):.1f}) and no corner lat/lons "
            "were supplied. VERIFY the column meaning before proceeding; do "
            "not guess, the target array would look fine either way.")
    return rasterise_points(row, col, k, n_frames, native_pix, native_pix)


def load_sevir(catalog_path: str,
               data_dir: str,
               event_ids: Sequence[str] | None = None,
               max_events: int | None = None,
               event_types: Sequence[str] | None = None,
               *,
               grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
               pixel_km: float = C.PIXEL_KM,
               freezing_level_km: float | None = None,
               mark_surrogates_invalid: bool = False,
               run_qc: bool = True,
               verbose: bool = True) -> list[dict]:
    """Load SEVIR events into the project's event interface.

    Args:
        catalog_path: path to CATALOG.csv.
        data_dir: root under which `file_name` paths resolve
            (e.g. .../sevir/data, containing vil/, ir107/, lght/ ...).
        event_ids: restrict to these SEVIR ids; None means all.
        event_types: restrict by the catalog's `event_type` column, e.g.
            ("Thunderstorm Wind", "Hail"). None means all.
        freezing_level_km: override config's 5.0 km Indian value. Pass ~3.5
            for CONUS; see the airmass-mismatch note in the block comment.
        mark_surrogates_invalid: set mask=False on refl_sfc / refl_m10 /
            echo_top instead of filling them with estimates. USE THIS AS AN
            ABLATION. If the model's scores barely move, the surrogates were
            contributing nothing but the appearance of physics.

    Returns a list of event dicts, each validated against the contract.
    """
    h5py = _require("h5py", "read SEVIR HDF5 files")

    rows = read_sevir_catalog(catalog_path)
    by_event: dict[str, dict[str, dict[str, str]]] = {}
    for r in rows:
        if event_types and r.get("event_type") not in event_types:
            continue
        by_event.setdefault(r["id"], {})[r["img_type"]] = r

    ids = list(by_event) if event_ids is None else [i for i in event_ids if i in by_event]
    # Require the three directly measured types. An event without vil has no
    # radar information at all and every radar channel would be a surrogate of
    # a surrogate.
    ids = [i for i in ids if "vil" in by_event[i]]
    if max_events is not None:
        ids = ids[:int(max_events)]
    if not ids:
        raise ValueError(
            f"no SEVIR events selected from {catalog_path} "
            f"(after event_ids/event_types filtering and requiring img_type='vil')")

    fl = C.FREEZING_LEVEL_KM if freezing_level_km is None else float(freezing_level_km)
    events: list[dict] = []
    for n, eid in enumerate(ids):
        grp = by_event[eid]
        ev = _empty_event(SEVIR_FRAMES, grid_h, grid_w)

        # ---- VIL (measured) ----
        r = grp["vil"]
        raw = _sevir_read_raster(h5py, os.path.join(data_dir, r["file_name"]),
                                 "vil", int(r["file_index"]))
        vil_native = _sevir_decode_vil(raw)
        vil, vil_ok = to_config_grid(vil_native, SEVIR_NATIVE_KM["vil"],
                                     grid_h=grid_h, grid_w=grid_w, pixel_km=pixel_km)
        ev["x"][:, C.CH["vil"]] = vil
        ev["mask"][:, C.CH["vil"]] = vil_ok

        # ---- IR 10.7 um (measured) ----
        if "ir107" in grp:
            r = grp["ir107"]
            raw = _sevir_read_raster(h5py, os.path.join(data_dir, r["file_name"]),
                                     "ir107", int(r["file_index"]))
            ir_native = _sevir_decode_ir(raw)
            ir, ir_ok = to_config_grid(ir_native, SEVIR_NATIVE_KM["ir107"],
                                       grid_h=grid_h, grid_w=grid_w, pixel_km=pixel_km)
            ev["x"][:, C.CH["ir_tb"]] = ir
            ev["mask"][:, C.CH["ir_tb"]] = ir_ok
        else:
            ir = np.full((SEVIR_FRAMES, grid_h, grid_w), 290.0, np.float32)
            ir_ok = np.zeros_like(ir, dtype=bool)

        # ---- Lightning (measured, but at 8 km) ----
        #
        # NOTE ON RESOLUTION. The lightning grid is 8 km native and we upsample
        # it 4x to reach the 2 km config grid. The target therefore does NOT
        # resolve 2 km structure: a single 8 km GLM cell becomes a solid 4x4
        # block of "lightning" pixels. Every metric evaluated at FSS radius 0
        # is measuring agreement at a scale the observation does not have, and
        # CSI at pixel level is correspondingly optimistic about localisation.
        # Report FSS at radius >= 2 px on SEVIR and say why.
        if "lght" in grp:
            r = grp["lght"]
            fl_arr = _sevir_read_lght(h5py, os.path.join(data_dir, r["file_name"]), eid)
            corners = None
            try:
                corners = (float(r["llcrnrlat"]), float(r["llcrnrlon"]),
                           float(r["urcrnrlat"]), float(r["urcrnrlon"]))
            except (KeyError, ValueError, TypeError):
                pass
            lg_native = sevir_lght_to_grid(
                fl_arr, SEVIR_NATIVE_PIX["lght"], SEVIR_FRAMES,
                corner_latlon=corners)
            # Counts must be conserved through the regrid, not averaged: use
            # area weights on the density then rescale by the area ratio.
            area_ratio = (SEVIR_NATIVE_KM["lght"] / pixel_km) ** 2
            lg, lg_ok = to_config_grid(lg_native, SEVIR_NATIVE_KM["lght"],
                                       valid=np.ones_like(lg_native, bool),
                                       grid_h=grid_h, grid_w=grid_w,
                                       pixel_km=pixel_km)
            lg = np.clip(lg / max(area_ratio, 1e-9), 0.0, None)
            ev["x"][:, C.CH["light_dens"]] = lg
            ev["mask"][:, C.CH["light_dens"]] = lg_ok
            ev["y"] = lg.astype(np.float32)
        else:
            ev["mask"][:, C.CH["light_dens"]] = False

        # ---- SURROGATES ----
        surrogates: list[str] = ["refl_sfc", "refl_m10", "echo_top"]
        if mark_surrogates_invalid:
            for ch in surrogates:
                ev["mask"][:, C.CH[ch]] = False
        else:
            sfc = estimate_refl_sfc_from_vil(vil)
            top = estimate_echo_top_from_ir(ir, vil)
            m10 = estimate_refl_m10(vil, ir, top, freezing_level_km=fl)
            base_ok = vil_ok & (ir_ok if "ir107" in grp else vil_ok)
            for ch, arr in (("refl_sfc", sfc), ("echo_top", top), ("refl_m10", m10)):
                ev["x"][:, C.CH[ch]] = arr.astype(np.float32)
                ev["mask"][:, C.CH[ch]] = base_ok
        ev["x"][~ev["mask"]] = C.FILL_VALUE

        ev["meta"] = {
            "source": "sevir",
            "event_id": eid,
            "event_type": grp["vil"].get("event_type"),
            "time_utc": grp["vil"].get("time_utc"),
            "img_types": sorted(grp),
            "corner_latlon": [grp["vil"].get(k) for k in
                              ("llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon")],
            "native_km": {k: SEVIR_NATIVE_KM[k] for k in grp if k in SEVIR_NATIVE_KM},
            "freezing_level_km": fl,
            "surrogate_channels": [] if mark_surrogates_invalid else surrogates,
            "surrogate_note": (
                "SEVIR has no reflectivity, no echo top and no -10C level. "
                "refl_sfc/echo_top/refl_m10 are ESTIMATED from VIL and IR and "
                "carry no independent vertical information. The project's "
                "central physics claim is NOT tested by a SEVIR run."),
            "target_native_km": SEVIR_NATIVE_KM["lght"],
            "target_note": ("lightning is 8 km native, upsampled 4x; do not "
                            "trust pixel-scale localisation metrics"),
            "schema_verified": False,
        }
        if run_qc:
            ev["x"], ev["mask"], qc = quality_control(ev["x"], ev["mask"])
            ev["meta"]["qc"] = {k: v for k, v in qc.items() if k != "flags"}
        validate_event(ev, name=f"sevir:{eid}")
        events.append(ev)
        if verbose and (n % 25 == 0 or n == len(ids) - 1):
            print(f"  sevir: {n + 1}/{len(ids)} events", flush=True)
    return events


# ===========================================================================
# GOES GLM
# ===========================================================================

def load_goes_glm(files: Sequence[str],
                  center_lat: float,
                  center_lon: float,
                  t_start: datetime | None = None,
                  n_frames: int | None = None,
                  *,
                  grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
                  pixel_km: float = C.PIXEL_KM,
                  timestep_min: float = C.TIMESTEP_MIN,
                  quality_flag_max: int = 0,
                  verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Accumulate GOES-16/18 GLM L2 LCFA flashes onto the config grid.

    GLM files are 20 s granules, so a 5 min frame is fifteen files. Returns
    (counts (n_frames, H, W) float32, meta).

    WHAT GLM ACTUALLY MEASURES, AND WHY IT IS NOT INTERCHANGEABLE WITH A
    GROUND NETWORK
    ---------------------------------------------------------------------
    GLM is an optical transient detector in geostationary orbit. It sees the
    777.4 nm oxygen emission scattered out of CLOUD TOP by any lightning
    discharge, so it measures TOTAL lightning and is dominated by INTRACLOUD
    flashes, which outnumber cloud-to-ground roughly 3:1 and more in vigorous
    convection. A ground-based VLF/LF network -- IITM's Damini LLN, WWLLN,
    ENTLN -- detects the sferic from the return stroke and is strongly biased
    toward CLOUD-TO-GROUND, with intracloud detection efficiency that is far
    lower and far more variable.

    THE CONSEQUENCE FOR THIS PROJECT. `y` means "lightning occurred in this
    pixel in this 5 min window", and the two sources answer different
    questions:

      * IC leads CG. Intracloud activity typically begins several minutes
        before the first cloud-to-ground stroke in a developing cell. A model
        trained on GLM is learning to predict an EARLIER event than a model
        trained on Damini, so it will appear to have longer effective lead
        time on the same storms while actually forecasting a different thing.
      * Base rates differ by a factor of several, so config.HP_*["pos_weight"]
        and every probability threshold in config.PROB_THRESHOLDS are tuned to
        whichever source produced the training targets.
      * The operational question -- "will lightning strike the ground here" --
        is the CG question. Damini is the right target for an Indian warning
        product; GLM is the right target for a total-lightning nowcast.

    Swapping the target source between training and evaluation changes what is
    being predicted while every metric still computes cleanly. Record the
    source in meta and check it before comparing two runs.

    COVERAGE, WHICH RULES GLM OUT FOR INDIA ENTIRELY. GOES-16/19 sits near
    75.2W and GOES-18 near 137.2W. India (roughly 68E-97E) is on the far side
    of the earth and is NOT in either field of view, at any quality. There is
    no geostationary optical lightning mapper covering India in operation
    (VERIFY: check the current status of any ISRO or EUMETSAT MTG-LI coverage
    before concluding -- MTG-LI at 0 deg longitude may reach western India at
    high view angle). For the Indian domain the target must come from a ground
    network. This function exists for the SEVIR/CONUS pretraining path.
    """
    nc4 = _require("netCDF4", "read GOES GLM L2 LCFA netCDF files")

    n_frames = C.INPUT_FRAMES + C.OUTPUT_FRAMES if n_frames is None else int(n_frames)
    step_s = float(timestep_min) * 60.0

    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    secs: list[np.ndarray] = []
    n_files_ok = 0
    n_flash_total = 0
    n_rejected_qc = 0
    t_min_seen: float | None = None

    for path in files:
        try:
            ds = nc4.Dataset(path, "r")
        except Exception as exc:
            if verbose:
                print(f"  glm: cannot open {path}: {exc}")
            continue
        try:
            # VERIFY: variable names recalled from the GLM L2 LCFA product
            # specification: flash_lat, flash_lon (float32 degrees),
            # flash_time_offset_of_first_event and
            # flash_time_offset_of_last_event (stored as int16 with
            # scale_factor/add_offset; netCDF4 applies them automatically and
            # yields seconds since the J2000 epoch 2000-01-01 12:00:00 UTC --
            # check the variable's `units` attribute rather than assuming),
            # flash_quality_flag, flash_energy, flash_area. Also present:
            # group_* and event_* hierarchies, and product_time.
            if "flash_lat" not in ds.variables:
                raise KeyError(
                    f"'flash_lat' not in {path}. Variables: "
                    f"{sorted(ds.variables)}. VERIFY the GLM schema.")
            fl_lat = np.asarray(ds.variables["flash_lat"][:], dtype=np.float64)
            fl_lon = np.asarray(ds.variables["flash_lon"][:], dtype=np.float64)
            tvar = ("flash_time_offset_of_first_event"
                    if "flash_time_offset_of_first_event" in ds.variables
                    else "flash_time_offset_of_last_event")
            fl_t = np.asarray(ds.variables[tvar][:], dtype=np.float64)

            if "flash_quality_flag" in ds.variables:
                qf = np.asarray(ds.variables["flash_quality_flag"][:], dtype=np.int64)
                good = qf <= int(quality_flag_max)
                n_rejected_qc += int((~good).sum())
                fl_lat, fl_lon, fl_t = fl_lat[good], fl_lon[good], fl_t[good]

            lats.append(fl_lat)
            lons.append(fl_lon)
            secs.append(fl_t)
            n_flash_total += fl_lat.size
            n_files_ok += 1
            if fl_t.size:
                m = float(np.nanmin(fl_t))
                t_min_seen = m if t_min_seen is None else min(t_min_seen, m)
        finally:
            ds.close()

    if not n_files_ok:
        raise ValueError(f"no readable GLM files among {len(files)} paths")

    lat = np.concatenate(lats) if lats else np.zeros(0)
    lon = np.concatenate(lons) if lons else np.zeros(0)
    sec = np.concatenate(secs) if secs else np.zeros(0)

    # Time origin. If the caller gave an absolute t_start we need the epoch the
    # file's seconds are counted from; GLM uses J2000 (2000-01-01 12:00 UTC).
    # VERIFY: confirm against the variable's `units` attribute -- silently
    # assuming the wrong epoch shifts every flash by decades and produces an
    # empty grid, which at least fails loudly, but assuming the wrong OFFSET
    # shifts by minutes and produces a plausible, wrong answer.
    J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    if t_start is None:
        t0_s = 0.0 if t_min_seen is None else t_min_seen
        t_start_dt = J2000 + timedelta(seconds=float(t0_s))
    else:
        t_start_dt = t_start if t_start.tzinfo else t_start.replace(tzinfo=timezone.utc)
        t0_s = (t_start_dt - J2000).total_seconds()

    frame = np.floor((sec - t0_s) / step_s).astype(np.int64)
    row, col = latlon_to_rowcol(lat, lon, center_lat, center_lon,
                                grid_h, grid_w, pixel_km)
    counts = rasterise_points(row, col, frame, n_frames, grid_h, grid_w)

    in_domain = int(((row >= 0) & (row < grid_h)
                     & (col >= 0) & (col < grid_w)).sum())
    meta = {
        "source": "goes_glm",
        "target_kind": "total_lightning_optical",
        "dominant_flash_type": "intracloud",
        "not_interchangeable_with": "ground VLF/LF CG networks (Damini, WWLLN, ENTLN)",
        "n_files": len(files), "n_files_ok": n_files_ok,
        "n_flashes_read": int(n_flash_total),
        "n_flashes_rejected_quality": int(n_rejected_qc),
        "n_flashes_in_domain": in_domain,
        "t_start_utc": t_start_dt.isoformat(),
        "timestep_min": float(timestep_min),
        "center_latlon": [float(center_lat), float(center_lon)],
        "india_coverage": False,
        "coverage_note": ("GOES-16/19 at 75.2W and GOES-18 at 137.2W do not "
                          "view India. Use a ground network for the Indian domain."),
        "schema_verified": False,
    }
    return counts, meta


# ===========================================================================
# MOSDAC INSAT-3D / 3DR
# ===========================================================================

INSAT_BAND_DATASETS = {   # VERIFY: recalled MOSDAC L1B HDF5 dataset names
    "TIR1": "IMG_TIR1",   # ~10.8 um thermal IR window -> the ir_tb channel
    "TIR2": "IMG_TIR2",   # ~12.0 um
    "MIR": "IMG_MIR",     # ~3.9 um
    "WV": "IMG_WV",       # ~6.8 um water vapour
    "VIS": "IMG_VIS",
    "SWIR": "IMG_SWIR",
}


def mosdac_token(required: bool = True) -> str:
    """Read the MOSDAC credential from the environment. Never hardcode a token.

    A token in source is a token in git history, in every clone, and in every
    log that echoes a traceback. Register at mosdac.gov.in and export:
        export MOSDAC_TOKEN=...
    """
    tok = os.environ.get("MOSDAC_TOKEN", "")
    if not tok and required:
        raise RuntimeError(
            "MOSDAC_TOKEN is not set. Register at https://mosdac.gov.in, "
            "obtain an API token, and export it:\n"
            "    export MOSDAC_TOKEN=<your token>\n"
            "Do not paste it into the source.")
    return tok


def mosdac_download(url: str, out_path: str, *, timeout: int = 120,
                    chunk: int = 1 << 20) -> str:
    """Fetch one MOSDAC file using the env token. Requires `requests`."""
    requests = _require("requests", "download from MOSDAC")
    tok = mosdac_token()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    # VERIFY: MOSDAC's current auth scheme. A bearer header is assumed here;
    # the portal has historically also used a session cookie from a form login
    # and a `token=` query parameter. Check what your account actually issues.
    headers = {"Authorization": f"Bearer {tok}"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for blk in r.iter_content(chunk_size=chunk):
                if blk:
                    fh.write(blk)
    return out_path


def _insat_time(h5file) -> datetime | None:
    """Acquisition start time from the file attributes.

    VERIFY: attribute recalled as `Acquisition_Start_Time`, formatted like
    '25MAY2023120000' or an ISO string depending on product version. Several
    formats are attempted; None is returned rather than a guess if none parse,
    because a wrong timestamp silently mis-associates satellite frames with
    radar frames and manufactures a fake lead time.
    """
    for key in ("Acquisition_Start_Time", "Acquisition_Time", "time_coverage_start"):
        v = h5file.attrs.get(key)
        if v is None:
            continue
        if isinstance(v, bytes):
            v = v.decode("utf-8", "ignore")
        v = str(v).strip().strip("[]'\" ")
        for fmt in ("%d%b%Y%H%M%S", "%d-%b-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
            try:
                return datetime.strptime(v.upper(), fmt.upper()).replace(
                    tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _insat_bt(h5file, band: str) -> np.ndarray:
    """Counts -> brightness temperature via the file's own lookup table.

    VERIFY: MOSDAC L1B stores the imagery as raw counts (uint16) in IMG_<BAND>
    and ships a companion LUT dataset IMG_<BAND>_TEMP that maps count -> K
    (and IMG_<BAND>_RADIANCE for radiance). Applying a linear scale/offset
    instead of the LUT is a common and quiet error: the sensor calibration is
    not linear in counts, and a linear approximation is right in the middle of
    the range and wrong at the cold end -- i.e. wrong precisely on convective
    cloud tops, which is the only part we care about.
    """
    ds_name = INSAT_BAND_DATASETS.get(band, f"IMG_{band}")
    if ds_name not in h5file:
        raise KeyError(f"{ds_name!r} not in file. Datasets: {list(h5file.keys())}. "
                       f"VERIFY the MOSDAC product layout.")
    counts = np.asarray(h5file[ds_name][:]).squeeze()
    lut_name = f"{ds_name}_TEMP"
    if lut_name in h5file:
        lut = np.asarray(h5file[lut_name][:]).astype(np.float64).ravel()
        idx = np.clip(counts.astype(np.int64), 0, lut.size - 1)
        return lut[idx]
    raise KeyError(
        f"{lut_name!r} (count->temperature LUT) not found; datasets present: "
        f"{list(h5file.keys())}. Refusing to guess a linear calibration -- it "
        f"would be wrong exactly on cold convective tops. VERIFY the product.")


def hold_last_valid(stack: np.ndarray, src_times: Sequence[datetime],
                    target_times: Sequence[datetime],
                    max_staleness_min: float = 20.0,
                    method: str = "hold"
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put a coarse-cadence field onto the radar time base, tracking staleness.

    Args:
        stack: (n_src, H, W).
        method: "hold" repeats the most recent earlier frame; "linear"
            interpolates between the bracketing frames.

    Returns (out (n_target, H, W), valid (n_target, H, W) bool,
             age_min (n_target,)).

    WHY STALENESS MUST BE A NUMBER AND NOT AN IMPLEMENTATION DETAIL
    ---------------------------------------------------------------
    INSAT-3D full-disk is nominally 30 min; the rapid-scan sector is 15 min;
    3D and 3DR are staggered so an Indian sector composite can approach 15 min
    (VERIFY: current operational scan schedule). The radar grid is 5 min. So
    an IR frame is on average 7-15 min old and can be 30 min old, and the
    system will happily hand the model a held value with no indication.

    That is not a small error during convective initiation. A cell that goes
    from cumulus to a 12 km top does so in 20-30 min. A 25 min old cloud-top
    temperature over that pixel says "warm, shallow, nothing here" at the exact
    moment the cell is exploding -- the held value is not merely imprecise, it
    is actively anti-correlated with the truth for the duration of the growth
    phase. And growth phase is when a lightning nowcast has any value at all;
    a mature storm needs no model.

    So: `valid` goes False past `max_staleness_min` (the model then sees
    FILL_VALUE and knows the channel is absent, which is honest), and `age_min`
    is surfaced in meta so metrics can be stratified by IR age. If skill is
    materially lower at age > 15 min, that is a measurement of how much the
    satellite cadence is costing you, and it argues for a source with better
    cadence rather than for a bigger model.

    "linear" is offered but should be used with care: interpolating brightness
    temperature across a 30 min gap invents a smooth cooling that did not
    happen, and unlike a held value it cannot be recognised as stale.
    """
    stack = np.asarray(stack, dtype=np.float64)
    n_src = stack.shape[0]
    if n_src != len(src_times):
        raise ValueError(f"stack has {n_src} frames but {len(src_times)} times")
    st = np.array([t.timestamp() for t in src_times], dtype=np.float64)
    tt = np.array([t.timestamp() for t in target_times], dtype=np.float64)
    order = np.argsort(st)
    st, stack = st[order], stack[order]

    H, W = stack.shape[1:]
    out = np.full((tt.size, H, W), C.FILL_VALUE, dtype=np.float32)
    valid = np.zeros((tt.size, H, W), dtype=bool)
    age = np.full(tt.size, np.inf, dtype=np.float64)
    max_s = float(max_staleness_min) * 60.0

    for k, t in enumerate(tt):
        i = int(np.searchsorted(st, t, side="right")) - 1
        if i < 0:
            continue                              # no earlier frame exists
        dt = t - st[i]
        age[k] = dt / 60.0
        if dt > max_s:
            continue                              # too stale to be usable
        if method == "linear" and i + 1 < n_src:
            span = st[i + 1] - st[i]
            w = 0.0 if span <= 0 else (t - st[i]) / span
            out[k] = ((1.0 - w) * stack[i] + w * stack[i + 1]).astype(np.float32)
        else:
            out[k] = stack[i].astype(np.float32)
        valid[k] = True
    return out, valid, age


def load_mosdac_insat(files: Sequence[str],
                      target_times: Sequence[datetime],
                      center_lat: float, center_lon: float,
                      *,
                      band: str = "TIR1",
                      grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
                      pixel_km: float = C.PIXEL_KM,
                      max_staleness_min: float = 20.0,
                      method: str = "hold",
                      verbose: bool = False
                      ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read INSAT-3D/3DR IR onto the config grid and time base.

    Returns (ir_tb (n_target, H, W) K, valid (n_target, H, W) bool, meta).
    `meta["ir_age_min"]` is the per-frame staleness in minutes -- carry it
    into the event meta and stratify metrics by it (see hold_last_valid).

    Geolocation. Two product families:
      * L1B full disk carries `Latitude` / `Longitude` datasets (VERIFY:
        recalled as int16 with a 0.01 scale factor). We bounding-box the
        domain in lat/lon and nearest-neighbour the survivors, which is fine
        for a 128 km box and avoids needing a KD-tree.
      * L1C sector products are already on a regular Mercator grid described by
        file attributes (VERIFY: recalled as left_longitude, right_longitude,
        upper_latitude, lower_latitude). That path is cheaper and preferred.
    """
    h5py = _require("h5py", "read MOSDAC INSAT HDF5 products")

    frames: list[np.ndarray] = []
    times: list[datetime] = []
    geo_mode = "unknown"
    for path in files:
        with h5py.File(path, "r") as hf:
            bt = _insat_bt(hf, band)
            t = _insat_time(hf)
            if t is None:
                if verbose:
                    print(f"  insat: no parsable acquisition time in {path}, skipped")
                continue

            if "Latitude" in hf and "Longitude" in hf:
                geo_mode = "latlon_arrays"
                lat = np.asarray(hf["Latitude"][:]).squeeze().astype(np.float64)
                lon = np.asarray(hf["Longitude"][:]).squeeze().astype(np.float64)
                # VERIFY: scale factor. If the arrays look like packed ints
                # (|values| >> 180) apply 0.01.
                if np.nanmax(np.abs(lat)) > 180.0:
                    lat, lon = lat * 0.01, lon * 0.01
                tile = _nearest_regrid(bt, lat, lon, center_lat, center_lon,
                                       grid_h, grid_w, pixel_km)
            else:
                geo_mode = "mercator_attrs"
                tile = _mercator_regrid(hf, bt, center_lat, center_lon,
                                        grid_h, grid_w, pixel_km)
            frames.append(tile)
            times.append(t)

    if not frames:
        raise ValueError(f"no usable INSAT frames among {len(files)} files")

    stack = np.stack(frames, axis=0)
    ir, valid, age = hold_last_valid(stack, times, target_times,
                                     max_staleness_min, method)
    meta = {
        "source": "mosdac_insat",
        "band": band,
        "n_files": len(files), "n_frames_read": len(frames),
        "geolocation_mode": geo_mode,
        "src_times_utc": [t.isoformat() for t in times],
        "ir_age_min": age.tolist(),
        "ir_age_min_mean": float(np.mean(age[np.isfinite(age)]))
                            if np.isfinite(age).any() else float("inf"),
        "ir_age_min_max": float(np.max(age[np.isfinite(age)]))
                           if np.isfinite(age).any() else float("inf"),
        "max_staleness_min": float(max_staleness_min),
        "n_target_frames_stale_dropped": int((~valid[:, 0, 0]).sum()),
        "cadence_note": ("INSAT-3D full disk 30 min, rapid-scan sector 15 min, "
                         "against a 5 min radar grid. A held IR frame is "
                         "anti-correlated with truth during rapid growth, "
                         "which is exactly the regime a nowcast is for. "
                         "VERIFY the current operational scan schedule."),
        "schema_verified": False,
    }
    return ir, valid, meta


def _nearest_regrid(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    center_lat: float, center_lon: float,
                    grid_h: int, grid_w: int, pixel_km: float) -> np.ndarray:
    """Nearest-neighbour regrid from irregular lat/lon arrays onto the grid.

    Brute force over a lat/lon bounding box. The box is a couple of hundred
    source pixels across for a 128 km domain even at 1 km resolution, so the
    O(n_src * n_dst) search is trivial; a KD-tree would need scipy and buy
    nothing here.
    """
    half_lat = (grid_h * pixel_km / 2.0) / 111.32 * 1.6
    half_lon = ((grid_w * pixel_km / 2.0)
                / (111.32 * max(math.cos(math.radians(center_lat)), 0.1)) * 1.6)
    sel = ((lat >= center_lat - half_lat) & (lat <= center_lat + half_lat)
           & (lon >= center_lon - half_lon) & (lon <= center_lon + half_lon)
           & np.isfinite(lat) & np.isfinite(lon))
    if not sel.any():
        return np.full((grid_h, grid_w), C.FILL_VALUE, np.float32)

    src_lat, src_lon = lat[sel], lon[sel]
    src_val = np.asarray(field, dtype=np.float64)[sel]
    srow, scol = latlon_to_rowcol(src_lat, src_lon, center_lat, center_lon,
                                  grid_h, grid_w, pixel_km)
    out = np.full((grid_h, grid_w), C.FILL_VALUE, np.float64)
    acc = np.zeros((grid_h, grid_w))
    cnt = np.zeros((grid_h, grid_w))
    r = np.rint(srow).astype(np.int64)
    c = np.rint(scol).astype(np.int64)
    keep = (r >= 0) & (r < grid_h) & (c >= 0) & (c < grid_w)
    np.add.at(acc, (r[keep], c[keep]), src_val[keep])
    np.add.at(cnt, (r[keep], c[keep]), 1.0)
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit]
    return out.astype(np.float32)


def _mercator_regrid(h5file, field: np.ndarray,
                     center_lat: float, center_lon: float,
                     grid_h: int, grid_w: int, pixel_km: float) -> np.ndarray:
    """Regrid an L1C Mercator sector using the file's corner attributes.

    VERIFY: attribute names recalled as left_longitude, right_longitude,
    upper_latitude, lower_latitude. Mercator is treated as locally linear in
    latitude, which over a 128 km box costs well under a pixel.
    """
    def _a(name: str) -> float:
        v = h5file.attrs.get(name)
        if v is None:
            raise KeyError(
                f"attribute {name!r} not found and no Latitude/Longitude "
                f"datasets present. Attributes: {list(h5file.attrs)}. "
                f"VERIFY the MOSDAC product geolocation.")
        return float(np.asarray(v).ravel()[0])

    llon, rlon = _a("left_longitude"), _a("right_longitude")
    ulat, blat = _a("upper_latitude"), _a("lower_latitude")
    nh, nw = field.shape[-2:]

    yy, xx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float64)
    km_lat, km_lon = 111.32, 111.32 * math.cos(math.radians(center_lat))
    dlat = ((grid_h - 1) / 2.0 - yy) * pixel_km / km_lat + center_lat
    dlon = (xx - (grid_w - 1) / 2.0) * pixel_km / km_lon + center_lon

    src_r = (ulat - dlat) / max(ulat - blat, 1e-9) * (nh - 1)
    src_c = (dlon - llon) / max(rlon - llon, 1e-9) * (nw - 1)
    r = np.clip(np.rint(src_r), 0, nh - 1).astype(np.int64)
    c = np.clip(np.rint(src_c), 0, nw - 1).astype(np.int64)
    out = np.asarray(field, dtype=np.float64)[r, c]
    oob = (src_r < 0) | (src_r > nh - 1) | (src_c < 0) | (src_c > nw - 1)
    out[oob] = C.FILL_VALUE
    return out.astype(np.float32)


# ===========================================================================
# IMD Doppler Weather Radar
# ===========================================================================
#
# The practical problem with IMD DWR is that a large share of what is actually
# obtainable is RENDERED IMAGERY, not data: MAX-Z, PPI(Z), VP and PAC products
# published as PNG/GIF with a colour bar, map overlay, station annotation and a
# timestamp burned in. Raw polar volumes (IRIS/Sigmet RAW, or ODIM_H5) exist and
# are obtainable by data request, but the imagery is what is on the web.
#
# Both paths are provided, and they are NOT equivalent:
#   polar_to_cartesian     real volumes -> real refl_m10, real echo top.
#   dequantize_colormap    rendered image -> approximate surface dBZ only.
#
# If you only have imagery, the charging-layer channel that this whole project
# is built around cannot be derived. Say so in the write-up rather than
# deriving it anyway from a MAX-Z picture.

# A placeholder colour scale. THIS IS A GUESS, NOT THE IMD SCALE. Supply the
# real legend -- crop the colour bar out of one product image and read the
# swatches -- before using dequantize_colormap on anything that matters.
DEFAULT_DBZ_LUT: list[tuple[float, tuple[int, int, int]]] = [
    (5.0, (4, 233, 231)), (10.0, (1, 159, 244)), (15.0, (3, 0, 244)),
    (20.0, (2, 253, 2)), (25.0, (1, 197, 1)), (30.0, (0, 142, 0)),
    (35.0, (253, 248, 2)), (40.0, (229, 188, 0)), (45.0, (253, 149, 0)),
    (50.0, (253, 0, 0)), (55.0, (212, 0, 0)), (60.0, (188, 0, 0)),
    (65.0, (248, 0, 253)), (70.0, (152, 84, 198)),
]


def dequantize_colormap(img: np.ndarray,
                        lut: Sequence[tuple[float, tuple[int, int, int]]] | None = None,
                        max_dist: float = 30.0,
                        ignore_colors: Sequence[tuple[int, int, int]] | None = None
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Invert a colour-mapped radar image back to approximate dBZ.

    Args:
        img: (H, W, 3) or (H, W, 4) RGB(A) array.
        lut: [(dbz, (r, g, b)), ...]. Defaults to DEFAULT_DBZ_LUT, WHICH IS A
            GUESS -- see the note on that constant.
        max_dist: Euclidean RGB distance beyond which a pixel is declared
            "not a data colour" and rejected.
        ignore_colors: extra colours to reject outright (background, coastline,
            grid lines).

    Returns (dbz (H, W) float32 with FILL_VALUE where rejected,
             valid (H, W) bool, report).

    THIS IS LOSSY AND THE LOSS IS NOT SMALL. Read this before using the output
    for anything but a picture:

      1. QUANTISATION. The rendering threw away the continuous field and kept
         one colour per class. With a 5 dBZ colour scale, every recovered value
         carries up to +/-2.5 dBZ of quantisation error by construction, and
         that is the BEST case, achieved only if the LUT is exactly right.
         Reflectivity enters rain rate as roughly Z^(1/1.4) and enters this
         project through a 40 dBZ charging-layer threshold; a 2.5 dBZ error
         straddling that threshold flips the classification of a pixel.

      2. CONTAMINATION. Rendered products have coastlines, district
         boundaries, range rings, lat/lon graticules, a north arrow, a colour
         bar and a timestamp drawn ON TOP of the data. Those pixels are not
         data and are not recoverable. Anti-aliasing makes it worse: every
         boundary between two colour classes, and every edge of every overlay
         glyph, produces blended pixels whose nearest LUT colour is arbitrary.
         The `max_dist` rejection catches the obvious ones and leaves the
         subtle ones. Expect a percent or two of the image to be quietly wrong.

      3. RESAMPLING AND COMPRESSION. The image was resampled to display size
         and often saved as JPEG or as a palettised GIF, both of which shift
         colours near edges. Nearest-colour matching then lands in the
         neighbouring class.

      4. IT IS USUALLY THE WRONG PRODUCT ANYWAY. The published image is
         typically MAX-Z (column maximum) or a low-tilt PPI. Neither is
         reflectivity AT THE -10C LEVEL. Column maximum in particular takes the
         peak wherever it occurs in the vertical, so a bright band at 5 km and
         a graupel core at 7 km produce the same picture. DERIVING refl_m10
         FROM A RENDERED IMAGE IS NOT POSSIBLE. Do not do it and then report a
         charging-layer skill score.

    Use this for a qualitative surface-reflectivity channel, for eyeballing a
    case, or for extending a record backwards where nothing else exists -- and
    mark those runs as image-derived.
    """
    lut = DEFAULT_DBZ_LUT if lut is None else list(lut)
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected (H, W, 3+) RGB image, got shape {a.shape}")
    rgb = a[:, :, :3].astype(np.float64)
    H, W = rgb.shape[:2]

    vals = np.array([v for v, _ in lut], dtype=np.float64)
    cols = np.array([c for _, c in lut], dtype=np.float64)

    best_d = np.full((H, W), np.inf)
    best_i = np.zeros((H, W), dtype=np.int64)
    for i in range(cols.shape[0]):      # loop over LUT: O(N) passes, O(H*W) memory
        d = np.sqrt(((rgb - cols[i]) ** 2).sum(axis=2))
        upd = d < best_d
        best_d[upd] = d[upd]
        best_i[upd] = i

    valid = best_d <= float(max_dist)
    if ignore_colors:
        for c in ignore_colors:
            hit = np.all(np.abs(rgb - np.asarray(c, dtype=np.float64)) < 8.0, axis=2)
            valid &= ~hit

    dbz = np.where(valid, vals[best_i], C.FILL_VALUE).astype(np.float32)

    steps = np.diff(np.sort(vals))
    report = {
        "n_pixels": int(H * W),
        "n_valid": int(valid.sum()),
        "frac_rejected": float(1.0 - valid.mean()),
        "lut_size": int(len(lut)),
        "lut_step_dbz": float(np.median(steps)) if steps.size else float("nan"),
        "quantisation_error_dbz": (float(np.median(steps) / 2.0)
                                   if steps.size else float("nan")),
        "median_match_distance": float(np.median(best_d[np.isfinite(best_d)])),
        "p99_match_distance": float(np.percentile(best_d[np.isfinite(best_d)], 99)),
        "lossy": True,
        "usable_for_refl_m10": False,
        "note": ("Quantisation error is +/- half the LUT step BEFORE any "
                 "overlay/anti-alias/compression contamination, and the source "
                 "product is usually MAX-Z, not a level-specific reflectivity."),
    }
    return dbz, valid, report


def load_dbz_lut(path: str) -> list[tuple[float, tuple[int, int, int]]]:
    """Load a colour scale from a CSV of `dbz,r,g,b`. Build it from the real
    colour bar of the product you are actually decoding."""
    out: list[tuple[float, tuple[int, int, int]]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                out.append((float(row[0]),
                            (int(row[1]), int(row[2]), int(row[3]))))
            except (ValueError, IndexError):
                continue
    if not out:
        raise ValueError(f"no usable 'dbz,r,g,b' rows in {path}")
    return out


def polar_to_cartesian(sweeps: Sequence[np.ndarray] | np.ndarray,
                       elevations_deg: Sequence[float],
                       azimuths_deg: Sequence[np.ndarray] | np.ndarray,
                       ranges_km: Sequence[np.ndarray] | np.ndarray,
                       *,
                       grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
                       pixel_km: float = C.PIXEL_KM,
                       radar_row: float | None = None,
                       radar_col: float | None = None,
                       radar_alt_km: float = 0.0,
                       target_alt_km: float | None = None,
                       echo_top_dbz: float = 18.0,
                       method: str = "nearest",
                       nodata: float = C.FILL_VALUE,
                       earth_factor: float = EARTH_FACTOR,
                       beamwidth_deg: float = 1.0,
                       north_up: bool = True) -> dict:
    """Grid a raw polar volume and derive the charging-layer and echo-top fields.

    Pure numpy. pyart does this better and with far more care about dual-pol,
    unfolding and clutter filters; it is unavailable in this sandbox and it is
    a heavy dependency, so this is a deliberately simple, auditable substitute.
    Use pyart if you have it.

    Args:
        sweeps: (n_elev, n_az, n_range) array, or a list of (n_az_i, n_range_i)
            arrays for volumes with per-sweep geometry.
        elevations_deg: one elevation angle per sweep.
        azimuths_deg: shared (n_az,) array, or one array per sweep.
        ranges_km: shared (n_range,) array, or one per sweep.
        radar_row, radar_col: radar position in grid pixels; default centre.
        target_alt_km: altitude for the reflectivity retrieval. Default
            config.CHARGING_LOWER_KM, the -10C level.
        method: "nearest" or "idw" (inverse distance over the four surrounding
            ray/gate samples).

    Returns a dict with refl_target, refl_target_valid, refl_lowest,
    column_max, echo_top_km, echo_top_valid, echo_top_is_lower_bound,
    sweep_grid, sweep_height_km, invalid_reason and meta.

    THE VERTICAL INTERPOLATION IS THE WHOLE POINT AND IT IS ALSO THE WEAKEST
    STEP. A radar samples a discrete stack of cones. To report reflectivity at
    6.5 km at a given horizontal distance we interpolate between the two tilts
    whose beams bracket that altitude. At 60 km range, consecutive tilts of a
    typical VCP are separated by 1-2 km in height, so the interpolation spans a
    gap comparable to the thickness of the charging layer; at 120 km the gap is
    larger than the layer. The retrieval is therefore genuinely uncertain, and
    that uncertainty is physical, not a coding shortcut -- it is why synth.py
    deliberately blurs and adds noise to refl_m10 rather than treating it as an
    oracle.

    `invalid_reason` records WHY each pixel has no value:
        0 ok, 1 overshoot (lowest beam already above the target),
        2 cone of silence (target above the highest beam),
        3 no valid data in the bracketing sweeps.
    Do not collapse these into a single "missing": overshoot and cone of
    silence are systematic geometry, not random dropout, and a model that sees
    them as random dropout will learn a spatial bias fixed to the radar site.
    """
    # --- normalise inputs to per-sweep lists ---
    if isinstance(sweeps, np.ndarray) and sweeps.ndim == 3:
        sw_list = [sweeps[i] for i in range(sweeps.shape[0])]
    else:
        sw_list = [np.asarray(s) for s in sweeps]
    n_elev = len(sw_list)
    if n_elev == 0:
        raise ValueError("no sweeps supplied")
    elevs = np.asarray(elevations_deg, dtype=np.float64)
    if elevs.size != n_elev:
        raise ValueError(f"{elevs.size} elevations for {n_elev} sweeps")

    # Azimuths and ranges may be given once (shared by every sweep, as in a
    # uniform VCP) or per sweep (as in a real volume where each tilt has its
    # own ray count). Accept both rather than forcing the caller to tile.
    if isinstance(azimuths_deg, (list, tuple)) and len(azimuths_deg) == n_elev \
            and np.ndim(azimuths_deg[0]) == 1:
        az_list = [np.asarray(a, dtype=np.float64) for a in azimuths_deg]
    else:
        az_list = [np.asarray(azimuths_deg, dtype=np.float64)] * n_elev
    if isinstance(ranges_km, (list, tuple)) and len(ranges_km) == n_elev \
            and np.ndim(ranges_km[0]) == 1:
        rg_list = [np.asarray(r, dtype=np.float64) for r in ranges_km]
    else:
        rg_list = [np.asarray(ranges_km, dtype=np.float64)] * n_elev

    target = C.CHARGING_LOWER_KM if target_alt_km is None else float(target_alt_km)
    rng_km, az_deg = range_azimuth_grid(grid_h, grid_w, pixel_km,
                                        radar_row, radar_col, north_up)

    sweep_grid = np.full((n_elev, grid_h, grid_w), np.nan, dtype=np.float64)
    sweep_h = np.zeros((n_elev, grid_h, grid_w), dtype=np.float64)

    for k in range(n_elev):
        data = np.asarray(sw_list[k], dtype=np.float64)
        az = az_list[k]
        rg = rg_list[k]
        if data.shape != (az.size, rg.size):
            raise ValueError(
                f"sweep {k} has shape {data.shape}, expected "
                f"({az.size}, {rg.size}) = (n_azimuth, n_range)")
        data = np.where(np.isfinite(data) & (data > nodata / 2.0), data, np.nan)

        # Angular distance on the circle to every ray, for every pixel.
        # H*W x n_az is ~1.5M entries for 64x64 and 360 rays -- brute force is
        # fine and avoids assuming the rays are uniformly spaced or sorted,
        # which real volumes routinely violate.
        d_az = np.abs(((az_deg.ravel()[:, None] - az[None, :]) + 180.0) % 360.0 - 180.0)
        j0 = np.argmin(d_az, axis=1)
        i0 = np.clip(np.searchsorted(rg, rng_km.ravel()) - 1, 0, rg.size - 1)
        i1 = np.clip(i0 + 1, 0, rg.size - 1)
        pick_hi = np.abs(rg[i1] - rng_km.ravel()) < np.abs(rg[i0] - rng_km.ravel())
        i_near = np.where(pick_hi, i1, i0)

        if method == "nearest":
            vals = data[j0, i_near]
        elif method == "idw":
            # Four surrounding samples: two rays x two gates, inverse distance.
            d_sorted = np.argsort(d_az, axis=1)[:, :2]
            jA, jB = d_sorted[:, 0], d_sorted[:, 1]
            num = np.zeros(rng_km.size)
            den = np.zeros(rng_km.size)
            for jj in (jA, jB):
                daz = d_az[np.arange(d_az.shape[0]), jj]
                for ii in (i0, i1):
                    v = data[jj, ii]
                    # Angular separation converted to a ground distance, so the
                    # two axes are weighted in comparable units.
                    darc = np.radians(daz) * np.maximum(rng_km.ravel(), 1e-3)
                    drad = np.abs(rg[ii] - rng_km.ravel())
                    w = 1.0 / np.maximum(np.hypot(darc, drad), 1e-3)
                    ok = np.isfinite(v)
                    num[ok] += w[ok] * v[ok]
                    den[ok] += w[ok]
            vals = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        else:
            raise ValueError(f"unknown method {method!r}; use 'nearest' or 'idw'")

        # Outside the sweep's own range coverage there is no measurement.
        out_of_range = (rng_km.ravel() < rg[0] - (rg[1] - rg[0] if rg.size > 1 else 0)) \
            | (rng_km.ravel() > rg[-1])
        vals = np.where(out_of_range, np.nan, vals)

        sweep_grid[k] = vals.reshape(grid_h, grid_w)
        sweep_h[k] = beam_height_km(rng_km, elevs[k], radar_alt_km, earth_factor)

    # Sweeps must be ordered by elevation for the vertical searches below.
    order = np.argsort(elevs)
    elevs_s = elevs[order]
    grid_s = sweep_grid[order]
    h_s = sweep_h[order]

    # ---- reflectivity at the target altitude ----
    below = h_s <= target                      # (n_elev, H, W)
    k_lo = below.sum(axis=0) - 1               # last sweep at or under target
    reason = np.zeros((grid_h, grid_w), dtype=np.int8)
    refl = np.full((grid_h, grid_w), nodata, dtype=np.float64)

    overshoot = k_lo < 0                       # even the lowest beam is above
    cone = k_lo >= (n_elev - 1)                # target above the highest beam
    ok_k = ~(overshoot | cone)
    reason[overshoot] = 1
    reason[cone] = 2

    if ok_k.any():
        kk = np.clip(k_lo, 0, n_elev - 2)
        ii, jj = np.nonzero(ok_k)
        kl = kk[ii, jj]
        v_lo = grid_s[kl, ii, jj]
        v_hi = grid_s[kl + 1, ii, jj]
        h_lo = h_s[kl, ii, jj]
        h_hi = h_s[kl + 1, ii, jj]
        good = np.isfinite(v_lo) & np.isfinite(v_hi)
        w = np.zeros_like(v_lo)
        span = np.maximum(h_hi - h_lo, 1e-6)
        w[good] = (target - h_lo[good]) / span[good]
        # Interpolate in dBZ, not in linear Z. Linear-Z interpolation is
        # dominated by whichever tilt is brighter and would bias the retrieval
        # toward the high-reflectivity sample; dBZ interpolation is the
        # convention in operational 3-D mosaics for exactly that reason.
        vals = (1.0 - w) * v_lo + w * v_hi
        refl[ii[good], jj[good]] = vals[good]
        bad = ok_k.copy()
        bad[ii[good], jj[good]] = False
        reason[bad] = 3
    refl_valid = reason == 0

    # ---- echo top: highest beam whose value reaches the threshold ----
    exceed = np.isfinite(grid_s) & (grid_s >= echo_top_dbz)
    any_ex = exceed.any(axis=0)
    k_top = (n_elev - 1) - np.argmax(exceed[::-1], axis=0)
    top = np.full((grid_h, grid_w), nodata, dtype=np.float64)
    lower_bound = np.zeros((grid_h, grid_w), dtype=bool)
    if any_ex.any():
        ii, jj = np.nonzero(any_ex)
        kt = k_top[ii, jj]
        h_at = h_s[kt, ii, jj]
        at_max = kt >= (n_elev - 1)
        # If a higher sweep exists, place the top where the profile crosses the
        # threshold between the last exceeding sweep and the one above it.
        kt_up = np.minimum(kt + 1, n_elev - 1)
        v_at = grid_s[kt, ii, jj]
        v_up = grid_s[kt_up, ii, jj]
        h_up = h_s[kt_up, ii, jj]
        interp = np.isfinite(v_up) & ~at_max & (v_at > v_up)
        frac = np.zeros_like(h_at)
        frac[interp] = ((v_at[interp] - echo_top_dbz)
                        / np.maximum(v_at[interp] - v_up[interp], 1e-6))
        est = np.where(interp, h_at + frac * (h_up - h_at), h_at)
        top[ii, jj] = est
        # Where the highest tilt still exceeds the threshold the storm extends
        # above the volume: the number is a LOWER BOUND, not a measurement.
        # Reporting it as an echo top systematically truncates the deepest --
        # i.e. the most electrified -- storms, which biases the very cases the
        # model most needs.
        lb = np.zeros((grid_h, grid_w), dtype=bool)
        lb[ii, jj] = at_max
        lower_bound = lb
    top_valid = any_ex

    lowest = grid_s[0]
    with np.errstate(invalid="ignore"):
        col_max = np.nanmax(grid_s, axis=0)
    col_max = np.where(np.isfinite(col_max), col_max, nodata)

    geom = radar_geometry_report(float(elevs_s[0]), float(elevs_s[-1]),
                                 beamwidth_deg)
    return {
        "refl_target": refl.astype(np.float32),
        "refl_target_valid": refl_valid,
        "target_alt_km": float(target),
        "invalid_reason": reason,
        "invalid_reason_key": {0: "ok", 1: "overshoot (lowest beam above target)",
                               2: "cone of silence (target above highest beam)",
                               3: "no valid data in bracketing sweeps"},
        "echo_top_km": top.astype(np.float32),
        "echo_top_valid": top_valid,
        "echo_top_is_lower_bound": lower_bound,
        "refl_lowest": np.where(np.isfinite(lowest), lowest, nodata).astype(np.float32),
        "column_max": col_max.astype(np.float32),
        "sweep_grid": np.where(np.isfinite(grid_s), grid_s, nodata).astype(np.float32),
        "sweep_height_km": h_s.astype(np.float32),
        "range_km": rng_km.astype(np.float32),
        "azimuth_deg": az_deg.astype(np.float32),
        "meta": {
            "n_sweeps": int(n_elev),
            "elevations_deg": elevs_s.tolist(),
            "method": method,
            "earth_factor": float(earth_factor),
            "beam_depth_km_at_100km": float(beam_width_km(100.0, beamwidth_deg)),
            "geometry": geom,
            "frac_target_valid": float(refl_valid.mean()),
            "frac_overshoot": float((reason == 1).mean()),
            "frac_cone_of_silence": float((reason == 2).mean()),
            "frac_echo_top_lower_bound": float(lower_bound.mean()),
        },
    }


def load_imd_dwr(paths: Sequence[str],
                 *,
                 kind: str = "auto",
                 grid_h: int = C.GRID_H, grid_w: int = C.GRID_W,
                 pixel_km: float = C.PIXEL_KM,
                 dbz_lut: Sequence[tuple[float, tuple[int, int, int]]] | None = None,
                 image_km_per_px: float | None = None,
                 elevations_deg: Sequence[float] | None = None,
                 radar_alt_km: float = 0.0,
                 verbose: bool = True) -> dict:
    """Load IMD DWR products, dispatching on what the files actually are.

    Args:
        paths: one file per time step, in time order.
        kind: "auto" | "odim" | "image". "auto" dispatches on the extension.
        image_km_per_px: required for the image path -- a rendered product has
            no georeferencing, so the scale must come from the product's own
            documentation (range rings help: a 250 km ring on a 500 px wide
            image is 1 km/px). Guessing it stretches the domain.

    Returns a partial event: the channels this source can supply, with mask
    False everywhere it cannot. It is a partial event on purpose -- IMD DWR
    provides no lightning, so `y` must be filled from a lightning network
    before the result is usable for training.

    WHICH PATH YOU ARE ON MATTERS ENORMOUSLY:
      ODIM/HDF5 polar volumes -> polar_to_cartesian -> real refl_m10 and a real
        echo top. This is the configuration the project's physics argument
        assumes.
      Rendered PNG/GIF -> dequantize_colormap -> an approximate surface
        reflectivity ONLY. refl_m10 and echo_top stay masked False. Do not
        synthesise them from a MAX-Z picture; see the dequantize_colormap
        docstring.
    """
    n = len(paths)
    ev = _empty_event(n, grid_h, grid_w)
    surrogate: list[str] = []
    mode_used = kind

    for t, p in enumerate(paths):
        ext = os.path.splitext(p)[1].lower()
        mode = kind
        if mode == "auto":
            mode = "image" if ext in (".png", ".gif", ".jpg", ".jpeg") else "odim"
        mode_used = mode

        if mode == "image":
            if image_km_per_px is None:
                raise ValueError(
                    "image_km_per_px is required for rendered DWR products: a "
                    "PNG carries no georeferencing. Read the scale off the "
                    "product's range rings or its documentation.")
            PIL = _require("PIL.Image", "read rendered IMD DWR product images")
            im = np.asarray(PIL.open(p).convert("RGB"))
            dbz, ok, rep = dequantize_colormap(im, dbz_lut)
            tile, tile_ok = to_config_grid(dbz, image_km_per_px, valid=ok,
                                           grid_h=grid_h, grid_w=grid_w,
                                           pixel_km=pixel_km)
            ev["x"][t, C.CH["refl_sfc"]] = tile
            ev["mask"][t, C.CH["refl_sfc"]] = tile_ok
            # refl_m10 and echo_top deliberately left masked False.
            if "refl_sfc(image-derived)" not in surrogate:
                surrogate.append("refl_sfc(image-derived)")
            if verbose and t == 0:
                print(f"  imd_dwr image: {rep['frac_rejected']:.1%} of pixels "
                      f"rejected as non-data colours; quantisation "
                      f"+/-{rep['quantisation_error_dbz']:.1f} dBZ")
        else:
            h5py = _require("h5py", "read IMD DWR ODIM_H5 polar volumes")
            vol = read_odim_volume(h5py, p, elevations_deg=elevations_deg)
            out = polar_to_cartesian(
                vol["sweeps"], vol["elevations_deg"], vol["azimuths_deg"],
                vol["ranges_km"], grid_h=grid_h, grid_w=grid_w,
                pixel_km=pixel_km, radar_alt_km=radar_alt_km)
            ev["x"][t, C.CH["refl_sfc"]] = out["refl_lowest"]
            ev["mask"][t, C.CH["refl_sfc"]] = out["refl_lowest"] > C.FILL_VALUE / 2
            ev["x"][t, C.CH["refl_m10"]] = out["refl_target"]
            ev["mask"][t, C.CH["refl_m10"]] = out["refl_target_valid"]
            ev["x"][t, C.CH["echo_top"]] = out["echo_top_km"]
            ev["mask"][t, C.CH["echo_top"]] = out["echo_top_valid"]

    ev["x"][~ev["mask"]] = C.FILL_VALUE
    ev["meta"] = {
        "source": "imd_dwr",
        "mode": mode_used,
        "paths": list(paths),
        "surrogate_channels": surrogate,
        "channels_supplied": (["refl_sfc"] if mode_used == "image"
                              else ["refl_sfc", "refl_m10", "echo_top"]),
        "missing_channels": ["vil", "ir_tb", "light_dens"]
                            + (["refl_m10", "echo_top"] if mode_used == "image" else []),
        "target_note": ("IMD DWR carries no lightning. `y` is all zeros here; "
                        "fill it from IITM Damini / WWLLN before training or "
                        "the model will learn that lightning never happens."),
        "schema_verified": False,
    }
    return ev


def read_odim_volume(h5py_mod, path: str,
                     quantity: str = "DBZH",
                     elevations_deg: Sequence[float] | None = None) -> dict:
    """Read an ODIM_H5 polar volume into sweeps / elevations / azimuths / ranges.

    VERIFY: ODIM_H5 layout recalled as
        /dataset<N>/where          attrs elangle, nbins, nrays, rscale (metres),
                                   rstart (km), a1gate
        /dataset<N>/data<M>/what   attrs quantity (b'DBZH'), gain, offset,
                                   nodata, undetect
        /dataset<N>/data<M>/data   uint8/uint16 (nrays, nbins)
        /where                     attrs lat, lon, height
    Decoding is value = gain * raw + offset, with `nodata` and `undetect`
    mapped to missing. Check gain/offset on a real file: forgetting them turns
    dBZ into raw counts, and counts are monotone in dBZ, so plots still look
    like weather while every threshold in the project is meaningless.

    IRIS/Sigmet RAW files are NOT handled here -- they need a real decoder
    (pyart, or wradlib). If your IMD data arrives as .RAW, install one.
    """
    sweeps, elevs, azs, rgs = [], [], [], []
    with h5py_mod.File(path, "r") as hf:
        ds_names = sorted((k for k in hf.keys() if k.startswith("dataset")),
                          key=lambda s: int(re.sub(r"\D", "", s) or 0))
        if not ds_names:
            raise ValueError(
                f"no /dataset* groups in {path}; datasets present: "
                f"{list(hf.keys())}. This may not be ODIM_H5. VERIFY.")
        for i, dn in enumerate(ds_names):
            g = hf[dn]
            where = g["where"].attrs if "where" in g else {}
            elang = float(np.asarray(where.get("elangle", np.nan)).ravel()[0]) \
                if "elangle" in where else (
                    float(elevations_deg[i]) if elevations_deg is not None else float("nan"))
            if not np.isfinite(elang):
                raise ValueError(
                    f"{path}:{dn} has no 'elangle' attribute and no fallback "
                    f"was supplied via elevations_deg. Refusing to guess the "
                    f"elevation: the whole vertical retrieval depends on it.")

            dgrp = None
            for k in sorted(g.keys()):
                if not k.startswith("data"):
                    continue
                q = g[k]["what"].attrs.get("quantity", b"") if "what" in g[k] else b""
                q = q.decode() if isinstance(q, bytes) else str(q)
                if q == quantity:
                    dgrp = g[k]
                    break
            if dgrp is None:
                continue

            what = dgrp["what"].attrs
            raw = np.asarray(dgrp["data"][:], dtype=np.float64)
            gain = float(np.asarray(what.get("gain", 1.0)).ravel()[0])
            offset = float(np.asarray(what.get("offset", 0.0)).ravel()[0])
            val = gain * raw + offset
            for key in ("nodata", "undetect"):
                if key in what:
                    sentinel = float(np.asarray(what[key]).ravel()[0])
                    val[raw == sentinel] = np.nan

            nrays, nbins = val.shape
            rscale_m = float(np.asarray(where.get("rscale", 1000.0)).ravel()[0])
            rstart_km = float(np.asarray(where.get("rstart", 0.0)).ravel()[0])
            rng = rstart_km + (np.arange(nbins) + 0.5) * rscale_m / 1000.0
            # ODIM rays start at north and run clockwise; a1gate gives the index
            # of the ray that was collected first, which we do not need since we
            # index by azimuth rather than by acquisition order.
            az = (np.arange(nrays) + 0.5) * (360.0 / nrays)

            sweeps.append(val)
            elevs.append(elang)
            azs.append(az)
            rgs.append(rng)

    if not sweeps:
        raise ValueError(f"no sweeps with quantity {quantity!r} found in {path}")
    return {"sweeps": sweeps, "elevations_deg": elevs,
            "azimuths_deg": azs, "ranges_km": rgs, "quantity": quantity}


# ===========================================================================
# Open-Meteo environmental covariates
# ===========================================================================

def load_openmeteo(lat: float, lon: float,
                   start_date: str, end_date: str,
                   *,
                   variables: Sequence[str] = ("cape", "convective_inhibition",
                                               "lifted_index"),
                   archive: bool = True,
                   timeout: int = 30) -> dict:
    """Hourly CAPE / CIN / lifted index at one point, as a per-event covariate.

    WHY THIS IS NOT A NOWCAST SOURCE, and must never be treated as one:

      * CADENCE. Hourly. The phenomenon we forecast evolves on 5 min. Between
        two Open-Meteo samples an entire pulse thunderstorm can initiate,
        electrify, produce fifty flashes and decay.
      * RESOLUTION. It is a reanalysis/model grid at roughly 9-25 km depending
        on the model. Our pixel is 2 km. It cannot resolve a convective cell,
        and in fact it is not trying to: CAPE is a property of a SOUNDING, an
        environmental profile, not of a storm.
      * CAUSALITY IS THE WRONG WAY ROUND. CAPE describes how much energy is
        AVAILABLE if a parcel is lifted. It says nothing about whether anything
        is lifting it. High CAPE with a cap and no trigger produces a blue sky
        all afternoon; moderate CAPE with strong forcing produces a squall
        line. CAPE is a necessary-ish condition, never a sufficient one.
      * LEAKAGE RISK. A reanalysis assimilates observations, so an ERA5 CAPE
        field for 15:00 has, in part, been informed by what the atmosphere
        actually did. Using an archive product as a "predictor" for a case in
        the same hour is a subtle way to leak the answer. Use the FORECAST
        endpoint (archive=False) valid before the event when building anything
        that claims operational realism.

    The legitimate use is one scalar (or a handful) per EVENT, conditioning the
    model on the airmass: was this a 3000 J/kg pre-monsoon Kalbaisakhi
    environment or a 700 J/kg monsoon-break day? That distinction genuinely
    changes the reflectivity-to-flash-rate relationship, and it is a per-event
    constant, which is exactly what a per-event covariate is for.

    VERIFY: Open-Meteo's hourly variable names and which of them each model
    exposes. `cape` is widely available; `convective_inhibition` and
    `lifted_index` are present for some models only, and the archive endpoint
    exposes a different set from the forecast endpoint. Missing variables are
    reported in the returned dict rather than silently becoming zeros.
    """
    base = ("https://archive-api.open-meteo.com/v1/archive" if archive
            else "https://api.open-meteo.com/v1/forecast")
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(variables), "timezone": "UTC",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base}?{query}"

    payload: dict
    try:
        requests = _require("requests", "query the Open-Meteo API")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except MissingDependency:
        # Stdlib fallback so a missing `requests` is not a hard blocker for a
        # single JSON GET.
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    got = {v: hourly.get(v) for v in variables}
    missing = [v for v, arr in got.items() if not arr]

    summary: dict[str, Any] = {}
    for v, arr in got.items():
        if not arr:
            continue
        a = np.array([np.nan if z is None else float(z) for z in arr])
        if np.isfinite(a).any():
            summary[f"{v}_mean"] = float(np.nanmean(a))
            summary[f"{v}_max"] = float(np.nanmax(a))
            summary[f"{v}_min"] = float(np.nanmin(a))

    return {
        "source": "openmeteo",
        "endpoint": "archive" if archive else "forecast",
        "url": url,
        "latlon": [float(lat), float(lon)],
        "time": hourly.get("time", []),
        "hourly": got,
        "summary": summary,
        "missing_variables": missing,
        "is_nowcast_source": False,
        "usage": "per-event environmental covariate only; never a 5 min predictor",
        "leakage_warning": ("the archive endpoint is a reanalysis and has seen "
                            "the observations; use archive=False for anything "
                            "claiming operational realism"),
        "schema_verified": False,
    }


# ===========================================================================
# Persistence
# ===========================================================================

_SAVE_FORMAT = 2


class _NpEncoder(json.JSONEncoder):
    """Meta dicts routinely contain numpy scalars, which json refuses."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)


def save_events(events: Sequence[dict], path: str, *,
                validate: bool = True) -> str:
    """Write a list of events to a compressed .npz.

    The manifest records config.CHANNELS, the grid geometry and FILL_VALUE at
    the time of writing. `load_events` refuses a file whose channel list does
    not match the current config. That check matters: adding one channel to
    config.CHANNELS silently shifts every channel index, so a cached array
    written before the change would load with reflectivity in the IR slot and
    the model would train on it without a single error being raised.
    """
    arrays: dict[str, np.ndarray] = {}
    metas: list[dict] = []
    for k, ev in enumerate(events):
        if validate:
            validate_event(ev, name=f"event[{k}]")
        arrays[f"x{k}"] = np.asarray(ev["x"], dtype=np.float32)
        arrays[f"m{k}"] = np.asarray(ev["mask"], dtype=bool)
        arrays[f"y{k}"] = np.asarray(ev["y"], dtype=np.float32)
        metas.append(ev.get("meta", {}))

    manifest = {
        "format": _SAVE_FORMAT,
        "n_events": len(events),
        "channels": list(C.CHANNELS),
        "grid": [C.GRID_H, C.GRID_W],
        "pixel_km": C.PIXEL_KM,
        "timestep_min": C.TIMESTEP_MIN,
        "fill_value": C.FILL_VALUE,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "metas": metas,
    }
    arrays["_manifest"] = np.array(
        json.dumps(manifest, cls=_NpEncoder), dtype=object)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_events(path: str, *, validate: bool = True,
                strict_channels: bool = True) -> list[dict]:
    """Read events written by `save_events`, checking the config still matches."""
    with np.load(path, allow_pickle=True) as z:
        if "_manifest" not in z:
            raise ValueError(
                f"{path} has no _manifest; it was not written by save_events")
        manifest = json.loads(str(z["_manifest"].item()))
        if manifest.get("channels") != list(C.CHANNELS) and strict_channels:
            raise ValueError(
                f"{path} was written with channels {manifest.get('channels')} "
                f"but config.CHANNELS is now {list(C.CHANNELS)}. Channel "
                f"indices have shifted; regenerate the cache. Pass "
                f"strict_channels=False only if you have checked the mapping "
                f"by hand.")
        n = int(manifest["n_events"])
        metas = manifest.get("metas", [{}] * n)
        events = []
        for k in range(n):
            ev = {"x": z[f"x{k}"], "mask": z[f"m{k}"], "y": z[f"y{k}"],
                  "meta": metas[k] if k < len(metas) else {}}
            if validate:
                validate_event(ev, name=f"{os.path.basename(path)}[{k}]")
            events.append(ev)
    return events


# ===========================================================================
# Self-test -- runs with NO network and NONE of the optional dependencies
# ===========================================================================

def _test_polar() -> None:
    """Grid a hand-made polar volume with one echo at a known range/azimuth.

    Constructing the volume by hand (rather than checking a real file, which
    the sandbox cannot reach) tests the part that is actually easy to get
    wrong: the azimuth convention, the row/column convention, the beam-height
    formula and the vertical interpolation between tilts.
    """
    print("-" * 74)
    print("polar_to_cartesian: synthetic volume, echo at r=30 km, az=45 deg")
    print("-" * 74)

    elevs = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0, 14.0, 19.5])
    az = np.arange(0.0, 360.0, 1.0)
    rg = np.arange(0.5, 120.0, 0.5)
    n_e, n_a, n_r = elevs.size, az.size, rg.size
    r_i = int(np.argmin(np.abs(rg - 30.0)))
    a_i = int(np.argmin(np.abs(az - 45.0)))

    def _build(lapse_db_per_km: float) -> np.ndarray:
        """One column at (30 km, 45 deg), 55 dBZ at the surface, decaying."""
        vol = np.full((n_e, n_a, n_r), -32.0)
        for k, e in enumerate(elevs):
            val = 55.0 - lapse_db_per_km * float(beam_height_km(rg[r_i], e))
            # A few gates and rays wide so nearest-neighbour sampling lands on it.
            vol[k, a_i - 2:a_i + 3, r_i - 2:r_i + 3] = val
        return vol

    # Case A: 6 dB/km, so the 18 dBZ top sits at (55-18)/6 = 6.17 km, inside
    # the volume at this range -- the fully sampled case.
    vol = _build(6.0)
    out = polar_to_cartesian(vol, elevs, az, rg,
                             target_alt_km=C.CHARGING_LOWER_KM, method="nearest")

    # Expected pixel: 30 km at 45 deg is 21.21 km north and 21.21 km east of
    # the radar, which sits at the grid centre (31.5, 31.5) at 2 km pixels.
    exp_row = int(round(31.5 - 21.213 / C.PIXEL_KM))
    exp_col = int(round(31.5 + 21.213 / C.PIXEL_KM))
    m = out["column_max"]
    got = np.unravel_index(int(np.argmax(m)), m.shape)
    print(f"  expected peak pixel (row, col) = ({exp_row}, {exp_col})")
    print(f"  actual   peak pixel (row, col) = ({got[0]}, {got[1]})   "
          f"column_max = {m[got]:.1f} dBZ")
    assert abs(got[0] - exp_row) <= 1 and abs(got[1] - exp_col) <= 1, \
        f"echo landed at {got}, expected ~({exp_row}, {exp_col}) -- azimuth or " \
        f"row/col convention is wrong"

    h_lowest = float(beam_height_km(30.0, 0.5))
    h_highest = float(beam_height_km(30.0, 19.5))
    print(f"  at r=30 km the volume spans {h_lowest:.2f} km (0.5 deg tilt) to "
          f"{h_highest:.2f} km (19.5 deg tilt)")

    exp_m10 = 55.0 - 6.0 * C.CHARGING_LOWER_KM
    got_m10 = float(out["refl_target"][exp_row, exp_col])
    print(f"  refl at {out['target_alt_km']:.1f} km (-10C), interpolated "
          f"between tilts: expected {exp_m10:.1f} dBZ, got {got_m10:.1f} dBZ, "
          f"valid={bool(out['refl_target_valid'][exp_row, exp_col])}")
    assert abs(got_m10 - exp_m10) < 1.0, "vertical interpolation is off"

    exp_top = (55.0 - 18.0) / 6.0
    got_top = float(out["echo_top_km"][exp_row, exp_col])
    lb = bool(out["echo_top_is_lower_bound"][exp_row, exp_col])
    print(f"  18 dBZ echo top     : expected {exp_top:.2f} km, got "
          f"{got_top:.2f} km, lower_bound={lb}")
    assert abs(got_top - exp_top) < 0.5, "echo top retrieval is off"
    assert not lb, "a fully sampled echo top was wrongly flagged as a lower bound"

    # Case B: 3 dB/km puts the true top at (55-18)/3 = 12.33 km, which at 30 km
    # range is ABOVE the highest tilt. The correct answer is not 12.33 km -- the
    # radar cannot see it -- but the top of the volume, flagged as a lower
    # bound. Reporting that number as an echo top without the flag would
    # systematically truncate the deepest, most electrified storms, and it
    # would do so worst near the radar where the cone of silence is widest.
    out_b = polar_to_cartesian(_build(3.0), elevs, az, rg,
                               target_alt_km=C.CHARGING_LOWER_KM)
    top_b = float(out_b["echo_top_km"][exp_row, exp_col])
    lb_b = bool(out_b["echo_top_is_lower_bound"][exp_row, exp_col])
    print(f"  same column at 3 dB/km (true top 12.33 km, above the volume): "
          f"reported {top_b:.2f} km, lower_bound={lb_b}")
    assert lb_b, "an unsampled echo top was not flagged as a lower bound"
    assert abs(top_b - h_highest) < 0.5, \
        "lower-bound echo top should be the highest sampled beam"

    g = out["meta"]["geometry"]
    print(f"  coverage: {out['meta']['frac_target_valid']:.1%} of pixels have a "
          f"valid -10C retrieval; {out['meta']['frac_cone_of_silence']:.1%} in "
          f"the cone of silence (r < {g['cone_of_silence_km']:.1f} km), "
          f"{out['meta']['frac_overshoot']:.1%} overshot")
    print(f"  beam depth at 100 km : "
          f"{out['meta']['beam_depth_km_at_100km']:.2f} km "
          f"(charging layer is only "
          f"{C.CHARGING_UPPER_KM - C.CHARGING_LOWER_KM:.1f} km thick)")

    # IDW must agree with nearest to within the sampling granularity.
    out2 = polar_to_cartesian(vol, elevs, az, rg,
                              target_alt_km=C.CHARGING_LOWER_KM, method="idw")
    d = float(abs(out2["refl_target"][exp_row, exp_col] - got_m10))
    print(f"  idw vs nearest at the echo pixel: {d:.2f} dBZ apart")


def _test_colormap() -> None:
    """Round-trip a dBZ field through a colour LUT and back."""
    print("-" * 74)
    print("dequantize_colormap: round-trip through a known LUT")
    print("-" * 74)

    lut = DEFAULT_DBZ_LUT
    vals = np.array([v for v, _ in lut])
    cols = np.array([c for _, c in lut], dtype=np.uint8)
    step = float(np.median(np.diff(vals)))

    rng = np.random.default_rng(7)
    truth = rng.uniform(vals.min(), vals.max(), size=(40, 40))
    idx = np.argmin(np.abs(truth[..., None] - vals[None, None, :]), axis=2)
    img = cols[idx]                                   # render
    # Burn in an annotation: a white timestamp box and a black graticule, both
    # of which are what real IMD product images actually have on them.
    img = img.copy()
    img[0:3, :] = (255, 255, 255)
    img[:, 0:2] = (0, 0, 0)

    dbz, valid, rep = dequantize_colormap(img, lut, max_dist=30.0)
    data = valid.copy()
    data[0:3, :] = False
    data[:, 0:2] = False
    err = np.abs(dbz[data] - truth[data])

    print(f"  LUT step {step:.1f} dBZ -> best-case quantisation error "
          f"+/-{rep['quantisation_error_dbz']:.2f} dBZ")
    print(f"  round-trip error on data pixels: max {err.max():.2f} dBZ, "
          f"mean {err.mean():.2f} dBZ")
    print(f"  annotation rejected: white rows "
          f"{'yes' if not valid[0:3, 2:].any() else 'NO'}, "
          f"black column {'yes' if not valid[3:, 0:2].any() else 'NO'}")
    print(f"  overall {rep['frac_rejected']:.1%} of pixels rejected as non-data "
          f"colours (lossy={rep['lossy']}, "
          f"usable_for_refl_m10={rep['usable_for_refl_m10']})")
    assert err.max() <= step / 2.0 + 1e-6, \
        f"round-trip error {err.max():.2f} exceeds half a LUT step"
    assert not valid[0:3, 2:].any(), "white annotation was accepted as data"


def _self_test() -> None:
    t0 = time.time()
    print("=" * 74)
    print("data_io.py SELF-TEST -- no network, no optional dependencies")
    print("=" * 74)
    print()
    print(format_sources_report())
    print()

    import synth

    # ---- 1. contract ----------------------------------------------------
    print("-" * 74)
    print("validate_event: synthetic event must satisfy the loader contract")
    print("-" * 74)
    # Seed 1 is chosen because it yields a ~1.6% lightning base rate, close to
    # the ~1% the rest of the config is tuned for. Several seeds produce a
    # completely null event; validate_event warns rather than raises on those
    # (a null storm day is real data, not a bug), but a null event would make
    # the QC demonstration below uninformative.
    rng = np.random.default_rng(1)
    ev = synth.generate_event(rng, n_frames=24)
    rep = validate_event(ev, name="synth")
    print(f"  shape {rep['shape']}  dtypes {rep['dtypes']}")
    print(f"  valid fraction {rep['valid_frac_overall']:.4f}   "
          f"base rate {rep['base_rate']:.5f}   "
          f"strikes {rep['total_strikes']:.0f}")
    for w in rep["warnings"]:
        print(f"  warning: {w}")

    # A check that never fires is not a check. Break the contract deliberately.
    for label, mutate in (
        ("channels-last", lambda e: {**e, "x": np.moveaxis(e["x"], 1, -1)}),
        ("NaN in x", lambda e: {**e, "x": _with_nan(e["x"])}),
        ("mask dtype", lambda e: {**e, "mask": e["mask"].astype(np.uint8)}),
    ):
        try:
            validate_event(mutate(ev), name=label)
            print(f"  FAIL: {label} was accepted")
        except EventContractError as exc:
            print(f"  rejects {label:14s}: {str(exc).splitlines()[0][:88]}")

    # ---- 2. quality control ---------------------------------------------
    print()
    xq = ev["x"].copy()
    mq = ev["mask"].copy()
    # (a) a sentinel that survived a loader: 9999 in VIL.
    xq[3, C.CH["vil"], 10, 10] = 9999.0
    # (b) a textbook ground-clutter target: strong, flat, stationary.
    xq[:, C.CH["refl_sfc"], 5:9, 5:9] = 48.0
    xq[:, C.CH["echo_top"], 5:9, 5:9] = 0.4
    mq[:, C.CH["refl_sfc"], 5:9, 5:9] = True
    mq[:, C.CH["echo_top"], 5:9, 5:9] = True
    # (c) a textbook bright band: moderate, slowly varying surface echo with an
    # echo top just above the 5.0 km freezing level and no charging-layer
    # signal. Injected rather than harvested from synth.py because the rule
    # must be shown to fire on a case whose ground truth is known by
    # construction; tuning it until it fires on fabricated data would be
    # tuning to the fabrication.
    tvec = np.arange(xq.shape[0], dtype=np.float32)
    xq[:, C.CH["refl_sfc"], 40:48, 40:48] = (38.0 + np.sin(tvec) * 0.8)[:, None, None]
    xq[:, C.CH["echo_top"], 40:48, 40:48] = 5.8
    xq[:, C.CH["refl_m10"], 40:48, 40:48] = 6.0
    for _ch in ("refl_sfc", "echo_top", "refl_m10"):
        mq[:, C.CH[_ch], 40:48, 40:48] = True
    xq[~mq] = C.FILL_VALUE

    # Radar at the grid centre with a deliberately short trusted range, so the
    # range-degradation rule fires inside this 128 km domain. A real S-band
    # DWR would use something like 150 km, which a 128 km box never reaches --
    # the rule matters for larger domains and for radars off to one side.
    x_qc, m_qc, qrep = quality_control(
        xq, mq, radar_center=(31.5, 31.5), max_range_km=40.0)
    print(format_qc_report(qrep))
    assert qrep["rules"]["range_check"]["per_channel"]["vil"] >= 1, \
        "the injected 9999 sentinel was not caught"
    assert qrep["rules"]["ap_clutter"]["masked"] > 0, \
        "the injected stationary ground clutter was not caught"
    assert qrep["rules"]["range_degradation"]["masked"] > 0, \
        "range degradation did not fire"
    assert qrep["rules"]["bright_band"]["n_pixels_flagged"] > 0, \
        "the injected bright band was not flagged"
    # The bright band is FLAGGED but the reflectivity is NOT blanked: it is a
    # real echo, and the flag is what metrics use to report the false-alarm
    # rate on the mimic population.
    assert m_qc[0, C.CH["refl_sfc"], 44, 44], \
        "bright-band surface reflectivity was masked; it should only be flagged"
    assert float(x_qc[3, C.CH["vil"], 10, 10]) == C.FILL_VALUE, \
        "out-of-range value was clamped instead of filled"
    ev_qc = {"x": x_qc, "mask": m_qc, "y": ev["y"], "meta": ev["meta"]}
    validate_event(ev_qc, name="post-qc")
    print("  post-QC event still satisfies the contract")

    # ---- 3. geometry ----------------------------------------------------
    print()
    print("-" * 74)
    print("beam geometry (why refl_m10 is unobservable in places)")
    print("-" * 74)
    for r in (10.0, 50.0, 100.0, 150.0, 200.0, 250.0):
        h_lo = float(beam_height_km(r, 0.5))
        h_hi = float(beam_height_km(r, 19.5))
        print(f"  r = {r:6.1f} km : lowest tilt (0.5 deg) centre {h_lo:6.2f} km, "
              f"highest (19.5 deg) {h_hi:6.2f} km, beam depth "
              f"{float(beam_width_km(r)):5.2f} km")
    g = radar_geometry_report()
    print(f"  charging layer {g['charging_lower_km']}-{g['charging_upper_km']} km "
          f"is sampled only for {g['cone_of_silence_km']:.0f} km < r < "
          f"{g['overshoot_range_km']:.0f} km")

    # ---- 4. polar gridding and colour dequantisation ---------------------
    print()
    _test_polar()
    print()
    _test_colormap()

    # ---- 5. persistence --------------------------------------------------
    print()
    print("-" * 74)
    print("save_events / load_events round trip")
    print("-" * 74)
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "cache", "_data_io_selftest.npz")
    ev_qc["meta"] = dict(ev_qc["meta"])
    ev_qc["meta"]["source"] = "synth"
    save_events([ev_qc], tmp)
    back = load_events(tmp)
    same = (np.array_equal(back[0]["x"], ev_qc["x"])
            and np.array_equal(back[0]["mask"], ev_qc["mask"])
            and np.array_equal(back[0]["y"], ev_qc["y"]))
    size_kb = os.path.getsize(tmp) / 1024.0
    print(f"  wrote {tmp} ({size_kb:.0f} kB), round trip exact: {same}")
    print(f"  meta survived: source={back[0]['meta'].get('source')!r}, "
          f"keys={sorted(back[0]['meta'])[:5]}")
    assert same
    try:
        os.remove(tmp)
    except OSError as exc:
        # Some sandboxes mount the workspace without unlink permission. The
        # round trip is what is being tested; failing the whole self-test over
        # a leftover scratch file would be noise.
        print(f"  (could not delete the scratch file: {exc})")

    # ---- 6. what is NOT tested ------------------------------------------
    print()
    print("=" * 74)
    print("NOT TESTED HERE (no network, no h5py/netCDF4/PIL in this sandbox)")
    print("=" * 74)
    for line in (
        "load_sevir       HDF5 layout, axis order, VIL/IR decoding, lght columns",
        "load_goes_glm    variable names, time epoch, quality flag semantics",
        "load_mosdac_insat dataset names, count->BT LUT, geolocation attributes",
        "load_imd_dwr     ODIM_H5 group layout, gain/offset, the real dBZ colour scale",
        "load_openmeteo   which hourly variables each model exposes",
    ):
        print(f"  {line}")
    print("  Every one of these carries a `# VERIFY:` comment at the point of "
          "use. Check them against a real file with h5ls / ncdump -h BEFORE "
          "trusting any number a real-data run produces.")
    print()
    print(f"ALL data_io.py ASSERTIONS PASSED in {time.time() - t0:.1f}s")


def _with_nan(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[0, 0, 0, 0] = np.nan
    return y


if __name__ == "__main__":
    _self_test()

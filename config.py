"""
Central configuration for the lightning nowcasting system.

Single source of truth for grid geometry, channel definitions, thermodynamic
levels, lead-time bins and per-backend hyperparameters.

DESIGN NOTE ON PHYSICS
----------------------
Lightning requires charge separation via *non-inductive charging*: graupel
colliding with ice crystals in the presence of supercooled liquid water.
This reaction is only efficient in the -10C to -20C temperature layer
(Takahashi 1978, Saunders 1993). Everything in this config that references
"charging layer" refers to that band.

The practical consequence: our predictors must sample reflectivity *at the
height of the -10C isotherm*, not at the surface. A 55 dBZ surface echo from
a warm-rain tropical shower has no ice aloft and produces no lightning; a
40 dBZ echo extending to -20C is very likely electrified. This distinction is
the single most important signal in the whole system, and it is why we carry
`refl_m10` as an explicit channel rather than only surface reflectivity.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------
# 64x64 at 2 km spacing = 128 km x 128 km domain. This is deliberately chosen:
#  - 2 km is close to the effective resolution of DWR reflectivity composites
#    after beam broadening at moderate range (IMD DWRs are 1 km gate spacing
#    but the beam is ~1.5 km wide at 100 km range, so 2 km is honest).
#  - 128 km is roughly one radar's reliable quantitative coverage and is large
#    enough to contain a mesoscale convective cluster's 2 h advection.
GRID_H = 64
GRID_W = 64
PIXEL_KM = 2.0

# ---------------------------------------------------------------------------
# Temporal specification
# ---------------------------------------------------------------------------
# 5 min is the DWR volume scan cadence in most operational modes.
TIMESTEP_MIN = 5

# Input: 6 frames = 30 min of history. Enough to establish growth/decay
# tendency and an advection vector; short enough that the storm has not
# fundamentally reorganised.
INPUT_FRAMES = 6

# Output: 12 frames = 60 min ahead at 5 min resolution.
#
# WHY 60 AND NOT 120: the physical lead time from "graupel appears in the
# charging layer" to "first strike" is roughly 5-15 min. Beyond ~30-45 min we
# are no longer forecasting charging physics from observed ice; we are
# forecasting *convective initiation*, which is chaotic and not skilfully
# predictable from radar echo extrapolation alone. Published deep-learning
# nowcasters show CSI collapsing past 60 min. We predict to 60 min honestly
# rather than to 120 min dishonestly. See LEAD_TIME_BINS for reporting.
OUTPUT_FRAMES = 12

# Report metrics in these lead-time bands (minutes, inclusive lower bound).
# Skill should visibly decay across these; if it does not, suspect leakage.
LEAD_TIME_BINS = [(5, 15), (20, 30), (35, 45), (50, 60)]

# ---------------------------------------------------------------------------
# Thermodynamic levels
# ---------------------------------------------------------------------------
# Heights (km AGL) of key isotherms. These vary with airmass; the values here
# are representative of the Indian monsoon/pre-monsoon warm-season sounding,
# which has a notably HIGH freezing level compared to mid-latitudes.
#
# WHY THIS MATTERS FOR INDIA: with a freezing level near 5 km, a storm must
# grow substantially deeper before any of its mass reaches the charging layer.
# Warm-rain processes efficiently remove condensate below the freezing level,
# so heavy-raining tropical cells can be electrically quiet. This is exactly
# the "mimic" population that drives false alarms, and it is why an approach
# tuned on US NEXRAD data (freezing level ~3.5 km) transfers poorly.
FREEZING_LEVEL_KM = 5.0      # 0C
CHARGING_LOWER_KM = 6.5      # -10C  <- lower bound of non-inductive charging
CHARGING_UPPER_KM = 8.0      # -20C  <- upper bound
# Operational threshold: reflectivity >= this value at the -10C level implies
# graupel present in the charging zone (Gremillion & Vincent 1998).
CHARGING_DBZ_THRESHOLD = 40.0

# ---------------------------------------------------------------------------
# Input channels
# ---------------------------------------------------------------------------
# Ordered; index into the channel axis of a Frame array.
CHANNELS = [
    "refl_sfc",     # dBZ, surface/lowest-tilt reflectivity composite
    "refl_m10",     # dBZ, reflectivity interpolated to the -10C level
    "echo_top",     # km, height of the 18 dBZ echo top
    "vil",          # kg/m^2, vertically integrated liquid
    "ir_tb",        # K, IR window (10.8 um) brightness temperature
    "light_dens",   # strikes / pixel / 5 min, from the LLN
]
N_CHANNELS = len(CHANNELS)
CH = {name: i for i, name in enumerate(CHANNELS)}

# Physical valid ranges, used for QC clipping on real data. Values outside
# these are set to the fill value and flagged, not silently clamped.
CHANNEL_RANGES = {
    "refl_sfc": (-32.0, 80.0),
    "refl_m10": (-32.0, 80.0),
    "echo_top": (0.0, 20.0),
    "vil": (0.0, 70.0),
    "ir_tb": (180.0, 320.0),
    "light_dens": (0.0, 500.0),
}

# Normalisation constants (subtract mean, divide by scale) so all channels
# land near unit variance for the network. Chosen from physical ranges rather
# than fitted to a dataset, so they are stable when data is swapped.
CHANNEL_NORM = {
    "refl_sfc": (20.0, 20.0),
    "refl_m10": (10.0, 20.0),
    "echo_top": (6.0, 4.0),
    "vil": (5.0, 10.0),
    "ir_tb": (270.0, 30.0),
    "light_dens": (0.0, 2.0),
}

# Sentinel for missing data. NaN would propagate through convolutions, so we
# use an explicit value plus a validity mask.
FILL_VALUE = -999.0

# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
# Binary target: at least this many strikes in the pixel during the 5 min
# window. 1 is the natural choice -- a single cloud-to-ground strike is a
# lethal event, so there is no justification for a higher bar.
LIGHTNING_THRESHOLD = 1

# Because a strike at pixel (i,j) and a strike at (i+1,j) are operationally
# the same warning, we evaluate with neighbourhood tolerance. These are the
# radii (in pixels) for the Fractions Skill Score.
FSS_RADII_PX = [0, 1, 2, 4, 8]

# Probability thresholds swept for contingency metrics.
PROB_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
SYNTH = {
    "seed": 20260905,
    "n_events": 60,             # independent storm days
    "frames_per_event": 48,     # 48 * 5 min = 4 h per event
    # Fraction of storm cells that are NON-ELECTRIFIED mimics. These are the
    # crux of the difficulty: heavy warm-rain showers and bright-band
    # stratiform produce high surface reflectivity but no charging-layer ice.
    # Without them the task is trivially separable on surface dBZ alone and
    # any model scores near-perfectly for the wrong reason.
    "mimic_fraction": 0.60,
    "cells_per_event": (2, 7),  # inclusive range
    # Per-event random steering flow, in px per frame (1 px = 2 km, 1 frame =
    # 5 min, so 1 px/frame = 24 km/h).
    #
    # CALIBRATION NOTE: an earlier version drew speed as
    # abs(normal(0,1)) * steering_px / 1.5, which gave a MEAN of only ~1.3
    # px/frame (~16 km/h) despite this constant reading 2.5. That is far too
    # slow for a thunderstorm and it accidentally removed the motion problem
    # from the benchmark: a 3x3 ConvLSTM propagates ~1 px/frame, so at 1.3
    # px/frame it can keep up and motion compensation buys nothing.
    #
    # Real values: Indian pre-monsoon squall lines (Kalbaisakhi nor'westers
    # over the eastern/northeastern plains) commonly move at 50-80 km/h, i.e.
    # 2-3.3 px/frame, and organised systems can exceed that. Monsoon-season
    # cells are slower, ~20-40 km/h. So the distribution must span roughly
    # 1-4 px/frame with a mean near 2.2. `steering_px` is now the MEAN speed
    # and the sampler uses a Rayleigh-like draw scaled to hit it.
    "steering_px": 2.2,
    "steering_px_max": 4.5,
    # Observation noise and sensor artefacts
    "radar_noise_db": 2.0,
    "ir_noise_k": 1.5,
    # LLN detection efficiency. Real networks miss strikes, especially
    # intracloud and at range. Modelling this prevents the target from being
    # a clean deterministic function of the physics channels.
    "lln_detection_eff": 0.85,
    # Probability that a whole channel is missing for an event, simulating
    # INSAT cadence gaps and radar outages.
    "channel_dropout_p": 0.10,
    # Anomalous propagation / ground clutter events -- high surface dBZ with
    # zero vertical extent. Trivially rejected IF the model uses echo_top,
    # which is the point.
    "ap_clutter_p": 0.15,
}

# ---------------------------------------------------------------------------
# Model hyperparameters, per backend
# ---------------------------------------------------------------------------
# The NumPy backend exists because pip/network are unavailable in some
# sandboxes. It is deliberately small: a hand-written ConvLSTM backward pass
# on 2 CPU cores cannot train a large model. The torch backend is used
# automatically when importable and is sized for a real GPU run.
HP_NUMPY = {
    "hidden": 16,
    "kernel": 3,
    "lr": 3e-3,
    "batch": 4,
    "epochs": 6,
    "grad_clip": 1.0,
    # Positive class weight. Base rate is ~1%, so unweighted BCE collapses to
    # predicting zero everywhere. We weight rather than resample because
    # resampling would break the temporal contiguity the ConvLSTM depends on.
    "pos_weight": 12.0,
    "downsample": 2,   # train on 32x32 for speed, evaluate at full res
}

HP_TORCH = {
    "hidden": 64,
    "kernel": 3,
    "lr": 1e-3,
    "batch": 8,
    "epochs": 40,
    "grad_clip": 1.0,
    "pos_weight": 12.0,
    "downsample": 1,
}

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
# Blocked by event with a temporal gap. Random sample-level splits would put
# frame t in train and frame t+5min in test -- these overlap in input window
# and are nearly identical, which leaks and inflates scores enormously.
SPLIT = {
    "train_frac": 0.65,
    "val_frac": 0.15,
    "test_frac": 0.20,
    # Frames discarded at each event boundary between splits.
    "gap_frames": INPUT_FRAMES + OUTPUT_FRAMES,
}

# ---------------------------------------------------------------------------
# Real data source registry
# ---------------------------------------------------------------------------
# Declarative description of each supported source so data_io can report what
# is available and what each one needs, without importing optional deps.
DATA_SOURCES = {
    "sevir": {
        "desc": "SEVIR: NEXRAD VIL + GOES IR/VIS + GLM lightning, 384x384 @ 1km, 5 min",
        "needs": ["h5py"],
        "auth": "none (AWS open data, s3://sevir)",
        "cadence_min": 5,
        "note": "Best public benchmark for this task. Already event-cropped.",
    },
    "nexrad": {
        "desc": "NEXRAD Level II raw radar volumes",
        "needs": ["boto3", "pyart"],
        "auth": "none (s3://noaa-nexrad-level2)",
        "cadence_min": 6,
        "note": "Needs gridding from polar to Cartesian; pyart does this.",
    },
    "goes_glm": {
        "desc": "GOES-16/18 GLM L2 lightning events/groups/flashes",
        "needs": ["netCDF4"],
        "auth": "none (s3://noaa-goes16)",
        "cadence_min": 0.33,
        "note": "20 s files. Accumulate to 5 min to match radar.",
    },
    "mosdac": {
        "desc": "MOSDAC INSAT-3D/3DR IR + IMD DWR products (India)",
        "needs": ["h5py", "requests"],
        "auth": "MOSDAC account token (register at mosdac.gov.in)",
        "cadence_min": 15,
        "note": (
            "INSAT-3D full-disk is 30 min, rapid-scan sector 15 min -- far "
            "coarser than GOES 1-5 min. The model must tolerate stale IR."
        ),
    },
    "imd_dwr": {
        "desc": "IMD Doppler Weather Radar reflectivity products",
        "needs": ["h5py"],
        "auth": "IMD data request / RSMC",
        "cadence_min": 10,
        "note": (
            "Coverage has real gaps over central India. Range degradation "
            "beyond ~150 km is severe; QC flags matter."
        ),
    },
    "openmeteo": {
        "desc": "Open-Meteo CAPE/CIN/LI reanalysis, for environmental context",
        "needs": ["requests"],
        "auth": "none for non-commercial",
        "cadence_min": 60,
        "note": "Not a nowcast source. Useful as a static per-event covariate.",
    },
}


def describe() -> str:
    """Human-readable configuration summary."""
    lines = []
    lines.append("LIGHTNING NOWCASTING CONFIGURATION")
    lines.append("=" * 62)
    lines.append(
        f"Grid          : {GRID_H}x{GRID_W} @ {PIXEL_KM} km "
        f"= {GRID_H * PIXEL_KM:.0f} x {GRID_W * PIXEL_KM:.0f} km"
    )
    lines.append(
        f"Timing        : {TIMESTEP_MIN} min steps, "
        f"{INPUT_FRAMES} in ({INPUT_FRAMES * TIMESTEP_MIN} min history) -> "
        f"{OUTPUT_FRAMES} out ({OUTPUT_FRAMES * TIMESTEP_MIN} min lead)"
    )
    lines.append(f"Channels      : {', '.join(CHANNELS)}")
    lines.append(
        f"Charging layer: {CHARGING_LOWER_KM}-{CHARGING_UPPER_KM} km "
        f"(-10C to -20C), threshold {CHARGING_DBZ_THRESHOLD} dBZ"
    )
    lines.append(f"Freezing level: {FREEZING_LEVEL_KM} km (warm tropical airmass)")
    lines.append(f"Target        : >= {LIGHTNING_THRESHOLD} strike / pixel / step")
    lines.append(f"FSS radii     : {FSS_RADII_PX} px")
    lines.append("")
    lines.append("DATA SOURCES")
    lines.append("-" * 62)
    for key, meta in DATA_SOURCES.items():
        lines.append(f"  {key:10s} {meta['desc']}")
        lines.append(f"  {'':10s} deps={meta['needs']} auth={meta['auth']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

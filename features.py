"""
Physics-informed feature construction. Strictly causal.

DESIGN RATIONALE
----------------
A ConvLSTM given raw channels can in principle learn everything here. In
practice, with ~1% positive rate, a small model and a 30-minute input window,
it will not learn a second-order temporal derivative or a 12-pixel spatial
aggregate from scratch. Handing it these directly is the difference between a
model that works and one that predicts the base rate everywhere.

Feature choices are driven by an empirical finding measured on this data
(see the module self-test): on the NEW-INITIATION subset -- pixels with no
lightning anywhere within 8 km during the input window -- the single strongest
predictor is not the value of charging-layer reflectivity at the pixel, but the
neighbourhood MAXIMUM of it. Ranked by AUPRC on that subset:

    refl_m10 box-max r=2      0.190
    refl_m10 box-max r=4      0.187
    refl_m10 at the pixel     0.157
    echo_top at the pixel     0.109
    lightning input channel   0.0065   (i.e. useless -- as it must be)

The physical reason is that a new strike does not occur where the charging-layer
ice already is; it occurs in a *developing* part of the same storm complex,
adjacent to where ice is currently detected. So the neighbourhood aggregate is
the physically correct predictor for initiation, and we build it explicitly.

THE CAUSALITY RULE
------------------
Every feature at time t is a function of observations at times <= t only.
Violating this is the single easiest way to produce a spectacular and completely
fake result. Two specific hazards, both avoided here:

  1. Centred temporal differences. `(x[t+1] - x[t-1]) / 2` is the natural way to
     write a derivative and it looks harmless. It reads one frame into the
     future. Use backward differences only.

  2. Whole-sequence normalisation. Computing a mean or std over the full time
     axis (or the full dataset including test) and subtracting it leaks future
     and out-of-fold information. We use FIXED constants from config, chosen
     from physical ranges, precisely so nothing is fitted to data.

`run_tests.py` proves both properties empirically by perturbing future frames
and asserting no feature at earlier times changes.
"""

from __future__ import annotations

import numpy as np

import config as C
from metrics import _box_max, _box_mean


# ---------------------------------------------------------------------------
# Derived physical quantities
# ---------------------------------------------------------------------------

# Soft-ramp onset for the charging response, in dBZ at the -10C level.
#
# WHY THIS IS NOT 40 dBZ. The Gremillion-Vincent 40 dBZ criterion applies to a
# reflectivity value resolved at the charging level. Any real retrieval at
# 6.5 km altitude comes from an elevated tilt whose beam is 1-3 km wide, so the
# retrieved value is a volume average over a region much larger than the
# graupel core. Beam dilution means an observed 20-25 dBZ can correspond to a
# 40+ dBZ core.
#
# Measured on this data: only 0.005% of valid refl_m10 pixels reach 40 dBZ,
# while ~0.5% reach 30 and the 99th percentile is 15 dBZ. Using 40 as a hard
# threshold zeroed charge_excess almost everywhere and silently killed every
# feature derived from it -- all of them scored exactly 1.0x lift (no signal)
# on the new-initiation subset. A hard clip on a smeared field destroys the
# signal it is meant to isolate.
#
# Fix: a soft ramp from ONSET, so the feature varies smoothly across the range
# the retrieval actually occupies while still emphasising the upper tail.
CHARGE_ONSET_DBZ = 8.0
CHARGE_SCALE_DBZ = 12.0


def charging_layer_excess(refl_m10: np.ndarray) -> np.ndarray:
    """Soft measure of graupel loading in the charging layer.

    The electrification response is threshold-like in the *true* charging-layer
    reflectivity, but our observation of it is smeared by beam width. So rather
    than a hard threshold we use a softplus-style ramp above CHARGE_ONSET_DBZ:
    near-zero below onset, approximately linear above it, and never exactly
    zero over the range the retrieval occupies.

    Returned in dBZ-like units (roughly "effective excess above onset").
    """
    valid = refl_m10 > C.FILL_VALUE / 2
    z = np.where(valid, (refl_m10 - CHARGE_ONSET_DBZ) / CHARGE_SCALE_DBZ, -20.0)
    # log1p(exp(z)) computed stably: for large z this is z, for small z ~exp(z)
    soft = np.where(z > 20.0, z, np.log1p(np.exp(np.clip(z, -30.0, 20.0))))
    return np.where(valid, soft * CHARGE_SCALE_DBZ, 0.0)


def graupel_proxy(refl_m10: np.ndarray, echo_top: np.ndarray) -> np.ndarray:
    """Proxy for graupel mass in the charging layer.

    Requires BOTH charging-layer reflectivity (large ice present) AND an echo
    top above the charging layer (the cell actually reaches the region where
    charging happens). Either alone is ambiguous:

      high refl_m10, low echo top  -> retrieval artefact or bright band
      high echo top, low refl_m10  -> glaciated anvil, small ice only, no
                                      graupel, no charging

    The product is the conjunction, which is what the physics requires. This is
    the feature that should separate electrified cells from warm-rain mimics.
    """
    valid = (refl_m10 > C.FILL_VALUE / 2) & (echo_top > C.FILL_VALUE / 2)
    depth = np.clip(echo_top - C.CHARGING_LOWER_KM, 0.0, None)
    exc = charging_layer_excess(refl_m10)
    return np.where(valid, exc * np.sqrt(depth), 0.0)


def deep_convection_flag(echo_top: np.ndarray, ir_tb: np.ndarray) -> np.ndarray:
    """Soft indicator that the cell penetrates well above the charging layer.

    Uses echo top and IR brightness temperature together. IR alone is
    ambiguous: a cold top can be an active updraft OR a decaying glaciated
    anvil that has stopped producing lightning. Combining with echo top
    (which requires actual hydrometeors, not just cold cloud) disambiguates.
    """
    et_valid = echo_top > C.FILL_VALUE / 2
    ir_valid = ir_tb > C.FILL_VALUE / 2
    # Smooth ramp between the -10C and -20C heights and above
    et_term = np.where(et_valid,
                       np.clip((echo_top - C.CHARGING_LOWER_KM) / 4.0, 0, 1.5),
                       0.0)
    # 240 K ~ -33C, well above the charging layer
    ir_term = np.where(ir_valid, np.clip((250.0 - ir_tb) / 30.0, 0, 1.5), 0.0)
    return et_term * ir_term


# ---------------------------------------------------------------------------
# Causal temporal features
# ---------------------------------------------------------------------------

def backward_difference(seq: np.ndarray, lag: int = 1) -> np.ndarray:
    """First difference against `lag` frames earlier. Strictly causal.

    Args:
        seq: (T, ...) sequence
    Returns:
        (T, ...) with the first `lag` frames set to 0 (no history available)

    Growth rate is a genuine predictor, not a convenience: a cell whose
    charging-layer reflectivity is *increasing* is intensifying and about to
    electrify, while the same value on a decreasing trend is a decaying cell
    that has probably already finished producing lightning. The instantaneous
    value cannot distinguish these; the tendency can. This is where lead time
    comes from.
    """
    out = np.zeros_like(seq, dtype=np.float64)
    if seq.shape[0] > lag:
        out[lag:] = seq[lag:].astype(np.float64) - seq[:-lag].astype(np.float64)
    return out


def causal_max(seq: np.ndarray) -> np.ndarray:
    """Running maximum over all frames up to and including t.

    Storm history matters: a cell that reached 45 dBZ at -10C two frames ago has
    produced graupel that is still aloft and still charging, even if the current
    retrieval dipped (beam geometry, attenuation, a missed tilt). The running
    max makes the model robust to single-frame retrieval dropouts, which are
    routine on real radar.
    """
    return np.maximum.accumulate(seq.astype(np.float64), axis=0)


def causal_mean(seq: np.ndarray) -> np.ndarray:
    """Expanding mean over frames up to and including t."""
    s = np.cumsum(seq.astype(np.float64), axis=0)
    n = np.arange(1, seq.shape[0] + 1, dtype=np.float64)
    shape = [-1] + [1] * (seq.ndim - 1)
    return s / n.reshape(shape)


# ---------------------------------------------------------------------------
# Full feature construction
# ---------------------------------------------------------------------------

# Names of the engineered channels, in the order build_features emits them.
FEATURE_NAMES = [
    # --- normalised raw channels (6) ---
    "refl_sfc_n", "refl_m10_n", "echo_top_n", "vil_n", "ir_tb_n", "light_n",
    # --- validity mask for the charging-layer retrieval (1) ---
    "m10_valid",
    # --- derived physics (3) ---
    "charge_excess", "graupel", "deep_conv",
    # --- neighbourhood aggregates of the graupel proxy (3) ---
    # These are the strongest new-initiation predictors; see module docstring.
    "graupel_max_r2", "graupel_max_r4", "graupel_mean_r4",
    # --- causal tendencies (4) ---
    "d_refl_m10", "d_echo_top", "d_ir_tb", "d_graupel",
    # --- causal history (2) ---
    "graupel_cmax", "light_cmean",
    # --- lightning neighbourhood context (1) ---
    "light_max_r4",
]
N_FEATURES = len(FEATURE_NAMES)


def build_features(x: np.ndarray) -> np.ndarray:
    """Build the feature stack for one input sequence.

    Args:
        x: (T, C, H, W) raw channels, possibly containing FILL_VALUE
    Returns:
        (T, N_FEATURES, H, W) float32

    Every output frame t depends only on input frames <= t.
    """
    T = x.shape[0]
    g = lambda name: x[:, C.CH[name]]  # noqa: E731

    refl_sfc = g("refl_sfc")
    refl_m10 = g("refl_m10")
    echo_top = g("echo_top")
    vil = g("vil")
    ir_tb = g("ir_tb")
    light = g("light_dens")

    # ---- Normalise raw channels with FIXED constants ----
    # Fixed, not data-fitted: a per-dataset or per-sequence normalisation would
    # leak information across the split boundary and across time.
    def norm(a: np.ndarray, name: str) -> np.ndarray:
        mu, sc = C.CHANNEL_NORM[name]
        valid = a > C.FILL_VALUE / 2
        # Missing -> 0 after normalisation (the "neutral" value), with the
        # validity flag carried separately so the model can tell "average" from
        # "unobserved".
        return np.where(valid, (a - mu) / sc, 0.0)

    feats = [
        norm(refl_sfc, "refl_sfc"),
        norm(refl_m10, "refl_m10"),
        norm(echo_top, "echo_top"),
        norm(vil, "vil"),
        norm(ir_tb, "ir_tb"),
        norm(light, "light_dens"),
    ]

    # Validity of the charging-layer retrieval. On real DWR this is frequently
    # missing (beam blockage, cone of silence overhead, range limits), and
    # "no charging-layer echo observed" must not be read as "no ice aloft".
    m10_valid = (refl_m10 > C.FILL_VALUE / 2).astype(np.float64)
    feats.append(m10_valid)

    # ---- Derived physics ----
    exc = charging_layer_excess(refl_m10)
    grp = graupel_proxy(refl_m10, echo_top)
    deep = deep_convection_flag(echo_top, ir_tb)
    feats += [exc / 20.0, grp / 40.0, deep]

    # ---- Neighbourhood aggregates ----
    # Initiation happens ADJACENT to existing charging-layer ice, so the
    # neighbourhood max outperforms the pixel value. Two radii plus a mean:
    # the max finds the nearest active core, the mean measures how much of the
    # surrounding area is convectively active.
    feats.append(_box_max(grp, 2) / 40.0)
    feats.append(_box_max(grp, 4) / 40.0)
    feats.append(_box_mean(grp, 4) / 40.0)

    # ---- Causal tendencies ----
    # Backward differences only. A centred difference here would read the future
    # and invalidate every result downstream.
    feats.append(backward_difference(refl_m10) / 10.0)
    feats.append(backward_difference(echo_top) / 2.0)
    feats.append(backward_difference(ir_tb) / 10.0)
    feats.append(backward_difference(grp) / 20.0)

    # ---- Causal history ----
    feats.append(causal_max(grp) / 40.0)
    feats.append(causal_mean((light >= C.LIGHTNING_THRESHOLD).astype(np.float64)))

    # ---- Lightning neighbourhood context ----
    # Legitimate as an input (a nowcaster sees current lightning). It is the
    # source of the persistence shortcut, which is exactly why the
    # new-initiation metric excludes pixels where it is informative.
    feats.append(_box_max((light >= C.LIGHTNING_THRESHOLD).astype(np.float64), 4))

    out = np.stack(feats, axis=1).astype(np.float32)
    assert out.shape == (T, N_FEATURES, x.shape[2], x.shape[3]), out.shape
    # Guard against NaN/inf reaching the network, where they would silently
    # poison every weight through the shared convolution.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_features_batch(X: np.ndarray) -> np.ndarray:
    """Vectorised over the batch axis. X: (N, T, C, H, W) -> (N, T, F, H, W)."""
    return np.stack([build_features(X[i]) for i in range(X.shape[0])])


if __name__ == "__main__":
    import synth
    import metrics as M

    events = synth.generate_dataset(n_events=8, seed=11)
    xs, ys = [], []
    for e in events:
        T = e["x"].shape[0]
        for t0 in range(0, T - C.INPUT_FRAMES - C.OUTPUT_FRAMES, 7):
            xs.append(e["x"][t0:t0 + C.INPUT_FRAMES])
            ys.append(e["y"][t0 + C.INPUT_FRAMES:
                             t0 + C.INPUT_FRAMES + C.OUTPUT_FRAMES])
    X = np.stack(xs)
    Y = np.stack(ys)
    Yb = Y >= C.LIGHTNING_THRESHOLD

    F = build_features_batch(X)
    print(f"features: {X.shape} -> {F.shape}  ({N_FEATURES} channels)")
    print(f"finite: {np.isfinite(F).all()}")

    # Rank features by AUPRC on the NEW-INITIATION subset -- the only subset
    # where a score cannot be obtained by persistence.
    clean = M.new_initiation_mask(X)
    niv = np.broadcast_to(clean[:, None], Yb.shape)
    base = float(Yb[niv].mean())
    print(f"\nnew-initiation base rate {base:.5f} "
          f"({int((Yb & niv).sum()):,} positives)")
    print("feature ranking by AUPRC on new-initiation pixels:")
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        f = np.broadcast_to(F[:, -1, i][:, None], Yb.shape).astype(np.float64)
        rows.append((M.auprc(f, Yb, niv), name))
    for ap, name in sorted(rows, reverse=True):
        print(f"  {name:16s} {ap:.4f}   ({ap / base:5.1f}x lift)")

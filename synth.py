"""
Physics-grounded synthetic storm generator.

PURPOSE
-------
Real DWR/LLN data is not reachable from this sandbox (no network, no h5py).
This module generates data in the *same interface* the real loaders produce,
so the entire pipeline -- features, model, metrics, leakage tests -- is
exercised and verified now, and real data drops in unchanged later.

THE CENTRAL DESIGN CONSTRAINT
-----------------------------
A synthetic generator is worthless if the task it creates is easy. The failure
mode (learned the hard way on a previous project) is that the generator ties
the label to some observable by construction, the model finds that shortcut,
scores 0.99, and the whole evaluation is meaningless.

Here the specific shortcut to avoid is: "high surface reflectivity => lightning".
If every strong echo is electrified, the model needs one channel and one
threshold. Real atmospheres are not like that, and India's especially is not:
with a freezing level near 5 km, warm-rain collision-coalescence efficiently
converts cloud water to rain *below* the freezing level. A tropical shower can
produce 50+ dBZ at the surface with almost no ice aloft and stay electrically
silent. Meanwhile a smaller but deeper cell that pushes graupel into the
-10 to -20C layer will electrify.

So we simulate two populations explicitly:

  ELECTRIFIED   deep updraft -> supercooled water above 0C level -> graupel in
                the charging layer -> non-inductive charging -> strikes
  MIMIC         warm-rain dominant, or bright-band stratiform, or ground
                clutter: strong low-level echo, negligible charging-layer ice,
                NO strikes

Separating them requires refl_m10 / echo_top / ir_tb, i.e. the vertical
structure. That is the actual physical discrimination task, and it is what we
want the model to have to learn.

WHAT IS DELIBERATELY IMPERFECT
------------------------------
  * LLN detection efficiency < 1: the target is a noisy observation of the
    latent charging state, not a deterministic function of it. Prevents the
    model from ever reaching perfect scores legitimately.
  * Stochastic charging delay: graupel presence does not instantly imply a
    strike. The lag is drawn per cell, so lead-time relationships are learnable
    in distribution but not exactly invertible.
  * Channel dropout: INSAT gaps and radar outages.
  * Observation noise on every channel.

None of these are cosmetic; each one closes a path to an unrealistically high
score.
"""

from __future__ import annotations

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gauss2d(h: int, w: int, cy: float, cx: float,
             sy: float, sx: float, theta: float = 0.0) -> np.ndarray:
    """Rotated 2-D Gaussian bump on a grid, peak 1.0.

    Used as the spatial footprint of a storm cell. Real cells are not
    Gaussian, but they are unimodal and elongated along the shear vector,
    which a rotated anisotropic Gaussian captures adequately.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dy = yy - cy
    dx = xx - cx
    ct, st = np.cos(theta), np.sin(theta)
    # Rotate into the cell's principal axes
    a = ct * dx + st * dy
    b = -st * dx + ct * dy
    return np.exp(-0.5 * ((a / max(sx, 1e-6)) ** 2 + (b / max(sy, 1e-6)) ** 2))


def _life_cycle(n: int, t_peak: float, growth: float, decay: float) -> np.ndarray:
    """Cell intensity envelope over time, in [0, 1].

    Asymmetric: convective cells grow faster than they decay (anvil and
    stratiform debris linger). Implemented as two exponentials joined at the
    peak, which is a reasonable first-order description of a pulse cell.
    """
    t = np.arange(n, dtype=np.float64)
    up = np.exp(-((t - t_peak) ** 2) / (2 * max(growth, 1e-6) ** 2))
    dn = np.exp(-((t - t_peak) ** 2) / (2 * max(decay, 1e-6) ** 2))
    return np.where(t <= t_peak, up, dn)


class _Cell:
    """One storm cell: kinematics, vertical structure, electrification state.

    The vertical structure is the whole point. Each cell carries an
    `ice_efficiency` describing how much of its condensate reaches the
    charging layer as graupel. Electrified cells have high values, mimics
    near-zero, and this is *not* correlated with surface reflectivity.
    """

    def __init__(self, rng: np.random.Generator, n_frames: int,
                 kind: str, steering: tuple[float, float]):
        self.kind = kind  # "electrified" | "warm_rain" | "stratiform" | "clutter"

        h, w = C.GRID_H, C.GRID_W
        # Start position, biased inward so cells spend time in-domain
        self.y0 = rng.uniform(0.15 * h, 0.85 * h)
        self.x0 = rng.uniform(0.15 * w, 0.85 * w)

        # Motion: shared event steering plus per-cell deviation. Clutter does
        # not move -- that is its defining signature and a legitimate cue.
        if kind == "clutter":
            self.vy, self.vx = 0.0, 0.0
        else:
            self.vy = steering[0] + rng.normal(0, 0.4)
            self.vx = steering[1] + rng.normal(0, 0.4)

        # Life cycle timing
        self.t_peak = rng.uniform(0.2 * n_frames, 0.8 * n_frames)
        if kind == "stratiform":
            # Broad, long-lived, slowly varying
            self.growth = rng.uniform(8, 16)
            self.decay = rng.uniform(12, 24)
        elif kind == "clutter":
            self.growth = rng.uniform(20, 40)
            self.decay = rng.uniform(20, 40)
        else:
            self.growth = rng.uniform(3, 7)
            self.decay = rng.uniform(5, 11)
        self.env = _life_cycle(n_frames, self.t_peak, self.growth, self.decay)

        # Spatial extent (px). Stratiform is broad and weak, convective is
        # compact and intense.
        if kind == "stratiform":
            self.sy = rng.uniform(10, 18)
            self.sx = rng.uniform(12, 22)
        elif kind == "clutter":
            self.sy = rng.uniform(2, 5)
            self.sx = rng.uniform(2, 5)
        else:
            self.sy = rng.uniform(3, 7)
            self.sx = rng.uniform(3, 8)
        self.theta = rng.uniform(0, np.pi)

        # ---- Vertical structure: the discriminating physics ----
        #
        # peak_dbz_sfc  surface reflectivity at maturity
        # ice_efficiency  fraction of condensate reaching charging layer as
        #                 graupel; drives refl_m10 and hence electrification
        # top_km        18 dBZ echo top at maturity
        #
        # Note carefully that peak_dbz_sfc RANGES OVERLAP between electrified
        # and warm_rain. That overlap is intentional and is what makes surface
        # dBZ alone insufficient.
        # CRITICAL: electrified and warm_rain draw surface reflectivity from
        # the SAME distribution. Any gap here would let the model separate the
        # two populations on surface dBZ alone, which is precisely the physical
        # error we are trying to force it to avoid. In the real atmosphere a
        # 55 dBZ warm-rain shower and a 55 dBZ electrified cell are genuinely
        # indistinguishable at the lowest tilt; the difference is aloft.
        if kind == "electrified":
            self.peak_dbz_sfc = rng.uniform(38, 62)
            self.ice_efficiency = rng.uniform(0.55, 0.95)
            self.top_km = rng.uniform(8.0, 16.0)
        elif kind == "warm_rain":
            # Heavy tropical shower: strong surface echo but warm-rain
            # dominated, so little of the condensate arrives in the charging
            # layer as graupel. Same surface dBZ range as electrified, and an
            # OVERLAPPING echo-top range: a monsoon cell can be 9 km deep and
            # still be a warm-rain machine whose condensate is depleted below
            # the freezing level. Overlap is required, otherwise echo_top alone
            # separates the populations perfectly and stands in for the physics.
            self.peak_dbz_sfc = rng.uniform(38, 62)
            self.ice_efficiency = rng.uniform(0.00, 0.14)
            self.top_km = rng.uniform(4.5, 10.5)
        elif kind == "stratiform":
            # Bright band: enhanced reflectivity right at the melting level
            # from aggregates coated in meltwater. Looks moderately strong,
            # has ice, but no graupel and no strong updraft -> no charging.
            self.peak_dbz_sfc = rng.uniform(26, 40)
            self.ice_efficiency = rng.uniform(0.05, 0.22)
            self.top_km = rng.uniform(7.0, 10.0)
        else:  # clutter (anomalous propagation, ground returns)
            self.peak_dbz_sfc = rng.uniform(35, 55)
            self.ice_efficiency = 0.0
            self.top_km = rng.uniform(0.3, 1.2)

        # Charging delay: time from graupel appearing in the charging layer to
        # the first strike, in frames (5 min each). Physically 1-3 frames.
        # Drawn per cell so the model must learn a distribution, not a constant.
        self.charge_delay = int(rng.integers(1, 4))

        # Per-cell breakdown threshold. Even with graupel present, charge must
        # accumulate past a threshold that varies with cell geometry, updraft
        # width and aerosol loading -- NONE of which radar observes. Drawing it
        # over a wide range is what keeps charging-layer reflectivity a
        # probabilistic rather than deterministic predictor.
        #
        # This matters: without it, refl_m10 becomes a near-perfect single
        # feature (AUROC ~0.98) and the model has nothing to learn beyond one
        # threshold. Real charging-layer reflectivity-to-flash-rate relations
        # scatter by an order of magnitude in flash rate at fixed dBZ.
        self.breakdown = rng.uniform(0.18, 1.05)

        # Unobservable multiplicative efficiency on the charging reaction
        # itself (mixed-phase microphysics detail radar cannot resolve).
        self.charge_gain = float(np.exp(rng.normal(0.0, 0.55)))

        # Flash rate scaling at maturity (strikes per 5 min in the core).
        self.flash_scale = rng.uniform(3.0, 22.0)

    def position(self, t: int) -> tuple[float, float]:
        return self.y0 + self.vy * t, self.x0 + self.vx * t

    def footprint(self, t: int) -> np.ndarray:
        cy, cx = self.position(t)
        # Cells expand as they mature (anvil spreading)
        grow = 1.0 + 0.45 * self.env[t]
        return _gauss2d(C.GRID_H, C.GRID_W, cy, cx,
                        self.sy * grow, self.sx * grow, self.theta)


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def generate_event(rng: np.random.Generator, n_frames: int | None = None,
                   mimic_fraction: float | None = None) -> dict:
    """Generate one storm event.

    Returns a dict with:
        x     (T, C, H, W) float32  observed channels, with FILL_VALUE gaps
        mask  (T, C, H, W) bool     True where observed
        y     (T, H, W)    float32  strike counts per pixel per frame
        meta  dict                  cell inventory, for diagnostics only

    The returned `y` is what the model predicts (binarised downstream). The
    latent charging state is deliberately NOT returned -- a model must infer
    it from the observable channels, which is the real problem.
    """
    n_frames = n_frames or C.SYNTH["frames_per_event"]
    mimic_fraction = (C.SYNTH["mimic_fraction"] if mimic_fraction is None
                      else mimic_fraction)
    h, w = C.GRID_H, C.GRID_W

    # Event-level steering flow. All cells share it -> coherent motion field,
    # which is what makes optical-flow advection a genuinely strong baseline.
    #
    # Rayleigh draw (magnitude of a 2-D Gaussian wind perturbation), which is
    # the natural distribution for wind speed and gives a realistic right tail
    # of fast-moving squall lines. Scaled so the MEAN equals SYNTH["steering_px"]:
    # for a Rayleigh with parameter s, mean = s*sqrt(pi/2), so s = mean/1.2533.
    mean_spd = C.SYNTH["steering_px"]
    s = mean_spd / np.sqrt(np.pi / 2.0)
    spd = float(np.clip(rng.rayleigh(s), 0.0, C.SYNTH["steering_px_max"]))
    ang = rng.uniform(0, 2 * np.pi)
    steering = (spd * np.sin(ang), spd * np.cos(ang))

    lo, hi = C.SYNTH["cells_per_event"]
    n_cells = int(rng.integers(lo, hi + 1))

    cells: list[_Cell] = []
    for _ in range(n_cells):
        if rng.random() < mimic_fraction:
            # Split the mimic population across the three non-electrified modes
            r = rng.random()
            kind = "warm_rain" if r < 0.5 else ("stratiform" if r < 0.85 else "clutter")
        else:
            kind = "electrified"
        cells.append(_Cell(rng, n_frames, kind, steering))

    # Guarantee at least one electrified cell in ~80% of events, and make ~20%
    # of events entirely non-electrified. Null events matter: a model must be
    # able to say "no lightning anywhere today", and if every event contains
    # lightning the model learns an unconditional positive prior.
    has_elec = any(c.kind == "electrified" for c in cells)
    if not has_elec and rng.random() < 0.8:
        cells.append(_Cell(rng, n_frames, "electrified", steering))

    # Accumulators
    refl_sfc = np.zeros((n_frames, h, w), np.float64)
    refl_m10 = np.zeros((n_frames, h, w), np.float64)
    echo_top = np.zeros((n_frames, h, w), np.float64)
    vil = np.zeros((n_frames, h, w), np.float64)
    # Latent graupel mass in the charging layer, per cell, for delay handling
    charge_layer = np.zeros((n_frames, h, w), np.float64)
    strikes = np.zeros((n_frames, h, w), np.float64)

    for cell in cells:
        for t in range(n_frames):
            e = cell.env[t]
            if e < 0.02:
                continue
            fp = cell.footprint(t)

            # Surface reflectivity: linear in envelope, in dBZ space.
            # Reflectivity in dBZ is logarithmic so additive combination of
            # cells is wrong in principle; we combine with max() below, which
            # is what radar composites effectively do.
            sfc = cell.peak_dbz_sfc * e * fp
            refl_sfc[t] = np.maximum(refl_sfc[t], sfc)

            # Charging-layer reflectivity. This is the physics: it is the
            # surface echo scaled by how efficiently this cell transports
            # condensate into the -10 to -20C layer as graupel. For warm-rain
            # cells the factor is near zero regardless of how strong the
            # surface echo is.
            m10 = cell.peak_dbz_sfc * e * fp * cell.ice_efficiency
            refl_m10[t] = np.maximum(refl_m10[t], m10)

            # Echo top follows the envelope but saturates
            top = cell.top_km * (0.45 + 0.55 * e) * (fp > 0.25)
            echo_top[t] = np.maximum(echo_top[t], top)

            # VIL scales with the column integral; roughly dBZ^(4/7) * depth
            v = 0.035 * (cell.peak_dbz_sfc * e) ** 1.35 * fp * min(cell.top_km / 8.0, 1.6)
            vil[t] = np.maximum(vil[t], v)

            # Latent charging: needs graupel in the layer. Expressed as a
            # normalised quantity so `breakdown` is comparable across cells.
            # `charge_gain` is the unobservable microphysical efficiency, so
            # the mapping from observed refl_m10 to charging state is
            # many-to-one and cannot be inverted exactly from radar.
            cl = (m10 / 60.0) * e * cell.charge_gain
            charge_layer[t] = np.maximum(charge_layer[t], cl)

            # ---- Electrification with delay ----
            # A strike at time t requires the charging layer to have been
            # active at t - charge_delay. This is the causal lag that makes
            # 0-30 min nowcasting physically possible: the predictor
            # (charging-layer ice) is observable BEFORE the consequence.
            ts = t + cell.charge_delay
            if ts < n_frames and cell.ice_efficiency > 0.2:
                active = cl > cell.breakdown * 0.35
                if active.any():
                    # Flash rate rises steeply once past breakdown -- flash
                    # rate scales roughly with updraft volume^2 in
                    # observations, hence the square.
                    excess = np.clip(cl - cell.breakdown * 0.35, 0, None)
                    rate = cell.flash_scale * (excess ** 2) * 12.0
                    lam = np.where(active, rate, 0.0)
                    strikes[ts] += rng.poisson(np.clip(lam, 0, 400))

    # ---- IR brightness temperature ----
    # Cold tops = deep convection. Derived from echo top via a lapse rate,
    # with a warm background. Note IR alone cannot distinguish a glaciated
    # anvil (cold, no longer electrified) from an active updraft, which is a
    # real ambiguity we want preserved.
    sfc_temp = 300.0
    lapse = 6.5  # K/km
    ir_tb = sfc_temp - lapse * echo_top
    # Anvil spreading: smooth the IR field, cloud tops are broader than echo
    for t in range(n_frames):
        ir_tb[t] = _box_blur(ir_tb[t], 3)
    ir_tb = np.clip(ir_tb, 190.0, 305.0)

    # ---- LLN observation: imperfect detection ----
    # Real networks miss strikes. Applying binomial thinning means the target
    # is a noisy realisation, so perfect prediction is impossible in principle.
    eff = C.SYNTH["lln_detection_eff"]
    observed = rng.binomial(strikes.astype(np.int64), eff).astype(np.float64)

    # ---- Beam broadening on the charging-layer retrieval ----
    # refl_m10 is NOT a direct measurement. To sample 6.5 km altitude the radar
    # must use an elevated tilt, where the beam is much wider than at the
    # lowest tilt (a 1 deg beam is ~1.7 km across at 100 km range and grows
    # linearly). The retrieval is therefore spatially smeared, and interpolating
    # between discrete tilts adds further error. Without this, refl_m10 is an
    # unrealistically clean oracle for the latent charging state.
    for t in range(n_frames):
        refl_m10[t] = _box_blur(refl_m10[t], 5)
    # Elevated-tilt returns are also noisier than the surface scan.
    refl_m10 += rng.normal(0, C.SYNTH["radar_noise_db"] * 2.0, refl_m10.shape)

    # ---- Observation noise ----
    refl_sfc += rng.normal(0, C.SYNTH["radar_noise_db"], refl_sfc.shape)
    ir_tb += rng.normal(0, C.SYNTH["ir_noise_k"], ir_tb.shape)
    vil += rng.normal(0, 0.3, vil.shape)
    echo_top += rng.normal(0, 0.2, echo_top.shape)

    refl_sfc = np.clip(refl_sfc, 0, 75)
    refl_m10 = np.clip(refl_m10, 0, 75)
    vil = np.clip(vil, 0, 70)
    echo_top = np.clip(echo_top, 0, 20)

    # Lightning density channel is the OBSERVED strikes at the current frame.
    # This is legitimate as an input (a nowcaster sees current lightning) but
    # it is also the source of the persistence shortcut, which is why
    # metrics.py scores new-initiation pixels separately.
    x = np.stack([refl_sfc, refl_m10, echo_top, vil, ir_tb, observed], axis=1)
    x = x.astype(np.float32)

    mask = np.ones_like(x, dtype=bool)

    # ---- Channel dropout ----
    # Simulates INSAT cadence gaps (IR stale/missing) and radar outages.
    for ci in range(C.N_CHANNELS):
        if rng.random() < C.SYNTH["channel_dropout_p"]:
            t0 = int(rng.integers(0, max(1, n_frames - 6)))
            t1 = min(n_frames, t0 + int(rng.integers(2, 8)))
            x[t0:t1, ci] = C.FILL_VALUE
            mask[t0:t1, ci] = False

    meta = {
        "n_cells": len(cells),
        "kinds": [c.kind for c in cells],
        "n_electrified": sum(c.kind == "electrified" for c in cells),
        "steering": steering,
        "total_strikes": float(observed.sum()),
    }
    return {"x": x, "mask": mask, "y": observed.astype(np.float32), "meta": meta}


def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur via cumulative sums. No scipy available."""
    if k <= 1:
        return a
    pad = k // 2
    out = a.astype(np.float64)
    for axis in (0, 1):
        p = np.pad(out, [(pad + 1, pad) if i == axis else (0, 0)
                         for i in range(2)], mode="edge")
        cs = np.cumsum(p, axis=axis)
        sl_hi = [slice(None)] * 2
        sl_lo = [slice(None)] * 2
        n = out.shape[axis]
        sl_hi[axis] = slice(k, k + n)
        sl_lo[axis] = slice(0, n)
        out = (cs[tuple(sl_hi)] - cs[tuple(sl_lo)]) / k
    return out


def generate_dataset(n_events: int | None = None, seed: int | None = None,
                     mimic_fraction: float | None = None,
                     frames_per_event: int | None = None) -> list[dict]:
    """Generate a list of events. Each event is one independent 'storm day'."""
    n_events = n_events or C.SYNTH["n_events"]
    seed = C.SYNTH["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    return [generate_event(rng, n_frames=frames_per_event,
                           mimic_fraction=mimic_fraction)
            for _ in range(n_events)]


def dataset_stats(events: list[dict]) -> dict:
    """Summary statistics, used to sanity-check difficulty.

    WHY THE HEADLINE AUROC IS MISLEADING HERE
    -----------------------------------------
    With a ~1% base rate, the overwhelming majority of negative pixels are
    clear air. An unconditional AUROC therefore mostly measures "can you tell
    a storm from empty sky", which is trivial and not the operational problem.
    Surface dBZ scores ~0.97 on that comparison even when it carries no
    information about electrification at all.

    The operationally meaningful question is CONDITIONAL: given that a storm
    is present in this pixel, will it produce lightning? That is the question
    a forecaster actually faces, and it is where false alarms come from. So the
    numbers to trust are the `*_stormy` variants, computed only over pixels
    with a meaningful echo (>= 30 dBZ surface).

    The key numbers to watch:
      base_rate            should be ~0.005-0.03.
      auc_sfc_stormy       AUROC of surface dBZ, restricted to stormy pixels.
                           Should be near 0.5-0.7. If high, the generator has
                           a surface-reflectivity shortcut and mimics are not
                           working.
      auc_m10_stormy       Same for charging-layer dBZ. Should be clearly
                           higher -- that gap IS the physics signal.
      physics_gap_stormy   auc_m10_stormy - auc_sfc_stormy. Want >= ~0.15.
    """
    y = np.concatenate([e["y"].ravel() for e in events])
    pos = (y >= C.LIGHTNING_THRESHOLD)

    sfc = np.concatenate([e["x"][:, C.CH["refl_sfc"]].ravel() for e in events])
    m10 = np.concatenate([e["x"][:, C.CH["refl_m10"]].ravel() for e in events])
    top = np.concatenate([e["x"][:, C.CH["echo_top"]].ravel() for e in events])
    valid = (sfc > C.FILL_VALUE / 2) & (m10 > C.FILL_VALUE / 2)

    # Storm-conditional subset: exclude clear air, keep the hard comparison.
    stormy = valid & (sfc >= 30.0)

    out = {
        "n_events": len(events),
        "n_frames": int(sum(e["y"].shape[0] for e in events)),
        "n_pixels": int(y.size),
        "base_rate": float(pos.mean()),
        "mean_strikes_per_frame": float(
            np.mean([e["y"].sum(axis=(1, 2)).mean() for e in events])),
        "frac_events_with_lightning": float(
            np.mean([e["y"].sum() > 0 for e in events])),
        # Unconditional -- reported for completeness, do NOT use to judge
        # difficulty (see docstring).
        "auc_sfc_all": _fast_auc(sfc[valid], pos[valid]),
        "auc_m10_all": _fast_auc(m10[valid], pos[valid]),
        # Storm-conditional -- these are the meaningful ones.
        "n_stormy_px": int(stormy.sum()),
        "base_rate_stormy": float(pos[stormy].mean()) if stormy.any() else float("nan"),
        "auc_sfc_stormy": _fast_auc(sfc[stormy], pos[stormy]),
        "auc_m10_stormy": _fast_auc(m10[stormy], pos[stormy]),
        "auc_top_stormy": _fast_auc(top[stormy], pos[stormy]),
    }
    out["physics_gap_stormy"] = out["auc_m10_stormy"] - out["auc_sfc_stormy"]
    kinds: dict[str, int] = {}
    for e in events:
        for k in e["meta"]["kinds"]:
            kinds[k] = kinds.get(k, 0) + 1
    out["cell_kinds"] = kinds
    return out


def _fast_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC. Subsamples for speed on large arrays."""
    if label.sum() == 0 or label.sum() == label.size:
        return float("nan")
    if score.size > 2_000_000:
        idx = np.random.default_rng(0).choice(score.size, 2_000_000, replace=False)
        score, label = score[idx], label[idx]
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    # Average ranks within ties
    s_sorted = score[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos = float(label.sum())
    n_neg = float(label.size - n_pos)
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


if __name__ == "__main__":
    ev = generate_dataset(n_events=12)
    st = dataset_stats(ev)
    print("SYNTHETIC DATASET STATISTICS")
    print("=" * 52)
    for k, v in st.items():
        print(f"  {k:28s} {v}")
    print()
    print("Difficulty check (STORM-CONDITIONAL -- the meaningful comparison)")
    print("-" * 52)
    print(f"  stormy pixels          : {st['n_stormy_px']:,}")
    print(f"  base rate | stormy     : {st['base_rate_stormy']:.3f}")
    print(f"  surface-dBZ AUROC      : {st['auc_sfc_stormy']:.3f}  (want < 0.75)")
    print(f"  charging-dBZ AUROC     : {st['auc_m10_stormy']:.3f}  (want > surface)")
    print(f"  echo-top AUROC         : {st['auc_top_stormy']:.3f}")
    print(f"  physics gap            : {st['physics_gap_stormy']:+.3f}  (want >= +0.15)")
    print()
    print("Unconditional (inflated by clear-air negatives, do not use):")
    print(f"  surface-dBZ AUROC      : {st['auc_sfc_all']:.3f}")

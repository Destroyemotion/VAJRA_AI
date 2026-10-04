"""
Baselines. These define what "skill" means for this problem.

WHY BASELINES ARE THE MOST IMPORTANT FILE HERE
----------------------------------------------
A CSI of 0.35 at 30 min lead sounds like a result. It is meaningless in
isolation. The question is always: better than what?

Nowcasting has a brutal property -- the trivial baselines are strong. Weather
at t+30min looks a great deal like weather at t, translated. Published deep
learning nowcasters have repeatedly been shown to add modest skill over
optical-flow extrapolation, and several have been found to add none once
evaluated properly. So we implement the honest competition:

  1. PERSISTENCE (Eulerian). Copy the last observed lightning field forward,
     unchanged. Zero physics, zero motion. This is the absolute floor.

  2. LAGRANGIAN PERSISTENCE. Estimate the motion field from the reflectivity
     sequence with optical flow, then advect the lightning field along it.
     This is what operational radar extrapolation nowcasting actually is
     (SCIT, TITAN, TREC all reduce to sophisticated versions of this). THIS is
     the bar. Beating persistence is trivial; beating Lagrangian persistence is
     the claim worth making.

  3. GREMILLION-VINCENT threshold rule. The operational meteorologist's rule:
     >= 40 dBZ at the -10C level implies lightning within ~5-15 min. Pure
     physics, one threshold, no learning. Advected forward to give it a fair
     chance at longer leads. If a deep model cannot beat this, the model has
     learned nothing the physics does not already say.

  4. CLIMATOLOGY. Constant probability = base rate everywhere. Establishes the
     AUPRC floor and validates that the probabilistic metrics behave.

A deep model must beat 2 AND 3, on the aggregate metrics AND on new
initiation, before any claim of value is defensible.
"""

from __future__ import annotations

import numpy as np

import config as C
from metrics import _box_max, _box_mean

# cv2 is available in this environment and provides Farneback dense optical
# flow. If it is missing we fall back to a cross-correlation motion estimate,
# which is cruder but keeps the baseline honest rather than absent.
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# 1. Eulerian persistence
# ---------------------------------------------------------------------------

def persistence(x_input: np.ndarray, n_out: int,
                smooth_radius: int = 1) -> np.ndarray:
    """Copy the last lightning frame forward.

    Args:
        x_input: (N, T_in, C, H, W)
        n_out:   number of forecast steps
    Returns:
        (N, n_out, H, W) probabilities

    A raw binary copy would be a terrible *probabilistic* forecast (only 0 or 1,
    so it is guaranteed badly calibrated and its AUPRC is degenerate). We apply
    a light spatial smoothing to turn it into a probability field, which is the
    strongest fair version of this baseline. Handicapping a baseline to make the
    model look good is self-deception.
    """
    lt = x_input[:, -1, C.CH["light_dens"]]              # (N, H, W)
    occ = (lt >= C.LIGHTNING_THRESHOLD).astype(np.float64)
    prob = _box_mean(occ, smooth_radius)
    prob = np.clip(prob, 0.0, 1.0)
    return np.repeat(prob[:, None], n_out, axis=1)


# ---------------------------------------------------------------------------
# Motion estimation
# ---------------------------------------------------------------------------

def estimate_motion(seq: np.ndarray) -> tuple[float, float]:
    """Estimate a single domain-mean motion vector (dy, dx) px/frame.

    Args:
        seq: (T, H, W) reflectivity sequence

    A single global vector is a simplification -- real storms have differential
    motion and rotation. But for a 128 km domain over 60 min it captures most
    of the displacement, and a global estimate is far more robust than a dense
    field when the echo is sparse. `estimate_motion_dense` provides the dense
    version for the advection baseline.
    """
    if seq.shape[0] < 2:
        return 0.0, 0.0
    vecs = []
    for t in range(1, seq.shape[0]):
        a, b = seq[t - 1], seq[t]
        if _HAS_CV2:
            a8 = _to_u8(a)
            b8 = _to_u8(b)
            flow = cv2.calcOpticalFlowFarneback(
                a8, b8, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            # Weight the flow by echo intensity: flow in clear air is noise.
            w = np.maximum(b - 15.0, 0.0)
            if w.sum() < 1e-6:
                continue
            dx = float((flow[..., 0] * w).sum() / w.sum())
            dy = float((flow[..., 1] * w).sum() / w.sum())
        else:
            dy, dx = _xcorr_shift(a, b)
        vecs.append((dy, dx))
    if not vecs:
        return 0.0, 0.0
    v = np.asarray(vecs)
    # Median over frame pairs, robust to a single bad estimate.
    return float(np.median(v[:, 0])), float(np.median(v[:, 1]))


def _to_u8(a: np.ndarray) -> np.ndarray:
    """Scale a dBZ field to uint8 for cv2. Missing data -> 0."""
    b = np.where(a > C.FILL_VALUE / 2, a, 0.0)
    b = np.clip((b - 0.0) / 70.0 * 255.0, 0, 255)
    return b.astype(np.uint8)


def _xcorr_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 6
                 ) -> tuple[float, float]:
    """Integer-shift cross-correlation fallback when cv2 is unavailable."""
    a = np.where(a > C.FILL_VALUE / 2, a, 0.0)
    b = np.where(b > C.FILL_VALUE / 2, b, 0.0)
    best, best_v = (0.0, 0.0), -np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = _shift_field(a, dy, dx)
            v = float((shifted * b).sum())
            if v > best_v:
                best_v, best = v, (float(dy), float(dx))
    return best


def _shift_field(a: np.ndarray, dy: float, dx: float,
                 fill: float = 0.0) -> np.ndarray:
    """Shift with bilinear interpolation, zero-filled at the boundary.

    Boundary handling matters: edge-replication would smear storm echoes across
    the whole downstream edge and create spurious skill there. Zero-fill means
    "unknown -> no event", which is the honest choice for an advection forecast
    of an area that has moved in from outside the domain.
    """
    h, w = a.shape
    y = np.arange(h)[:, None] - dy
    x = np.arange(w)[None, :] - dx
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    fy = y - y0
    fx = x - x0

    def _g(yy, xx):
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yyc = np.clip(yy, 0, h - 1)
        xxc = np.clip(xx, 0, w - 1)
        return np.where(ok, a[yyc, xxc], fill)

    return (_g(y0, x0) * (1 - fy) * (1 - fx)
            + _g(y0 + 1, x0) * fy * (1 - fx)
            + _g(y0, x0 + 1) * (1 - fy) * fx
            + _g(y0 + 1, x0 + 1) * fy * fx)


# ---------------------------------------------------------------------------
# 2. Lagrangian persistence (optical-flow advection)
# ---------------------------------------------------------------------------

def advection(x_input: np.ndarray, n_out: int,
              smooth_radius: int = 1,
              decay_per_step: float = 0.97) -> np.ndarray:
    """Advect the last lightning field along the estimated motion.

    This is the real bar to beat. Args/returns as `persistence`.

    `decay_per_step` slowly reduces probability with lead time, reflecting that
    confidence in an extrapolated field decays. This makes the baseline better
    calibrated (and therefore harder to beat), which is the point.
    """
    n = x_input.shape[0]
    out = np.zeros((n, n_out, C.GRID_H, C.GRID_W), np.float64)
    refl_ch = C.CH["refl_sfc"]
    lt_ch = C.CH["light_dens"]

    for i in range(n):
        refl_seq = x_input[i, :, refl_ch]
        dy, dx = estimate_motion(refl_seq)
        occ = (x_input[i, -1, lt_ch] >= C.LIGHTNING_THRESHOLD).astype(np.float64)
        base = _box_mean(occ, smooth_radius)
        for t in range(n_out):
            step = t + 1
            adv = _shift_field(base, dy * step, dx * step, fill=0.0)
            # Growing dispersion with lead time: position uncertainty increases,
            # so the probability field should broaden. Radius grows ~sqrt(t),
            # which matches how displacement error accumulates for a random-walk
            # error in the motion vector.
            r = int(round(np.sqrt(step)))
            if r > 0:
                adv = _box_mean(adv, r)
            out[i, t] = np.clip(adv * (decay_per_step ** step), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 3. Gremillion-Vincent charging-layer threshold rule
# ---------------------------------------------------------------------------

def charging_layer_rule(x_input: np.ndarray, n_out: int,
                        dbz_threshold: float | None = None,
                        advect: bool = True,
                        dilate: int = 2) -> np.ndarray:
    """Operational physics rule: >= 40 dBZ at the -10C level implies lightning.

    Reference: Gremillion & Vincent (1998), Wea. Forecasting -- the 40 dBZ at
    -10C criterion gave the best combination of POD and lead time for
    Florida thunderstorms and is still used operationally.

    Args:
        advect: if True, extrapolate the diagnosed region along the motion
                field, giving the rule a fair chance at longer lead times
                rather than crippling it at t=0.
        dilate: spatial dilation radius. A strike does not occur exactly at the
                reflectivity maximum; it occurs somewhere in the cell.

    This baseline is important for a specific reason: it is the null hypothesis
    that *there is nothing to learn*. All the physics is one threshold on one
    derived channel. A deep model earns its complexity only by beating this.
    """
    thr = C.CHARGING_DBZ_THRESHOLD if dbz_threshold is None else dbz_threshold
    n = x_input.shape[0]
    out = np.zeros((n, n_out, C.GRID_H, C.GRID_W), np.float64)
    m10_ch = C.CH["refl_m10"]
    refl_ch = C.CH["refl_sfc"]

    for i in range(n):
        m10 = x_input[i, -1, m10_ch]
        # Missing charging-layer data -> cannot diagnose. Emit zero rather than
        # treating FILL_VALUE as a very low reflectivity, which would be a
        # confident "no lightning" on the basis of no observation.
        valid = m10 > C.FILL_VALUE / 2
        diag = (m10 >= thr) & valid
        field = _box_max(diag[None], dilate)[0].astype(np.float64)
        field = _box_mean(field, 1)

        if advect:
            dy, dx = estimate_motion(x_input[i, :, refl_ch])
        else:
            dy = dx = 0.0

        for t in range(n_out):
            step = t + 1
            f = _shift_field(field, dy * step, dx * step, fill=0.0) if advect else field
            r = int(round(np.sqrt(step)))
            if r > 0:
                f = _box_mean(f, r)
            # The rule's own skill decays: it diagnoses current charging state,
            # which becomes progressively less relevant as lead time grows.
            out[i, t] = np.clip(f * (0.95 ** step), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 4. Climatology
# ---------------------------------------------------------------------------

def climatology(x_input: np.ndarray, n_out: int,
                base_rate: float = 0.015) -> np.ndarray:
    """Constant base-rate probability everywhere.

    AUPRC of this baseline must equal the base rate. If it does not, the metric
    implementation is wrong -- so this doubles as a metric sanity check.
    """
    n = x_input.shape[0]
    return np.full((n, n_out, C.GRID_H, C.GRID_W), base_rate, np.float64)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES = {
    "climatology": climatology,
    "persistence": persistence,
    "advection": advection,
    "charging_rule": charging_layer_rule,
}


def run_all(x_input: np.ndarray, n_out: int,
            base_rate: float = 0.015) -> dict[str, np.ndarray]:
    """Run every baseline, returning name -> (N, n_out, H, W) probabilities."""
    out = {}
    for name, fn in BASELINES.items():
        if name == "climatology":
            out[name] = fn(x_input, n_out, base_rate=base_rate)
        else:
            out[name] = fn(x_input, n_out)
    return out


if __name__ == "__main__":
    import synth
    import metrics as M

    print("Generating a small dataset to smoke-test baselines...")
    events = synth.generate_dataset(n_events=6, seed=11)

    # Build a few forecast samples
    xs, ys = [], []
    for e in events:
        T = e["x"].shape[0]
        for t0 in range(0, T - C.INPUT_FRAMES - C.OUTPUT_FRAMES, 9):
            xs.append(e["x"][t0:t0 + C.INPUT_FRAMES])
            ys.append(e["y"][t0 + C.INPUT_FRAMES:
                             t0 + C.INPUT_FRAMES + C.OUTPUT_FRAMES])
    X = np.stack(xs)
    Y = np.stack(ys)
    print(f"  X {X.shape}  Y {Y.shape}  base rate "
          f"{(Y >= 1).mean():.4f}")

    br = float((Y >= 1).mean())
    for name, fn in BASELINES.items():
        p = fn(X, C.OUTPUT_FRAMES, base_rate=br) if name == "climatology" \
            else fn(X, C.OUTPUT_FRAMES)
        ev = M.evaluate(p, Y, x_input=X, threshold=0.15, label=name)
        print()
        print(M.format_report(ev))

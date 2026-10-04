"""
Verification metrics for rare-event spatial forecasts.

WHY NOT ACCURACY, AND WHY NOT AUROC
-----------------------------------
Lightning occupies ~1-2% of pixels. Two consequences:

  Accuracy is useless. Predicting "no lightning" everywhere scores 98-99%.

  AUROC is misleading. It is the probability a random positive outranks a
  random negative. With 98% negatives, almost every negative drawn is clear
  air far from any storm, so AUROC mostly measures "storm vs no storm" and
  saturates near 1.0 while the forecast is still operationally poor. AUROC
  does not depend on the base rate, which sounds like a virtue but here means
  it hides the thing that matters: how many false alarms you issue per hit.

  AUPRC (average precision) is the right scalar summary, because precision is
  exactly "of the warnings I issued, how many verified" -- the quantity that
  destroys public trust when it is low.

THE THREE DIAGNOSTICS THAT ACTUALLY MATTER HERE
-----------------------------------------------
1. CSI / POD / FAR at operating thresholds. CSI (critical success index)
   ignores true negatives entirely, which is what you want when true negatives
   are trivially abundant.

2. FSS (Fractions Skill Score) at several neighbourhood radii. A forecast that
   puts the storm 6 km from where it verified is operationally fine but is
   scored as both a miss and a false alarm by pixel-exact CSI ("double
   penalty"). FSS with a radius credits near-misses. Reporting FSS across radii
   also reveals the *spatial scale* at which the forecast has skill, which is
   more informative than any single number.

3. NEW-INITIATION skill. This is the one that separates a real nowcaster from
   a persistence echo, and it is the metric most papers under-report.

THE PERSISTENCE SHORTCUT, STATED PRECISELY
------------------------------------------
Lightning is highly autocorrelated in space and time. A large majority of
pixels that have lightning at t+30 min are close to pixels that had lightning
at t. So a model that simply copies (or advects) its lightning input channel
forward scores well on aggregate CSI while providing no warning of anything
new -- and new initiation over a previously quiet area is precisely the
situation that kills people, because nobody has taken shelter yet.

The remedy implemented here: `new_initiation_mask_lagrangian` restricts scoring
to pixels outside the corridor that existing lightning sweeps through as it is
advected along the motion estimated from the input window. Skill on that subset
cannot be obtained by persistence or by advection. Expect it to be much lower
than aggregate skill -- that gap is honest and should be reported, not hidden.

A NOTE ON HOW THIS WAS GOT WRONG FIRST
--------------------------------------
The original implementation excluded a STATIC 8 km neighbourhood. That is the
natural thing to write and it is wrong as soon as storms move: at 2.2 px/frame
a cell clears 4 px in under two frames, so from 10 min of lead the "new
initiation" subset was quietly readmitting pixels downstream of existing
lightning. Advection scored 4.3x lift on a subset that was documented as
guaranteeing ~0. The lesson generalises: an anti-shortcut metric has to be
defined in the same reference frame as the shortcut it is defending against.

All functions are pure NumPy (no scipy/sklearn available).
"""

from __future__ import annotations

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Contingency table
# ---------------------------------------------------------------------------

def contingency(pred_bin: np.ndarray, obs_bin: np.ndarray,
                valid: np.ndarray | None = None) -> dict:
    """2x2 contingency counts and derived scores.

    Args:
        pred_bin: boolean predicted-event array
        obs_bin:  boolean observed-event array
        valid:    optional boolean mask; only these elements are scored

    Returns dict with hits/misses/false_alarms/correct_negatives and the
    standard verification scores.
    """
    p = np.asarray(pred_bin).astype(bool)
    o = np.asarray(obs_bin).astype(bool)
    if valid is not None:
        v = np.asarray(valid).astype(bool)
        p, o = p[v], o[v]

    hits = int(np.count_nonzero(p & o))
    fa = int(np.count_nonzero(p & ~o))
    miss = int(np.count_nonzero(~p & o))
    cn = int(np.count_nonzero(~p & ~o))
    n = hits + fa + miss + cn

    def _safe(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    pod = _safe(hits, hits + miss)              # probability of detection
    far = _safe(fa, hits + fa)                  # false alarm ratio
    csi = _safe(hits, hits + miss + fa)         # critical success index
    bias = _safe(hits + fa, hits + miss)        # frequency bias
    # Heidke skill score: skill relative to random chance forecasts with the
    # same marginals. Unlike CSI it accounts for correct negatives, so it is
    # optimistic for rare events; reported for comparability with literature.
    exp_correct = (_safe((hits + miss) * (hits + fa), n)
                   + _safe((cn + miss) * (cn + fa), n))
    hss = _safe(hits + cn - exp_correct, n - exp_correct)
    # Equitable threat score / Gilbert skill score
    hits_random = _safe((hits + miss) * (hits + fa), n)
    ets = _safe(hits - hits_random, hits + miss + fa - hits_random)

    return {
        "hits": hits, "false_alarms": fa, "misses": miss,
        "correct_negatives": cn, "n": n,
        "POD": pod, "FAR": far, "CSI": csi, "BIAS": bias,
        "HSS": hss, "ETS": ets,
        "obs_rate": _safe(hits + miss, n),
        "pred_rate": _safe(hits + fa, n),
    }


def sweep_thresholds(prob: np.ndarray, obs_bin: np.ndarray,
                     thresholds: list[float] | None = None,
                     valid: np.ndarray | None = None) -> list[dict]:
    """Contingency scores across probability thresholds.

    Reporting a single threshold hides the trade-off. Operational users pick a
    threshold based on their tolerance for false alarms, so the whole curve is
    the deliverable.
    """
    thresholds = thresholds or C.PROB_THRESHOLDS
    rows = []
    for t in thresholds:
        r = contingency(prob >= t, obs_bin, valid)
        r["threshold"] = t
        rows.append(r)
    return rows


def best_csi(prob: np.ndarray, obs_bin: np.ndarray,
             thresholds: list[float] | None = None,
             valid: np.ndarray | None = None) -> dict:
    """Row with maximum CSI from a threshold sweep.

    NOTE: choosing the threshold on the same data you report is optimistic.
    The pipeline selects the threshold on the validation split and applies it
    unchanged to test. This helper is for diagnostics, not headline numbers.
    """
    rows = sweep_thresholds(prob, obs_bin, thresholds, valid)
    rows = [r for r in rows if not np.isnan(r["CSI"])]
    if not rows:
        return {}
    return max(rows, key=lambda r: r["CSI"])


# ---------------------------------------------------------------------------
# Neighbourhood / scale-aware verification
# ---------------------------------------------------------------------------

def _box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edge-padded. Last two axes are spatial.

    Implemented with cumulative sums so cost is independent of radius.
    """
    if radius <= 0:
        return a.astype(np.float64)
    k = 2 * radius + 1
    out = a.astype(np.float64)
    for axis in (-2, -1):
        n = out.shape[axis]
        pad_width = [(0, 0)] * out.ndim
        pad_width[axis] = (radius + 1, radius)
        p = np.pad(out, pad_width, mode="edge")
        cs = np.cumsum(p, axis=axis)
        hi = [slice(None)] * out.ndim
        lo = [slice(None)] * out.ndim
        hi[axis] = slice(k, k + n)
        lo[axis] = slice(0, n)
        out = (cs[tuple(hi)] - cs[tuple(lo)]) / k
    return out


def _box_max(a: np.ndarray, radius: int) -> np.ndarray:
    """Max over a (2r+1)^2 window, edge-clamped. Dilates event masks.

    Deliberately NOT implemented with np.roll: roll wraps around, so a pixel on
    the left edge of the domain would see events from the right edge. For the
    new-initiation mask that would spuriously mark edge pixels as "had
    lightning nearby" and silently remove them from the most important
    diagnostic in the whole system.
    """
    if radius <= 0:
        return a
    out = a
    for axis in (-2, -1):
        pad_width = [(0, 0)] * out.ndim
        pad_width[axis] = (radius, radius)
        p = np.pad(out, pad_width, mode="edge")
        n = out.shape[axis]
        acc = None
        for shift in range(2 * radius + 1):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(shift, shift + n)
            chunk = p[tuple(sl)]
            acc = chunk if acc is None else np.maximum(acc, chunk)
        out = acc
    return out


def fss(prob: np.ndarray, obs_bin: np.ndarray, radius: int,
        valid: np.ndarray | None = None) -> float:
    """Fractions Skill Score at a given neighbourhood radius.

    FSS = 1 - MSE(Pf, Po) / (MSE_ref), where Pf and Po are the forecast and
    observed event *fractions* within each neighbourhood, and MSE_ref is the
    worst-case MSE for those fraction fields (sum of squares).

    FSS = 1 is perfect; FSS = 0 is no skill. The conventional "useful skill"
    threshold is 0.5 + f/2 where f is the domain event frequency; for a rare
    event that is ~0.5.

    Interpreting across radii: the smallest radius at which FSS crosses 0.5 is
    the finest scale at which the forecast is useful. For a 2 km grid, radius 4
    means skill only at ~18 km scale, which is still operationally relevant for
    a district-level warning but is not "pin-point".
    """
    p = np.asarray(prob, dtype=np.float64)
    o = np.asarray(obs_bin, dtype=np.float64)
    if valid is not None:
        v = np.asarray(valid, dtype=np.float64)
        # Zero out invalid pixels in both fields so they contribute nothing.
        p = p * v
        o = o * v
    pf = _box_mean(p, radius)
    po = _box_mean(o, radius)
    if valid is not None:
        v_any = _box_mean(np.asarray(valid, dtype=np.float64), radius) > 0
        pf, po = pf[v_any], po[v_any]
    num = float(np.mean((pf - po) ** 2))
    den = float(np.mean(pf ** 2) + np.mean(po ** 2))
    if den <= 0:
        return float("nan")
    return 1.0 - num / den


def fss_profile(prob: np.ndarray, obs_bin: np.ndarray,
                radii: list[int] | None = None,
                valid: np.ndarray | None = None) -> dict[int, float]:
    radii = radii or C.FSS_RADII_PX
    return {r: fss(prob, obs_bin, r, valid) for r in radii}


def neighborhood_contingency(prob: np.ndarray, obs_bin: np.ndarray,
                             threshold: float, radius: int,
                             valid: np.ndarray | None = None) -> dict:
    """Contingency with spatial tolerance.

    A prediction counts as a hit if an observed event occurs anywhere within
    `radius`. This is the fair way to score a forecast whose position error is
    smaller than the warning area, and it directly addresses the double-penalty
    problem that makes pixel-exact CSI pessimistic for sharp forecasts.
    """
    pred = np.asarray(prob) >= threshold
    obs = np.asarray(obs_bin).astype(bool)
    obs_dilated = _box_max(obs, radius)
    pred_dilated = _box_max(pred, radius)
    # Hit: predicted and an observation nearby. Miss: observed but nothing
    # predicted nearby. This asymmetric treatment is standard practice.
    hits = int(np.count_nonzero(pred & obs_dilated
                                & (valid if valid is not None else True)))
    fa = int(np.count_nonzero(pred & ~obs_dilated
                              & (valid if valid is not None else True)))
    miss = int(np.count_nonzero(obs & ~pred_dilated
                                & (valid if valid is not None else True)))

    def _safe(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    return {
        "radius_px": radius, "radius_km": radius * C.PIXEL_KM,
        "threshold": threshold,
        "hits": hits, "false_alarms": fa, "misses": miss,
        "POD": _safe(hits, hits + miss),
        "FAR": _safe(fa, hits + fa),
        "CSI": _safe(hits, hits + miss + fa),
    }


# ---------------------------------------------------------------------------
# Probabilistic scores
# ---------------------------------------------------------------------------

def auprc(prob: np.ndarray, obs_bin: np.ndarray,
          valid: np.ndarray | None = None) -> float:
    """Average precision (area under precision-recall curve).

    Computed exactly by sorting, using the step-wise definition
    AP = sum_k (R_k - R_{k-1}) * P_k, which does not reward the optimistic
    interpolation that trapezoidal PR-AUC does.

    This is the primary scalar metric for this task. Its no-skill value equals
    the base rate, so an AP of 0.15 at a 1.5% base rate is a 10x lift, which
    is respectable -- whereas an AUROC of 0.95 on the same forecast tells you
    almost nothing.
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    o = np.asarray(obs_bin).astype(bool).ravel()
    if valid is not None:
        v = np.asarray(valid).astype(bool).ravel()
        p, o = p[v], o[v]
    n_pos = int(o.sum())
    if n_pos == 0 or n_pos == o.size:
        return float("nan")

    order = np.argsort(-p, kind="mergesort")
    o_sorted = o[order]
    p_sorted = p[order]
    tp = np.cumsum(o_sorted)
    fp = np.cumsum(~o_sorted)

    # Collapse tied scores to a single operating point; otherwise AP depends
    # on the arbitrary order of equal-probability pixels.
    last_of_tie = np.append(p_sorted[1:] != p_sorted[:-1], True)
    tp = tp[last_of_tie]
    fp = fp[last_of_tie]

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    prev_recall = np.append(0.0, recall[:-1])
    return float(np.sum((recall - prev_recall) * precision))


def auroc(prob: np.ndarray, obs_bin: np.ndarray,
          valid: np.ndarray | None = None) -> float:
    """AUROC, reported only for comparison with published work.

    See module docstring: for a 1% base rate this saturates and should not be
    used to judge operational value.
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    o = np.asarray(obs_bin).astype(bool).ravel()
    if valid is not None:
        v = np.asarray(valid).astype(bool).ravel()
        p, o = p[v], o[v]
    n_pos = float(o.sum())
    n_neg = float(o.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1)
    ps = p[order]
    i = 0
    while i < ps.size:
        j = i
        while j + 1 < ps.size and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[o].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier_decomposition(prob: np.ndarray, obs_bin: np.ndarray,
                        n_bins: int = 10,
                        valid: np.ndarray | None = None) -> dict:
    """Brier score with reliability / resolution / uncertainty decomposition.

    BS = reliability - resolution + uncertainty

      reliability  how far conditional observed frequency departs from the
                   forecast probability. LOWER is better. This is calibration.
      resolution   how much the conditional frequencies vary across bins.
                   HIGHER is better. This is discrimination.
      uncertainty  base-rate variance; a property of the data, not the model.

    Calibration matters operationally: if the system says 30% and it verifies
    70% of the time, authorities will either over-evacuate or stop trusting it.
    """
    p = np.asarray(prob, dtype=np.float64).ravel()
    o = np.asarray(obs_bin).astype(np.float64).ravel()
    if valid is not None:
        v = np.asarray(valid).astype(bool).ravel()
        p, o = p[v], o[v]
    if p.size == 0:
        return {}
    bs = float(np.mean((p - o) ** 2))
    obar = float(o.mean())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    rel = 0.0
    res = 0.0
    bins = []
    for b in range(n_bins):
        sel = idx == b
        nk = int(sel.sum())
        if nk == 0:
            bins.append({"bin": b, "n": 0, "mean_prob": float("nan"),
                         "obs_freq": float("nan")})
            continue
        pk = float(p[sel].mean())
        ok = float(o[sel].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - obar) ** 2
        bins.append({"bin": b, "n": nk,
                     "prob_lo": float(edges[b]), "prob_hi": float(edges[b + 1]),
                     "mean_prob": pk, "obs_freq": ok})
    rel /= p.size
    res /= p.size
    unc = obar * (1.0 - obar)
    return {
        "brier": bs, "reliability": rel, "resolution": res,
        "uncertainty": unc, "base_rate": obar,
        # Brier skill score vs a constant climatological forecast.
        "bss": float(1.0 - bs / unc) if unc > 0 else float("nan"),
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# The new-initiation subset: the anti-persistence diagnostic
# ---------------------------------------------------------------------------

def new_initiation_mask(x_input: np.ndarray, radius: int = 4,
                        lightning_channel: int | None = None) -> np.ndarray:
    """Mask of pixels with NO lightning nearby during the input window.

    Args:
        x_input: (T_in, C, H, W) or (N, T_in, C, H, W) input sequence
        radius:  neighbourhood radius in pixels. 4 px = 8 km, chosen so that
                 a forecast cannot score by advecting existing lightning a
                 short distance.
        lightning_channel: index of the lightning density channel.

    Returns:
        (H, W) or (N, H, W) boolean mask, True where the pixel is "clean".

    Scoring restricted to this mask answers: can the model warn about
    lightning where there was none? Eulerian persistence scores exactly 0 CSI
    here by construction, because it never places probability outside the
    dilated footprint this mask removes.

    IMPORTANT: this uses only the INPUT window, so applying it introduces no
    leakage -- it is computable at forecast time.

    THIS MASK IS NOT SUFFICIENT ON ITS OWN. MEASURED FAILURE
    --------------------------------------------------------
    An earlier version of this docstring claimed advection also scores ~0 here.
    That claim was FALSE and `run_tests.py` caught it. A static radius is a
    valid exclusion only at zero lead time. Storms in this configuration move
    at a mean 2.2 px/frame, so they clear a 4 px (8 km) exclusion in under two
    frames. From 10 min of lead onward, pixels this mask calls "new initiation"
    include pixels merely DOWNSTREAM of existing lightning, which Lagrangian
    persistence predicts trivially. Measured with the static mask: advection
    reached 4.3x AUPRC lift overall and 19.5x at the 10 min lead, with
    CSI 0.066 -- not the 0.0 the metric was supposed to guarantee.

    Use `new_initiation_mask_lagrangian` for any lead time beyond the first
    frame. This function is retained because it is the correct primitive for
    single-frame and feature-ranking use, and because the Lagrangian version
    is built from it.
    """
    ch = C.CH["light_dens"] if lightning_channel is None else lightning_channel
    x = np.asarray(x_input)
    if x.ndim == 4:
        lt = x[:, ch]                      # (T, H, W)
        had = (lt >= C.LIGHTNING_THRESHOLD).any(axis=0)
        return ~_box_max(had, radius)
    elif x.ndim == 5:
        lt = x[:, :, ch]                   # (N, T, H, W)
        had = (lt >= C.LIGHTNING_THRESHOLD).any(axis=1)
        return ~_box_max(had, radius)
    raise ValueError(f"expected 4 or 5 dims, got {x.ndim}")


def _shift_bool(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer-shift a boolean field on the last two axes, zero-filled.

    Deliberately NOT np.roll, for the same reason `_box_max` is not: roll wraps,
    so a storm advected off the eastern edge would reappear in the west and
    exclude pixels 128 km upstream of anything real. Zero-fill means "advected
    out of the domain", which is the honest answer.
    """
    out = np.zeros_like(a, dtype=bool)
    h, w = a.shape[-2], a.shape[-1]
    if abs(dy) >= h or abs(dx) >= w:
        return out
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    out[..., ys_dst, xs_dst] = a[..., ys_src, xs_src]
    return out


def new_initiation_mask_lagrangian(
        x_input: np.ndarray, n_out: int, radius: int = 4,
        lightning_channel: int | None = None,
        motion: tuple[float, float] | np.ndarray | None = None,
        dispersion: bool = True) -> np.ndarray:
    """Lead-resolved new-initiation mask: excludes the SWEPT CORRIDOR.

    Args:
        x_input: (N, T_in, C, H, W) input sequences
        n_out:   number of forecast steps to build a mask for
        radius:  base exclusion radius in px (4 px = 8 km)
        motion:  optional (dy, dx) px/frame, or (N, 2) per sample. If None it
                 is estimated from the input window by optical flow.
        dispersion: also grow the corridor by round(sqrt(step)) px, matching
                 the dispersion the advection baseline applies to its own
                 forecast, so the corridor covers everything that baseline can
                 claim.

    Returns:
        (N, n_out, H, W) boolean, True where the pixel is scoreable.

    WHY A CORRIDOR AND NOT A BIGGER CIRCLE
    --------------------------------------
    The obvious repair for the static-radius bug is to grow the radius with
    lead time, r(t) = 4 + ceil(speed * t). That does work -- it drives
    advection to exactly 1.00x lift at every lead, which is how the cause was
    proved. But it is isotropic, and by the 60 min lead it reaches 31 px on a
    64 px domain, so a single lightning pixel erases almost the whole grid and
    the subset stops containing enough events to measure anything.

    Storms do not disperse isotropically; they travel. So we exclude the region
    the existing lightning actually sweeps through: displace the footprint by
    the estimated motion at every step from 1 to `step`, dilate each position,
    and take the union. That removes precisely what Lagrangian persistence can
    claim while leaving the upstream and cross-stream domain scoreable.

    The motion estimate comes from the INPUT window only, so this is still
    computable at forecast time and introduces no leakage. It is also the SAME
    estimator the advection baseline uses, which is deliberate: whatever
    advection does with its flow vector, this mask has already excluded.
    """
    x = np.asarray(x_input)
    if x.ndim != 5:
        raise ValueError(f"expected (N, T, C, H, W), got {x.ndim} dims")
    n, _, _, h, w = x.shape
    ch = C.CH["light_dens"] if lightning_channel is None else lightning_channel
    had = (x[:, :, ch] >= C.LIGHTNING_THRESHOLD).any(axis=1)   # (N, H, W)

    # Lazy import: baselines imports _box_mean from this module, so a top-level
    # import here would be circular.
    if motion is None:
        import baselines as _B
        mv = np.array([_B.estimate_motion(x[i, :, C.CH["refl_sfc"]])
                       for i in range(n)], dtype=np.float64)
    else:
        mv = np.asarray(motion, dtype=np.float64)
        if mv.ndim == 1:
            mv = np.broadcast_to(mv, (n, 2))

    out = np.ones((n, n_out, h, w), dtype=bool)
    for i in range(n):
        if not had[i].any():
            continue                      # nothing to exclude beyond nothing
        dy, dx = float(mv[i, 0]), float(mv[i, 1])
        swept = _box_max(had[i], radius)  # step 0 footprint
        for t in range(n_out):
            step = t + 1
            moved = _shift_bool(had[i], int(round(dy * step)),
                                int(round(dx * step)))
            r = radius
            if dispersion:
                r += int(round(np.sqrt(step)))
            swept = swept | _box_max(moved, r)
            out[i, t] = ~swept            # union is monotone in step
    return out


# ---------------------------------------------------------------------------
# Lead-time resolved evaluation
# ---------------------------------------------------------------------------

def evaluate(prob: np.ndarray, obs: np.ndarray,
             x_input: np.ndarray | None = None,
             threshold: float = 0.15,
             valid: np.ndarray | None = None,
             label: str = "") -> dict:
    """Full evaluation of a forecast.

    Args:
        prob:    (N, T_out, H, W) forecast probabilities in [0, 1]
        obs:     (N, T_out, H, W) observed strike counts (or binary)
        x_input: (N, T_in, C, H, W) inputs, needed for the new-initiation subset
        threshold: probability threshold for the headline contingency scores.
                   MUST have been selected on validation data, not here.
        valid:   optional (N, T_out, H, W) mask of scoreable pixels
        label:   name for reporting

    Returns a nested dict. The fields that matter most:
        overall.AUPRC              primary scalar
        overall.CSI                at the given threshold
        by_lead                    skill decay with lead time
        fss                        scale-dependent skill
        new_initiation             skill excluding persistence
        calibration                reliability
    """
    prob = np.asarray(prob, dtype=np.float64)
    obs_bin = np.asarray(obs) >= C.LIGHTNING_THRESHOLD

    out: dict = {"label": label, "threshold": threshold,
                 "n_forecasts": int(prob.shape[0]),
                 "shape": list(prob.shape)}

    out["overall"] = contingency(prob >= threshold, obs_bin, valid)
    out["overall"]["AUPRC"] = auprc(prob, obs_bin, valid)
    out["overall"]["AUROC"] = auroc(prob, obs_bin, valid)

    out["sweep"] = sweep_thresholds(prob, obs_bin, valid=valid)

    # ---- Lead-time decay ----
    # Physically, skill MUST fall with lead time. If it does not, something is
    # leaking: either the target is visible in the input, or the splits overlap.
    by_lead = []
    for t in range(prob.shape[1]):
        lead_min = (t + 1) * C.TIMESTEP_MIN
        v = None if valid is None else valid[:, t]
        r = contingency(prob[:, t] >= threshold, obs_bin[:, t], v)
        r["lead_min"] = lead_min
        r["AUPRC"] = auprc(prob[:, t], obs_bin[:, t], v)
        r["FSS_r2"] = fss(prob[:, t], obs_bin[:, t], 2, v)
        by_lead.append(r)
    out["by_lead"] = by_lead

    # Aggregated into the configured bands
    banded = []
    for lo, hi in C.LEAD_TIME_BINS:
        ts = [t for t in range(prob.shape[1])
              if lo <= (t + 1) * C.TIMESTEP_MIN <= hi]
        if not ts:
            continue
        v = None if valid is None else valid[:, ts]
        r = contingency(prob[:, ts] >= threshold, obs_bin[:, ts], v)
        r["band"] = f"{lo}-{hi}min"
        r["AUPRC"] = auprc(prob[:, ts], obs_bin[:, ts], v)
        banded.append(r)
    out["by_lead_band"] = banded

    # ---- Scale-dependent skill ----
    out["fss"] = {f"r{r}_{int(r * C.PIXEL_KM)}km":
                  fss(prob, obs_bin, r, valid) for r in C.FSS_RADII_PX}
    out["neighborhood"] = [
        neighborhood_contingency(prob, obs_bin, threshold, r, valid)
        for r in C.FSS_RADII_PX
    ]

    # ---- Calibration ----
    out["calibration"] = brier_decomposition(prob, obs_bin, valid=valid)

    # ---- New initiation ----
    if x_input is not None:
        # Lagrangian (swept-corridor) exclusion, not a static radius. With a
        # static radius the advection baseline scored 4.3x lift here, which
        # silently invalidated the whole point of the subset. See
        # new_initiation_mask_lagrangian for the measurement and the fix.
        ni_valid = new_initiation_mask_lagrangian(x_input, prob.shape[1])
        if valid is not None:
            ni_valid = ni_valid & valid
        n_clean_pos = int(np.count_nonzero(obs_bin & ni_valid))
        # Static mask kept alongside for comparison only -- it is the quantity
        # older results were computed with, so reporting both makes the change
        # auditable instead of a silent renumbering.
        static_valid = np.broadcast_to(
            new_initiation_mask(x_input)[:, None], prob.shape)
        ni = {
            "n_scored_px": int(np.count_nonzero(ni_valid)),
            "n_positive_px": n_clean_pos,
            "n_scored_px_static": int(np.count_nonzero(static_valid)),
            "exclusion": "lagrangian_swept_corridor",
            "note": ("Pixels outside the corridor swept by existing lightning "
                     "advected along the input-window motion estimate. Both "
                     "Eulerian and Lagrangian persistence score 0 here."),
        }
        if n_clean_pos > 0:
            ni.update(contingency(prob >= threshold, obs_bin, ni_valid))
            ni["AUPRC"] = auprc(prob, obs_bin, ni_valid)
            ni["FSS_r4"] = fss(prob, obs_bin, 4, ni_valid)
            # Also resolve by lead time -- new-initiation skill should decay
            # even faster than aggregate skill.
            ni["by_lead"] = []
            for t in range(prob.shape[1]):
                v = ni_valid[:, t]
                if np.count_nonzero(obs_bin[:, t] & v) < 10:
                    continue
                r = contingency(prob[:, t] >= threshold, obs_bin[:, t], v)
                r["lead_min"] = (t + 1) * C.TIMESTEP_MIN
                r["AUPRC"] = auprc(prob[:, t], obs_bin[:, t], v)
                ni["by_lead"].append(r)
        out["new_initiation"] = ni
        if n_clean_pos > 0:
            ni.update(contingency(prob >= threshold, obs_bin, ni_valid))
            ni["AUPRC"] = auprc(prob, obs_bin, ni_valid)
            ni["FSS_r4"] = fss(prob, obs_bin, 4, ni_valid)
            # Also resolve by lead time -- new-initiation skill should decay
            # even faster than aggregate skill.
            ni["by_lead"] = []
            for t in range(prob.shape[1]):
                v = ni_valid[:, t]
                if np.count_nonzero(obs_bin[:, t] & v) < 10:
                    continue
                r = contingency(prob[:, t] >= threshold, obs_bin[:, t], v)
                r["lead_min"] = (t + 1) * C.TIMESTEP_MIN
                r["AUPRC"] = auprc(prob[:, t], obs_bin[:, t], v)
                ni["by_lead"].append(r)
        out["new_initiation"] = ni

    return out


def format_report(ev: dict) -> str:
    """Readable summary of an `evaluate` result."""
    L = []
    ov = ev["overall"]
    L.append(f"=== {ev.get('label', 'forecast')} ===")
    L.append(f"  forecasts={ev['n_forecasts']}  shape={ev['shape']}  "
             f"threshold={ev['threshold']:.3f}")
    L.append(f"  base rate      {ov['obs_rate']:.4f}   "
             f"pred rate {ov['pred_rate']:.4f}")
    L.append(f"  AUPRC          {ov['AUPRC']:.4f}   "
             f"(no-skill = base rate {ov['obs_rate']:.4f}, "
             f"lift {ov['AUPRC'] / max(ov['obs_rate'], 1e-9):.1f}x)")
    L.append(f"  AUROC          {ov['AUROC']:.4f}   "
             f"(inflated for rare events -- do not headline)")
    L.append(f"  CSI            {ov['CSI']:.4f}")
    L.append(f"  POD            {ov['POD']:.4f}   FAR {ov['FAR']:.4f}   "
             f"BIAS {ov['BIAS']:.3f}")
    L.append(f"  HSS            {ov['HSS']:.4f}   ETS {ov['ETS']:.4f}")

    cal = ev.get("calibration") or {}
    if cal:
        L.append(f"  Brier          {cal['brier']:.5f}  "
                 f"BSS {cal['bss']:+.4f}  "
                 f"rel {cal['reliability']:.5f} (lower=better)  "
                 f"res {cal['resolution']:.5f} (higher=better)")

    L.append("  FSS by scale:")
    for k, v in ev["fss"].items():
        flag = " <- useful" if (not np.isnan(v) and v >= 0.5) else ""
        L.append(f"    {k:14s} {v:.4f}{flag}")

    L.append("  Skill vs lead time (CSI / AUPRC):")
    for r in ev["by_lead_band"]:
        L.append(f"    {r['band']:12s} CSI {r['CSI']:.4f}  "
                 f"AUPRC {r['AUPRC']:.4f}  POD {r['POD']:.4f}  "
                 f"FAR {r['FAR']:.4f}")

    ni = ev.get("new_initiation")
    if ni:
        L.append("  NEW INITIATION (no lightning within 8 km in input window):")
        L.append(f"    scored px {ni['n_scored_px']:,}  "
                 f"positives {ni['n_positive_px']:,}")
        if "CSI" in ni:
            L.append(f"    CSI {ni['CSI']:.4f}  POD {ni['POD']:.4f}  "
                     f"FAR {ni['FAR']:.4f}  AUPRC {ni['AUPRC']:.4f}")
            L.append(f"    FSS_r4 {ni['FSS_r4']:.4f}")
        else:
            L.append("    too few positives to score")
    return "\n".join(L)


if __name__ == "__main__":
    # Self-check on synthetic arrays with known properties.
    rng = np.random.default_rng(0)
    obs = (rng.random((4, 6, 32, 32)) < 0.02)
    # A forecast that is the truth plus noise should beat a random one.
    good = np.clip(obs * 0.8 + rng.random(obs.shape) * 0.15, 0, 1)
    bad = rng.random(obs.shape)
    print("perfect-ish AUPRC:", round(auprc(good, obs), 4))
    print("random     AUPRC:", round(auprc(bad, obs), 4),
          " (should be ~base rate", round(obs.mean(), 4), ")")
    print("perfect FSS r0  :", round(fss(obs.astype(float), obs, 0), 4),
          "(should be 1.0)")

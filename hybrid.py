"""
Motion-compensated (Lagrangian) nowcasting, and a calibrated advection/learned blend.

THE MEASURED PROBLEM THIS FILE SOLVES
-------------------------------------
A single 3x3 ConvLSTM layer propagates information at exactly one pixel per
recurrent step: the state at (i, j) is updated from a 3x3 window of the previous
state, so after T steps a pixel can only have been influenced by inputs within T
pixels. Over the 6-frame input window that is a reach of about 6 px.

The steering flow in config.SYNTH is 2.5 px/frame, so a cell moves ~15 px across
the 6-frame history and ~30 px across the 12-frame forecast -- half the 64 px
domain. The receptive field is therefore smaller than the displacement by a
factor of two to five. The consequence is not "slightly worse accuracy": the
network structurally CANNOT see where a fast cell came from, and it cannot move
its own prediction to where the cell is going. Everything it learns is
necessarily local and quasi-Eulerian. This is an architectural limit, not a
training-budget limit, and no number of epochs fixes it.

Making the kernel large enough (15+ px reach) would need either a 31x31 kernel
or a deep multi-scale stack. Both are cheap in torch on a GPU and both are
prohibitive here: cost scales with k^2 in the fused convolution, so a 15x15
kernel is 25x the FLOPs of 3x3, on 2 CPU cores, in NumPy.

THE FIX: SPLIT THE PROBLEM THE WAY OPERATIONAL NOWCASTING DOES
--------------------------------------------------------------
Nowcasting decomposes cleanly into two questions with completely different
mathematical characters:

    WHERE will the storm be?      Kinematics. Smooth, large-scale, and solved
                                  well by optical flow. Non-local by nature.
    WILL it electrify?            Microphysics. Graupel in the -10 to -20C
                                  layer, charge accumulation, breakdown. Local
                                  by nature -- it happens inside the cell.

Advection answers the first. A small-receptive-field ConvLSTM is a perfectly
appropriate model for the second, because the process really is local. The
mistake is asking one 3x3 recurrence to do both.

So `MotionCompensatedNowcaster`:

  1. estimates the motion field from the reflectivity sequence
     (baselines.estimate_motion, Farneback optical flow, intensity-weighted);
  2. warps the input feature sequence into a LAGRANGIAN reference frame, so a
     given storm cell sits at approximately the same pixel in every input
     frame. The ConvLSTM now sees a quasi-stationary storm, and 6 px of reach is
     ample to relate a cell's state at frame 1 to its state at frame 6. Its
     capacity goes into intensification and electrification instead of being
     spent -- unsuccessfully -- on translation;
  3. trains against targets warped into the same Lagrangian frame, so the
     learning problem is "does this cell electrify", with translation removed
     from both sides;
  4. warps the predicted probability fields BACK to the Earth-relative frame
     using the same motion scaled by lead time, because a warning has to be
     issued for a place on the ground.

WHY ZERO-FILL AND NOT EDGE-REPLICATE AT THE WARP BOUNDARY
---------------------------------------------------------
Both warps expose a strip of the grid whose source lies outside the domain.
baselines._shift_field zero-fills it, and that is the right choice in all three
places it is used here:

  * Input features. features.build_features already emits exactly 0.0 for any
    pixel it could not observe (a radar outage, a missing INSAT slot), with the
    validity flag carried in a separate channel. Zero-filling the warped-in strip
    therefore encodes it in the representation the encoder ALREADY associates
    with "not observed", which is precisely true of an area that has advected in
    from beyond the radar. Being exact about the one imperfection: because
    config.CHANNEL_NORM uses a non-zero offset, 0.0 on `refl_sfc_n`
    de-normalises to 20 dBZ rather than to clear air, so the strip reads as
    "unobserved / weak unclassified echo" rather than "definitely clear". That
    is a bounded artefact confined to a border of width |v| * (T_in - 1), and it
    is the conservative direction: it cannot fabricate a strong echo.
    Edge-replication, by contrast, would copy the domain-edge storm state across
    the whole upstream border, manufacturing a 15-30 px band of fictitious deep
    convection that the encoder would treat as a genuine observation.

  * Targets. Replication would smear real strikes along the boundary and create
    a band of positives that never happened, teaching the model to fire on the
    domain edge -- a bias that is invisible in aggregate metrics because it sits
    where verification is sparsest.

  * Output probabilities. Replication would paint the entire downstream
    boundary with whatever probability the edge pixel had, which at a 1% base
    rate means a permanent stripe of false alarms.

The honest cost of zero-fill is that some real positives are pushed out of the
warped target and are labelled negative. That is a genuine loss of training
signal, it is quantified in the self-test rather than hidden, and it is bounded
by capping the motion magnitude.

THE BLEND, AND WHY ITS WEIGHTS ARE FITTED ON VALIDATION ONLY
------------------------------------------------------------
`BlendedNowcaster` combines an advection baseline with a learned model using a
per-lead-time weight. The physical expectation is that advection dominates at
short lead (the storm is where it was, moved a little; nothing about the cell
has had time to change) and the learned model matters more at longer lead (the
cell has evolved, and evolution is what the network models). Whether the data
agrees is a testable prediction, so the fitted weights are reported rather than
assumed, and the monotone-decline constraint is available but OFF by default.

The weights are fitted on the VALIDATION split and never on test. This is not a
formality: with 12 free weights and a threshold, fitting on test would let a
model with no skill at all pick out the noise pattern of the test set. Anything
fitted after the model is a model parameter and must be estimated out of sample.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import numpy as np

import baselines as B
import config as C
import features as FT
from convlstm_np import ConvLSTMNowcaster, sigmoid


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------

def _shift_stack(a: np.ndarray, dy: float, dx: float,
                 fill: float = 0.0) -> np.ndarray:
    """Batched form of baselines._shift_field: shifts the LAST TWO axes.

    Identical arithmetic to baselines._shift_field -- bilinear interpolation
    with zero-fill outside the source grid -- but the index arrays are built
    once and applied to every leading axis at the same time. The compensation of
    one training sample touches T_in * N_FEATURES = 120 fields, so calling the
    2-D version in a Python loop would cost ~17 ms per sample and dominate the
    data preparation. The self-test asserts EXACT equality with
    baselines._shift_field, so this remains a performance detail and not a
    second, subtly different implementation.
    """
    h, w = a.shape[-2:]
    y = np.arange(h)[:, None] - dy
    x = np.arange(w)[None, :] - dx
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = y - y0
    fx = x - x0

    def _g(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yyc = np.clip(yy, 0, h - 1)
        xxc = np.clip(xx, 0, w - 1)
        return np.where(ok, a[..., yyc, xxc], fill)

    return (_g(y0, x0) * ((1 - fy) * (1 - fx))
            + _g(y0 + 1, x0) * (fy * (1 - fx))
            + _g(y0, x0 + 1) * ((1 - fy) * fx)
            + _g(y0 + 1, x0 + 1) * (fy * fx))


def motion_from_features(F: np.ndarray) -> np.ndarray:
    """Per-sample (dy, dx) px/frame, estimated from the feature stack.

    Args:
        F: (N, T_in, N_FEATURES, H, W) output of features.build_features_batch
    Returns:
        (N, 2) float64 array of (dy, dx)

    Feature 0 is `refl_sfc_n`, i.e. surface reflectivity normalised with the
    FIXED constants in config.CHANNEL_NORM. We invert that affine map before
    calling the flow estimator rather than running the estimator on the
    normalised field, because baselines._to_u8 quantises its input assuming a
    0-70 dBZ range: normalised values live in roughly [-1, 2.8] and would all
    collapse into the bottom 4% of the uint8 range, leaving optical flow with
    ~10 distinct grey levels to work with.

    Deriving motion from the features rather than taking the raw channels as a
    second argument keeps this class signature-compatible with
    ConvLSTMNowcaster, so the pipeline can hold both in one dict and call them
    identically. The self-test checks that motion recovered this way matches
    motion computed from the raw channels.

    One caveat, stated because it is a real approximation: pixels where
    refl_sfc was missing are 0 in the normalised feature and de-normalise to the
    channel mean (20 dBZ), not to FILL_VALUE. They therefore contribute a weak
    static signal to the flow estimate instead of being excluded. The intensity
    weighting inside estimate_motion (weight = max(dBZ - 15, 0)) keeps that
    contribution small, and a static region biases the estimate towards zero
    motion, which is the conservative direction.
    """
    mu, sc = C.CHANNEL_NORM["refl_sfc"]
    out = np.zeros((F.shape[0], 2), dtype=np.float64)
    for i in range(F.shape[0]):
        refl = F[i, :, 0].astype(np.float64) * sc + mu
        dy, dx = B.estimate_motion(refl)
        out[i] = (dy, dx)
    return out


def _cap_motion(motion: np.ndarray, max_px: float) -> np.ndarray:
    """Limit |v| so a wild flow estimate cannot warp everything off the grid.

    Farneback on a sparse, noisy 64x64 dBZ field occasionally returns a large
    spurious vector (a decaying cell that vanishes between frames looks like
    very fast motion). Uncapped, one such sample would have its entire target
    warped out of the domain and would train the network on an all-negative
    label field. The cap is a magnitude clip that preserves direction.
    """
    if max_px is None or max_px <= 0:
        return motion
    mag = np.sqrt((motion ** 2).sum(axis=1, keepdims=True))
    scale = np.where(mag > max_px, max_px / np.maximum(mag, 1e-9), 1.0)
    return motion * scale


def compensate_inputs(F: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Warp an input feature sequence into the Lagrangian frame.

    Args:
        F:      (N, T_in, Ch, H, W)
        motion: (N, 2) (dy, dx) px/frame
    Returns:
        same shape, every frame aligned to the position of the LAST input frame.

    The reference frame is the last observation, r = T_in - 1, because that is
    the frame the forecast is issued from and the only one whose geometry is
    known exactly. A cell observed at frame t has moved by v * (r - t) by the
    time of frame r, so frame t's content is shifted by +v * (r - t). Frame r
    itself is untouched, which means the network's view of "now" is the true
    observation with no interpolation error -- the frames that get resampled are
    the older, less important ones.
    """
    n, t_in = F.shape[0], F.shape[1]
    r = t_in - 1
    out = np.empty_like(F)
    for i in range(n):
        dy, dx = float(motion[i, 0]), float(motion[i, 1])
        for t in range(t_in):
            lag = r - t
            if lag == 0 or (dy == 0.0 and dx == 0.0):
                out[i, t] = F[i, t]
            else:
                out[i, t] = _shift_stack(F[i, t], dy * lag, dx * lag,
                                         fill=0.0).astype(F.dtype, copy=False)
    return out


def compensate_targets(Y: np.ndarray, motion: np.ndarray,
                       binarise_at: float = 0.5) -> np.ndarray:
    """Warp binary targets from the Earth frame into the Lagrangian frame.

    Args:
        Y:      (N, T_out, H, W) strike counts or binary
        motion: (N, 2) px/frame
    Returns:
        (N, T_out, H, W) BOOLEAN targets in the reference frame.

    The target at lead k sits k frames after the reference, so its content must
    be moved backwards along the flow by v * k to arrive in the reference frame.

    TWO TRAPS, BOTH OF WHICH SILENTLY DESTROY THE LABELS.

    First: binarise BEFORE warping, never after. Bilinear interpolation of a
    binary field returns values in (0, 1); config.LIGHTNING_THRESHOLD is 1, and
    ConvLSTMNowcaster._prepare applies `y >= LIGHTNING_THRESHOLD` to any
    non-boolean target. A warped-then-thresholded count field would therefore
    map every interpolated positive to 0 and hand the network an all-negative
    label array, which trains perfectly happily to a constant output.

    Second: return a genuine bool array. _prepare passes bool through untouched
    but re-thresholds anything else, so returning float 0.0/1.0 would work only
    by accident of the >= comparison and would break the moment
    LIGHTNING_THRESHOLD changed.

    `binarise_at` = 0.5 makes the resample effectively nearest-neighbour for an
    isolated positive pixel. Sub-pixel fidelity is not worth chasing: the
    displacement is a single domain-mean vector whose own error is comfortably
    more than one pixel.
    """
    yb = np.asarray(Y)
    yb = yb if yb.dtype == bool else (yb >= C.LIGHTNING_THRESHOLD)
    n, t_out = yb.shape[0], yb.shape[1]
    out = np.zeros(yb.shape, dtype=bool)
    for i in range(n):
        dy, dx = float(motion[i, 0]), float(motion[i, 1])
        if dy == 0.0 and dx == 0.0:
            out[i] = yb[i]
            continue
        for k in range(t_out):
            step = k + 1
            w = _shift_stack(yb[i, k].astype(np.float64),
                             -dy * step, -dx * step, fill=0.0)
            out[i, k] = w >= binarise_at
    return out


def decompensate_probs(P: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Warp Lagrangian-frame probabilities back to the Earth-relative frame.

    Args:
        P:      (N, T_out, H, W) probabilities in the reference frame
        motion: (N, 2) px/frame
    Returns:
        same shape, Earth-relative.

    This is the exact inverse displacement of `compensate_targets`: content at
    lead k moves forward along the flow by v * k. Using the SAME motion vector
    for the forward and backward warps is what makes the pair consistent -- if
    the network learned to predict a cell staying put in the Lagrangian frame,
    the round trip places it where advection says it will be, and the model's
    contribution is purely the change in electrification probability.

    Zero-fill here means "no forecast issued" for the upstream strip, which is
    honest: that area's storm history is outside the domain, so there is no
    basis for a warning.
    """
    n, t_out = P.shape[0], P.shape[1]
    out = np.zeros(P.shape, dtype=np.float64)
    for i in range(n):
        dy, dx = float(motion[i, 0]), float(motion[i, 1])
        if dy == 0.0 and dx == 0.0:
            out[i] = P[i]
            continue
        for k in range(t_out):
            step = k + 1
            out[i, k] = _shift_stack(P[i, k].astype(np.float64),
                                     dy * step, dx * step, fill=0.0)
    return np.clip(out, 0.0, 1.0)


def lagrangian_coverage(Y: np.ndarray, motion: np.ndarray) -> dict:
    """Per-lead ceiling on recall imposed by the coordinate change itself.

    THIS IS THE HONEST COST OF GOING LAGRANGIAN, AND IT MUST BE REPORTED.

    An Earth-frame pixel q at lead k has Lagrangian coordinate q - v*k. If that
    coordinate lies off the grid, the pixel's upstream history was never
    observed, the compensated input carries no information about it, and the
    inverse warp assigns it probability zero. Such a pixel is a GUARANTEED MISS
    however good the network is. With |v| ~ 2.5 px/frame and 12 lead steps the
    displacement reaches 30 px on a 64 px grid, so at the longest leads a large
    fraction of the domain is structurally unforecastable.

    This function pushes the observed target through compensate -> decompensate
    and counts how many positives come back. That fraction is the maximum POD
    the motion-compensated model can attain at each lead, and it explains
    (rather than excuses) any long-lead deficit in the results table. The plain
    Eulerian model does not pay this cost -- it can emit a probability anywhere
    -- which is a genuine advantage of staying in Earth coordinates and is the
    reason the blend exists.

    The mirror-image effect does NOT bias the forecast, and the symmetry is worth
    stating: reference-frame pixels whose forward trajectory leaves the domain
    receive a fabricated negative label from zero-fill, but those are exactly the
    pixels the inverse warp pushes off-grid at predict time. Their predictions
    are discarded, so the fabricated labels cannot leak into anything that gets
    scored. They only dilute the gradient with easy negatives.
    """
    yb = np.asarray(Y)
    yb = yb if yb.dtype == bool else (yb >= C.LIGHTNING_THRESHOLD)
    back = decompensate_probs(
        compensate_targets(yb, motion).astype(np.float64), motion)
    recovered = yb & (back >= 0.5)
    # Purely geometric coverage: push a field of ones through the same warp pair.
    # Unlike the positives-based fraction this is defined at every lead even
    # when a lead happens to contain no observed lightning, so it is what the
    # monotonicity assertion uses.
    ones = np.ones((yb.shape[0], yb.shape[1]) + yb.shape[2:], dtype=np.float64)
    geo = decompensate_probs(compensate_targets(ones >= 0.5, motion
                                                ).astype(np.float64), motion)
    per_lead = []
    for k in range(yb.shape[1]):
        n_pos = int(yb[:, k].sum())
        per_lead.append({
            "lead_min": (k + 1) * C.TIMESTEP_MIN,
            "n_positive": n_pos,
            "recoverable_frac": (float(recovered[:, k].sum()) / n_pos
                                 if n_pos else float("nan")),
            "grid_coverage_frac": float((geo[:, k] >= 0.5).mean()),
        })
    tot = int(yb.sum())
    return {
        "per_lead": per_lead,
        "overall_recoverable_frac": (float(recovered.sum()) / tot
                                     if tot else float("nan")),
        "overall_grid_coverage_frac": float((geo >= 0.5).mean()),
        "note": ("Fraction of observed positives whose Lagrangian coordinate is "
                 "on-grid. This is the POD ceiling for the motion-compensated "
                 "model; the rest are guaranteed misses because their upstream "
                 "history is outside the domain."),
    }


# ---------------------------------------------------------------------------
# The motion-compensated model
# ---------------------------------------------------------------------------

class MotionCompensatedNowcaster:
    """ConvLSTM trained and evaluated in a Lagrangian reference frame.

    Mirrors ConvLSTMNowcaster's constructor and its fit / predict / save / load
    surface, so pipeline.py can hold both in one registry and drive them with
    identical calls. Everything specific to this class happens inside those four
    methods; the network itself is unmodified, which is deliberate -- the claim
    being tested is that the COORDINATE SYSTEM is what limits the plain model,
    not its parameter count.
    """

    def __init__(self, in_ch: int, t_out: int = C.OUTPUT_FRAMES,
                 hidden: int | None = None, kernel: int | None = None,
                 lr: float | None = None, pos_weight: float | None = None,
                 grad_clip: float | None = None, downsample: int | None = None,
                 batch: int | None = None, seed: int = 0,
                 dtype: Any = np.float32,
                 max_motion_px: float = 4.0,
                 compensate: bool = True,
                 target_binarise: float = 0.5) -> None:
        self.net = ConvLSTMNowcaster(
            in_ch=in_ch, t_out=t_out, hidden=hidden, kernel=kernel, lr=lr,
            pos_weight=pos_weight, grad_clip=grad_clip, downsample=downsample,
            batch=batch, seed=seed, dtype=dtype)
        self.in_ch = self.net.in_ch
        self.t_out = self.net.t_out
        # Cap in px/frame. 4.0 px/frame = 8 km/5 min = 96 km/h, above almost any
        # real storm motion, so the cap only ever fires on a bad flow estimate.
        self.max_motion_px = float(max_motion_px)
        # `compensate=False` turns this class into a pass-through wrapper around
        # ConvLSTMNowcaster. That is the ablation that isolates the value of the
        # coordinate change from every other difference, and it is one keyword
        # rather than a second code path.
        self.compensate = bool(compensate)
        self.target_binarise = float(target_binarise)

    # -- proxies so the pipeline can report on either model identically ----
    @property
    def epochs_done(self) -> int:
        return self.net.epochs_done

    @property
    def history(self) -> dict:
        return self.net.history

    @property
    def downsample(self) -> int:
        return self.net.downsample

    def n_params(self) -> int:
        return self.net.n_params()

    # -- motion ------------------------------------------------------------
    def motion(self, F: np.ndarray) -> np.ndarray:
        """(N, 2) capped (dy, dx) px/frame for a feature batch."""
        if not self.compensate:
            return np.zeros((F.shape[0], 2), dtype=np.float64)
        return _cap_motion(motion_from_features(F), self.max_motion_px)

    # -- training ----------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: np.ndarray | None = None, Y_val: np.ndarray | None = None,
            epochs: int | None = None, batch: int | None = None,
            max_seconds: float | None = None, verbose: bool = True,
            checkpoint: str | None = None) -> dict:
        """Compensate, then train the inner network. Same signature as the net.

        The compensation is recomputed on every call rather than cached on the
        instance. Chunked training calls fit() repeatedly, and a cache keyed on
        anything other than the array contents would eventually be served a
        stale motion field for a different batch -- a bug that would show up as
        an unexplained skill drop and would be very hard to find. Optical flow
        on this data costs ~10 ms per sample, so recomputation is a few seconds
        per chunk and buys correctness outright.
        """
        t0 = time.time()
        mo = self.motion(X)
        Fc = compensate_inputs(X, mo) if self.compensate else X
        Yc = (compensate_targets(Y, mo, self.target_binarise)
              if self.compensate
              else (Y if Y.dtype == bool else Y >= C.LIGHTNING_THRESHOLD))

        Fv = Yv = None
        if X_val is not None and Y_val is not None:
            mv = self.motion(X_val)
            Fv = compensate_inputs(X_val, mv) if self.compensate else X_val
            Yv = (compensate_targets(Y_val, mv, self.target_binarise)
                  if self.compensate
                  else (Y_val if Y_val.dtype == bool
                        else Y_val >= C.LIGHTNING_THRESHOLD))

        if verbose:
            spd = np.sqrt((mo ** 2).sum(axis=1))
            kept = float(Yc.sum()) / max(float(
                (np.asarray(Y) >= C.LIGHTNING_THRESHOLD).sum()), 1.0)
            print(f"  motion-compensated: |v| mean {spd.mean():.2f} "
                  f"max {spd.max():.2f} px/frame; "
                  f"{kept:.1%} of target positives survive the warp "
                  f"(rest advected outside the domain); "
                  f"prep {time.time() - t0:.1f}s")

        net_ckpt = None
        if checkpoint:
            net_ckpt = self._net_path(checkpoint)
            # Write the wrapper metadata first so a run killed mid-epoch still
            # leaves a loadable pair on disk: the net checkpoint is rewritten by
            # net.fit after every epoch, and the meta file never changes.
            self._save_meta(checkpoint)
        hist = self.net.fit(Fc, Yc, Fv, Yv, epochs=epochs, batch=batch,
                            max_seconds=max_seconds, verbose=verbose,
                            checkpoint=net_ckpt)
        if checkpoint:
            self._save_meta(checkpoint)
        return hist

    # -- inference ---------------------------------------------------------
    def predict(self, X: np.ndarray, batch: int = 8) -> np.ndarray:
        """Earth-relative probabilities, (N, T_out, H, W). Same API as the net.

        Three steps: forward warp of the inputs, the network, backward warp of
        the output. The two warps use one motion estimate, so they compose to
        the identity on position and the network's job reduces to the change in
        electrification probability along the trajectory.
        """
        mo = self.motion(X)
        Fc = compensate_inputs(X, mo) if self.compensate else X
        Pc = self.net.predict(Fc, batch=batch)
        if not self.compensate:
            return Pc
        return decompensate_probs(Pc, mo)

    def eval_loss(self, X: np.ndarray, Y: np.ndarray, batch: int = 8) -> float:
        """Weighted BCE in the Lagrangian frame (the frame it is trained in)."""
        mo = self.motion(X)
        Fc = compensate_inputs(X, mo) if self.compensate else X
        Yc = (compensate_targets(Y, mo, self.target_binarise)
              if self.compensate
              else (Y if Y.dtype == bool else Y >= C.LIGHTNING_THRESHOLD))
        return self.net.eval_loss(Fc, Yc, batch=batch)

    # -- checkpointing -----------------------------------------------------
    @staticmethod
    def _net_path(path: str) -> str:
        base = path[:-4] if path.endswith(".npz") else path
        return base + "_net.npz"

    @staticmethod
    def _meta_path(path: str) -> str:
        base = path[:-4] if path.endswith(".npz") else path
        return base + "_hybrid.json"

    def _save_meta(self, path: str) -> str:
        meta = {
            "kind": "MotionCompensatedNowcaster",
            "max_motion_px": self.max_motion_px,
            "compensate": self.compensate,
            "target_binarise": self.target_binarise,
            "net_file": os.path.basename(self._net_path(path)),
        }
        mp = self._meta_path(path)
        with open(mp, "w") as fh:
            json.dump(meta, fh, indent=2)
        return mp

    def save(self, path: str) -> str:
        """Write the inner net checkpoint plus a small JSON of wrapper state.

        Two files rather than one because ConvLSTMNowcaster.save owns its npz
        layout and reaching into it to add keys would couple this file to that
        one's internals. The net file name is stored as a BASENAME so the pair
        can be moved between directories together.
        """
        self.net.save(self._net_path(path))
        return self._save_meta(path)

    @classmethod
    def load(cls, path: str) -> "MotionCompensatedNowcaster":
        with open(cls._meta_path(path)) as fh:
            meta = json.load(fh)
        net_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                meta["net_file"])
        net = ConvLSTMNowcaster.load(net_path)
        m = cls(in_ch=net.in_ch, t_out=net.t_out,
                max_motion_px=float(meta["max_motion_px"]),
                compensate=bool(meta["compensate"]),
                target_binarise=float(meta["target_binarise"]))
        m.net = net
        m.in_ch, m.t_out = net.in_ch, net.t_out
        return m


# ---------------------------------------------------------------------------
# Calibrated blend of advection and a learned model
# ---------------------------------------------------------------------------

def _pav_nonincreasing(v: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators projection onto non-increasing sequences.

    Used only when `enforce_monotone` is on. Implemented directly because there
    is no sklearn here; the arrays are 12 long so an O(n^2) pooling loop is
    irrelevant to runtime.
    """
    out = [float(x) for x in v]
    i = 0
    while i < len(out) - 1:
        if out[i] < out[i + 1] - 1e-12:
            # Pool the violating pair (and then re-check backwards, because
            # pooling can create a new violation with the previous block).
            j = i
            while j >= 0 and out[j] < out[j + 1] - 1e-12:
                m = 0.5 * (out[j] + out[j + 1])
                out[j] = out[j + 1] = m
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    return np.asarray(out, dtype=np.float64)


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2, dtype=np.float64))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


class BlendedNowcaster:
    """Lead-time-weighted convex blend of advection and a learned model.

        p_blend(k) = w_k * p_advection(k) + (1 - w_k) * p_learned(k)
        p_final    = sigmoid(a * logit(p_blend) + b)

    The blend is convex in probability space rather than in logit space. Logit
    blending is the more usual choice for combining classifiers, but here one of
    the two components (advection) legitimately emits exact zeros over most of
    the domain -- there is no echo to extrapolate -- and logit(0) is -inf, so a
    logit blend would need a clipping constant that then dominates the result
    wherever advection is zero. A convex combination in probability space has
    the further property that its calibration is bounded by the calibration of
    its inputs, so the blend cannot be worse calibrated than both parents.

    (a, b) is a Platt-style recalibration fitted after blending. Convex mixing
    pulls sharp probabilities towards the middle, which costs resolution at the
    operating threshold; one global two-parameter rescaling recovers most of it.
    It is fitted by grid search because there is no optimiser available, and a
    5x7 grid on a proper score is plenty for two parameters.

    EVERYTHING here is fitted on VALIDATION and applied unchanged to test:
    12 weights plus 2 calibration parameters. Fitted on test they would be 14
    free parameters tuned on the reported number, which is how a model with no
    skill produces a table that looks like skill.
    """

    def __init__(self, learned: Any,
                 advection_fn: Callable[..., np.ndarray] | None = None,
                 n_out: int = C.OUTPUT_FRAMES,
                 weight_grid: np.ndarray | None = None,
                 smooth_weights: bool = True,
                 enforce_monotone: bool = False,
                 calibrate: bool = True,
                 label: str = "blend") -> None:
        self.learned = learned
        self.advection_fn = advection_fn or B.advection
        self.n_out = int(n_out)
        self.weight_grid = (np.linspace(0.0, 1.0, 21)
                            if weight_grid is None else np.asarray(weight_grid))
        self.smooth_weights = bool(smooth_weights)
        self.enforce_monotone = bool(enforce_monotone)
        self.calibrate = bool(calibrate)
        self.label = label
        # Start at "advection only". If fit() is never called, predict() then
        # returns the baseline rather than an arbitrary half-and-half mix, so a
        # forgotten fit shows up as "identical to advection" instead of as a
        # plausible-but-meaningless new column in the results table.
        self.w = np.ones(self.n_out, dtype=np.float64)
        self.w_raw = self.w.copy()
        self.cal_a, self.cal_b = 1.0, 0.0
        self.fit_info: dict = {}

    # -- component probabilities ------------------------------------------
    def components(self, X_raw: np.ndarray, F: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
        """(p_advection, p_learned), both (N, T_out, H, W).

        Advection needs the RAW channels (it advects the observed lightning
        field), the learned model needs the FEATURES. That is why this class
        does not mirror ConvLSTMNowcaster's single-array API: pretending it
        could would mean recovering the raw lightning channel by inverting the
        feature normalisation, which is possible but is exactly the sort of
        implicit coupling that breaks silently when a feature is inserted.
        """
        p_adv = self.advection_fn(X_raw, self.n_out)
        p_net = self.learned.predict(F)
        return np.asarray(p_adv, np.float64), np.asarray(p_net, np.float64)

    def blend(self, p_adv: np.ndarray, p_net: np.ndarray) -> np.ndarray:
        w = self.w.reshape(1, -1, 1, 1)
        p = w * p_adv + (1.0 - w) * p_net
        if self.calibrate and (self.cal_a != 1.0 or self.cal_b != 0.0):
            p = sigmoid(self.cal_a * _logit(p) + self.cal_b)
        return np.clip(p, 0.0, 1.0)

    # -- fitting, on validation only ---------------------------------------
    def fit(self, X_raw_val: np.ndarray, F_val: np.ndarray, Y_val: np.ndarray,
            verbose: bool = True) -> dict:
        """Choose per-lead weights and the calibration on the validation split.

        The weights are chosen to minimise the BRIER score, not to maximise CSI.
        Brier is a proper score, so its minimiser is the honest probability;
        CSI is defined only at a threshold, and at a 1% base rate the CSI
        surface over w is a noisy step function whose argmax jumps between
        neighbouring grid points from run to run. Selecting a probability blend
        on a threshold metric also entangles the blend with the threshold
        selection that happens later in the pipeline, and the two would then
        have to be optimised jointly.
        """
        t0 = time.time()
        p_adv, p_net = self.components(X_raw_val, F_val)
        y = (np.asarray(Y_val) >= C.LIGHTNING_THRESHOLD).astype(np.float64)

        raw = np.empty(self.n_out, dtype=np.float64)
        per_lead: list[dict] = []
        for k in range(self.n_out):
            best_w, best_bs = 1.0, np.inf
            for w in self.weight_grid:
                bs = _brier(w * p_adv[:, k] + (1.0 - w) * p_net[:, k], y[:, k])
                if bs < best_bs:
                    best_bs, best_w = bs, float(w)
            raw[k] = best_w
            per_lead.append({
                "lead_min": (k + 1) * C.TIMESTEP_MIN,
                "w_advection_raw": best_w,
                "brier_blend": best_bs,
                "brier_advection": _brier(p_adv[:, k], y[:, k]),
                "brier_learned": _brier(p_net[:, k], y[:, k]),
            })

        self.w_raw = raw.copy()
        w = raw.copy()
        if self.smooth_weights and self.n_out >= 3:
            # 3-point moving average. Each weight is fitted from one lead time's
            # worth of validation pixels, so the raw sequence is noisy; the
            # underlying dependence on lead time is physically smooth, and
            # smoothing trades a little validation fit for variance we would
            # otherwise carry straight into test.
            pad = np.concatenate(([w[0]], w, [w[-1]]))
            w = (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0
        if self.enforce_monotone:
            w = _pav_nonincreasing(w)
        self.w = np.clip(w, 0.0, 1.0)

        # ---- calibration, after the weights are fixed --------------------
        self.cal_a, self.cal_b = 1.0, 0.0
        p_mix = self.blend(p_adv, p_net)
        bs0 = _brier(p_mix, y)
        best = (bs0, 1.0, 0.0)
        if self.calibrate:
            z = _logit(np.clip(self.w.reshape(1, -1, 1, 1) * p_adv
                               + (1.0 - self.w.reshape(1, -1, 1, 1)) * p_net,
                               0.0, 1.0))
            for a in (0.6, 0.8, 1.0, 1.25, 1.6, 2.0):
                for b in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
                    bs = _brier(sigmoid(a * z + b), y)
                    if bs < best[0]:
                        best = (bs, a, b)
            self.cal_a, self.cal_b = best[1], best[2]

        # Does the fitted weighting actually follow the physical expectation
        # (advection dominant early, learned model later)? Reported, not
        # assumed. A flat or rising profile is informative -- it says the
        # learned model adds nothing at any lead, or nothing early.
        slope = float(np.polyfit(np.arange(self.n_out), self.w, 1)[0])
        self.fit_info = {
            "per_lead": per_lead,
            "w_raw": self.w_raw.tolist(),
            "w_used": self.w.tolist(),
            "cal_a": self.cal_a, "cal_b": self.cal_b,
            "brier_uncalibrated": bs0,
            "brier_calibrated": best[0],
            "brier_advection_only": _brier(p_adv, y),
            "brier_learned_only": _brier(p_net, y),
            "weight_slope_per_step": slope,
            "advection_weight_declines_with_lead": bool(slope < 0),
            "fit_seconds": time.time() - t0,
            "fitted_on": "validation split only",
        }
        if verbose:
            print(f"  blend fitted on VALIDATION ({F_val.shape[0]} samples, "
                  f"{time.time() - t0:.1f}s)")
            print("    advection weight by lead (min): "
                  + " ".join(f"{(k + 1) * C.TIMESTEP_MIN}:{self.w[k]:.2f}"
                             for k in range(self.n_out)))
            print(f"    weight slope {slope:+.4f}/step -> advection weight "
                  f"{'declines' if slope < 0 else 'does NOT decline'} with lead"
                  f" (physical expectation: declines)")
            print(f"    Brier: advection {self.fit_info['brier_advection_only']:.6f}"
                  f"  learned {self.fit_info['brier_learned_only']:.6f}"
                  f"  blend {bs0:.6f}  calibrated {best[0]:.6f}"
                  f"  (a={self.cal_a}, b={self.cal_b})")
        return self.fit_info

    def predict(self, X_raw: np.ndarray, F: np.ndarray) -> np.ndarray:
        p_adv, p_net = self.components(X_raw, F)
        return self.blend(p_adv, p_net)

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> str:
        """Save the blend parameters only; the learned model saves separately.

        Storing a copy of the network inside the blend file would mean two
        divergent copies of the same weights on disk, and the blend is 14
        numbers.
        """
        if not path.endswith(".json"):
            path = path + ".json"
        with open(path, "w") as fh:
            json.dump({
                "kind": "BlendedNowcaster",
                "label": self.label,
                "n_out": self.n_out,
                "w": self.w.tolist(), "w_raw": self.w_raw.tolist(),
                "cal_a": self.cal_a, "cal_b": self.cal_b,
                "calibrate": self.calibrate,
                "smooth_weights": self.smooth_weights,
                "enforce_monotone": self.enforce_monotone,
                "fit_info": self.fit_info,
            }, fh, indent=2, default=float)
        return path

    @classmethod
    def load(cls, path: str, learned: Any,
             advection_fn: Callable[..., np.ndarray] | None = None
             ) -> "BlendedNowcaster":
        if not path.endswith(".json"):
            path = path + ".json"
        with open(path) as fh:
            d = json.load(fh)
        m = cls(learned, advection_fn=advection_fn, n_out=int(d["n_out"]),
                smooth_weights=bool(d["smooth_weights"]),
                enforce_monotone=bool(d["enforce_monotone"]),
                calibrate=bool(d["calibrate"]), label=d.get("label", "blend"))
        m.w = np.asarray(d["w"], dtype=np.float64)
        m.w_raw = np.asarray(d["w_raw"], dtype=np.float64)
        m.cal_a, m.cal_b = float(d["cal_a"]), float(d["cal_b"])
        m.fit_info = d.get("fit_info", {})
        return m


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_warp_agreement() -> None:
    """_shift_stack must agree EXACTLY with baselines._shift_field."""
    rng = np.random.default_rng(0)
    a = rng.random((3, 4, 64, 64))
    worst = 0.0
    for dy, dx in ((2.5, -1.3), (-7.0, 11.25), (0.5, 0.5), (-0.2, 0.0)):
        got = _shift_stack(a, dy, dx)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ref = B._shift_field(a[i, j], dy, dx)
                worst = max(worst, float(np.abs(got[i, j] - ref).max()))
    print(f"  _shift_stack vs baselines._shift_field: max abs diff {worst:.3e}"
          f"  ({'exact' if worst == 0.0 else 'MISMATCH'})")
    assert worst == 0.0, "vectorised warp diverges from the reference warp"


def _test_roundtrip() -> None:
    """Warp forward then back must recover the field away from the boundary.

    The interior/boundary distinction is not a fudge: zero-fill genuinely
    destroys the strip whose source was off-grid, and a round trip cannot
    invent it back. Requiring exact recovery everywhere would only be possible
    with wrap-around, which is physically wrong (a storm leaving the east edge
    does not reappear in the west). So the test asserts near-exact recovery in
    the interior and reports the boundary error separately.

    Integer displacements round-trip to machine precision. Fractional ones do
    not: bilinear interpolation is a low-pass filter, applying it twice blurs
    twice, and no inverse exists. The tolerance below reflects that, and the
    integer case is checked separately so an actual indexing error cannot hide
    behind the interpolation tolerance.
    """
    rng = np.random.default_rng(1)
    # Smooth field: a real reflectivity field is spatially correlated, and
    # bilinear resampling error on white noise is dominated by the noise, which
    # would make the tolerance meaningless.
    base = rng.random((64, 64))
    from synth import _box_blur
    field = _box_blur(base, 7) * 50.0

    print("  warp round trip (forward then inverse):")
    for dy, dx, tol in ((3.0, -2.0, 1e-12), (2.5, -1.5, 2.0), (-1.25, 4.75, 2.0)):
        fwd = _shift_stack(field, dy, dx)
        back = _shift_stack(fwd, -dy, -dx)
        pad = int(np.ceil(max(abs(dy), abs(dx)))) + 2
        interior = np.s_[pad:-pad, pad:-pad]
        err_in = float(np.abs(back[interior] - field[interior]).max())
        rel_in = err_in / float(np.abs(field).max())
        edge = np.abs(back - field).max()
        kind = "integer" if (dy == int(dy) and dx == int(dx)) else "fractional"
        print(f"    dy={dy:+.2f} dx={dx:+.2f} ({kind:10s}) interior max err "
              f"{err_in:.3e} ({rel_in:.2%} of field range), "
              f"whole-grid max err {edge:.3e} (boundary strip is "
              f"irrecoverable by construction)")
        assert err_in <= tol, (dy, dx, err_in, tol)


def _test_compensation_reduces_motion(F: np.ndarray, X: np.ndarray) -> None:
    """The point of compensation: storms become quasi-stationary.

    Measured as the mean absolute difference between each input frame and the
    LAST input frame on the reflectivity channel. If compensation works this
    falls, and it must fall most for the earliest frame (which is displaced
    most). A sign error here is otherwise invisible -- the model still trains
    and still produces plausible maps, just worse ones -- so the test also runs
    the INVERTED warp and requires it to make the mismatch worse. A test that
    cannot fail proves nothing.

    The comparison is restricted to the domain interior, and that restriction is
    not a fudge to make the number look good. Zero-fill deliberately writes 0.0
    into the strip whose source is off-grid; on `refl_sfc_n` the value 0.0
    de-normalises to 20 dBZ (config.CHANNEL_NORM offset), not to clear air, so
    the strip registers as a large change on this diagnostic even though it is
    the intended "unobserved" encoding. The boundary effect is real and is
    reported separately below rather than being averaged into the headline.
    """
    mo_raw = np.stack([B.estimate_motion(X[i, :, C.CH["refl_sfc"]])
                       for i in range(X.shape[0])])
    mo_feat = motion_from_features(F)
    d = np.abs(mo_raw - mo_feat).max()
    print(f"  motion from features vs from raw channels: max abs diff "
          f"{d:.3f} px/frame (mean |v| {np.sqrt((mo_raw ** 2).sum(1)).mean():.2f})")

    mo = _cap_motion(mo_feat, 4.0)
    Fc = compensate_inputs(F, mo)
    # Inverted warp, for the falsification arm of the test.
    r = F.shape[1] - 1
    Fw = np.empty_like(F)
    for i in range(F.shape[0]):
        for t in range(F.shape[1]):
            lag = r - t
            Fw[i, t] = (F[i, t] if lag == 0 else
                        _shift_stack(F[i, t], -mo[i, 0] * lag, -mo[i, 1] * lag))

    # Interior margin: wide enough to exclude the strip exposed by the largest
    # displacement in the batch, |v| * (T_in - 1).
    pad = int(np.ceil(float(np.abs(mo).max()) * r)) + 2
    pad = min(pad, min(F.shape[-2], F.shape[-1]) // 3)
    sl = np.s_[pad:-pad, pad:-pad]

    def mism(A: np.ndarray, t: int) -> float:
        return float(np.abs(A[:, t, 0][:, sl[0], sl[1]]
                            - A[:, r, 0][:, sl[0], sl[1]]).mean())

    print(f"  frame-to-reference mismatch on refl_sfc_n, interior only "
          f"(margin {pad} px), mean |F[t] - F[last]|:")
    gains: list[float] = []
    for t in range(r):
        before, after, wrong = mism(F, t), mism(Fc, t), mism(Fw, t)
        gains.append(before - after)
        print(f"    lag {r - t}: before {before:.4f} -> compensated "
              f"{after:.4f} ({(before - after) / max(before, 1e-9):+.1%})   "
              f"inverted warp {wrong:.4f} "
              f"({(before - wrong) / max(before, 1e-9):+.1%})")
        assert after < before, (
            f"compensation increased the mismatch at lag {r - t}: "
            f"check the warp sign")
        assert wrong > after, (
            f"the inverted warp is not worse at lag {r - t}: the diagnostic "
            f"cannot distinguish the two signs and proves nothing")
    # The earliest frame is displaced the most, so it must gain the most.
    print(f"    absolute gain: {gains[0]:.5f} at lag {r} vs {gains[-1]:.5f} at "
          f"lag 1 (displacement scales with lag, so the gain must too)")
    assert gains[0] > gains[-1]

    # Quantify the honest cost of zero-fill: how much of the grid is filled.
    frac = float(np.mean(np.abs(mo).max(axis=1) * r)) / F.shape[-1]
    print(f"  zero-filled boundary strip is ~{frac:.1%} of the grid width on "
          f"the earliest frame (unobserved upstream inflow, encoded exactly as "
          f"features.build_features encodes a radar outage: 0.0)")


def _self_test() -> None:
    import cv as CV
    import metrics as M
    import synth

    t_start = time.time()
    print("=" * 78)
    print("hybrid.py SELF-TEST")
    print("=" * 78)

    print("\n1. WARP PRIMITIVES")
    _test_warp_agreement()
    _test_roundtrip()

    print("\n2. DATA (tiny: 4 events, 24 frames each)")
    events = synth.generate_dataset(n_events=4, seed=13, frames_per_event=24)
    splits = CV.split_events(4, train_frac=0.5, val_frac=0.25, test_frac=0.25,
                             seed=1)
    sets = CV.build_split_sequences(events, splits, stride=3, val_stride=3)
    Xtr, Ytr = sets["train"]["X"], sets["train"]["Y"]
    Xva, Yva = sets["val"]["X"], sets["val"]["Y"]
    Ftr = FT.build_features_batch(Xtr)
    Fva = FT.build_features_batch(Xva)
    print(f"  train X {Xtr.shape} F {Ftr.shape} base rate "
          f"{(Ytr >= 1).mean():.4f}   val {Xva.shape[0]} samples "
          f"base rate {(Yva >= 1).mean():.4f}")

    print("\n3. MOTION COMPENSATION")
    _test_compensation_reduces_motion(Ftr, Xtr)

    print("\n4. TARGET WARP -- positives must survive, and stay BOOLEAN")
    mo = _cap_motion(motion_from_features(Ftr), 4.0)
    Yc = compensate_targets(Ytr, mo)
    n0 = int((Ytr >= C.LIGHTNING_THRESHOLD).sum())
    print(f"  dtype {Yc.dtype} (must be bool so _prepare passes it through)")
    print(f"  positives {n0:,} -> {int(Yc.sum()):,} "
          f"({int(Yc.sum()) / max(n0, 1):.1%} survive; the rest advect off-grid,"
          f" which zero-fill labels negative)")
    assert Yc.dtype == bool
    # The trap this guards: warping first and thresholding at
    # LIGHTNING_THRESHOLD=1 afterwards annihilates every interpolated positive.
    wrong = _shift_stack((Ytr[0, 0] >= 1).astype(np.float64), -0.5, -0.5)
    print(f"  counter-check: bilinear-warped binary field has max "
          f"{wrong.max():.3f}, so thresholding it at LIGHTNING_THRESHOLD="
          f"{C.LIGHTNING_THRESHOLD} would keep "
          f"{int((wrong >= C.LIGHTNING_THRESHOLD).sum())} of "
          f"{int((Ytr[0, 0] >= 1).sum())} positives")

    print("\n5. STRUCTURAL RECALL CEILING OF THE LAGRANGIAN COORDINATE CHANGE")
    # Push the observed target through compensate -> decompensate. Whatever
    # comes back is what the model could in principle predict; the rest are
    # guaranteed misses because their upstream history is off-grid. Reporting
    # this is what turns a long-lead deficit in the results table from a mystery
    # into a known, quantified property of the method.
    cov = lagrangian_coverage(Ytr, mo)
    for row in cov["per_lead"]:
        if row["lead_min"] % 15 == 0 or row["lead_min"] == 5:
            rf = row["recoverable_frac"]
            rf_s = "   n/a" if not np.isfinite(rf) else f"{rf:6.3f}"
            print(f"    lead {row['lead_min']:2d} min: grid coverage "
                  f"{row['grid_coverage_frac']:.3f};  reachable positives "
                  f"{rf_s} of {row['n_positive']:5d} observed")
    print(f"    overall: grid coverage {cov['overall_grid_coverage_frac']:.3f}, "
          f"reachable positives {cov['overall_recoverable_frac']:.3f}, "
          f"at mean |v| {np.sqrt((mo ** 2).sum(1)).mean():.2f} px/frame")
    print("    (the Eulerian model pays none of this cost -- it can emit a "
          "probability anywhere. That is a real advantage of Earth coordinates "
          "and is why BlendedNowcaster exists.)")
    lead1 = cov["per_lead"][0]["recoverable_frac"]
    assert lead1 > 0.8, (
        f"only {lead1:.2f} of 5-minute-lead positives are reachable; at one "
        f"step the displacement is |v| px, so this must be near 1 -- suspect "
        f"the lead-time indexing in the warp pair")
    # Coverage must fall with lead, because displacement grows with lead. Uses
    # the geometric measure, which is defined even for leads with no positives.
    gc = [r["grid_coverage_frac"] for r in cov["per_lead"]]
    assert gc[-1] < gc[0], (
        f"grid coverage did not fall with lead ({gc[0]:.3f} -> {gc[-1]:.3f}): "
        f"the warp is probably not scaling with lead time")

    print("\n6. FIT / PREDICT (a few seconds of training only)")
    model = MotionCompensatedNowcaster(in_ch=FT.N_FEATURES, hidden=8,
                                       downsample=2, batch=4, lr=6e-3, seed=2)
    print(f"  {model.n_params():,} params")
    hist = model.fit(Ftr, Ytr, Fva, Yva, epochs=2, max_seconds=18.0,
                     verbose=True)
    P = model.predict(Fva)
    print(f"  predict {P.shape} range [{P.min():.4f}, {P.max():.4f}] "
          f"mean {P.mean():.4f}")
    assert P.shape == Yva.shape
    assert np.isfinite(P).all() and P.min() >= 0.0 and P.max() <= 1.0
    ap = M.auprc(P, Yva >= 1)
    print(f"  val AUPRC {ap:.4f} vs base rate {(Yva >= 1).mean():.4f} "
          f"-- {hist['epochs_done']} epochs, not a result")

    print("\n7. SAVE / LOAD ROUND TRIP")
    import tempfile
    ck = os.path.join(tempfile.gettempdir(), "hybrid_selftest.npz")
    model.save(ck)
    m2 = MotionCompensatedNowcaster.load(ck)
    d = float(np.abs(m2.predict(Fva) - P).max())
    print(f"  max |dprob| after save/load: {d:.2e}")
    assert d == 0.0

    print("\n8. BLENDED NOWCASTER (weights fitted on validation only)")
    bl = BlendedNowcaster(model)
    info = bl.fit(Xva, Fva, Yva, verbose=True)
    Pb = bl.predict(Xva, Fva)
    assert Pb.shape == Yva.shape and np.isfinite(Pb).all()
    print(f"  blend predict {Pb.shape} range [{Pb.min():.4f}, {Pb.max():.4f}]")
    bp = os.path.join(tempfile.gettempdir(), "hybrid_selftest_blend.json")
    bl.save(bp)
    bl2 = BlendedNowcaster.load(bp, learned=m2)
    d2 = float(np.abs(bl2.predict(Xva, Fva) - Pb).max())
    print(f"  max |dprob| after blend save/load: {d2:.2e}")
    assert d2 == 0.0
    # Monotone projection sanity: the PAV output must be non-increasing.
    mono = _pav_nonincreasing(np.array([0.2, 0.9, 0.4, 0.8, 0.1]))
    print(f"  PAV non-increasing projection: {np.round(mono, 3).tolist()}")
    assert np.all(np.diff(mono) <= 1e-9)
    assert info["fitted_on"] == "validation split only"

    print(f"\nALL HYBRID ASSERTIONS PASSED in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    _self_test()

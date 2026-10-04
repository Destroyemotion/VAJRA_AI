"""
Verification suite for the lightning nowcasting system. No pytest available.

WHY THIS FILE EXISTS
--------------------
Every number this project reports is worthless unless two things are true:
the metrics compute what they claim to compute, and no feature, split or model
input can see the future. Both failures are silent. A leaking pipeline does not
crash -- it produces a beautiful CSI and a paper-shaped result that is entirely
fake. So the properties are asserted here, mechanically, against hand-computed
values and against controlled perturbation experiments.

WHAT EACH GROUP CATCHES
-----------------------
A. METRIC ANCHORS. Exact hand-computed values on tiny arrays. Catches the
   ordinary arithmetic slips that a smoke test cannot see: a POD/FAR swap, an
   off-by-one in the AUPRC recall increments, trapezoidal PR interpolation
   sneaking in (which inflates AP for rare events), a tie-handling bug in AUROC
   (which silently rewards a constant forecast), a sign error in the Brier
   reliability term. Also anchors FSS at its two fixed points -- exactly 1 for a
   perfect forecast, exactly 0 for a spatially disjoint one -- and at the
   closed-form no-skill value for a climatological constant, 2c/(1+c) at
   radius 0. And it pins the neighbourhood dilation used by the new-initiation
   mask to a NON-WRAPPING implementation: np.roll would let an event on the left
   edge mark the right edge as "lightning nearby", quietly deleting real
   domain-edge pixels from the single most important diagnostic in the project.

B. CAUSALITY AND LEAKAGE PROOFS. The most important group. Four independent
   experiments, all of the same form: change something the pipeline must not be
   able to see, and require the output to be BITWISE identical.
     1. Perturb every value in the future frames of a raw sequence. Every
        feature at earlier frames must be unchanged. This is what catches a
        centred temporal difference, a whole-sequence normalisation, or a
        forward-looking running statistic. A local centred-difference
        implementation is run through the same check to prove the check fires.
     2. Truncate the future instead of perturbing it. Features aligned at the
        start of the sequence must not depend on how much data follows.
     3. Split integrity: no event in two splits, and window arithmetic verified
        by tagging every frame with its own index so an off-by-one in t0 cannot
        hide. The leakage a naive sample-level split WOULD cause is measured,
        not asserted from a docstring.
     4. Model input path: perturb the raw frames of the TARGET window and
        require the model's probabilities to be bitwise identical, and require
        the new-initiation mask to be a pure function of the input window.
        Plus batch invariance -- a prediction that changes when its batch-mates
        change means some statistic is being pooled across samples, which on a
        test set is leakage.

C. GENERATOR HEALTH. A synthetic benchmark is worthless if it is easy. The
   specific shortcut this project is built to avoid is "strong surface echo =>
   lightning". If the storm-conditional AUROC of surface reflectivity ever
   approaches that of charging-layer reflectivity, the two populations have
   stopped overlapping and every downstream score is measuring a shortcut.

D. MODEL SANITY. The analytic BPTT gradient is checked against central finite
   differences in float64, then the network is trained for a few seconds and
   required to beat the best CONSTANT forecast under the same weighted loss.
   "Loss went down" is not evidence: most of the early decrease is the model
   discovering the base rate, which a single scalar can do.

E. BASELINES HONOUR THE RULES. On the new-initiation subset, BOTH Eulerian
   persistence and optical-flow advection must score at the base-rate floor
   with ~zero CSI, at every lead time. That is the entire basis of the claim
   that skill on this subset is genuine rather than a persistence echo, so it
   is asserted, not merely reported. Climatology AUPRC must equal its own base
   rate to machine precision, which validates the AUPRC implementation from
   the other direction.

   The advection half of that assertion USED TO FAIL, and the failure was real:
   metrics.new_initiation_mask excluded a STATIC 8 km neighbourhood while the
   storms move, so from 10 min lead onward the "new initiation" subset
   contained pixels merely downstream of existing lightning and advection
   reached 16x lift. metrics.evaluate now uses the Lagrangian swept-corridor
   exclusion instead. The test still computes the static mask alongside, so it
   reports both numbers and the fix is demonstrated by measurement rather than
   assumed.

OUTCOMES
--------
PASS / FAIL / KNOWN-ISSUE. The third is reserved for a property the code under
test DOCUMENTS but does not deliver, where the gap is judged tolerable and
measured so it cannot be forgotten. It is NOT a place to park a broken
correctness guarantee: anything that would invalidate a reported metric is a
FAIL. Known issues are printed loudly in the summary and do not set the exit
code; only FAIL does.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import numpy as np

import baselines as B
import config as C
import convlstm_np as NN
import cv as CV
import features as FT
import metrics as M
import synth


# ---------------------------------------------------------------------------
# Minimal test registry (no pytest in this environment)
# ---------------------------------------------------------------------------

class KnownIssue(Exception):
    """A documented property that the code under test does not actually deliver.

    Raised instead of AssertionError when the measurement is correct and
    reproducible but contradicts a claim in the source. Reported separately so
    that weakening a test and finding a real defect are never confused.
    """


SECTIONS: list[str] = [
    "A. METRIC ANCHORS (hand-computed exact values)",
    "B. CAUSALITY / LEAKAGE PROOFS (perturb the future, require no change)",
    "C. GENERATOR HEALTH AND DIFFICULTY (no surface-reflectivity shortcut)",
    "D. MODEL SANITY (gradients correct, learns more than a scalar)",
    "E. BASELINES HONOUR THE RULES (new-initiation subset is clean)",
]

_REGISTRY: list[dict[str, Any]] = []


def test(section: str, name: str) -> Callable:
    """Register a test. The function returns a one-line evidence string."""
    def deco(fn: Callable[[], str]) -> Callable[[], str]:
        _REGISTRY.append({"section": section, "name": name, "fn": fn,
                          "doc": (fn.__doc__ or "").strip().split("\n")[0]})
        return fn
    return deco


def req(cond: bool, msg: str) -> None:
    """Assertion with a mandatory explanation of the failure mode."""
    if not cond:
        raise AssertionError(msg)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return bool(abs(float(a) - float(b)) <= tol)


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality, checked twice: array compare and raw byte compare.

    np.allclose with atol=0 still applies rtol; array_equal is exact but treats
    NaN as unequal, which is the behaviour we want here (features are
    nan-scrubbed, so a NaN is itself a bug). The byte comparison additionally
    catches a -0.0/+0.0 or dtype difference that array_equal would forgive.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b)) and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# Shared fixtures (generated once; the whole suite must run in well under 90 s)
# ---------------------------------------------------------------------------

_FIX: dict[str, Any] = {}

# Deliberately the project's configured seed, not a seed chosen because it
# passes: these checks are about the dataset the project actually reports on.
FIX_SEED = C.SYNTH["seed"]
FIX_EVENTS = 10
FIX_FRAMES = 36
FIX_STRIDE = 6


def fx_events() -> list[dict]:
    if "events" not in _FIX:
        _FIX["events"] = synth.generate_dataset(
            n_events=FIX_EVENTS, seed=FIX_SEED, frames_per_event=FIX_FRAMES)
    return _FIX["events"]


def fx_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supervised windows cut from the fixture events, event-blocked layout."""
    if "windows" not in _FIX:
        _FIX["windows"] = CV.make_sequences(
            fx_events(), range(FIX_EVENTS), stride=FIX_STRIDE)
    return _FIX["windows"]


def fx_ni() -> dict[str, Any]:
    """New-initiation masks and base rates for the fixture windows.

    Both masks are built: `niv` is the Lagrangian swept-corridor exclusion the
    pipeline now scores against, and `niv_static` is the old static-radius one,
    kept so the regression test can show the two side by side rather than
    asserting the historical bug is gone on trust.
    """
    if "ni" not in _FIX:
        X, Y, _ = fx_windows()
        yb = Y >= C.LIGHTNING_THRESHOLD
        niv = M.new_initiation_mask_lagrangian(X, yb.shape[1])   # (N, T, H, W)
        clean = M.new_initiation_mask(X)                          # (N, H, W)
        niv_static = np.broadcast_to(clean[:, None], yb.shape)
        _FIX["ni"] = {
            "yb": yb, "niv": niv, "niv_static": niv_static,
            "base_all": float(yb.mean()),
            "base_ni": float(yb[niv].mean()),
            "base_ni_static": float(yb[niv_static].mean()),
            "n_pos_ni": int(np.count_nonzero(yb & niv)),
        }
    return _FIX["ni"]


def fx_baseline_probs() -> dict[str, np.ndarray]:
    if "bprobs" not in _FIX:
        X, _, _ = fx_windows()
        _FIX["bprobs"] = B.run_all(X, C.OUTPUT_FRAMES,
                                   base_rate=fx_ni()["base_all"])
    return _FIX["bprobs"]


# ===========================================================================
# A. METRIC ANCHORS
# ===========================================================================

SEC_A = SECTIONS[0]


@test(SEC_A, "contingency_exact")
def t_contingency_exact() -> str:
    """All 2x2 scores against hand arithmetic; catches POD/FAR/CSI/HSS slips."""
    # hits=3, false alarms=2, misses=4, correct negatives=11, n=20.
    pred = np.array([1] * 3 + [1] * 2 + [0] * 4 + [0] * 11, bool)
    obs = np.array([1] * 3 + [0] * 2 + [1] * 4 + [0] * 11, bool)

    # Hand arithmetic, written as fractions so the expected value cannot be
    # copied back from a failing run.
    hits, fa, miss, cn, n = 3, 2, 4, 11, 20
    exp_correct = (hits + miss) * (hits + fa) / n + (cn + miss) * (cn + fa) / n
    want = {
        "hits": hits, "false_alarms": fa, "misses": miss,
        "correct_negatives": cn, "n": n,
        "POD": 3 / 7, "FAR": 2 / 5, "CSI": 3 / 9, "BIAS": 5 / 7,
        "HSS": (hits + cn - exp_correct) / (n - exp_correct),   # 2.5/8.5
        "ETS": (hits - 1.75) / (9 - 1.75),                      # 1.25/7.25
        "obs_rate": 7 / 20, "pred_rate": 5 / 20,
    }
    got = M.contingency(pred, obs)
    for k, v in want.items():
        req(close(got[k], v, 1e-12),
            f"contingency[{k}] = {got[k]!r}, hand-computed {v!r}")

    # exp_correct = 35/20 + 195/20 = 11.5, so HSS = 5/17 and ETS = 5/29.
    req(close(got["HSS"], 5 / 17, 1e-12), f"HSS {got['HSS']} != 5/17")
    req(close(got["ETS"], 5 / 29, 1e-12), f"ETS {got['ETS']} != 5/29")

    # The `valid` mask must actually mask. Append 30 elements of garbage that
    # would wreck every score if they were scored, then mask them out and
    # require the identical answer. A valid= argument silently ignored is a
    # leak of unscoreable pixels into every reported number.
    pad_p = np.array([1] * 15 + [0] * 15, bool)
    pad_o = np.array([0] * 15 + [1] * 15, bool)
    pred2 = np.concatenate([pred, pad_p])
    obs2 = np.concatenate([obs, pad_o])
    valid = np.concatenate([np.ones(20, bool), np.zeros(30, bool)])
    got2 = M.contingency(pred2, obs2, valid)
    for k, v in want.items():
        req(close(got2[k], v, 1e-12),
            f"valid mask ignored: contingency[{k}] = {got2[k]!r} != {v!r}")

    # Degenerate case: no observations and no predictions -> nan, not 0.0.
    # Returning 0.0 would let an empty subset silently drag a mean CSI down.
    z = M.contingency(np.zeros(5, bool), np.zeros(5, bool))
    req(np.isnan(z["CSI"]) and np.isnan(z["POD"]),
        "empty contingency returned a number instead of nan")
    return (f"hits/fa/miss/cn = 3/2/4/11 exact; POD=3/7 FAR=2/5 CSI=1/3 "
            f"BIAS=5/7 HSS=5/17 ETS=5/29 to 1e-12; valid mask honoured; "
            f"empty subset -> nan")


@test(SEC_A, "fss_fixed_points")
def t_fss_fixed_points() -> str:
    """FSS = 1 identical, exactly 0 disjoint, 2c/(1+c) for a constant."""
    obs = np.zeros((8, 8))
    obs[3, 3] = 1.0
    obs[3, 4] = 1.0

    # Perfect forecast: numerator is identically zero at every radius.
    for r in (0, 1, 2, 4):
        req(close(M.fss(obs, obs, r), 1.0, 1e-12),
            f"FSS(perfect, r={r}) = {M.fss(obs, obs, r)} != 1.0")

    # Spatially disjoint binary forecast at radius 0 is the worst case that
    # defines the denominator, so FSS is exactly 0. This is the anchor that
    # catches a wrong reference term (e.g. using MSE of the raw fields).
    dis = np.zeros((8, 8))
    dis[3, 6] = 1.0
    dis[3, 7] = 1.0
    req(close(M.fss(dis, obs, 0), 0.0, 1e-12),
        f"FSS(disjoint, r=0) = {M.fss(dis, obs, 0)} != 0.0")

    # No-skill climatological constant. With p = c = mean(o) and radius 0,
    #   num = mean((c-o)^2) = c(1-c),  den = c^2 + c
    #   FSS = 1 - c(1-c)/(c(c+1)) = 2c/(1+c)
    rng = np.random.default_rng(0)
    ob = (rng.random((40, 40)) < 0.05).astype(np.float64)
    c = float(ob.mean())
    got = M.fss(np.full_like(ob, c), ob, 0)
    want = 2 * c / (1 + c)
    req(close(got, want, 1e-12),
        f"FSS(constant at base rate, r=0) = {got} != 2c/(1+c) = {want}")
    req(got < 0.15, f"no-skill FSS {got} is not near zero")

    # Double-penalty relief: a 3 px displacement scores 0 at radius 0 but must
    # earn credit at a radius that contains the error. If FSS did not rise with
    # radius, the whole scale-dependent reading of the metric is broken.
    r0, r4 = M.fss(dis, obs, 0), M.fss(dis, obs, 4)
    req(r0 == 0.0 and r4 > 0.5,
        f"FSS gives no neighbourhood credit: r0={r0:.4f} r4={r4:.4f}")
    return (f"FSS=1 identical (r=0,1,2,4); FSS=0 disjoint exact; constant "
            f"forecast = 2c/(1+c) = {want:.6f} to 1e-12; 3px displacement "
            f"r0={r0:.3f} -> r4={r4:.3f}")


@test(SEC_A, "auprc_exact")
def t_auprc_exact() -> str:
    """AP by hand on a known ranking; step-wise, no optimistic interpolation."""
    scores = np.arange(10, 0, -1).astype(np.float64)     # strictly decreasing

    # Positives at ranks 1, 3, 4, 8 of the descending order.
    #   k=1: P=1/1, dR=1/4 -> 1/4
    #   k=3: P=2/3, dR=1/4 -> 1/6
    #   k=4: P=3/4, dR=1/4 -> 3/16
    #   k=8: P=4/8, dR=1/4 -> 1/8
    #   total = 12/48 + 8/48 + 9/48 + 6/48 = 35/48
    lab = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0], bool)
    got = M.auprc(scores, lab)
    req(close(got, 35 / 48, 1e-12), f"AP = {got}, hand-computed 35/48")

    # Positives exactly the top k: precision is 1 at every recall step, AP = 1.
    top = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    req(close(M.auprc(scores, top), 1.0, 1e-12),
        f"AP for a perfect ranking = {M.auprc(scores, top)} != 1.0")

    # All scores tied: one operating point, precision = base rate, recall = 1.
    # A tie-splitting bug shows up here as AP > base rate, which would make a
    # constant forecast look skilful -- the exact failure that would let the
    # climatology baseline appear to beat its own floor.
    tie = np.array([1] * 5 + [0] * 5, bool)
    req(close(M.auprc(np.ones(10), tie), 0.5, 1e-12),
        f"AP of a constant forecast = {M.auprc(np.ones(10), tie)} != base rate")

    # Reversed ranking: worst case, AP must be well below the base rate.
    rev = M.auprc(scores, np.array([0] * 6 + [1] * 4, bool))
    req(rev < 0.5, f"AP of an inverted ranking = {rev}, should be < base rate")

    # valid mask must exclude, not reweight.
    pad_s = np.full(10, 100.0)                   # would top the ranking
    pad_l = np.zeros(10, bool)                   # ... as false alarms
    got_v = M.auprc(np.concatenate([scores, pad_s]),
                    np.concatenate([lab, pad_l]),
                    np.concatenate([np.ones(10, bool), np.zeros(10, bool)]))
    req(close(got_v, 35 / 48, 1e-12),
        f"AP ignored the valid mask: {got_v} != 35/48")

    # No positives -> nan, never 0.0. A 0.0 would average into a comparison
    # table as a real (bad) score for a split that simply cannot be scored.
    req(np.isnan(M.auprc(scores, np.zeros(10, bool))),
        "AP with zero positives returned a number instead of nan")
    return (f"AP=35/48 on a hand-ranked case, 1.0 for top-k, exactly the base "
            f"rate under full ties, {rev:.4f} inverted; valid mask honoured; "
            f"nan with no positives")


@test(SEC_A, "auroc_exact")
def t_auroc_exact() -> str:
    """AUROC from the rank-sum identity, plus exact tie averaging."""
    scores = np.arange(10, 0, -1).astype(np.float64)
    # Positives carry scores 10, 9, 8, 6, 3 -> ascending ranks 10, 9, 8, 6, 3.
    # AUROC = (sum_ranks_pos - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    #       = (36 - 15) / 25 = 21/25 = 0.84
    lab = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], bool)
    got = M.auroc(scores, lab)
    req(close(got, 21 / 25, 1e-12), f"AUROC = {got}, hand-computed 0.84")

    # Fully tied scores must give exactly 0.5. Without mid-rank averaging this
    # returns 0.0 or 1.0 depending on argsort order, and a constant forecast
    # then looks either perfect or anti-perfect.
    tie = np.array([1] * 5 + [0] * 5, bool)
    req(close(M.auroc(np.ones(10), tie), 0.5, 1e-12),
        f"AUROC under full ties = {M.auroc(np.ones(10), tie)} != 0.5")

    # Perfect and perfectly inverted separations.
    req(close(M.auroc(scores, np.array([1] * 5 + [0] * 5, bool)), 1.0, 1e-12),
        "AUROC of a perfect separation != 1.0")
    req(close(M.auroc(scores, np.array([0] * 5 + [1] * 5, bool)), 0.0, 1e-12),
        "AUROC of an inverted separation != 0.0")
    return (f"AUROC = 21/25 = 0.84 exact from the rank sum; 0.5 under full "
            f"ties; 1.0 / 0.0 at the extremes")


@test(SEC_A, "brier_decomposition")
def t_brier_decomposition() -> str:
    """Reliability = 0 for a perfectly calibrated forecast; decomposition adds."""
    # Five bins, each holding one forecast value whose observed frequency
    # equals it exactly. Within-bin forecast variance is zero, so the Murphy
    # decomposition BS = reliability - resolution + uncertainty holds exactly.
    probs: list[float] = []
    obs: list[int] = []
    for pv, n in [(0.05, 100), (0.25, 100), (0.45, 100), (0.65, 100),
                  (0.85, 100)]:
        k = int(round(pv * n))
        probs += [pv] * n
        obs += [1] * k + [0] * (n - k)
    p = np.asarray(probs)
    o = np.asarray(obs, bool)
    bd = M.brier_decomposition(p, o)

    req(bd["reliability"] < 1e-12,
        f"perfectly calibrated forecast has reliability {bd['reliability']:.3e}, "
        f"must be 0 -- the reliability term is mis-binned or mis-signed")
    ident = bd["reliability"] - bd["resolution"] + bd["uncertainty"]
    req(close(ident, bd["brier"], 1e-12),
        f"decomposition does not sum: rel-res+unc = {ident:.12f} vs "
        f"BS = {bd['brier']:.12f}")
    # obar = 225/500 = 0.45, so uncertainty = 0.45*0.55 = 0.2475 exactly.
    req(close(bd["uncertainty"], 0.2475, 1e-12),
        f"uncertainty {bd['uncertainty']} != 0.45*0.55")
    req(bd["resolution"] > 0.05,
        f"resolution {bd['resolution']:.4f} too small for a forecast that "
        f"spans 0.05-0.85")

    # A badly miscalibrated forecast must produce a LARGE reliability term.
    # Without this the previous assertion is satisfied by a function that
    # always returns zero.
    bad = M.brier_decomposition(np.full(200, 0.9),
                                np.array([1] * 20 + [0] * 180, bool))
    req(close(bad["reliability"], 0.64, 1e-9),
        f"reliability of a 0.9 forecast verifying 0.1 = "
        f"{bad['reliability']:.6f}, hand-computed (0.9-0.1)^2 = 0.64")
    req(bad["bss"] < 0, f"BSS of a grossly miscalibrated forecast is "
                        f"{bad['bss']:.3f}, should be negative")
    return (f"reliability {bd['reliability']:.2e} (~0) when calibrated, exactly "
            f"0.64 when 0.9 verifies at 0.1; BS = rel-res+unc to 1e-12; "
            f"uncertainty = 0.2475 exact")


@test(SEC_A, "new_initiation_mask_no_wrap")
def t_ni_mask_no_wrap() -> str:
    """Dilation is edge-clamped, not circular; exact count of excluded pixels."""
    h = w = 12
    t_in = 4
    radius = 4
    ch = C.CH["light_dens"]

    # One strike on the LEFT edge, mid-height.
    x = np.zeros((t_in, C.N_CHANNELS, h, w), np.float32)
    x[1, ch, 6, 0] = 3.0
    mask = M.new_initiation_mask(x, radius=radius)
    req(mask.shape == (h, w), f"mask shape {mask.shape} != {(h, w)}")

    # Excluded region is the clamped square rows 2..10 x cols 0..4 = 45 px.
    n_excl = int(np.count_nonzero(~mask))
    req(n_excl == 9 * 5,
        f"excluded {n_excl} pixels, the clamped (2r+1)^2 square clipped at the "
        f"left edge is exactly 45 -- dilation radius or clamping is wrong")
    req(not mask[6, 0], "the pixel containing the strike is marked clean")
    req(not mask[6, radius], "pixel at exactly radius is not excluded")
    req(mask[6, radius + 1], "pixel at radius+1 was excluded (radius too large)")
    req(not mask[2, 0] and mask[1, 0],
        "vertical extent of the dilation is wrong")

    # THE WRAP TEST. A circular dilation makes the right edge see the left one.
    req(mask[6, w - 1],
        "right-edge pixel excluded by a strike on the LEFT edge: the dilation "
        "wraps around the domain (np.roll), which silently deletes real "
        "domain-edge pixels from the new-initiation subset")
    # Prove the test can fire: the roll-based version does mark that pixel.
    had = (x[:, ch] >= C.LIGHTNING_THRESHOLD).any(axis=0)
    rolled = np.zeros_like(had)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rolled |= np.roll(np.roll(had, dy, axis=0), dx, axis=1)
    req(bool(rolled[6, w - 1]),
        "the wrap test is vacuous: even np.roll does not reach the far edge")

    # Same on the top edge, and the batched (N, T, C, H, W) entry point.
    x2 = np.zeros((t_in, C.N_CHANNELS, h, w), np.float32)
    x2[0, ch, 0, 6] = 1.0
    mask2 = M.new_initiation_mask(x2, radius=radius)
    req(mask2[h - 1, 6], "bottom row excluded by a strike in the top row (wrap)")
    req(int(np.count_nonzero(~mask2)) == 5 * 9,
        "top-edge clamped exclusion count is wrong")
    m5 = M.new_initiation_mask(np.stack([x, x2]), radius=radius)
    req(m5.shape == (2, h, w) and np.array_equal(m5[0], mask)
        and np.array_equal(m5[1], mask2),
        "the (N, T, C, H, W) path disagrees with the (T, C, H, W) path")
    return (f"exclusion is exactly the clamped 9x5=45 px square; radius and "
            f"radius+1 boundary correct; left/top edge events do NOT reach the "
            f"opposite edge (a roll-based dilation would, and was verified to)")


# ===========================================================================
# B. CAUSALITY / LEAKAGE PROOFS
# ===========================================================================

SEC_B = SECTIONS[1]

# Features at frames [0, T_in - 1) must not move when frames >= T_in - 1 change.
# T_in - 1 is included in the perturbation on purpose: it is the boundary frame,
# where an off-by-one in a temporal window would show up.
_PERTURB_FROM = C.INPUT_FRAMES - 1


def _raw_sequence(n_frames: int = 18, seed: int = 3) -> np.ndarray:
    """A real generated storm sequence -- not noise, so features are non-trivial."""
    key = f"raw{n_frames}_{seed}"
    if key not in _FIX:
        ev = synth.generate_dataset(n_events=1, seed=seed,
                                    frames_per_event=max(n_frames, 24))[0]
        _FIX[key] = ev["x"][:n_frames].copy()
    return _FIX[key]


@test(SEC_B, "future_perturbation_features")
def t_future_perturbation() -> str:
    """Perturbing future frames must not alter any earlier feature, bitwise."""
    x = _raw_sequence(18)
    f0 = FT.build_features(x)

    rng = np.random.default_rng(1234)
    x_pert = x.copy()
    # Replace EVERY value from the boundary frame onward with unrelated draws
    # spanning the physical range, including the FILL_VALUE sentinel, so any
    # dependence at all shows up.
    shp = x_pert[_PERTURB_FROM:].shape
    x_pert[_PERTURB_FROM:] = rng.uniform(-30.0, 80.0, shp).astype(np.float32)
    x_pert[_PERTURB_FROM, C.CH["refl_m10"]] = C.FILL_VALUE
    f1 = FT.build_features(x_pert)

    past0, past1 = f0[:_PERTURB_FROM], f1[:_PERTURB_FROM]
    req(bitwise_equal(past0, past1),
        f"FEATURES LOOK AHEAD: perturbing frames >= {_PERTURB_FROM} changed "
        f"features at frames < {_PERTURB_FROM}. Max abs delta "
        f"{float(np.abs(past0.astype(np.float64) - past1.astype(np.float64)).max()):.3e}. "
        f"Channels affected: "
        f"{[FT.FEATURE_NAMES[i] for i in np.where((past0 != past1).any(axis=(0, 2, 3)))[0]]}")

    # And the perturbation must actually have done something, otherwise the
    # assertion above is satisfied by a build_features that ignores its input.
    req(not np.array_equal(f0[_PERTURB_FROM:], f1[_PERTURB_FROM:]),
        "the perturbation changed no feature at all -- the test is vacuous")

    # Per-channel report so a future failure names the guilty feature.
    n_changed = int((f0[_PERTURB_FROM:] != f1[_PERTURB_FROM:]
                     ).any(axis=(0, 2, 3)).sum())
    return (f"all {FT.N_FEATURES} channels x frames 0..{_PERTURB_FROM - 1} "
            f"bitwise identical after randomising frames "
            f"{_PERTURB_FROM}..{x.shape[0] - 1}; {n_changed}/{FT.N_FEATURES} "
            f"channels did change at frames >= {_PERTURB_FROM} (test is live)")


@test(SEC_B, "backward_difference_is_backward")
def t_backward_difference() -> str:
    """d[t] may use x[t] and x[t-1] only; frames before t must not move."""
    rng = np.random.default_rng(7)
    seq = rng.standard_normal((8, 3, 3))
    d = FT.backward_difference(seq)

    req(np.array_equal(d[0], np.zeros((3, 3))),
        "frame 0 of a backward difference must be 0 (no history available)")
    req(np.allclose(d[1:], seq[1:] - seq[:-1], atol=0, rtol=0),
        "backward_difference is not x[t] - x[t-1]")

    tp = 4
    s2 = seq.copy()
    s2[tp] += 5.0
    d2 = FT.backward_difference(s2)

    # The causal statement. d[t] legitimately depends on x[t]; what must NEVER
    # happen is d[t-1] moving, which is exactly what a centred difference does.
    req(bitwise_equal(d[:tp], d2[:tp]),
        f"NON-CAUSAL DERIVATIVE: perturbing frame {tp} changed the difference "
        f"at frames < {tp}. A centred difference (x[t+1]-x[t-1])/2 does this "
        f"and reads one frame of the future")
    req(not np.allclose(d[tp], d2[tp]) and not np.allclose(d[tp + 1], d2[tp + 1]),
        f"perturbing frame {tp} changed neither d[{tp}] nor d[{tp + 1}] -- "
        f"the difference is not being computed")
    req(bitwise_equal(d[tp + 2:], d2[tp + 2:]),
        f"a single-frame perturbation propagated past frame {tp + 1}")

    # Prove the check has teeth by running a centred implementation through it.
    def centred(s: np.ndarray) -> np.ndarray:
        out = np.zeros_like(s, dtype=np.float64)
        out[1:-1] = (s[2:] - s[:-2]) / 2.0
        return out

    req(not np.array_equal(centred(seq)[:tp], centred(s2)[:tp]),
        "the causality check does not detect a centred difference, so it "
        "cannot detect the bug it exists for")

    # Multi-lag form: the first `lag` frames must be zero, not wrapped.
    for lag in (1, 2, 3):
        dl = FT.backward_difference(seq, lag=lag)
        req(np.array_equal(dl[:lag], np.zeros((lag, 3, 3))),
            f"backward_difference(lag={lag}) did not zero the first {lag} frames")
        req(np.allclose(dl[lag:], seq[lag:] - seq[:-lag], atol=0, rtol=0),
            f"backward_difference(lag={lag}) has the wrong offset")
    return ("d[0]=0; d[t]=x[t]-x[t-1] exactly; perturbing frame 4 leaves "
            "d[0..3] bitwise unchanged and moves only d[4], d[5]; a centred "
            "difference put through the same check IS rejected; lag=1,2,3 ok")


@test(SEC_B, "truncation_invariance")
def t_truncation_invariance() -> str:
    """Features aligned at the start cannot depend on how much future exists."""
    ev = synth.generate_dataset(n_events=1, seed=3, frames_per_event=24)[0]
    f_long = FT.build_features(ev["x"][:24])
    for k in (4, 8, C.INPUT_FRAMES, 12):
        f_short = FT.build_features(ev["x"][:k])
        req(bitwise_equal(f_long[:k], f_short),
            f"truncating the sequence to {k} frames changed the features at "
            f"frames 0..{k - 1}: something is normalised or aggregated over the "
            f"whole time axis, which leaks the future into the past")

    # Causal history is window-relative by design (each sample's running max
    # starts at its own window start). That is causal, but it means a window
    # starting at t0 > 0 is NOT a slice of the long-sequence features. What must
    # hold is the inequality: a running max over a longer history can only be
    # larger. A sign or axis error in causal_max breaks this.
    ci = FT.FEATURE_NAMES.index("graupel_cmax")
    t0, k = 6, 8
    f_win = FT.build_features(ev["x"][t0:t0 + k])
    req(bool((f_long[t0:t0 + k, ci] >= f_win[:, ci] - 1e-7).all()),
        "causal_max over a longer history came out SMALLER than over a "
        "sub-window: the running maximum is accumulating along the wrong axis")
    strictly = bool((f_long[t0:t0 + k, ci] > f_win[:, ci] + 1e-7).any())
    return (f"features on 24 frames match features on 4/8/6/12 frames bitwise "
            f"at the aligned prefix; causal_max monotone in history length "
            f"(strict somewhere: {strictly})")


@test(SEC_B, "window_arithmetic_no_target_in_input")
def t_window_arithmetic() -> str:
    """Every X frame is an input frame and every Y frame a target frame, exactly."""
    # Tag every frame with its own global index so an off-by-one in t0 cannot
    # hide behind visually similar radar frames.
    h = w = 4
    n_frames = 20
    fake: list[dict] = []
    for e in range(3):
        xx = np.zeros((n_frames, C.N_CHANNELS, h, w), np.float32)
        yy = np.zeros((n_frames, h, w), np.float32)
        for t in range(n_frames):
            xx[t] = e * 1000 + t
            yy[t] = e * 1000 + t
        fake.append({"x": xx, "y": yy, "mask": np.ones_like(xx, bool),
                     "meta": {"kinds": []}})

    wins = CV.enumerate_windows(fake, [0, 1, 2], stride=2)
    X, Y, sid = CV.make_sequences(fake, [0, 1, 2], stride=2)
    req(len(wins) == X.shape[0], "enumerate_windows and make_sequences disagree")

    for k, (e, t0) in enumerate(wins):
        req(int(sid[k]) == e, f"sample {k} claims event {sid[k]}, window says {e}")
        for i in range(C.INPUT_FRAMES):
            req(bool((X[k, i] == e * 1000 + t0 + i).all()),
                f"X[{k},{i}] holds frame {float(X[k, i].flat[0]) % 1000:.0f}, "
                f"expected {t0 + i}: the input window is mis-indexed")
        for j in range(C.OUTPUT_FRAMES):
            want = e * 1000 + t0 + C.INPUT_FRAMES + j
            req(bool((Y[k, j] == want).all()),
                f"Y[{k},{j}] holds frame {float(Y[k, j].flat[0]) % 1000:.0f}, "
                f"expected {t0 + C.INPUT_FRAMES + j}: the target window is "
                f"mis-indexed and may overlap the input")
        # The decisive property: no target frame index appears in the input.
        in_idx = {t0 + i for i in range(C.INPUT_FRAMES)}
        out_idx = {t0 + C.INPUT_FRAMES + j for j in range(C.OUTPUT_FRAMES)}
        req(not (in_idx & out_idx),
            f"input and target windows overlap at t0={t0}: {in_idx & out_idx}")
        req(t0 + C.INPUT_FRAMES + C.OUTPUT_FRAMES <= n_frames,
            "a window ran off the end of its event")
    return (f"{len(wins)} windows over 3 frame-tagged events: X holds exactly "
            f"frames t0..t0+{C.INPUT_FRAMES - 1}, Y exactly "
            f"t0+{C.INPUT_FRAMES}..t0+{C.INPUT_FRAMES + C.OUTPUT_FRAMES - 1}, "
            f"no index shared")


@test(SEC_B, "split_integrity")
def t_split_integrity() -> str:
    """No event in two splits, at any size or seed; the check itself fires."""
    for n in (3, 6, 9, 12, 20, 37, 60):
        for seed in (0, 1, 42, 20260905):
            s = CV.split_events(n, seed=seed)
            ids = s["train"] + s["val"] + s["test"]
            req(len(ids) == len(set(ids)),
                f"n={n} seed={seed}: an event landed in two splits: {s}")
            req(set(ids) == set(range(n)),
                f"n={n} seed={seed}: splits do not cover every event exactly "
                f"once -- an event was silently dropped by the fraction "
                f"arithmetic")
            for k, v in s.items():
                req(len(v) > 0, f"n={n} seed={seed}: {k} split is empty, so "
                                f"threshold selection or scoring is undefined")
            req(s == CV.split_events(n, seed=seed),
                f"split_events is not deterministic at n={n} seed={seed}, so a "
                f"resumed run would train on former test events")

    # A check that never fires is indistinguishable from no check.
    fired = False
    try:
        CV.verify_split_disjoint({"train": [0, 1], "val": [1], "test": [2]},
                                 n_events=3)
    except AssertionError:
        fired = True
    req(fired, "verify_split_disjoint ACCEPTED a split with a shared event")
    fired = False
    try:
        CV.verify_split_disjoint({"train": [0], "val": [1], "test": []},
                                 n_events=4)
    except AssertionError:
        fired = True
    req(fired, "verify_split_disjoint accepted a split that drops an event")

    # Now on the real arrays: samples from one event must never straddle splits.
    events = fx_events()
    splits = CV.split_events(FIX_EVENTS, seed=0)
    sets = CV.build_split_sequences(events, splits, stride=FIX_STRIDE,
                                   val_stride=FIX_STRIDE)
    owner: dict[int, str] = {}
    for name, d in sets.items():
        for e in np.unique(d["sid"]).tolist():
            req(int(e) not in owner,
                f"event {e} supplies samples to both {owner.get(int(e))} and "
                f"{name}: consecutive windows from one storm are near-identical, "
                f"so this is memorisation scored as generalisation")
            owner[int(e)] = name
        req(set(np.unique(d["sid"]).tolist()) <= set(splits[name]),
            f"{name} contains samples from events outside its own index list")

    # Windows are (event, t0) pairs; with events blocked, no pair can be shared.
    tr = set(CV.enumerate_windows(events, splits["train"], stride=FIX_STRIDE))
    te = set(CV.enumerate_windows(events, splits["test"], stride=FIX_STRIDE))
    req(not (tr & te), f"{len(tr & te)} identical windows in train and test")
    req(not ({e for e, _ in tr} & {e for e, _ in te}),
        "train and test windows come from a shared event")
    return (f"disjoint, covering, non-empty and deterministic for 7 sizes x 4 "
            f"seeds; verify_split_disjoint rejects both an overlap and a "
            f"dropped event; {len(owner)} events each owned by exactly one "
            f"split; 0 shared (event,t0) windows")


@test(SEC_B, "naive_split_leakage_is_real")
def t_naive_leakage_measured() -> str:
    """Quantify the leak event-blocking prevents, so the rule is not folklore."""
    events = fx_events()
    span = C.INPUT_FRAMES + C.OUTPUT_FRAMES
    lk1 = CV.naive_sample_split_leakage(events, stride=1, seed=1)
    req(lk1["frac_test_overlapping_train"] > 0.9,
        f"a random SAMPLE-level split at stride 1 overlapped only "
        f"{lk1['frac_test_overlapping_train']:.1%} of test windows with train. "
        f"If this is low the measurement is broken, and the justification for "
        f"event blocking is unsupported")
    req(lk1["overlap_span_frames"] == span,
        f"overlap span {lk1['overlap_span_frames']} != T_in+T_out = {span}")
    lk6 = CV.naive_sample_split_leakage(events, stride=FIX_STRIDE, seed=1)
    return (f"random sample-level split leaks "
            f"{lk1['frac_test_overlapping_train']:.0%} of test windows at "
            f"stride 1 and {lk6['frac_test_overlapping_train']:.0%} at stride "
            f"{FIX_STRIDE} (overlap if |dt| < {span} frames); event blocking "
            f"makes both 0 by construction")


@test(SEC_B, "model_input_path_ignores_target_window")
def t_model_path_causal() -> str:
    """X -> features -> predict must be bitwise invariant to the target frames."""
    events = synth.generate_dataset(n_events=2, seed=5, frames_per_event=30)
    X, Y, _ = CV.make_sequences(events, range(2), stride=8)
    model = NN.ConvLSTMNowcaster(in_ch=FT.N_FEATURES, t_out=C.OUTPUT_FRAMES,
                                 hidden=4, downsample=2, seed=2)
    p0 = model.predict(FT.build_features_batch(X))

    # Overwrite the whole TARGET window of every event with noise, then rebuild
    # the samples through the same code path. X is sliced from frames
    # [t0, t0+T_in) so nothing may change; if the prediction moves, some part of
    # the path reaches past the input window.
    rng = np.random.default_rng(99)
    pert = []
    for ev in events:
        e2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in ev.items()}
        e2["x"][C.INPUT_FRAMES:] = rng.uniform(
            -30.0, 80.0, e2["x"][C.INPUT_FRAMES:].shape).astype(np.float32)
        e2["y"][C.INPUT_FRAMES:] = (rng.random(
            e2["y"][C.INPUT_FRAMES:].shape) < 0.3).astype(np.float32) * 9.0
        pert.append(e2)
    X2, Y2, _ = CV.make_sequences(pert, range(2), stride=8)
    p1 = model.predict(FT.build_features_batch(X2))

    # Only the first window (t0 = 0) has its whole input window before the
    # perturbation boundary; later windows legitimately read perturbed frames
    # as INPUT. So restrict the invariance claim to those windows.
    wins = CV.enumerate_windows(events, range(2), stride=8)
    safe = [k for k, (_, t0) in enumerate(wins)
            if t0 + C.INPUT_FRAMES <= C.INPUT_FRAMES]
    req(bool(safe), "no window sits entirely before the perturbed target frames")
    for k in safe:
        req(bitwise_equal(X[k], X2[k]),
            f"sample {k}: the input window itself changed when only target "
            f"frames were perturbed -- make_sequences reads past t0+T_in")
        req(bitwise_equal(p0[k], p1[k]),
            f"MODEL SEES THE TARGET: prediction for sample {k} moved by "
            f"{float(np.abs(p0[k] - p1[k]).max()):.3e} when only the target "
            f"window was perturbed")
    req(not np.array_equal(Y[safe[0]], Y2[safe[0]]),
        "the target perturbation did not change Y -- the test is vacuous")

    # Batch invariance. A prediction that depends on its batch-mates means some
    # statistic is pooled across samples; at test time that is leakage between
    # test cases, and it inflates scores in a way no split can prevent.
    p_full = model.predict(FT.build_features_batch(X), batch=8)
    p_one = np.concatenate([model.predict(FT.build_features_batch(X[i:i + 1]),
                                          batch=1)
                            for i in range(X.shape[0])])
    req(bitwise_equal(p_full, p_one),
        f"predictions depend on batch composition (max delta "
        f"{float(np.abs(p_full - p_one).max()):.3e}): a statistic is pooled "
        f"across samples, which leaks between test cases")
    req(bitwise_equal(model.predict(FT.build_features_batch(X)), p0),
        "predict is not deterministic on identical input")
    req(float(p0.min()) >= 0.0 and float(p0.max()) <= 1.0
        and bool(np.isfinite(p0).all()),
        "predict returned values outside [0,1] or non-finite")
    return (f"{len(safe)} window(s) fully inside the input region: X and the "
            f"predicted probabilities bitwise unchanged after randomising every "
            f"target frame; predictions identical whether run in a batch of "
            f"{X.shape[0]} or one at a time; deterministic; range [0,1]")


@test(SEC_B, "new_initiation_mask_is_pure_in_x")
def t_ni_mask_pure() -> str:
    """The anti-shortcut mask must be computable at forecast time, from x only."""
    h = w = 12
    ch = C.CH["light_dens"]
    x = np.zeros((4, C.N_CHANNELS, h, w), np.float32)
    x[1, ch, 6, 6] = 2.0
    base = M.new_initiation_mask(x, radius=4)

    req(np.array_equal(M.new_initiation_mask(x, radius=4), base),
        "new_initiation_mask is not a pure function of its input")

    # Only the lightning channel may matter. If any other channel did, the mask
    # would stop meaning "no observed lightning nearby".
    rng = np.random.default_rng(1)
    x_other = x.copy()
    for ci in range(C.N_CHANNELS):
        if ci != ch:
            x_other[:, ci] = rng.uniform(-999.0, 80.0, (4, h, w))
    req(np.array_equal(M.new_initiation_mask(x_other, radius=4), base),
        "the mask changed when a non-lightning channel changed")
    x_ltg = x.copy()
    x_ltg[2, ch, 1, 1] = 5.0
    req(not np.array_equal(M.new_initiation_mask(x_ltg, radius=4), base),
        "the mask did NOT change when a strike was added: it is not reading "
        "the lightning channel")

    # And through the pipeline entry point: the scored subset must be identical
    # for two completely different observation fields with the same input.
    X, Y, _ = fx_windows()
    prob = np.full(Y.shape, 0.02)
    rng2 = np.random.default_rng(0)
    Y_alt = (rng2.random(Y.shape) < 0.3).astype(np.float32) * 7.0
    e1 = M.evaluate(prob[:4], Y[:4], x_input=X[:4], threshold=0.15)
    e2 = M.evaluate(prob[:4], Y_alt[:4], x_input=X[:4], threshold=0.15)
    n1 = e1["new_initiation"]["n_scored_px"]
    n2 = e2["new_initiation"]["n_scored_px"]
    req(n1 == n2,
        f"the new-initiation subset changed with the OBSERVATIONS "
        f"({n1:,} -> {n2:,} px): the mask is reading y, so the diagnostic is "
        f"not computable at forecast time and is itself leaking")
    req(n1 > 0, "the new-initiation subset is empty")
    return (f"pure in x, sensitive only to the lightning channel; scored "
            f"subset {n1:,} px identical under two unrelated observation "
            f"fields, so the mask uses the input window only")


# ===========================================================================
# C. GENERATOR HEALTH AND DIFFICULTY
# ===========================================================================

SEC_C = SECTIONS[2]


def fx_stats() -> dict:
    if "stats" not in _FIX:
        _FIX["stats"] = synth.dataset_stats(fx_events())
    return _FIX["stats"]


@test(SEC_C, "base_rate_in_band")
def t_base_rate() -> str:
    """Base rate must be rare but scoreable, on the configured seed and others."""
    st = fx_stats()
    br = st["base_rate"]
    req(0.003 < br < 0.04,
        f"base rate {br:.5f} outside 0.003-0.04. Too low and every split is "
        f"degenerate (all rare-event metrics nan); too high and the rare-event "
        f"machinery is not being exercised at all")

    # Seed robustness: the band must not be a property of one lucky draw.
    spread = []
    for s in (1, 7, 11, 2024):
        ev = synth.generate_dataset(n_events=5, seed=s,
                                    frames_per_event=FIX_FRAMES)
        y = np.concatenate([e["y"].ravel() for e in ev])
        spread.append(float((y >= C.LIGHTNING_THRESHOLD).mean()))
    req(all(0.0005 < v < 0.06 for v in spread),
        f"base rate across seeds {['%.5f' % v for v in spread]} leaves the "
        f"plausible range for lightning occupancy")
    req(st["mean_strikes_per_frame"] > 0,
        "no strikes at all in the fixture dataset")
    return (f"base rate {br:.5f} on seed {FIX_SEED} (band 0.003-0.04); across "
            f"4 other seeds {min(spread):.5f}-{max(spread):.5f}; "
            f"{st['mean_strikes_per_frame']:.1f} strikes/frame mean")


@test(SEC_C, "no_surface_reflectivity_shortcut")
def t_physics_gap() -> str:
    """Storm-conditional charging-layer AUROC must beat surface dBZ clearly."""
    st = fx_stats()
    sfc = st["auc_sfc_stormy"]
    m10 = st["auc_m10_stormy"]
    gap = st["physics_gap_stormy"]
    req(st["n_stormy_px"] > 5000,
        f"only {st['n_stormy_px']} storm-conditional pixels: the comparison "
        f"below is noise")
    req(gap > 0.05,
        f"physics gap {gap:+.3f} <= 0.05. The generator has regressed to a "
        f"SURFACE-REFLECTIVITY SHORTCUT: surface dBZ (AUROC {sfc:.3f}) "
        f"discriminates electrified from non-electrified cells about as well as "
        f"charging-layer dBZ ({m10:.3f}), so the warm-rain mimic population is "
        f"not doing its job and every downstream score measures the shortcut, "
        f"not the physics")
    req(m10 > 0.75,
        f"charging-layer AUROC {m10:.3f} is too low for the physics to be "
        f"learnable at all -- the label is no longer tied to charging-layer ice")
    req(sfc < 0.90,
        f"surface-dBZ AUROC {sfc:.3f} is nearly diagnostic on its own")
    req(st["auc_sfc_all"] > sfc,
        "unconditional surface AUROC is not higher than the storm-conditional "
        "one, so the clear-air inflation this metric warns about is absent and "
        "the warning is misleading")

    # Seed robustness. A physics gap that only exists on the configured seed is
    # a property of one lucky draw, not of the generator. The shortcut this
    # guards against (electrified and mimic cells separating on surface dBZ)
    # would come and go with the cell-kind lottery at small n, so the gap is
    # re-measured on independent seeds and every one of them must clear the bar.
    others: list[tuple[int, float, float, float]] = []
    for s in (101, 777):
        ev = synth.generate_dataset(n_events=5, seed=s,
                                    frames_per_event=FIX_FRAMES)
        s_st = synth.dataset_stats(ev)
        others.append((s, s_st["auc_sfc_stormy"], s_st["auc_m10_stormy"],
                       s_st["physics_gap_stormy"]))
    for s, s_sfc, s_m10, s_gap in others:
        req(s_gap > 0.05,
            f"physics gap on seed {s} is {s_gap:+.3f} <= 0.05 (surface "
            f"{s_sfc:.3f} vs charging-layer {s_m10:.3f}). The gap on the "
            f"configured seed {FIX_SEED} ({gap:+.3f}) is therefore a property "
            f"of one draw, not of the generator: on a different storm "
            f"population the surface-reflectivity shortcut reappears")

    # The generator's own docstring targets a gap >= 0.15; report against that.
    soft = "" if gap >= 0.15 else (
        f"  [note: gap {gap:+.3f} is below synth.py's own >= +0.15 target]")
    return (f"storm-conditional AUROC: surface {sfc:.3f} vs charging-layer "
            f"{m10:.3f}, gap {gap:+.3f} (> 0.05 required); unconditional "
            f"surface AUROC {st['auc_sfc_all']:.3f} confirms clear-air "
            f"inflation; independent seeds "
            + ", ".join(f"{s}: {g:+.3f}" for s, _, _, g in others)
            + f"{soft}")


@test(SEC_C, "null_events_present")
def t_null_events() -> str:
    """Some events must be entirely lightning-free, and some must not be."""
    st = fx_stats()
    frac = st["frac_events_with_lightning"]
    req(0.0 < frac < 1.0,
        f"frac_events_with_lightning = {frac:.3f}. If 1.0, every event contains "
        f"lightning and the model can learn an unconditional positive prior; if "
        f"0.0 there is nothing to predict")
    req(0.25 < frac < 1.0,
        f"only {frac:.1%} of events contain any lightning: with an event-blocked "
        f"split a whole test split of null events becomes likely, and every "
        f"rare-event metric on it returns nan")
    spread = []
    for s in (1, 7, 11, 2024):
        ev = synth.generate_dataset(n_events=6, seed=s,
                                    frames_per_event=FIX_FRAMES)
        spread.append(float(np.mean([e["y"].sum() > 0 for e in ev])))
    req(all(0.0 < v < 1.0 for v in spread),
        f"across seeds {['%.2f' % v for v in spread]}: some seed produced "
        f"either all-null or all-active events")
    note = ""
    if frac < 0.75:
        note = (f"  [note: synth.py aims for ~20% null events, i.e. frac ~0.8; "
                f"measured {frac:.2f} here and "
                f"{min(spread):.2f}-{max(spread):.2f} across seeds, so the null "
                f"fraction is well above the documented intent]")
    return (f"{frac:.1%} of the {FIX_EVENTS} fixture events carry lightning "
            f"(both classes of event present); across 4 other seeds "
            f"{min(spread):.2f}-{max(spread):.2f}{note}")


@test(SEC_C, "mimic_population_present")
def t_mimic_population() -> str:
    """Warm-rain and electrified cells must coexist, none dominating."""
    kinds = fx_stats()["cell_kinds"]
    total = sum(kinds.values())
    req(total > 0, "no cells generated at all")
    req(kinds.get("electrified", 0) > 0,
        "no electrified cells: there is no positive class to predict")
    req(kinds.get("warm_rain", 0) > 0,
        "no warm_rain cells: without the heavy-echo / no-ice mimic population "
        "the task collapses to a surface-reflectivity threshold, which is the "
        "exact failure this generator is designed to prevent")
    worst = max(kinds.items(), key=lambda kv: kv[1])
    req(worst[1] / total < 0.9,
        f"cell kind {worst[0]!r} is {worst[1] / total:.0%} of all cells; the "
        f"population has collapsed to one mode")
    mimics = sum(v for k, v in kinds.items() if k != "electrified")
    req(mimics >= kinds["electrified"] * 0.5,
        f"only {mimics} mimic cells against {kinds['electrified']} electrified: "
        f"far below the configured mimic_fraction "
        f"{C.SYNTH['mimic_fraction']:.2f}")
    return (f"{total} cells: {kinds}; largest kind {worst[1] / total:.0%} "
            f"(< 90%); {mimics} mimics vs {kinds['electrified']} electrified")


# ===========================================================================
# D. MODEL SANITY
# ===========================================================================

SEC_D = SECTIONS[3]


@test(SEC_D, "gradient_check")
def t_gradient_check() -> str:
    """Analytic BPTT vs central finite differences in float64, every tensor."""
    res = NN.gradient_check(verbose=False, tol=1e-4)
    overall = res["_overall"]
    req(overall < 1e-4,
        f"max relative gradient error {overall:.3e} >= 1e-4: the backward pass "
        f"is wrong. Training will still reduce the loss, so this is invisible "
        f"downstream and every learned result would be untrustworthy")
    for name in NN.ConvLSTMNowcaster.PARAM_NAMES:
        req(name in res, f"gradient_check skipped tensor {name}")
        req(res[name] < 1e-4,
            f"{name} gradient max rel err {res[name]:.3e} >= 1e-4")
    # A checker that probes nothing would also report 0.0; require it to have
    # produced a non-degenerate, finite error for at least one tensor.
    finite = [v for k, v in res.items() if not k.startswith("_")]
    req(all(np.isfinite(v) for v in finite),
        "gradient_check produced a non-finite relative error")
    return (f"all {len(NN.ConvLSTMNowcaster.PARAM_NAMES)} tensors "
            f"({', '.join(NN.ConvLSTMNowcaster.PARAM_NAMES)}) max rel err "
            f"{overall:.2e} < 1e-4")


@test(SEC_D, "learns_more_than_a_constant")
def t_train_beats_constant() -> str:
    """Loss must fall AND end below the best constant forecast's loss."""
    events = synth.generate_dataset(n_events=4, seed=7, frames_per_event=36)
    X, Y, _ = CV.make_sequences(events, range(4), stride=FIX_STRIDE)
    feats = FT.build_features_batch(X)
    yb = Y >= C.LIGHTNING_THRESHOLD

    model = NN.ConvLSTMNowcaster(in_ch=FT.N_FEATURES, t_out=C.OUTPUT_FRAMES,
                                 hidden=8, downsample=2, batch=4, lr=8e-3,
                                 seed=1)
    hist = model.fit(feats, yb, epochs=20, verbose=False)
    tr = hist["train_loss"]
    req(len(tr) == 20, f"only {len(tr)} epochs recorded")

    # Best CONSTANT probability under the same positive-weighted loss. This is
    # the number that matters: most of the early decrease in any loss curve is
    # the model discovering the base rate, which needs no spatial skill.
    yd = model._down(yb.astype(np.float64), as_dtype=np.float64)
    r = float(yd.mean())
    w = model.pos_weight
    p_star = w * r / (w * r + (1.0 - r))
    z_star = float(np.log(p_star / (1.0 - p_star)))
    const_loss = NN.weighted_bce_logits(np.full_like(yd, z_star), yd, w)

    req(tr[-1] < tr[0],
        f"train loss did not fall: {tr[0]:.4f} -> {tr[-1]:.4f}. The optimiser, "
        f"the gradient or the learning rate is broken")
    req(tr[-1] < 0.75 * tr[0],
        f"train loss fell only from {tr[0]:.4f} to {tr[-1]:.4f} "
        f"({tr[-1] / tr[0]:.0%} of the start): not a real decrease")
    req(tr[-1] < const_loss,
        f"final train loss {tr[-1]:.4f} is not below the best CONSTANT "
        f"forecast's loss {const_loss:.4f} (p*={p_star:.3f} at coarse base rate "
        f"{r:.4f}). The network has learned nothing a single scalar could not")
    req(all(np.isfinite(v) for v in tr), "a non-finite loss appeared")

    prob = model.predict(feats[:2])
    req(prob.shape == (2,) + yb.shape[1:], f"predict shape {prob.shape}")
    req(bool(np.isfinite(prob).all()) and 0.0 <= float(prob.min())
        and float(prob.max()) <= 1.0, "predict left [0,1] or went non-finite")
    return (f"{X.shape[0]} sequences, hidden=8, downsample=2, 20 epochs: loss "
            f"{tr[0]:.4f} -> {tr[-1]:.4f} (min {min(tr):.4f}), best constant "
            f"forecast {const_loss:.4f} -- beaten by "
            f"{const_loss - tr[-1]:.4f} nats")


# ===========================================================================
# E. BASELINES HONOUR THE RULES
# ===========================================================================

SEC_E = SECTIONS[4]


@test(SEC_E, "climatology_auprc_equals_base_rate")
def t_climatology_floor() -> str:
    """A constant forecast must score exactly its base rate; validates AUPRC."""
    X, Y, _ = fx_windows()
    ni = fx_ni()
    yb, niv = ni["yb"], ni["niv"]
    prob = fx_baseline_probs()["climatology"]

    ap_all = M.auprc(prob, yb)
    req(close(ap_all, ni["base_all"], 1e-6),
        f"climatology AUPRC {ap_all:.8f} != its base rate "
        f"{ni['base_all']:.8f}. The no-skill floor of average precision IS the "
        f"base rate, so the AUPRC implementation is wrong and every reported "
        f"lift is wrong with it")
    ap_ni = M.auprc(prob, yb, niv)
    req(close(ap_ni, ni["base_ni"], 1e-6),
        f"climatology AUPRC on the new-initiation subset {ap_ni:.8f} != subset "
        f"base rate {ni['base_ni']:.8f}")
    csi = M.contingency(prob >= 0.05, yb, niv)["CSI"]
    req(close(csi, 0.0, 1e-12) or np.isnan(csi),
        f"a constant forecast below threshold scored CSI {csi}")
    req(np.isnan(M.auroc(prob, yb)) or close(M.auroc(prob, yb), 0.5, 1e-9),
        "AUROC of a constant forecast != 0.5")
    return (f"AUPRC {ap_all:.8f} vs base rate {ni['base_all']:.8f} "
            f"(delta {abs(ap_all - ni['base_all']):.2e}); on the "
            f"new-initiation subset {ap_ni:.8f} vs {ni['base_ni']:.8f}; "
            f"AUROC 0.5")


@test(SEC_E, "persistence_scores_zero_on_new_initiation")
def t_persistence_ni_zero() -> str:
    """Eulerian persistence must be exactly blind on new-initiation pixels."""
    ni = fx_ni()
    yb, niv = ni["yb"], ni["niv"]
    prob = fx_baseline_probs()["persistence"]

    req(ni["n_pos_ni"] > 1000,
        f"only {ni['n_pos_ni']} positive pixels on the new-initiation subset: "
        f"the assertions below would be noise")

    # Persistence smooths the last observed lightning field by radius 1; the
    # mask excludes everything within radius 4 of ANY input-window strike. So
    # its probability on a clean pixel must be identically zero.
    max_clean = float(prob[niv].max())
    req(max_clean == 0.0,
        f"persistence assigns up to {max_clean:.4f} probability to pixels it is "
        f"defined to know nothing about. The new-initiation exclusion radius no "
        f"longer covers the persistence smoothing radius, so the headline "
        f"anti-shortcut metric is contaminated")

    for thr in (0.02, 0.05, 0.15):
        c = M.contingency(prob >= thr, yb, niv)
        req(c["CSI"] == 0.0,
            f"persistence CSI {c['CSI']:.6f} != 0 at threshold {thr} on the "
            f"new-initiation subset")
        req(c["hits"] == 0 and c["false_alarms"] == 0,
            f"persistence produced {c['hits']} hits / {c['false_alarms']} false "
            f"alarms on pixels where its forecast is identically zero")

    ap = M.auprc(prob, yb, niv)
    req(close(ap, ni["base_ni"], 1e-9),
        f"persistence AUPRC on the new-initiation subset {ap:.8f} != base rate "
        f"{ni['base_ni']:.8f}: a degenerate all-zero forecast is being scored "
        f"above its floor, which means tie handling in AUPRC is rewarding it")

    # It must nonetheless be strong on the AGGREGATE metric -- that is the
    # shortcut the subset exists to remove. If persistence were weak overall,
    # the new-initiation diagnostic would be solving a problem that is absent.
    ap_all = M.auprc(prob, yb)
    lift_all = ap_all / ni["base_all"]
    req(lift_all > 3.0,
        f"persistence aggregate AUPRC lift is only {lift_all:.2f}x: the "
        f"persistence shortcut this project is built around is not present in "
        f"the data, so the new-initiation metric proves nothing")
    return (f"max probability on clean pixels exactly 0.0; CSI exactly 0 at "
            f"thresholds 0.02/0.05/0.15; AUPRC {ap:.6f} == subset base rate "
            f"{ni['base_ni']:.6f}; meanwhile aggregate lift {lift_all:.1f}x "
            f"({ni['n_pos_ni']:,} NI positives scored)")


@test(SEC_E, "charging_rule_near_floor_on_new_initiation")
def t_charging_rule_ni() -> str:
    """The physics threshold rule must not score on new-initiation pixels either."""
    ni = fx_ni()
    yb, niv = ni["yb"], ni["niv"]
    prob = fx_baseline_probs()["charging_rule"]
    ap = M.auprc(prob, yb, niv)
    lift = ap / ni["base_ni"]
    req(lift < 3.0,
        f"the advected 40 dBZ rule reaches {lift:.2f}x base rate on the "
        f"new-initiation subset, above the 3x tolerance")
    csi = M.contingency(prob >= 0.05, yb, niv)["CSI"]
    req(csi < 0.05,
        f"charging-rule CSI {csi:.4f} on new-initiation pixels is not near zero")
    lift_all = M.auprc(prob, yb) / ni["base_all"]
    req(lift_all > lift,
        "the charging rule scores no better on the aggregate set than on the "
        "new-initiation subset, which contradicts the persistence-shortcut model")
    return (f"AUPRC lift {lift:.2f}x on new-initiation pixels (< 3x), CSI "
            f"{csi:.4f}; aggregate lift {lift_all:.1f}x")


@test(SEC_E, "advection_new_initiation_containment")
def t_advection_ni() -> str:
    """Advection must score at the floor on new-initiation pixels, at EVERY lead.

    This is the assertion the whole "honest evaluation" claim rests on. If
    Lagrangian persistence can score on the new-initiation subset, then a
    model's new-initiation score is not evidence that it predicts initiation.

    History: this test failed against the original static-radius mask, which
    excluded a fixed 8 km neighbourhood while storms move 2.2 px/frame. It
    guarded exactly one lead step and then leaked, peaking at 16x lift. The
    static mask is still computed here as a regression witness, so the fix is
    demonstrated by measurement rather than asserted.
    """
    ni = fx_ni()
    yb, niv, niv_s = ni["yb"], ni["niv"], ni["niv_static"]
    prob = fx_baseline_probs()["advection"]

    def lift_at(t: int, v: np.ndarray) -> float:
        y, vt = yb[:, t], v[:, t]
        if np.count_nonzero(y & vt) < 10:
            return float("nan")
        return M.auprc(prob[:, t], y, vt) / float(y[vt].mean())

    lifts = [lift_at(t, niv) for t in range(C.OUTPUT_FRAMES)]
    lifts_s = [lift_at(t, niv_s) for t in range(C.OUTPUT_FRAMES)]
    finite = [v for v in lifts if np.isfinite(v)]
    worst = float(max(finite)) if finite else float("nan")
    worst_lead = (int(np.nanargmax(lifts) + 1) * C.TIMESTEP_MIN
                  if finite else -1)
    fin_s = [v for v in lifts_s if np.isfinite(v)]
    worst_s = float(max(fin_s)) if fin_s else float("nan")

    lift_all = M.auprc(prob, yb) / ni["base_all"]
    lift_ni = M.auprc(prob, yb, niv) / ni["base_ni"]
    lift_ni_s = M.auprc(prob, yb, niv_s) / ni["base_ni_static"]
    csi_ni = M.contingency(prob >= 0.05, yb, niv)["CSI"]

    diag = (
        f"advection is NOT contained by the new-initiation exclusion: "
        f"overall lift {lift_ni:.2f}x, worst per-lead {worst:.2f}x at "
        f"{worst_lead} min, CSI@0.05 {csi_ni:.4f}. The subset is supposed to "
        f"remove everything Lagrangian persistence can claim, so any positive "
        f"advection score here means a model's new-initiation number is "
        f"partly motion extrapolation and cannot be read as initiation skill. "
        f"Check that metrics.evaluate is calling "
        f"new_initiation_mask_lagrangian (swept corridor along the "
        f"input-window motion estimate) and not the static-radius "
        f"new_initiation_mask. For reference the static mask scores "
        f"{lift_ni_s:.2f}x overall / {worst_s:.2f}x worst-lead on this same "
        f"forecast.")

    req(worst < 1.5, diag)
    req(lift_ni < 1.5, diag)
    req(csi_ni < 0.01, diag)

    # The subset must still be big enough to measure a model on. The naive
    # repair -- growing the radius isotropically with lead time -- also passes
    # the containment assertions above but erases the domain, so assert that
    # this one did not.
    n_pos = int(np.count_nonzero(yb & niv))
    frac = float(niv.mean())
    req(n_pos >= 200 and frac > 0.5,
        f"the exclusion contains advection but leaves only {n_pos} positives "
        f"over {frac:.1%} of the domain -- too little to score a model on. An "
        f"isotropically growing radius does this; use the swept corridor")

    return (f"contained at every lead: worst {worst:.2f}x at {worst_lead} min, "
            f"overall {lift_ni:.2f}x, CSI {csi_ni:.4f} (aggregate lift off the "
            f"subset is {lift_all:.1f}x). Static-radius mask on the identical "
            f"forecast leaks to {lift_ni_s:.2f}x / {worst_s:.2f}x worst-lead. "
            f"{n_pos:,} positives survive over {frac:.1%} of the domain")


@test(SEC_E, "new_initiation_subset_is_harder")
def t_ni_subset_harder() -> str:
    """Every baseline must lose skill on the subset, or the subset is not doing work."""
    ni = fx_ni()
    yb, niv = ni["yb"], ni["niv"]
    req(ni["base_ni"] < ni["base_all"],
        f"the new-initiation subset base rate {ni['base_ni']:.5f} is not below "
        f"the aggregate {ni['base_all']:.5f}: the mask is not removing the "
        f"lightning-adjacent pixels it is meant to remove")
    rows = []
    for name, prob in fx_baseline_probs().items():
        lift_all = M.auprc(prob, yb) / ni["base_all"]
        lift_ni = M.auprc(prob, yb, niv) / ni["base_ni"]
        rows.append((name, lift_all, lift_ni))
        req(lift_ni <= lift_all + 1e-6,
            f"{name} scores a HIGHER lift on the new-initiation subset "
            f"({lift_ni:.2f}x) than on the full set ({lift_all:.2f}x). New "
            f"initiation is strictly the harder problem, so this means the "
            f"subset or the masking in auprc is wrong")
    frac = float(np.count_nonzero(niv[:, 0])) / niv[:, 0].size
    return (f"subset base rate {ni['base_ni']:.5f} < aggregate "
            f"{ni['base_all']:.5f}; {frac:.1%} of pixels are clean; lifts "
            f"(all -> NI): "
            + ", ".join(f"{n} {a:.1f}x->{b:.1f}x" for n, a, b in rows))


# ===========================================================================
# Runner
# ===========================================================================

def main() -> int:
    t_start = time.time()
    width = 78
    print("=" * width)
    print("LIGHTNING NOWCASTING -- VERIFICATION SUITE")
    print("Metric anchors, causality proofs, generator difficulty, model "
          "sanity, baselines")
    print("=" * width)

    results: list[dict[str, Any]] = []
    for section in SECTIONS:
        tests = [t for t in _REGISTRY if t["section"] == section]
        if not tests:
            continue
        print()
        print(section)
        print("-" * width)
        for spec in tests:
            t0 = time.time()
            try:
                evidence = spec["fn"]()
                status, detail = "PASS", (evidence or spec["doc"])
            except KnownIssue as exc:
                status, detail = "KNOWN-ISSUE", str(exc)
            except AssertionError as exc:
                status, detail = "FAIL", str(exc)
            except Exception as exc:                      # noqa: BLE001
                status = "ERROR"
                detail = (f"{type(exc).__name__}: {exc}\n"
                          + "".join(traceback.format_exc(limit=6)))
            dt = time.time() - t0
            results.append({"section": section, "name": spec["name"],
                            "status": status, "detail": detail, "seconds": dt})
            print(f"  [{status:11s}] {spec['name']:44s} {dt:6.2f}s")
            print(f"        why: {spec['doc']}")
            for line in _wrap(detail, width - 14):
                print(f"        {line}")

    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] in ("FAIL", "ERROR") for r in results)
    n_known = sum(r["status"] == "KNOWN-ISSUE" for r in results)

    print()
    print("=" * width)
    print(f"SUMMARY  {n_pass} passed, {n_fail} failed, {n_known} known issues "
          f"in {time.time() - t_start:.1f}s")
    print("=" * width)
    for r in results:
        if r["status"] != "PASS":
            print(f"  {r['status']}: {r['name']}")
    if n_known:
        print()
        print("KNOWN ISSUES -- measured, reproducible, and NOT fixed. These are")
        print("defects in the code under test, not softened assertions:")
        for r in results:
            if r["status"] == "KNOWN-ISSUE":
                print(f"\n  * {r['name']}")
                for line in _wrap(r["detail"], width - 6):
                    print(f"    {line}")
    if n_fail:
        print()
        print("FAILURES -- do not report any metric from this pipeline until "
              "these are fixed.")
        for r in results:
            if r["status"] in ("FAIL", "ERROR"):
                print(f"\n  * {r['name']}")
                for line in _wrap(r["detail"], width - 6):
                    print(f"    {line}")
        return 1
    print()
    print("All causality and leakage proofs hold. No feature, split or model "
          "input")
    print("path can see the future, and the metric anchors match hand "
          "arithmetic.")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    """Wrap without importing textwrap's paragraph handling; keep newlines."""
    out: list[str] = []
    for para in str(text).split("\n"):
        words, line = para.split(), ""
        if not words:
            out.append("")
            continue
        for word in words:
            if line and len(line) + 1 + len(word) > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        out.append(line)
    return out


if __name__ == "__main__":
    raise SystemExit(main())

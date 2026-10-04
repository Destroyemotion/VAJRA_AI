"""Verify the Lagrangian new-initiation fix does what it claims.

Three things must hold, or the fix is cosmetic:
  1. Advection lift on the new-initiation subset collapses to ~1.0x at EVERY
     lead time (it was 4.3x overall, 19.5x at 10 min, with the static mask).
  2. Persistence stays at exactly 0 CSI (the fix must not have broken the
     property that already held).
  3. Enough of the domain survives to still measure anything -- the naive
     "grow the radius" repair passes (1) but leaves almost no scoreable
     pixels, which is why the corridor formulation was chosen instead.
"""

from __future__ import annotations

import numpy as np

import config as C
import metrics as M
import synth
import baselines as B

rng = np.random.default_rng(7)
events = synth.generate_dataset(n_events=6, seed=4242)

xs, ys = [], []
for e in events:
    T = e["x"].shape[0]
    for t0 in range(0, T - C.INPUT_FRAMES - C.OUTPUT_FRAMES, 6):
        xs.append(e["x"][t0:t0 + C.INPUT_FRAMES])
        ys.append(e["y"][t0 + C.INPUT_FRAMES:
                         t0 + C.INPUT_FRAMES + C.OUTPUT_FRAMES])
X = np.stack(xs)
Y = np.stack(ys) >= C.LIGHTNING_THRESHOLD
print(f"{X.shape[0]} samples, aggregate base rate {Y.mean():.5f}")

adv = B.advection(X, C.OUTPUT_FRAMES)
per = B.persistence(X, C.OUTPUT_FRAMES)

static = np.broadcast_to(M.new_initiation_mask(X)[:, None], adv.shape)
lagr = M.new_initiation_mask_lagrangian(X, C.OUTPUT_FRAMES)

print(f"\nscoreable pixels: static {static.mean():.3f} of domain, "
      f"lagrangian {lagr.mean():.3f}")
print(f"positives kept  : static {int((Y & static).sum()):,}, "
      f"lagrangian {int((Y & lagr).sum()):,}")


def lift(prob, mask, obs=Y):
    base = float(obs[mask].mean()) if mask.any() else float("nan")
    if not np.isfinite(base) or base <= 0:
        return float("nan"), float("nan"), 0
    return M.auprc(prob, obs, mask) / base, base, int((obs & mask).sum())


print("\n                    STATIC MASK          LAGRANGIAN MASK")
print("lead   npos_s npos_l   adv_lift  per_lift   adv_lift  per_lift")
worst = 0.0
for t in range(C.OUTPUT_FRAMES):
    ms, ml = static[:, t], lagr[:, t]
    a_s, _, ns = lift(adv[:, t], ms, Y[:, t])
    p_s, _, _ = lift(per[:, t], ms, Y[:, t])
    a_l, _, nl = lift(adv[:, t], ml, Y[:, t])
    p_l, _, _ = lift(per[:, t], ml, Y[:, t])
    worst = max(worst, 0.0 if not np.isfinite(a_l) else a_l)
    print(f"{(t + 1) * 5:4d}  {ns:6d} {nl:6d}   "
          f"{a_s:7.2f}x {p_s:8.2f}x   {a_l:7.2f}x {p_l:8.2f}x")

a_s_all, _, _ = lift(adv, static)
a_l_all, base_l, npos_l = lift(adv, lagr)
p_l_all, _, _ = lift(per, lagr)
print(f"\noverall advection lift : static {a_s_all:.2f}x -> "
      f"lagrangian {a_l_all:.2f}x")
print(f"overall persistence lift: lagrangian {p_l_all:.2f}x")

csi_a = M.contingency(adv >= 0.05, Y, lagr)["CSI"]
csi_p = M.contingency(per >= 0.05, Y, lagr)["CSI"]
print(f"CSI@0.05 on lagrangian subset: advection {csi_a:.4f}  "
      f"persistence {csi_p:.4f}")

ok = True
if not (worst < 1.5):
    ok = False
    print(f"\nFAIL: worst per-lead advection lift {worst:.2f}x >= 1.5x")
if not (csi_a < 0.01):
    ok = False
    print(f"\nFAIL: advection CSI {csi_a:.4f} on the subset is not ~0")
if npos_l < 200:
    ok = False
    print(f"\nFAIL: only {npos_l} positives survive -- subset too small")
print("\n" + ("PASS: advection contained, subset still measurable" if ok
              else "FAIL"))

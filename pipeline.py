"""
Experiment orchestration: split, train, evaluate everything, and say plainly who won.

WHAT THIS FILE IS FOR
---------------------
Every other module in this project produces a number. This one decides which
numbers are allowed to be believed, and it is written around three rules that
exist because breaking any of them turns a real evaluation into a press release.

RULE 1: THE DECISION THRESHOLD IS CHOSEN ON VALIDATION, NEVER ON TEST.
CSI, POD and FAR are all defined at an operating threshold. Sweeping thresholds
on the test set and reporting the best one is the most common way to inflate a
nowcasting result, and it is nearly invisible in a paper: the number is real,
the sweep is standard practice, and nobody states which split it ran on. On this
data the gap between "best threshold on test" and "validation threshold applied
to test" is routinely 0.02-0.05 CSI, which is the same size as the entire claimed
improvement of most published deep nowcasters over optical flow. So here the
threshold is selected on validation, for EVERY method including every baseline
(a threshold tuned for the model but not for the baseline is the same cheat with
extra steps), and both the validation and test scores are reported side by side.

RULE 2: EVERY BASELINE IS EVALUATED ON EXACTLY THE SAME SAMPLES, WITH THE SAME
CODE PATH. Not a number quoted from a previous run, not a baseline evaluated on
a convenient subset. `baselines.run_all` and the learned models are both fed the
identical (X, Y) test arrays and both go through `metrics.evaluate`.

RULE 3: NEW-INITIATION SKILL IS REPORTED NEXT TO AGGREGATE SKILL, ALWAYS.
Aggregate CSI on this task is dominated by the persistence of existing lightning.
A model can score respectably on aggregate CSI while providing zero warning of
anything new, and new lightning over a previously quiet area is the case that
actually hurts people. metrics.evaluate computes the subset; this file puts it in
the headline table rather than an appendix.

THE VERDICT SECTION
-------------------
`verdict()` compares the learned models against `advection` (Lagrangian
persistence -- what operational radar extrapolation actually is) and against
`charging_rule` (the Gremillion-Vincent 40 dBZ at -10C rule -- the null
hypothesis that there is nothing to learn beyond one physical threshold). It
states the outcome in plain language including the case where the learned model
loses, which at a small training budget is the expected outcome and is a
legitimate result. Nothing in this file rounds in a favourable direction,
re-picks a metric after seeing the numbers, or drops a baseline that won.

TIME AND MEMORY
---------------
2 CPU cores, 3 GB RAM, and any single sandbox call killed at ~120 s. Two
consequences shape the interface:

  * `max_seconds` is a real training budget, threaded down to
    ConvLSTMNowcaster.fit which checks it BETWEEN minibatches. Training stops
    cleanly, writes a checkpoint, and the next invocation resumes from it with
    the Adam moments intact. So a 20 minute training run is 12 sequential calls,
    none of which can be killed mid-update.
  * The feature stack is the memory hog: (N, 6, 20, 64, 64) float32 is 2 MB per
    sample, and motion compensation needs a second copy. `estimate_memory`
    prints the projection before anything is allocated, because discovering the
    limit by being OOM-killed halfway through costs the whole run.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable

import numpy as np

import baselines as B
import config as C
import cv as CV
import features as FT
import hybrid as HY
import metrics as M
import synth
from convlstm_np import ConvLSTMNowcaster

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None
    _HAS_PANDAS = False


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "results")
DEFAULT_CACHE = os.path.join(HERE, "cache")

# One shared threshold grid for every method.
#
# WHY IT IS WIDER THAN config.PROB_THRESHOLDS. A network trained with
# pos_weight=12 is deliberately over-confident, while advection emits calibrated
# but small probabilities. If the grid started at 0.02 the advection baseline
# would be forced to operate at a threshold well below its own useful range and
# would look worse than it is; if it stopped at 0.70 a sharp model would be
# denied its best operating point. The grid must span both, and -- this is the
# part that makes it fair -- the SAME grid and the SAME validation-selection
# rule are applied to every model and every baseline.
THRESHOLD_GRID = sorted(set(C.PROB_THRESHOLDS)
                        | {0.005, 0.01, 0.60, 0.80, 0.90})

# Metrics pulled into the comparison table, in display order.
TABLE_COLUMNS = [
    "model", "kind", "thr_val", "AUPRC", "CSI", "POD", "FAR",
    "FSS_r2", "FSS_r4", "BSS", "NI_CSI", "NI_AUPRC",
    "val_AUPRC", "val_CSI",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_or_make_events(n_events: int, seed: int,
                        frames_per_event: int | None = None,
                        use_cache: bool = True,
                        cache_dir: str = DEFAULT_CACHE,
                        verbose: bool = True) -> list[dict]:
    """Generate the synthetic dataset, caching it so repeated calls are cheap.

    The cache key includes every generator argument that changes the data. It
    deliberately does NOT include anything else: a cache keyed on a partial
    signature is worse than no cache, because a later run silently trains on the
    previous run's data and the discrepancy surfaces as an unreproducible score.

    `mask` is not cached. It is currently all-True except where x == FILL_VALUE,
    which is recoverable, and storing it would double the cache size for no gain.
    """
    n_frames = C.SYNTH["frames_per_event"] if frames_per_event is None \
        else int(frames_per_event)
    path = os.path.join(cache_dir,
                        f"synth_e{n_events}_s{seed}_f{n_frames}.npz")
    if use_cache and os.path.exists(path):
        t0 = time.time()
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"].item()))
        events = [{"x": z[f"x{i}"], "y": z[f"y{i}"], "meta": meta[i]}
                  for i in range(int(z["n"]))]
        if verbose:
            print(f"  loaded {len(events)} cached events from "
                  f"{os.path.basename(path)} in {time.time() - t0:.1f}s")
        return events

    t0 = time.time()
    events = synth.generate_dataset(n_events=n_events, seed=seed,
                                    frames_per_event=frames_per_event)
    if verbose:
        print(f"  generated {len(events)} events "
              f"({events[0]['x'].shape[0]} frames each) in "
              f"{time.time() - t0:.1f}s")
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        blob: dict[str, Any] = {"n": np.array(len(events))}
        for i, e in enumerate(events):
            blob[f"x{i}"] = e["x"]
            blob[f"y{i}"] = e["y"]
        blob["meta_json"] = np.array(json.dumps(
            [{k: (list(v) if isinstance(v, tuple) else v)
              for k, v in e["meta"].items()} for e in events]))
        np.savez_compressed(path, **blob)
        if verbose:
            size = os.path.getsize(path) / 1e6
            print(f"  cached -> {os.path.basename(path)} ({size:.0f} MB)")
    return events


def estimate_memory(n_train: int, n_val: int, n_test: int,
                    verbose: bool = True) -> dict:
    """Project peak memory before allocating anything.

    The dominant terms, per sample:
        raw X    T_in * N_CHANNELS  * H * W * 4 B = 0.6 MB
        features T_in * N_FEATURES  * H * W * 4 B = 2.0 MB
    and motion compensation holds a second copy of one split's features. On a
    3 GB budget the crossover is around 350 training samples, which is why the
    default stride is INPUT_FRAMES rather than 1.
    """
    px = C.GRID_H * C.GRID_W
    raw_per = C.INPUT_FRAMES * C.N_CHANNELS * px * 4 / 1e6
    feat_per = C.INPUT_FRAMES * FT.N_FEATURES * px * 4 / 1e6
    tgt_per = C.OUTPUT_FRAMES * px * 4 / 1e6
    n_all = n_train + n_val + n_test
    resident = n_all * (raw_per + feat_per + tgt_per)
    # Compensation copy of the largest split, plus the model's downsampled view.
    transient = max(n_train, n_val, n_test) * feat_per * 1.3
    est = {
        "raw_mb_per_sample": raw_per,
        "features_mb_per_sample": feat_per,
        "resident_mb": resident,
        "transient_mb": transient,
        "peak_mb_estimate": resident + transient,
    }
    if verbose:
        print(f"  memory projection: {n_all} samples resident "
              f"{resident:.0f} MB + transient {transient:.0f} MB "
              f"= peak ~{resident + transient:.0f} MB")
        if resident + transient > 2200:
            print("  WARNING: projected peak is close to the 3 GB limit. "
                  "Raise --stride or set max_samples to cut sample count.")
    return est


# ---------------------------------------------------------------------------
# Threshold selection -- on validation, for everyone
# ---------------------------------------------------------------------------

def select_threshold(prob_val: np.ndarray, y_val: np.ndarray,
                     grid: list[float] | None = None,
                     default: float = 0.15) -> dict:
    """Pick the CSI-maximising threshold on the VALIDATION split.

    Returned dict carries the validation CSI and AUPRC at that choice so the
    caller can report the in-sample (validation) and out-of-sample (test) numbers
    together. A large gap between them is the signal that the threshold choice
    itself has overfitted, which happens when the validation split is small; it
    is visible in the table rather than buried.

    `default` is used only when every threshold gives an undefined CSI, i.e. the
    forecast never crosses any threshold and there are no hits, misses or false
    alarms to count. Returning a default rather than nan keeps the pipeline
    running so the rest of the table still prints -- with the fallback flagged.
    """
    grid = list(grid or THRESHOLD_GRID)
    yb = np.asarray(y_val) >= C.LIGHTNING_THRESHOLD
    row = M.best_csi(prob_val, yb, thresholds=grid)
    if not row:
        return {"threshold": float(default), "val_CSI": float("nan"),
                "val_AUPRC": M.auprc(prob_val, yb), "fallback": True}
    return {"threshold": float(row["threshold"]),
            "val_CSI": float(row["CSI"]),
            "val_POD": float(row["POD"]),
            "val_FAR": float(row["FAR"]),
            "val_AUPRC": M.auprc(prob_val, yb),
            "fallback": False}


# ---------------------------------------------------------------------------
# Training, budgeted and resumable
# ---------------------------------------------------------------------------

def train_chunked(make: Callable[[], Any], ckpt: str,
                  F: np.ndarray, Y: np.ndarray,
                  F_val: np.ndarray | None = None,
                  Y_val: np.ndarray | None = None,
                  epochs: int = 40, budget: float = 60.0,
                  chunk_seconds: float | None = None,
                  loader: Callable[[str], Any] | None = None,
                  resume: bool = True, verbose: bool = True) -> tuple[Any, dict]:
    """Train under a wall-clock budget, resuming from a checkpoint if present.

    The genuinely important chunking here is ACROSS PROCESS INVOCATIONS: any one
    sandbox call is killed at ~120 s, so a long run has to be a sequence of short
    ones that each pick up exactly where the last stopped. That works because
    ConvLSTMNowcaster.save stores the Adam moment estimates and the epoch
    counter, so a resume is a continuation and not a restart with a loss spike --
    and because cv.split_events is deterministic given (n_events, seed), so the
    resumed call is guaranteed to be looking at the same training events.

    `chunk_seconds` splits the budget WITHIN one call as well. It defaults to the
    whole budget, i.e. one fit() call, because fit() already checks the budget
    between minibatches and checkpoints after every epoch. Splitting further has
    a real cost for MotionCompensatedNowcaster, which recomputes optical flow on
    every fit() call by design.
    """
    model = None
    resumed = False
    if resume and loader is not None:
        try:
            model = loader(ckpt)
            resumed = True
            if verbose:
                print(f"  resumed from {os.path.basename(ckpt)} at epoch "
                      f"{model.epochs_done}")
        except (FileNotFoundError, OSError, KeyError, ValueError):
            model = None
    if model is None:
        model = make()
        if budget <= 0 and verbose:
            # Silence here would be dangerous: an untrained network still emits
            # probabilities and still produces a full row in the comparison
            # table, and nothing downstream can tell the difference.
            print("  WARNING: no checkpoint found and no training budget -- "
                  "this model is RANDOMLY INITIALISED. Its row in the table is "
                  "not a result.")

    info = {"resumed": resumed, "budget_seconds": float(budget),
            "epochs_target": int(epochs)}
    if budget <= 0 or model.epochs_done >= epochs:
        info["trained_seconds"] = 0.0
        info["epochs_done"] = int(model.epochs_done)
        info["complete"] = model.epochs_done >= epochs
        if verbose:
            print(f"  no training performed (epochs_done="
                  f"{model.epochs_done}/{epochs}, budget={budget:.0f}s)")
        return model, info

    chunk = float(budget if chunk_seconds is None else chunk_seconds)
    t_start = time.time()
    n_calls = 0
    while True:
        remaining = budget - (time.time() - t_start)
        if remaining <= 1.0 or model.epochs_done >= epochs:
            break
        hist = model.fit(F, Y, F_val, Y_val,
                         epochs=int(epochs - model.epochs_done),
                         max_seconds=min(chunk, remaining),
                         verbose=verbose, checkpoint=ckpt)
        n_calls += 1
        if not hist.get("stopped_early", False):
            break
    model.save(ckpt)
    info.update({
        "trained_seconds": time.time() - t_start,
        "epochs_done": int(model.epochs_done),
        "fit_calls": n_calls,
        "complete": bool(model.epochs_done >= epochs),
        "train_loss": list(model.history.get("train_loss", [])),
        "val_loss": list(model.history.get("val_loss", [])),
        "checkpoint": ckpt,
    })
    if verbose:
        print(f"  trained {info['epochs_done']}/{epochs} epochs in "
              f"{info['trained_seconds']:.1f}s "
              f"({'complete' if info['complete'] else 'budget-limited'})"
              f" -> {os.path.basename(ckpt)}")
    return model, info


# ---------------------------------------------------------------------------
# Forecast production
# ---------------------------------------------------------------------------

def model_probs(name: str, model: Any, X_raw: np.ndarray, F: np.ndarray,
                batch: int = 8) -> np.ndarray:
    """Uniform way to get (N, T_out, H, W) probabilities from anything.

    ConvLSTMNowcaster and MotionCompensatedNowcaster both take the feature
    stack. BlendedNowcaster additionally needs the raw channels, because the
    advection component advects the observed lightning channel and that channel
    is only recoverable from the features by inverting a normalisation. Rather
    than pretend to a single signature, the difference is handled in one place.
    """
    if isinstance(model, HY.BlendedNowcaster):
        return model.predict(X_raw, F)
    return model.predict(F, batch=batch)


def evaluate_all(probs: dict[str, np.ndarray],
                 X_test: np.ndarray, Y_test: np.ndarray,
                 probs_val: dict[str, np.ndarray],
                 Y_val: np.ndarray,
                 verbose: bool = True) -> dict[str, dict]:
    """Select each method's threshold on val, then score it on test.

    One loop for models and baselines alike -- see RULE 2 in the module
    docstring. The threshold is chosen per method, which is the fair comparison:
    it asks "how well does each method do at its own best operating point,
    chosen without seeing test", not "how well does each method do at the
    operating point that suits the model".
    """
    out: dict[str, dict] = {}
    for name in probs:
        t0 = time.time()
        sel = select_threshold(probs_val[name], Y_val)
        ev = M.evaluate(probs[name], Y_test, x_input=X_test,
                        threshold=sel["threshold"], label=name)
        ev["threshold_selection"] = sel
        ev["threshold_selected_on"] = "validation split"
        out[name] = ev
        if verbose:
            ov = ev["overall"]
            ni = ev.get("new_initiation", {})
            print(f"    {name:22s} thr={sel['threshold']:.3f} (val CSI "
                  f"{sel['val_CSI']:.3f})  test AUPRC {ov['AUPRC']:.4f}  "
                  f"CSI {ov['CSI']:.4f}  NI-CSI "
                  f"{ni.get('CSI', float('nan')):.4f}  "
                  f"[{time.time() - t0:.1f}s]")
    return out


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

LEARNED = {"convlstm", "convlstm_eulerian", "hybrid", "motion_compensated"}


def _kind(name: str) -> str:
    if name in B.BASELINES:
        return "baseline"
    if name.startswith("blend"):
        return "blend"
    return "learned"


def table_rows(evals: dict[str, dict]) -> list[dict]:
    rows = []
    for name, ev in evals.items():
        ov = ev["overall"]
        ni = ev.get("new_initiation", {})
        cal = ev.get("calibration", {})
        sel = ev.get("threshold_selection", {})
        rows.append({
            "model": name,
            "kind": _kind(name),
            "thr_val": sel.get("threshold", ev.get("threshold")),
            "AUPRC": ov.get("AUPRC"),
            "CSI": ov.get("CSI"),
            "POD": ov.get("POD"),
            "FAR": ov.get("FAR"),
            "FSS_r2": ev["fss"].get(f"r2_{int(2 * C.PIXEL_KM)}km"),
            "FSS_r4": ev["fss"].get(f"r4_{int(4 * C.PIXEL_KM)}km"),
            "BSS": cal.get("bss"),
            "NI_CSI": ni.get("CSI", float("nan")),
            "NI_AUPRC": ni.get("AUPRC", float("nan")),
            "val_AUPRC": sel.get("val_AUPRC"),
            "val_CSI": sel.get("val_CSI"),
        })
    return rows


def comparison_table(evals: dict[str, dict], sort_by: str = "AUPRC"):
    """Sorted comparison table. Returns a DataFrame when pandas is importable.

    Sorted by AUPRC descending so the reader's eye lands on the ranking
    immediately and can see at a glance whether `advection` and `charging_rule`
    are above or below the learned rows -- which is the only question the table
    exists to answer.
    """
    rows = table_rows(evals)
    if _HAS_PANDAS:
        df = pd.DataFrame(rows, columns=TABLE_COLUMNS)
        df = df.sort_values(sort_by, ascending=False,
                            na_position="last").reset_index(drop=True)
        return df
    rows.sort(key=lambda r: (-1e9 if r[sort_by] is None
                             or (isinstance(r[sort_by], float)
                                 and math.isnan(r[sort_by]))
                             else -r[sort_by]))
    return rows


def format_table(tbl: Any) -> str:
    """Fixed-width rendering that does not depend on pandas display options."""
    rows = (tbl.to_dict("records") if _HAS_PANDAS and hasattr(tbl, "to_dict")
            else tbl)
    head = (f"{'model':22s} {'kind':9s} {'thr':>6s} "
            + " ".join(f"{c:>9s}" for c in TABLE_COLUMNS[3:]))
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for c in TABLE_COLUMNS[3:]:
            v = r.get(c)
            cells.append("      n/a" if v is None
                         or (isinstance(v, float) and math.isnan(v))
                         else f"{v:9.4f}")
        lines.append(f"{str(r['model']):22s} {str(r['kind']):9s} "
                     f"{r['thr_val']:6.3f} " + " ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}"


def _compare(name_a: str, a: dict, name_b: str, b: dict, metric: str) -> dict:
    va, vb = a.get(metric), b.get(metric)
    ok = (va is not None and vb is not None
          and not (isinstance(va, float) and math.isnan(va))
          and not (isinstance(vb, float) and math.isnan(vb)))
    return {
        "metric": metric, "learned": name_a, "reference": name_b,
        "learned_value": va, "reference_value": vb,
        "comparable": ok,
        "learned_wins": bool(ok and va > vb),
        "margin": (va - vb) if ok else None,
        "relative": ((va - vb) / vb) if (ok and vb not in (0, None)) else None,
    }


def verdict(evals: dict[str, dict], problems: list[str] | None = None,
            verbose: bool = True) -> dict:
    """State plainly whether the learned models beat the honest competition.

    The two references are chosen for specific reasons stated in baselines.py:
    `advection` is Lagrangian persistence, which is what operational radar
    extrapolation nowcasting reduces to and therefore the bar a new method must
    clear to be worth deploying; `charging_rule` is the Gremillion-Vincent
    threshold, which is the null hypothesis that all the skill available is one
    number applied to one physically chosen channel.

    Four comparisons are made per learned model: AUPRC and CSI on aggregate, and
    AUPRC and CSI on the new-initiation subset. A learned model is only called a
    win if it beats a reference on the aggregate PRIMARY metric (AUPRC); the
    other three are reported regardless, including when they contradict.

    `problems` is cv.stratify_report's list of split-quality failures, and it is
    a REQUIRED input in practice rather than a nicety. The first small run of
    this pipeline produced a test split with 154 positive pixels, all from one
    event: every baseline scored exactly the base rate because it never crossed
    any threshold, the network's over-confidence therefore "won" every
    comparison by a factor of 40, and the verdict text cheerfully announced that
    the learned model beat both references. The numbers were arithmetically
    correct and completely meaningless. So when the test split fails its quality
    checks this function refuses to declare a winner, and says why.

    No metric is chosen after the fact, nothing is rounded in a favourable
    direction, and a loss is reported as a loss.
    """
    problems = list(problems or [])
    # A problem naming the test split invalidates ranking; one naming train or
    # val degrades the model but leaves the comparison itself readable.
    test_problems = [p for p in problems if p.strip().startswith("test")]
    test_usable = not test_problems

    rows = {r["model"]: r for r in table_rows(evals)}
    learned = [n for n, r in rows.items() if r["kind"] in ("learned", "blend")]
    refs = [n for n in ("advection", "charging_rule") if n in rows]

    comparisons: list[dict] = []
    for ln in learned:
        for rn in refs:
            for metric in ("AUPRC", "CSI", "NI_AUPRC", "NI_CSI"):
                comparisons.append(_compare(ln, rows[ln], rn, rows[rn], metric))

    def _best(names: list[str], metric: str) -> str | None:
        cand = [(rows[n][metric], n) for n in names
                if rows[n][metric] is not None
                and not (isinstance(rows[n][metric], float)
                         and math.isnan(rows[n][metric]))]
        return max(cand)[1] if cand else None

    best_learned = _best(learned, "AUPRC")
    best_overall = _best(list(rows), "AUPRC")
    best_ni = _best(list(rows), "NI_AUPRC")

    text: list[str] = []
    text.append("VERDICT")
    text.append("=" * 78)
    if problems:
        text.append("SPLIT-QUALITY CHECK FAILED:")
        for p in problems:
            text.append(f"  ! {p}")
        text.append("")
    if not test_usable:
        text.append("THIS RUN IS NOT INTERPRETABLE AS A MODEL COMPARISON.")
        text.append("The test split does not contain enough independent "
                    "lightning to rank anything. Read the numbers below as a "
                    "check that the code runs, not as evidence. Specifically:")
        text.append("  - with a base rate this low, a baseline that never "
                    "crosses its threshold scores AUPRC == base rate, which "
                    "any over-confident model beats without being better;")
        text.append("  - CSI, POD and FAR are computed from a handful of "
                    "positive pixels drawn from one storm, so their sampling "
                    "error is larger than any difference between methods.")
        text.append("Fix the data, not the presentation: more events, or a "
                    "lower synth mimic_fraction so more events electrify.")
        text.append("")
    if best_learned is None:
        text.append("No learned model produced a scoreable AUPRC. Nothing to "
                    "claim.")
    else:
        text.append(f"Best learned model by test AUPRC : {best_learned} "
                    f"({_fmt(rows[best_learned]['AUPRC'])})")
    text.append(f"Best of ALL methods by test AUPRC: {best_overall} "
                f"({_fmt(rows[best_overall]['AUPRC']) if best_overall else 'n/a'})")
    text.append(f"Best on NEW INITIATION (AUPRC)  : {best_ni} "
                f"({_fmt(rows[best_ni]['NI_AUPRC']) if best_ni else 'n/a'})")
    text.append("")

    summary: dict[str, dict] = {}
    for ln in learned:
        text.append(f"{ln}:")
        per_ref: dict[str, dict] = {}
        for rn in refs:
            agg = _compare(ln, rows[ln], rn, rows[rn], "AUPRC")
            csi = _compare(ln, rows[ln], rn, rows[rn], "CSI")
            nia = _compare(ln, rows[ln], rn, rows[rn], "NI_AUPRC")
            nic = _compare(ln, rows[ln], rn, rows[rn], "NI_CSI")
            per_ref[rn] = {"AUPRC": agg, "CSI": csi,
                           "NI_AUPRC": nia, "NI_CSI": nic}
            if not agg["comparable"]:
                text.append(f"  vs {rn:14s} NOT COMPARABLE (a metric is nan -- "
                            f"check the stratification warnings)")
                continue
            word = "BEATS" if agg["learned_wins"] else "does NOT beat"
            text.append(
                f"  vs {rn:14s} aggregate AUPRC {_fmt(agg['learned_value'])} "
                f"vs {_fmt(agg['reference_value'])} -> {word} it "
                f"({agg['margin']:+.4f}"
                + (f", {agg['relative']:+.1%})" if agg['relative'] is not None
                   else ")"))
            text.append(
                f"  {'':17s} aggregate CSI   {_fmt(csi['learned_value'])} "
                f"vs {_fmt(csi['reference_value'])} -> "
                f"{'better' if csi['learned_wins'] else 'worse or equal'}")
            if nia["comparable"]:
                text.append(
                    f"  {'':17s} NEW-INIT AUPRC  {_fmt(nia['learned_value'])} "
                    f"vs {_fmt(nia['reference_value'])} -> "
                    f"{'better' if nia['learned_wins'] else 'worse or equal'}"
                    f";  NEW-INIT CSI {_fmt(nic['learned_value'])} vs "
                    f"{_fmt(nic['reference_value'])} -> "
                    f"{'better' if nic['learned_wins'] else 'worse or equal'}")
            else:
                text.append(f"  {'':17s} NEW-INIT: too few positives in the "
                            f"clean subset to score")
        summary[ln] = per_ref
        text.append("")

    # ---- the one-paragraph bottom line ----
    if best_learned is not None and refs:
        beat_adv = summary.get(best_learned, {}).get("advection", {}) \
            .get("AUPRC", {}).get("learned_wins", False)
        beat_chg = summary.get(best_learned, {}).get("charging_rule", {}) \
            .get("AUPRC", {}).get("learned_wins", False)
        ni_adv = summary.get(best_learned, {}).get("advection", {}) \
            .get("NI_AUPRC", {}).get("learned_wins", False)
        ni_chg = summary.get(best_learned, {}).get("charging_rule", {}) \
            .get("NI_AUPRC", {}).get("learned_wins", False)
        text.append("BOTTOM LINE")
        text.append("-" * 78)
        if not test_usable:
            text.append(
                "WITHHELD. The test split failed its quality checks, so no "
                "claim about which method is better is supportable from this "
                "run -- in either direction. Re-run with more events before "
                "quoting any of the numbers above.")
        elif beat_adv and beat_chg:
            text.append(
                f"The learned approach ({best_learned}) beat BOTH the advection "
                f"baseline and the charging-layer rule on aggregate AUPRC.")
        elif beat_adv:
            text.append(
                f"{best_learned} beat advection on aggregate AUPRC but did NOT "
                f"beat the charging-layer rule. The physics threshold is still "
                f"the better forecast, so the network has not learned anything "
                f"beyond it.")
        elif beat_chg:
            text.append(
                f"{best_learned} beat the charging-layer rule but did NOT beat "
                f"advection. Since advection is what operational extrapolation "
                f"nowcasting already does, this is not yet a deployable gain.")
        else:
            text.append(
                f"{best_learned} did NOT beat advection and did NOT beat the "
                f"charging-layer rule on aggregate AUPRC. At this training "
                f"budget that is the expected outcome and it is reported as-is: "
                f"the learned model currently adds no skill over the trivial "
                f"competition.")
        if not test_usable:
            pass
        elif ni_adv or ni_chg:
            who = " and ".join(
                [n for n, f in (("advection", ni_adv),
                                ("charging_rule", ni_chg)) if f])
            text.append(
                f"On NEW INITIATION -- the subset persistence and advection "
                f"cannot score on -- {best_learned} beat {who}. That subset is "
                f"where genuine forecast value lives, so this is the more "
                f"meaningful of the two results even when the aggregate "
                f"comparison goes the other way.")
        else:
            text.append(
                f"On NEW INITIATION {best_learned} did not beat the references "
                f"either. There is no subset on which the learned model is "
                f"currently ahead.")
        text.append("")
        text.append("Caveats that apply to every number above, stated so they "
                    "are not discovered later:")
        text.append("  * the training budget is wall-clock seconds on 2 CPU "
                    "cores, so the network is underfitted by design, not "
                    "converged;")
        text.append("  * the data is synthetic, and the generator's difficulty "
                    "was tuned by hand (see synth.dataset_stats);")
        text.append("  * with a handful of test events the confidence interval "
                    "on any CSI here is wide -- read the ranking, not the "
                    "third decimal.")

    if verbose:
        print()
        print("\n".join(text))
    return {"comparisons": comparisons, "summary": summary,
            "best_learned": best_learned, "best_overall": best_overall,
            "best_new_initiation": best_ni,
            "split_problems": problems,
            "test_split_usable": bool(test_usable),
            "interpretable": bool(test_usable and best_learned is not None),
            "text": text}


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars and non-finite floats for json.dump.

    NaN is mapped to None rather than left as a bare NaN token: json.dump will
    happily emit `NaN`, which is not valid JSON and is rejected by most parsers,
    so a results file would be unreadable by anything except Python.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run_experiment(n_events: int = 24,
                   seed: int = C.SYNTH["seed"],
                   frames_per_event: int | None = None,
                   split_seed: int = 0,
                   stride: int = C.INPUT_FRAMES,
                   eval_stride: int | None = None,
                   max_samples: dict[str, int] | None = None,
                   models: tuple[str, ...] = ("convlstm", "hybrid", "blend"),
                   hidden: int | None = None,
                   downsample: int | None = None,
                   lr: float | None = None,
                   epochs: int = 60,
                   max_seconds: float = 60.0,
                   chunk_seconds: float | None = None,
                   outdir: str = DEFAULT_OUTDIR,
                   tag: str = "run",
                   resume: bool = True,
                   use_cache: bool = True,
                   do_train: bool = True,
                   do_eval: bool = True,
                   verbose: bool = True) -> dict:
    """End-to-end experiment. Returns a dict; also writes JSON and checkpoints.

    Args:
        n_events: independent storm events to generate. config.SYNTH sets 60 for
            a full run; the default here is smaller so that a single sandbox
            call finishes.
        stride / eval_stride: window spacing in frames. INPUT_FRAMES makes
            consecutive training samples share no input frames.
        models: any of "convlstm" (plain Eulerian ConvLSTM), "hybrid"
            (motion-compensated), "blend" (calibrated advection + best learned).
        epochs / max_seconds: BOTH are limits; whichever binds first stops
            training. max_seconds is the one that normally binds here.
        do_train / do_eval: split the work across sandbox calls. `do_train=False`
            with resume=True evaluates whatever the checkpoints currently hold.

    Every knob that changes the data or the split is recorded in the output JSON,
    because a results file that cannot be traced back to the configuration that
    produced it is not a result.
    """
    t_run = time.time()
    os.makedirs(outdir, exist_ok=True)
    eval_stride = stride if eval_stride is None else int(eval_stride)

    out: dict[str, Any] = {
        "tag": tag,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_events": n_events, "seed": seed,
            "frames_per_event": frames_per_event, "split_seed": split_seed,
            "stride": stride, "eval_stride": eval_stride,
            "models": list(models), "hidden": hidden,
            "downsample": downsample, "lr": lr,
            "epochs": epochs, "max_seconds": max_seconds,
            "grid": [C.GRID_H, C.GRID_W], "input_frames": C.INPUT_FRAMES,
            "output_frames": C.OUTPUT_FRAMES,
            "n_features": FT.N_FEATURES,
            "threshold_grid": THRESHOLD_GRID,
            "threshold_selected_on": "validation split, maximising CSI",
        },
    }

    # ---- 1. data ---------------------------------------------------------
    if verbose:
        print("=" * 78)
        print(f"EXPERIMENT '{tag}'  --  {n_events} events, budget "
              f"{max_seconds:.0f}s, models {list(models)}")
        print("=" * 78)
        print("\n[1] DATA")
    events = load_or_make_events(n_events, seed, frames_per_event,
                                use_cache=use_cache, verbose=verbose)

    # ---- 2. event-blocked split -----------------------------------------
    if verbose:
        print("\n[2] EVENT-BLOCKED SPLIT")
    splits = CV.split_events(len(events), seed=split_seed)
    disj = CV.verify_split_disjoint(splits, n_events=len(events))
    if verbose:
        print(f"  train {len(splits['train'])} / val {len(splits['val'])} "
              f"/ test {len(splits['test'])} events; disjoint="
              f"{disj['disjoint']}, covers all={disj['covers_all']}")
        print(f"  test event ids: {splits['test']}")
    out["splits"] = splits
    out["split_check"] = disj

    sets = CV.build_split_sequences(events, splits, stride=stride,
                                    val_stride=eval_stride,
                                    max_samples=max_samples, seed=split_seed)
    out["sample_disjoint_check"] = CV.verify_sample_disjoint(sets)
    if verbose:
        print()
    strat = CV.stratify_report(sets, verbose=verbose)
    out["stratification"] = strat
    if verbose:
        print()
        estimate_memory(sets["train"]["X"].shape[0], sets["val"]["X"].shape[0],
                        sets["test"]["X"].shape[0])

    # Free the event list: the arrays are copied into the split tensors, and on
    # a 60-event run the originals are ~280 MB that nothing reads again.
    del events

    # ---- 3. features -----------------------------------------------------
    if verbose:
        print("\n[3] FEATURES (physics channels, strictly causal)")
    t0 = time.time()
    for name in ("train", "val", "test"):
        sets[name]["F"] = FT.build_features_batch(sets[name]["X"])
    if verbose:
        print(f"  built {FT.N_FEATURES} channels for "
              f"{sum(sets[n]['F'].shape[0] for n in sets)} samples in "
              f"{time.time() - t0:.1f}s")

    Xtr, Ytr, Ftr = sets["train"]["X"], sets["train"]["Y"], sets["train"]["F"]
    Xva, Yva, Fva = sets["val"]["X"], sets["val"]["Y"], sets["val"]["F"]
    Xte, Yte, Fte = sets["test"]["X"], sets["test"]["Y"], sets["test"]["F"]

    # ---- 4. train --------------------------------------------------------
    learned_names = [m for m in models if m in ("convlstm", "hybrid")]
    budget_each = (max_seconds / max(len(learned_names), 1)) if do_train else 0.0
    trained: dict[str, Any] = {}
    train_info: dict[str, dict] = {}

    if verbose:
        print(f"\n[4] TRAINING ({'budget %.0fs each' % budget_each if do_train else 'skipped, loading checkpoints'})")

    hp = dict(hidden=hidden, downsample=downsample, lr=lr)

    if "convlstm" in models:
        if verbose:
            print("  --- convlstm (plain Eulerian) ---")
        ck = os.path.join(outdir, f"{tag}_convlstm.npz")
        mdl, info = train_chunked(
            make=lambda: ConvLSTMNowcaster(in_ch=FT.N_FEATURES,
                                           t_out=C.OUTPUT_FRAMES, seed=1, **hp),
            ckpt=ck, F=Ftr, Y=Ytr, F_val=Fva, Y_val=Yva,
            epochs=epochs, budget=budget_each, chunk_seconds=chunk_seconds,
            loader=ConvLSTMNowcaster.load, resume=resume, verbose=verbose)
        trained["convlstm"] = mdl
        train_info["convlstm"] = info

    if "hybrid" in models:
        if verbose:
            print("  --- hybrid (motion-compensated / Lagrangian) ---")
        ck = os.path.join(outdir, f"{tag}_hybrid.npz")
        mdl, info = train_chunked(
            make=lambda: HY.MotionCompensatedNowcaster(
                in_ch=FT.N_FEATURES, t_out=C.OUTPUT_FRAMES, seed=1, **hp),
            ckpt=ck, F=Ftr, Y=Ytr, F_val=Fva, Y_val=Yva,
            epochs=epochs, budget=budget_each, chunk_seconds=chunk_seconds,
            loader=HY.MotionCompensatedNowcaster.load,
            resume=resume, verbose=verbose)
        trained["hybrid"] = mdl
        train_info["hybrid"] = info
        # Explains any long-lead deficit for this model before it appears in
        # the table: the Lagrangian coordinate change makes some Earth pixels
        # structurally unforecastable.
        cov = HY.lagrangian_coverage(
            Yte, mdl.motion(Fte)) if mdl.compensate else None
        if cov and verbose:
            print(f"  Lagrangian recall ceiling on test: overall "
                  f"{cov['overall_recoverable_frac']:.3f} of positives "
                  f"reachable; grid coverage at 5 min "
                  f"{cov['per_lead'][0]['grid_coverage_frac']:.3f} -> at 60 min "
                  f"{cov['per_lead'][-1]['grid_coverage_frac']:.3f}")
        out["lagrangian_coverage_test"] = cov
    out["training"] = train_info

    if not trained and verbose:
        print("  no learned models requested")

    if not do_eval:
        out["seconds"] = time.time() - t_run
        out["note"] = "training only; run with do_eval=True to score"
        _write_json(out, outdir, tag, verbose)
        return out

    # ---- 5. blend, fitted on validation only -----------------------------
    if "blend" in models and trained:
        if verbose:
            print("\n[5] BLEND (advection + learned, weights fitted on VAL)")
        # Which learned model to blend is itself a choice, so it is made on
        # validation AUPRC -- never on test. Recorded in the output so the
        # choice is auditable.
        val_ap = {}
        for nm, mdl in trained.items():
            val_ap[nm] = M.auprc(model_probs(nm, mdl, Xva, Fva),
                                 Yva >= C.LIGHTNING_THRESHOLD)
        pick = max(val_ap, key=lambda k: (-np.inf if math.isnan(val_ap[k])
                                          else val_ap[k]))
        if verbose:
            print(f"  learned component chosen on VAL AUPRC: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in val_ap.items())
                  + f"  -> {pick}")
        bl = HY.BlendedNowcaster(trained[pick], label=f"blend({pick})")
        bl.fit(Xva, Fva, Yva, verbose=verbose)
        bl.save(os.path.join(outdir, f"{tag}_blend.json"))
        trained[f"blend({pick})"] = bl
        out["blend"] = {"component": pick,
                        "val_auprc_by_model": _jsonable(val_ap),
                        "fit_info": _jsonable(bl.fit_info)}

    # ---- 6. forecasts ----------------------------------------------------
    if verbose:
        print("\n[6] FORECASTS (baselines + models, identical samples)")
    base_rate_tr = float((Ytr >= C.LIGHTNING_THRESHOLD).mean())
    t0 = time.time()
    probs_te = B.run_all(Xte, C.OUTPUT_FRAMES, base_rate=base_rate_tr)
    probs_va = B.run_all(Xva, C.OUTPUT_FRAMES, base_rate=base_rate_tr)
    if verbose:
        print(f"  {len(probs_te)} baselines on val+test in "
              f"{time.time() - t0:.1f}s (climatology base rate "
              f"{base_rate_tr:.4f}, taken from TRAIN)")
    for nm, mdl in trained.items():
        t0 = time.time()
        probs_te[nm] = model_probs(nm, mdl, Xte, Fte)
        probs_va[nm] = model_probs(nm, mdl, Xva, Fva)
        if verbose:
            print(f"  {nm:22s} predicted in {time.time() - t0:.1f}s")

    # ---- 7. evaluate -----------------------------------------------------
    if verbose:
        print("\n[7] EVALUATION (threshold from VAL, applied unchanged to TEST)")
    evals = evaluate_all(probs_te, Xte, Yte, probs_va, Yva, verbose=verbose)
    out["evaluations"] = _jsonable(evals)

    # Metric sanity check that costs nothing and catches a broken AUPRC:
    # climatology's average precision must equal the test base rate.
    clim = evals.get("climatology", {}).get("overall", {})
    if clim:
        br = clim.get("obs_rate", float("nan"))
        ap = clim.get("AUPRC", float("nan"))
        out["metric_sanity"] = {
            "climatology_AUPRC": ap, "test_base_rate": br,
            "agrees": bool(abs(ap - br) < 1e-6) if
            (isinstance(ap, float) and isinstance(br, float)
             and math.isfinite(ap) and math.isfinite(br)) else None,
        }
        if verbose:
            print(f"  metric sanity: climatology AUPRC {ap:.5f} vs test base "
                  f"rate {br:.5f} (must match; AUPRC's no-skill value IS the "
                  f"base rate)")

    # ---- 8. table and verdict --------------------------------------------
    tbl = comparison_table(evals)
    out["table"] = _jsonable(table_rows(evals))
    if verbose:
        print("\n[8] COMPARISON TABLE (test split; sorted by AUPRC)")
        print()
        print(format_table(tbl))
        print()
        print("  thr      = probability threshold, chosen on VALIDATION by max CSI")
        print("  FSS_r2/4 = Fractions Skill Score at 4 km / 8 km neighbourhood")
        print("  BSS      = Brier skill score vs climatology (>0 is useful)")
        print("  NI_*     = NEW INITIATION subset: pixels with no observed")
        print("             lightning within 8 km during the whole input window."
              )
        print("             Persistence scores 0 here by construction, so this")
        print("             column is the one that cannot be faked.")
        print("  val_*    = the same metrics on the validation split, for")
        print("             comparison with the test column")

    out["verdict"] = _jsonable(verdict(evals,
                                       problems=strat.get("problems"),
                                       verbose=verbose))
    out["seconds"] = time.time() - t_run
    if verbose:
        print(f"\ntotal experiment wall clock {out['seconds']:.1f}s")
    _write_json(out, outdir, tag, verbose)
    return out


def _write_json(out: dict, outdir: str, tag: str, verbose: bool = True) -> str:
    path = os.path.join(outdir, f"{tag}_results.json")
    with open(path, "w") as fh:
        json.dump(_jsonable(out), fh, indent=2, allow_nan=False)
    if verbose:
        print(f"results -> {path}")
    return path


# ---------------------------------------------------------------------------
# Small end-to-end self-test
# ---------------------------------------------------------------------------

def _self_test() -> dict:
    """Small but complete run: proves the whole chain wires up and produces a table.

    Sized from a failure. The first version used 10 events, which after a
    65/15/20 event split left 2 test events -- and with SYNTH mimic_fraction at
    0.60 the chance that both are non-electrified is high. It duly happened: the
    test split held 154 positive pixels from a single storm, every baseline
    scored exactly the base rate, and the verdict declared a sweeping victory
    for the network. The event-blocked split was working correctly; there simply
    were not enough independent electrified events on the far side of it.

    24 events puts ~5 events in test, of which ~2 electrify on average. That is
    still a small sample -- the verdict says so -- but it is enough for the
    baselines to be scoreable, which is the difference between a comparison and
    a coin flip.
    """
    return run_experiment(
        n_events=24, seed=101, frames_per_event=None, split_seed=3,
        stride=6, eval_stride=6,
        models=("convlstm", "hybrid", "blend"),
        hidden=12, downsample=2, epochs=40,
        max_seconds=40.0, tag="selftest", resume=False, use_cache=True,
        verbose=True)


if __name__ == "__main__":
    _self_test()

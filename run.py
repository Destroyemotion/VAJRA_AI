"""
Command-line entry point. Everything the project can do, without a GUI.

WHY A CLI AT ALL
----------------
Two constraints shape this file. The first is the sandbox: any single call is
killed at ~120 s, so training has to be startable, stoppable and restartable
from the shell rather than living inside one long-running process. `train`
therefore consumes a wall-clock budget, checkpoints, and exits; run it again and
it resumes. `eval` is a separate subcommand for the same reason -- scoring
seven methods is its own chunk of time and does not need to share a process with
training.

The second is that there is no display. A nowcast is a spatial field, and any
verification metric is a summary that has already thrown away the thing you most
need to look at: WHERE the model put its probability. `demo` therefore renders a
forecast as a coarse ASCII map next to a hit/miss/false-alarm map at the chosen
threshold, so a wrong forecast can be diagnosed as "right place, wrong time",
"right time, wrong place", or "smeared everything" -- distinctions that CSI
collapses into a single number.

The ASCII map deliberately shows the block MAXIMUM rather than the block mean.
Coarsening a field whose base rate is ~1% by averaging shows nothing: a single
high-probability pixel in a 4x4 block averages to a quarter of its value and
disappears into the background. The maximum is also the operationally relevant
aggregate, since a warning is issued for a district if any part of it is at risk.

SUBCOMMANDS
    describe     configuration, data sources, and what is on disk
    make-synth   generate and cache a synthetic dataset
    baselines    score the baselines only -- no training, fast, run this first
    train        train under a wall-clock budget, checkpoint, exit
    eval         load checkpoints, score everything, print the table and verdict
    demo         one sequence, forecast rendered as ASCII with observed overlay

Defaults are chosen so that every subcommand runs bare, and so that the whole
sequence `make-synth`, `train`, `train`, `eval` fits in four sandbox calls.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

import baselines as B
import config as C
import cv as CV
import features as FT
import hybrid as HY
import metrics as M
import pipeline as P
import synth
from convlstm_np import ConvLSTMNowcaster

HERE = os.path.dirname(os.path.abspath(__file__))


def _importable(name: str) -> bool:
    """True if a module could be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

# Probability ramp. The breakpoints are not linear: they are dense at the low end
# because that is where the decision-relevant range lives for a 1% base-rate
# event. A linear ramp would render almost every useful forecast as blank.
PROB_LEVELS = [(0.02, " "), (0.05, "."), (0.10, ":"), (0.20, "-"),
               (0.35, "+"), (0.55, "*"), (0.75, "#"), (1.01, "@")]


def _symbol(v: float, levels=PROB_LEVELS) -> str:
    for hi, ch in levels:
        if v < hi:
            return ch
    return levels[-1][1]


def coarsen_max(field: np.ndarray, block: int = 4) -> np.ndarray:
    """Block maximum. See the module docstring for why max and not mean."""
    h, w = field.shape
    bh, bw = h // block, w // block
    return field[:bh * block, :bw * block].reshape(bh, block, bw, block) \
        .max(axis=(1, 3))


def ascii_panel(field: np.ndarray, block: int = 4,
                levels=PROB_LEVELS) -> list[str]:
    c = coarsen_max(np.asarray(field, dtype=np.float64), block)
    return ["".join(_symbol(v, levels) for v in row) for row in c]


def ascii_contingency(prob: np.ndarray, obs: np.ndarray, threshold: float,
                      block: int = 4) -> list[str]:
    """Hit / miss / false-alarm map at the operating threshold.

    H hit, M miss, F false alarm, '.' correct null. This is the panel that
    diagnoses a forecast: a field of M with F displaced a few blocks away is a
    position error, whereas M and F interleaved is a timing error, and neither
    looks different in the CSI.
    """
    f = coarsen_max(np.asarray(prob, dtype=np.float64), block) >= threshold
    o = coarsen_max((np.asarray(obs) >= C.LIGHTNING_THRESHOLD).astype(
        np.float64), block) > 0
    rows = []
    for i in range(f.shape[0]):
        rows.append("".join("H" if (f[i, j] and o[i, j]) else
                            "M" if o[i, j] else
                            "F" if f[i, j] else "."
                            for j in range(f.shape[1])))
    return rows


def side_by_side(left: list[str], right: list[str],
                 lhead: str = "", rhead: str = "", gap: int = 4) -> str:
    w = max([len(r) for r in left] + [len(lhead)])
    pad = " " * gap
    out = [f"{lhead:<{w}}{pad}{rhead}"]
    for a, b in zip(left, right):
        out.append(f"{a:<{w}}{pad}{b}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def cmd_describe(args: argparse.Namespace) -> int:
    print(C.describe())
    print()
    print("FEATURES")
    print("-" * 78)
    print(f"  {FT.N_FEATURES} engineered channels from {C.N_CHANNELS} raw:")
    for i in range(0, FT.N_FEATURES, 5):
        print("    " + ", ".join(FT.FEATURE_NAMES[i:i + 5]))
    print()
    print("BASELINES (every one of these is scored alongside the models)")
    print("-" * 78)
    for name in B.BASELINES:
        doc = (B.BASELINES[name].__doc__ or "").strip().split("\n")[0]
        print(f"  {name:16s} {doc}")
    print()
    print("MODELS")
    print("-" * 78)
    print("  convlstm       pure-NumPy ConvLSTM encoder-forecaster (Eulerian)")
    print("  hybrid         same net trained in a motion-compensated "
          "(Lagrangian) frame")
    print("  blend          calibrated advection + learned, lead-dependent "
          "weight, fitted on val")
    print()
    print("DATA SOURCES")
    print("-" * 78)
    print("  synth        [AVAILABLE]  in-repo charging-physics simulator "
          "(synth.py); no deps")
    for key, src in C.DATA_SOURCES.items():
        needs = list(src.get("needs", []))
        missing = [n for n in needs if not _importable(n)]
        state = "AVAILABLE" if not missing else \
            "missing " + ",".join(missing)
        print(f"  {key:12s} [{state}]  {src.get('desc', '')}")
        print(f"  {'':12s} cadence {src.get('cadence_min', '?')} min; "
              f"auth: {src.get('auth', '?')}")
        if src.get("note"):
            print(f"  {'':12s} {src['note']}")
    print("  NOTE: network access is unavailable in this environment, so every")
    print("        real source above is documented rather than reachable. The")
    print("        synthetic generator exists precisely so the pipeline can be")
    print("        validated end-to-end without one.")
    print()
    print("ON DISK")
    print("-" * 78)
    for label, pattern in (("cached datasets", os.path.join(P.DEFAULT_CACHE,
                                                            "*.npz")),
                           ("checkpoints", os.path.join(P.DEFAULT_OUTDIR,
                                                        "*.npz")),
                           ("results", os.path.join(P.DEFAULT_OUTDIR,
                                                    "*_results.json"))):
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"  {label:16s} (none)")
        for h in hits:
            print(f"  {label:16s} {os.path.basename(h):40s} "
                  f"{os.path.getsize(h) / 1e6:7.1f} MB")
    return 0


# ---------------------------------------------------------------------------
# make-synth
# ---------------------------------------------------------------------------

def cmd_make_synth(args: argparse.Namespace) -> int:
    t0 = time.time()
    events = P.load_or_make_events(args.n_events, args.seed, args.frames,
                                   use_cache=not args.no_cache, verbose=True)
    print()
    print(synth.dataset_stats(events))
    print()
    # The split is deterministic given (n_events, split_seed), so reporting the
    # stratification here -- before any training -- is what tells you whether the
    # dataset is even usable. A test split with no lightning in it makes every
    # downstream number meaningless, and it is far cheaper to discover now.
    splits = CV.split_events(len(events), seed=args.split_seed)
    sets = CV.build_split_sequences(events, splits, stride=args.stride,
                                    val_stride=args.stride)
    rep = CV.stratify_report(sets, verbose=True)
    if rep["problems"]:
        print()
        print("This dataset is NOT usable for ranking models as split. Either "
              "raise --n-events, or change --split-seed and re-check.")
        return 1
    print(f"\nok in {time.time() - t0:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# shared data preparation
# ---------------------------------------------------------------------------

def _prepare(args: argparse.Namespace, keep: tuple[str, ...] = ("test",),
             features: tuple[str, ...] = (), verbose: bool = True) -> dict:
    """Events -> split -> sequences -> features, for the requested splits only.

    `keep` and `features` are separate arguments rather than one, because the two
    costs are very different and the two needs do not coincide. Raw channels are
    0.6 MB/sample and every baseline needs them; the feature stack is 2 MB/sample
    and only the network needs it. Collapsing them into one flag made
    `run.py baselines` build 20 physics channels it never read.
    """
    events = P.load_or_make_events(args.n_events, args.seed, args.frames,
                                   use_cache=True, verbose=verbose)
    splits = CV.split_events(len(events), seed=args.split_seed)
    sets = CV.build_split_sequences(events, splits, stride=args.stride,
                                    val_stride=args.eval_stride)
    # Captured before anything is freed. climatology's constant probability must
    # come from TRAIN, exactly as it does in pipeline.run_experiment: taking it
    # from the split being scored would hand the trivial baseline the answer.
    base_rate = float((sets["train"]["Y"] >= C.LIGHTNING_THRESHOLD).mean())
    for k in list(sets):
        if k not in keep:
            sets[k].pop("X", None)
            sets[k].pop("Y", None)
        elif k in features:
            sets[k]["F"] = FT.build_features_batch(sets[k]["X"])
    sets["_splits"] = splits
    sets["_base_rate_train"] = base_rate
    return sets


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def cmd_baselines(args: argparse.Namespace) -> int:
    """Baselines only. Fast, needs no checkpoint, and it sets the bar.

    Worth running before any training: if advection already scores 0.4 AUPRC on
    your data then a network has very little room, and if it scores 0.02 then the
    task may be badly posed. Either way you learn it in ten seconds.
    """
    sets = _prepare(args, keep=("val", "test"))
    Xva, Yva = sets["val"]["X"], sets["val"]["Y"]
    Xte, Yte = sets["test"]["X"], sets["test"]["Y"]
    base_rate = sets["_base_rate_train"]
    print(f"\nscoring {len(B.BASELINES)} baselines on {Xte.shape[0]} test "
          f"sequences (thresholds from {Xva.shape[0]} val sequences, "
          f"climatology rate {base_rate:.4f} from train)")
    pv = B.run_all(Xva, C.OUTPUT_FRAMES, base_rate=base_rate)
    pt = B.run_all(Xte, C.OUTPUT_FRAMES, base_rate=base_rate)
    print()
    evals = P.evaluate_all(pt, Xte, Yte, pv, Yva, verbose=True)
    print()
    print(P.format_table(P.comparison_table(evals)))
    if args.report:
        for name in evals:
            print()
            print(M.format_report(evals[name]))
    return 0


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------

def _experiment_kwargs(args: argparse.Namespace) -> dict:
    return dict(n_events=args.n_events, seed=args.seed,
                frames_per_event=args.frames, split_seed=args.split_seed,
                stride=args.stride, eval_stride=args.eval_stride,
                models=tuple(args.models.split(",")),
                hidden=args.hidden, downsample=args.downsample, lr=args.lr,
                epochs=args.epochs, max_seconds=args.max_seconds,
                outdir=args.outdir, tag=args.tag, resume=not args.no_resume,
                verbose=not args.quiet)


def cmd_train(args: argparse.Namespace) -> int:
    """Consume a wall-clock budget, checkpoint, exit. Re-run to continue.

    Training does NOT evaluate. Scoring seven methods on the test split takes
    ~15 s of its own, and spending it at the end of every training chunk both
    wastes budget and tempts you to watch the test number move -- which is how a
    test split stops being a test split.
    """
    out = P.run_experiment(do_train=True, do_eval=False,
                           **_experiment_kwargs(args))
    ti = out.get("training", {})
    if ti and not args.quiet:
        done = all(v.get("complete") for v in ti.values())
        print()
        print("epochs: " + ", ".join(f"{k}={v['epochs_done']}/"
                                    f"{v['epochs_target']}"
                                    for k, v in ti.items()))
        print("run `python3 run.py train` again to continue, or "
              "`python3 run.py eval` to score."
              if not done else
              "epoch target reached; run `python3 run.py eval` to score.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    kw = _experiment_kwargs(args)
    kw["max_seconds"] = 0.0
    out = P.run_experiment(do_train=False, do_eval=True, **kw)
    v = out.get("verdict", {})
    return 0 if v.get("test_split_usable", True) else 2


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def _load_model(name: str, tag: str, outdir: str, in_ch: int):
    """Load a trained model by short name, or raise with a useful message."""
    if name == "convlstm":
        return ConvLSTMNowcaster.load(os.path.join(outdir, f"{tag}_convlstm.npz"))
    if name == "hybrid":
        return HY.MotionCompensatedNowcaster.load(
            os.path.join(outdir, f"{tag}_hybrid.npz"))
    if name.startswith("blend"):
        bp = os.path.join(outdir, f"{tag}_blend.json")
        with open(bp) as fh:
            label = json.load(fh).get("label", "blend(convlstm)")
        comp = label[label.find("(") + 1:label.rfind(")")] or "convlstm"
        return HY.BlendedNowcaster.load(bp, _load_model(comp, tag, outdir,
                                                        in_ch))
    raise SystemExit(f"unknown model '{name}' "
                     f"(expected convlstm, hybrid or blend)")


def cmd_demo(args: argparse.Namespace) -> int:
    """One test sequence, rendered so a wrong forecast can be diagnosed by eye."""
    sets = _prepare(args, keep=("test",), features=("test",))
    Xte, Yte, Fte = sets["test"]["X"], sets["test"]["Y"], sets["test"]["F"]
    n = Xte.shape[0]

    idx = args.sample
    if idx < 0:
        # Default to the most electrified test sequence. A demo on a quiet
        # sequence shows two blank panels and teaches nothing.
        idx = int(np.argmax((Yte >= C.LIGHTNING_THRESHOLD)
                            .reshape(n, -1).sum(axis=1)))
        print(f"no --sample given; using the most electrified test sequence "
              f"({idx} of {n})")
    if not 0 <= idx < n:
        raise SystemExit(f"--sample must be in [0, {n})")

    model = _load_model(args.model, args.tag, args.outdir, FT.N_FEATURES)
    x1, f1, y1 = Xte[idx:idx + 1], Fte[idx:idx + 1], Yte[idx:idx + 1]
    prob = P.model_probs(args.model, model, x1, f1)[0]
    obs = y1[0]
    ev_id = int(sets["test"]["sid"][idx])

    thr = args.threshold
    if thr is None:
        # Prefer the threshold this experiment selected on VALIDATION. Picking a
        # threshold to make the demo look good would be exactly the failure the
        # rest of the code is built to prevent.
        rp = os.path.join(args.outdir, f"{args.tag}_results.json")
        thr = 0.15
        src = "default (no results file found)"
        if os.path.exists(rp):
            with open(rp) as fh:
                res = json.load(fh)
            for row in res.get("table", []):
                if row["model"].startswith(args.model):
                    thr = float(row["thr_val"])
                    src = f"selected on VALIDATION in {args.tag}_results.json"
                    break
    else:
        src = "given on the command line"

    print()
    print("=" * 78)
    print(f"DEMO  model={args.model}  test sample {idx}/{n}  storm event "
          f"{ev_id}  threshold={thr:.3f} ({src})")
    print(f"grid {C.GRID_H}x{C.GRID_W} at {C.PIXEL_KM:.0f} km, rendered as "
          f"{C.GRID_H // args.block}x{C.GRID_W // args.block} blocks of "
          f"{args.block * C.PIXEL_KM:.0f} km (block MAXIMUM)")
    print("=" * 78)

    # ---- what the model was looking at ----
    last = x1[0, -1]
    m10 = last[C.CH["refl_m10"]]
    m10 = np.where(m10 > C.FILL_VALUE / 2, m10, 0.0)
    ltg_in = x1[0, :, C.CH["light_dens"]].max(axis=0)
    dbz_levels = [(5, " "), (10, "."), (15, ":"), (20, "-"), (25, "+"),
                  (30, "*"), (35, "#"), (1e9, "@")]
    print()
    print(side_by_side(
        ascii_panel(m10, args.block, dbz_levels),
        ascii_panel((ltg_in >= C.LIGHTNING_THRESHOLD).astype(float),
                    args.block, [(0.5, "."), (1.01, "L")]),
        "INPUT t=0: charging-layer dBZ", "INPUT: any lightning in window"))
    print("  left  ' '<5 .<10 :<15 -<20 +<25 *<30 #<35 @>=35 dBZ at -10C")
    print("  right L = at least one strike here during the 30 min input window")
    print("        blocks marked '.' on the right are where a NEW-INITIATION")
    print("        forecast has to come from physics, not from persistence")

    # ---- forecasts ----
    leads = [int(v) for v in args.leads.split(",")]
    for lead in leads:
        t = lead // C.TIMESTEP_MIN - 1
        if not 0 <= t < prob.shape[0]:
            continue
        p, o = prob[t], obs[t]
        print()
        print(side_by_side(
            ascii_panel(p, args.block),
            ascii_contingency(p, o, thr, args.block),
            f"+{lead:2d} min FORECAST  (max p={p.max():.2f})",
            f"+{lead:2d} min H=hit M=miss F=false alarm"))
        nb = coarsen_max((o >= C.LIGHTNING_THRESHOLD).astype(float),
                         args.block) > 0
        fb = coarsen_max(p, args.block) >= thr
        h = int((nb & fb).sum())
        m_ = int((nb & ~fb).sum())
        fa = int((~nb & fb).sum())
        csi = h / max(h + m_ + fa, 1)
        print(f"  blocks: {h} hit, {m_} miss, {fa} false alarm  ->  "
              f"block CSI {csi:.3f}   "
              f"({int((o >= C.LIGHTNING_THRESHOLD).sum())} strike pixels "
              f"observed)")
    print()
    print("  probability ramp: ' '<0.02 .<0.05 :<0.10 -<0.20 +<0.35 *<0.55 "
          "#<0.75 @>=0.75")

    # ---- numeric summary, because eyes are unreliable about calibration ----
    print()
    print("  lead   max p   mean p   obs px   forecast px>=thr")
    for t in range(prob.shape[0]):
        pk = prob[t]
        print(f"  {(t + 1) * C.TIMESTEP_MIN:3d}min  {pk.max():.3f}   "
              f"{pk.mean():.4f}   {int((obs[t] >= C.LIGHTNING_THRESHOLD).sum()):6d}"
              f"   {int((pk >= thr).sum()):6d}")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _add_data_args(p: argparse.ArgumentParser) -> None:
    """Arguments that select the DATA. Identical across subcommands on purpose.

    The split is a deterministic function of (n_events, seed, split_seed), so as
    long as these three agree, `train`, `eval` and `demo` are guaranteed to be
    looking at the same test events across separate process invocations. If they
    disagree, `eval` would score a model on events it was trained on -- which is
    why they are one shared block of arguments rather than copied per command.
    """
    p.add_argument("--n-events", type=int, default=24,
                   help="independent storm events (config.SYNTH has %d)"
                        % C.SYNTH["n_events"])
    p.add_argument("--seed", type=int, default=101,
                   help="synthetic data seed")
    p.add_argument("--frames", type=int, default=None,
                   help="frames per event (default config.SYNTH)")
    p.add_argument("--split-seed", type=int, default=3,
                   help="seed for the event-blocked split")
    p.add_argument("--stride", type=int, default=C.INPUT_FRAMES,
                   help="training window stride in frames")
    p.add_argument("--eval-stride", type=int, default=C.INPUT_FRAMES,
                   help="val/test window stride in frames")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", default="convlstm,hybrid,blend",
                   help="comma list of convlstm,hybrid,blend")
    p.add_argument("--hidden", type=int, default=12)
    p.add_argument("--downsample", type=int, default=2,
                   help="model's internal grid stride (targets too, so "
                        "calibration is preserved)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=40,
                   help="epoch ceiling; the time budget usually binds first")
    p.add_argument("--max-seconds", type=float, default=70.0,
                   help="TOTAL wall-clock training budget for this call, split "
                        "evenly across the learned models requested; keep under "
                        "~100 so one sandbox call cannot be killed mid-update")
    p.add_argument("--tag", default="run", help="checkpoint/result name prefix")
    p.add_argument("--outdir", default=P.DEFAULT_OUTDIR)
    p.add_argument("--no-resume", action="store_true",
                   help="ignore existing checkpoints and start from scratch")
    p.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="Early lightning nowcasting, pure NumPy. "
                    "Typical session: make-synth, baselines, train, train, "
                    "eval, demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("describe", help="configuration, data sources, disk")
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("make-synth", help="generate and cache a dataset")
    _add_data_args(p)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_make_synth)

    p = sub.add_parser("baselines", help="score baselines only (no training)")
    _add_data_args(p)
    p.add_argument("--report", action="store_true",
                   help="also print the full per-lead report for each")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("train", help="train under a budget, checkpoint, exit")
    _add_data_args(p)
    _add_model_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="score everything, print table and verdict")
    _add_data_args(p)
    _add_model_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("demo", help="one sequence as an ASCII forecast map")
    _add_data_args(p)
    p.add_argument("--model", default="hybrid",
                   help="convlstm, hybrid or blend")
    p.add_argument("--tag", default="run")
    p.add_argument("--outdir", default=P.DEFAULT_OUTDIR)
    p.add_argument("--sample", type=int, default=-1,
                   help="test sequence index; default picks the most "
                        "electrified one")
    p.add_argument("--leads", default="5,15,30,60",
                   help="comma list of lead times in minutes")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the validation-selected threshold")
    p.add_argument("--block", type=int, default=4,
                   help="pixels per ASCII cell")
    p.set_defaults(func=cmd_demo)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        print()
        print("no subcommand given; showing configuration instead\n")
        return cmd_describe(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

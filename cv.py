"""
Event-blocked splitting and sequence construction.

WHY THIS FILE IS A CORRECTNESS FILE, NOT A CONVENIENCE FILE
-----------------------------------------------------------
The single easiest way to produce a spectacular and completely fake nowcasting
result is to split at the level of SAMPLES rather than EVENTS.

Consider two training samples drawn from the same storm event at t0 = 30 and
t0 = 31. Their input windows are frames 30-35 and 31-36: five of six frames are
literally the same array, and consecutive 5 min radar frames are near-identical
anyway. Their targets overlap in eleven of twelve frames. If one lands in train
and the other in test, the "test" score is measuring memorisation of an image
the model already fitted. The inflation is not subtle -- it is the difference
between a CSI of 0.2 and a CSI of 0.6, i.e. between an honest result and a
fabricated one.

So the rule enforced here is absolute: an EVENT (one independent storm day)
belongs to exactly one split, and every sample inherits its event's split. No
frame from an event ever appears on both sides of a boundary.
`naive_sample_split_leakage` quantifies exactly how much leakage the wrong
choice produces on this data, so the claim above is measured rather than
asserted.

THE SECOND FAILURE MODE: A SPLIT WITH NO LIGHTNING
-------------------------------------------------
Base rate is ~1%, and config.SYNTH deliberately makes ~20% of events entirely
non-electrified. With a small number of events it is entirely possible for the
test split to contain zero positive pixels. Every rare-event metric then returns
nan (metrics.auprc returns nan when n_pos == 0, by design), the pipeline prints
a table of nans, and the run is silently worthless. `stratify_report` exists to
make that condition loud rather than silent, and it reports both the pixel base
rate and the fraction of SAMPLES that contain any lightning -- the second number
matters because a single very active event can supply all the positives in a
split, in which case the effective sample size for the headline metric is one.

WHAT `gap_frames` IS FOR
------------------------
config.SPLIT["gap_frames"] is INPUT_FRAMES + OUTPUT_FRAMES: the full temporal
footprint of one sample. For the synthetic generator it is not needed, because
each event is generated from independent random draws and there is no temporal
continuity between event k and event k+1 -- event blocking alone is sufficient.
It becomes essential on real data, where "events" are usually contiguous slices
carved out of one continuous radar archive: there, adjacent slices touch, and
`guard_frames` must be set to SPLIT["gap_frames"] so the last sample of one
slice and the first sample of the next cannot overlap in time.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Event-level splitting
# ---------------------------------------------------------------------------

def split_events(n_events: int,
                 train_frac: float | None = None,
                 val_frac: float | None = None,
                 test_frac: float | None = None,
                 seed: int = 0,
                 mode: str = "shuffle") -> dict[str, list[int]]:
    """Partition event indices into train / val / test.

    Args:
        n_events: number of independent events available.
        train_frac, val_frac, test_frac: default to config.SPLIT.
        seed: makes the partition deterministic. The same seed and n_events
            always give the same partition, which is what allows a chunked,
            resumable training run spread over several process invocations to
            keep training on the same data it started on. Getting this wrong
            would mean a "resumed" run silently trains on former test events.
        mode: "shuffle" assigns events at random; "chronological" assigns the
            first events to train and the last to test.

    WHICH MODE TO USE. For the synthetic generator the events are exchangeable
    (independent draws from one distribution), so "shuffle" is correct and
    gives better-balanced splits at small n. For a real archive use
    "chronological": storm climatology drifts with season and with the monsoon
    onset, and a shuffled split lets the model see the same synoptic regime --
    sometimes the same mesoscale system on the same afternoon -- in both train
    and test. That is a subtler leak than frame overlap but it inflates scores
    in the same direction.

    Returns:
        {"train": [...], "val": [...], "test": [...]} of event indices.
    """
    sp = C.SPLIT
    ftr = sp["train_frac"] if train_frac is None else float(train_frac)
    fva = sp["val_frac"] if val_frac is None else float(val_frac)
    fte = sp["test_frac"] if test_frac is None else float(test_frac)
    tot = ftr + fva + fte
    if not np.isclose(tot, 1.0):
        # Renormalise rather than fail: a caller asking for 60/20/20 as
        # (0.6, 0.2, 0.2) and a caller asking for (3, 1, 1) both mean the same
        # thing, and silently dropping the remainder would be worse.
        ftr, fva, fte = ftr / tot, fva / tot, fte / tot

    if n_events < 3:
        raise ValueError(
            f"need at least 3 events to form three splits, got {n_events}")

    idx = np.arange(n_events)
    if mode == "shuffle":
        idx = np.random.default_rng(seed).permutation(idx)
    elif mode == "chronological":
        pass
    else:
        raise ValueError(f"unknown mode {mode!r}")

    n_tr = int(round(ftr * n_events))
    n_va = int(round(fva * n_events))
    # Guarantee every split is non-empty. Rounding on small n can otherwise
    # produce an empty val split, and then threshold selection silently falls
    # back to whatever the caller does with an empty array -- usually nan.
    n_tr = max(1, min(n_tr, n_events - 2))
    n_va = max(1, min(n_va, n_events - n_tr - 1))

    out = {
        "train": sorted(int(i) for i in idx[:n_tr]),
        "val": sorted(int(i) for i in idx[n_tr:n_tr + n_va]),
        "test": sorted(int(i) for i in idx[n_tr + n_va:]),
    }
    verify_split_disjoint(out, n_events=n_events)
    return out


def verify_split_disjoint(splits: dict[str, Sequence[int]],
                          n_events: int | None = None,
                          strict: bool = True) -> dict:
    """Assert that no event index appears in more than one split.

    This is cheap and it is checked on every call to `split_events`, because the
    cost of the check is microseconds and the cost of the bug is an entire
    invalid set of results. `strict` also requires that the splits cover every
    event exactly once, which catches an event silently dropped by a rounding
    error in the fraction arithmetic.

    Returns a diagnostics dict; raises AssertionError on violation.
    """
    seen: dict[int, str] = {}
    collisions: list[tuple[int, str, str]] = []
    for name, ids in splits.items():
        for e in ids:
            if e in seen:
                collisions.append((int(e), seen[e], name))
            else:
                seen[int(e)] = name
    assert not collisions, (
        "EVENT APPEARS IN MORE THAN ONE SPLIT -- results would be invalid: "
        + ", ".join(f"event {e} in both {a} and {b}" for e, a, b in collisions))

    info = {
        "sizes": {k: len(v) for k, v in splits.items()},
        "n_unique_events": len(seen),
        "disjoint": True,
    }
    if n_events is not None:
        info["n_events"] = int(n_events)
        info["covers_all"] = (len(seen) == n_events)
        if strict:
            missing = sorted(set(range(n_events)) - set(seen))
            assert not missing, f"events dropped by the split: {missing}"
    return info


# ---------------------------------------------------------------------------
# Window enumeration
# ---------------------------------------------------------------------------

def enumerate_windows(events: Sequence[dict],
                      event_ids: Iterable[int],
                      stride: int = 1,
                      guard_frames: int = 0,
                      input_frames: int | None = None,
                      output_frames: int | None = None
                      ) -> list[tuple[int, int]]:
    """List of (event_id, t0) for every admissible sample window.

    A window at t0 occupies frames [t0, t0 + T_in + T_out). Windows never cross
    an event boundary, because a boundary is a discontinuity in the physical
    state and a sample straddling it would ask the model to forecast a
    different storm from the one it was shown.

    Args:
        stride: step between consecutive t0 within an event. stride=1 gives the
            most samples but they are almost perfectly correlated, so the
            effective sample size is far below the nominal count and the loss
            curve becomes over-optimistic about how much data there is. A stride
            of roughly T_in makes successive samples share no input frames.
        guard_frames: frames dropped at both ends of each event. Zero is right
            for independent synthetic events; set to config.SPLIT["gap_frames"]
            when events are contiguous slices of one archive.
    """
    t_in = C.INPUT_FRAMES if input_frames is None else int(input_frames)
    t_out = C.OUTPUT_FRAMES if output_frames is None else int(output_frames)
    span = t_in + t_out
    stride = max(1, int(stride))

    windows: list[tuple[int, int]] = []
    for e in event_ids:
        ev = events[int(e)]
        n_frames = int(ev["x"].shape[0])
        lo = int(guard_frames)
        hi = n_frames - int(guard_frames) - span      # last admissible t0, +1
        if hi < lo:
            continue
        for t0 in range(lo, hi + 1, stride):
            windows.append((int(e), int(t0)))
    return windows


def make_sequences(events: Sequence[dict],
                   event_ids: Iterable[int],
                   stride: int = 1,
                   guard_frames: int = 0,
                   input_frames: int | None = None,
                   output_frames: int | None = None,
                   max_samples: int | None = None,
                   seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice the given events into supervised (X, Y) samples.

    Args:
        events: list of event dicts from synth.generate_dataset (or a real
            loader producing the same interface: "x" (T, C, H, W) raw channels,
            "y" (T, H, W) strike counts).
        event_ids: which events to use -- normally one split's index list.
        max_samples: cap, applied by uniform random subsampling of the window
            list rather than truncation. Truncating would keep only the
            earliest events and quietly change the population being trained on.

    Returns:
        X   (N, T_in, C, H, W)  float32 raw channels, FILL_VALUE preserved
        Y   (N, T_out, H, W)    float32 strike COUNTS (not binarised)
        sid (N,)                int32 event id for every sample

    Y is left as counts, not binarised, for two reasons: metrics.evaluate
    applies config.LIGHTNING_THRESHOLD itself, and keeping counts leaves the
    door open to a rate-based target later without regenerating the arrays.

    `sid` is the traceability handle. Every downstream assertion about leakage,
    and every per-event error breakdown, needs to map a sample back to the
    event it came from; reconstructing that after the fact from array positions
    is exactly the kind of bookkeeping that goes wrong silently.
    """
    t_in = C.INPUT_FRAMES if input_frames is None else int(input_frames)
    t_out = C.OUTPUT_FRAMES if output_frames is None else int(output_frames)

    windows = enumerate_windows(events, event_ids, stride, guard_frames,
                                t_in, t_out)
    if max_samples is not None and len(windows) > max_samples:
        keep = np.random.default_rng(seed).choice(
            len(windows), size=int(max_samples), replace=False)
        windows = [windows[i] for i in sorted(keep.tolist())]

    if not windows:
        raise ValueError(
            "no admissible windows: events are shorter than "
            f"{t_in + t_out} frames + 2*{guard_frames} guard frames")

    xs = np.empty((len(windows), t_in) + events[windows[0][0]]["x"].shape[1:],
                  dtype=np.float32)
    ys = np.empty((len(windows), t_out) + events[windows[0][0]]["y"].shape[1:],
                  dtype=np.float32)
    sid = np.empty(len(windows), dtype=np.int32)

    for k, (e, t0) in enumerate(windows):
        ev = events[e]
        xs[k] = ev["x"][t0:t0 + t_in]
        ys[k] = ev["y"][t0 + t_in:t0 + t_in + t_out]
        sid[k] = e
    return xs, ys, sid


def build_split_sequences(events: Sequence[dict],
                          splits: dict[str, Sequence[int]],
                          stride: int = 1,
                          val_stride: int | None = None,
                          guard_frames: int = 0,
                          max_samples: dict[str, int] | None = None,
                          seed: int = 0) -> dict[str, dict]:
    """Convenience: run make_sequences for every split and re-verify disjointness.

    `val_stride` defaults to `stride`; the pipeline usually wants a coarser
    stride on val/test because evaluation cost scales with N and a denser
    evaluation set does not make the estimate meaningfully more precise (the
    samples are correlated, so extra windows add little independent
    information).
    """
    vs = stride if val_stride is None else int(val_stride)
    strides = {"train": stride, "val": vs, "test": vs}
    out: dict[str, dict] = {}
    for name, ids in splits.items():
        cap = None if max_samples is None else max_samples.get(name)
        X, Y, sid = make_sequences(events, ids, stride=strides.get(name, stride),
                                   guard_frames=guard_frames,
                                   max_samples=cap, seed=seed)
        out[name] = {"X": X, "Y": Y, "sid": sid, "event_ids": list(ids)}
    verify_sample_disjoint(out)
    return out


def verify_sample_disjoint(sets: dict[str, dict]) -> dict:
    """Assert no event id contributes samples to more than one split.

    This checks the property on the arrays actually built, not on the intended
    index lists. The two can differ if a caller edits the split lists between
    `split_events` and `make_sequences`, which is exactly the sort of edit that
    happens while debugging and is never noticed.
    """
    owner: dict[int, str] = {}
    bad: list[tuple[int, str, str]] = []
    for name, d in sets.items():
        for e in np.unique(d["sid"]).tolist():
            e = int(e)
            if e in owner and owner[e] != name:
                bad.append((e, owner[e], name))
            owner[e] = name
    assert not bad, (
        "SAMPLES FROM ONE EVENT LANDED IN TWO SPLITS: "
        + ", ".join(f"event {e} in {a} and {b}" for e, a, b in bad))
    return {"n_events_used": len(owner),
            "per_split": {k: int(np.unique(v["sid"]).size)
                          for k, v in sets.items()}}


# ---------------------------------------------------------------------------
# Stratification report
# ---------------------------------------------------------------------------

def split_stats(Y: np.ndarray, sid: np.ndarray | None = None) -> dict:
    """Base rate and lightning-bearing fractions for one split's targets."""
    yb = np.asarray(Y) >= C.LIGHTNING_THRESHOLD
    n = int(yb.shape[0])
    per_sample = yb.reshape(n, -1).any(axis=1)
    out = {
        "n_samples": n,
        "n_events": int(np.unique(sid).size) if sid is not None else -1,
        "n_positive_px": int(yb.sum()),
        "base_rate": float(yb.mean()),
        "frac_samples_with_lightning": float(per_sample.mean()),
    }
    if sid is not None and n:
        # Which events actually supply the positives. If one event supplies
        # nearly all of them the headline metric has an effective sample size
        # of about one, and its confidence interval is enormous however many
        # windows were cut from it.
        ev_pos = {}
        for e in np.unique(sid).tolist():
            m = sid == e
            ev_pos[int(e)] = int(yb[m].sum())
        tot = max(sum(ev_pos.values()), 1)
        out["n_events_with_lightning"] = int(sum(v > 0 for v in ev_pos.values()))
        out["max_event_share_of_positives"] = float(max(ev_pos.values()) / tot)
    return out


def stratify_report(sets: dict[str, dict], verbose: bool = True) -> dict:
    """Print and return per-split base rates, and flag unusable splits.

    THE FAILURE THIS EXISTS TO CATCH. If the test split happens to contain no
    lightning at all, AUPRC is nan by construction (metrics.auprc returns nan
    when there are no positives), CSI is nan, and the comparison table is a
    grid of nans that reads as "something went wrong with the code" rather than
    "the split is degenerate". Worse, a split with very few positives does not
    produce nan -- it produces a plausible-looking number with a confidence
    interval so wide that any ranking of models from it is noise. Both cases
    are reported here, before any training time is spent.
    """
    stats = {name: split_stats(d["Y"], d.get("sid")) for name, d in sets.items()}
    problems: list[str] = []
    for name, s in stats.items():
        if s["n_positive_px"] == 0:
            problems.append(
                f"{name} split has ZERO positive pixels: every rare-event "
                f"metric on it will be nan. Increase n_events or reseed.")
        elif s["n_positive_px"] < 500:
            problems.append(
                f"{name} split has only {s['n_positive_px']} positive pixels: "
                f"metrics will be extremely noisy, do not rank models on it.")
        if s.get("n_events_with_lightning", 1) == 1 and s["n_events"] > 1:
            problems.append(
                f"{name} split: all positives come from a single event, so the "
                f"effective sample size for AUPRC/CSI is 1 event.")

    if verbose:
        print("SPLIT STRATIFICATION")
        print("-" * 78)
        print(f"  {'split':6s} {'events':>7s} {'samples':>8s} {'pos px':>10s} "
              f"{'base rate':>10s} {'frac samp w/ ltg':>17s} {'top-event pos':>14s}")
        for name in ("train", "val", "test"):
            if name not in stats:
                continue
            s = stats[name]
            share = s.get("max_event_share_of_positives", float("nan"))
            print(f"  {name:6s} {s['n_events']:7d} {s['n_samples']:8d} "
                  f"{s['n_positive_px']:10,d} {s['base_rate']:10.5f} "
                  f"{s['frac_samples_with_lightning']:17.3f} {share:14.3f}")
        for name in stats:
            if name in ("train", "val", "test"):
                continue
            s = stats[name]
            print(f"  {name:6s} {s['n_events']:7d} {s['n_samples']:8d} "
                  f"{s['n_positive_px']:10,d} {s['base_rate']:10.5f} "
                  f"{s['frac_samples_with_lightning']:17.3f}")
        if problems:
            print()
            for p in problems:
                print(f"  WARNING: {p}")
        else:
            print("  all splits carry enough positives to be scoreable")
    return {"stats": stats, "problems": problems}


# ---------------------------------------------------------------------------
# Leakage quantification: what the WRONG split would have given us
# ---------------------------------------------------------------------------

def naive_sample_split_leakage(events: Sequence[dict],
                               event_ids: Iterable[int] | None = None,
                               stride: int = 1,
                               test_frac: float = 0.2,
                               seed: int = 0,
                               guard_frames: int = 0) -> dict:
    """Measure the leakage a random SAMPLE-level split would introduce.

    Two windows from the same event at t0 = a and t0 = b occupy frame ranges
    [a, a + L) and [b, b + L) with L = T_in + T_out, which overlap exactly when
    |a - b| < L. So the count of test windows that share at least one frame
    with a training window is computable in closed form -- no need to compare
    arrays.

    This function exists so the docstring claim at the top of the file is a
    measurement. On this data with stride 1 it reports that essentially every
    test sample overlaps a training sample, which is why event blocking is
    mandatory rather than tidy.
    """
    if event_ids is None:
        event_ids = range(len(events))
    windows = enumerate_windows(events, event_ids, stride, guard_frames)
    span = C.INPUT_FRAMES + C.OUTPUT_FRAMES
    n = len(windows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test_w = [windows[i] for i in perm[:n_test]]
    train_w = [windows[i] for i in perm[n_test:]]

    by_event: dict[int, list[int]] = {}
    for e, t0 in train_w:
        by_event.setdefault(e, []).append(t0)
    for e in by_event:
        by_event[e].sort()

    leaked = 0
    min_dists: list[int] = []
    for e, t0 in test_w:
        ts = by_event.get(e)
        if not ts:
            min_dists.append(10 ** 6)
            continue
        arr = np.asarray(ts)
        d = int(np.min(np.abs(arr - t0)))
        min_dists.append(d)
        if d < span:
            leaked += 1

    return {
        "n_windows": n,
        "n_test_windows": n_test,
        "n_test_overlapping_train": int(leaked),
        "frac_test_overlapping_train": float(leaked) / max(n_test, 1),
        "median_min_frame_distance": float(np.median(min_dists)),
        "overlap_span_frames": int(span),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import time

    import synth

    t_start = time.time()
    n_events = 9
    print("=" * 78)
    print("cv.py SELF-TEST -- event-blocked splitting must be provably disjoint")
    print("=" * 78)

    # Short events keep this fast; the split logic does not care about length.
    events = synth.generate_dataset(n_events=n_events, seed=5,
                                    frames_per_event=30)
    print(f"generated {len(events)} events of {events[0]['x'].shape[0]} frames "
          f"in {time.time() - t_start:.1f}s")

    # ---- 1. determinism -------------------------------------------------
    a = split_events(n_events, seed=42)
    b = split_events(n_events, seed=42)
    c = split_events(n_events, seed=43)
    assert a == b, "split_events is not deterministic for a fixed seed"
    print(f"\nsplit (seed 42) : {a}")
    print(f"split (seed 43) : {c}")
    print(f"  deterministic for fixed seed : {a == b}")
    print(f"  changes with seed            : {a != c}")

    # ---- 2. event-level disjointness ------------------------------------
    info = verify_split_disjoint(a, n_events=n_events)
    print(f"  event-disjoint               : {info['disjoint']}  "
          f"covers all {info['n_events']} events: {info['covers_all']}")

    # A deliberately corrupted split must be REJECTED. A check that never
    # fires is indistinguishable from no check at all.
    bad = {"train": a["train"] + a["test"][:1], "val": a["val"],
           "test": a["test"]}
    try:
        verify_split_disjoint(bad, n_events=n_events)
        raise SystemExit("FAIL: verify_split_disjoint accepted an overlap")
    except AssertionError as exc:
        print(f"  rejects a corrupted split    : yes ({str(exc)[:58]}...)")

    # ---- 3. sequences carry event provenance and stay disjoint ----------
    sets = build_split_sequences(events, a, stride=3, val_stride=4)
    for name, d in sets.items():
        print(f"\n{name:5s} X {d['X'].shape}  Y {d['Y'].shape}  "
              f"events {sorted(set(d['sid'].tolist()))}")
        assert d["X"].shape[1] == C.INPUT_FRAMES
        assert d["X"].shape[2] == C.N_CHANNELS
        assert d["Y"].shape[1] == C.OUTPUT_FRAMES
        assert d["sid"].shape[0] == d["X"].shape[0]
        assert set(np.unique(d["sid"]).tolist()) <= set(a[name])

    ids = {name: set(np.unique(d["sid"]).tolist()) for name, d in sets.items()}
    for x in ("train", "val", "test"):
        for y in ("train", "val", "test"):
            if x < y:
                inter = ids[x] & ids[y]
                assert not inter, f"{x}/{y} share events {inter}"
    print(f"\npairwise event intersections : "
          f"train^val {ids['train'] & ids['val']}, "
          f"train^test {ids['train'] & ids['test']}, "
          f"val^test {ids['val'] & ids['test']}  (all empty)")

    # ---- 4. windows never cross an event boundary -----------------------
    # Reconstruct each sample from its event and assert an exact match. This
    # catches an off-by-one in the t0 arithmetic, which would silently mix the
    # tail of one event with the head of the next in a concatenated layout.
    win = enumerate_windows(events, a["test"], stride=4)
    Xt, Yt, st = make_sequences(events, a["test"], stride=4)
    worst = 0.0
    for k, (e, t0) in enumerate(win):
        assert st[k] == e
        worst = max(worst, float(np.abs(
            Xt[k] - events[e]["x"][t0:t0 + C.INPUT_FRAMES]).max()))
        worst = max(worst, float(np.abs(
            Yt[k] - events[e]["y"][t0 + C.INPUT_FRAMES:
                                   t0 + C.INPUT_FRAMES + C.OUTPUT_FRAMES]
        ).max()))
        span_end = t0 + C.INPUT_FRAMES + C.OUTPUT_FRAMES
        assert span_end <= events[e]["x"].shape[0], "window ran off event end"
    print(f"exact reconstruction from (event, t0): max abs diff {worst:.1e} "
          f"over {len(win)} windows")

    # ---- 5. stratification ----------------------------------------------
    print()
    rep = stratify_report(sets)

    # ---- 6. what the wrong split would have cost us ---------------------
    print()
    print("LEAKAGE OF A NAIVE RANDOM SAMPLE-LEVEL SPLIT (the thing we avoid)")
    print("-" * 78)
    for s in (1, 3, 6):
        lk = naive_sample_split_leakage(events, stride=s, seed=1)
        print(f"  stride {s}: {lk['n_test_overlapping_train']:4d} of "
              f"{lk['n_test_windows']:4d} test windows "
              f"({lk['frac_test_overlapping_train']:6.1%}) share >=1 frame "
              f"with a train window; median nearest train window is "
              f"{lk['median_min_frame_distance']:.0f} frames away "
              f"(overlap if < {lk['overlap_span_frames']})")
    print("  Event-blocked splitting makes all of these exactly 0 by "
          "construction.")

    # And prove that claim on the real splits: no (event, t0) pair can be
    # shared, because no event is shared.
    tr_w = set(enumerate_windows(events, a["train"], stride=3))
    te_w = set(enumerate_windows(events, a["test"], stride=4))
    shared_events = {e for e, _ in tr_w} & {e for e, _ in te_w}
    print(f"  event-blocked: shared windows {len(tr_w & te_w)}, "
          f"shared events {len(shared_events)}")
    assert not (tr_w & te_w) and not shared_events

    ok = not rep["problems"]
    print()
    print(f"ALL SPLIT ASSERTIONS PASSED in {time.time() - t_start:.1f}s"
          + ("" if ok else "  (with stratification warnings above)"))


if __name__ == "__main__":
    _self_test()

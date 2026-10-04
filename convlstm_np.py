"""
ConvLSTM encoder-forecaster in pure NumPy, with a hand-written backward pass.

WHY THIS BACKEND EXISTS
-----------------------
No torch, no scipy, no sklearn, no network. Everything here -- convolution,
the LSTM recurrence, backpropagation through time, Adam -- is written against
numpy primitives only. That constraint is not cosmetic: it forces the design
towards a small number of large matrix multiplies (which BLAS makes fast) and
away from anything that loops over pixels in Python (which would be ~1000x
slower and make training impossible on 2 cores).

WHY A ConvLSTM RATHER THAN A PER-PIXEL CLASSIFIER
-------------------------------------------------
Lightning initiation is a *neighbourhood* process. As measured in features.py,
the strongest predictor of a new strike is not the charging-layer reflectivity
AT the pixel but the neighbourhood MAXIMUM of the graupel proxy: new strikes
appear beside where charging-layer ice already is, in the developing flank of
the same complex. A per-pixel model cannot represent "graupel 8 km upstream is
advecting towards me". A convolutional recurrence can, because the state at
(i, j) is updated from a 3x3 window of the previous state, so information
travels one pixel per frame in every direction: over the 6 encoder frames each
output pixel sees a 13x13 px (26 km) window of the inputs.

AND WHERE THAT ARGUMENT RUNS OUT
--------------------------------
One 3x3 layer propagates information at exactly 1 px/frame, while the steering
flow in config.SYNTH is ~2.5 px/frame -- about 15 px across the input window,
well beyond the 6 px reach of this recurrence. So a single-layer 3x3 ConvLSTM
physically cannot track the fastest cells across the whole 30 min history on
its own. Two things compensate, and both are deliberate: the input features
already contain neighbourhood maxima at radius 2 and 4 px (features.py), which
hand the model a pre-advected view of where ice is; and the encoder-forecaster
split means the forecaster only has to evolve an already-summarised state. The
honest expectation is still that this backend is weakest on fast-moving cells,
and that a larger kernel, dilation, or a downsampled second layer -- all cheap
in torch, all expensive here -- is where the next gain would come from.

DESIGN CHOICES, AND THE SPECIFIC FAILURE MODE EACH ONE PREVENTS
---------------------------------------------------------------
1. ONE convolution emitting 4*hidden channels, then split into gates.
   Four separate convolutions over the same input would re-run im2col four
   times and issue four small matmuls. Measured here (N=4, 36 in-channels,
   hidden=16, 32x32, 3x3): fused 2.1 ms, four separate convs 5.7 ms, and
   3.5 ms even if they share one im2col. So the fusion is worth ~2.8x, and
   speed is the binding constraint on whether this model can be trained at all.

2. Convolution via im2col + matmul.
   The naive alternative (Python loops over kernel offsets AND output pixels)
   is not merely slow, it is unusable: a single epoch would take hours. im2col
   turns convolution into GEMM, which numpy dispatches to BLAS.

3. Forget-gate bias initialised to 1.0.
   With b_f = 0 the forget gate starts at sigmoid(0) = 0.5, so the cell state
   is halved every step: after the 6 encoder frames only ~1.5% of frame-1
   information survives, and the gradient reaching frame 1 is attenuated by the
   same factor. The model then cannot learn motion across the input window and
   degenerates into a single-frame classifier. sigmoid(1) = 0.73 retains ~15%
   over 6 steps, which is enough to learn from. This is standard practice
   (Jozefowicz et al. 2015) for exactly this reason.

4. Weighted BCE computed FROM LOGITS, never sigmoid-then-log.
   sigmoid(-40) rounds to 0 in float32, so log(sigmoid(z)) returns -inf and one
   confident-and-wrong pixel poisons every weight in the network through the
   shared convolution. The stable identity
       BCE(z, y) = max(z, 0) - z*y + log1p(exp(-|z|))
   has no overflow and no log(0) for any finite z.
   The pos_weight factor is not optional: at a ~1% base rate the minimiser of
   unweighted BCE is "predict the base rate everywhere", which scores an
   excellent loss and a CSI of exactly zero.

5. Encoder-forecaster, with the encoder summary re-injected at EVERY decoder
   step. The forecaster is a second ConvLSTM initialised from the encoder's
   final (h, c). Its per-step input is the encoder's final hidden state, held
   constant across lead times. The alternative (feed zeros, the plain Shi et
   al. decoder) forces the entire 60 min forecast through the cell state alone,
   and with only 8-16 hidden channels that state is a narrow bottleneck; giving
   the decoder a direct look at the encoder summary at every step costs one
   extra input block in a convolution we are doing anyway. Lead-time
   differentiation comes from the forecaster's own recurrence, which is why one
   shared 1x1 head can serve all 12 output frames.

6. Gradient clipping by GLOBAL norm.
   Recurrences produce occasional very large gradients (one storm cell with a
   saturated gate, backpropagated through 18 steps). Clipping per-tensor would
   change the relative scaling between the encoder and the head; clipping the
   global norm rescales the whole gradient vector and so preserves its
   direction, which is the point of clipping.

7. Optional strided DOWNSAMPLING of inputs and targets during training.
   Cost is ~O(H*W), so training on 32x32 is 4x cheaper. Targets are subsampled
   by the SAME striding, deliberately NOT max-pooled: "any strike in the 4 km
   cell" is a different and higher-probability event than "a strike at this
   2 km pixel", so max-pooling the target would inflate the learned
   probabilities by up to d^2 and destroy calibration the moment predict()
   upsamples back to the label grid by pixel repetition.

8. gradient_check() against central finite differences.
   This is the most important function in the file. A hand-written BPTT with a
   sign error, a swapped gate, or a missing accumulation still trains, still
   produces a decreasing loss, and still yields plausible-looking maps -- it
   just converges to something worse than it should, silently. Finite
   differences are the only cheap way to know the analytic gradient is right.

MEMORY
------
The per-step im2col matrix is the largest array in the forward pass
(in_ch*k*k times the size of one frame). It is NOT cached: with 18 steps that
would dominate the 3 GB budget. It is recomputed in the backward pass, which
costs a handful of slice copies against two matmuls we have to do anyway.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import config as C

# Adam constants. Left as module-level rather than hyperparameters because
# there is no evidence in this project that tuning them matters, and every
# extra knob is another thing to overfit on a 60-event synthetic dataset.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Gate order inside the fused 4*hidden output channel block. Fixed once here;
# forward and backward must agree, and the bias part of gradient_check() probes
# one index inside EVERY slab precisely so a mismatch cannot slip through.
GATE_ORDER = ("input", "forget", "output", "candidate")


# ---------------------------------------------------------------------------
# Numerically safe elementwise helpers
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function.

    exp(+large) overflows to inf and then inf/inf = nan, which would propagate
    into every gate. Evaluating exp only at -|z| keeps the argument <= 0, where
    exp is bounded by 1, and the two algebraically identical branches are
    selected by sign.
    """
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def weighted_bce_logits(z: np.ndarray, y: np.ndarray,
                        pos_weight: float) -> float:
    """Mean positive-weighted binary cross-entropy computed from logits.

    Uses the stable identity
        -log sigmoid(z)      = max(z, 0) - z + log1p(exp(-|z|))
        -log (1 - sigmoid(z)) = max(z, 0)     + log1p(exp(-|z|))
    which combine, for y in {0, 1}, into
        wt * (max(z, 0) - z*y + log1p(exp(-|z|)))
    with wt = pos_weight for positives and 1 for negatives.

    Never form sigmoid(z) and then take its log: for |z| >~ 40 that is exactly
    0 or 1 in float32 and the loss becomes inf.
    """
    wt = 1.0 + (pos_weight - 1.0) * y
    core = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(wt * core, dtype=np.float64))


def weighted_bce_logits_grad(z: np.ndarray, y: np.ndarray,
                             pos_weight: float) -> np.ndarray:
    """d/dz of weighted_bce_logits. Shape of z.

    The derivative of the weighted loss collapses to wt * (sigmoid(z) - y),
    which is bounded in [-pos_weight, pos_weight] for any finite logit. That
    boundedness is why logit-space BCE is safe where sigmoid-then-log is not.
    """
    wt = 1.0 + (pos_weight - 1.0) * y
    return (wt * (sigmoid(z) - y) / z.size).astype(z.dtype, copy=False)


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------
# Layout: cols has shape (N, C*kh*kw, H*W) so that a weight tensor reshaped to
# (M, C*kh*kw) can be applied with a single broadcast matmul:
#     np.matmul(W2d, cols) -> (N, M, H*W)
# No transpose of the output is needed, which matters because a transpose of a
# (N, M, H*W) array is a full copy of the largest intermediate in the model.
#
# Both directions are implemented as a loop over the kh*kw kernel offsets
# (9 iterations for a 3x3 kernel) rather than with np.lib.stride_tricks. Strided
# tricks are marginally faster to build the forward matrix but col2im then needs
# scatter-add into overlapping windows, and np.add.at is famously slow. The
# offset loop makes the backward direction plain slice-accumulation, and each
# offset writes a disjoint set of column slots, so there is no aliasing.


def im2col(x: np.ndarray, kh: int, kw: int,
           pad_h: int, pad_w: int) -> np.ndarray:
    """(N, C, H, W) -> (N, C*kh*kw, Ho*Wo) patch matrix. Zero padding."""
    n, c, h, w = x.shape
    if pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    ho = h + 2 * pad_h - kh + 1
    wo = w + 2 * pad_w - kw + 1
    k = kh * kw
    cols = np.empty((n, c * k, ho * wo), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            # Slot index for channel ch and offset (i, j) is ch*k + i*kw + j,
            # so a strided slice with step k fills all C channels at once and
            # in the order that W.reshape(M, C*kh*kw) expects.
            cols[:, i * kw + j::k, :] = x[:, :, i:i + ho, j:j + wo].reshape(
                n, c, ho * wo)
    return cols


def col2im(cols: np.ndarray, x_shape: tuple[int, int, int, int],
           kh: int, kw: int, pad_h: int, pad_w: int) -> np.ndarray:
    """Adjoint of im2col: scatter-accumulate patch gradients back to (N,C,H,W).

    Overlapping windows must ADD, not overwrite -- a pixel participates in up
    to kh*kw output positions and receives gradient from all of them. Getting
    this wrong scales the input gradient by 1/9 for interior pixels while
    leaving the edges alone, which the gradient check catches immediately.
    """
    n, c, h, w = x_shape
    ho = h + 2 * pad_h - kh + 1
    wo = w + 2 * pad_w - kw + 1
    k = kh * kw
    xp = np.zeros((n, c, h + 2 * pad_h, w + 2 * pad_w), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + ho, j:j + wo] += cols[:, i * kw + j::k, :].reshape(
                n, c, ho, wo)
    if pad_h or pad_w:
        return xp[:, :, pad_h:pad_h + h, pad_w:pad_w + w]
    return xp


def conv2d_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray,
                   pad_h: int, pad_w: int) -> np.ndarray:
    """Cross-correlation with 'same'-style padding. x (N,C,H,W), W (M,C,kh,kw)."""
    m, c, kh, kw = W.shape
    assert x.shape[1] == c, (x.shape, W.shape)
    n, _, h, w = x.shape
    ho = h + 2 * pad_h - kh + 1
    wo = w + 2 * pad_w - kw + 1
    cols = im2col(x, kh, kw, pad_h, pad_w)
    out = np.matmul(W.reshape(m, c * kh * kw), cols)      # (N, M, Ho*Wo)
    out += b.reshape(1, m, 1)
    return out.reshape(n, m, ho, wo)


def conv2d_backward(dout: np.ndarray, x: np.ndarray, W: np.ndarray,
                    pad_h: int, pad_w: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients (dx, dW, db) for conv2d_forward. Recomputes im2col(x)."""
    m, c, kh, kw = W.shape
    n = dout.shape[0]
    p = dout.shape[2] * dout.shape[3]
    dout2 = dout.reshape(n, m, p)
    cols = im2col(x, kh, kw, pad_h, pad_w)               # (N, C*kh*kw, P)
    # dW[m, k] = sum over samples and output positions of dout * col.
    dW = np.tensordot(dout2, cols, axes=([0, 2], [0, 2])).reshape(W.shape)
    db = dout2.sum(axis=(0, 2))
    dcols = np.matmul(W.reshape(m, c * kh * kw).T, dout2)  # (N, C*kh*kw, P)
    dx = col2im(dcols, x.shape, kh, kw, pad_h, pad_w)
    return dx, dW.astype(W.dtype, copy=False), db.astype(W.dtype, copy=False)


# ---------------------------------------------------------------------------
# ConvLSTM cell
# ---------------------------------------------------------------------------

class _CellCache:
    """Activations one cell step needs in the backward pass.

    Stores the (cheap) gate activations and inputs, not the (expensive) im2col
    matrix -- see the module docstring on memory. `c` itself is not stored
    either, only tanh(c), because that is the only form the backward pass uses.
    """

    __slots__ = ("x", "h_prev", "c_prev", "i", "f", "o", "g", "tanh_c", "h_out")

    def __init__(self, x, h_prev, c_prev, i, f, o, g, tanh_c, h_out):
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.i = i
        self.f = f
        self.o = o
        self.g = g
        self.tanh_c = tanh_c
        self.h_out = h_out


class ConvLSTMCell:
    """One ConvLSTM step, gates from a SINGLE convolution over [x, h].

        z          = conv([x_t, h_{t-1}]) + b          (4*hidden channels)
        i, f, o    = sigmoid(z_i), sigmoid(z_f), sigmoid(z_o)
        g          = tanh(z_g)
        c_t        = f * c_{t-1} + i * g
        h_t        = o * tanh(c_t)

    The convolution is over the CONCATENATION of input and hidden state, which
    is what makes this a ConvLSTM rather than a fully-connected LSTM applied
    per pixel: gate values at (i, j) depend on a 3x3 neighbourhood of both the
    observations and the previous state, so the cell can express advection.
    """

    def __init__(self, in_ch: int, hidden: int, kernel: int = 3,
                 dtype: Any = np.float32,
                 rng: np.random.Generator | None = None,
                 name: str = "cell") -> None:
        assert kernel % 2 == 1, "even kernels cannot be 'same'-padded symmetrically"
        rng = rng or np.random.default_rng(0)
        self.in_ch = int(in_ch)
        self.hidden = int(hidden)
        self.kernel = int(kernel)
        self.pad = self.kernel // 2
        self.dtype = np.dtype(dtype)
        self.name = name

        fan_in = (in_ch + hidden) * kernel * kernel
        # Xavier-style scaling on the true fan-in of the fused convolution.
        # Too large and the gates saturate at step 1 (sigmoid' -> 0, nothing
        # learns); too small and every gate sits at exactly 0.5 with tanh
        # near-linear, which is a symmetric point where the four gate slabs
        # receive nearly identical gradients and take many steps to
        # differentiate.
        scale = 1.0 / np.sqrt(fan_in)
        self.W = (rng.standard_normal(
            (4 * hidden, in_ch + hidden, kernel, kernel)) * scale
        ).astype(self.dtype)
        self.b = np.zeros(4 * hidden, dtype=self.dtype)
        # Forget slab to 1.0: see module docstring item 3.
        self.b[hidden:2 * hidden] = 1.0

    # -- slabs -------------------------------------------------------------
    def _split(self, z: np.ndarray) -> tuple[np.ndarray, ...]:
        h = self.hidden
        return z[:, :h], z[:, h:2 * h], z[:, 2 * h:3 * h], z[:, 3 * h:]

    # -- forward -----------------------------------------------------------
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                cache: bool = True
                ) -> tuple[np.ndarray, np.ndarray, _CellCache | None]:
        cat = np.concatenate((x, h_prev), axis=1)
        z = conv2d_forward(cat, self.W, self.b, self.pad, self.pad)
        zi, zf, zo, zg = self._split(z)
        i = sigmoid(zi)
        f = sigmoid(zf)
        o = sigmoid(zo)
        g = np.tanh(zg)
        c = f * c_prev + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c
        cc = _CellCache(x, h_prev, c_prev, i, f, o, g, tanh_c, h) if cache else None
        return h, c, cc

    # -- backward ----------------------------------------------------------
    def backward(self, cc: _CellCache, dh: np.ndarray, dc_next: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray]:
        """Returns (dx, dh_prev, dc_prev, dW, db).

        `dc_next` is the gradient arriving along the CELL path from step t+1.
        Forgetting to add it (using only dh) is the classic BPTT bug: the loss
        still decreases, but long-range memory never trains, so the model
        silently becomes a 1-2 frame nowcaster.
        """
        tanh_c = cc.tanh_c
        # h = o * tanh(c)
        do = dh * tanh_c
        dc = dc_next + dh * cc.o * (1.0 - tanh_c * tanh_c)
        # c = f * c_prev + i * g
        di = dc * cc.g
        dg = dc * cc.i
        df = dc * cc.c_prev
        dc_prev = dc * cc.f
        # through the nonlinearities
        dzi = di * cc.i * (1.0 - cc.i)
        dzf = df * cc.f * (1.0 - cc.f)
        dzo = do * cc.o * (1.0 - cc.o)
        dzg = dg * (1.0 - cc.g * cc.g)
        dz = np.concatenate((dzi, dzf, dzo, dzg), axis=1)

        cat = np.concatenate((cc.x, cc.h_prev), axis=1)
        dcat, dW, db = conv2d_backward(dz, cat, self.W, self.pad, self.pad)
        cin = cc.x.shape[1]
        return dcat[:, :cin], dcat[:, cin:], dc_prev, dW, db


# ---------------------------------------------------------------------------
# Encoder-forecaster model
# ---------------------------------------------------------------------------

class ConvLSTMNowcaster:
    """ConvLSTM encoder-forecaster producing per-pixel lightning logits.

        encoder     : consumes T_in feature frames -> (h_enc, c_enc)
        forecaster  : initialised from (h_enc, c_enc), run T_out steps with
                      h_enc as a constant context input
        head        : 1x1 conv, hidden -> 1, shared across all lead times

    The head emits a LOGIT, not a probability: the loss is defined on logits
    for numerical stability (module docstring item 4), and predict() applies
    the sigmoid only at the very end.
    """

    PARAM_NAMES = ("W_enc", "b_enc", "W_fc", "b_fc", "W_head", "b_head")

    def __init__(self, in_ch: int, t_out: int = C.OUTPUT_FRAMES,
                 hidden: int | None = None, kernel: int | None = None,
                 lr: float | None = None, pos_weight: float | None = None,
                 grad_clip: float | None = None, downsample: int | None = None,
                 batch: int | None = None, seed: int = 0,
                 dtype: Any = np.float32) -> None:
        hp = C.HP_NUMPY
        self.in_ch = int(in_ch)
        self.t_out = int(t_out)
        self.hidden = int(hp["hidden"] if hidden is None else hidden)
        self.kernel = int(hp["kernel"] if kernel is None else kernel)
        self.lr = float(hp["lr"] if lr is None else lr)
        self.pos_weight = float(hp["pos_weight"] if pos_weight is None
                                else pos_weight)
        self.grad_clip = float(hp["grad_clip"] if grad_clip is None
                               else grad_clip)
        self.downsample = int(hp["downsample"] if downsample is None
                              else downsample)
        self.batch = int(hp["batch"] if batch is None else batch)
        self.seed = int(seed)
        self.dtype = np.dtype(dtype)

        rng = np.random.default_rng(seed)
        self.enc = ConvLSTMCell(self.in_ch, self.hidden, self.kernel,
                                self.dtype, rng, name="enc")
        # The forecaster's input is the encoder summary h_enc (hidden channels),
        # hence in_ch = hidden. See module docstring item 5.
        self.fc = ConvLSTMCell(self.hidden, self.hidden, self.kernel,
                               self.dtype, rng, name="fc")
        # 1x1 head. A larger head would let the output depend on neighbouring
        # hidden states, but the recurrence already mixes space; keeping the
        # head pointwise makes the hidden state itself the spatial forecast,
        # which is easier to inspect when debugging.
        self.W_head = (rng.standard_normal((1, self.hidden, 1, 1))
                       / np.sqrt(self.hidden)).astype(self.dtype)
        # Bias initialised to the logit of the climatological base rate so the
        # first forward pass already predicts roughly the right frequency.
        # Starting at 0 means predicting p=0.5 everywhere against a 1% base
        # rate: the first few hundred updates are spent purely on the bias, and
        # with a weighted loss those updates are large enough to wreck the
        # freshly initialised convolution kernels first.
        base_rate = 0.01
        self.b_head = np.array([np.log(base_rate / (1.0 - base_rate))],
                               dtype=self.dtype)

        # Adam state, kept alongside the parameters so a checkpoint can restore
        # an EXACT continuation rather than restarting the moment estimates
        # (which produces a visible loss spike on every resume).
        self.opt_m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.opt_v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.opt_t = 0
        self.epochs_done = 0
        # Loss curves plus run bookkeeping. Kept on the instance (and in the
        # checkpoint) so a run split across several calls reports one continuous
        # curve instead of a series of disconnected fragments.
        self.history: dict[str, Any] = {"train_loss": [], "val_loss": []}

    # -- parameter plumbing ------------------------------------------------
    def _params(self) -> dict[str, np.ndarray]:
        """Live references to every trainable array.

        References, not copies: Adam mutates them in place, so nothing needs to
        be written back and there is exactly one source of truth.
        """
        return {
            "W_enc": self.enc.W, "b_enc": self.enc.b,
            "W_fc": self.fc.W, "b_fc": self.fc.b,
            "W_head": self.W_head, "b_head": self.b_head,
        }

    def n_params(self) -> int:
        return int(sum(p.size for p in self._params().values()))

    # -- forward -----------------------------------------------------------
    def _forward(self, fx: np.ndarray, cache: bool = True
                 ) -> tuple[np.ndarray, tuple[list, list] | None]:
        """fx: (N, T_in, in_ch, H, W) -> logits (N, T_out, H, W)."""
        n, t_in, cin, h, w = fx.shape
        assert cin == self.in_ch, f"expected {self.in_ch} channels, got {cin}"
        state = np.zeros((n, self.hidden, h, w), dtype=self.dtype)
        hs, cs = state, state.copy()

        enc_caches: list = []
        for t in range(t_in):
            hs, cs, cc = self.enc.forward(
                np.ascontiguousarray(fx[:, t], dtype=self.dtype), hs, cs, cache)
            if cache:
                enc_caches.append(cc)

        ctx = hs                       # encoder summary, reused every lead time
        hf, cf = hs, cs
        fc_caches: list = []
        logits = np.empty((n, self.t_out, h, w), dtype=self.dtype)
        for t in range(self.t_out):
            hf, cf, cc = self.fc.forward(ctx, hf, cf, cache)
            if cache:
                fc_caches.append(cc)
            z = conv2d_forward(hf, self.W_head, self.b_head, 0, 0)
            logits[:, t] = z[:, 0]
        return logits, ((enc_caches, fc_caches) if cache else None)

    # -- backward ----------------------------------------------------------
    def _backward(self, dlogits: np.ndarray,
                  caches: tuple[list, list]) -> dict[str, np.ndarray]:
        """Backpropagation through time over forecaster then encoder."""
        enc_caches, fc_caches = caches
        grads = {k: np.zeros_like(v) for k, v in self._params().items()}

        shape = fc_caches[0].h_prev.shape
        dh_next = np.zeros(shape, dtype=self.dtype)
        dc_next = np.zeros(shape, dtype=self.dtype)
        # h_enc feeds the forecaster in TWO ways: as its initial hidden state
        # and as the context input at every step. Both paths must be summed.
        # Dropping the second was a real bug during development: the loss still
        # fell, and only the gradient check located it.
        dctx = np.zeros(shape, dtype=self.dtype)

        for t in reversed(range(self.t_out)):
            cc = fc_caches[t]
            dz = np.ascontiguousarray(dlogits[:, t])[:, None]
            dh_head, dwh, dbh = conv2d_backward(dz, cc.h_out, self.W_head, 0, 0)
            grads["W_head"] += dwh
            grads["b_head"] += dbh
            dx, dh_prev, dc_prev, dW, db = self.fc.backward(
                cc, dh_head + dh_next, dc_next)
            grads["W_fc"] += dW
            grads["b_fc"] += db
            dctx += dx
            dh_next, dc_next = dh_prev, dc_prev

        # Gradient wrt the encoder's final state: recurrent path + context path.
        dh, dc = dh_next + dctx, dc_next
        for t in reversed(range(len(enc_caches))):
            _dx, dh, dc, dW, db = self.enc.backward(enc_caches[t], dh, dc)
            grads["W_enc"] += dW
            grads["b_enc"] += db
        # dx at the encoder inputs is discarded: the features are fixed data.
        return grads

    def _loss_and_grad(self, fx: np.ndarray, y: np.ndarray
                       ) -> tuple[float, dict[str, np.ndarray]]:
        logits, caches = self._forward(fx, cache=True)
        loss = weighted_bce_logits(logits, y, self.pos_weight)
        dlogits = weighted_bce_logits_grad(logits, y, self.pos_weight)
        return loss, self._backward(dlogits, caches)

    # -- optimiser ---------------------------------------------------------
    def _adam_step(self, grads: dict[str, np.ndarray]) -> float:
        """Clip by global norm, then one Adam update. Returns the pre-clip norm."""
        sq = 0.0
        for g in grads.values():
            sq += float(np.sum(g * g, dtype=np.float64))
        gnorm = float(np.sqrt(sq))
        scale = 1.0
        if self.grad_clip > 0 and gnorm > self.grad_clip:
            # Rescale the whole gradient vector so its DIRECTION is unchanged.
            scale = self.grad_clip / (gnorm + 1e-12)

        self.opt_t += 1
        bc1 = 1.0 - ADAM_BETA1 ** self.opt_t
        bc2 = 1.0 - ADAM_BETA2 ** self.opt_t
        for name, p in self._params().items():
            g = grads[name] if scale == 1.0 else grads[name] * scale
            m, v = self.opt_m[name], self.opt_v[name]
            m *= ADAM_BETA1
            m += (1.0 - ADAM_BETA1) * g
            v *= ADAM_BETA2
            v += (1.0 - ADAM_BETA2) * (g * g)
            step = self.lr * (m / bc1) / (np.sqrt(v / bc2) + ADAM_EPS)
            p -= step.astype(p.dtype, copy=False)
        return gnorm

    # -- grid handling -----------------------------------------------------
    def _down(self, a: np.ndarray, as_dtype: Any | None = None) -> np.ndarray:
        d = self.downsample
        out = a if d <= 1 else a[..., ::d, ::d]
        return np.ascontiguousarray(out, dtype=as_dtype or self.dtype)

    def _prepare(self, x: np.ndarray, y: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Feature/target pair on the training grid, target binarised."""
        yb = y if y.dtype == bool else (y >= C.LIGHTNING_THRESHOLD)
        assert x.shape[0] == y.shape[0], (x.shape, y.shape)
        assert y.shape[1] == self.t_out, (y.shape, self.t_out)
        return self._down(x), self._down(yb.astype(self.dtype))

    # -- training ----------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray,
            X_val: np.ndarray | None = None, Y_val: np.ndarray | None = None,
            epochs: int | None = None, batch: int | None = None,
            max_seconds: float | None = None, verbose: bool = True,
            checkpoint: str | None = None) -> dict:
        """Train with Adam.

        Args:
            X: (N, T_in, in_ch, H, W) features from features.build_features_batch
            Y: (N, T_out, H, W) strike counts or binary labels
            X_val, Y_val: optional held-out set, scored once per epoch
            epochs: default from config
            batch: default from config
            max_seconds: wall-clock budget. Checked BETWEEN minibatches so a
                sandbox that kills long calls cannot lose the run: training
                stops cleanly, the checkpoint is written, and the next call
                resumes from it.
            checkpoint: path written after every completed epoch and on
                early stop.

        Returns the history dict (also kept on the instance, so a resumed run
        appends rather than restarting the curve).
        """
        epochs = int(C.HP_NUMPY["epochs"] if epochs is None else epochs)
        batch = int(self.batch if batch is None else batch)
        xd, yd = self._prepare(X, Y)
        has_val = X_val is not None and Y_val is not None
        if has_val:
            xv, yv = self._prepare(X_val, Y_val)

        n = xd.shape[0]
        t_start = time.time()
        stopped = False
        if verbose:
            print(f"  ConvLSTM(hidden={self.hidden}, k={self.kernel}, "
                  f"in={self.in_ch}, T_out={self.t_out}) "
                  f"{self.n_params():,} params, dtype={self.dtype.name}")
            print(f"  train {n} seq on {xd.shape[-2]}x{xd.shape[-1]} grid "
                  f"(downsample={self.downsample}), batch={batch}, "
                  f"lr={self.lr}, pos_weight={self.pos_weight}, "
                  f"clip={self.grad_clip}")

        for _ in range(epochs):
            if max_seconds is not None and time.time() - t_start >= max_seconds:
                stopped = True
                break
            # Permutation seeded by (seed, epochs_done) so that a resumed run
            # sees the same sample order it would have seen without the
            # interruption. Reproducibility across a checkpoint boundary is
            # what makes "resume" mean something.
            rng = np.random.default_rng(self.seed * 100003 + self.epochs_done)
            perm = rng.permutation(n)
            ep_t0 = time.time()
            tot, seen, gn = 0.0, 0, 0.0
            partial = False
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                loss, grads = self._loss_and_grad(xd[idx], yd[idx])
                gn = self._adam_step(grads)
                tot += loss * idx.size
                seen += idx.size
                if (max_seconds is not None
                        and time.time() - t_start >= max_seconds):
                    partial = True
                    break
            train_loss = tot / max(seen, 1)
            self.history["train_loss"].append(train_loss)
            msg = (f"  epoch {self.epochs_done + 1:2d}  "
                   f"train {train_loss:.5f}")
            if has_val:
                vl = self.eval_loss(xv, yv, prepared=True)
                self.history["val_loss"].append(vl)
                msg += f"  val {vl:.5f}"
            msg += f"  |g| {gn:.3f}  {time.time() - ep_t0:.1f}s"
            if partial:
                # Do NOT count a truncated epoch as done: on resume the same
                # permutation is regenerated and the remaining minibatches are
                # visited, so no sample is silently skipped.
                msg += "  [budget reached mid-epoch]"
                stopped = True
            else:
                self.epochs_done += 1
            if verbose:
                print(msg, flush=True)
            if checkpoint:
                self.save(checkpoint)
            if stopped:
                break

        elapsed = time.time() - t_start
        if verbose and stopped:
            print(f"  stopped after {elapsed:.1f}s "
                  f"(budget {max_seconds}s), {self.epochs_done} full epochs done"
                  + (f", checkpoint -> {checkpoint}" if checkpoint else ""))
        self.history["epochs_done"] = self.epochs_done
        self.history["stopped_early"] = stopped
        self.history["seconds"] = elapsed
        return self.history

    # -- inference ---------------------------------------------------------
    def eval_loss(self, X: np.ndarray, Y: np.ndarray, batch: int = 8,
                  prepared: bool = False) -> float:
        """Weighted BCE on the training grid, no caches kept."""
        if prepared:
            xd, yd = X, Y
        else:
            xd, yd = self._prepare(X, Y)
        tot, seen = 0.0, 0
        for s in range(0, xd.shape[0], batch):
            xb, yb = xd[s:s + batch], yd[s:s + batch]
            logits, _ = self._forward(xb, cache=False)
            tot += weighted_bce_logits(logits, yb, self.pos_weight) * xb.shape[0]
            seen += xb.shape[0]
        return tot / max(seen, 1)

    def predict(self, X: np.ndarray, batch: int = 8) -> np.ndarray:
        """Probabilities in [0, 1], shape (N, T_out, H, W) on the FULL grid.

        If training used downsample > 1, the forward pass runs on the coarse
        grid and the result is upsampled by pixel repetition so the output
        always matches the label grid. Nearest-neighbour repetition (rather
        than bilinear) is deliberate: a probability field should not be smoothed
        on its way out, because smoothing lowers the peaks and would make every
        threshold-based contingency score look worse than the model actually is.
        """
        n, _, cin, h, w = X.shape
        d = self.downsample
        out = np.empty((n, self.t_out, h, w), dtype=np.float64)
        for s in range(0, n, batch):
            xb = self._down(X[s:s + batch])
            logits, _ = self._forward(xb, cache=False)
            p = sigmoid(logits.astype(np.float64))
            if d > 1:
                p = np.repeat(np.repeat(p, d, axis=-2), d, axis=-1)
                # Crop rather than pad: strided subsampling of an odd-sized
                # grid rounds up, so the repeated field can be larger than the
                # label grid but never smaller.
                p = p[..., :h, :w]
            out[s:s + xb.shape[0]] = p
        return out

    # -- checkpointing -----------------------------------------------------
    def save(self, path: str) -> str:
        """np.savez checkpoint: parameters + Adam state + schedule position.

        Optimiser state is included on purpose. Reloading only the weights and
        restarting Adam from zero moments produces a large first step (the
        bias-correction divides by 1 - beta at t = 1) and a visible loss spike,
        so a "resumed" run would not be the same run.
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        blob: dict[str, Any] = {}
        for k, v in self._params().items():
            blob["p_" + k] = v
            blob["m_" + k] = self.opt_m[k]
            blob["v_" + k] = self.opt_v[k]
        blob.update(
            in_ch=np.array(self.in_ch), t_out=np.array(self.t_out),
            hidden=np.array(self.hidden), kernel=np.array(self.kernel),
            lr=np.array(self.lr), pos_weight=np.array(self.pos_weight),
            grad_clip=np.array(self.grad_clip),
            downsample=np.array(self.downsample), batch=np.array(self.batch),
            seed=np.array(self.seed), opt_t=np.array(self.opt_t),
            epochs_done=np.array(self.epochs_done),
            dtype=np.array(self.dtype.name),
            train_loss=np.asarray(self.history["train_loss"], dtype=np.float64),
            val_loss=np.asarray(self.history["val_loss"], dtype=np.float64),
        )
        np.savez(path, **blob)
        return path

    @classmethod
    def load(cls, path: str) -> "ConvLSTMNowcaster":
        """Reconstruct a model saved by save(). Exact continuation of training."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        z = np.load(path)
        m = cls(in_ch=int(z["in_ch"]), t_out=int(z["t_out"]),
                hidden=int(z["hidden"]), kernel=int(z["kernel"]),
                lr=float(z["lr"]), pos_weight=float(z["pos_weight"]),
                grad_clip=float(z["grad_clip"]),
                downsample=int(z["downsample"]), batch=int(z["batch"]),
                seed=int(z["seed"]), dtype=np.dtype(str(z["dtype"].item())))
        for k, p in m._params().items():
            # In-place copy keeps the references handed out by _params() valid.
            p[...] = z["p_" + k]
            m.opt_m[k][...] = z["m_" + k]
            m.opt_v[k][...] = z["v_" + k]
        m.opt_t = int(z["opt_t"])
        m.epochs_done = int(z["epochs_done"])
        m.history = {"train_loss": [float(v) for v in z["train_loss"]],
                     "val_loss": [float(v) for v in z["val_loss"]]}
        return m


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

def _probe_indices(name: str, param: np.ndarray, hidden: int,
                   n_samples: int, rng: np.random.Generator) -> list[int]:
    """Flat indices to finite-difference for one tensor.

    For the gate biases the indices are forced to cover ALL FOUR slabs
    (input / forget / output / candidate). A purely random sample of a
    4*hidden vector can miss a slab, and the bugs that matter most here are
    slab-specific: a swapped gate order, or a sign error in one gate's
    derivative. Those are invisible in the loss curve and fatal to skill.
    """
    if name.startswith("b_") and param.size == 4 * hidden:
        idx = [g * hidden for g in range(4)]                  # one per gate
        idx += [g * hidden + hidden - 1 for g in range(4)]     # and the far end
        return sorted(set(i for i in idx if i < param.size))
    if param.size <= n_samples:
        return list(range(param.size))
    return sorted(rng.choice(param.size, size=n_samples, replace=False).tolist())


def gradient_check(seed: int = 3, n_samples: int = 5, eps: float = 1e-5,
                   tol: float = 1e-4, verbose: bool = True) -> dict[str, float]:
    """Verify analytic gradients against central finite differences.

    Tiny by design: hidden=2, 1 input channel, 6x6 grid, T_in=2, T_out=2,
    batch=1. Small enough to be exhaustive in milliseconds, large enough that
    every code path runs -- 'same' padding with overlapping windows (so col2im
    accumulation matters), two recurrent steps in each of the two cells (so the
    dc_next path matters), and the shared head across lead times (so gradient
    accumulation over output steps matters).

    float64 everywhere. In float32 the representation error alone is ~1e-7
    relative, so a central difference with eps = 1e-5 has ~1e-2 relative noise
    and the check would be meaningless -- it would pass with real bugs present.

    Central differences (f(x+e) - f(x-e)) / 2e are used rather than forward
    differences because their error is O(eps^2), which is what makes a 1e-4
    tolerance achievable at all.
    """
    rng = np.random.default_rng(seed)
    model = ConvLSTMNowcaster(in_ch=1, t_out=2, hidden=2, kernel=3,
                              pos_weight=C.HP_NUMPY["pos_weight"],
                              downsample=1, seed=seed, dtype=np.float64)

    # Jitter EVERY parameter away from its initialisation. At init the gate
    # biases are 0 (or exactly 1.0 for forget) and the point is near-symmetric;
    # a check performed at a symmetric point can pass while a real asymmetry
    # bug hides. A generic random point has no such cancellations.
    for p in model._params().values():
        p += rng.standard_normal(p.shape) * 0.35

    x = rng.standard_normal((1, 2, 1, 6, 6))
    # A target with a healthy mix of both classes, so both branches of the
    # weighted loss (and hence the pos_weight factor) are exercised.
    y = (rng.random((1, 2, 6, 6)) < 0.3).astype(np.float64)

    loss0, grads = model._loss_and_grad(x, y)

    def loss_at(name: str, flat_idx: int, delta: float) -> float:
        p = model._params()[name]
        flat = p.reshape(-1)
        old = flat[flat_idx]
        flat[flat_idx] = old + delta
        logits, _ = model._forward(x, cache=False)
        out = weighted_bce_logits(logits, y, model.pos_weight)
        flat[flat_idx] = old
        return out

    results: dict[str, float] = {}
    worst: tuple[float, str, int, float, float] = (0.0, "", -1, 0.0, 0.0)
    for name, p in model._params().items():
        max_rel = 0.0
        for fi in _probe_indices(name, p, model.hidden, n_samples, rng):
            num = (loss_at(name, fi, eps) - loss_at(name, fi, -eps)) / (2 * eps)
            ana = float(grads[name].reshape(-1)[fi])
            denom = max(abs(num) + abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
            if rel > worst[0]:
                worst = (rel, name, fi, ana, num)
        results[name] = max_rel

    overall = max(results.values())
    if verbose:
        print("gradient check (float64, central differences, "
              f"eps={eps:g}, tol={tol:g})")
        print(f"  loss at check point: {loss0:.8f}")
        for name, p in model._params().items():
            flag = "ok" if results[name] < tol else "FAIL"
            print(f"  {name:8s} shape {str(tuple(p.shape)):16s} "
                  f"max rel err {results[name]:.3e}  {flag}")
        print(f"  worst: {worst[1]}[{worst[2]}] analytic {worst[3]:+.8e} "
              f"numeric {worst[4]:+.8e}")
        print(f"  OVERALL max rel err {overall:.3e}")
    assert overall < tol, (
        f"backward pass is wrong: max relative error {overall:.3e} >= {tol:g}; "
        f"worst tensor {worst[1]} index {worst[2]} "
        f"(analytic {worst[3]:.6e} vs numeric {worst[4]:.6e})")
    results["_overall"] = overall
    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _smoke_train(seconds: float = 40.0, epochs: int = 12) -> None:
    """Train briefly on synthetic data and show that the loss actually falls.

    Deliberately compared against the best CONSTANT forecast. "Loss went down"
    is a weak claim, because most of the early decrease is the model learning
    the base rate, which requires no spatial skill at all. The constant-optimal
    weighted loss is the number to beat; anything above it means the network has
    learned nothing a single scalar could not do.

    Only a few epochs on a handful of sequences: this proves the machinery
    trains, nothing more. Left running on 16 sequences it overfits by epoch ~25
    (train keeps falling, val turns upward), which is expected and is the
    pipeline's problem to solve with real data volume and early stopping.
    """
    import os
    import tempfile

    import synth
    import features
    import metrics as M

    t0 = time.time()
    events = synth.generate_dataset(n_events=4, seed=7)
    xs, ys = [], []
    for e in events:
        t_max = e["x"].shape[0] - C.INPUT_FRAMES - C.OUTPUT_FRAMES
        for s in range(0, t_max, 7):
            xs.append(e["x"][s:s + C.INPUT_FRAMES])
            ys.append(e["y"][s + C.INPUT_FRAMES:
                             s + C.INPUT_FRAMES + C.OUTPUT_FRAMES])
    X_raw = np.stack(xs)
    Y = np.stack(ys) >= C.LIGHTNING_THRESHOLD
    F = features.build_features_batch(X_raw)
    # Split by EVENT order, not at random: consecutive windows overlap in their
    # input frames, so a random split would leak (see config.SPLIT).
    n_val = max(1, F.shape[0] // 5)
    Ftr, Ytr = F[:-n_val], Y[:-n_val]
    Fva, Yva = F[-n_val:], Y[-n_val:]
    print(f"  data: {F.shape} features, {Y.shape} targets, "
          f"base rate {Y.mean():.4f}, prep {time.time() - t0:.1f}s")

    model = ConvLSTMNowcaster(in_ch=features.N_FEATURES,
                              t_out=C.OUTPUT_FRAMES, hidden=8,
                              downsample=2, batch=4, lr=6e-3, seed=1)
    ckpt = os.path.join(tempfile.gettempdir(), "convlstm_smoke.npz")
    hist = model.fit(Ftr, Ytr, Fva, Yva, epochs=epochs, max_seconds=seconds,
                     verbose=True, checkpoint=ckpt)

    tr = hist["train_loss"]
    print(f"  train loss trajectory: "
          f"{' -> '.join(f'{v:.4f}' for v in tr)}")
    if hist["val_loss"]:
        print(f"  val   loss trajectory: "
              f"{' -> '.join(f'{v:.4f}' for v in hist['val_loss'])}")
        print("  (val is one held-out event with roughly twice the base rate, "
              "so its weighted loss")
        print("   sits at a different LEVEL than train by construction -- "
              "read the trend, not the gap)")

    # Reference: best possible constant probability under the weighted loss.
    r = float(model._down(Ytr.astype(np.float64), as_dtype=np.float64).mean())
    w = model.pos_weight
    p_star = w * r / (w * r + (1.0 - r))
    z_star = np.log(p_star / (1.0 - p_star))
    yd = model._down(Ytr.astype(np.float64), as_dtype=np.float64)
    const_loss = weighted_bce_logits(np.full_like(yd, z_star), yd, w)
    print(f"  best CONSTANT forecast loss {const_loss:.4f} "
          f"(p*={p_star:.3f} at coarse base rate {r:.4f})")
    print(f"  final train {tr[-1]:.4f} -> "
          f"{'beats' if tr[-1] < const_loss else 'DOES NOT beat'} "
          f"the constant baseline")

    prob = model.predict(Fva)
    print(f"  predict: {prob.shape} range [{prob.min():.4f}, {prob.max():.4f}]"
          f"  mean {prob.mean():.4f}")
    assert prob.shape == Yva.shape, (prob.shape, Yva.shape)
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
    ap = M.auprc(prob, Yva)
    print(f"  val AUPRC {ap:.4f} vs base rate {Yva.mean():.4f} "
          f"({ap / max(Yva.mean(), 1e-9):.1f}x lift) "
          f"-- a few seconds of training, not a result")

    # ---- checkpoint round-trip -------------------------------------------
    # Two things must hold, and only the second one is interesting:
    #   (a) reloaded weights reproduce the same probabilities;
    #   (b) reloaded OPTIMISER state reproduces the same next update. Without
    #       (b) every resume restarts Adam's moment estimates, which produces a
    #       loss spike and makes a chunked run worse than an uninterrupted one.
    m2 = ConvLSTMNowcaster.load(ckpt)
    d_pred = float(np.abs(m2.predict(Fva[:1]) - prob[:1]).max())
    model.fit(Ftr, Ytr, epochs=1, verbose=False)
    m2.fit(Ftr, Ytr, epochs=1, verbose=False)
    d_param = max(float(np.abs(a - b).max())
                  for a, b in zip(model._params().values(),
                                  m2._params().values()))
    print(f"  checkpoint: max |dprob| {d_pred:.2e}, "
          f"max |dparam| after one further epoch on each {d_param:.2e}, "
          f"adam_t {m2.opt_t}")
    assert d_pred == 0.0 and d_param == 0.0

    # ---- wall-clock budget ------------------------------------------------
    # The reason max_seconds exists: this sandbox kills any call at ~120 s, so a
    # long training run has to be resumable in slices. Ask for 50 epochs with a
    # 1 s budget and check it stops cleanly mid-run instead of being killed.
    m3 = ConvLSTMNowcaster.load(ckpt)
    e0 = m3.epochs_done
    h3 = m3.fit(Ftr, Ytr, epochs=50, max_seconds=1.0, verbose=False)
    print(f"  budget stop: asked 50 epochs with a 1.0s budget, ran "
          f"{m3.epochs_done - e0} full epochs in {h3['seconds']:.2f}s, "
          f"stopped_early={h3['stopped_early']}")
    assert h3["stopped_early"] and m3.epochs_done - e0 < 50


if __name__ == "__main__":
    t_script = time.time()
    print("=" * 70)
    print("GRADIENT CHECK -- analytic BPTT vs central finite differences")
    print("=" * 70)
    gradient_check()

    print()
    print("=" * 70)
    print("SMOKE TRAIN -- tiny model, few seconds, loss must decrease")
    print("=" * 70)
    _smoke_train(seconds=40.0, epochs=12)
    print(f"\ntotal script time {time.time() - t_script:.1f}s")

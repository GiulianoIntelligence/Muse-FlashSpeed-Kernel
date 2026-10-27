"""Micro-probes for the suspect Triton ops, run under TRITON_INTERPRET=1."""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from nvfp4 import dequantize  # noqa: E402
import submission as S  # noqa: E402


@triton.jit
def _probe_bf16(x_ptr, o_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    v = tl.load(x_ptr + offs)          # bf16 -> fp32
    tl.store(o_ptr + offs, v)


@triton.jit
def _probe_dq(w_ptr, sf_ptr, gs_ptr, o_ptr, H: tl.constexpr, BR: tl.constexpr, KC: tl.constexpr):
    """Dequant rows [j0, j0+BR) cols [0, KC) exactly as _k1 does; store fp32 [BR, KC]."""
    j = tl.arange(0, BR)
    kkh = tl.arange(0, KC // 2)
    kk = tl.arange(0, KC)
    gs = tl.load(gs_ptr)
    b = tl.load(w_ptr + j[:, None] * (H // 2) + kkh[None, :])
    s = tl.load(sf_ptr + j[:, None] * (H // 16) + (kk // 16)[None, :])
    w = (_dq_operands(b) * (s.to(tl.float32) * gs))
    tl.store(o_ptr + j[:, None] * KC + kk[None, :], w)


@triton.jit
def _dq_operands(b):
    lo = S._e2m1((b & 0x0F).to(tl.int32))
    hi = S._e2m1((b >> 4).to(tl.int32))
    return tl.interleave(lo, hi)


@triton.jit
def _probe_dot(x_ptr, w_ptr, o_ptr, CP: tl.constexpr, KC: tl.constexpr, BR: tl.constexpr):
    """acc = dot(a [CP,KC] bf16, trans(w [BR,KC] bf16)) with w computed in-register."""
    cp = tl.arange(0, CP)
    kk = tl.arange(0, KC)
    br = tl.arange(0, BR)
    a = tl.load(x_ptr + cp[:, None] * KC + kk[None, :])              # bf16
    w = tl.load(w_ptr + br[:, None] * KC + kk[None, :])              # bf16
    w2 = w * 1.0                                                      # force register compute
    acc = tl.dot(a, tl.trans(w2), out_dtype=tl.float32)
    tl.store(o_ptr + cp[:, None] * BR + br[None, :], acc)


def main():
    H, I, BR, KC = 64, 48, 16, 32
    torch.manual_seed(1)
    w13 = torch.randint(0, 256, (I, H // 2), dtype=torch.uint8)
    sf = torch.randn((I, H // 16)).to(torch.float8_e4m3fn)
    gs = torch.tensor([0.25], dtype=torch.float32)

    # C: bf16 round-trip
    x = torch.randn((32,), dtype=torch.bfloat16)
    o = torch.empty((32,), dtype=torch.float32)
    _probe_bf16[(1,)](x, o, N=32)
    print("C bf16 roundtrip max err:", (o - x.float()).abs().max().item(),
          "| finite:", bool(torch.isfinite(o).all()))

    # A: dequant path vs nvfp4
    o = torch.full((BR, KC), float("nan"), dtype=torch.float32)
    _probe_dq[(1,)](w13, sf, gs, o, H=H, BR=BR, KC=KC)
    want = dequantize(w13[:BR, : KC // 2], sf[:BR, : KC // 16], gs)
    err = (o - want).abs().max().item()
    rel = err / want.abs().max().item()
    print(f"A dequant max abs err {err:.3e} rel {rel:.3e} | got finite: {bool(torch.isfinite(o).all())}")

    # B: dot(a, trans(w_computed))
    a = torch.randn((16, KC), dtype=torch.bfloat16)
    wt = torch.randn((BR, KC), dtype=torch.bfloat16)
    o = torch.full((16, BR), float("nan"), dtype=torch.float32)
    _probe_dot[(1,)](a, wt, o, CP=16, KC=KC, BR=BR)
    want = a.float() @ wt.float().T
    err = (o - want).abs().max().item()
    print(f"B dot/trans max abs err {err:.3e} | got finite: {bool(torch.isfinite(o).all())} "
          f"| sample got {o[0, :3].tolist()} want {want[0, :3].tolist()}")


if __name__ == "__main__":
    main()

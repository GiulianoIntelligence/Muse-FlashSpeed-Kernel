"""Local (no-GPU) validation of submission.py's Triton kernels via TRITON_INTERPRET=1.

Runs the four kernels on CPU with tiny shapes and fp32 buffers (kernels are
dtype-generic; production passes bf16), comparing every stage against a plain torch
fp32 reference, so semantic bugs are found here instead of in an `okc check` run.
Usage: /tmp/venv/bin/python test_local.py
"""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import triton  # noqa: E402

import submission as S  # noqa: E402
from nvfp4 import dequantize, random_weights  # noqa: E402

torch.manual_seed(0)

H, I = 64, 128           # I%KC==0 required (no k-tail mask); I%BR!=0 exercises jm masks
N_EXP = 16
START, COUNT = 2, 4      # nonzero start exercises the gid = start + e mapping
TOPK = 8
BR1, BR2, KC = 32, 32, 64


def make_case(M):
    g = torch.Generator().manual_seed(M)
    w13, w13_sf, gs13 = random_weights(2 * I, H, generator=g, device="cpu", leading=(N_EXP,))
    w2, w2_sf, gs2 = random_weights(H, I, generator=g, device="cpu", leading=(N_EXP,))
    weights = {
        "w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(N_EXP).contiguous(),
        "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(N_EXP).contiguous(),
        "post_norm_w": (0.1 * torch.randn((H,), generator=g)).to(torch.bfloat16),
    }
    x = torch.randn((M, H), generator=g).to(torch.bfloat16)
    ids = torch.randint(0, N_EXP, (M, TOPK), generator=g, dtype=torch.int32)
    gw = torch.rand((M, TOPK), generator=g)
    gw = (gw / gw.sum(-1, keepdim=True)).float()
    return x, ids, gw, weights


def ref(x, ids, gw, weights):
    """fp32 reference, mirroring reference.py's ref_kernel."""
    xf = x.float()
    out = torch.zeros((x.shape[0], H), dtype=torch.float32)
    for e in range(START, START + COUNT):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e])
        h = xf[tok] @ w13.T
        inter = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        w2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e])
        y = inter @ w2.T
        s = y * (1.0 + weights["post_norm_w"].float())
        yn = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, yn * gw[tok, slot].unsqueeze(1))
    return out


def run(M):
    x, ids, gw, weights = make_case(M)
    state = S.prepare(weights)
    xf = x.float()
    M8 = M * TOPK

    list_ws = torch.full((COUNT, M8), -1, dtype=torch.int32)
    counts = torch.zeros((COUNT,), dtype=torch.int32)
    S._route_prep[(COUNT,)](ids, list_ws, counts, M8, START, M8,
                            BLOCK=max(16, triton.next_power_of_2(M8)), num_warps=1)

    for e in range(COUNT):
        want = sorted(t * TOPK + k for t in range(M) for k in range(TOPK)
                      if ids[t, k].item() == START + e)
        got = sorted(list_ws[e, :counts[e]].tolist())
        assert got == want, f"routing e={e}: got {got} want {want}"
    print(f"M={M}: routing OK {counts.tolist()}")

    CP = min(max(triton.next_power_of_2(M), 16), 128)
    inter = torch.full((M8, I), float("nan"), dtype=torch.float32)
    y = torch.full((M8, H), float("nan"), dtype=torch.float32)
    out = torch.empty((M, H), dtype=torch.float32)

    if M <= 8:
        RPAD = min(max(triton.next_power_of_2(M), 1), 8)
        S._gv1[(COUNT, I // 16)](xf, list_ws, counts, state["w13T"], state["w13_sfT"],
                                 state["w13_gs"], inter, M8, START, M8,
                                 H=H, I=I, RP=16, KC=KC, RPAD=RPAD, num_warps=1)
        S._gv2[(COUNT, H // 16)](inter, list_ws, counts, state["w2T"], state["w2_sfT"],
                                 state["w2_gs"], y, START, M8,
                                 H=H, I=I, RP=16, KC=KC, RPAD=RPAD, num_warps=1)
    else:
        S._k1[(COUNT, triton.cdiv(I, BR1))](xf, list_ws, counts, state["w13T"], state["w13_sfT"],
                                            state["w13_gs"], inter, M8, START, M8,
                                            H=H, I=I, BR=BR1, KC=KC, CP=CP, num_warps=1)
        S._k2[(COUNT, triton.cdiv(H, BR2))](inter, list_ws, counts, state["w2T"], state["w2_sfT"],
                                            state["w2_gs"], y, START, M8,
                                            H=H, I=I, BR=BR2, KC=KC, CP=CP, num_warps=1)
    S._k3[(M,)](y, ids, gw, state["post_norm_w"], out, START, COUNT,
                H=H, EPS=1e-8, num_warps=1)

    for t in range(M):
        for k in range(TOPK):
            e = ids[t, k].item()
            if not (START <= e < START + COUNT):
                continue
            w13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e])
            h = xf[t] @ w13.T
            want = torch.nn.functional.silu(h[:I]) * h[I:]
            got = inter[t * TOPK + k]
            err = (got - want).norm() / want.norm().clamp(min=1e-9)
            assert err < 1e-3, f"inter t={t} k={k} e={e}: rel err {err:.5f}"
    print(f"M={M}: inter OK")

    r = ref(x, ids, gw, weights)
    if counts.sum() == 0:
        assert torch.isfinite(out).all() and out.abs().max() == 0, \
            f"no local routes: out must be exact zeros, got max {out.abs().max()}"
        print(f"M={M}: no local routes -> zeros OK")
        return
    rel = (out - r).norm() / r.norm().clamp(min=1e-9)
    row = (out - r).reshape(M, -1).norm(dim=-1) / r.reshape(M, -1).norm(dim=-1).clamp(min=1e-9)
    print(f"M={M}: out rel L2 {rel:.6f} worst row {row.max():.6f}  (pass < 0.015 / 0.075)")
    assert rel < 1e-3 and row.max() < 1e-2, "output mismatch"


if __name__ == "__main__":
    for M in (1, 5, 64):
        run(M)
    print("ALL OK")

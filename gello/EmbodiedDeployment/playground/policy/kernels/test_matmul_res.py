import time

import pandas as pd
import torch
import triton
import triton.language as tl


# from dexmal realtime vla
@triton.jit
def matmul_small_res_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    res_ptr,
    seq_len: tl.constexpr,
    hidden: tl.constexpr,
    features: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_N)
    grid_j = tl.cdiv(hidden, BLOCK_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_N
        j = (p % grid_j) * BLOCK_M
        acc = tl.load(
            res_ptr
            + (i + tl.arange(0, BLOCK_N))[:, None] * hidden
            + (j + tl.arange(0, BLOCK_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_M))[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        for k in range(0, features, BLOCK_K):
            x = tl.load(
                inp_ptr
                + (i + tl.arange(0, BLOCK_N))[:, None] * features
                + (k + tl.arange(0, BLOCK_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_N))[:, None] < seq_len)
                & ((k + tl.arange(0, BLOCK_K))[None, :] < features),
                other=0.0,
            )
            w = tl.load(
                weight_ptr
                + (k + tl.arange(0, BLOCK_K))[:, None] * hidden
                + (j + tl.arange(0, BLOCK_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_K))[:, None] < features)
                & ((j + tl.arange(0, BLOCK_M))[None, :] < hidden),
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr
            + (i + tl.arange(0, BLOCK_N))[:, None] * hidden
            + (j + tl.arange(0, BLOCK_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_N))[:, None] < seq_len)
            & ((j + tl.arange(0, BLOCK_M))[None, :] < hidden),
        )


def triton_matmul_res(x, weight, residual):
    seq_len, features = x.shape
    hidden = residual.shape[1]
    out = torch.empty_like(residual)

    BLOCK_N = 128
    BLOCK_M = 64
    BLOCK_K = 64
    num_warps = 8
    num_stages = 4

    # grid = (triton.cdiv(seq_len, BLOCK_N) * triton.cdiv(hidden, BLOCK_M),)
    grid = (((seq_len + BLOCK_N - 1) // BLOCK_N) * (2048 // 64),)

    matmul_small_res_kernel[grid](
        x,
        weight,
        out,
        residual,
        seq_len=seq_len,
        hidden=hidden,
        features=features,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# ====================== 2. 基准测试函数 ======================
def benchmark(fn, x, weight, residual, name):
    # warmup
    for _ in range(10):
        out = fn(x, weight, residual)
        torch.cuda.synchronize()

    iters = 50
    start = time.time()
    for _ in range(iters):
        out = fn(x, weight, residual)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters * 1000  # ms
    return elapsed


def debugpy_listen():
    import debugpy

    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for client to attach 10092...")
    debugpy.wait_for_client()


if __name__ == "__main__":
    # debugpy_listen()

    torch.manual_seed(0)
    device = torch.device("cuda")

    results = []

    test_cases = [
        # (seq_len, hidden, features, case_name)
        (778, 2048, 16384, "matmul_n_16384_2048_res size"),
    ]

    for seq_len, hidden, features, name in test_cases:
        print(f"\n=== Testing {name} ===")
        x = torch.randn(
            seq_len, features, device=device, dtype=torch.bfloat16
        ).contiguous()
        weight = torch.randn(
            features, hidden, device=device, dtype=torch.bfloat16
        ).contiguous()
        residual = torch.randn(
            seq_len, hidden, device=device, dtype=torch.bfloat16
        ).contiguous()

        # 1. torch.compile (matmul + add)
        def torch_compile_fn(x, w, r):
            return (x @ w) + r

        torch_compile_fn = torch.compile(
            torch_compile_fn, mode="max-autotune", fullgraph=True
        )

        t_torch = benchmark(
            torch_compile_fn, x, weight, residual, "torch.compile"
        )

        # 2. 我们的融合 kernel
        t_triton = benchmark(
            triton_matmul_res, x, weight, residual, "triton_fused"
        )

        speedup = t_torch / t_triton

        results.append(
            {
                "场景": name,
                "seq_len": seq_len,
                "hidden": hidden,
                "features (K)": features,
                "torch.compile (ms)": f"{t_torch:.2f}",
                "matmul_small_res (ms)": f"{t_triton:.2f}",
                "加速比": f"{speedup:.2f}x",
            }
        )

    # ====================== 4. 美观打印表格 ======================
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("实测数据（A100-80GB PCIe，bf16，Triton 2.2）")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

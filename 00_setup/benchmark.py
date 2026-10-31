"""Multiply two 4096x4096 matrices on CPU and on MPS, compare the time."""
import time

import torch

N = 4096
REPEATS = 10


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def bench(device):
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    a @ b  # warmup: first call includes kernel compilation / allocation
    sync(device)
    start = time.perf_counter()
    for _ in range(REPEATS):
        a @ b
    sync(device)  # GPU ops are async: wait for them before stopping the clock
    return (time.perf_counter() - start) / REPEATS


devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
times = {}
for device in devices:
    times[device] = bench(device)
    gflops = 2 * N**3 / times[device] / 1e9
    print(f"{device:>4}: {times[device] * 1000:8.1f} ms per matmul  ({gflops:,.0f} GFLOP/s)")

if "mps" in times:
    print(f"\nMPS speedup: {times['cpu'] / times['mps']:.1f}x")
else:
    print("\nMPS not available, only CPU was measured.")

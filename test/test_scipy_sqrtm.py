import sys
from pathlib import Path
from itertools import cycle
import time

src_path = Path(__file__).parents[1].joinpath("src")
assert src_path.exists(), f"Path does not exist: {src_path}"
sys.path.append(src_path.as_posix())

import torch
import numpy as np
import scipy
from scipy import linalg as spla
from transformers import set_seed
from tqdm import tqdm

from loqer.statistic_profiler.scale import sqrt_newton_schulz


def compute_sqrtm_cuda(A, numIters=200):
    A_sqrt = sqrt_newton_schulz(A, numIters)
    return A_sqrt


def compute_sqrtm_numpy(A):
    A_sqrt = spla.sqrtm(A)
    return A_sqrt


if __name__ == "__main__":
    set_seed(42)
    num_runs = 4
    matrix_size = (13824, 13824)  # CUDA: 76277.03 ms = 1.1min, NumPy kraken: 6min

    A = torch.randn(*matrix_size).cuda()
    print("Warmup")
    for _ in range(40):
        _ = torch.matmul(A, A.transpose(0, 1))

    # CUDA
    print("CUDA")
    A_list = [np.random.randn(*matrix_size).astype(np.float32) for _ in range(5)]
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in tqdm(range(num_runs), total=num_runs):
        A = A_list[i % 5]
        A = torch.from_numpy(A).cuda().reshape(1, matrix_size[0], matrix_size[1])
        start_events[i].record()
        compute_sqrtm_cuda(A, numIters=200)
        end_events[i].record()

    torch.cuda.synchronize()
    cuda_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f"CUDA: {np.mean(cuda_times):.2f} ms")

    # NumPy
    print("NumPy")
    start_times = []
    end_times = []
    for i in tqdm(range(num_runs), total=num_runs):
        A = A_list[i % 5]
        start_times.append(time.time())
        compute_sqrtm_numpy(A)
        end_times.append(time.time())

    print(f"NumPy: {np.mean(np.array(end_times) - np.array(start_times)):.2f} ms")

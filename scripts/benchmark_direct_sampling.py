#!/usr/bin/env python3
"""Time Direct Sampling on the NumPy reference and on the PyTorch CUDA scorer, each at its own default chunk, and
check that they select the same candidates. A GPU's presence is not a speedup: this measures where each backend wins.
Local only (not run in CI).

Usage: python scripts/benchmark_direct_sampling.py OUT_JSON
"""

import json
import platform
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from geocond.direct_sampling import direct_sampling


def channels(shape):
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    x, y, z = grids
    return (np.sin(2 * np.pi * x / 12 + 1.2 * np.sin(2 * np.pi * y / 20) + 0.3 * np.sin(2 * np.pi * z / 9)) > 0.3)


def main(out):
    import torch

    smooth = gaussian_filter(np.random.default_rng(0).normal(size=(64, 64, 16)), sigma=3)
    cases = [
        ("categorical channels, TI 64x64x1, grid 32x32x1, threshold 0, full scan", channels((64, 64, 1)).astype(float),
         (32, 32, 1), "categorical", 20, 0.0, 1.0),
        ("categorical channels, TI 128x128x1, grid 32x32x1, threshold 0, full scan",
         channels((128, 128, 1)).astype(float), (32, 32, 1), "categorical", 20, 0.0, 1.0),
        ("categorical channels, TI 40x40x40, grid 16x16x16, threshold 0, scan 0.25", channels((40, 40, 40)).astype(float),
         (16, 16, 16), "categorical", 20, 0.0, 0.25),
        ("categorical channels, TI 64x64x64, grid 8x8x8, threshold 0, full scan", channels((64, 64, 64)).astype(float),
         (8, 8, 8), "categorical", 30, 0.0, 1.0),
        ("continuous smooth field, TI 64x64x16, grid 10x10x4, threshold 0.02, full scan", smooth, (10, 10, 4),
         "continuous", 20, 0.02, 1.0),
    ]
    rows = []
    for name, ti, grid, kind, neighbours, threshold, scan in cases:
        kw = {"variable_kinds": [kind], "max_neighbors": neighbours, "threshold": threshold, "scan_fraction": scan, "seed": 1}
        t0 = time.perf_counter()
        a = direct_sampling(ti, grid, **kw)
        t1 = time.perf_counter()
        b = direct_sampling(ti, grid, backend="torch", **kw)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        done = a.candidate >= 0
        rows.append({"case": name, "nodes": int(done.sum()), "numpy_s": round(t1 - t0, 2), "cuda_s": round(t2 - t1, 2),
                     "same_candidates": bool(np.array_equal(a.candidate, b.candidate)),
                     "same_scores": bool(np.array_equal(a.score, b.score, equal_nan=True)),
                     "mean_scanned": round(float(a.scanned[done].mean())), "valid_centers": a.parameters["valid_centers"],
                     "fallbacks": int(a.fallback.sum())})
        print(json.dumps(rows[-1]))
    report = {"cpu": platform.processor() or platform.machine(), "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "numpy": np.__version__,
              "chunks": {"numpy": 1024, "torch": 65536}, "rows": rows}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])

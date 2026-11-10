"""Window statistics, batch aggregation and trajectory bootstrap."""
import numpy as np
import torch
import torch.nn.functional as F


def motion_weights(gt):
    """Per-pixel weights from the largest change relative to the first frame."""
    return 0.01 + np.abs(gt - gt[:1]).max(axis=3, keepdims=True).max(axis=0, keepdims=True)


def ssim_frames(x, y):
    """Per-frame SSIM with a 7x7 uniform window for (T, H, W, 3) tensors in [0, 1]."""
    x = torch.as_tensor(x, dtype=torch.float64).permute(0, 3, 1, 2)
    y = torch.as_tensor(y, dtype=torch.float64).permute(0, 3, 1, 2)
    pool = lambda z: F.avg_pool2d(z, 7, 1)
    unbiased = 49 / 48
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ux, uy = pool(x), pool(y)
    vx = unbiased * (pool(x * x) - ux * ux)
    vy = unbiased * (pool(y * y) - uy * uy)
    vxy = unbiased * (pool(x * y) - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return s.mean(dim=(2, 3)).mean(dim=1)


def window_stats(pred, gt, device="cpu", ssim=True):
    """Return (MSE, weighted squared-error sum, weight sum, SSIM) of one window."""
    se = (pred - gt) ** 2
    w = np.broadcast_to(motion_weights(gt), gt.shape)
    s = 0.0
    if ssim:
        a = torch.from_numpy(np.ascontiguousarray(pred)).to(device)
        b = torch.from_numpy(np.ascontiguousarray(gt)).to(device)
        s = float(ssim_frames(a, b).mean())
    return float(se.mean()), float((w * se).sum(dtype=np.float64)), float(w.sum(dtype=np.float64)), s


def aggregate(rows):
    """Average window rows (key, MSE, weighted sum, weight sum, SSIM) in consecutive batches of four."""
    rows = sorted(rows, key=lambda r: r[0])
    out = {"mse": [], "mmse": [], "psnr": [], "ssim": []}
    for b in range(0, len(rows), 4):
        batch = rows[b:b + 4]
        mse = float(np.mean([r[1] for r in batch]))
        out["mse"].append(mse)
        out["mmse"].append(sum(r[2] for r in batch) / (sum(r[3] for r in batch) + 1e-8))
        out["psnr"].append(10 * np.log10(1.0 / (mse + 1e-8)))
        out["ssim"].append(float(np.mean([r[4] for r in batch])))
    return {k: float(np.mean(v)) for k, v in out.items()}, len(rows)


def per_trajectory(rows):
    """Mean motion-weighted error of each trajectory, ordered by trajectory index."""
    by = {}
    for r in rows:
        by.setdefault(r[5], []).append(r[2] / (r[3] + 1e-8))
    return np.array([np.mean(by[k]) for k in sorted(by)])


def bootstrap_gain(x, c, rng, draws=20000):
    """Percentage error reduction of x relative to c: [point, 2.5%, 97.5%, 5%, trajectories]."""
    n = len(x)
    g = np.empty(draws)
    for b in range(draws):
        i = rng.integers(0, n, n)
        g[b] = 100 * (1 - x[i].mean() / c[i].mean())
    return [float(100 * (1 - x.mean() / c.mean())), float(np.quantile(g, .025)), float(np.quantile(g, .975)), float(np.quantile(g, .05)), n]


def load_records(folder):
    """Merge the per-shard window rows of an evaluation folder."""
    import glob
    import os
    rec, seen, shards = {}, [], None
    for f in sorted(glob.glob(os.path.join(folder, "shard_*.npz"))):
        d = np.load(f, allow_pickle=True)
        seen.append(int(d["shard"]))
        shards = int(d["shards"])
        for k, v in d["rec"].item().items():
            rec.setdefault(k, []).extend(v)
    if shards is None or sorted(seen) != list(range(shards)):
        raise SystemExit(f"missing shards in {folder}: found {sorted(seen)}")
    return rec

"""Environment table, video and metadata access, and the renderer's training cache."""
import argparse
import functools
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
import torch

ROOT = os.environ.get("ACWM_ROOT", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 120
WINDOW = 37
SPLITS = ("ind_test", "ood_test")

ENVS = {
    "push_cube": dict(path="rigid_dynamics/push_block", action_dim=2, frame_hw=(240, 240), checkpoint="VideoDiT_S_push_cube_240x240"),
    "stack_cube": dict(path="rigid_dynamics/stack_cube", action_dim=7, frame_hw=(240, 240), checkpoint="VideoDiT_S_stack_cube_240x240"),
    "push_rope": dict(path="deformable/push_rope", action_dim=2, frame_hw=(240, 240), checkpoint="VideoDiT_S_push_rope_240x240",
                      frames=64, action_scale=1.0, max_episodes=None, max_rope_length=2.6, fit_rope_length=2.7),
    "push_sand": dict(path="particle/push_sand", action_dim=7, frame_hw=(240, 400), checkpoint="VideoDiT_S_push_sand_240x400"),
    "pour_water": dict(path="particle/pour_water", action_dim=4, frame_hw=(240, 240), checkpoint="VideoDiT_S_pour_water_240x240",
                       frames=300, action_scale=40.0, max_episodes=None, holdout="splits/pour_water_holdout.json"),
    "robot_arm": dict(path="kinematics/robot_arm_64", action_dim=7, frame_hw=(240, 240), checkpoint="VideoDiT_S_robot_arm_240x240"),
    "reacher": dict(path="kinematics/reacher", action_dim=2, frame_hw=(240, 240), checkpoint="VideoDiT_S_reacher_240x240"),
}


def data_dir(env):
    return os.path.join(ROOT, "data", *ENVS[env]["path"].split("/"))


def cache_dir(env):
    return os.path.join(ROOT, "cache", env)


def output_dir(*parts):
    path = os.path.join(ROOT, "outputs", *parts)
    os.makedirs(path, exist_ok=True)
    return path


@functools.lru_cache(maxsize=None)
def metadata(env, split):
    """Episode list of a split."""
    return torch.load(os.path.join(data_dir(env), split, "metadata.pt"), weights_only=False)


def video_path(env, split, episode):
    return os.path.join(data_dir(env), split, os.path.basename(episode["video_path"]))


def read_video(path, hw):
    """Decode a video to (T, H, W, 3) uint8, resized to hw when the stored size differs."""
    import av
    frames = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            a = frame.to_ndarray(format="rgb24")
            if a.shape[:2] != tuple(hw):
                a = cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
            frames.append(a)
    return np.stack(frames)


@functools.lru_cache(maxsize=2)
def _video(path, hw):
    return read_video(path, hw)


def episode_actions(episode):
    """Actions of an episode as a (T, action_dim) float array."""
    return np.asarray(episode["actions"], np.float32).reshape(len(episode["actions"]), -1)


def pad(x, n):
    """Cut to n steps, repeating the last step when shorter."""
    x = x[:n]
    if len(x) < n:
        x = np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)])
    return x


@functools.lru_cache(maxsize=None)
def action_scale(env):
    return float(np.load(os.path.join(cache_dir(env), "meta.npz"))["action_scale"])


def load_window(env, split, ti, s0):
    """Ground-truth frames in [0, 1] and unscaled actions of one window."""
    e = metadata(env, split)[ti]
    video = _video(video_path(env, split, e), ENVS[env]["frame_hw"])
    frames = pad(video[s0:], WINDOW).astype(np.float32) / 255
    return frames, pad(episode_actions(e)[s0:], WINDOW)


def forecast_files(env, which):
    """Forecast shard files of the fitting or test windows, in name order."""
    folder = output_dir(env, "forecasts")
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.startswith(f"{which}_shard")]


def iter_forecasts(path, wanted):
    """Yield (row, frames) for the wanted rows of a forecast file without loading the whole array."""
    wanted = set(wanted)
    with zipfile.ZipFile(path) as z, z.open("pred.npy") as fh:
        version = np.lib.format.read_magic(fh)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read_header(fh)
        size = int(np.prod(shape[1:])) * dtype.itemsize
        for i in range(shape[0]):
            buf = fh.read(size)
            if i in wanted:
                yield i, np.frombuffer(buf, dtype).reshape(shape[1:])


def _decode(job):
    path, n = job
    return pad(read_video(path, (SIZE, SIZE)), n)


def prepare(env, workers=8):
    """Build the training cache: frames at 120 pixels, actions, holdout split, background and kinematic matrix."""
    cfg = ENVS[env]
    episodes = metadata(env, "ind_train")
    limit = cfg.get("max_rope_length")
    keep = [i for i, e in enumerate(episodes)
            if (limit is None or e["rope_length"] <= limit) and os.path.exists(video_path(env, "ind_train", e))]
    keep = keep[:cfg.get("max_episodes", 2000)]
    chosen = [episodes[i] for i in keep]
    T = cfg.get("frames") or max(int(np.median([int(e["length"]) for e in chosen])), WINDOW)

    cache = cache_dir(env)
    os.makedirs(cache, exist_ok=True)
    frames = np.lib.format.open_memmap(os.path.join(cache, "frames.npy"), mode="w+", dtype=np.uint8, shape=(len(chosen), T, SIZE, SIZE, 3))
    with Pool(workers) as pool:
        jobs = [(video_path(env, "ind_train", e), T) for e in chosen]
        for n, v in enumerate(pool.imap(_decode, jobs, chunksize=4)):
            frames[n] = v
    frames.flush()

    actions = np.stack([pad(episode_actions(e), T) for e in chosen])
    scale = cfg.get("action_scale") or float(1.0 / max(actions.std(), 1e-6))

    if "holdout" in cfg:
        # Listed episodes of the split, as positions in the cache.
        position = {e: n for n, e in enumerate(keep)}
        holdout = [position[e] for e in json.load(open(os.path.join(HERE, cfg["holdout"]))) if e in position]
    else:
        rng = np.random.default_rng(123)
        holdout = sorted(int(i) for i in rng.choice(len(chosen), max(1, len(chosen) // 20), replace=False))
    held = set(holdout)
    train = [i for i in range(len(chosen)) if i not in held]

    # Rough background for foreground detection: median over three frames of sampled training episodes.
    sample = np.random.default_rng(0).choice(train, min(200, len(train)), replace=False)
    rough = np.median(np.concatenate([frames[e, [0, T // 2, T - 1]] for e in sample]), 0).astype(np.float32) / 255

    # Clean background: median of the last frame over the training episodes.
    clean = np.median(np.stack([frames[e, T - 1] for e in train]).astype(np.float32) / 255, 0).transpose(2, 0, 1)

    # Kinematic matrix: least-squares map from cumulative scaled action to foreground centroid shift in pixels.
    scaled = actions * scale
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    rng = np.random.default_rng(0)
    moves, shifts = [], []
    for e in rng.choice(train, min(300, len(train)), replace=False):
        s = int(rng.integers(0, T - WINDOW + 1))
        f = frames[e, s:s + WINDOW].astype(np.float32) / 255
        fg = np.abs(f - rough[None]).max(-1)
        w = (fg > 0.15).astype(np.float32) * fg
        mass = w.sum((1, 2)) + 1e-6
        cx = (w * xx).sum((1, 2)) / mass
        cy = (w * yy).sum((1, 2)) / mass
        cum = np.cumsum(scaled[e, s:s + WINDOW - 1], 0)
        for k in (6, 12, 24, 36):
            moves.append(cum[k - 1])
            shifts.append([cx[k] - cx[0], cy[k] - cy[0]])
    kinematics, *_ = np.linalg.lstsq(np.array(moves), np.array(shifts), rcond=None)

    np.savez(os.path.join(cache, "meta.npz"), actions=actions, episode_index=np.array(keep), action_scale=scale)
    json.dump(holdout, open(os.path.join(cache, "holdout.json"), "w"))
    np.save(os.path.join(cache, "background.npy"), clean.astype(np.float32))
    np.save(os.path.join(cache, "kinematics.npy"), kinematics.astype(np.float32))
    print(f"{env}: {len(chosen)} episodes x {T} frames, {len(holdout)} held out, action scale {scale:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the renderer's training cache for one environment.")
    parser.add_argument("env", choices=sorted(ENVS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.env, args.workers)

"""Forecasts of the released ACWM-DiT-S model on the fitting and test windows."""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch
import yaml

import data

STEPS = 50
SCHEDULER_KEYS = {"scheduler.sigmas", "scheduler.timesteps", "scheduler.linear_timesteps_weights"}


def load_model(env, device):
    """Build the model with the official code and load the released checkpoint."""
    repo = os.path.join(data.ROOT, "repo")
    checkpoints = os.path.join(data.ROOT, "checkpoints")
    sys.path.insert(0, repo)
    cwd = os.getcwd()
    os.chdir(repo)
    from acwm.dataset.data_config import get_config_by_name
    from acwm.model.interface import get_dynamics_class
    cfg = yaml.safe_load(open(f"configs/envs/{env}.yaml"))
    cfg["model_config"]["action_dim"] = get_config_by_name(cfg["dataset"]["name"]).action_dim
    cfg["model_config"].update(yaml.safe_load(open(f"configs/model/{cfg['model_type']}.yaml")).get("model_config", {}))
    cfg["model_config"]["vae_config"] = [os.path.join(checkpoints, "Wan2.1_VAE.pth")]
    model = get_dynamics_class(cfg["dynamics_class"])(cfg["model_name"], cfg["model_config"]).eval()
    os.chdir(cwd)
    state = torch.load(os.path.join(checkpoints, data.ENVS[env]["checkpoint"], "latest.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k not in SCHEDULER_KEYS}
    model.load_state_dict(state, strict=False)
    return model.to(device)


def windows(env, which):
    """Windows as (split, trajectory, cached episode, start frame)."""
    L = data.WINDOW
    if which == "fit":
        cache = data.cache_dir(env)
        holdout = json.load(open(os.path.join(cache, "holdout.json")))
        meta = np.load(os.path.join(cache, "meta.npz"))
        T = int(meta["actions"].shape[1])
        long_rope = data.ENVS[env].get("fit_rope_length")
        # Held-out training episodes, four evenly spaced windows each: the second half, or all of them for push_rope.
        held = holdout[len(holdout) // 2:] if long_rope is None else holdout
        wins = [("ind_train", int(meta["episode_index"][e]), int(e), int(s)) for e in held for s in np.linspace(0, T - L, 4, dtype=int)]
        if long_rope is not None:
            # Training episodes with a rope longer than the limit, two evenly spaced windows each.
            wins += [("ind_train", ti, -1, int(s)) for ti, e in enumerate(data.metadata(env, "ind_train"))
                     if e["rope_length"] > long_rope for s in np.linspace(0, T - L, 2, dtype=int)]
        return wins
    # Every test trajectory, ten evenly spaced windows each.
    return [(split, ti, -1, int(s)) for split in data.SPLITS for ti, e in enumerate(data.metadata(env, split))
            for s in np.linspace(0, int(e["length"]) - L, 10, dtype=int)]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.env, device)
    wins = windows(args.env, args.which)[args.shard::args.shards]
    out = {"split": [], "ti": [], "s0": [], "pred": []}
    for b in range(0, len(wins), args.batch):
        batch = wins[b:b + args.batch]
        first, actions = [], []
        for split, ti, _, s0 in batch:
            frames, act = data.load_window(args.env, split, ti, s0)
            first.append(frames[0])
            actions.append(act)
        seed = zlib.crc32(repr(batch).encode())
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            pred = model.generate(torch.from_numpy(np.stack(first)).to(device), torch.from_numpy(np.stack(actions)).to(device),
                                  num_inference_steps=STEPS, noise_level=0.0, mode="parallel").float().cpu()
        pred = (pred[:, :data.WINDOW].clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
        for k, (split, ti, _, s0) in enumerate(batch):
            out["split"].append(split)
            out["ti"].append(ti)
            out["s0"].append(s0)
            out["pred"].append(pred[k])
        print(f"{args.env} {args.which} shard {args.shard}: {min(b + args.batch, len(wins))}/{len(wins)} windows", flush=True)
    path = os.path.join(data.output_dir(args.env, "forecasts"), f"{args.which}_shard{args.shard}.npz")
    np.savez_compressed(path, **{k: np.array(v) for k, v in out.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast the fitting or test windows of one environment.")
    parser.add_argument("env", choices=sorted(data.ENVS))
    parser.add_argument("which", choices=("fit", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    main(args)

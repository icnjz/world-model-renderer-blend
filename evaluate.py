"""Blend of the video model's forecast with the renderer: coefficient fitting, test evaluation and report."""
import argparse
import collections
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import data
import metrics
import renderer

CONTRASTS = (("blend", "model"), ("blend", "blend_identity"), ("blend_identity", "model"), ("renderer", "model"))


def run_dir(env, variant, seed):
    return data.output_dir(env, f"blend_{variant}_seed{seed}")


def render(net, env, gt, actions, device, identity=False):
    """Renderer frames 1..36 of a window at full resolution as (36, H, W, 3) arrays."""
    first = F.interpolate(torch.from_numpy(gt[0]).permute(2, 0, 1)[None], size=(data.SIZE, data.SIZE), mode="area").to(device)
    act = torch.from_numpy(actions * data.action_scale(env))[None].to(device)
    index = torch.arange(1, data.WINDOW, device=device)[None]
    with torch.no_grad():
        out = net(first, act, index, identity=identity)
    full = lambda z: F.interpolate(z.clamp(0, 1), size=data.ENVS[env]["frame_hw"], mode="bilinear", align_corners=False).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return (full(out[0]), full(out[1])) if identity else full(out)


def blend(forecast, rendering, beta):
    """Keep the first frame of the forecast and mix frames 1..36 with the rendering."""
    return np.concatenate([forecast[:1], forecast[1:] + beta * (rendering - forecast[1:])])


def fit(args):
    """Closed-form mixing coefficient under the motion-weighted squared error on the fitting windows."""
    net = renderer.load(args.env, args.variant, args.seed, args.device)
    num = {"blend": 0.0, "blend_identity": 0.0}
    den = dict.fromkeys(num, 0.0)
    model_error = []
    for f in data.forecast_files(args.env, "fit"):
        d = np.load(f)
        split, ti, s0 = d["split"], d["ti"], d["s0"]
        for i, pred in data.iter_forecasts(f, range(len(split))):
            gt, actions = data.load_window(args.env, str(split[i]), int(ti[i]), int(s0[i]))
            G = pred.astype(np.float32)[1:] / 255
            Y = gt[1:]
            w = metrics.motion_weights(gt)
            total = float(np.broadcast_to(w, Y.shape).sum(dtype=np.float64))
            model_error.append(float((w * (Y - G) ** 2).sum()) / total)
            for name, R in zip(num, render(net, args.env, gt, actions, args.device, identity=True)):
                step = R - G
                num[name] += float((w * step * (Y - G)).sum(dtype=np.float64)) / total
                den[name] += float((w * step * step).sum(dtype=np.float64)) / total
    beta = {k: float(np.clip(num[k] / den[k], 0, 1)) if den[k] > 0 else 0.0 for k in num}
    out = {"beta": beta, "windows": len(model_error), "model_error": float(np.mean(model_error))}
    json.dump(out, open(os.path.join(run_dir(args.env, args.variant, args.seed), "coefficient.json"), "w"), indent=1)
    print(args.env, json.dumps(out), flush=True)


def run(args):
    """Window statistics of every prediction on one shard of the test windows."""
    folder = run_dir(args.env, args.variant, args.seed)
    net = renderer.load(args.env, args.variant, args.seed, args.device)
    beta = json.load(open(os.path.join(folder, "coefficient.json")))["beta"]
    rec, count = {}, 0
    for f in data.forecast_files(args.env, "test"):
        d = np.load(f)
        split, ti, s0 = d["split"], d["ti"], d["s0"]
        mine = [i for i in range(len(split)) if (count + 1 + i) % args.shards == args.shard]
        count += len(split)
        for i, pred in data.iter_forecasts(f, mine):
            sp, t, s = str(split[i]), int(ti[i]), int(s0[i])
            gt, actions = data.load_window(args.env, sp, t, s)
            G = pred.astype(np.float32) / 255
            R, R_identity = render(net, args.env, gt, actions, args.device, identity=True)
            predictions = {"model": G, "blend": blend(G, R, beta["blend"]), "blend_identity": blend(G, R_identity, beta["blend_identity"]),
                           "renderer": np.concatenate([G[:1], R]), "copy_first": np.repeat(gt[:1], data.WINDOW, 0)}
            for name, p in predictions.items():
                rec.setdefault((name, sp), []).append((t * 1000 + s,) + metrics.window_stats(np.clip(p, 0, 1), gt, args.device) + (t,))
    np.savez(os.path.join(folder, f"shard_{args.shard}.npz"), rec=np.array(rec, dtype=object), shard=args.shard, shards=args.shards)


def report(args):
    """Tables and bootstrap gains over all test trajectories and over the first 50 of each split."""
    folder = run_dir(args.env, args.variant, args.seed)
    rec = metrics.load_records(folder)
    for key, rows in rec.items():
        if set(collections.Counter(r[5] for r in rows).values()) != {10}:
            raise SystemExit(f"{key}: expected 10 windows per trajectory")
    out = {"env": args.env, "coefficient": json.load(open(os.path.join(folder, "coefficient.json")))}
    for label, first50 in (("all_trajectories", False), ("first_50_trajectories", True)):
        rows = lambda name, sp: [r for r in rec[(name, sp)] if r[5] < 50] if first50 else rec[(name, sp)]
        table = {name: {sp: metrics.aggregate([r[:5] for r in rows(name, sp)]) for sp in data.SPLITS} for name in sorted({k[0] for k in rec})}
        rng = np.random.default_rng(128)
        gains = {}
        for a, c in CONTRASTS:
            gains[f"{a}_vs_{c}"] = {sp: metrics.bootstrap_gain(metrics.per_trajectory(rows(a, sp)), metrics.per_trajectory(rows(c, sp)), rng) for sp in data.SPLITS}
        out[label] = {"table": table, "gains": gains}
        print(f"\n{args.env} ({label})", flush=True)
        cell = lambda z: f"MSE {1e3 * z[0]['mse']:.3f} M-MSE {1e3 * z[0]['mmse']:6.2f} PSNR {z[0]['psnr']:5.2f} SSIM {z[0]['ssim']:.4f} (n{z[1]})"
        for name, t in table.items():
            print(f"  {name:15s} InD {cell(t['ind_test'])} | OoD {cell(t['ood_test'])}", flush=True)
        for name, g in gains.items():
            print(f"  {name:28s} OoD gain {g['ood_test'][0]:+.2f}% [{g['ood_test'][1]:+.2f}, {g['ood_test'][2]:+.2f}]  InD gain {g['ind_test'][0]:+.2f}% (5th pct {g['ind_test'][3]:+.2f})", flush=True)
    json.dump(out, open(os.path.join(folder, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the mixing coefficient, evaluate the test windows, or print the report.")
    parser.add_argument("command", choices=("fit", "run", "report"))
    parser.add_argument("env", choices=sorted(data.ENVS))
    parser.add_argument("--variant", choices=renderer.VARIANTS, default="exchange")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    {"fit": fit, "run": run, "report": report}[args.command](args)

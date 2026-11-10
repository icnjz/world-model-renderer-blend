"""Comparison of the two colour rules under the same blend: per environment and pooled."""
import argparse
import json
import os

import numpy as np
import torch

import data
import evaluate
import metrics
import renderer

NAMES = ("model",) + renderer.VARIANTS
CONTRASTS = (("exchange", "free"), ("exchange", "model"), ("free", "model"))
DRAWS = 20000


def run_dir(env, seed):
    return data.output_dir(env, f"compare_seed{seed}")


def run(env, args):
    """Window statistics of the forecast and of both blends on one shard of the test windows."""
    nets = {v: renderer.load(env, v, args.seed, args.device) for v in renderer.VARIANTS}
    beta = {v: json.load(open(os.path.join(evaluate.run_dir(env, v, args.seed), "coefficient.json")))["beta"]["blend"] for v in renderer.VARIANTS}
    rec, count = {}, 0
    for f in data.forecast_files(env, "test"):
        d = np.load(f)
        split, ti, s0 = d["split"], d["ti"], d["s0"]
        mine = [i for i in range(len(split)) if (count + 1 + i) % args.shards == args.shard]
        count += len(split)
        for i, pred in data.iter_forecasts(f, mine):
            sp, t, s = str(split[i]), int(ti[i]), int(s0[i])
            gt, actions = data.load_window(env, sp, t, s)
            G = pred.astype(np.float32) / 255
            predictions = {"model": G}
            for v in renderer.VARIANTS:
                predictions[v] = evaluate.blend(G, evaluate.render(nets[v], env, gt, actions, args.device), beta[v])
            for name, p in predictions.items():
                rec.setdefault((name, sp), []).append((t * 1000 + s,) + metrics.window_stats(np.clip(p, 0, 1), gt, ssim=False) + (t,))
    np.savez(os.path.join(run_dir(env, args.seed), f"shard_{args.shard}.npz"), rec=np.array(rec, dtype=object), shard=args.shard, shards=args.shards)


def report(env, args):
    """Table and bootstrap gains over all test trajectories of one environment."""
    folder = run_dir(env, args.seed)
    rec = metrics.load_records(folder)
    table = {}
    for name in NAMES:
        table[name] = {}
        for sp in data.SPLITS:
            m = metrics.aggregate([r[:5] for r in rec[(name, sp)]])[0]
            table[name][sp] = {k: m[k] for k in ("mse", "mmse", "psnr")}
    rng = np.random.default_rng(128)
    gains = {}
    for a, c in CONTRASTS:
        gains[f"{a}_vs_{c}"] = {sp: metrics.bootstrap_gain(metrics.per_trajectory(rec[(a, sp)]), metrics.per_trajectory(rec[(c, sp)]), rng, DRAWS) for sp in data.SPLITS}
    print(f"\n{env}", flush=True)
    cell = lambda m: f"MSE {1e3 * m['mse']:.3f} M-MSE {1e3 * m['mmse']:6.2f} PSNR {m['psnr']:5.2f}"
    for name, t in table.items():
        print(f"  {name:9s} InD {cell(t['ind_test'])} | OoD {cell(t['ood_test'])}", flush=True)
    span = lambda g: f"{g[0]:+.2f}% [{g[1]:+.2f}, {g[2]:+.2f}]"
    for name, g in gains.items():
        print(f"  {name:18s} OoD gain {span(g['ood_test'])}  InD gain {span(g['ind_test'])}", flush=True)
    json.dump({"env": env, "table": table, "gains": gains}, open(os.path.join(folder, "report.json"), "w"), indent=1)


def pool(args):
    """Mean over environments of the exchange-vs-free gain, with trajectories resampled within each environment."""
    rows = {}
    for env in args.envs:
        rec = metrics.load_records(run_dir(env, args.seed))
        rows[env] = {sp: (metrics.per_trajectory(rec[("exchange", sp)]), metrics.per_trajectory(rec[("free", sp)])) for sp in data.SPLITS}
    gain = lambda x, c: 100 * (1 - x.mean() / c.mean())
    rng = np.random.default_rng(128)
    out = {}
    for sp in ("ood_test", "ind_test"):
        point = {env: float(gain(*rows[env][sp])) for env in args.envs}
        draws = np.empty(DRAWS)
        for b in range(DRAWS):
            g = []
            for env in args.envs:
                x, c = rows[env][sp]
                i = rng.integers(0, len(x), len(x))
                g.append(gain(x[i], c[i]))
            draws[b] = np.mean(g)
        out[sp] = {"pooled": float(np.mean(list(point.values()))), "lo95": float(np.quantile(draws, .025)),
                   "hi95": float(np.quantile(draws, .975)), "per_env": point}
        each = "  ".join(f"{env} {v:+.2f}" for env, v in point.items())
        print(f"{sp}: pooled {out[sp]['pooled']:+.2f}% [{out[sp]['lo95']:+.2f}, {out[sp]['hi95']:+.2f}]  {each}", flush=True)
    json.dump(out, open(os.path.join(data.output_dir(), f"compare_seed{args.seed}_pooled.json"), "w"), indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate both colour rules on the test windows, report one environment, or pool several.")
    parser.add_argument("command", choices=("run", "report", "pool"))
    parser.add_argument("envs", nargs="+", choices=sorted(data.ENVS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.command == "pool":
        pool(args)
    else:
        for env in args.envs:
            {"run": run, "report": report}[args.command](env, args)

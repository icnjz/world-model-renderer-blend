"""One row per environment from the evaluation reports."""
import argparse
import json
import os

import data
import evaluate
import renderer


def main(args):
    cell = lambda m: f"{1e3 * m['mse']:.3f}/{1e3 * m['mmse']:.2f}/{m['psnr']:.2f}/{m['ssim']:.4f}"
    out = {}
    for env in args.envs:
        report = json.load(open(os.path.join(evaluate.run_dir(env, args.variant, args.seed), "report.json")))
        table, gains = report["all_trajectories"]["table"], report["all_trajectories"]["gains"]
        beta = report["coefficient"]["beta"]["blend"]
        ood, ind = gains["blend_vs_model"]["ood_test"], gains["blend_vs_model"]["ind_test"]
        colour = gains["blend_vs_blend_identity"]["ood_test"]
        out[env] = {"beta": beta, "blend_vs_model": gains["blend_vs_model"], "blend_vs_blend_identity": gains["blend_vs_blend_identity"],
                    "model": {sp: table["model"][sp][0] for sp in data.SPLITS}, "blend": {sp: table["blend"][sp][0] for sp in data.SPLITS}}
        print(f"{env:11s} beta {beta:.3f}  OoD gain {ood[0]:+.2f}% [{ood[1]:+.2f}, {ood[2]:+.2f}]  InD gain {ind[0]:+.2f}% (5th pct {ind[3]:+.2f})  "
              f"colour rule OoD gain {colour[0]:+.2f}%  OoD MSE/M-MSE/PSNR/SSIM {cell(out[env]['model']['ood_test'])} -> {cell(out[env]['blend']['ood_test'])}", flush=True)
    json.dump(out, open(os.path.join(data.output_dir(), f"summary_{args.variant}_seed{args.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise the evaluation reports of several environments.")
    parser.add_argument("envs", nargs="+", choices=sorted(data.ENVS))
    parser.add_argument("--variant", choices=renderer.VARIANTS, default="exchange")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())

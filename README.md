# World model and structured renderer blend

Code for blending the forecast of an action-conditioned video world model with a small structured renderer on the ACWM-Phys benchmark.

For each environment the code

- trains a renderer that predicts future frames from the first frame and the action sequence;
- runs the released ACWM-DiT-S model, kept frozen, on the same windows;
- mixes the two predictions with one coefficient fitted on held-out training windows;
- evaluates the result on the in-distribution and out-of-distribution test splits.

Environments: `push_cube`, `stack_cube`, `push_rope`, `push_sand`, `pour_water`, `robot_arm`, `reacher`.

## Method

**Renderer** (`renderer.py`, 120 x 120 pixels)

1. Matting. Over a fixed background `B`, the coverage `m0` is the smallest one that explains the first frame, and `P0` is the premultiplied colour.
2. Motion. A backward warp moves the material. The displacement is linear in the cumulative action, with a matrix initialised by least squares on training data, plus a learned residual flow.
3. Colour. Each channel follows a two-pool exchange with nonnegative rates `a` and `b`: `P_t = lam * P0 + (1 - lam) * a / (a + b) * m0`, with `lam = exp(-(a + b) * t)` and `t` the frame index divided by 36. Coverage is moved, never created.
4. Frame. `I_t = warp(P_t) + (1 - warp(m0)) * B`.

`data.py` caches at most the first 2000 episodes of `ind_train` whose video file is present (all such episodes for `pour_water`; for `push_rope` all with rope length up to 2.6). The renderer is trained on the cached episodes, except the held-out ones described below, by minimising the mean squared error on 8 random frames of random 37-frame windows.

**Blend** (`evaluate.py`)

With `G` the model forecast and `R` the rendering upsampled to the frame size, the prediction is `G + beta * (R - G)` on frames 1-36 and the forecast itself on frame 0. `beta` minimises the motion-weighted squared error on the fitting windows in closed form and is clipped to [0, 1].

Fitting windows. A random 5 % of the cached episodes (for `pour_water` the episodes listed in `splits/pour_water_holdout.json`) are held out from renderer training. `beta` is fitted on the second half of that list, with four evenly spaced 37-frame windows per episode; the first half is left unused. For `push_rope`, `beta` is fitted on all held-out episodes (four windows each) together with the `ind_train` episodes whose rope is longer than 2.7 (two windows each).

**Free colour rule** (`--variant free`)

The same renderer with the colour step replaced by an unconstrained per-pixel logit shift. `compare.py` evaluates both colour rules under the same blend.

## Setup

```bash
pip install -r requirements.txt
```

The scripts read and write under the directory given by the environment variable `ACWM_ROOT` (default: the current directory).

| Path | Content |
|---|---|
| `data/` | ACWM-Phys data (Hugging Face dataset `t1an/ACWM-Phys`), e.g. `data/rigid_dynamics/push_block/{ind_train,ind_test,ood_test}/` |
| `repo/` | Official ACWM-Phys code |
| `checkpoints/` | Released DiT-S checkpoints (Hugging Face `t1an/ACWM-Phys-checkpoints`), e.g. `checkpoints/VideoDiT_S_push_cube_240x240/latest.pt`, and `Wan2.1_VAE.pth` |
| `cache/` | Training caches and renderer checkpoints, written by `data.py` and `renderer.py` |
| `outputs/` | Forecasts and evaluation results, written by the other scripts |

`forecast.py` imports the official code, so the requirements of `repo/` must be installed as well. Forecasting uses a GPU when one is available and is slow without one. Renderer training and evaluation also run on CPU.

## Usage

One environment:

```bash
python data.py push_cube
python renderer.py push_cube
python forecast.py push_cube fit
python forecast.py push_cube test
python evaluate.py fit push_cube
python evaluate.py run push_cube
python evaluate.py report push_cube
```

Summary over environments:

```bash
python summary.py push_cube stack_cube push_rope push_sand pour_water robot_arm reacher
```

Comparison of the two colour rules:

```bash
python renderer.py push_cube --variant free
python evaluate.py fit push_cube --variant free
python compare.py run push_cube
python compare.py report push_cube
python compare.py pool push_cube stack_cube push_rope push_sand pour_water robot_arm reacher
```

`forecast.py`, `evaluate.py run` and `compare.py run` take `--shard i --shards n` to split the windows over several processes. `renderer.py`, `evaluate.py`, `compare.py` and `summary.py` take `--seed` for the renderer seed. `renderer.py`, `evaluate.py` and `summary.py` take `--variant exchange|free` (default `exchange`).

## Evaluation

Every test trajectory contributes 10 evenly spaced windows of 37 frames. Window statistics are averaged in batches of four windows. The report gives MSE, motion-weighted MSE (M-MSE), PSNR and SSIM on both test splits, over all trajectories and over the first 50 of each split, and the percentage M-MSE reduction between predictions with a bootstrap interval over trajectories.

Predictions in the report:

| Name | Prediction |
|---|---|
| `model` | Forecast of the released model |
| `blend` | Forecast mixed with the rendering |
| `blend_identity` | Forecast mixed with a rendering that keeps the first-frame colour |
| `renderer` | Rendering on frames 1-36 after frame 0 of the forecast |
| `copy_first` | First frame repeated |

## Outputs

| Path | Content |
|---|---|
| `outputs/<env>/forecasts/` | Model forecasts of the fitting and test windows |
| `outputs/<env>/blend_<variant>_seed<k>/coefficient.json` | Mixing coefficients |
| `outputs/<env>/blend_<variant>_seed<k>/report.json` | Tables and gains of one environment |
| `outputs/summary_<variant>_seed<k>.json` | One row per environment |
| `outputs/<env>/compare_seed<k>/report.json` | Both colour rules on one environment |
| `outputs/compare_seed<k>_pooled.json` | Gain of the exchange rule over the free rule, averaged over environments |

## Files

| File | Function |
|---|---|
| `data.py` | Environment table, video and metadata access, training cache |
| `renderer.py` | Renderer model and training |
| `forecast.py` | Forecasts of the released model |
| `evaluate.py` | Mixing coefficient, test evaluation, report |
| `compare.py` | Comparison of the two colour rules |
| `summary.py` | Summary over environments |
| `metrics.py` | Window statistics, aggregation, bootstrap |
| `splits/pour_water_holdout.json` | Held-out training episodes of `pour_water` |

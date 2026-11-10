"""Structured renderer: mats the first frame over a fixed background, moves the material with an
action-conditioned warp and updates its colour. Run as a script to train it."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data

GRID = 15
VARIANTS = ("exchange", "free")


def conv(i, o, stride=2):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1), nn.GroupNorm(8, o), nn.SiLU())


def up(i, o):
    return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(16, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(16, o), nn.SiLU())


def sample(image, grid):
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class Renderer(nn.Module):
    """Predicts future frames from the first frame and the action sequence.

    variant "exchange": each colour channel follows a two-pool exchange with nonnegative rates.
    variant "free": each colour channel follows an unconstrained per-pixel logit shift.
    """

    def __init__(self, background, kinematics, action_dim, variant="exchange"):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        self.frame_encoder = nn.Sequential(conv(3, 64), conv(64, 128), conv(128, 256), conv(256, 256), nn.Flatten(), nn.Linear(4096, 512))
        self.action_encoder = nn.GRU(action_dim, 256, batch_first=True)
        self.feature_encoder = nn.Sequential(conv(3, 32), conv(32, 64), conv(64, 64), conv(64, 64, 1))
        self.skip = nn.Conv2d(64, 32, 1)
        self.film1 = nn.Linear(768, 128)
        self.film2 = nn.Linear(768, 512)
        for layer in (self.film1, self.film2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.block1 = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.GroupNorm(16, 256), nn.SiLU(),
                                    nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(16, 256), nn.SiLU())
        self.block2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(16, 256), nn.SiLU(), nn.Conv2d(256, 256, 3, 1, 1))
        self.fuse = nn.Conv2d(256 + 32, 256, 1)
        # Output channels: 2 residual flow, then 6 exchange rates ("exchange") or 3 colour logit shifts ("free").
        head = nn.Conv2d(32, 8 if variant == "exchange" else 5, 3, 1, 1)
        with torch.no_grad():
            head.weight[:2].zero_()
            head.bias[:2].zero_()
            if variant == "exchange":
                head.weight[2:].zero_()
                head.bias[2:].fill_(-6.0)
        self.decoder = nn.Sequential(up(256, 128), up(128, 64), up(64, 32), head)
        self.kinematics = nn.Parameter(torch.as_tensor(kinematics, dtype=torch.float32))
        self.register_buffer("background", torch.as_tensor(background, dtype=torch.float32).clamp(1e-3, 1 - 1e-3))

    def forward(self, first, actions, index, identity=False):
        """Render the frames at the given indices.

        first: (B, 3, 120, 120) first frame in [0, 1].
        actions: (B, T, action_dim) scaled actions.
        index: (B, K) frame indices within the window.
        Returns (B * K, 3, 120, 120). With identity=True also returns the rendering that keeps the first-frame colour.
        """
        B, K = index.shape
        index = torch.cat([torch.zeros_like(index[:, :1]), index], 1)
        Q = K + 1

        past = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], 1)
        h, _ = self.action_encoder(past)
        h = torch.gather(h, 1, index[..., None].expand(B, Q, 256))
        f = self.frame_encoder(F.interpolate(first, size=(64, 64), mode="area"))[:, None].expand(B, Q, 512)
        code = torch.cat([f, h], -1).reshape(B * Q, -1)

        z = self.feature_encoder(first)
        skip = self.skip(z)[:, None].expand(B, Q, 32, GRID, GRID).reshape(B * Q, 32, GRID, GRID)
        z = z[:, None].expand(B, Q, 64, GRID, GRID).reshape(B * Q, 64, GRID, GRID)
        scale, shift = self.film1(code).chunk(2, -1)
        x = self.block1(z * (1 + scale[..., None, None]) + shift[..., None, None])
        scale, shift = self.film2(code).chunk(2, -1)
        x = self.block2(x * (1 + scale[..., None, None]) + shift[..., None, None])
        out = self.decoder(self.fuse(torch.cat([x, skip], 1)))
        H = out.shape[-1]
        out = out.reshape(B, Q, -1, H, H)

        # Backward warp: residual flow relative to the zero-horizon query minus the kinematic displacement.
        flow = (out[:, 1:, :2] - out[:, :1, :2]).reshape(B * K, 2, H, H).permute(0, 2, 3, 1)
        cumulative = torch.cumsum(past, 1)
        move = torch.gather(cumulative, 1, index[:, 1:, None].expand(B, K, cumulative.shape[-1])) @ self.kinematics
        axis = torch.linspace(-1, 1, H, device=out.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack([xx, yy], -1)[None] + (flow - move.reshape(B * K, 1, 1, 2)) * (2.0 / (H - 1))

        # Matting of the first frame: coverage m0 and premultiplied colour P0.
        bg = self.background[None]
        m0 = torch.maximum((first - bg).clamp_min(0) / (1 - bg).clamp_min(1e-3), (bg - first).clamp_min(0) / bg.clamp_min(1e-3))
        m0 = m0.amax(1, keepdim=True).clamp(0, 1)
        P0 = torch.minimum((first - (1 - m0) * bg).clamp_min(0), m0)
        m0 = m0[:, None].expand(B, K, 1, H, H).reshape(B * K, 1, H, H)
        P0 = P0[:, None].expand(B, K, 3, H, H).reshape(B * K, 3, H, H)

        if self.variant == "exchange":
            rates = F.softplus(out[:, 1:, 2:8].reshape(B * K, 6, H, H))
            a, b = rates[:, :3], rates[:, 3:]
            t = (index[:, 1:].float() / (data.WINDOW - 1)).reshape(B * K, 1, 1, 1)
            keep = torch.exp(-(a + b) * t)
            P = keep * P0 + (1 - keep) * (a / (a + b + 1e-6)) * m0
        else:
            delta = (out[:, 1:, 2:5] - out[:, :1, 2:5]).reshape(B * K, 3, H, H)
            colour = (P0 / m0.clamp_min(1e-3)).clamp(1e-3, 1 - 1e-3)
            P = m0 * torch.sigmoid(torch.logit(colour) + delta)

        uncovered = (1 - sample(m0, grid)) * bg
        frames = sample(P, grid) + uncovered
        if identity:
            return frames, sample(P0, grid) + uncovered
        return frames


def checkpoint_path(env, variant, seed):
    return os.path.join(data.cache_dir(env), f"renderer_{variant}_seed{seed}.pt")


def build(env, variant):
    """Create a renderer with the background and kinematic matrix of an environment."""
    cache = data.cache_dir(env)
    return Renderer(np.load(os.path.join(cache, "background.npy")), np.load(os.path.join(cache, "kinematics.npy")),
                    data.ENVS[env]["action_dim"], variant)


def load(env, variant, seed, device="cpu"):
    """Load a trained renderer in evaluation mode."""
    net = build(env, variant)
    net.load_state_dict(torch.load(checkpoint_path(env, variant, seed), map_location="cpu"))
    return net.to(device).eval()


def train(args):
    cache = data.cache_dir(args.env)
    frames = np.load(os.path.join(cache, "frames.npy"), mmap_mode="r")
    actions = np.load(os.path.join(cache, "meta.npz"))["actions"] * data.action_scale(args.env)
    holdout = set(json.load(open(os.path.join(cache, "holdout.json"))))
    episodes = np.array([i for i in range(len(actions)) if i not in holdout])

    torch.manual_seed(args.seed)
    net = build(args.env, args.variant).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps + 1)
    rng = np.random.default_rng(1 + args.seed)
    L = data.WINDOW
    B, K = args.batch // 8, 8
    path = checkpoint_path(args.env, args.variant, args.seed)
    image = lambda x: torch.from_numpy(np.stack(x)).permute(0, 3, 1, 2).float().to(args.device) / 255
    start = time.time()
    for step in range(args.steps + 1):
        ep = rng.choice(episodes, B)
        s0 = rng.integers(0, frames.shape[1] - L + 1, B)
        index = np.stack([np.sort(rng.choice(np.arange(0, L), K, replace=False)) for _ in range(B)])
        first = image([frames[e, s] for e, s in zip(ep, s0)])
        target = image([frames[e, s + k] for e, s, ks in zip(ep, s0, index) for k in ks])
        act = torch.from_numpy(np.stack([actions[e, s:s + L] for e, s in zip(ep, s0)])).float().to(args.device)
        loss = F.mse_loss(net(first, act, torch.from_numpy(index).to(args.device)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"{args.env} {args.variant} seed {args.seed} step {step} loss {loss.item():.5f} "
                  f"({(time.time() - start) / max(step, 1):.2f}s/step)", flush=True)
        if step % 500 == 0 and step:
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, path)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the renderer of one environment.")
    parser.add_argument("env", choices=sorted(data.ENVS))
    parser.add_argument("--variant", choices=VARIANTS, default="exchange")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    train(args)

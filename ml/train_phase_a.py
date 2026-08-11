"""kev03 Phase A — self-supervised cartography training.

Trains ONLY the belief machinery (ObsEncoder + ConvGRU + AuxHeads): roll the
belief map over a T-step window of fog-masked observations and supervise the
interpretable channels against the full simulator truth:

  * obstacle logits      BCE            (vs later-revealed / full-sim truth)
  * gold mean + logvar   Gaussian NLL   (log1p domain, on supervised cells)
  * bomb logits          BCE
  * NPC next-round occ.  BCE
  * opponent next-round  BCE
  * time-since-observed  MSE            (bookkeeping channel)
  * snapshot consistency Huber          (per region: predicted fog gold +
                                         known visible gold vs gold_remaining;
                                         ONLY fog cells get gradient)

Usage:
  python3 ml/train_phase_a.py --data 'logs/*.npz' --steps 5000 --out ckpt/
  python3 ml/train_phase_a.py --smoke        # 2 steps on synthetic data

Blocked on real data: until the official logs (/share/data.tar.gz + match
downloads) are converted to the npz contract in ml/dataset.py, --data can
only point at our own simulator's dumps, i.e. the belief learns OUR guesses
about spawns/NPCs/bombs. See docs/KEV03.md section 7.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time


def _require_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        sys.stderr.write(
            "[kev03] PyTorch is not installed — train_phase_a.py cannot run.\n"
            "        Install with:  pip install -r ml/requirements.txt\n"
            "        (Deployment does not need torch; see ml/csrc/.)\n")
        return False


if not _require_torch():
    raise SystemExit(2)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds  # noqa: E402
from model import (C_BELIEF, C_OBS, GRID, GoldRushNet,  # noqa: E402
                   get_interpretable_channels)

N_REGIONS = ds.N_REGIONS

LOSS_WEIGHTS = {
    "obstacle": 1.0,
    "gold_nll": 1.0,
    "bomb": 1.0,
    "npc_next": 0.5,
    "opp_next": 0.5,
    "time_since": 0.25,
    "snapshot": 0.5,
}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def masked_bce(logit, target, mask):
    """logit/target/mask [B, 17, 17] -> scalar (mean over mask)."""
    per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (per * mask).sum() / mask.sum().clamp(min=1.0)


def gaussian_nll(mean, logvar, target, mask):
    """Heteroscedastic NLL in the log1p-gold domain, mean over mask."""
    logvar = logvar.clamp(-6.0, 6.0)
    per = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    return (per * mask).sum() / mask.sum().clamp(min=1.0)


def snapshot_consistency(gold_mean_log, fog, gold_visible_raw,
                         snap_remaining, snap_valid, region_map):
    """Huber(region prediction, snapshot gold_remaining), fog-only gradient.

    gold_mean_log    [B, 17, 17]  predicted E[log1p(gold)] at this round
    fog              [B, 17, 17]  1 under fog
    gold_visible_raw [B, 17, 17]  actual raw gold on visible cells (no grad —
                                  exactly known, masked out of the gradient
                                  path by construction: it is input data)
    snap_remaining   [B, 5]       snapshot gold_remaining per region
    snap_valid       [B]          1 when this round carries a snapshot
    region_map       [17, 17]     long, ids 1..5
    """
    if snap_valid.sum() < 1:
        return gold_mean_log.sum() * 0.0
    pred_raw = torch.expm1(gold_mean_log.clamp(-2.0, 8.0))   # raw-gold domain
    pred_fog = pred_raw * fog                                # grad ONLY on fog
    vis = gold_visible_raw.detach()                          # constants
    losses = []
    for r in range(1, N_REGIONS + 1):
        m = (region_map == r).float()                        # [17, 17]
        region_pred = (pred_fog * m).sum(dim=(-1, -2)) + \
                      (vis * m).sum(dim=(-1, -2))            # [B]
        target = snap_remaining[:, r - 1]                    # [B]
        per = F.huber_loss(region_pred, target, delta=10.0, reduction="none")
        losses.append(per * snap_valid)
    return torch.stack(losses).sum() / (snap_valid.sum() * N_REGIONS).clamp(min=1.0)


# ---------------------------------------------------------------------------
# One BPTT window
# ---------------------------------------------------------------------------

def rollout_losses(net: GoldRushNet, batch: dict, device) -> dict:
    """batch: numpy dict from dataset.sample_batch (or synthetic).

    Rolls belief over T steps and accumulates all phase-A losses.
    Returns dict of scalar tensors incl. 'total'.
    """
    tt = lambda k, dt=torch.float32: torch.as_tensor(batch[k], dtype=dt, device=device)
    obs = tt("obs")                          # [B, T, C_OBS, 17, 17]
    B, T = obs.shape[:2]
    region_map = torch.as_tensor(batch["region_map"], dtype=torch.long,
                                 device=device)                     # [17, 17]
    belief = net.initial_belief(B, device)
    acc = {k: obs.new_zeros(()) for k in LOSS_WEIGHTS}
    for t in range(T):
        belief = net.belief_step(obs[:, t], belief)
        aux = net.aux(belief)
        m = tt("sup_mask")[:, t]
        acc["obstacle"] = acc["obstacle"] + masked_bce(
            aux["obstacle_logit"], tt("obstacle")[:, t], m)
        acc["gold_nll"] = acc["gold_nll"] + gaussian_nll(
            aux["gold_mean_log"], aux["gold_logvar"], tt("gold_log")[:, t], m)
        acc["bomb"] = acc["bomb"] + masked_bce(
            aux["bomb_logit"], tt("bomb")[:, t], m)
        acc["npc_next"] = acc["npc_next"] + masked_bce(
            aux["npc_logit"], tt("npc_next")[:, t], m)
        acc["opp_next"] = acc["opp_next"] + masked_bce(
            aux["opp_logit"], tt("opp_next")[:, t], m)
        acc["time_since"] = acc["time_since"] + F.mse_loss(
            aux["time_since"], tt("time_since")[:, t])
        acc["snapshot"] = acc["snapshot"] + snapshot_consistency(
            aux["gold_mean_log"], tt("fog")[:, t], tt("gold_visible_raw")[:, t],
            tt("snap_remaining")[:, t], tt("snap_valid")[:, t], region_map)
    total = obs.new_zeros(())
    out = {}
    for k, w in LOSS_WEIGHTS.items():
        out[k] = acc[k] / T
        total = total + w * out[k]
    out["total"] = total
    return out


# ---------------------------------------------------------------------------
# Synthetic batches (--smoke)
# ---------------------------------------------------------------------------

def synthetic_batch(B: int = 2, T: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    g17 = (GRID, GRID)
    batch = {
        "obs": rng.random((B, T, C_OBS, *g17), dtype=np.float32),
        "obstacle": (rng.random((B, T, *g17)) < 0.1).astype(np.float32),
        "bomb": (rng.random((B, T, *g17)) < 0.05).astype(np.float32),
        "gold_log": rng.random((B, T, *g17), dtype=np.float32) * 4.0,
        "npc_next": (rng.random((B, T, *g17)) < 0.03).astype(np.float32),
        "opp_next": (rng.random((B, T, *g17)) < 0.01).astype(np.float32),
        "time_since": rng.random((B, T, *g17), dtype=np.float32),
        "sup_mask": np.ones((B, T, *g17), np.float32),
        "fog": (rng.random((B, T, *g17)) < 0.7).astype(np.float32),
        "gold_visible_raw": (rng.random((B, T, *g17), dtype=np.float32) * 20),
        "snap_valid": (np.arange(T)[None, :].repeat(B, 0) % 5 == 0).astype(np.float32),
        "snap_remaining": rng.random((B, T, N_REGIONS), dtype=np.float32) * 100,
        "region_map": ds.build_region_map().astype(np.int64),
    }
    return batch


def run_smoke() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = GoldRushNet(with_critic=False).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(2):
        batch = synthetic_batch(seed=step)
        losses = rollout_losses(net, batch, device)
        assert losses["total"].shape == (), "total loss must be scalar"
        for k, v in losses.items():
            assert torch.isfinite(v).all(), f"non-finite loss '{k}': {v}"
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        print(f"[kev03][smoke] step {step}: " +
              " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    # shape assertions on the belief pathway
    belief = net.initial_belief(2, device)
    belief = net.belief_step(torch.randn(2, C_OBS, GRID, GRID), belief)
    assert belief.shape == (2, C_BELIEF, GRID, GRID)
    assert set(net.aux(belief)) == set(get_interpretable_channels())
    print("[kev03][smoke] phase A smoke PASSED (2 optimizer steps, "
          "all losses finite, shapes OK)")
    return 0


# ---------------------------------------------------------------------------
# Real training loop
# ---------------------------------------------------------------------------

def run_training(args) -> int:
    paths = sorted(glob.glob(args.data))
    if not paths:
        sys.stderr.write(f"[kev03] no .npz files match --data '{args.data}'.\n"
                         "        Phase A is blocked on simulator dumps / "
                         "official logs (docs/KEV03.md sec. 7).\n")
        return 1
    device = torch.device(args.device)
    data = ds.TrajectoryDataset(paths, window=args.bptt, seed=args.seed)
    torch.manual_seed(args.seed)
    net = GoldRushNet(with_critic=False).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    for step in range(1, args.steps + 1):
        batch = data.sample_batch(args.batch)
        losses = rollout_losses(net, batch, device)
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"[kev03][A] step {step}/{args.steps} ({time.time()-t_start:.0f}s) {msg}")
        if step % args.ckpt_every == 0 or step == args.steps:
            path = os.path.join(args.out, f"phase_a_{step:07d}.pt")
            torch.save({"step": step, "model": net.state_dict(),
                        "optim": opt.state_dict()}, path)
            print(f"[kev03][A] checkpoint -> {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="", help="glob of trajectory .npz files")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--bptt", type=int, default=16, help="belief rollout length T")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="ml/ckpt", help="checkpoint directory")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic 2-step self-test, exit 0 on success")
    args = ap.parse_args(argv)
    if args.smoke:
        return run_smoke()
    if not args.data:
        ap.error("--data is required unless --smoke")
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())

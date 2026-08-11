"""kev03 Phase C — imitation (behavior cloning) of a scripted teacher.

The teacher is our deterministic bot (src/player.cpp) played inside the
simulator; its per-round decision [m1..m6, k, order, vp] is stored in the
npz `actions` array (contract in ml/dataset.py).  We minimize cross-entropy
over the SAME autoregressive factorization the actor samples with:

    L = CE(order) + CE(k) + sum_i CE(m_i) + CE(vp)

with the actor teacher-forced (each component's logits are conditioned on
the TEACHER's previous components, not on its own samples).  The belief
ConvGRU is rolled over the window exactly as at inference, so BC gradients
also shape the belief map; add --aux-weight to co-train phase-A map losses
and keep the interpretable channels grounded.

Usage:
  python3 ml/train_imitation.py --data 'logs/*.npz' --steps 20000 --out ckpt/
  python3 ml/train_imitation.py --init ckpt/phase_a_0005000.pt ...
  python3 ml/train_imitation.py --smoke     # 2 steps on synthetic data
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
            "[kev03] PyTorch is not installed — train_imitation.py cannot run.\n"
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
import train_phase_a as phase_a  # noqa: E402  (reuses aux losses)
from model import (C_OBS, GRID, N_MOVES, GoldRushNet)  # noqa: E402


def teacher_dict(teacher_t: torch.Tensor) -> dict:
    """teacher_t [B, 9] int64 -> actor-forcing dict (clamped to valid ranges)."""
    return {
        "moves": teacher_t[:, 0:6].clamp(0, 4),
        "k": teacher_t[:, 6].clamp(0, 6),
        "order": teacher_t[:, 7].clamp(0, 1),
        "vp": teacher_t[:, 8].clamp(0, 2),
    }


def bc_step_loss(actor_out: dict, teacher: dict) -> dict:
    """Cross-entropy of every factor vs the teacher action.  Scalars."""
    lg = actor_out["logits"]
    out = {
        "order": F.cross_entropy(lg["order"], teacher["order"]),
        "k": F.cross_entropy(lg["k"], teacher["k"]),
        "vp": F.cross_entropy(lg["vp"], teacher["vp"]),
        "moves": F.cross_entropy(
            lg["moves"].reshape(-1, lg["moves"].shape[-1]),   # [B*6, 5]
            teacher["moves"].reshape(-1)),
    }
    out["total"] = out["order"] + out["k"] + out["moves"] * N_MOVES + out["vp"]
    return out


def rollout_bc(net: GoldRushNet, batch: dict, device, aux_weight: float = 0.0):
    """Roll belief over the window, teacher-force the actor each step."""
    tt = lambda k, dt=torch.float32: torch.as_tensor(batch[k], dtype=dt, device=device)
    obs = tt("obs")                                  # [B, T, C_OBS, 17, 17]
    unit_pos = tt("unit_pos", torch.long)            # [B, T, 2, 2]
    unit_gold = tt("unit_gold")                      # [B, T, 2]
    teacher = tt("teacher", torch.long)              # [B, T, 9]
    B, T = obs.shape[:2]
    belief = net.initial_belief(B, device)
    acc = {k: obs.new_zeros(()) for k in ("order", "k", "moves", "vp", "total")}
    n_correct = obs.new_zeros(())
    for t in range(T):
        belief = net.belief_step(obs[:, t], belief)
        forced = teacher_dict(teacher[:, t])
        actor_out = net.actor(belief, unit_pos[:, t], unit_gold[:, t],
                              actions=forced, legality_fn=None)
        losses = bc_step_loss(actor_out, forced)
        for k in acc:
            acc[k] = acc[k] + losses[k]
        pred = actor_out["logits"]["moves"].argmax(-1)       # [B, 6]
        n_correct = n_correct + (pred == forced["moves"]).float().mean()
    out = {k: v / T for k, v in acc.items()}
    out["move_acc"] = n_correct / T
    if aux_weight > 0.0:
        aux_losses = phase_a.rollout_losses(net, batch, device)
        out["aux"] = aux_losses["total"]
        out["total"] = out["total"] + aux_weight * out["aux"]
    return out


# ---------------------------------------------------------------------------
# Synthetic batches (--smoke)
# ---------------------------------------------------------------------------

def synthetic_batch(B: int = 2, T: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    teacher = np.zeros((B, T, 9), np.int64)
    teacher[..., 0:6] = rng.integers(0, 5, (B, T, 6))
    teacher[..., 6] = rng.integers(0, 7, (B, T))
    teacher[..., 7] = rng.integers(0, 2, (B, T))
    teacher[..., 8] = rng.integers(0, 3, (B, T))
    return {
        "obs": rng.random((B, T, C_OBS, GRID, GRID), dtype=np.float32),
        "unit_pos": rng.integers(0, GRID, (B, T, 2, 2)),
        "unit_gold": rng.integers(0, 80, (B, T, 2)).astype(np.float32),
        "teacher": teacher,
    }


def run_smoke() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = GoldRushNet(with_critic=False).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(2):
        batch = synthetic_batch(seed=step)
        losses = rollout_bc(net, batch, device, aux_weight=0.0)
        assert losses["total"].shape == ()
        for k, v in losses.items():
            assert torch.isfinite(v).all(), f"non-finite '{k}': {v}"
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        print(f"[kev03][smoke] step {step}: " +
              " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    # sampled (non-forced) path must emit a full legal-range GameOutput
    b = synthetic_batch(seed=9)
    belief = net.initial_belief(2, device)
    belief = net.belief_step(torch.as_tensor(b["obs"][:, 0], device=device), belief)
    out = net.actor(belief,
                    torch.as_tensor(b["unit_pos"][:, 0], device=device),
                    torch.as_tensor(b["unit_gold"][:, 0], device=device),
                    actions=None, sample=True)
    a = out["actions"]
    assert a["moves"].shape == (2, 6) and a["moves"].min() >= 0 and a["moves"].max() <= 4
    assert a["k"].min() >= 0 and a["k"].max() <= 6
    assert a["order"].min() >= 0 and a["order"].max() <= 1
    assert a["vp"].min() >= 0 and a["vp"].max() <= 2
    print("[kev03][smoke] imitation smoke PASSED (2 optimizer steps, "
          "all losses finite, sampled action ranges legal)")
    return 0


# ---------------------------------------------------------------------------
# Real training loop
# ---------------------------------------------------------------------------

def run_training(args) -> int:
    paths = sorted(glob.glob(args.data))
    if not paths:
        sys.stderr.write(f"[kev03] no .npz files match --data '{args.data}'.\n"
                         "        Generate teacher logs by running the "
                         "deterministic bot in the simulator first.\n")
        return 1
    device = torch.device(args.device)
    data = ds.TrajectoryDataset(paths, window=args.bptt, seed=args.seed)
    torch.manual_seed(args.seed)
    net = GoldRushNet(with_critic=False).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device)
        missing, unexpected = net.load_state_dict(ck["model"], strict=False)
        print(f"[kev03][C] init from {args.init} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    for step in range(1, args.steps + 1):
        batch = data.sample_batch(args.batch)
        losses = rollout_bc(net, batch, device, aux_weight=args.aux_weight)
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"[kev03][C] step {step}/{args.steps} ({time.time()-t_start:.0f}s) {msg}")
        if step % args.ckpt_every == 0 or step == args.steps:
            path = os.path.join(args.out, f"imitation_{step:07d}.pt")
            torch.save({"step": step, "model": net.state_dict(),
                        "optim": opt.state_dict()}, path)
            print(f"[kev03][C] checkpoint -> {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="", help="glob of teacher .npz files")
    ap.add_argument("--init", default="", help="phase-A checkpoint to start from")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--bptt", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="co-train phase-A map losses (keeps belief grounded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="ml/ckpt")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
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

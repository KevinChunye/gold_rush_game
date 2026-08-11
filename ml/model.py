"""kev03 model zoo — neural belief-map agent for GoldRush 2.0.

The game (see /src/game_api.h and /README.md) is a 17x17 POMDP: each round we
see a fog-masked grid (5x5/7x7/9x9 around our two units), and every 5 rounds a
coarse per-region Snapshot. kev03 maintains a *neural belief map*

    B_t  in  R^[C_BELIEF x 17 x 17],   C_BELIEF = C_INTERP + C_LATENT

updated each round by a ConvGRU:

    B_t = ConvGRU(B_{t-1}, encode(O_t))

The first C_INTERP channels are interpretable BY CONSTRUCTION: the auxiliary
heads are (per-channel affine) reads of those exact channels, so the training
gradient forces the ConvGRU to write, e.g., the obstacle log-odds into
channel 0.  The remaining C_LATENT channels are free capacity.

Modules
-------
  ObsEncoder        C_OBS input planes  -> C_ENC feature planes
  ConvGRUCell       PyTorch-GRU gate math (r, z, n) with 3x3 convs
  BeliefCore        wraps the ConvGRUCell; owns initial_belief()
  AuxHeads          interpretable-channel reads + affine scales
  UnitTokenActor    2 unit tokens, cross-attention over the belief map,
                    autoregressive joint action: order -> k -> m1..m6 -> vp
  PrivilegedCritic  TRAINING ONLY — sees the full unmasked simulator state
  GoldRushNet       wrapper bundling all of the above

Input plane layout (built by ml/dataset.py:build_observation_planes) —
keep the two files in sync:

  idx  plane
  0    fog mask                (1 where grid == -5)
  1    obstacle (visible)      (grid == -1)
  2    bomb (visible)          (grid == -3)
  3    empty (visible)         (grid == 0)
  4    gold present (visible)  (grid >= 1)
  5    log1p(gold) / 4.0       (visible gold amount)
  6    my unit 0 position      (one-hot plane)
  7    my unit 1 position      (one-hot plane)
  8    my unit 0 gold          (broadcast scalar, log1p/6)
  9    my unit 1 gold          (broadcast scalar, log1p/6)
  10   visible enemies         (one-hot-ish, from packed visible_enemies)
  11   visible NPCs            (count per cell / 3, clipped to 1)
  12   round / 500             (broadcast scalar)
  13   snapshot: log1p(gold_remaining of cell's region) / 6      \
  14   snapshot: log1p(gold_generated of cell's region) / 6       |  masked by
  15   snapshot: log1p(gold_collected of cell's region) / 6       |  region-id
  16   snapshot: (enter - leave) of cell's region / 20, clipped   |  geometry
  17   snapshot: occupants of cell's region / 10                 /

Snapshot planes are zero until the first snapshot arrives, then hold the most
recent snapshot's values (the round plane lets the net infer staleness).  The
region geometry is a *pluggable* function (dataset.region_id_default) — the
default 5-region layout is a GUESS pending the official logs.

Every tensor comment uses B=batch, T=time, H=W=17.
"""

from __future__ import annotations

import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised on torch-less machines
    raise SystemExit(
        "[kev03] PyTorch is not installed in this environment.\n"
        "        ml/model.py (and everything importing it) needs torch for "
        "training-side work.\n"
        "        Install with:  pip install -r ml/requirements.txt\n"
        "        (The deployed bot does NOT need torch — inference is the "
        "dependency-free C++ engine in ml/csrc/.)"
    )

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

GRID = 17                      # board side
N_CELLS = GRID * GRID          # 289
C_INTERP = 7                   # interpretable belief channels (see below)
C_LATENT = 32                  # learned latent belief channels
C_BELIEF = C_INTERP + C_LATENT # 39
C_OBS = 18                     # observation input planes (layout in docstring)
C_ENC = 32                     # ObsEncoder output channels
C_FULL = 12                    # PrivilegedCritic input planes (dataset.py)
D_TOKEN = 64                   # unit-token / attention width
N_ATTN_HEADS = 4
N_ATTN_ROUNDS = 2              # cross-attention refinement rounds
N_MOVES = 6                    # moves per round (S in game_api.h)
N_MOVE_CHOICES = 5             # 0=up 1=down 2=left 3=right 4=stay
N_K = 7                        # split point in [0, 6]
N_ORDER = 2
N_VP = 3                       # vision purchase 0/1/2

# (drow, dcol) for moves 0..4 — MUST match src/constants.h / the engine.
MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


def get_interpretable_channels() -> dict:
    """Name -> channel index into B_t for the by-construction channels.

    obstacle_logit / bomb_logit / npc_logit / opp_logit are log-odds (after
    AuxHeads' affine); gold_mean_log is the predicted E[log1p(gold)];
    gold_logvar the log-variance in the same log1p domain; time_since is the
    predicted rounds-since-last-observed / 500.
    """
    return {
        "obstacle_logit": 0,
        "gold_mean_log": 1,
        "gold_logvar": 2,
        "bomb_logit": 3,
        "npc_logit": 4,       # next-round NPC occupancy
        "opp_logit": 5,       # next-round opponent occupancy
        "time_since": 6,
    }


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

class ObsEncoder(nn.Module):
    """[B, C_OBS, 17, 17] -> [B, C_ENC, 17, 17].

    Two 3x3 same-padding convs.  Deliberately shallow: the deployed budget is
    ~2M MACs total (see docs/KEV03.md section 7), and the ConvGRU eats most
    of it.  Widths are config args so the shipped net can shrink to 16-24ch.
    """

    def __init__(self, c_in: int = C_OBS, c_out: int = C_ENC):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(obs))          # [B, C_ENC, 17, 17]
        return F.relu(self.conv2(x))         # [B, C_ENC, 17, 17]


# ---------------------------------------------------------------------------
# ConvGRU
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell with PyTorch nn.GRUCell gate math.

    Gate order in the stacked 3*H channel dim is (r, z, n) — the C++ engine
    (ml/csrc/nn_infer.cpp) mirrors this exactly; change both or neither.

        r  = sigmoid(Wir * x + Whr * h)
        z  = sigmoid(Wiz * x + Whz * h)
        n  = tanh  (Win * x + r ⊙ (Whn * h))
        h' = (1 - z) ⊙ n + z ⊙ h

    h stays in (-1, 1) elementwise when initialized at 0, which is why
    AuxHeads applies a learnable affine before treating channels as logits.
    """

    def __init__(self, c_in: int, c_hidden: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.c_hidden = c_hidden
        self.conv_ih = nn.Conv2d(c_in, 3 * c_hidden, kernel, padding=pad)
        self.conv_hh = nn.Conv2d(c_hidden, 3 * c_hidden, kernel, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, c_in, 17, 17], h: [B, c_hidden, 17, 17]
        i_r, i_z, i_n = self.conv_ih(x).chunk(3, dim=1)
        h_r, h_z, h_n = self.conv_hh(h).chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h         # [B, c_hidden, 17, 17]


class BeliefCore(nn.Module):
    """Owns the recurrent belief map: B_t = ConvGRU(B_{t-1}, encode(O_t))."""

    def __init__(self, c_enc: int = C_ENC, c_belief: int = C_BELIEF):
        super().__init__()
        self.c_belief = c_belief
        self.cell = ConvGRUCell(c_enc, c_belief)

    def initial_belief(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.c_belief, GRID, GRID, device=device)

    def forward(self, enc_obs: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        # enc_obs: [B, C_ENC, 17, 17], belief: [B, C_BELIEF, 17, 17]
        return self.cell(enc_obs, belief)    # [B, C_BELIEF, 17, 17]


# ---------------------------------------------------------------------------
# Auxiliary heads (interpretable by construction)
# ---------------------------------------------------------------------------

class AuxHeads(nn.Module):
    """Reads the first C_INTERP channels of B_t through a per-channel affine.

    The GRU hidden state lives in (-1, 1); a learnable scale (init 4.0) and
    bias let the channel express confident log-odds (±4 ≈ p 0.982) while the
    channel itself remains a monotone view of the prediction — that is what
    makes it interpretable AND supervisable at once.  No conv parameters:
    the aux gradient goes straight into the ConvGRU's channels.
    """

    def __init__(self, c_interp: int = C_INTERP):
        super().__init__()
        self.scale = nn.Parameter(torch.full((c_interp,), 4.0))
        self.bias = nn.Parameter(torch.zeros(c_interp))

    def forward(self, belief: torch.Tensor) -> dict:
        # belief: [B, C_BELIEF, 17, 17]
        ch = get_interpretable_channels()
        s, b = self.scale, self.bias
        out = {}
        for name, i in ch.items():
            out[name] = belief[:, i] * s[i] + b[i]      # each [B, 17, 17]
        return out


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

def make_obstacle_legality_fn(obstacle_map: torch.Tensor):
    """Build the default legality-masking hook.

    obstacle_map: [B, 17, 17] bool — True where a KNOWN obstacle sits (from
    the persistent obstacle memory / belief threshold; fog is NOT an
    obstacle).  Returns fn(logits [B,5], pos [B,2]) -> masked logits with
    -inf on moves that leave the board or enter a known obstacle.  'stay'
    (4) is always legal, so the distribution never becomes empty.
    """
    B = obstacle_map.shape[0]

    def fn(logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(logits)                  # [B, 5]
        for m, (dr, dc) in enumerate(MOVE_DELTAS[:4]):
            nr = pos[:, 0] + dr                          # [B]
            nc = pos[:, 1] + dc
            off = (nr < 0) | (nr >= GRID) | (nc < 0) | (nc >= GRID)
            nr_c = nr.clamp(0, GRID - 1)
            nc_c = nc.clamp(0, GRID - 1)
            blocked = obstacle_map[torch.arange(B, device=logits.device), nr_c, nc_c]
            mask[:, m] = torch.where(off | blocked,
                                     torch.full_like(mask[:, m], float("-inf")),
                                     torch.zeros_like(mask[:, m]))
        return logits + mask

    return fn


class UnitTokenActor(nn.Module):
    """Two unit tokens + cross-attention over B_t + autoregressive joint head.

    Factorization (each component conditioned on all previous ones):

        p(a) = p(order) p(k | order) prod_i p(m_i | order, k, m_<i) p(vp | ...)

    order/k/moves match GameOutput; vp (vision purchase, 3-way) is appended
    to the chain so the actor emits a complete GameOutput — the team-lead
    spec lists order -> k -> moves, vp is the natural tail and is present in
    the npz teacher actions (9 ints per player).

    Move i belongs to unit 0 when i < k else unit 1; its logits are
    conditioned on the *owner's* token and masked by the legality hook using
    the owner's simulated position (tracked as moves are decoded, matching
    the engine's skip-illegal semantics for known cells).
    """

    def __init__(self, c_belief: int = C_BELIEF, d: int = D_TOKEN):
        super().__init__()
        self.d = d
        # --- token seeding: belief@pos (C_BELIEF) + log-gold (1) + one-hot idx (2)
        self.seed = nn.Linear(c_belief + 1 + 2, d)
        # --- cross-attention over the flattened belief map
        self.kv_proj = nn.Conv2d(c_belief, d, kernel_size=1)
        self.pos_emb = nn.Parameter(torch.randn(N_CELLS, d) * 0.02)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d, N_ATTN_HEADS, batch_first=True)
            for _ in range(N_ATTN_ROUNDS))
        self.ln_q = nn.ModuleList(nn.LayerNorm(d) for _ in range(N_ATTN_ROUNDS))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(),
                          nn.Linear(2 * d, d))
            for _ in range(N_ATTN_ROUNDS))
        # --- heads
        self.order_head = nn.Linear(3 * d, N_ORDER)
        self.order_emb = nn.Embedding(N_ORDER, d)
        self.k_head = nn.Linear(4 * d, N_K)
        self.k_emb = nn.Embedding(N_K, d)
        self.move_bos = nn.Parameter(torch.zeros(d))
        self.move_emb = nn.Embedding(N_MOVE_CHOICES, d)
        self.step_emb = nn.Embedding(N_MOVES, d)
        self.move_in = nn.Linear(3 * d, d)   # prev-move + step + owner token
        self.move_gru = nn.GRUCell(d, d)
        self.move_head = nn.Linear(2 * d, N_MOVE_CHOICES)  # state + owner token
        self.h0_proj = nn.Linear(3 * d, d)   # ctx + order_e + k_e -> h0
        self.vp_head = nn.Linear(5 * d, N_VP)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _gather_belief_at(belief: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """belief [B, C, 17, 17], pos [B, 2, 2] int -> [B, 2, C]."""
        B, C, _, _ = belief.shape
        flat = belief.flatten(2)                             # [B, C, 289]
        idx = (pos[..., 0] * GRID + pos[..., 1]).clamp(0, N_CELLS - 1)  # [B, 2]
        idx = idx.unsqueeze(1).expand(B, C, 2)               # [B, C, 2]
        return flat.gather(2, idx).transpose(1, 2)           # [B, 2, C]

    def _pick(self, logits, given, sample):
        """Sample (or take given) + accumulate log-prob/entropy."""
        logp = F.log_softmax(logits, dim=-1)
        if given is not None:
            a = given.long()
        elif sample:
            a = torch.multinomial(logp.exp(), 1).squeeze(-1)
        else:
            a = logits.argmax(dim=-1)
        lp = logp.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1)
        return a, lp, ent

    # -- forward ----------------------------------------------------------
    def forward(self, belief, unit_pos, unit_gold,
                actions: dict | None = None, legality_fn=None, sample: bool = True):
        """One decision.

        belief    [B, C_BELIEF, 17, 17]
        unit_pos  [B, 2, 2] long   (row, col) of my two units
        unit_gold [B, 2] float     raw gold held (normalized internally)
        actions   optional teacher dict {order [B], k [B], moves [B,6], vp [B]}
                  -> teacher-forced conditioning (behavior cloning / PPO eval)
        legality_fn  optional fn(logits [B,5], pos [B,2]) -> masked logits
        sample    sample (True) vs argmax (False) when actions is None

        Returns dict:
          logits   {order [B,2], k [B,7], moves [B,6,5], vp [B,3]}
          actions  {order [B], k [B], moves [B,6], vp [B]}
          log_prob [B]   (joint, sum of chosen-component log-probs)
          entropy  [B]   (sum of component entropies)
        """
        B = belief.shape[0]
        dev = belief.device
        given = actions or {}

        # ---- unit tokens -------------------------------------------------
        feats = self._gather_belief_at(belief, unit_pos)     # [B, 2, C_BELIEF]
        goldn = torch.log1p(unit_gold.float().clamp(min=0)).unsqueeze(-1) / 6.0
        idx_oh = torch.eye(2, device=dev).unsqueeze(0).expand(B, 2, 2)
        tok = self.seed(torch.cat([feats, goldn, idx_oh], dim=-1))   # [B, 2, D]

        kv = self.kv_proj(belief).flatten(2).transpose(1, 2)  # [B, 289, D]
        kv = kv + self.pos_emb.unsqueeze(0)
        for i in range(N_ATTN_ROUNDS):
            q = self.ln_q[i](tok)
            att, _ = self.attn[i](q, kv, kv, need_weights=False)
            tok = tok + att
            tok = tok + self.ffn[i](tok)                      # [B, 2, D]
        t0, t1 = tok[:, 0], tok[:, 1]                         # [B, D] each
        ctx = kv.mean(dim=1)                                  # [B, D]

        logits, acts = {}, {}
        # ---- order -------------------------------------------------------
        logits["order"] = self.order_head(torch.cat([ctx, t0, t1], -1))
        acts["order"], lp, ent = self._pick(logits["order"], given.get("order"), sample)
        log_prob, entropy = lp, ent
        order_e = self.order_emb(acts["order"])               # [B, D]

        # ---- k -----------------------------------------------------------
        logits["k"] = self.k_head(torch.cat([ctx, t0, t1, order_e], -1))
        acts["k"], lp, ent = self._pick(logits["k"], given.get("k"), sample)
        log_prob, entropy = log_prob + lp, entropy + ent
        k_e = self.k_emb(acts["k"])                           # [B, D]

        # ---- moves m_1..m_6 (autoregressive GRU decoder) ----------------
        h = torch.tanh(self.h0_proj(torch.cat([ctx, order_e, k_e], -1)))  # [B, D]
        prev = self.move_bos.unsqueeze(0).expand(B, self.d)   # [B, D]
        sim_pos = unit_pos.clone().long()                     # [B, 2, 2] tracked
        move_logits, move_acts = [], []
        given_moves = given.get("moves")                      # [B, 6] or None
        for i in range(N_MOVES):
            owner = (torch.full((B,), i, device=dev) >= acts["k"]).long()  # [B]
            own_tok = torch.where(owner.unsqueeze(-1).bool(), t1, t0)      # [B, D]
            own_pos = sim_pos.gather(
                1, owner.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)         # [B, 2]
            step = self.step_emb(torch.full((B,), i, device=dev, dtype=torch.long))
            x = self.move_in(torch.cat([prev, step, own_tok], -1))
            h = self.move_gru(x, h)
            li = self.move_head(torch.cat([h, own_tok], -1))               # [B, 5]
            if legality_fn is not None:
                li = legality_fn(li, own_pos)
            gm = given_moves[:, i] if given_moves is not None else None
            ai, lp, ent = self._pick(li, gm, sample)
            log_prob, entropy = log_prob + lp, entropy + ent
            move_logits.append(li)
            move_acts.append(ai)
            prev = self.move_emb(ai)
            # advance the owner's simulated position (clamped to the board;
            # the engine skips illegal moves, the mask handles known cells)
            deltas = torch.tensor(MOVE_DELTAS, device=dev, dtype=torch.long)
            npos = (own_pos + deltas[ai]).clamp(0, GRID - 1)               # [B, 2]
            sim_pos = sim_pos.scatter(
                1, owner.view(B, 1, 1).expand(B, 1, 2), npos.unsqueeze(1))
        logits["moves"] = torch.stack(move_logits, dim=1)     # [B, 6, 5]
        acts["moves"] = torch.stack(move_acts, dim=1)         # [B, 6]

        # ---- vp ----------------------------------------------------------
        logits["vp"] = self.vp_head(torch.cat([ctx, t0, t1, order_e, k_e], -1))
        acts["vp"], lp, ent = self._pick(logits["vp"], given.get("vp"), sample)
        log_prob, entropy = log_prob + lp, entropy + ent

        return {"logits": logits, "actions": acts,
                "log_prob": log_prob, "entropy": entropy}


# ---------------------------------------------------------------------------
# Privileged critic (training only — NEVER shipped)
# ---------------------------------------------------------------------------

class PrivilegedCritic(nn.Module):
    """V(s) from the FULL unmasked simulator state (asymmetric actor-critic).

    Input: [B, C_FULL, 17, 17] built by dataset.build_full_state_planes —
    no fog, both players' units, all NPCs, true gold field.  Only used to
    reduce PPO variance during phase D; export_weights.py excludes it by
    default and it must never enter the submitted .so.
    """

    def __init__(self, c_in: int = C_FULL, width: int = 24):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, full_state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(full_state))       # [B, W, 17, 17]
        x = F.relu(self.conv2(x))                # [B, W, 17, 17]
        x = x.mean(dim=(2, 3))                   # [B, W] global average pool
        x = F.relu(self.fc1(x))                  # [B, 64]
        return self.fc2(x).squeeze(-1)           # [B]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class GoldRushNet(nn.Module):
    """encoder + belief core + aux heads + actor (+ optional critic)."""

    def __init__(self, with_critic: bool = True):
        super().__init__()
        self.encoder = ObsEncoder()
        self.core = BeliefCore()
        self.aux = AuxHeads()
        self.actor = UnitTokenActor()
        self.critic = PrivilegedCritic() if with_critic else None

    def initial_belief(self, batch: int, device=None) -> torch.Tensor:
        return self.core.initial_belief(batch, device)

    def belief_step(self, obs: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        """obs [B, C_OBS, 17, 17], belief [B, C_BELIEF, 17, 17] -> new belief."""
        return self.core(self.encoder(obs), belief)

    def forward(self, obs, belief, unit_pos=None, unit_gold=None,
                actions=None, legality_fn=None, sample=True,
                full_state=None):
        """One environment step.  Returns (new_belief, aux, actor_out, value).

        actor_out is None when unit_pos is None (phase-A cartography needs no
        policy); value is None unless full_state given and critic present.
        """
        belief = self.belief_step(obs, belief)
        aux = self.aux(belief)
        actor_out = None
        if unit_pos is not None:
            actor_out = self.actor(belief, unit_pos, unit_gold,
                                   actions=actions, legality_fn=legality_fn,
                                   sample=sample)
        value = None
        if full_state is not None and self.critic is not None:
            value = self.critic(full_state)
        return belief, aux, actor_out, value


# ---------------------------------------------------------------------------
# Introspection helper
# ---------------------------------------------------------------------------

def parameter_report(model: nn.Module) -> str:
    lines, total = [], 0
    for name, p in model.named_parameters():
        lines.append(f"  {name:60s} {tuple(p.shape)!s:20s} {p.numel():>9d}")
        total += p.numel()
    lines.append(f"  {'TOTAL':60s} {'':20s} {total:>9d}"
                 f"  (~{total * 4 / 1e6:.2f} MB fp32)")
    return "\n".join(lines)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = GoldRushNet(with_critic=True)
    print("[kev03] GoldRushNet parameter report:")
    print(parameter_report(net))
    B = 2
    obs = torch.randn(B, C_OBS, GRID, GRID)
    belief = net.initial_belief(B)
    pos = torch.randint(0, GRID, (B, 2, 2))
    gold = torch.randint(0, 50, (B, 2)).float()
    fs = torch.randn(B, C_FULL, GRID, GRID)
    leg = make_obstacle_legality_fn(torch.zeros(B, GRID, GRID, dtype=torch.bool))
    belief, aux, act, val = net(obs, belief, pos, gold,
                                legality_fn=leg, full_state=fs)
    assert belief.shape == (B, C_BELIEF, GRID, GRID)
    assert aux["obstacle_logit"].shape == (B, GRID, GRID)
    assert act["logits"]["moves"].shape == (B, N_MOVES, N_MOVE_CHOICES)
    assert val.shape == (B,)
    print("[kev03] forward smoke OK "
          f"(belief {tuple(belief.shape)}, joint log_prob {tuple(act['log_prob'].shape)})")

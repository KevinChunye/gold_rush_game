"""kev03 trajectory container + fog-masking utilities (numpy only, no torch).

===========================================================================
NPZ TRAJECTORY CONTRACT (simulator team: please match this exactly)
===========================================================================
One .npz file per match, T = number of recorded rounds (<= 500), containing:

  board_terrain  int  [T, 17, 17]  static terrain + bombs as the engine codes
                                   them: -1 obstacle, -3 bomb, 0 empty.
                                   NO fog (-5) and NO gold here — this is the
                                   FULL simulator state; gold is separate.
  gold           int  [T, 17, 17]  gold amount per cell, >= 0.
  units          int  [T, 4, 2]    (row, col); units 0,1 = player 0's units,
                                   units 2,3 = player 1's units. (-1,-1) if
                                   a unit does not exist (should not happen).
  unit_gold      int  [T, 4]       gold held by each of the 4 units.
  npcs           int  [T, 7, 2]    NPC positions, (-1,-1) for absent slots.
                                   Slot index == stable NPC id - 1 if
                                   possible (ids are stable across rounds).
  actions        int  [T, 2, 9]    per player p: actions[t, p] =
                                   [m1..m6, k, order, vp] — the 6 moves in
                                   [0,4], split k in [0,6], order bit, and
                                   vision purchase vp in {0,1,2} CHOSEN at
                                   round t (so vp affects vision at t+1).
  rewards        float [T, 2]      per-player per-round reward (e.g. gold
                                   delta); training scripts may recompute.

Optional (nice to have, used when present):
  npc_ids        int  [T, 7]       stable NPC ids, 0 for empty slots.
  latency_us     float [T, 2]      decision latency — who moved first.

All integers int32 or int64; this loader casts as needed.
===========================================================================

This module reconstructs, from the FULL state above, exactly what each
player's moveDecision() saw (fog-masked grid, visible enemies/NPCs, regional
snapshot) so the model trains on the same observation distribution as the
deployed bot.  Vision is 5x5 by default and 7x7 / 9x9 on the round AFTER a
vp=1 / vp=2 purchase (Chebyshev radius 2/3/4 around each of the player's two
units).

Region geometry is PLUGGABLE (`region_fn(row, col) -> 1..5`): the official
region layout has not been published, so `region_id_default` is our GUESS
(center 9x9 = region 5, ring quadrants NW/NE/SW/SE = 1..4). Recalibrate it
from official logs before trusting snapshot-derived features.
"""

from __future__ import annotations

import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    raise SystemExit(
        "[kev03] numpy is not installed. ml/dataset.py is numpy-only "
        "(no torch needed);  pip install numpy  (or -r ml/requirements.txt)."
    )

GRID = 17
FOG, BOMB, OBSTACLE, EMPTY = -5, -3, -1, 0
N_REGIONS = 5
SNAPSHOT_PERIOD = 5
MAX_ROUNDS = 500
VISION_RADIUS = {0: 2, 1: 3, 2: 4}          # vp bought last round -> radius
MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))

C_OBS = 18                                   # keep in sync with ml/model.py
C_FULL = 12


# ---------------------------------------------------------------------------
# Region geometry (pluggable — DEFAULT IS A GUESS)
# ---------------------------------------------------------------------------

def region_id_default(row: int, col: int) -> int:
    """GUESSED 5-region layout, to be replaced once official logs arrive.

    Center 9x9 block (rows/cols 4..12 inclusive) = region 5 (the gold-rich
    spawn area).  The remaining ring is split into quadrants at row 8/col 8:
    NW=1, NE=2, SW=3, SE=4.
    """
    if 4 <= row <= 12 and 4 <= col <= 12:
        return 5
    if row <= 8:
        return 1 if col <= 8 else 2
    return 3 if col <= 8 else 4


def build_region_map(region_fn=region_id_default) -> np.ndarray:
    """[17, 17] int32 of region ids 1..5."""
    m = np.zeros((GRID, GRID), dtype=np.int32)
    for r in range(GRID):
        for c in range(GRID):
            m[r, c] = region_fn(r, c)
    return m


# ---------------------------------------------------------------------------
# Fog masking
# ---------------------------------------------------------------------------

def vision_radius_at(actions: np.ndarray, t: int, player: int) -> int:
    """Radius in effect at round t: vp chosen at t-1 upgrades this round."""
    if t <= 0:
        return VISION_RADIUS[0]
    vp = int(actions[t - 1, player, 8])
    return VISION_RADIUS.get(vp, VISION_RADIUS[0])


def visibility_mask(unit_pos: np.ndarray, radius: int) -> np.ndarray:
    """unit_pos [2, 2] -> [17, 17] bool, Chebyshev balls around both units."""
    rows = np.arange(GRID)[:, None]
    cols = np.arange(GRID)[None, :]
    vis = np.zeros((GRID, GRID), dtype=bool)
    for u in range(unit_pos.shape[0]):
        r, c = int(unit_pos[u, 0]), int(unit_pos[u, 1])
        if r < 0 or c < 0:
            continue
        vis |= (np.abs(rows - r) <= radius) & (np.abs(cols - c) <= radius)
    return vis


def fog_mask_observation(board_terrain, gold, units, unit_gold, npcs,
                         actions, t: int, player: int) -> dict:
    """Rebuild player `player`'s GameInput-equivalent view at round t.

    Returns dict:
      grid            [17,17] int  -5 fog / -3 bomb / -1 obstacle / 0 / gold
      my_units        [2,2] int
      my_units_gold   [2] int
      gold_opp        int (opponent total — always known per the ABI)
      visible_enemies [2,2] int, packed from index 0, empty (-1,-1)
      visible_npcs    [7,2] int, packed, empty (-1,-1)
      vis             [17,17] bool (True where visible)
    """
    my = units[t, 2 * player: 2 * player + 2]              # [2, 2]
    opp = units[t, 2 * (1 - player): 2 * (1 - player) + 2] # [2, 2]
    radius = vision_radius_at(actions, t, player)
    vis = visibility_mask(my, radius)

    grid = np.full((GRID, GRID), FOG, dtype=np.int64)
    terr = board_terrain[t]
    g = gold[t]
    grid[vis] = np.where(g[vis] > 0, g[vis], terr[vis])

    enemies = np.full((2, 2), -1, dtype=np.int64)
    n = 0
    for u in range(2):
        r, c = int(opp[u, 0]), int(opp[u, 1])
        if r >= 0 and vis[r, c]:
            enemies[n] = (r, c)                            # packed, unit id hidden
            n += 1

    vnpcs = np.full((7, 2), -1, dtype=np.int64)
    n = 0
    for i in range(npcs.shape[1]):
        r, c = int(npcs[t, i, 0]), int(npcs[t, i, 1])
        if r >= 0 and vis[r, c]:
            vnpcs[n] = (r, c)
            n += 1

    return {
        "grid": grid,
        "my_units": my.astype(np.int64),
        "my_units_gold": unit_gold[t, 2 * player: 2 * player + 2].astype(np.int64),
        "gold_opp": int(unit_gold[t, 2 * (1 - player): 2 * (1 - player) + 2].sum()),
        "visible_enemies": enemies,
        "visible_npcs": vnpcs,
        "vis": vis,
    }


# ---------------------------------------------------------------------------
# Snapshot reconstruction (approximate where the logs would be exact)
# ---------------------------------------------------------------------------

def compute_snapshot(gold, units, npcs, region_map, t: int,
                     window: int = SNAPSHOT_PERIOD) -> dict:
    """Approximate the engine's Snapshot for the window ending at round t.

    gold_remaining is EXACT (sum of gold[t] per region).  gold_generated /
    gold_collected are approximated from frame-to-frame gold diffs (positive
    diffs = generated, negative = collected; a same-round spawn+pickup on
    one cell aliases — official logs will calibrate).  enter/leave count
    region transitions of all 4 units + NPCs; occupants counts them at t.
    """
    t0 = max(0, t - window + 1)
    out = {"window_begin": t0, "window_end": t, "regions": []}
    gen = np.zeros(N_REGIONS + 1)
    col = np.zeros(N_REGIONS + 1)
    ent = np.zeros(N_REGIONS + 1, dtype=np.int64)
    lea = np.zeros(N_REGIONS + 1, dtype=np.int64)

    # A round tt's events (gold diff, region transitions tt-1 -> tt) belong
    # to the window that CONTAINS tt, so consecutive windows tile with no
    # dropped boundary round. Convention is a guess pending log calibration.
    for tt in range(max(1, t0), t + 1):
        diff = gold[tt].astype(np.int64) - gold[tt - 1].astype(np.int64)
        for rgn in range(1, N_REGIONS + 1):
            m = region_map == rgn
            d = diff[m]
            gen[rgn] += d[d > 0].sum()
            col[rgn] += -d[d < 0].sum()
        # region transitions of every actor (4 units + NPCs)
        for arr in (units, npcs):
            for i in range(arr.shape[1]):
                r0, c0 = int(arr[tt - 1, i, 0]), int(arr[tt - 1, i, 1])
                r1, c1 = int(arr[tt, i, 0]), int(arr[tt, i, 1])
                if r0 < 0 or r1 < 0:
                    continue
                a, b = region_map[r0, c0], region_map[r1, c1]
                if a != b:
                    lea[a] += 1
                    ent[b] += 1

    occ = np.zeros(N_REGIONS + 1, dtype=np.int64)
    for arr in (units, npcs):
        for i in range(arr.shape[1]):
            r, c = int(arr[t, i, 0]), int(arr[t, i, 1])
            if r >= 0:
                occ[region_map[r, c]] += 1

    for rgn in range(1, N_REGIONS + 1):
        out["regions"].append({
            "id": rgn,
            "enter": int(ent[rgn]),
            "leave": int(lea[rgn]),
            "gold_generated": int(gen[rgn]),
            "gold_collected": int(col[rgn]),
            "gold_remaining": int(gold[t][region_map == rgn].sum()),
            "occupants": int(occ[rgn]),
        })
    return out


# ---------------------------------------------------------------------------
# Model input planes  (layout documented in ml/model.py — keep in sync)
# ---------------------------------------------------------------------------

def build_observation_planes(obs: dict, snapshot: dict | None, round_no: int,
                             region_map: np.ndarray) -> np.ndarray:
    """obs from fog_mask_observation -> [C_OBS, 17, 17] float32."""
    p = np.zeros((C_OBS, GRID, GRID), dtype=np.float32)
    grid = obs["grid"]
    p[0] = grid == FOG
    p[1] = grid == OBSTACLE
    p[2] = grid == BOMB
    p[3] = grid == EMPTY
    p[4] = grid >= 1
    p[5] = np.log1p(np.maximum(grid, 0)) / 4.0
    for u in range(2):
        r, c = int(obs["my_units"][u, 0]), int(obs["my_units"][u, 1])
        if r >= 0:
            p[6 + u, r, c] = 1.0
        p[8 + u] = np.log1p(max(int(obs["my_units_gold"][u]), 0)) / 6.0
    for e in range(2):
        r, c = int(obs["visible_enemies"][e, 0]), int(obs["visible_enemies"][e, 1])
        if r >= 0:
            p[10, r, c] = 1.0
    for i in range(obs["visible_npcs"].shape[0]):
        r, c = int(obs["visible_npcs"][i, 0]), int(obs["visible_npcs"][i, 1])
        if r >= 0:
            p[11, r, c] += 1.0 / 3.0
    p[11] = np.minimum(p[11], 1.0)
    p[12] = round_no / float(MAX_ROUNDS)
    if snapshot is not None:
        for rg in snapshot["regions"]:
            m = region_map == rg["id"]
            p[13][m] = np.log1p(max(rg["gold_remaining"], 0)) / 6.0
            p[14][m] = np.log1p(max(rg["gold_generated"], 0)) / 6.0
            p[15][m] = np.log1p(max(rg["gold_collected"], 0)) / 6.0
            p[16][m] = np.clip((rg["enter"] - rg["leave"]) / 20.0, -1.0, 1.0)
            p[17][m] = min(rg["occupants"] / 10.0, 1.0)
    return p


def build_full_state_planes(board_terrain, gold, units, unit_gold, npcs,
                            t: int, player: int) -> np.ndarray:
    """FULL unmasked state -> [C_FULL, 17, 17] float32 (PrivilegedCritic).

      0 obstacle  1 bomb  2 log1p(gold)/4
      3,4 my unit 0/1     5,6 opp unit 0/1     7 NPC count/3 (clipped)
      8,9 my unit 0/1 gold (log1p/6, broadcast)
      10 opp total gold (log1p/7, broadcast)   11 round/500 (broadcast)
    """
    p = np.zeros((C_FULL, GRID, GRID), dtype=np.float32)
    p[0] = board_terrain[t] == OBSTACLE
    p[1] = board_terrain[t] == BOMB
    p[2] = np.log1p(np.maximum(gold[t], 0)) / 4.0
    my = 2 * player
    op = 2 * (1 - player)
    for j, u in enumerate((my, my + 1, op, op + 1)):
        r, c = int(units[t, u, 0]), int(units[t, u, 1])
        if r >= 0:
            p[3 + j, r, c] = 1.0
    for i in range(npcs.shape[1]):
        r, c = int(npcs[t, i, 0]), int(npcs[t, i, 1])
        if r >= 0:
            p[7, r, c] += 1.0 / 3.0
    p[7] = np.minimum(p[7], 1.0)
    p[8] = np.log1p(max(int(unit_gold[t, my]), 0)) / 6.0
    p[9] = np.log1p(max(int(unit_gold[t, my + 1]), 0)) / 6.0
    p[10] = np.log1p(max(int(unit_gold[t, op] + unit_gold[t, op + 1]), 0)) / 7.0
    p[11] = t / float(MAX_ROUNDS)
    return p


# ---------------------------------------------------------------------------
# Trajectory container
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "board_terrain": ("T", GRID, GRID),
    "gold": ("T", GRID, GRID),
    "units": ("T", 4, 2),
    "unit_gold": ("T", 4),
    "npcs": ("T", 7, 2),
    "actions": ("T", 2, 9),
    "rewards": ("T", 2),
}


class Trajectory:
    """One match loaded from an .npz following the contract above."""

    def __init__(self, arrays: dict):
        T = arrays["board_terrain"].shape[0]
        for key, shape in REQUIRED_KEYS.items():
            if key not in arrays:
                raise ValueError(f"npz missing required array '{key}'")
            want = tuple(T if s == "T" else s for s in shape)
            got = tuple(arrays[key].shape)
            if got != want:
                raise ValueError(f"'{key}': shape {got}, expected {want}")
        self.T = T
        for key in REQUIRED_KEYS:
            setattr(self, key, np.asarray(arrays[key]))
        self.npc_ids = np.asarray(arrays["npc_ids"]) if "npc_ids" in arrays else None
        self.latency_us = (np.asarray(arrays["latency_us"])
                           if "latency_us" in arrays else None)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})


def occupancy_plane(positions: np.ndarray) -> np.ndarray:
    """[N, 2] positions -> [17, 17] float32 binary occupancy."""
    p = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(positions.shape[0]):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        if r >= 0:
            p[r, c] = 1.0
    return p


def make_training_window(traj: Trajectory, player: int, t0: int, T: int,
                         region_fn=region_id_default) -> dict:
    """Slice [t0, t0+T) into model inputs + phase-A/BC targets (all numpy).

    Returns dict of float32 arrays:
      obs         [T, C_OBS, 17, 17]   fog-masked model input
      full_state  [T, C_FULL, 17, 17]  privileged critic input
      unit_pos    [T, 2, 2]            my units (int64)
      unit_gold   [T, 2]
      teacher     [T, 9]               my recorded action (int64)
      reward      [T]
      obstacle    [T, 17, 17]  bomb [T,17,17]  gold_log [T,17,17]
      npc_next    [T, 17, 17]  opp_next [T,17,17]  time_since [T,17,17]
      sup_mask    [T, 17, 17]  supervision mask for the map losses (all ones
                               when training from the simulator, which knows
                               the full truth; from real logs it would be
                               'revealed at any later round')
      fog         [T, 17, 17]  1 where fog at t (this player's view)
      gold_visible_raw [T,17,17]  raw gold on visible cells, 0 under fog
      snap_valid  [T]          1 on rounds carrying a (new) snapshot
      snap_remaining [T, 5]    per-region gold_remaining on those rounds
      region_map  [17, 17]     int64 region ids 1..5
    """
    assert 0 <= t0 and t0 + T <= traj.T
    region_map = build_region_map(region_fn)
    out = {
        "obs": np.zeros((T, C_OBS, GRID, GRID), np.float32),
        "full_state": np.zeros((T, C_FULL, GRID, GRID), np.float32),
        "unit_pos": np.zeros((T, 2, 2), np.int64),
        "unit_gold": np.zeros((T, 2), np.float32),
        "teacher": np.zeros((T, 9), np.int64),
        "reward": np.zeros((T,), np.float32),
        "obstacle": np.zeros((T, GRID, GRID), np.float32),
        "bomb": np.zeros((T, GRID, GRID), np.float32),
        "gold_log": np.zeros((T, GRID, GRID), np.float32),
        "npc_next": np.zeros((T, GRID, GRID), np.float32),
        "opp_next": np.zeros((T, GRID, GRID), np.float32),
        "time_since": np.zeros((T, GRID, GRID), np.float32),
        "sup_mask": np.ones((T, GRID, GRID), np.float32),
        "fog": np.zeros((T, GRID, GRID), np.float32),
        "gold_visible_raw": np.zeros((T, GRID, GRID), np.float32),
        "snap_valid": np.zeros((T,), np.float32),
        "snap_remaining": np.zeros((T, N_REGIONS), np.float32),
        "region_map": region_map.astype(np.int64),
    }
    # time-since-observed bookkeeping starts fresh at the window start
    last_seen = np.full((GRID, GRID), -1, dtype=np.int64)
    snapshot = None
    for i in range(T):
        t = t0 + i
        obs = fog_mask_observation(traj.board_terrain, traj.gold, traj.units,
                                   traj.unit_gold, traj.npcs, traj.actions,
                                   t, player)
        last_seen[obs["vis"]] = t
        if t % SNAPSHOT_PERIOD == 0 and t > 0:
            snapshot = compute_snapshot(traj.gold, traj.units, traj.npcs,
                                        region_map, t)
            out["snap_valid"][i] = 1.0
            for j, rg in enumerate(snapshot["regions"]):
                out["snap_remaining"][i, j] = rg["gold_remaining"]
        out["obs"][i] = build_observation_planes(obs, snapshot, t, region_map)
        out["full_state"][i] = build_full_state_planes(
            traj.board_terrain, traj.gold, traj.units, traj.unit_gold,
            traj.npcs, t, player)
        out["unit_pos"][i] = obs["my_units"]
        out["unit_gold"][i] = obs["my_units_gold"]
        out["teacher"][i] = traj.actions[t, player]
        out["reward"][i] = traj.rewards[t, player]
        out["obstacle"][i] = traj.board_terrain[t] == OBSTACLE
        out["bomb"][i] = traj.board_terrain[t] == BOMB
        out["gold_log"][i] = np.log1p(np.maximum(traj.gold[t], 0))
        tn = min(t + 1, traj.T - 1)
        out["npc_next"][i] = occupancy_plane(traj.npcs[tn])
        opp = traj.units[tn, 2 * (1 - player): 2 * (1 - player) + 2]
        out["opp_next"][i] = occupancy_plane(opp)
        unseen = last_seen < 0
        age = np.where(unseen, t + 1, t - last_seen)
        out["time_since"][i] = age / float(MAX_ROUNDS)
        out["fog"][i] = obs["grid"] == FOG
        out["gold_visible_raw"][i] = np.where(obs["grid"] >= 1, obs["grid"], 0)
    return out


class TrajectoryDataset:
    """Round-robin sampler of fixed-length windows across many .npz matches."""

    def __init__(self, paths: list, window: int = 16, seed: int = 0):
        if not paths:
            raise ValueError("TrajectoryDataset: no .npz paths given")
        self.trajs = [Trajectory.load(p) for p in paths]
        self.window = window
        self.rng = np.random.default_rng(seed)

    def sample_window(self, region_fn=region_id_default) -> dict:
        traj = self.trajs[self.rng.integers(len(self.trajs))]
        player = int(self.rng.integers(2))
        t0 = int(self.rng.integers(0, max(1, traj.T - self.window + 1)))
        return make_training_window(traj, player, t0,
                                    min(self.window, traj.T), region_fn)

    def sample_batch(self, batch: int, region_fn=region_id_default) -> dict:
        ws = [self.sample_window(region_fn) for _ in range(batch)]
        out = {}
        for k in ws[0]:
            if k == "region_map":
                out[k] = ws[0][k]
            else:
                out[k] = np.stack([w[k] for w in ws])   # [B, ...]
        return out


# ---------------------------------------------------------------------------
# Self-test (numpy only — runs without torch)
# ---------------------------------------------------------------------------

def _fabricate_trajectory(T: int = 12, seed: int = 0) -> Trajectory:
    rng = np.random.default_rng(seed)
    terr = np.zeros((T, GRID, GRID), np.int64)
    terr[:, 2, 2] = OBSTACLE
    terr[:, 14, 3] = BOMB
    gold = np.zeros((T, GRID, GRID), np.int64)
    for t in range(T):
        for _ in range(6):
            gold[t, rng.integers(4, 13), rng.integers(4, 13)] = rng.integers(1, 30)
    units = rng.integers(0, GRID, (T, 4, 2))
    unit_gold = rng.integers(0, 100, (T, 4))
    npcs = rng.integers(0, GRID, (T, 7, 2))
    npcs[:, 5:] = -1
    actions = np.zeros((T, 2, 9), np.int64)
    actions[:, :, :6] = rng.integers(0, 5, (T, 2, 6))
    actions[:, :, 6] = rng.integers(0, 7, (T, 2))
    actions[:, :, 7] = rng.integers(0, 2, (T, 2))
    actions[:, :, 8] = rng.integers(0, 3, (T, 2))
    rewards = rng.random((T, 2)).astype(np.float32)
    return Trajectory({"board_terrain": terr, "gold": gold, "units": units,
                       "unit_gold": unit_gold, "npcs": npcs,
                       "actions": actions, "rewards": rewards})


def _selftest() -> None:
    rmap = build_region_map()
    assert rmap.shape == (GRID, GRID)
    assert sorted(np.unique(rmap).tolist()) == [1, 2, 3, 4, 5]
    assert rmap[8, 8] == 5 and rmap[0, 0] == 1 and rmap[0, 16] == 2
    assert rmap[16, 0] == 3 and rmap[16, 16] == 4

    traj = _fabricate_trajectory(T=12)
    w = make_training_window(traj, player=0, t0=0, T=12)
    assert w["obs"].shape == (12, C_OBS, GRID, GRID)
    assert w["full_state"].shape == (12, C_FULL, GRID, GRID)
    assert w["teacher"].shape == (12, 9)
    # fog + one-hot terrain partition every cell
    onehot = w["obs"][:, 0] + w["obs"][:, 1] + w["obs"][:, 2] + \
        w["obs"][:, 3] + w["obs"][:, 4]
    assert np.all(onehot == 1.0), "fog/terrain planes must partition the board"
    # visibility: cells at Chebyshev distance <= 2 of a unit are never fog
    t = 3
    r, c = traj.units[t, 0]
    assert w["fog"][t, r, c] == 0.0
    # snapshot rounds flagged (t=5, 10 within this window)
    assert w["snap_valid"][5] == 1.0 and w["snap_valid"][10] == 1.0
    assert w["snap_valid"][4] == 0.0
    # snapshot gold_remaining consistency: fog gold + visible gold == region sum
    snap = compute_snapshot(traj.gold, traj.units, traj.npcs, rmap, 5)
    for rg in snap["regions"]:
        m = rmap == rg["id"]
        assert rg["gold_remaining"] == traj.gold[5][m].sum()

    # dataset batch path
    ds = TrajectoryDataset.__new__(TrajectoryDataset)
    ds.trajs, ds.window, ds.rng = [traj], 8, np.random.default_rng(0)
    batch = ds.sample_batch(3)
    assert batch["obs"].shape == (3, 8, C_OBS, GRID, GRID)
    assert batch["region_map"].shape == (GRID, GRID)
    print("[kev03] dataset selftest OK "
          f"(window obs {w['obs'].shape}, batch obs {batch['obs'].shape})")


if __name__ == "__main__":
    _selftest()

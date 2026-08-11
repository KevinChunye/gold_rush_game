# GoldRush 2.0 — C++ Bot (MVP)

Minimal viable submission for the **GoldRush 2.0 编程掘金争夺赛**. The goal of
this version is not a clever algorithm — it is a bot that is *guaranteed
legal, crash-free and fast*, so we can validate the whole pipeline first:

> build `player.so` → upload to the contest site → play a match → watch the replay.

Once the pipeline is proven, we iterate on strategy inside one file
(`src/player.cpp`) without touching anything else.

---

## 1. What the competition is

Two players fight over gold on a **17×17 grid** for **500 rounds**; whoever
holds more gold at the end wins (ties go to the player with lower P90
decision latency — speed is also a separate prize for C++ teams).

Each player controls **2 units**. Each round the engine calls our
`moveDecision()` with what our units can currently see, and we answer with
**6 moves split between the two units** — within **300 ms** (overruns drain a
60 s per-match pool; an empty pool, a crash, or a malformed answer loses the
match instantly).

The interesting constraints:

| Mechanic | Public-beta value |
|---|---|
| Vision | 5×5 around each unit; rest of the grid is fog (`-5`). Can buy 7×7 (2 gold) or 9×9 (3 gold) for the *next* round |
| Gold spawns | central 9×9 every round; occasionally outside it |
| Pickup | entering a gold cell grabs **65 %** of it (ceil); re-entering grabs 65 % of the rest |
| Bombs (`-3`) | stepping on one costs **10 %** of held gold; bombs respawn every 20 rounds |
| Obstacles (`-1`) | impassable, fixed per map |
| NPCs | 7 of them, spawn at center, up to 3 steps/round, also grab gold; a cell with **≥ 3 NPCs** tramples us for 5 % of held gold |
| Global snapshot | every 5 rounds: per-region gold generated / collected / remaining, traffic |
| Round order | gold+bombs spawn → both strategies queried on the same state → faster player moves → NPCs move → slower player moves |

So the game is essentially **exploration vs. exploitation under fog of war
with a latency scoreboard**: find gold you cannot fully see, route two units
efficiently around hazards, and answer quickly.

## 2. What the bot does (v2.4 — "memory + value chains, seat-adaptive")

**v2.4 headline:** the engine executes the *faster-answering* bot's moves
first each round, then the NPCs, then the slower bot — and simulator
experiments showed the two seats want opposite strategies (details in §6).
The seat isn't in the API, but it's observable: when we move first, piles
we planned to grab are still there; when we move second they keep
vanishing. The bot audits planned-vs-realized pickups each round and
switches profile: **first seat** = aggressive (free-roaming exploration,
long gold memory, contest everything); **second seat** = defensive
(fog rationed to one priced step per round, fast center-memory decay,
contested piles discounted). Everything below describes the shared
machinery.

Everything lives in [`src/player.cpp`](src/player.cpp) (~380 lines, no STL,
no allocation, no I/O). The decision architecture is expected-value
maximization over the 6 moves, not nearest-gold greed:

1. **Belief memory across rounds** (the engine keeps the `.so` loaded):
   obstacles are remembered forever (they never move), bombs stay feared
   for ~one respawn window after sighting, and gold piles are remembered
   with a belief that decays over ~40 rounds (someone may take them while
   we're away). State auto-resets when a new match starts.
2. **Fog is traversable, priced by wealth.** Crossing unknown cells risks
   a hidden bomb (−10 % of held gold), so fog crossings are charged
   against a target's value in proportion to the unit's gold: poor units
   explore boldly, rich units stick to charted ground.
3. **Value-aware chained targeting.** Each unit repeatedly takes the best
   pickup-per-step option — a pile (65 % of believed value at BFS
   distance) or re-entering the pile underfoot (2 moves for 65 % of the
   remainder) — and keeps going until its move budget runs out; with
   nothing in sight or memory it drifts toward the center spawn area.
4. **Dynamic split & order.** Each round the 6 moves are divided by
   maximizing the two units' combined value curves over every `k` in
   [0,6]; both execution orders are evaluated; and the second mover
   re-plans around the piles the first mover will already have taken
   (65 % claim discount) and its final cell.
5. **Safety unchanged from v1**: never steps into known obstacles, bombs,
   enemies, or ≥ 3-NPC cells, and always returns a legal `GameOutput`
   (worst case: stand still). No vision purchases yet (`vp=0`).
6. **Speed**: a decision costs ~25 µs p90 (see `make test`) against the
   300 ms budget — no timeout risk, and still a strong speed-prize entry.

## 3. Repo layout

```
├── Makefile              make -> player.so (the upload artifact); make test
├── src/
│   ├── game_api.h        official contest ABI — byte-identical copy of
│   │                     reference/game_api.h (keep it that way)
│   ├── constants.h       cell values + action codes from the rules
│   └── player.cpp        the bot — the only file strategy work touches
├── test/
│   └── local_test.cpp    offline harness: dlopen()s player.so like the real
│                         engine, runs scripted scenarios, checks legality,
│                         measures p50/p90 latency
├── reference/            official 参考代码 from the organizers (game.zip):
│                         canonical game_api.h, a random-walk sample player
│                         (C++ and Python), their Makefile
├── sim/                  local match engine: dlopen()s two bot .so files,
│                         plays full 500-round rule-faithful matches (§6)
├── ml/                   kev03 neural-cartographer scaffold (§7)
└── docs/KEV03.md         kev03 design doc, phase plan, review backlog
```

## 4. Quick start

Needs Linux x86_64 with `g++` and `make` (the contest runs Linux — see the
warning below).

```bash
make        # -> player.so  (this exact file is what we upload)
make test   # -> builds + runs the harness against player.so
```

`make test` prints each scenario's board and the bot's decision, then a
latency summary, and ends with `ALL CHECKS PASSED`.

### Submitting

1. Build `player.so` **on Linux** (best: the contest dev server, which
   matches the judge machines):
   ```bash
   rsync -av --exclude .git . <account>@8.153.76.120:~/gold_rush_game/
   ssh <account>@8.153.76.120 "cd ~/gold_rush_game && make test"
   scp <account>@8.153.76.120:~/gold_rush_game/player.so .
   ```
   (SSH account comes from the registration email.)
2. Upload `player.so` (≤ 16 MB — ours is ~16 KB) on the contest site:
   **http://47.103.127.219** (plain HTTP; disable VPN/proxy if it won't load).
3. Start a match against another team's bot (500 challenges/day limit),
   then **watch the replay** on the site.

> **⚠️ Do not build on macOS.** A Mac produces a Mach-O `.dylib`, which the
> Linux engine cannot `dlopen`. Always produce the submission artifact on
> Linux.

### First-upload checklist

- [ ] Match completes all 500 rounds — no crash / format loss / timeout.
- [ ] Replay shows units moving as intended (confirms our direction mapping
      0=up/1=down/2=left/3=right and row/col orientation match the engine).
- [ ] Units route around obstacles and bombs, and collect gold.
- [ ] P90 latency on the leaderboard is microseconds-tiny.
- [x] ABI verified: `src/game_api.h` **is** the official header from the
      reference code, copied verbatim (`diff reference/game_api.h
      src/game_api.h` must stay empty). The official sample player is a
      random walker, so this bot already beats the provided baseline.

## 5. The interface in 30 seconds

Every round the engine calls:

```cpp
extern "C" GameOutput moveDecision(const GameInput* input);
```

- **In** (`GameInput`): `round`; `grid[17][17]` with `-5` fog / `-3` bomb /
  `-1` obstacle / `0` empty / `≥1` gold amount; our two unit positions and
  gold; opponent's total gold; visible enemy positions (packed from index 0,
  empty slots `(-1,-1)` — and we are *not* told which enemy unit is which);
  visible NPCs; and every 5 rounds a region `snapshot`.
- **Out** (`GameOutput`): `actions[6]` (0=up 1=down 2=left 3=right 4=stay),
  `k` (unit 0 executes `actions[0:k]`, unit 1 executes `actions[k:6]`),
  `order` (which of our units moves first), `vp` (vision purchase).
- Illegal moves (wall / off-board / other player unit) are simply skipped —
  only a *malformed* output or a crash/timeout loses the match.
- One unit executes all of its steps, then the other; the grid we receive is
  the pre-move state, and only the final position's vision comes back next
  round (no en-route vision).

## 6. Local simulator (`sim/`)

A rules-faithful headless engine that `dlopen`s two real submission `.so`
files (isolated copies, so both can export `moveDecision` and keep private
global state) and plays full 500-round matches: decision-time move
ordering at nanosecond precision like the real engine, fog/vision/vp,
pickups, bombs, trampling, NPCs, snapshots, mirrored maps. Spawn rates,
NPC policy, bomb waves and region geometry are **guesses** collected in
`SimConfig`, to be calibrated from official logs.

```bash
make -C sim
sim/simulator botA.so botB.so --games 10 --seed 42 --mirror \
    [--quantum-ns 1000000000]   # neutralize latency ordering
    [--force-first A|B]         # pin the first-moving bot
```

Key findings from the v1/v2.x experiments (mean net gold, 20 mirrored
plays, identical seeds):

| bot | as first mover | as second (vs v1) | order neutralized |
|---|---|---|---|
| v1 (nearest-visible BFS) | 1184–1252 | 324–364 | 770–779 |
| v2.0 (aggressive memory) | 1786 | 356 | 1032 |
| v2.2 (cautious memory) | 1428 | 403 | 906 |
| **v2.4 (seat-adaptive)** | **1863** | **415** | — |

Interpretation: in a mirror matchup, first-mover status dominates
strategy (the second mover collects scraps regardless); the first seat
rewards aggression, the second rewards caution — hence v2.4's adaptive
switch. The kev02-vs-kev01 ladder blowout (431 vs 1195) is reproduced
almost exactly by the sim at nanosecond ordering (356 vs 1252), which is
what validated this analysis.

## 7. ML track (`ml/`, `docs/KEV03.md`)

Scaffold for the "neural cartographer" third-family submission: a
ConvGRU spatial belief map with auxiliary mapping losses and
snapshot-consistency weak supervision, unit-token cross-attention, an
autoregressive joint action head, and a privileged critic — plus a
dependency-free C++ inference skeleton (`ml/csrc/`) for embedding the
trained policy in the contest `.so`. Training is **blocked on the
official logs** (`/share/data.tar.gz` + per-match downloads) for
simulator calibration; see `docs/KEV03.md` for the phase plan,
deployment latency budget, and open review findings.

## 8. Status → roadmap

Done: value/distance targeting, gold + bomb memory with decay, re-pickup
chains, dynamic `k` + `order`, priced fog crossing, seat inference with
adaptive profiles, a rules-faithful local simulator, ML-track scaffold.

Next, in rough order of expected value:

1. **Snapshot macro layer** — steer units toward regions with high
   `gold_remaining` and low traffic, and de-correlate the two units
   (one farms the center, one works an under-contested region). Needs
   the region geometry, to be extracted from the official game logs.
2. **Vision purchases as investment** — buy 7×7/9×9 when the end-of-round
   position is information-rich (frontier cells, supposedly-rich region,
   enough rounds left to amortize). The baseline bot shows the failure
   mode: it bought vision every snapshot round and finished negative.
3. **Opponent & NPC modeling** — we nearly always move first (µs
   latency), so contested piles are worth more to us than raw distance
   suggests: race the favorable contests, concede the lost ones; track
   NPC headings via their stable ids.
4. **Terminal-position value** — prefer plans that end next to spawn-rich
   ground, not just plans that collect the most now.
5. **Learned candidate scorer (second submission)** — mine the official
   logs for spawn/NPC statistics, train a small model (boosted trees) to
   score candidate move plans, and upload it under a separate model name
   so it can be A/B-tested against this bot on the ladder.

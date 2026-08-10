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

## 2. What this MVP does

Everything lives in [`src/player.cpp`](src/player.cpp) (~200 lines, no STL,
no allocation, no I/O):

1. **Fixed split**: each unit gets 3 of the 6 moves (`k=3`, `order=0`, no
   vision purchase).
2. **Gold seeking**: BFS over the *currently visible, safe* cells to the
   nearest gold pile; the two units pick different piles when possible.
3. **Fallback**: with no visible gold, walk toward the center (8,8), where
   gold spawns every round.
4. **Safety**: never steps into fog, obstacles, bombs, visible enemies,
   cells with ≥ 3 NPCs, or our own other unit. Worst case it stands still —
   which is always a legal answer.
5. **Persistent memory**: the engine keeps `player.so` loaded for the whole
   match, so globals survive between rounds. We use that to remember every
   obstacle ever seen (the map is static), and we reset state when the round
   counter restarts (= a new match began). This is the scaffold any future
   strategy state (gold heatmaps, opponent tracking) will reuse.
6. **Speed**: a decision costs ~1 µs (see `make test` output) against the
   300 ms budget — no timeout risk, and a strong entry for the speed prize.

## 3. Repo layout

```
├── Makefile              make -> player.so (the upload artifact); make test
├── src/
│   ├── game_api.h        contest ABI structs, transcribed from the rules
│   └── player.cpp        the bot — the only file strategy work touches
└── test/
    └── local_test.cpp    offline harness: dlopen()s player.so like the real
                          engine, runs scripted scenarios, checks legality,
                          measures p50/p90 latency
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
- [ ] If the official 参考代码 ships its own `game_api.h`, diff it against
      `src/game_api.h` and adopt the official one if they differ at all.

## 5. The interface in 30 seconds

Every round the engine calls:

```cpp
extern "C" GameOutput moveDecision(const GameInput* input);
```

- **In** (`GameInput`): `round`; `grid[17][17]` with `-5` fog / `-3` bomb /
  `-1` obstacle / `0` empty / `≥1` gold amount; our two unit positions and
  gold; opponent's total gold; visible enemy positions; visible NPCs; and
  every 5 rounds a region `snapshot`.
- **Out** (`GameOutput`): `actions[6]` (0=up 1=down 2=left 3=right 4=stay),
  `k` (unit 0 executes `actions[0:k]`, unit 1 executes `actions[k:6]`),
  `order` (which of our units moves first), `vp` (vision purchase).
- Illegal moves (wall / off-board / other player unit) are simply skipped —
  only a *malformed* output or a crash/timeout loses the match.
- One unit executes all of its steps, then the other; the grid we receive is
  the pre-move state, and only the final position's vision comes back next
  round (no en-route vision).

## 6. Known limitations → roadmap

Deliberately out of scope for the MVP, in rough order of expected value:

1. **Use the snapshot** — steer units toward regions with high
   `gold_remaining` instead of blind center-walking.
2. **Gold memory & heatmap** — remember seen-but-uncollected gold and spawn
   statistics across rounds (the persistence scaffold already exists).
3. **Dynamic move split** — vary `k` (e.g. 4/2 when one unit is in a rich
   area) and pick `order` deliberately.
4. **Vision purchases** — buy 7×7/9×9 when the expected information beats
   the 2–3 gold price.
5. **Re-pickup loops** — a cell keeps 35 % after pickup; oscillating on rich
   piles is profitable.
6. **Smarter fog handling** — currently fog is a wall; model it instead of
   avoiding it (risk: hidden bombs cost 10 %).
7. **Opponent/NPC modeling** — infer turn order from state diffs, avoid
   contested piles we'd lose, race the ones we'd win.

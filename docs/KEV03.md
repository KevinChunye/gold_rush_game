# kev03 — Neural Belief-Map Agent for GoldRush 2.0

*Track: machine learning. Status: scaffolding (`ml/`) — no trained model yet.
The deterministic bot (`src/player.cpp`) remains the submission until kev03
beats it under the criteria in section 7.*

Code map: `ml/model.py` (modules), `ml/dataset.py` (npz contract + fog
masking), `ml/train_phase_a.py` (cartography), `ml/train_imitation.py`
(behavior cloning), `ml/export_weights.py` (blob export),
`ml/csrc/nn_infer.cpp` (dependency-free C++ inference skeleton).

---

## 1. Framing: a mapping POMDP, not a localization one

GoldRush 2.0 is a two-player partially observable Markov game on a 17×17
grid over 500 rounds. Each round we see only 5×5 (optionally purchased
7×7/9×9) windows around our two units plus, every 5 rounds, a coarse
5-region Snapshot (gold generated/collected/remaining, enter/leave traffic,
occupants). Everything else is fog (`-5`).

Unlike classic robotics POMDPs there is **no localization problem** — we
always know exactly where our units are. The entire epistemic burden is
**mapping**: where are the obstacles (static, knowable forever once seen),
where are the bombs (respawn every ~20 rounds — beliefs must decay), where
is the gold (spawns every round, mostly in the central 9×9, gets eaten by
NPCs and the opponent — beliefs must decay *and* regenerate), and where are
the moving agents likely to be next round.

The right sufficient statistic for this is a **belief state over the map**.
The deterministic bot already keeps a hand-written one (obstacle memory,
bomb fear timers, gold belief decay); kev03 replaces those hand-tuned
scalar decay rules with a *learned, spatial, recurrent* belief:

    B_t ∈ R^{17×17×(C_interp + C_latent)},   B_t = ConvGRU(B_{t-1}, encode(O_t))

The first `C_interp = 7` channels are forced (by the training losses, see
§3-A) to be a calibrated map: obstacle log-odds, expected gold and its
log-variance, bomb log-odds, next-round NPC and opponent occupancy
log-odds, and time-since-observed. The other `C_latent = 32` channels are
free capacity for whatever the policy needs (NPC headings, opponent intent,
spawn rhythm). The policy then acts on B_t — acting on a belief state is
the standard reduction of a POMDP to an MDP; here the belief is neural and
approximate rather than exact Bayesian.

## 2. Architecture

```
                    fog-masked GameInput (round t)
                               |
              +----------------+-----------------+
              | 18 input planes (terrain one-hot,|
              | log1p gold, fog, 2 unit planes,  |
              | unit-gold/round broadcasts,      |
              | enemies, NPCs, 5 snapshot planes |
              | broadcast through region map)    |
              +----------------+-----------------+
                               v
                      ObsEncoder (2x conv3x3)          [C_ENC=32]
                               v
   B_{t-1} ----------->  ConvGRUCell  ----------------> B_t   [C_BELIEF=39]
   [39,17,17]           (gates r,z,n)                   |
                               ^                        +--> AuxHeads (affine
                               |                        |    reads of ch 0..6):
                        (recurrent, kept                |    obstacle | gold μ,logσ² |
                         across rounds)                 |    bomb | NPC_t+1 | opp_t+1 |
                                                        |    time-since-observed
                                                        |
              +-----------------------------------------+
              v                                         v
   UnitTokenActor                              PrivilegedCritic (TRAIN ONLY)
   2 tokens seeded from                        full unmasked sim state ->
   (B_t @ unit pos, unit gold, idx)            conv3x3 x2 -> GAP -> V(s)
   2 rounds cross-attention over               (never exported, never shipped)
   flattened 17x17 belief (+pos emb)
              v
   autoregressive joint action:
   order (2) -> k (7) -> m1..m6 (5 each, GRU decoder,
   owner = unit0 if i<k else unit1, legality mask hook
   zeroes moves into known obstacles / off-board) -> vp (3)
```

Design notes:

- **Interpretable-by-construction aux heads.** The aux heads are per-channel
  affine reads of B_t's first 7 channels (not separate convs), so the
  supervision gradient forces the ConvGRU itself to write a human-readable
  map into those channels. Debugging = imshow the belief.
- **Snapshot broadcast planes** paint each cell with its region's latest
  snapshot statistics through a **pluggable region-geometry function**
  (`ml/dataset.py:region_id_default`). Default — center 9×9 (rows/cols
  4..12) = region 5, surrounding ring split into NW/NE/SW/SE quadrants at
  row 8 / col 8 — **is a guess**; extract the real geometry from official
  logs/replays before trusting it.
- **Autoregressive joint action.** order→k→moves→vp exactly matches the
  `GameOutput` degrees of freedom; each factor conditions on the previous
  ones, so "k=2 because unit 1 has the lucrative chain" is expressible. The
  legality hook masks moves into *known* obstacles/off-board during both
  sampling and deployment (the engine would skip them anyway — masking just
  stops us wasting probability mass); fog is deliberately NOT masked.
- **Asymmetric actor-critic.** The critic sees the full simulator state
  (both players, true gold field, all NPCs). That is sound for policy
  gradient training (the critic is only a baseline) and free at deployment
  because the critic is never exported.

## 3. Training phases

**A. Self-supervised cartography** (`ml/train_phase_a.py`) — train
encoder+ConvGRU+aux only: roll the belief over T-step fog-masked windows of
logged/simulated games; supervise obstacle/bomb (BCE), gold (heteroscedastic
Gaussian NLL in log1p domain), NPC/opponent next-round occupancy (BCE),
time-since-observed (MSE), and the **snapshot-consistency loss**: for each
region, Huber( Σ_fog-cells predicted gold + Σ_visible-cells actual gold −
snapshot `gold_remaining` ), where only the fog-cell predictions carry
gradient (visible cells are exactly known constants). This teaches the net
to redistribute the snapshot's aggregate onto plausible fog cells. Cheap,
stable, and produces an inspectable map before any RL is attempted.

**B. World model.** Extend phase A with action-conditioned prediction:
predict next-round belief targets given our action (and later, learned NPC
and opponent transition heads). Gives (i) calibrated NPC/bomb dynamics,
(ii) a rollout model for search later if we want it. Explicitly blocked on
official logs (§7) — a world model fit to our simulator learns our guesses.

**C. Imitation bootstrap** (`ml/train_imitation.py`) — behavior-clone the
deterministic bot (and any stronger scripted variants) via cross-entropy on
the autoregressive factorization, teacher-forced. Warm-starts from the
phase-A checkpoint; `--aux-weight` keeps the map channels grounded during
BC. Output: a policy roughly as strong as the scripted bot but
differentiable — the safe starting point for RL.

**D. Recurrent PPO.** Fine-tune the whole net with PPO in the simulator:
belief GRU unrolled through rollouts (truncated BPTT), privileged critic,
reward = per-round gold delta (+ small terminal win bonus), entropy bonus
decayed, KL leash to the BC policy early on. Latency is modeled by the
simulator's execution order (§7).

**E. Self-play league.** Checkpoint pool + scripted opponents (the
deterministic bot v1/v2, greedy, camper, vision-buyer) with
prioritized-by-loss-rate sampling, so the policy neither overfits one rival
nor forgets how to beat simple bots. Gate promotions on held-out map seeds
(§4) and on ladder matches.

## 4. Map randomization and held-out seeds

We do not know the tournament map pool (obstacles are "fixed per map").
Train on procedurally randomized maps: obstacle count/topology (scattered,
walls, rooms), bomb layouts and respawn phases, gold spawn rate in/out of
the central 9×9, NPC count/aggression, opponent unit spawn corners. Hold
out a fixed set of ~50 map seeds never trained on; every promotion gate
(BC → PPO checkpoints → league) reports on held-out seeds only. When
official logs arrive, refit the randomization ranges to the observed
distributions and REGENERATE the held-out set from the calibrated ranges.

## 5. Oracle distillation and the partial-observability gap

Train a **full-information teacher**: same actor architecture but fed the
unmasked state (no fog) — it plays near-optimally with respect to
information. Then distill: fog-masked student minimizes KL to the oracle's
action distribution on the same states (DAgger-style, on states the student
visits).

The useful diagnostic is the **partial-observability gap**: (oracle score −
student score) on identical seeds. It decomposes our losses into "didn't
know" (gap) vs "knew but played badly" (oracle's own losses vs opponents),
and it prices vision purchases: if the gap is large and buying 9×9 vision
shrinks the student→oracle action divergence more than the 3-gold cost, the
vp head has something real to learn. Track the gap over training; a
shrinking gap means the belief map is doing its job.

## 6. Metrics

Belief quality (phase A/B, held-out seeds):
- **Obstacle F1** (and IoU) on fog cells vs ground truth.
- **Gold MAE under fog** (raw-gold domain), split by cell age since last
  observation (MAE@5, @20, @50 rounds unseen).
- **Bomb AUROC** on fog cells (bombs are rare — ROC, not accuracy).
- **NPC / opponent next-step log-loss** vs ground truth occupancy.
- **Snapshot consistency error**: | predicted region gold − snapshot
  `gold_remaining` | at snapshot rounds (the training loss, as a metric).
- **Calibration**: reliability curves / ECE for the obstacle & bomb
  probabilities; z-score histogram of (true−μ)/σ for gold (should be ~N(0,1)
  if the log-variance head is honest).

Policy quality:
- Win rate + mean gold margin vs the deterministic bot and each scripted
  opponent, on held-out seeds (paired by seed, report CIs).
- Gold collected per 100 rounds; bomb hits per match; trample events;
  vision purchases and their measured information value (§5).
- Partial-observability gap (§5).
- **Latency**: p50/p90 µs per decision through the C++ engine (never the
  Python path), and win rate as a function of injected extra latency in the
  simulator — the win-rate-vs-latency tradeoff curve (§7).
- Ladder: rating vs the public pool, and specifically the A/B result vs our
  own deterministic bot upload.

## 7. Constraints and risks (read this before believing anything above)

**Hardware & size.** The judge machine is a **32-core AMD EPYC CPU with no
GPU**, and the uploaded `.so` is capped at **16 MB** (weights + code + any
tables share that budget). Everything must run as plain CPU code inside
`moveDecision()`.

**Latency is not just a 300 ms timeout — it is part of the game.** Round
order is: spawns → both strategies queried → **faster player moves** →
NPCs move → slower player moves. Moving first wins contested piles and
picks gold before NPCs eat it. Our deterministic bot answers in ~25 µs p90;
a fat network would surrender that edge on every single round. Budget for
the deployed net: **≲ 200 µs per decision, roughly ≤ 2M MACs** — think
16-24 channels, 2-3 convs, one small ConvGRU, tiny token/actor head — with
int8 or AVX-512 kernels as the later escape hatch. The training-side
architecture in `ml/model.py` (32-ch encoder, 39-ch belief, 64-d tokens) is
deliberately larger for learning headroom; the shipped net must be a
shrunk/distilled variant. Reality check from this repo's skeleton
(`ml/csrc/nn_infer.cpp`, naive scalar loops, 4-core dev sandbox): a
4.8M-MAC belief stack costs **~3.2 ms at -O2** and ~1.1 ms at
`-O3 -march=native` — i.e. 5-16× over budget before vectorization and
channel-slimming. The plan is honest about this: measure the
**win-rate-vs-latency tradeoff in the simulator** (which models execution
order) and only pay for milliseconds that buy more expected gold than
moving first does. If a 2M-MAC net can't beat the scripted bot, a bigger
net that always moves second must beat it by even more — that bar rises
with size.

**Everything trainable is blocked on the official logs.** All phases (A-E)
currently would consume OUR simulator's guesses about gold spawn
distributions, NPC policy, bomb respawn details, and region geometry. Until
we obtain `/share/data.tar.gz` from the dev server and the per-match
downloadable game logs, and convert them to the npz contract in
`ml/dataset.py`: phase A learns to predict our own generator (fine as a
pipeline test, meaningless as calibration); phase B would be a world model
of the wrong world; PPO would exploit simulator artifacts. Log acquisition
and spawn/NPC/bomb calibration is the critical path, not model code.

**Region geometry is a guess** (center 9×9 = region 5, ring quadrants =
1-4). Snapshot features and the consistency loss inherit any error here.
Verify against logs (snapshot `gold_remaining` vs summed visible gold per
candidate geometry makes it identifiable) before trusting either.

**Imitation-first de-risks PPO.** Recurrent PPO from scratch on a sparse,
partially observed, adversarial game is where projects die. BC from the
deterministic bot gives a known-strength starting policy, verifies the
whole obs→net→action path end-to-end, and turns PPO into fine-tuning.

**The C++ gap is real engineering.** `nn_infer.cpp` is a skeleton: the full
forward graph (attention, autoregressive decode), plane construction from
`GameInput`, and the export-name wiring are unwritten (checklist in
`ml/csrc/README.md`). Numerical parity tests (PyTorch vs C++, same weights,
≤1e-5) are mandatory before any ladder upload.

**Fallback discipline.** The deterministic bot stays the submitted bot
until kev03 (through the C++ engine, at deployed size) beats it **both** on
held-out simulator seeds **and** in A/B matches on the ladder. No
enthusiasm-driven uploads.

---

## Appendix: open review findings (adversarial pass, 2026-08-11)

An adversarial review of this scaffold produced the following accepted-but-
not-yet-applied items (the snapshot-window undercount in `dataset.py` was
already fixed):

1. `model.py` — `vp` head does not condition on the sampled moves; either
   feed the move-decoder state into it or amend the factorization docs.
2. `csrc/nn_infer.cpp` — LayerNorm primitive and a composed multi-head
   attention are still missing for the exported actor.
3. `export_weights.py --smoke` writes to a temp dir and ignores `--out`,
   so the C++ byte-exact cross-check cannot run yet; the README checkbox
   overstates this until a real torch export has been diffed.
4. `dataset.py` — trajectories shorter than the sample window break
   `sample_batch` (pad + mask, or reject at load).
5. `dataset.py` — `gold_opp` is reconstructed but never encoded as an
   observation plane; add a broadcast plane (C_OBS 18 -> 19) or document
   the omission.
6. `dataset.py` — final-round `npc_next`/`opp_next` targets silently reuse
   the same round; emit a `next_valid` mask and apply it in the Phase A
   losses.

All six are scaffold-stage issues; none block Phase A once torch and the
official logs are available.

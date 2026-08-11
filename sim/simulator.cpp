// sim/simulator.cpp — headless local match simulator for GoldRush 2.0.
//
// Plays full matches between two contestant shared objects, each exporting
//     extern "C" GameOutput moveDecision(const GameInput*);
//
// Usage:
//     ./simulator botA.so botB.so [--games N] [--seed S] [--mirror]
//                                 [--csv file] [--quiet] [--reload]
//
//     --games N   number of base games (default 1); game i uses seed S+i
//     --seed S    base seed (default 1); one std::mt19937 per game
//     --mirror    play each seed twice with P1/P2 swapped; aggregate both
//     --csv file  write one row per played game (header documented below)
//     --quiet     suppress per-game summary lines (aggregate still printed)
//     --reload    fresh temp copy + dlopen of both bots between games;
//                 default keeps handles across games like the real engine,
//                 so bots must detect a new match by seeing round == 0
//
// dlopen isolation: both bots export the same symbol, and glibc dedups
// dlopen of the same file, so each bot is first copied to a unique temp
// file (mkstemp under $TMPDIR or /tmp) and the copy is dlopen'ed with
// RTLD_NOW | RTLD_LOCAL. That gives each bot its own globals even when
// botA.so and botB.so are the same file. The temp copy is unlinked
// immediately after dlopen (the mapping stays alive), so no temp files
// are left behind even on a crash. Note: if your /tmp is mounted noexec,
// point TMPDIR somewhere executable.
//
// Determinism: all gameplay randomness comes from one std::mt19937 seeded
// with (seed + game index), so a seed reproduces the WORLD exactly — but
// the real engine orders the two bots' moves by wall-clock decision time
// at nanosecond precision (official FAQ: exact ties are near-impossible;
// a true tie goes to P1), and wall-clock time is inherently irreproducible
// run to run. The default (order_quantum_ns = 1) mirrors the real engine,
// so move ORDER — and hence outcomes — can vary across runs for bots of
// similar speed; note also that P1's decision runs first each round
// (cache-warm asymmetry). For fully reproducible or controlled-order
// experiments use --quantum-ns (large values force ties -> P1) or
// --force-first A|B, which pin the ordering deterministically.
//
// ROUND FLOW (official semantics):
//   1. spawn gold/bombs for this round
//   2. both bots' moveDecision() called on the identical pre-move state
//      (each fog-masked to its own vision); wall-clock timed (steady_clock)
//   3. faster bot's two units move (order field picks which unit goes
//      first; a unit runs ALL its steps before the other starts)
//   4. NPCs move (up to 3 steps each)
//   5. slower bot's units move
//   6. vision purchases from this round's outputs take effect next round
//      only, for exactly one round; fees only tallied (deducted at scoring)

#include "../src/game_api.h"
#include "../src/constants.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SimConfig — ALL gameplay constants in one place. Values marked GUESS are
// not published in the official rules; they are best estimates pending
// calibration against official match logs.
// ---------------------------------------------------------------------------

struct World;
static int defaultRegionOf(int row, int col);
static int defaultNpcStep(World& w, int npcIdx);

using RegionFn    = int (*)(int row, int col);      // -> region id 1..5
using NpcPolicyFn = int (*)(World& w, int npcIdx);  // -> one action 0..4

struct SimConfig {
    int rounds = 500;

    // --- gold spawning (GUESS: rates/amounts pending log calibration) ---
    int center_lo = 4, center_hi = 12;   // center 9x9 = rows/cols 4..12
    int center_gold_per_round = 2;       // piles per round inside the center
    int center_gold_min = 1, center_gold_max = 8;
    int outer_gold_period = 7;           // GUESS: round % period == phase
    int outer_gold_phase = 0;            //        (so rounds 0,7,14,...)
    int outer_gold_count = 1;
    int outer_gold_min = 2, outer_gold_max = 16;

    // --- bombs (GUESS) ---
    int bomb_period = 20;                // wave when round % period == phase
    int bomb_phase = 0;                  // (so rounds 0,20,40,...)
    int bomb_count = 4;

    // --- NPCs ---
    int npc_count = MAX_NPCS;            // 7; all spawn at (8,8) at round 0
    int npc_steps = 3;                   // max steps per NPC per round
    bool npc_trigger_bombs = false;      // default: NPCs ignore bombs...
    bool npc_avoid_bombs = false;        // ...and do not path around them
    NpcPolicyFn npc_policy = defaultNpcStep;   // pluggable

    // --- trampling ---
    int trample_npc_threshold = 3;       // >= 3 NPCs on the entered cell

    // --- map ---
    int obstacle_count = 24;             // GUESS; placed as mirrored pairs
                                         // (odd values round up to even)

    // --- snapshot ---
    int snapshot_period = 5;
    int snapshot_phase = 4;              // valid when round % period == phase
    RegionFn region_of = defaultRegionOf;  // region geometry is NOT public

    // --- vision ---
    int base_vision = 2;                 // Chebyshev radius 2 = 5x5
    int vp1_fee = 2, vp2_fee = 3;        // vp=1 -> 7x7, vp=2 -> 9x9

    // Force a fixed first-moving seat (0/1) regardless of decision time;
    // -1 = order by measured decision time (the real engine's behavior).
    int force_first_seat = -1;

    // --- decision-time move ordering (see file header) ---
    // The real engine compares at nanosecond precision (official FAQ: exact
    // ties are near-impossible; on a true tie P1 moves first), so the
    // default is no quantization. Raise this to model a coarser engine.
    long long order_quantum_ns = 1;
};

// Pluggable region functions must return 1..REGION_COUNT; clamp defensively
// so a bad plug-in cannot index the window/occupancy arrays out of bounds.
static inline int regionIdx(const SimConfig& cfg, int r, int c) {
    const int id = cfg.region_of(r, c);
    return id < 1 ? 1 : id > REGION_COUNT ? REGION_COUNT : id;
}

// ---------------------------------------------------------------------------
// World state
// ---------------------------------------------------------------------------

struct Unit { Position pos; int gold; };
struct Npc  { int id; Position pos; int gold; };

// rolling per-region accumulators for the snapshot window (index 1..5)
struct RegionWindow {
    int enter[REGION_COUNT + 1]     = {};
    int leave[REGION_COUNT + 1]     = {};
    int generated[REGION_COUNT + 1] = {};
    int collected[REGION_COUNT + 1] = {};
    void reset() { *this = RegionWindow{}; }
};

struct World {
    const SimConfig* cfg = nullptr;
    std::mt19937 rng;
    int round = 0;
    bool obstacle[GRID_SIZE][GRID_SIZE] = {};
    int  gold[GRID_SIZE][GRID_SIZE]     = {};
    bool bomb[GRID_SIZE][GRID_SIZE]     = {};
    Unit units[2][2] = {};               // [player][unit]
    Npc  npcs[MAX_NPCS] = {};
    RegionWindow window;                 // stats since the last snapshot
    int windowBegin = 0;                 // label for the next snapshot

    int randint(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    }
};

static bool inBounds(int r, int c) {
    return r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE;
}

static bool unitAt(const World& w, int r, int c) {
    for (int p = 0; p < 2; ++p)
        for (int u = 0; u < 2; ++u)
            if (w.units[p][u].pos.row == r && w.units[p][u].pos.col == c)
                return true;
    return false;
}

static int npcCountAt(const World& w, int r, int c) {
    int n = 0;
    for (int i = 0; i < w.cfg->npc_count; ++i)
        if (w.npcs[i].pos.row == r && w.npcs[i].pos.col == c) ++n;
    return n;
}

// exact integer versions of the rule fractions (all ceil, gold >= 0)
static int pickupAmount(int g)      { return (13 * g + 19) / 20; }        // ceil(0.65*g)
static int bombLossOf(int gold)     { return gold > 0 ? (gold + 9)  / 10 : 0; } // ceil(10%)
static int trampleLossOf(int gold)  { return gold > 0 ? (gold + 19) / 20 : 0; } // ceil(5%)

// DEFAULT REGION GUESS (real geometry is not public): region 5 = center 9x9
// (rows/cols 4..12); the remaining ring splits into quadrants at row 8 /
// col 8: 1=NW 2=NE 3=SW 4=SE (ties r==8/c==8 go north/west).
static int defaultRegionOf(int row, int col) {
    if (row >= 4 && row <= 12 && col >= 4 && col <= 12) return 5;
    if (row <= 8) return col <= 8 ? 1 : 2;
    return col <= 8 ? 3 : 4;
}

// DEFAULT NPC POLICY (pluggable; the real one is unknown): each step, walk
// one step toward the nearest gold pile by Manhattan distance; obstacle-
// aware only in that it never proposes stepping into an obstacle; random
// tie-breaks (among nearest piles and among improving directions) use the
// game RNG; stay when there is no gold or no improving step. Gold on the
// NPC's own cell is ignored as a target — standing still never picks up,
// so a distance-0 target would freeze the NPC forever.
static int defaultNpcStep(World& w, int npcIdx) {
    const SimConfig& cfg = *w.cfg;
    const Npc& npc = w.npcs[npcIdx];
    int best = INT_MAX;
    std::vector<Position> targets;
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (w.gold[r][c] <= 0) continue;
            if (r == npc.pos.row && c == npc.pos.col) continue;
            int d = std::abs(r - npc.pos.row) + std::abs(c - npc.pos.col);
            if (d < best) { best = d; targets.clear(); }
            if (d == best) targets.push_back({r, c});
        }
    if (targets.empty()) return ACT_STAY;
    Position t = targets[targets.size() == 1
                             ? 0
                             : (size_t)w.randint(0, (int)targets.size() - 1)];
    static const int DR[4] = {-1, 1, 0, 0}, DC[4] = {0, 0, -1, 1};
    int cand[4], n = 0;
    int cur = std::abs(t.row - npc.pos.row) + std::abs(t.col - npc.pos.col);
    for (int a = 0; a < 4; ++a) {
        int nr = npc.pos.row + DR[a], nc = npc.pos.col + DC[a];
        if (!inBounds(nr, nc) || w.obstacle[nr][nc]) continue;
        if (cfg.npc_avoid_bombs && w.bomb[nr][nc]) continue;
        if (std::abs(t.row - nr) + std::abs(t.col - nc) < cur) cand[n++] = a;
    }
    if (n == 0) return ACT_STAY;
    return n == 1 ? cand[0] : cand[w.randint(0, n - 1)];
}

// ---------------------------------------------------------------------------
// Map generation: obstacle_count obstacles as pairs mirrored under
// (r,c) -> (16-r,16-c); never on the 4 spawn corners or (8,8); reject maps
// where flood fill from (0,0) does not reach every non-obstacle cell.
// ---------------------------------------------------------------------------

static bool fullyConnected(const World& w) {
    bool seen[GRID_SIZE][GRID_SIZE] = {};
    std::vector<Position> stack;
    stack.push_back({0, 0});
    seen[0][0] = true;
    int reached = 0;
    static const int DR[4] = {-1, 1, 0, 0}, DC[4] = {0, 0, -1, 1};
    while (!stack.empty()) {
        Position p = stack.back();
        stack.pop_back();
        ++reached;
        for (int d = 0; d < 4; ++d) {
            int nr = p.row + DR[d], nc = p.col + DC[d];
            if (inBounds(nr, nc) && !w.obstacle[nr][nc] && !seen[nr][nc]) {
                seen[nr][nc] = true;
                stack.push_back({nr, nc});
            }
        }
    }
    int open = 0;
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c)
            if (!w.obstacle[r][c]) ++open;
    return reached == open;
}

static void generateMap(World& w) {
    const SimConfig& cfg = *w.cfg;
    auto forbidden = [](int r, int c) {
        return (r == 0 && c == 0) || (r == 0 && c == GRID_SIZE - 1) ||
               (r == GRID_SIZE - 1 && c == 0) ||
               (r == GRID_SIZE - 1 && c == GRID_SIZE - 1) ||
               (r == 8 && c == 8);
    };
    for (;;) {
        std::memset(w.obstacle, 0, sizeof w.obstacle);
        int placed = 0, guard = 0;
        while (placed < cfg.obstacle_count && ++guard < 100000) {
            int r = w.randint(0, GRID_SIZE - 1), c = w.randint(0, GRID_SIZE - 1);
            int mr = GRID_SIZE - 1 - r, mc = GRID_SIZE - 1 - c;
            if (forbidden(r, c) || forbidden(mr, mc)) continue;
            if (r == mr && c == mc) continue;  // self-mirrored center
            if (w.obstacle[r][c] || w.obstacle[mr][mc]) continue;
            w.obstacle[r][c] = w.obstacle[mr][mc] = true;
            placed += 2;
        }
        if (fullyConnected(w)) return;  // else reject and retry
    }
}

// ---------------------------------------------------------------------------
// Phase 1: spawning
// ---------------------------------------------------------------------------

static void spawnGoldPile(World& w, bool center) {
    const SimConfig& cfg = *w.cfg;
    std::vector<Position> cells;
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c) {
            bool inC = r >= cfg.center_lo && r <= cfg.center_hi &&
                       c >= cfg.center_lo && c <= cfg.center_hi;
            if (inC != center) continue;
            if (w.obstacle[r][c] || w.bomb[r][c]) continue;  // gold+bomb never coexist
            cells.push_back({r, c});
        }
    if (cells.empty()) return;
    Position p = cells[(size_t)w.randint(0, (int)cells.size() - 1)];
    int amt = center ? w.randint(cfg.center_gold_min, cfg.center_gold_max)
                     : w.randint(cfg.outer_gold_min, cfg.outer_gold_max);
    w.gold[p.row][p.col] += amt;  // gold accumulates on repeat spawns
    w.window.generated[regionIdx(cfg, p.row, p.col)] += amt;
}

static void spawnOneBomb(World& w) {
    std::vector<Position> cells;
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (w.obstacle[r][c] || w.bomb[r][c] || w.gold[r][c] > 0) continue;
            if (unitAt(w, r, c) || npcCountAt(w, r, c) > 0) continue;
            cells.push_back({r, c});
        }
    if (cells.empty()) return;
    Position p = cells[(size_t)w.randint(0, (int)cells.size() - 1)];
    w.bomb[p.row][p.col] = true;
}

static void spawnPhase(World& w) {
    const SimConfig& cfg = *w.cfg;
    for (int i = 0; i < cfg.center_gold_per_round; ++i) spawnGoldPile(w, true);
    if (cfg.outer_gold_period > 0 &&
        w.round % cfg.outer_gold_period == cfg.outer_gold_phase)
        for (int i = 0; i < cfg.outer_gold_count; ++i) spawnGoldPile(w, false);
    if (cfg.bomb_period > 0 && w.round % cfg.bomb_period == cfg.bomb_phase)
        for (int i = 0; i < cfg.bomb_count; ++i) spawnOneBomb(w);
}

// ---------------------------------------------------------------------------
// Snapshot (window semantics: delivered right after the spawn phase of a
// snapshot round R, covering spawns of rounds R-4..R and movement/pickups
// of the four rounds after the previous snapshot; the accumulators reset at
// delivery, so round R's own moves fall into the NEXT window)
// ---------------------------------------------------------------------------

static Snapshot makeSnapshot(const World& w, bool valid) {
    Snapshot s{};
    if (!valid) {
        s.window_begin = -1;
        s.window_end = -1;
        return s;
    }
    s.window_begin = w.windowBegin;
    s.window_end = w.round;
    int remaining[REGION_COUNT + 1] = {}, occ[REGION_COUNT + 1] = {};
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c)
            if (w.gold[r][c] > 0)
                remaining[regionIdx(*w.cfg, r, c)] += w.gold[r][c];
    for (int p = 0; p < 2; ++p)
        for (int u = 0; u < 2; ++u)
            ++occ[regionIdx(*w.cfg, w.units[p][u].pos.row, w.units[p][u].pos.col)];
    for (int i = 0; i < w.cfg->npc_count; ++i)
        ++occ[regionIdx(*w.cfg, w.npcs[i].pos.row, w.npcs[i].pos.col)];
    for (int i = 0; i < REGION_COUNT; ++i) {
        int id = i + 1;
        s.regions[i].id = id;
        s.regions[i].enter = w.window.enter[id];
        s.regions[i].leave = w.window.leave[id];
        s.regions[i].gold_generated = w.window.generated[id];
        s.regions[i].gold_collected = w.window.collected[id];
        s.regions[i].gold_remaining = remaining[id];
        s.regions[i].occupants = occ[id];
    }
    return s;
}

// regions of all 11 actors (4 units then npc_count NPCs), for enter/leave
static void collectRegions(const World& w, int out[]) {
    int i = 0;
    for (int p = 0; p < 2; ++p)
        for (int u = 0; u < 2; ++u)
            out[i++] = regionIdx(*w.cfg, w.units[p][u].pos.row, w.units[p][u].pos.col);
    for (int n = 0; n < w.cfg->npc_count; ++n)
        out[i++] = regionIdx(*w.cfg, w.npcs[n].pos.row, w.npcs[n].pos.col);
}

// ---------------------------------------------------------------------------
// Phase 2: fog-masked GameInput for player p (absolute coordinates for BOTH
// players; the real engine does not mirror P2's view)
// ---------------------------------------------------------------------------

static GameInput buildInput(const World& w, int p, int radius,
                            const Snapshot& snap, bool snapValid) {
    GameInput in{};
    in.round = w.round;
    bool vis[GRID_SIZE][GRID_SIZE] = {};
    for (int u = 0; u < 2; ++u) {
        Position pos = w.units[p][u].pos;
        for (int r = std::max(0, pos.row - radius);
             r <= std::min(GRID_SIZE - 1, pos.row + radius); ++r)
            for (int c = std::max(0, pos.col - radius);
                 c <= std::min(GRID_SIZE - 1, pos.col + radius); ++c)
                vis[r][c] = true;
    }
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (!vis[r][c])            in.grid[r][c] = CELL_FOG;
            else if (w.obstacle[r][c]) in.grid[r][c] = CELL_OBSTACLE;
            else if (w.bomb[r][c])     in.grid[r][c] = CELL_BOMB;
            else                       in.grid[r][c] = w.gold[r][c]; // 0 or >=1
        }
    for (int u = 0; u < 2; ++u) {
        in.my_units[u] = w.units[p][u].pos;          // always exact
        in.my_units_gold[u] = w.units[p][u].gold;
    }
    in.gold_opp = w.units[1 - p][0].gold + w.units[1 - p][1].gold;  // GROSS
    int ne = 0;
    for (int u = 0; u < 2; ++u) {
        Position ep = w.units[1 - p][u].pos;
        if (vis[ep.row][ep.col]) in.visible_enemies[ne++] = ep;  // packed
    }
    for (; ne < 2; ++ne) in.visible_enemies[ne] = {-1, -1};
    int nn = 0;
    for (int i = 0; i < w.cfg->npc_count; ++i)
        if (vis[w.npcs[i].pos.row][w.npcs[i].pos.col])
            in.visible_npcs[nn++] = {w.npcs[i].id, w.npcs[i].pos};  // stable ids
    in.num_visible_npcs = nn;
    for (; nn < MAX_NPCS; ++nn) in.visible_npcs[nn] = {0, {-1, -1}};
    in.snapshot_valid = snapValid ? 1 : 0;
    in.snapshot = snap;
    return in;
}

// ---------------------------------------------------------------------------
// Phases 3/5: player unit movement
// ---------------------------------------------------------------------------

struct SlotStats {                       // per player slot, per game
    long long vision = 0;                // vision fees tallied (phase 6)
    long long bombLoss = 0, trampleLoss = 0;
    long long illegal = 0, stay = 0, malformed = 0;
    int firstMoves = 0;                  // rounds this slot moved first
    std::vector<long long> decNs;        // per-round decision times
};

static const int DR5[5] = {-1, 1, 0, 0, 0}, DC5[5] = {0, 0, -1, 1, 0};

// one step of one unit; illegal steps are skipped but later steps still run
static void stepUnit(World& w, int p, int u, int a, SlotStats& st) {
    Unit& unit = w.units[p][u];
    if (a == ACT_STAY) {                 // standing still never picks up
        ++st.stay;
        return;
    }
    int nr = unit.pos.row + DR5[a], nc = unit.pos.col + DC5[a];
    // legal iff in bounds, not an obstacle, and not occupied by any of the
    // 4 player units (a unit occupies its cell until it actually moves, so
    // swaps/pass-throughs are impossible). NPCs never block.
    if (!inBounds(nr, nc) || w.obstacle[nr][nc] || unitAt(w, nr, nc)) {
        ++st.illegal;
        return;
    }
    unit.pos = {nr, nc};
    if (w.bomb[nr][nc]) {                // bomb: pay ceil(10%), bomb gone
        int loss = bombLossOf(unit.gold);
        unit.gold -= loss;
        st.bombLoss += loss;
        w.bomb[nr][nc] = false;          // unit keeps moving afterwards
    }
    if (w.gold[nr][nc] > 0) {            // pickup BEFORE trample
        int take = pickupAmount(w.gold[nr][nc]);
        unit.gold += take;
        w.gold[nr][nc] -= take;          // cell keeps the remainder
        w.window.collected[regionIdx(*w.cfg, nr, nc)] += take;
    }
    if (npcCountAt(w, nr, nc) >= w.cfg->trample_npc_threshold) {
        int loss = trampleLossOf(unit.gold);   // trample on entry only
        unit.gold -= loss;
        st.trampleLoss += loss;
    }
}

// execute a whole GameOutput for player p, clamping+counting malformed fields
static void applyBotMoves(World& w, int p, const GameOutput& out, SlotStats& st) {
    int k = out.k;
    if (k < 0 || k > S) { ++st.malformed; k = k < 0 ? 0 : S; }
    int order = out.order;
    if (order != 0 && order != 1) { ++st.malformed; order = order < 0 ? 0 : 1; }
    for (int i = 0; i < 2; ++i) {
        int u = (order == 0) ? i : 1 - i;      // first unit finishes ALL its
        int lo = (u == 0) ? 0 : k;             // steps before the other starts
        int hi = (u == 0) ? k : S;
        for (int s = lo; s < hi; ++s) {
            int a = out.actions[s];
            const bool bad = (a < 0 || a > 4);
            if (bad) { ++st.malformed; a = ACT_STAY; }
            const long long staysBefore = st.stay;
            stepUnit(w, p, u, a, st);
            if (bad) st.stay = staysBefore;  // don't count garbage as a stay
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4: NPC movement
// ---------------------------------------------------------------------------

static void moveNpcs(World& w) {
    const SimConfig& cfg = *w.cfg;
    for (int i = 0; i < cfg.npc_count; ++i) {
        Npc& npc = w.npcs[i];
        for (int s = 0; s < cfg.npc_steps; ++s) {
            int a = cfg.npc_policy(w, i);
            if (a < 0 || a > 4 || a == ACT_STAY) continue;
            int nr = npc.pos.row + DR5[a], nc = npc.pos.col + DC5[a];
            if (!inBounds(nr, nc) || w.obstacle[nr][nc]) continue;  // policy guard
            if (cfg.npc_avoid_bombs && w.bomb[nr][nc]) continue;
            npc.pos = {nr, nc};          // NPCs are never blocked by actors
            if (w.bomb[nr][nc] && cfg.npc_trigger_bombs) {
                npc.gold -= bombLossOf(npc.gold);
                w.bomb[nr][nc] = false;
            }
            if (w.gold[nr][nc] > 0) {    // NPC pickup: same 65% rule; the
                int take = pickupAmount(w.gold[nr][nc]);   // gold never scores
                npc.gold += take;
                w.gold[nr][nc] -= take;
                w.window.collected[regionIdx(cfg, nr, nc)] += take;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bot loading (see dlopen note in the file header)
// ---------------------------------------------------------------------------

using MoveFn = GameOutput (*)(const GameInput*);

static bool copyAll(int src, int dst) {
    char buf[1 << 16];
    ssize_t n;
    while ((n = read(src, buf, sizeof buf)) > 0) {
        ssize_t off = 0;
        while (off < n) {
            ssize_t m = write(dst, buf + off, (size_t)(n - off));
            if (m < 0) return false;
            off += m;
        }
    }
    return n == 0;
}

struct Bot {
    std::string origPath;   // as given on the command line
    void* handle = nullptr;
    MoveFn fn = nullptr;

    bool load() {
        int src = open(origPath.c_str(), O_RDONLY);
        if (src < 0) { std::perror(origPath.c_str()); return false; }
        const char* tmpdir = getenv("TMPDIR");
        if (!tmpdir || !*tmpdir) tmpdir = "/tmp";
        std::string t = std::string(tmpdir) + "/simbot_XXXXXX";
        std::vector<char> path(t.begin(), t.end());
        path.push_back('\0');
        int dst = mkstemp(path.data());
        if (dst < 0) { std::perror("mkstemp"); close(src); return false; }
        bool ok = copyAll(src, dst);
        close(src);
        close(dst);
        if (!ok) {
            std::fprintf(stderr, "copy of %s failed\n", origPath.c_str());
            unlink(path.data());
            return false;
        }
        handle = dlopen(path.data(), RTLD_NOW | RTLD_LOCAL);
        unlink(path.data());   // mapping stays alive; nothing left on disk
        if (!handle) {
            std::fprintf(stderr, "dlopen %s: %s\n", origPath.c_str(), dlerror());
            return false;
        }
        fn = reinterpret_cast<MoveFn>(dlsym(handle, "moveDecision"));
        if (!fn) {
            std::fprintf(stderr, "dlsym moveDecision in %s: %s\n",
                         origPath.c_str(), dlerror());
            dlclose(handle);
            handle = nullptr;
            return false;
        }
        return true;
    }
    void unload() {
        if (handle) { dlclose(handle); handle = nullptr; fn = nullptr; }
    }
    bool reload() { unload(); return load(); }
};

static GameOutput timedCall(const Bot& b, const GameInput& in, long long& ns) {
    auto t0 = std::chrono::steady_clock::now();
    GameOutput out = b.fn(&in);
    auto t1 = std::chrono::steady_clock::now();
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return out;
}

// nearest-rank percentile of decision times, reported in whole microseconds
static long long percentileUs(std::vector<long long> ns, double p) {
    if (ns.empty()) return 0;
    for (long long& t : ns) t /= 1000;
    std::sort(ns.begin(), ns.end());
    size_t n = ns.size();
    size_t rank = (size_t)std::ceil(p * (double)n);
    if (rank < 1) rank = 1;
    if (rank > n) rank = n;
    return ns[rank - 1];
}

// ---------------------------------------------------------------------------
// One full game
// ---------------------------------------------------------------------------

struct GameRecord {
    long long gross[2] = {}, vision[2] = {}, net[2] = {};
    long long bombLoss[2] = {}, trampleLoss[2] = {};
    long long illegal[2] = {}, stay[2] = {}, malformed[2] = {};
    int firstMoves[2] = {};
    long long p50us[2] = {}, p90us[2] = {};
    int winner = -1;                     // 0, 1, or -1 = tie
    std::vector<long long> decNs[2];
};

static GameRecord playGame(const SimConfig& cfgIn, Bot* bot[2], unsigned seed) {
    SimConfig cfgLocal = cfgIn;
    if (cfgLocal.npc_count > MAX_NPCS) cfgLocal.npc_count = MAX_NPCS;  // arrays are sized MAX_NPCS
    if (cfgLocal.npc_count < 0) cfgLocal.npc_count = 0;
    const SimConfig& cfg = cfgLocal;
    World w;
    w.cfg = &cfg;
    w.rng.seed(seed);
    generateMap(w);
    w.units[0][0] = {{0, 0}, 0};                         // P1 spawns
    w.units[0][1] = {{GRID_SIZE - 1, GRID_SIZE - 1}, 0};
    w.units[1][0] = {{0, GRID_SIZE - 1}, 0};             // P2 spawns
    w.units[1][1] = {{GRID_SIZE - 1, 0}, 0};
    for (int i = 0; i < cfg.npc_count; ++i) w.npcs[i] = {i + 1, {8, 8}, 0};

    SlotStats st[2];
    int radius[2] = {cfg.base_vision, cfg.base_vision};
    const int actors = 4 + cfg.npc_count;

    for (w.round = 0; w.round < cfg.rounds; ++w.round) {
        int startReg[4 + MAX_NPCS], endReg[4 + MAX_NPCS];
        collectRegions(w, startReg);

        // (1) spawn, then build this round's snapshot if due
        spawnPhase(w);
        bool snapDue = cfg.snapshot_period > 0 &&
                       w.round % cfg.snapshot_period == cfg.snapshot_phase;
        Snapshot snap = makeSnapshot(w, snapDue);
        if (snapDue) {                    // next window starts with this
            w.window.reset();             // round's own movement
            // Movement accumulators reset mid-round R (after delivery), so
            // the next window's enter/leave/collected include R's own moves:
            // label it as starting at R rather than R+1.
            w.windowBegin = w.round;
        }

        // (2) both bots decide on the identical pre-move state
        GameInput in[2];
        for (int p = 0; p < 2; ++p)
            in[p] = buildInput(w, p, radius[p], snap, snapDue);
        GameOutput out[2];
        long long ns[2];
        for (int p = 0; p < 2; ++p) {
            out[p] = timedCall(*bot[p], in[p], ns[p]);
            st[p].decNs.push_back(ns[p]);
        }

        // (6) vision purchases: fee tallied now, effect next round only
        int nextRadius[2];
        for (int p = 0; p < 2; ++p) {
            int vp = out[p].vp;
            if (vp < 0 || vp > 2) { ++st[p].malformed; vp = vp < 0 ? 0 : 2; }
            nextRadius[p] = cfg.base_vision + vp;        // 1 -> 7x7, 2 -> 9x9
            st[p].vision += vp == 1 ? cfg.vp1_fee : vp == 2 ? cfg.vp2_fee : 0;
        }

        // (3,4,5) faster bot moves, then NPCs, then slower bot
        long long q = cfg.order_quantum_ns > 0 ? cfg.order_quantum_ns : 1;
        int first = cfg.force_first_seat >= 0 ? cfg.force_first_seat
                  : (ns[1] / q < ns[0] / q) ? 1 : 0;       // tie -> P1
        ++st[first].firstMoves;
        applyBotMoves(w, first, out[first], st[first]);
        moveNpcs(w);
        applyBotMoves(w, 1 - first, out[1 - first], st[1 - first]);

        // enter/leave accounting: region at round start vs round end
        collectRegions(w, endReg);
        for (int a = 0; a < actors; ++a)
            if (startReg[a] != endReg[a]) {
                ++w.window.leave[startReg[a]];
                ++w.window.enter[endReg[a]];
            }

        radius[0] = nextRadius[0];
        radius[1] = nextRadius[1];
    }

    GameRecord rec;
    for (int p = 0; p < 2; ++p) {
        rec.gross[p] = w.units[p][0].gold + w.units[p][1].gold;
        rec.vision[p] = st[p].vision;
        rec.net[p] = rec.gross[p] - rec.vision[p];       // fees deducted here
        rec.bombLoss[p] = st[p].bombLoss;
        rec.trampleLoss[p] = st[p].trampleLoss;
        rec.illegal[p] = st[p].illegal;
        rec.stay[p] = st[p].stay;
        rec.malformed[p] = st[p].malformed;
        rec.firstMoves[p] = st[p].firstMoves;
        rec.p50us[p] = percentileUs(st[p].decNs, 0.50);
        rec.p90us[p] = percentileUs(st[p].decNs, 0.90);
        rec.decNs[p] = std::move(st[p].decNs);
    }
    if (rec.net[0] != rec.net[1])
        rec.winner = rec.net[0] > rec.net[1] ? 0 : 1;
    else if (rec.p90us[0] != rec.p90us[1])               // tie -> lower p90
        rec.winner = rec.p90us[0] < rec.p90us[1] ? 0 : 1;
    return rec;
}

// ---------------------------------------------------------------------------
// CLI, aggregation, CSV
// ---------------------------------------------------------------------------

struct BotAgg {
    std::string name;
    int plays = 0, wins = 0, draws = 0;
    double sumNet = 0, sumGross = 0, sumVision = 0;
    double sumBomb = 0, sumTrample = 0, sumFirst = 0;
    long long illegal = 0, stay = 0, malformed = 0;
    std::vector<long long> allDecNs;
};

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s botA.so botB.so [--games N] [--seed S] [--mirror] "
        "[--csv file] [--quiet] [--reload] [--quantum-ns Q] [--force-first A|B]\n"
        "  --quantum-ns: decision-time quantum for move ordering (default 1 =\n"
        "  nanosecond compare like the real engine; large values neutralize\n"
        "  latency, e.g. 1000000000 forces alternating/tie ordering)\n", prog);
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    std::string csvPath;
    int games = 1;
    unsigned seed = 1;
    long long quantum = -1;
    int forceFirstBot = -1;
    bool mirror = false, quiet = false, reload = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--games" && i + 1 < argc) games = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = (unsigned)std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--mirror") mirror = true;
        else if (a == "--csv" && i + 1 < argc) csvPath = argv[++i];
        else if (a == "--quiet") quiet = true;
        else if (a == "--reload") reload = true;
        else if (a == "--quantum-ns" && i + 1 < argc) quantum = std::atoll(argv[++i]);
        else if (a == "--force-first" && i + 1 < argc) {
            std::string b = argv[++i];
            forceFirstBot = (b == "A" || b == "a") ? 0 : (b == "B" || b == "b") ? 1 : -1;
        }
        else if (a.rfind("--", 0) == 0) { usage(argv[0]); return 2; }
        else pos.push_back(a);
    }
    if (pos.size() != 2 || games < 1) { usage(argv[0]); return 2; }

    SimConfig cfg;
    if (quantum > 0) cfg.order_quantum_ns = quantum;
    Bot botA, botB;
    botA.origPath = pos[0];
    botB.origPath = pos[1];
    if (!botA.load()) return 1;
    if (!botB.load()) { botA.unload(); return 1; }

    std::FILE* csv = nullptr;
    if (!csvPath.empty()) {
        csv = std::fopen(csvPath.c_str(), "w");
        if (!csv) { std::perror(csvPath.c_str()); botA.unload(); botB.unload(); return 1; }
        std::fprintf(csv, "game,seed,mirrored,p1,p2,net1,net2,gross1,gross2,"
                          "vision1,vision2,bombloss1,bombloss2,trample1,trample2,"
                          "illegal1,illegal2,stay1,stay2,p90us1,p90us2,"
                          "firstmover1,winner\n");
    }

    BotAgg agg[2];
    agg[0].name = botA.origPath;
    agg[1].name = botB.origPath;
    int totalPlays = 0;

    for (int g = 0; g < games; ++g) {
        unsigned gseed = seed + (unsigned)g;
        int plays = mirror ? 2 : 1;
        for (int m = 0; m < plays; ++m) {
            if (reload && totalPlays > 0) {
                if (!botA.reload() || !botB.reload()) return 1;
            }
            Bot* slot[2] = {m == 0 ? &botA : &botB, m == 0 ? &botB : &botA};
            SimConfig cfgPlay = cfg;
            if (forceFirstBot >= 0)
                cfgPlay.force_first_seat = (slot[0] == (forceFirstBot == 0 ? &botA : &botB)) ? 0 : 1;
            GameRecord rec = playGame(cfgPlay, slot, gseed);
            ++totalPlays;

            const char* winStr = rec.winner == 0 ? "P1"
                               : rec.winner == 1 ? "P2" : "tie";
            if (!quiet)
                std::printf("game %d%s seed %u: P1=%s net=%lld (gross %lld, vis %lld)"
                            " | P2=%s net=%lld (gross %lld, vis %lld)"
                            " | P1 first %d/%d | winner=%s\n",
                            g, m ? " (mirrored)" : "", gseed,
                            slot[0]->origPath.c_str(), rec.net[0], rec.gross[0], rec.vision[0],
                            slot[1]->origPath.c_str(), rec.net[1], rec.gross[1], rec.vision[1],
                            rec.firstMoves[0], cfg.rounds, winStr);
            if (csv)
                std::fprintf(csv, "%d,%u,%d,%s,%s,%lld,%lld,%lld,%lld,%lld,%lld,"
                                  "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%s\n",
                             g, gseed, m,
                             slot[0]->origPath.c_str(), slot[1]->origPath.c_str(),
                             rec.net[0], rec.net[1], rec.gross[0], rec.gross[1],
                             rec.vision[0], rec.vision[1],
                             rec.bombLoss[0], rec.bombLoss[1],
                             rec.trampleLoss[0], rec.trampleLoss[1],
                             rec.illegal[0], rec.illegal[1],
                             rec.stay[0], rec.stay[1],
                             rec.p90us[0], rec.p90us[1],
                             rec.firstMoves[0],
                             rec.winner == -1 ? "tie" : (rec.winner == 0 ? "1" : "2"));

            for (int s = 0; s < 2; ++s) {           // map slot -> bot A/B
                BotAgg& a = agg[slot[s] == &botA ? 0 : 1];
                ++a.plays;
                if (rec.winner == s) ++a.wins;
                if (rec.winner == -1) ++a.draws;
                a.sumNet += (double)rec.net[s];
                a.sumGross += (double)rec.gross[s];
                a.sumVision += (double)rec.vision[s];
                a.sumBomb += (double)rec.bombLoss[s];
                a.sumTrample += (double)rec.trampleLoss[s];
                a.sumFirst += (double)rec.firstMoves[s];
                a.illegal += rec.illegal[s];
                a.stay += rec.stay[s];
                a.malformed += rec.malformed[s];
                a.allDecNs.insert(a.allDecNs.end(),
                                  rec.decNs[s].begin(), rec.decNs[s].end());
            }
        }
    }
    if (csv) std::fclose(csv);

    std::printf("\n=== aggregate: %d base game(s)%s = %d play(s) of %d rounds ===\n",
                games, mirror ? " (mirrored)" : "", totalPlays, cfg.rounds);
    for (int b = 0; b < 2; ++b) {
        const BotAgg& a = agg[b];
        double n = a.plays > 0 ? (double)a.plays : 1.0;
        std::printf("[%c] %s\n", 'A' + b, a.name.c_str());
        std::printf("    wins %d  draws %d  losses %d  (of %d plays)\n",
                    a.wins, a.draws, a.plays - a.wins - a.draws, a.plays);
        std::printf("    mean net %.1f  mean gross %.1f  mean vision spend %.1f\n",
                    a.sumNet / n, a.sumGross / n, a.sumVision / n);
        std::printf("    mean bomb loss %.1f  mean trample loss %.1f\n",
                    a.sumBomb / n, a.sumTrample / n);
        std::printf("    illegal steps %lld  stay steps %lld  malformed outputs %lld\n",
                    a.illegal, a.stay, a.malformed);
        std::printf("    decision time p50 %lld us  p90 %lld us\n",
                    percentileUs(a.allDecNs, 0.50), percentileUs(a.allDecNs, 0.90));
        std::printf("    mean rounds moved first %.1f\n", a.sumFirst / n);
    }

    botA.unload();
    botB.unload();
    return 0;
}

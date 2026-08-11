// player.cpp — GoldRush 2.0 bot, v2.1 "memory + value chains, disciplined".
//
// v2 lost its ladder A/B against v1 (431 vs 1195). Post-mortem pointed at
// three causes, each fixed here:
//
//   A. Fog-blind pathing. v2 planned shortest paths through fog for free
//      when the unit was poor. A hidden obstacle skips that step and every
//      LATER step then executes from the wrong cell (the engine keeps
//      trying the rest of the sequence), so one unseen wall could turn a
//      whole round of moves into garbage — and hidden bombs taxed us too.
//      Fix: every fog step now carries a base price (hidden obstacles
//      waste moves even when we hold no gold) plus the wealth-scaled bomb
//      price, and any emitted path is TRUNCATED at its first fog cell: a
//      unit steps into at most one unknown cell per round, re-planning
//      with fresh vision next round. No more cascading desync.
//
//   B. Phantom gold. v2 remembered piles for ~40 rounds everywhere. In
//      the center 9x9 — where the opponent and 7 NPCs live — a pile seen
//      5 rounds ago is almost certainly gone, and units marched to
//      nothing. Fix: center memories decay ~30% per round; only outer
//      piles (rarely visited by anyone) keep the slow decay.
//
//   C. Contested piles. Whoever answers faster moves first and takes 65%
//      of any contested cell, and NPCs move before the slower player too.
//      Fix: piles near a visible enemy are worth 50%, piles near a
//      visible NPC 60%, so we stop paying full price to arrive second.
//      (Latency itself was also trimmed: candidate lists instead of full
//      grid scans, and order enumeration only when the units are close
//      enough to interact.)
//
// Everything else carries over from v2: persistent obstacle/bomb/gold
// memory, value-per-step chained targeting with off-and-back re-pickup,
// dynamic k over [0,6] and both execution orders, second mover re-planning
// around the first mover's claims. No STL, no allocation, no I/O.

#include <cstring>

#include "constants.h"

namespace {

constexpr int kCells = GRID_SIZE * GRID_SIZE;
constexpr int kCenter = GRID_SIZE / 2;   // (8,8)
constexpr int kCenterLo = 4, kCenterHi = 12;  // the gold-rich central 9x9

// row/col deltas indexed by ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT.
// dir^1 is the opposite direction (used for the off-and-back re-pickup).
constexpr int kDr[4] = {-1, 1, 0, 0};
constexpr int kDc[4] = {0, 0, -1, 1};

constexpr int kGoldTTLOuter = 40;   // slow linear decay outside the center
constexpr int kBombTTL = 25;        // rounds a remembered bomb keeps blocking
constexpr int kFogBaseCost = 20;    // centi-gold per fog step: hidden walls waste moves
constexpr int kApproachCredit = 5;  // centi-gold/step for closing on a far pile
constexpr int kCenterCredit = 2;    // centi-gold/step for drifting to the center
constexpr int kMaxCand = 64;        // piles are sparse; more than this never happens

// ---------------------------------------------------------------------------
// Persistent state (survives across rounds; reset when a new match starts).
// ---------------------------------------------------------------------------
int g_last_round = -1;
unsigned char g_obstacle[GRID_SIZE][GRID_SIZE];  // ever seen an obstacle here
int g_bomb_round[GRID_SIZE][GRID_SIZE];          // last round a bomb was seen here
int g_gold_amt[GRID_SIZE][GRID_SIZE];            // last seen pile size
int g_gold_round[GRID_SIZE][GRID_SIZE];          // round of that sighting

void ResetMatchState() {
    g_last_round = -1;
    std::memset(g_obstacle, 0, sizeof(g_obstacle));
    std::memset(g_gold_amt, 0, sizeof(g_gold_amt));
    std::memset(g_gold_round, 0, sizeof(g_gold_round));
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c) g_bomb_round[r][c] = -1000000;
}

bool InBounds(int r, int c) { return r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE; }

bool InCenter(int r, int c) {
    return r >= kCenterLo && r <= kCenterHi && c >= kCenterLo && c <= kCenterHi;
}

int Abs(int x) { return x < 0 ? -x : x; }

// ceil(0.65 * x): what one entry onto a pile of x actually pays.
int Ceil65(int x) { return (65 * x + 99) / 100; }

void UpdateMemory(const GameInput* in) {
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            const int v = in->grid[r][c];
            if (v == CELL_FOG) continue;  // no news about this cell
            if (v == CELL_OBSTACLE) g_obstacle[r][c] = 1;
            if (v == CELL_BOMB) g_bomb_round[r][c] = in->round;
            else g_bomb_round[r][c] = -1000000;  // visible and bomb-free
            if (v >= 1) {
                g_gold_amt[r][c] = v;
                g_gold_round[r][c] = in->round;
            } else {
                g_gold_amt[r][c] = 0;  // visible and empty: forget any old pile
            }
        }
    }
}

// How much gold we believe sits on (r,c): exact when visible; when fogged,
// a memory decayed by how contested the ground is. The center is picked
// clean within a few rounds by the opponent and 7 NPCs; the outer ring is
// rarely visited, so memories there stay credible far longer.
int BelievedGold(const GameInput* in, int r, int c) {
    const int v = in->grid[r][c];
    if (v >= 1) return v;
    if (v != CELL_FOG || g_gold_amt[r][c] <= 0) return 0;
    const int age = in->round - g_gold_round[r][c];
    if (InCenter(r, c)) {
        if (age > 10) return 0;
        int val = g_gold_amt[r][c];
        for (int i = 0; i < age; ++i) val = val * 70 / 100;  // ~30%/round gone
        return val;
    }
    if (age > kGoldTTLOuter) return 0;
    return g_gold_amt[r][c] * (100 - (65 * age) / kGoldTTLOuter) / 100;
}

// ---------------------------------------------------------------------------
// BFS over the blocked mask. dist = -1 for unreachable cells.
// ---------------------------------------------------------------------------
struct Bfs {
    int dist[kCells];
    int prev_cell[kCells];
    int prev_dir[kCells];
};

void RunBfs(const unsigned char blocked[GRID_SIZE][GRID_SIZE], int other_cell, int start_idx,
            Bfs* b) {
    std::memset(b->dist, -1, sizeof(b->dist));
    int queue[kCells];
    int head = 0, tail = 0;
    b->dist[start_idx] = 0;
    queue[tail++] = start_idx;
    while (head < tail) {
        const int cur = queue[head++];
        const int r = cur / GRID_SIZE, c = cur % GRID_SIZE;
        for (int d = 0; d < 4; ++d) {
            const int nr = r + kDr[d], nc = c + kDc[d];
            if (!InBounds(nr, nc) || blocked[nr][nc]) continue;
            const int ni = nr * GRID_SIZE + nc;
            if (ni == other_cell || b->dist[ni] != -1) continue;
            b->dist[ni] = b->dist[cur] + 1;
            b->prev_cell[ni] = cur;
            b->prev_dir[ni] = d;
            queue[tail++] = ni;
        }
    }
}

// Path from BFS start to `target`: forward-ordered step dirs and the cell
// each step lands on. Returns length.
int ExtractPath(const Bfs& b, int start_idx, int target, int dirs[], int cells[]) {
    int len = 0;
    for (int cur = target; cur != start_idx; cur = b.prev_cell[cur]) {
        dirs[len] = b.prev_dir[cur];
        cells[len] = cur;
        ++len;
    }
    for (int i = 0; i < len / 2; ++i) {  // reverse into forward order
        int t = dirs[i]; dirs[i] = dirs[len - 1 - i]; dirs[len - 1 - i] = t;
        t = cells[i]; cells[i] = cells[len - 1 - i]; cells[len - 1 - i] = t;
    }
    return len;
}

// Fog cells a path crosses, excluding the target cell itself (gold and
// bombs never share a cell, so a believed-pile cell is safe to enter).
int FogSteps(const GameInput* in, const int cells[], int len) {
    int n = 0;
    for (int i = 0; i + 1 < len; ++i)
        if (in->grid[cells[i] / GRID_SIZE][cells[i] % GRID_SIZE] == CELL_FOG) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// Chained plan for one unit: up to `budget` moves, the cumulative value
// curve cum[s] (centi-gold after s steps, used by the k-allocator), and
// the piles it picks up along the way (used to discount the other unit).
// ---------------------------------------------------------------------------
struct Chain {
    int moves[S];
    int n;
    int cum[S + 1];
    int pick_cell[S];
    int pick_step[S];  // 1-based step count at which the pile is entered
    int npicks;
};

void RunChain(const GameInput* in, Position start, int budget,
              const unsigned char blocked[GRID_SIZE][GRID_SIZE], int other_cell,
              const unsigned char claim_pct[GRID_SIZE][GRID_SIZE], int fog_cost_centi,
              Chain* out) {
    out->n = 0;
    out->npicks = 0;
    out->cum[0] = 0;

    // Sparse candidate list with contest-discounted values. Claims by the
    // other unit scale a pile to its post-pickup remainder.
    int cand[kMaxCand];
    int cand_val[kCells];  // indexed by cell; zero everywhere but candidates
    std::memset(cand_val, 0, sizeof(cand_val));
    int ncand = 0;
    for (int r = 0; r < GRID_SIZE && ncand < kMaxCand; ++r) {
        for (int c = 0; c < GRID_SIZE && ncand < kMaxCand; ++c) {
            int val = BelievedGold(in, r, c) * claim_pct[r][c] / 100;
            if (val < 1) continue;
            for (int e = 0; e < 2; ++e) {  // enemy nearby: we may arrive second
                const Position p = in->visible_enemies[e];
                if (p.row >= 0 && Abs(p.row - r) + Abs(p.col - c) <= 4) val = val * 50 / 100;
            }
            for (int i = 0; i < in->num_visible_npcs && i < MAX_NPCS; ++i) {
                const Position p = in->visible_npcs[i].pos;
                if (p.row >= 0 && Abs(p.row - r) + Abs(p.col - c) <= 2) {
                    val = val * 60 / 100;
                    break;  // one NPC discount is enough
                }
            }
            if (val < 1) continue;
            const int idx = r * GRID_SIZE + c;
            cand[ncand++] = idx;
            cand_val[idx] = val;
        }
    }

    static Bfs bfs;  // scratch; the engine calls us single-threaded
    static int dirs[kCells], cells[kCells];
    int pos = start.row * GRID_SIZE + start.col;
    int remaining = budget;
    int value_now = 0;

    while (remaining > 0) {
        RunBfs(blocked, other_cell, pos, &bfs);

        // Best reachable pile by fog-priced pickup-per-step, plus the best
        // out-of-range pile to walk toward.
        int best = -1, best_score = 0, best_gain = 0, best_dist = 0;
        int far = -1, far_score = 0;
        for (int ci = 0; ci < ncand; ++ci) {
            const int idx = cand[ci];
            if (idx == pos || cand_val[idx] < 1 || bfs.dist[idx] <= 0) continue;
            const int pk = Ceil65(cand_val[idx]);
            if (pk < 1) continue;
            // Upper bound (zero fog penalty) prune before the path walk.
            const int d = bfs.dist[idx];
            if (pk * 100 / d <= (d <= remaining ? best_score : far_score)) continue;
            const int len = ExtractPath(bfs, pos, idx, dirs, cells);
            const int gain = pk * 100 - fog_cost_centi * FogSteps(in, cells, len);
            if (gain < 1) continue;
            const int score = gain / len;
            if (len <= remaining) {
                if (score > best_score || (score == best_score && best >= 0 && len < best_dist)) {
                    best = idx;
                    best_score = score;
                    best_gain = gain;
                    best_dist = len;
                }
            } else if (score > far_score) {
                far = idx;
                far_score = score;
            }
        }

        // Re-entering the pile underfoot: 2 moves for 65% of the remainder.
        // The step-off cell must be a KNOWN-safe neighbor — stepping off
        // into fog could hit a hidden wall and desync the return move.
        const int here_pk = Ceil65(cand_val[pos] >= 1 ? cand_val[pos] : 0);
        int osc_dir = -1;
        if (remaining >= 2 && here_pk >= 1 && here_pk * 100 / 2 > best_score) {
            const int r = pos / GRID_SIZE, c = pos % GRID_SIZE;
            for (int d = 0; d < 4; ++d) {
                const int nr = r + kDr[d], nc = c + kDc[d];
                if (InBounds(nr, nc) && !blocked[nr][nc] && in->grid[nr][nc] != CELL_FOG &&
                    nr * GRID_SIZE + nc != other_cell) {
                    osc_dir = d;
                    break;
                }
            }
        }

        if (osc_dir >= 0) {
            out->moves[out->n++] = osc_dir;
            out->cum[out->n] = value_now;
            out->moves[out->n++] = osc_dir ^ 1;  // straight back on
            value_now += here_pk * 100;
            out->cum[out->n] = value_now;
            out->pick_cell[out->npicks] = pos;
            out->pick_step[out->npicks] = out->n;
            ++out->npicks;
            cand_val[pos] -= here_pk;
            remaining -= 2;
            continue;
        }

        if (best >= 0) {
            const int len = ExtractPath(bfs, pos, best, dirs, cells);
            // Emit the path, but stop after the first step into fog (that
            // step still executes — one unknown cell per round is the most
            // we gamble; next round we re-plan with fresh vision there).
            bool truncated = false;
            int emitted = 0;
            for (int i = 0; i < len; ++i) {
                out->moves[out->n++] = dirs[i];
                ++emitted;
                const bool is_fog =
                    in->grid[cells[i] / GRID_SIZE][cells[i] % GRID_SIZE] == CELL_FOG;
                if (is_fog && cells[i] != best) {
                    truncated = true;
                    out->cum[out->n] = value_now;  // filled properly below
                    break;
                }
                out->cum[out->n] = value_now;
            }
            if (truncated) {
                // Progress credit only; the pickup didn't happen this round.
                for (int i = out->n - emitted; i < out->n; ++i)
                    out->cum[i + 1] = out->cum[i] + kApproachCredit;
                value_now = out->cum[out->n];
                break;
            }
            value_now += best_gain;
            out->cum[out->n] = value_now;
            out->pick_cell[out->npicks] = best;
            out->pick_step[out->npicks] = out->n;
            ++out->npicks;
            cand_val[best] -= Ceil65(cand_val[best]);
            remaining -= len;
            pos = best;
            continue;
        }

        if (far >= 0) {
            const int len = ExtractPath(bfs, pos, far, dirs, cells);
            const int take = remaining < len ? remaining : len;
            for (int i = 0; i < take; ++i) {
                out->moves[out->n++] = dirs[i];
                value_now += kApproachCredit;
                out->cum[out->n] = value_now;
                if (in->grid[cells[i] / GRID_SIZE][cells[i] % GRID_SIZE] == CELL_FOG)
                    break;  // frontier reached: one unknown step, then stop
            }
            break;
        }

        // No gold anywhere in sight or memory: drift toward the center,
        // where new gold spawns every round. Same one-fog-step discipline.
        int r = pos / GRID_SIZE, c = pos % GRID_SIZE;
        while (remaining > 0) {
            int best_dir = -1;
            int best_d = Abs(r - kCenter) + Abs(c - kCenter);
            for (int d = 0; d < 4; ++d) {
                const int nr = r + kDr[d], nc = c + kDc[d];
                if (!InBounds(nr, nc) || blocked[nr][nc]) continue;
                if (nr * GRID_SIZE + nc == other_cell) continue;
                const int nd = Abs(nr - kCenter) + Abs(nc - kCenter);
                if (nd < best_d) {
                    best_d = nd;
                    best_dir = d;
                }
            }
            if (best_dir < 0) break;
            r += kDr[best_dir];
            c += kDc[best_dir];
            out->moves[out->n++] = best_dir;
            value_now += kCenterCredit;
            out->cum[out->n] = value_now;
            --remaining;
            if (in->grid[r][c] == CELL_FOG) break;  // one unknown step max
        }
        break;
    }

    for (int i = out->n; i < S; ++i) out->cum[i + 1] = value_now;
}

Position EndAfter(Position start, const Chain& ch, int steps) {
    const int n = steps < ch.n ? steps : ch.n;
    for (int i = 0; i < n; ++i) {
        start.row += kDr[ch.moves[i]];
        start.col += kDc[ch.moves[i]];
    }
    return start;
}

unsigned char g_no_claims[GRID_SIZE][GRID_SIZE];  // all 100, set up on first call

}  // namespace

extern "C" GameOutput moveDecision(const GameInput* input) {
    // Baseline output is always legal even if everything below bails out.
    GameOutput out = {};
    for (int i = 0; i < S; ++i) out.actions[i] = ACT_STAY;
    out.k = S / 2;
    out.order = 0;
    out.vp = 0;
    if (input == nullptr) return out;

    // A round counter that went backwards means a new match started.
    if (input->round == 0 || input->round <= g_last_round) ResetMatchState();
    g_last_round = input->round;

    UpdateMemory(input);

    // Trampling: >= 3 NPCs on one cell costs 5% of held gold on entry.
    unsigned char npc_count[GRID_SIZE][GRID_SIZE];
    std::memset(npc_count, 0, sizeof(npc_count));
    for (int i = 0; i < input->num_visible_npcs && i < MAX_NPCS; ++i) {
        const Position p = input->visible_npcs[i].pos;
        if (InBounds(p.row, p.col)) ++npc_count[p.row][p.col];
    }

    // Cells we refuse to plan through. Fog itself is passable (but priced
    // and rationed to one step per round); known/remembered hazards block.
    unsigned char blocked[GRID_SIZE][GRID_SIZE];
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            const int v = input->grid[r][c];
            blocked[r][c] = (v == CELL_BOMB || v == CELL_OBSTACLE || g_obstacle[r][c] ||
                             input->round - g_bomb_round[r][c] <= kBombTTL ||
                             npc_count[r][c] >= 3)
                                ? 1
                                : 0;
        }
    }
    for (int i = 0; i < 2; ++i) {
        const Position e = input->visible_enemies[i];
        if (InBounds(e.row, e.col)) blocked[e.row][e.col] = 1;
    }

    if (g_no_claims[0][0] != 100) std::memset(g_no_claims, 100, sizeof(g_no_claims));

    // Fog price per step: a base charge (hidden obstacles waste moves even
    // for a broke unit) plus the expected hidden-bomb cost (~2% density x
    // 10% of held gold). Poor units explore; rich units keep to the map.
    const int fog_cost[2] = {kFogBaseCost + input->my_units_gold[0] / 5,
                             kFogBaseCost + input->my_units_gold[1] / 5};

    const int u_idx[2] = {input->my_units[0].row * GRID_SIZE + input->my_units[0].col,
                          input->my_units[1].row * GRID_SIZE + input->my_units[1].col};

    // Full-budget chains for both units, each with the other parked on its
    // current cell. Exact for whoever moves first; provisional otherwise.
    static Chain base[2], replan[2];
    RunChain(input, input->my_units[0], S, blocked, u_idx[1], g_no_claims, fog_cost[0], &base[0]);
    RunChain(input, input->my_units[1], S, blocked, u_idx[0], g_no_claims, fog_cost[1], &base[1]);

    // Execution order only matters when the units can actually interact
    // this round; when they are far apart, skip the second enumeration.
    const int unit_gap = Abs(input->my_units[0].row - input->my_units[1].row) +
                         Abs(input->my_units[0].col - input->my_units[1].col);
    const int num_orders = unit_gap > 2 * S ? 1 : 2;

    int best_total = -1, best_k = S / 2, best_order = 0;
    for (int ord = 0; ord < num_orders; ++ord) {
        const int f = ord;      // unit index moving first
        const int s = 1 - ord;  // unit index moving second
        int k_pick = S / 2, k_total = -1;
        for (int k = 0; k <= S; ++k) {
            const int fb = (f == 0) ? k : S - k;  // first mover's budget
            const int total = base[f].cum[fb] + base[s].cum[S - fb];
            if (total > k_total || (total == k_total && Abs(k - S / 2) < Abs(k_pick - S / 2))) {
                k_total = total;
                k_pick = k;
            }
        }
        const int fb = (f == 0) ? k_pick : S - k_pick;
        const Position f_end = EndAfter(input->my_units[f], base[f], fb);

        unsigned char claims[GRID_SIZE][GRID_SIZE];
        std::memset(claims, 100, sizeof(claims));
        for (int i = 0; i < base[f].npicks && base[f].pick_step[i] <= fb; ++i) {
            const int r = base[f].pick_cell[i] / GRID_SIZE, c = base[f].pick_cell[i] % GRID_SIZE;
            claims[r][c] = static_cast<unsigned char>(claims[r][c] * 35 / 100);
        }
        RunChain(input, input->my_units[s], S - fb, blocked,
                 f_end.row * GRID_SIZE + f_end.col, claims, fog_cost[s], &replan[s]);

        const int realized = base[f].cum[fb] + replan[s].cum[S - fb];
        if (realized > best_total) {
            best_total = realized;
            best_k = k_pick;
            best_order = ord;
        }
    }

    // Assemble: unit 0 owns actions[0..k), unit 1 owns actions[k..6),
    // regardless of who executes first (that's what `order` is for).
    out.k = best_k;
    out.order = best_order;
    const Chain& plan0 = (best_order == 0) ? base[0] : replan[0];
    const Chain& plan1 = (best_order == 0) ? replan[1] : base[1];
    for (int i = 0; i < best_k && i < plan0.n; ++i) out.actions[i] = plan0.moves[i];
    for (int i = 0; i < S - best_k && i < plan1.n; ++i) out.actions[best_k + i] = plan1.moves[i];
    return out;
}

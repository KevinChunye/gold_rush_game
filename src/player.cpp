// player.cpp — GoldRush 2.0 bot, v2 "memory + value chains".
//
// v1 proved the pipeline: always-legal output, microsecond decisions.
// v2 keeps both guarantees and changes the decision architecture from
// "nearest gold by BFS" to "expected-value maximization":
//
//   1. Belief memory across rounds (the engine keeps this .so loaded):
//      - obstacles: permanent (they never move),
//      - bombs: feared for ~one respawn window after last sighting,
//      - gold piles: remembered with a decaying belief (someone else
//        may grab them while we are away).
//   2. Fog is traversable, priced by wealth. A hidden obstacle only
//      wastes a step (the engine skips illegal moves), but a hidden bomb
//      costs 10% of held gold — so crossing fog charges a candidate
//      target in proportion to the unit's gold. Poor units explore,
//      rich units stay on charted ground.
//   3. Value-aware chained targeting: each unit repeatedly takes the
//      option with the best expected pickup-per-step — a reachable pile,
//      or stepping off and back onto the pile it stands on (each entry
//      pays 65% of what remains) — then continues with leftover moves.
//   4. Dynamic split and order: the 6 moves go to whichever unit turns
//      them into more value (k chosen over [0,6] from both units' value
//      curves), both execution orders are tried, and the second mover
//      re-plans around what the first will already have grabbed.
//
// Still no STL, no allocation, no I/O; a round costs a few microseconds.

#include <cstring>

#include "constants.h"

namespace {

constexpr int kCells = GRID_SIZE * GRID_SIZE;
constexpr int kCenter = GRID_SIZE / 2;  // (8,8)

// row/col deltas indexed by ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT.
// dir^1 is the opposite direction (used for the off-and-back re-pickup).
constexpr int kDr[4] = {-1, 1, 0, 0};
constexpr int kDc[4] = {0, 0, -1, 1};

constexpr int kGoldTTL = 40;        // rounds until a remembered pile is written off
constexpr int kBombTTL = 25;        // rounds a remembered bomb keeps blocking
constexpr int kApproachCredit = 5;  // centi-gold/step for closing on a far pile
constexpr int kCenterCredit = 2;    // centi-gold/step for drifting to the center

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

// How much gold we believe sits on (r,c): exact when visible, a decaying
// memory when fogged (linear fade to ~35% credibility over kGoldTTL rounds).
int BelievedGold(const GameInput* in, int r, int c) {
    const int v = in->grid[r][c];
    if (v >= 1) return v;
    if (v == CELL_FOG && g_gold_amt[r][c] > 0) {
        const int age = in->round - g_gold_round[r][c];
        if (age > kGoldTTL) return 0;
        return g_gold_amt[r][c] * (100 - (65 * age) / kGoldTTL) / 100;
    }
    return 0;
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

// Fog cells crossed on the way to `target` (excluding the target itself:
// gold and bombs never share a cell, so the pile cell is safe to enter).
int FogSteps(const GameInput* in, const Bfs& b, int start_idx, int target) {
    int n = 0;
    for (int cur = b.prev_cell[target]; cur != start_idx; cur = b.prev_cell[cur])
        if (in->grid[cur / GRID_SIZE][cur % GRID_SIZE] == CELL_FOG) ++n;
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

    // What each cell is worth to this unit (piles the other unit reaches
    // first are scaled down to their post-pickup remainder).
    int value_left[kCells];
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c)
            value_left[r * GRID_SIZE + c] = BelievedGold(in, r, c) * claim_pct[r][c] / 100;

    static Bfs bfs;  // scratch; the engine calls us single-threaded
    int pos = start.row * GRID_SIZE + start.col;
    int remaining = budget;
    int value_now = 0;

    while (remaining > 0) {
        RunBfs(blocked, other_cell, pos, &bfs);

        // Best reachable pile by (fog-priced) pickup-per-step, best far
        // pile to walk toward, and the re-pickup of the pile underfoot.
        int best = -1, best_score = 0, best_dist = 0, best_gain = 0;
        int far = -1, far_score = 0;
        for (int idx = 0; idx < kCells; ++idx) {
            if (idx == pos || value_left[idx] < 1 || bfs.dist[idx] <= 0) continue;
            const int pk = Ceil65(value_left[idx]);
            if (pk < 1) continue;
            const int gain = pk * 100 - fog_cost_centi * FogSteps(in, bfs, pos, idx);
            if (gain < 1) continue;
            const int score = gain / bfs.dist[idx];
            if (bfs.dist[idx] <= remaining) {
                if (score > best_score ||
                    (score == best_score && best >= 0 && bfs.dist[idx] < best_dist)) {
                    best = idx;
                    best_score = score;
                    best_dist = bfs.dist[idx];
                    best_gain = gain;
                }
            } else if (score > far_score) {
                far = idx;
                far_score = score;
            }
        }

        // Re-entering the pile underfoot costs 2 moves for 65% of the rest.
        const int here_pk = Ceil65(value_left[pos]);
        int osc_dir = -1;
        if (remaining >= 2 && here_pk >= 1 && here_pk * 100 / 2 > best_score) {
            const int r = pos / GRID_SIZE, c = pos % GRID_SIZE;
            for (int d = 0; d < 4; ++d) {
                const int nr = r + kDr[d], nc = c + kDc[d];
                if (InBounds(nr, nc) && !blocked[nr][nc] && nr * GRID_SIZE + nc != other_cell) {
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
            value_left[pos] -= here_pk;
            remaining -= 2;
            continue;
        }

        if (best >= 0) {
            // Walk to the pile; its value is credited on the arrival step.
            int path[kCells];
            int len = 0;
            for (int cur = best; cur != pos; cur = bfs.prev_cell[cur]) path[len++] = bfs.prev_dir[cur];
            for (int i = len - 1; i >= 0; --i) {
                out->moves[out->n++] = path[i];
                out->cum[out->n] = value_now;
            }
            value_now += best_gain;
            out->cum[out->n] = value_now;
            out->pick_cell[out->npicks] = best;
            out->pick_step[out->npicks] = out->n;
            ++out->npicks;
            value_left[best] -= Ceil65(value_left[best]);
            remaining -= len;
            pos = best;
            continue;
        }

        if (far >= 0) {
            // Close the distance to the best out-of-range pile.
            int path[kCells];
            int len = 0;
            for (int cur = far; cur != pos; cur = bfs.prev_cell[cur]) path[len++] = bfs.prev_dir[cur];
            const int take = remaining < len ? remaining : len;
            for (int i = 0; i < take; ++i) {
                out->moves[out->n++] = path[len - 1 - i];
                value_now += kApproachCredit;
                out->cum[out->n] = value_now;
            }
            remaining = 0;
            break;
        }

        // No gold anywhere in sight or memory: drift toward the center,
        // where new gold spawns every round.
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

    // Cells we refuse to plan through. Fog itself is passable now; known
    // and remembered hazards still block.
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

    // Expected bomb cost per fog step, per unit: ~2% bomb density x 10% of
    // held gold => gold/5 centi-gold. Poor units explore, rich ones don't.
    const int fog_cost[2] = {input->my_units_gold[0] / 5, input->my_units_gold[1] / 5};

    const int u_idx[2] = {input->my_units[0].row * GRID_SIZE + input->my_units[0].col,
                          input->my_units[1].row * GRID_SIZE + input->my_units[1].col};

    // Full-budget chains for both units, each with the other parked on its
    // current cell. Exact for whoever moves first; provisional otherwise.
    static Chain base[2], replan[2];
    RunChain(input, input->my_units[0], S, blocked, u_idx[1], g_no_claims, fog_cost[0], &base[0]);
    RunChain(input, input->my_units[1], S, blocked, u_idx[0], g_no_claims, fog_cost[1], &base[1]);

    // Try both execution orders; for each, pick the split k maximizing the
    // combined curves, then re-plan the second mover around the first
    // mover's end cell and claimed piles. Keep whichever order realizes more.
    int best_total = -1, best_k = S / 2, best_order = 0;
    for (int ord = 0; ord < 2; ++ord) {
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

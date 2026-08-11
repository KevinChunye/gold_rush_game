// player.cpp — GoldRush 2.0 bot, v2.5 "seat-adaptive, at speed".
//
// Same decisions as v2.4, engineered for latency. The engine executes the
// faster-answering bot's moves first each round (nanosecond comparison),
// then the NPCs, then the slower bot — so latency is part of the game:
// staying under the opponent's decision time wins first pick of every
// contested pile. v2.4 played the right moves at ~20us; v2.5 plays the
// same moves several times faster:
//
//   - One BFS per pile-hop, and none at all for re-pickup oscillations
//     (position doesn't change, so the previous search stays valid).
//   - Epoch-stamped search state: no per-search memset of dist arrays.
//   - The candidate pile list is built once per round and shared by all
//     plans (claims applied as per-plan multipliers on the small list).
//   - Order and split are chosen from the two base value curves; the
//     second mover is re-planned only when the plans actually collide
//     (shared pile or path through the first mover's final cell).
//   - Pile-hops per plan capped at 3, search depth capped at 14 steps —
//     bounding the worst-case round near the typical one (P90 hygiene).
//   - A dlopen-time constructor runs two synthetic decisions to fault in
//     pages and warm the code path: the official FAQ counts the FIRST
//     moveDecision (including any lazy init) toward P90, and loading has
//     its own generous 10 s budget — so we pay the cold-start there.
//
// Strategy (unchanged from v2.4): persistent obstacle/bomb/gold memory
// with seat-dependent decay, value-per-step chained targeting with
// off-and-back re-pickup, dynamic k and order, and seat inference —
// auditing planned-vs-realized pickups to detect whether we move first
// (aggressive profile) or last (defensive profile).
// No STL, no allocation, no I/O.

#include <cstring>

#include "constants.h"

namespace {

constexpr int kCells = GRID_SIZE * GRID_SIZE;
constexpr int kCenter = GRID_SIZE / 2;        // (8,8)
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
constexpr int kMaxHops = 3;         // pile-hops per plan (bounds worst-case latency)
constexpr int kBfsDepth = 14;       // search horizon; farther is next round's problem

// Wealth threshold for second-seat fog rationing (see fog policy below).
constexpr int kExploreCap = 30;

// ---------------------------------------------------------------------------
// Persistent state (survives across rounds; reset when a new match starts).
// ---------------------------------------------------------------------------
int g_last_round = -1;
unsigned char g_obstacle[GRID_SIZE][GRID_SIZE];  // ever seen an obstacle here
int g_bomb_round[GRID_SIZE][GRID_SIZE];          // last round a bomb was seen here
int g_gold_amt[GRID_SIZE][GRID_SIZE];            // last seen pile size
int g_gold_round[GRID_SIZE][GRID_SIZE];          // round of that sighting

// Seat inference. Forced-order simulator experiments showed the two seats
// want opposite strategies (first mover: aggressive, ~1863 net; second
// mover: defensive, aggressive play collapses to ~356). The seat is not
// in the API but it is observable: when we move first, piles we planned
// to grab are still there; when we move last they keep vanishing first.
bool g_second_seat = false;
int g_seat_ema = 0;        // 0..100, exponential average of shortfall rounds
int g_prev_gold_sum = -1;  // our total gold after last round's plan
int g_planned_gain = 0;    // visible-pile pickups last plan promised

void ResetMatchState() {
    g_last_round = -1;
    g_second_seat = false;
    g_seat_ema = 0;
    g_prev_gold_sum = -1;
    g_planned_gain = 0;
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

// How much gold we believe sits on (r,c): exact when visible; when fogged,
// a decayed memory. Second seat: the opponent and NPCs sweep the center
// before we ever move, so center memories rot ~30% per round there.
int BelievedGold(const GameInput* in, int r, int c) {
    const int v = in->grid[r][c];
    if (v >= 1) return v;
    if (v != CELL_FOG || g_gold_amt[r][c] <= 0) return 0;
    const int age = in->round - g_gold_round[r][c];
    if (g_second_seat && InCenter(r, c)) {
        if (age > 10) return 0;
        int val = g_gold_amt[r][c];
        for (int i = 0; i < age; ++i) val = val * 70 / 100;
        return val;
    }
    if (age > kGoldTTLOuter) return 0;
    return g_gold_amt[r][c] * (100 - (65 * age) / kGoldTTLOuter) / 100;
}

// ---------------------------------------------------------------------------
// BFS scratch, epoch-stamped so no per-search clearing is needed. dist and
// parents are valid for a cell only when its stamp matches the epoch.
// ---------------------------------------------------------------------------
int g_dist[kCells];
int g_prev_cell[kCells];
int g_prev_dir[kCells];
int g_stamp[kCells];  // zero-initialized; epoch starts above zero
int g_epoch = 0;
int g_queue[kCells];

int BfsDist(int idx) { return g_stamp[idx] == g_epoch ? g_dist[idx] : -1; }

void RunBfs(const unsigned char blocked[GRID_SIZE][GRID_SIZE], int other_cell, int start_idx) {
    ++g_epoch;
    int head = 0, tail = 0;
    g_dist[start_idx] = 0;
    g_stamp[start_idx] = g_epoch;
    g_queue[tail++] = start_idx;
    while (head < tail) {
        const int cur = g_queue[head++];
        const int d = g_dist[cur];
        if (d >= kBfsDepth) continue;
        const int r = cur / GRID_SIZE, c = cur % GRID_SIZE;
        for (int k = 0; k < 4; ++k) {
            const int nr = r + kDr[k], nc = c + kDc[k];
            if (!InBounds(nr, nc) || blocked[nr][nc]) continue;
            const int ni = nr * GRID_SIZE + nc;
            if (ni == other_cell || g_stamp[ni] == g_epoch) continue;
            g_stamp[ni] = g_epoch;
            g_dist[ni] = d + 1;
            g_prev_cell[ni] = cur;
            g_prev_dir[ni] = k;
            g_queue[tail++] = ni;
        }
    }
}

// Path from the BFS start to `target`: forward-ordered step dirs and the
// cell each step lands on. Returns length. Valid only for reached targets.
int ExtractPath(int start_idx, int target, int dirs[], int cells[]) {
    int len = 0;
    for (int cur = target; cur != start_idx; cur = g_prev_cell[cur]) {
        dirs[len] = g_prev_dir[cur];
        cells[len] = cur;
        ++len;
    }
    for (int i = 0; i < len / 2; ++i) {
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
// Round-level shared context: the candidate pile list (with seat-dependent
// contest discounts already applied) is built once and shared by all plans.
// ---------------------------------------------------------------------------
int g_cand_idx[kMaxCand];
int g_cand_val[kMaxCand];
int g_ncand = 0;

void BuildCandidates(const GameInput* in) {
    g_ncand = 0;
    for (int r = 0; r < GRID_SIZE && g_ncand < kMaxCand; ++r) {
        for (int c = 0; c < GRID_SIZE && g_ncand < kMaxCand; ++c) {
            int val = BelievedGold(in, r, c);
            if (val < 1) continue;
            if (g_second_seat) {  // moving last: contested piles get swept
                for (int e = 0; e < 2; ++e) {
                    const Position p = in->visible_enemies[e];
                    if (p.row >= 0 && Abs(p.row - r) + Abs(p.col - c) <= 4) val = val * 50 / 100;
                }
                for (int i = 0; i < in->num_visible_npcs && i < MAX_NPCS; ++i) {
                    const Position p = in->visible_npcs[i].pos;
                    if (p.row >= 0 && Abs(p.row - r) + Abs(p.col - c) <= 2) {
                        val = val * 60 / 100;
                        break;
                    }
                }
                if (val < 1) continue;
            }
            g_cand_idx[g_ncand] = r * GRID_SIZE + c;
            g_cand_val[g_ncand] = val;
            ++g_ncand;
        }
    }
}

// ---------------------------------------------------------------------------
// Chained plan for one unit: up to `budget` moves, the cumulative value
// curve cum[s] (centi-gold after s steps, for the k-allocator), the cells
// stepped on (for collision tests), and the piles picked along the way.
// ---------------------------------------------------------------------------
struct Chain {
    int moves[S];
    int n;
    int cum[S + 1];
    int cellseq[S];    // cell landed on after each emitted move
    int pick_cell[S];
    int pick_step[S];  // 1-based step count at which the pile is entered
    int npicks;
};

// claim_pct: nullptr = no claims; otherwise percent left per cell.
void RunChain(const GameInput* in, Position start, int budget,
              const unsigned char blocked[GRID_SIZE][GRID_SIZE], int other_cell,
              const unsigned char (*claim_pct)[GRID_SIZE], int fog_cost_centi,
              bool full_emit, Chain* out) {
    out->n = 0;
    out->npicks = 0;
    out->cum[0] = 0;

    int val[kMaxCand];
    for (int i = 0; i < g_ncand; ++i) {
        const int idx = g_cand_idx[i];
        val[i] = claim_pct ? g_cand_val[i] * claim_pct[idx / GRID_SIZE][idx % GRID_SIZE] / 100
                           : g_cand_val[i];
    }

    static int dirs[kCells], cells[kCells];
    int pos = start.row * GRID_SIZE + start.col;
    int remaining = budget;
    int value_now = 0;
    int hops = 0;
    bool bfs_valid = false;

    while (remaining > 0) {
        if (!bfs_valid) {
            RunBfs(blocked, other_cell, pos);
            bfs_valid = true;
        }

        // Best reachable pile by fog-priced pickup-per-step, and the best
        // out-of-range pile to walk toward.
        int best = -1, best_ci = -1, best_score = 0, best_gain = 0, best_dist = 0;
        int far = -1, far_score = 0;
        for (int ci = 0; ci < g_ncand; ++ci) {
            const int idx = g_cand_idx[ci];
            if (idx == pos || val[ci] < 1) continue;
            const int d = BfsDist(idx);
            if (d <= 0) continue;
            const int pk = Ceil65(val[ci]);
            if (pk < 1) continue;
            // Upper bound (zero fog penalty) prune before the path walk.
            if (pk * 100 / d <= (d <= remaining ? best_score : far_score)) continue;
            const int len = ExtractPath(pos, idx, dirs, cells);
            const int gain = pk * 100 - fog_cost_centi * FogSteps(in, cells, len);
            if (gain < 1) continue;
            const int score = gain / len;
            if (len <= remaining) {
                if (score > best_score || (score == best_score && best >= 0 && len < best_dist)) {
                    best = idx;
                    best_ci = ci;
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
        // The step-off cell must be KNOWN-safe (never fog): a hidden wall
        // would desync the return move. Costs no BFS — position is kept.
        int here_ci = -1;
        for (int ci = 0; ci < g_ncand; ++ci)
            if (g_cand_idx[ci] == pos) { here_ci = ci; break; }
        const int here_pk = here_ci >= 0 ? Ceil65(val[here_ci]) : 0;
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
            const int r = pos / GRID_SIZE, c = pos % GRID_SIZE;
            out->cellseq[out->n] = (r + kDr[osc_dir]) * GRID_SIZE + (c + kDc[osc_dir]);
            out->moves[out->n++] = osc_dir;
            out->cum[out->n] = value_now;
            out->cellseq[out->n] = pos;
            out->moves[out->n++] = osc_dir ^ 1;  // straight back on
            value_now += here_pk * 100;
            out->cum[out->n] = value_now;
            out->pick_cell[out->npicks] = pos;
            out->pick_step[out->npicks] = out->n;
            ++out->npicks;
            val[here_ci] -= here_pk;
            remaining -= 2;
            continue;  // BFS still valid: we ended where we started
        }

        if (best >= 0) {
            const int len = ExtractPath(pos, best, dirs, cells);
            // Emit the path; in the defensive profile stop after the first
            // step into fog (one unknown cell per round is all we gamble).
            bool truncated = false;
            int emitted = 0;
            for (int i = 0; i < len; ++i) {
                out->cellseq[out->n] = cells[i];
                out->moves[out->n++] = dirs[i];
                ++emitted;
                const bool is_fog =
                    in->grid[cells[i] / GRID_SIZE][cells[i] % GRID_SIZE] == CELL_FOG;
                out->cum[out->n] = value_now;
                if (!full_emit && is_fog && cells[i] != best) {
                    truncated = true;
                    break;
                }
            }
            if (truncated) {
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
            val[best_ci] -= Ceil65(val[best_ci]);
            remaining -= len;
            pos = best;
            bfs_valid = false;
            if (++hops >= kMaxHops) break;
            continue;
        }

        if (far >= 0) {
            const int len = ExtractPath(pos, far, dirs, cells);
            const int take = remaining < len ? remaining : len;
            for (int i = 0; i < take; ++i) {
                out->cellseq[out->n] = cells[i];
                out->moves[out->n++] = dirs[i];
                value_now += kApproachCredit;
                out->cum[out->n] = value_now;
                if (!full_emit &&
                    in->grid[cells[i] / GRID_SIZE][cells[i] % GRID_SIZE] == CELL_FOG)
                    break;  // frontier reached: one unknown step, then stop
            }
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
            out->cellseq[out->n] = r * GRID_SIZE + c;
            out->moves[out->n++] = best_dir;
            value_now += kCenterCredit;
            out->cum[out->n] = value_now;
            --remaining;
            if (!full_emit && in->grid[r][c] == CELL_FOG) break;  // one unknown step max
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

    // Seat inference: did last round's promised visible pickups arrive?
    const int gold_sum = input->my_units_gold[0] + input->my_units_gold[1];
    if (g_prev_gold_sum >= 0 && g_planned_gain > 0) {
        const int realized = gold_sum - g_prev_gold_sum;
        g_seat_ema = (7 * g_seat_ema + (realized < g_planned_gain ? 100 : 0)) / 8;
        g_second_seat = g_seat_ema >= 35;
    }

    // Trampling: >= 3 NPCs on one cell costs 5% of held gold on entry.
    unsigned char npc_count[GRID_SIZE][GRID_SIZE];
    std::memset(npc_count, 0, sizeof(npc_count));
    for (int i = 0; i < input->num_visible_npcs && i < MAX_NPCS; ++i) {
        const Position p = input->visible_npcs[i].pos;
        if (InBounds(p.row, p.col)) ++npc_count[p.row][p.col];
    }

    // One fused pass: update persistent memory from the visible grid and
    // build the blocked mask. Fog is passable (priced/rationed by the fog
    // policy); known and remembered hazards block.
    unsigned char blocked[GRID_SIZE][GRID_SIZE];
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            const int v = input->grid[r][c];
            if (v != CELL_FOG) {
                if (v == CELL_OBSTACLE) g_obstacle[r][c] = 1;
                if (v == CELL_BOMB) g_bomb_round[r][c] = input->round;
                else g_bomb_round[r][c] = -1000000;
                if (v >= 1) {
                    g_gold_amt[r][c] = v;
                    g_gold_round[r][c] = input->round;
                } else {
                    g_gold_amt[r][c] = 0;
                }
            }
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

    BuildCandidates(input);

    // Fog policy per unit, by seat profile. First seat: explore freely at
    // a small wealth-scaled price (our information is fresh — we move
    // before everyone; forced-order experiments showed aggression beating
    // caution by ~50% here). Second seat: poor units still explore, rich
    // units pay ~1% of held gold per fog step and get path truncation.
    bool explore[2];
    int fog_cost[2];
    for (int u = 0; u < 2; ++u) {
        const int g = input->my_units_gold[u];
        if (!g_second_seat) {
            explore[u] = true;
            fog_cost[u] = g / 5;
        } else {
            explore[u] = g < kExploreCap;
            fog_cost[u] = explore[u] ? kFogBaseCost + g / 5 : kFogBaseCost + g;
        }
    }

    const int u_idx[2] = {input->my_units[0].row * GRID_SIZE + input->my_units[0].col,
                          input->my_units[1].row * GRID_SIZE + input->my_units[1].col};

    // Full-budget chains for both units, each with the other parked on its
    // current cell. Exact for whoever moves first; provisional otherwise.
    static Chain base[2], replan;
    RunChain(input, input->my_units[0], S, blocked, u_idx[1], nullptr, fog_cost[0],
             explore[0], &base[0]);
    RunChain(input, input->my_units[1], S, blocked, u_idx[0], nullptr, fog_cost[1],
             explore[1], &base[1]);

    // Choose order and split from the two base value curves (no extra
    // planning): for each order, the best k; keep the better order.
    int best_k = S / 2, best_order = 0, best_total = -1;
    for (int ord = 0; ord < 2; ++ord) {
        const int f = ord;
        for (int k = 0; k <= S; ++k) {
            const int fb = (f == 0) ? k : S - k;
            const int total = base[f].cum[fb] + base[1 - f].cum[S - fb];
            if (total > best_total ||
                (total == best_total && Abs(k - S / 2) < Abs(best_k - S / 2))) {
                best_total = total;
                best_k = k;
                best_order = ord;
            }
        }
    }
    const int f = best_order, s = 1 - best_order;
    const int fb = (f == 0) ? best_k : S - best_k;
    const int sb = S - fb;
    const Position f_end = EndAfter(input->my_units[f], base[f], fb);
    const int f_end_idx = f_end.row * GRID_SIZE + f_end.col;

    // Re-plan the second mover only if the base plans actually collide:
    // it walks through the first mover's final cell, or they picked the
    // same pile within their executed prefixes.
    bool collide = false;
    const int sn = sb < base[s].n ? sb : base[s].n;
    for (int i = 0; i < sn && !collide; ++i)
        if (base[s].cellseq[i] == f_end_idx) collide = true;
    for (int i = 0; i < base[f].npicks && base[f].pick_step[i] <= fb && !collide; ++i)
        for (int j = 0; j < base[s].npicks && base[s].pick_step[j] <= sb; ++j)
            if (base[f].pick_cell[i] == base[s].pick_cell[j]) { collide = true; break; }

    const Chain* plan_s = &base[s];
    if (collide) {
        unsigned char claims[GRID_SIZE][GRID_SIZE];
        std::memset(claims, 100, sizeof(claims));
        for (int i = 0; i < base[f].npicks && base[f].pick_step[i] <= fb; ++i) {
            const int r = base[f].pick_cell[i] / GRID_SIZE, c = base[f].pick_cell[i] % GRID_SIZE;
            claims[r][c] = static_cast<unsigned char>(claims[r][c] * 35 / 100);
        }
        RunChain(input, input->my_units[s], sb, blocked, f_end_idx, claims, fog_cost[s],
                 explore[s], &replan);
        plan_s = &replan;
    }

    // Assemble: unit 0 owns actions[0..k), unit 1 owns actions[k..6),
    // regardless of who executes first (that's what `order` is for).
    out.k = best_k;
    out.order = best_order;
    const Chain* plan0 = (f == 0) ? &base[0] : plan_s;
    const Chain* plan1 = (f == 0) ? plan_s : &base[1];
    for (int i = 0; i < best_k && i < plan0->n; ++i) out.actions[i] = plan0->moves[i];
    for (int i = 0; i < S - best_k && i < plan1->n; ++i) out.actions[best_k + i] = plan1->moves[i];

    // Remember what this plan promises so next round can audit the seat:
    // each unit's first pickup that targets a currently VISIBLE pile.
    g_planned_gain = 0;
    const Chain* plans[2] = {plan0, plan1};
    const int budgets[2] = {best_k, S - best_k};
    for (int u = 0; u < 2; ++u) {
        for (int i = 0; i < plans[u]->npicks && plans[u]->pick_step[i] <= budgets[u]; ++i) {
            const int r = plans[u]->pick_cell[i] / GRID_SIZE;
            const int c = plans[u]->pick_cell[i] % GRID_SIZE;
            if (input->grid[r][c] >= 1) {
                g_planned_gain += Ceil65(input->grid[r][c]);
                break;
            }
        }
    }
    g_prev_gold_sum = gold_sum;
    return out;
}

namespace {

// dlopen-time warmup: fault in pages, resolve lazy bindings, and warm the
// code path so the FIRST real moveDecision (which counts toward P90 per
// the official FAQ) runs at steady-state speed. Loading has its own 10 s
// budget, so this is free. State is wiped afterwards.
__attribute__((constructor)) void WarmUp() {
    static GameInput fake;  // zeroed static: the struct is too big for stack comfort
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c) fake.grid[r][c] = CELL_FOG;
    fake.my_units[0] = {0, 0};
    fake.my_units[1] = {GRID_SIZE - 1, GRID_SIZE - 1};
    fake.visible_enemies[0] = {-1, -1};
    fake.visible_enemies[1] = {-1, -1};
    for (int i = 0; i < MAX_NPCS; ++i) fake.visible_npcs[i] = {0, {-1, -1}};
    fake.snapshot.window_begin = -1;
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c) fake.grid[r][c] = (r == 2 && c == 2) ? 7 : CELL_EMPTY;
    fake.round = 0;
    moveDecision(&fake);
    fake.round = 1;
    moveDecision(&fake);
    ResetMatchState();
}

}  // namespace

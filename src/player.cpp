// player.cpp — GoldRush 2.0 MVP bot.
//
// Deliberately simple: the goal of this version is a submission that never
// crashes, never times out, and always returns a legal GameOutput, so the
// whole upload -> match -> replay pipeline can be validated end to end.
//
// Strategy per round:
//   1. Each unit gets 3 of the 6 shared moves (k = 3).
//   2. BFS over the currently visible, safe cells to the nearest gold pile;
//      the two units avoid picking the same pile when an alternative exists.
//   3. If a unit sees no reachable gold, it walks toward the map center
//      (8,8), where gold spawns every round.
//   4. Cells treated as walls: fog, obstacles (also remembered across
//      rounds — the map is static), bombs, visible enemy units, cells
//      holding >= 3 NPCs (trampling penalty), and our own other unit.
//   5. No vision purchase (vp = 0), snapshot ignored.
//
// No STL, no allocation, no I/O: a round costs a few microseconds, which
// also keeps our P90 latency (the tiebreaker + speed prize metric) tiny.

#include <cstring>

#include "game_api.h"

namespace {

constexpr int kCells = GRID_N * GRID_N;
constexpr int kStepsPerUnit = MAX_MOVES / 2;  // 3 moves per unit
constexpr int kCenter = GRID_N / 2;           // (8,8)

// row/col deltas indexed by ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT
constexpr int kDr[4] = {-1, 1, 0, 0};
constexpr int kDc[4] = {0, 0, -1, 1};

// ---------------------------------------------------------------------------
// Persistent state. The engine keeps player.so loaded for the entire match
// (and possibly across matches), so globals survive between rounds. We use
// that to remember obstacles, which are fixed for a given map.
// ---------------------------------------------------------------------------
int g_last_round = -1;
unsigned char g_known_obstacle[GRID_N][GRID_N];

void ResetMatchState() {
    g_last_round = -1;
    std::memset(g_known_obstacle, 0, sizeof(g_known_obstacle));
}

bool InBounds(int r, int c) { return r >= 0 && r < GRID_N && c >= 0 && c < GRID_N; }

int Abs(int x) { return x < 0 ? -x : x; }

// BFS from `start` over non-blocked cells to the nearest gold cell
// (optionally skipping `avoid_target`, so our two units spread out).
// Writes up to `steps` moves into `moves`, returns how many were written,
// or -1 if no gold is reachable. `target_out` reports the chosen pile.
int PlanTowardGold(const GameInput* in, Position start,
                   const unsigned char blocked[GRID_N][GRID_N],
                   int avoid_target, int steps, int moves[],
                   int* target_out) {
    int dist[kCells];
    int prev_cell[kCells];
    int prev_dir[kCells];
    int queue[kCells];
    std::memset(dist, -1, sizeof(dist));

    int head = 0, tail = 0;
    const int start_idx = start.row * GRID_N + start.col;
    dist[start_idx] = 0;
    queue[tail++] = start_idx;

    int target = -1;
    while (head < tail) {
        const int cur = queue[head++];
        const int r = cur / GRID_N, c = cur % GRID_N;
        if (cur != start_idx && in->grid[r][c] >= 1 && cur != avoid_target) {
            target = cur;  // BFS order => nearest pile
            break;
        }
        for (int d = 0; d < 4; ++d) {
            const int nr = r + kDr[d], nc = c + kDc[d];
            if (!InBounds(nr, nc) || blocked[nr][nc]) continue;
            const int ni = nr * GRID_N + nc;
            if (dist[ni] != -1) continue;
            dist[ni] = dist[cur] + 1;
            prev_cell[ni] = cur;
            prev_dir[ni] = d;
            queue[tail++] = ni;
        }
    }
    if (target < 0) return -1;
    *target_out = target;

    // Walk the parent chain back to the start to recover the move list.
    int path[kCells];
    int len = 0;
    for (int cur = target; cur != start_idx; cur = prev_cell[cur]) path[len++] = prev_dir[cur];
    const int n = len < steps ? len : steps;
    for (int i = 0; i < n; ++i) moves[i] = path[len - 1 - i];
    return n;
}

// Fallback when no gold is visible: greedy-walk toward the center, only
// onto cells known to be safe, stopping early if boxed in.
int PlanTowardCenter(const unsigned char blocked[GRID_N][GRID_N], Position start,
                     int steps, int moves[]) {
    int r = start.row, c = start.col;
    int n = 0;
    for (int s = 0; s < steps; ++s) {
        int best_dir = -1;
        int best_dist = Abs(r - kCenter) + Abs(c - kCenter);
        for (int d = 0; d < 4; ++d) {
            const int nr = r + kDr[d], nc = c + kDc[d];
            if (!InBounds(nr, nc) || blocked[nr][nc]) continue;
            const int nd = Abs(nr - kCenter) + Abs(nc - kCenter);
            if (nd < best_dist) {
                best_dist = nd;
                best_dir = d;
            }
        }
        if (best_dir < 0) break;  // no safe step that makes progress
        r += kDr[best_dir];
        c += kDc[best_dir];
        moves[n++] = best_dir;
    }
    return n;
}

// Plan one unit: gold first, center fallback. Fills exactly `steps` slots
// (padding with STAY) and returns the unit's final position.
Position PlanUnit(const GameInput* in, Position start,
                  const unsigned char blocked[GRID_N][GRID_N],
                  int avoid_target, int steps, int out_actions[],
                  int* target_out) {
    int moves[MAX_MOVES];
    int n = PlanTowardGold(in, start, blocked, avoid_target, steps, moves, target_out);
    if (n < 0 && avoid_target >= 0) {
        // Only one pile in sight: sharing it still beats idling
        // (each pickup takes 65% of whatever is left on the cell).
        n = PlanTowardGold(in, start, blocked, -1, steps, moves, target_out);
    }
    if (n < 0) {
        *target_out = -1;
        n = PlanTowardCenter(blocked, start, steps, moves);
    }
    Position end = start;
    for (int i = 0; i < steps; ++i) {
        if (i < n) {
            out_actions[i] = moves[i];
            end.row += kDr[moves[i]];
            end.col += kDc[moves[i]];
        } else {
            out_actions[i] = ACT_STAY;
        }
    }
    return end;
}

}  // namespace

extern "C" GameOutput moveDecision(const GameInput* input) {
    // Baseline output is always legal even if everything below bails out.
    GameOutput out = {};
    for (int i = 0; i < MAX_MOVES; ++i) out.actions[i] = ACT_STAY;
    out.k = kStepsPerUnit;  // unit 0: actions[0..2], unit 1: actions[3..5]
    out.order = 0;
    out.vp = 0;
    if (input == nullptr) return out;

    // A round counter that went backwards means a new match started:
    // wipe anything remembered from the previous map.
    if (input->round == 0 || input->round <= g_last_round) ResetMatchState();
    g_last_round = input->round;

    // Remember every obstacle we have ever seen (they never move).
    for (int r = 0; r < GRID_N; ++r)
        for (int c = 0; c < GRID_N; ++c)
            if (input->grid[r][c] == CELL_OBSTACLE) g_known_obstacle[r][c] = 1;

    // Count NPCs per cell; >= 3 on one cell tramples us for 5% of our gold.
    unsigned char npc_count[GRID_N][GRID_N];
    std::memset(npc_count, 0, sizeof(npc_count));
    for (int i = 0; i < input->num_visible_npcs && i < MAX_NPCS; ++i) {
        const Position p = input->visible_npcs[i].pos;
        if (InBounds(p.row, p.col)) ++npc_count[p.row][p.col];
    }

    // Cells we refuse to step on. Fog is treated as a wall: it could hide
    // an obstacle (wasted move) or a bomb (-10% gold), and our vision
    // recenters on us next round anyway.
    unsigned char blocked[GRID_N][GRID_N];
    for (int r = 0; r < GRID_N; ++r) {
        for (int c = 0; c < GRID_N; ++c) {
            const int v = input->grid[r][c];
            blocked[r][c] = (v == CELL_FOG || v == CELL_BOMB || v == CELL_OBSTACLE ||
                             g_known_obstacle[r][c] || npc_count[r][c] >= 3)
                                ? 1
                                : 0;
        }
    }
    for (int i = 0; i < 2; ++i) {
        const Position e = input->visible_enemies[i];
        if (InBounds(e.row, e.col)) blocked[e.row][e.col] = 1;  // can't pass through enemies
    }

    // Unit 0 plans first (order = 0). While it moves, unit 1 is still parked
    // on its current cell, so that cell is a wall for unit 0...
    unsigned char blocked0[GRID_N][GRID_N];
    std::memcpy(blocked0, blocked, sizeof(blocked));
    blocked0[input->my_units[1].row][input->my_units[1].col] = 1;
    int target0 = -1;
    const Position end0 = PlanUnit(input, input->my_units[0], blocked0, -1,
                                   kStepsPerUnit, &out.actions[0], &target0);

    // ...and by the time unit 1 moves, unit 0 sits on its final cell.
    unsigned char blocked1[GRID_N][GRID_N];
    std::memcpy(blocked1, blocked, sizeof(blocked));
    blocked1[end0.row][end0.col] = 1;
    int target1 = -1;
    PlanUnit(input, input->my_units[1], blocked1, target0, kStepsPerUnit,
             &out.actions[kStepsPerUnit], &target1);

    return out;
}

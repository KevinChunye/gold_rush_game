// local_test.cpp — offline smoke test for the submission artifact.
//
// Loads ./player.so with dlopen(), exactly like the contest engine does,
// then feeds it hand-crafted rounds and checks that every answer is a
// well-formed, sane GameOutput. Also measures decision latency, since the
// contest uses P90 latency as tiebreaker and for the speed prize.
//
// This is NOT a full game simulator (NPC policy, spawn distributions and
// bomb schedules are secret) — real matches are played on the contest site,
// which produces replays. This harness just proves the pipeline:
// build .so -> engine can load it -> every round gets a legal answer, fast.

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/game_api.h"

using MoveFn = GameOutput (*)(const GameInput*);

static int g_failures = 0;

#define CHECK(cond, ...)                              \
    do {                                              \
        if (!(cond)) {                                \
            std::printf("  [FAIL] " __VA_ARGS__);     \
            std::printf("\n");                        \
            ++g_failures;                             \
        }                                             \
    } while (0)

static const int kDr[4] = {-1, 1, 0, 0};
static const int kDc[4] = {0, 0, -1, 1};

static GameInput MakeEmptyInput() {
    GameInput in;
    std::memset(&in, 0, sizeof(in));
    for (int r = 0; r < GRID_N; ++r)
        for (int c = 0; c < GRID_N; ++c) in.grid[r][c] = CELL_FOG;
    in.visible_enemies[0] = {-1, -1};
    in.visible_enemies[1] = {-1, -1};
    for (int i = 0; i < MAX_NPCS; ++i) in.visible_npcs[i] = {0, {-1, -1}};
    in.snapshot.window_begin = -1;
    return in;
}

// Reveal the 5x5 vision box around (r,c) as empty ground (clipped at edges).
static void RevealVision(GameInput* in, int r, int c) {
    for (int dr = -2; dr <= 2; ++dr)
        for (int dc = -2; dc <= 2; ++dc) {
            const int nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= GRID_N || nc < 0 || nc >= GRID_N) continue;
            if (in->grid[nr][nc] == CELL_FOG) in->grid[nr][nc] = CELL_EMPTY;
        }
}

static void PrintBoard(const GameInput& in) {
    char board[GRID_N][GRID_N];
    for (int r = 0; r < GRID_N; ++r)
        for (int c = 0; c < GRID_N; ++c) {
            const int v = in.grid[r][c];
            char ch = '?';
            if (v == CELL_FOG) ch = '~';
            else if (v == CELL_OBSTACLE) ch = '#';
            else if (v == CELL_BOMB) ch = 'X';
            else if (v == CELL_EMPTY) ch = '.';
            else if (v >= 1) ch = static_cast<char>('0' + (v > 9 ? 9 : v));
            board[r][c] = ch;
        }
    for (int i = 0; i < in.num_visible_npcs; ++i) {
        const Position p = in.visible_npcs[i].pos;
        if (p.row >= 0) board[p.row][p.col] = 'n';
    }
    for (int i = 0; i < 2; ++i) {
        const Position e = in.visible_enemies[i];
        if (e.row >= 0) board[e.row][e.col] = 'E';
    }
    board[in.my_units[0].row][in.my_units[0].col] = 'A';
    board[in.my_units[1].row][in.my_units[1].col] = 'B';

    std::printf("     (~ fog  # obstacle  X bomb  . empty  1-9 gold  A/B us  E enemy  n NPC)\n");
    for (int r = 0; r < GRID_N; ++r) {
        std::printf("     ");
        for (int c = 0; c < GRID_N; ++c) std::printf("%c ", board[r][c]);
        std::printf("\n");
    }
}

static const char* ActName(int a) {
    switch (a) {
        case ACT_UP: return "UP";
        case ACT_DOWN: return "DOWN";
        case ACT_LEFT: return "LEFT";
        case ACT_RIGHT: return "RIGHT";
        case ACT_STAY: return "STAY";
    }
    return "??";
}

// Format check: an out-of-range field means instant match loss, so this is
// the single most important property of the bot.
static void ValidateFormat(const GameOutput& out) {
    for (int i = 0; i < MAX_MOVES; ++i)
        CHECK(out.actions[i] >= 0 && out.actions[i] <= 4, "actions[%d]=%d out of [0,4]", i,
              out.actions[i]);
    CHECK(out.k >= 0 && out.k <= MAX_MOVES, "k=%d out of [0,%d]", out.k, MAX_MOVES);
    CHECK(out.order == 0 || out.order == 1, "order=%d not in {0,1}", out.order);
    CHECK(out.vp >= 0 && out.vp <= 2, "vp=%d not in {0,1,2}", out.vp);
}

// Sanity check: the first step of each unit must not walk into a cell the
// input already shows as fatal/wasteful (wall, bomb, off-board, 3+ NPCs).
static void ValidateFirstSteps(const GameInput& in, const GameOutput& out) {
    for (int unit = 0; unit < 2; ++unit) {
        const int begin = unit == 0 ? 0 : out.k;
        const int end = unit == 0 ? out.k : MAX_MOVES;
        if (begin >= end) continue;  // unit got no moves this round
        const int a = out.actions[begin];
        if (a == ACT_STAY) continue;
        const int nr = in.my_units[unit].row + kDr[a];
        const int nc = in.my_units[unit].col + kDc[a];
        CHECK(nr >= 0 && nr < GRID_N && nc >= 0 && nc < GRID_N,
              "unit %d first move leaves the board", unit);
        if (nr < 0 || nr >= GRID_N || nc < 0 || nc >= GRID_N) continue;
        CHECK(in.grid[nr][nc] != CELL_OBSTACLE, "unit %d first move hits an obstacle", unit);
        CHECK(in.grid[nr][nc] != CELL_BOMB, "unit %d first move steps on a bomb", unit);
        int npcs = 0;
        for (int i = 0; i < in.num_visible_npcs; ++i)
            if (in.visible_npcs[i].pos.row == nr && in.visible_npcs[i].pos.col == nc) ++npcs;
        CHECK(npcs < 3, "unit %d first move enters a trampling cell", unit);
    }
}

static GameOutput RunScenario(MoveFn move, const char* title, const GameInput& in,
                              bool print_board) {
    std::printf("\n== %s (round %d) ==\n", title, in.round);
    if (print_board) PrintBoard(in);
    const GameOutput out = move(&in);
    std::printf("  decision: k=%d order=%d vp=%d\n", out.k, out.order, out.vp);
    std::printf("    unit A: ");
    for (int i = 0; i < out.k; ++i) std::printf("%s ", ActName(out.actions[i]));
    std::printf("\n    unit B: ");
    for (int i = out.k; i < MAX_MOVES; ++i) std::printf("%s ", ActName(out.actions[i]));
    std::printf("\n");
    ValidateFormat(out);
    ValidateFirstSteps(in, out);
    return out;
}

int main() {
    void* handle = dlopen("./player.so", RTLD_NOW);
    if (!handle) {
        std::printf("[FAIL] dlopen(./player.so): %s\n", dlerror());
        return 1;
    }
    auto move = reinterpret_cast<MoveFn>(dlsym(handle, "moveDecision"));
    if (!move) {
        std::printf("[FAIL] dlsym(moveDecision): %s\n", dlerror());
        return 1;
    }
    std::printf("player.so loaded, moveDecision symbol resolved (same path the engine uses).\n");

    // --- Scenario 1: opening position, gold visible to both units -------
    GameInput s1 = MakeEmptyInput();
    s1.round = 0;
    s1.my_units[0] = {0, 0};
    s1.my_units[1] = {16, 16};
    RevealVision(&s1, 0, 0);
    RevealVision(&s1, 16, 16);
    s1.grid[1][1] = CELL_OBSTACLE;  // in unit A's way
    s1.grid[2][1] = 5;              // gold for unit A
    s1.grid[15][15] = CELL_BOMB;    // in unit B's way
    s1.grid[14][15] = 2;            // gold for unit B
    const GameOutput o1 =
        RunScenario(move, "opening: route to gold around obstacle/bomb", s1, true);
    CHECK(o1.actions[0] != ACT_STAY, "unit A should move toward visible gold");
    CHECK(o1.actions[o1.k] != ACT_STAY, "unit B should move toward visible gold");

    // --- Scenario 2: nothing visible -> head for the center --------------
    GameInput s2 = MakeEmptyInput();
    s2.round = 1;
    s2.my_units[0] = {0, 0};
    s2.my_units[1] = {16, 16};
    RevealVision(&s2, 0, 0);
    RevealVision(&s2, 16, 16);
    const GameOutput o2 = RunScenario(move, "no gold in sight: converge on center", s2, false);
    CHECK(o2.actions[0] == ACT_DOWN || o2.actions[0] == ACT_RIGHT,
          "unit A should walk toward the center");
    CHECK(o2.actions[o2.k] == ACT_UP || o2.actions[o2.k] == ACT_LEFT,
          "unit B should walk toward the center");

    // --- Scenario 3: unit A boxed in by obstacles -> must stay, not crash -
    GameInput s3 = MakeEmptyInput();
    s3.round = 2;
    s3.my_units[0] = {0, 0};
    s3.my_units[1] = {16, 16};
    RevealVision(&s3, 0, 0);
    RevealVision(&s3, 16, 16);
    s3.grid[0][1] = CELL_OBSTACLE;
    s3.grid[1][0] = CELL_OBSTACLE;
    s3.grid[1][1] = CELL_OBSTACLE;
    const GameOutput o3 = RunScenario(move, "boxed-in corner: stay put safely", s3, false);
    CHECK(o3.actions[0] == ACT_STAY, "boxed-in unit A must stay");

    // --- Scenario 4: gold guarded by an NPC crowd -> take the safe pile ---
    GameInput s4 = MakeEmptyInput();
    s4.round = 3;
    s4.my_units[0] = {8, 8};
    s4.my_units[1] = {16, 16};
    RevealVision(&s4, 8, 8);
    RevealVision(&s4, 16, 16);
    for (int c = 5; c <= 10; ++c) s4.grid[8][c] = (s4.grid[8][c] == CELL_FOG) ? 0 : s4.grid[8][c];
    s4.grid[8][10] = 9;  // rich pile, but three NPCs sit on it
    s4.grid[8][6] = 4;   // safe pile
    s4.num_visible_npcs = 3;
    s4.visible_npcs[0] = {1, {8, 10}};
    s4.visible_npcs[1] = {2, {8, 10}};
    s4.visible_npcs[2] = {3, {8, 10}};
    const GameOutput o4 = RunScenario(move, "3 NPCs camp the big pile: dodge it", s4, true);
    CHECK(o4.actions[0] == ACT_LEFT, "unit A should choose the un-camped pile to its left");

    // --- Scenario 5: enemy visible next to us --------------------------
    GameInput s5 = MakeEmptyInput();
    s5.round = 4;
    s5.my_units[0] = {8, 8};
    s5.my_units[1] = {0, 16};
    RevealVision(&s5, 8, 8);
    RevealVision(&s5, 0, 16);
    s5.grid[8][9] = 3;              // gold right of unit A...
    s5.visible_enemies[0] = {8, 9}; // ...but an enemy stands on it
    s5.my_units_gold[0] = 12;
    s5.gold_opp = 40;
    RunScenario(move, "enemy standing on the nearest gold: no illegal bump", s5, false);

    // --- Latency: the contest's other scoreboard ------------------------
    // 300 ms/round hard-ish limit, P90 latency breaks ties + speed prize.
    const int kIters = 20000;
    std::vector<double> us(kIters);
    for (int i = 0; i < kIters; ++i) {
        GameInput in = s1;
        in.round = 5 + (i % 490);  // avoid the round==0 state reset path
        const auto t0 = std::chrono::steady_clock::now();
        const GameOutput out = move(&in);
        const auto t1 = std::chrono::steady_clock::now();
        us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (out.k < 0) return 1;  // keep the call from being optimized away
    }
    std::sort(us.begin(), us.end());
    std::printf("\n== latency over %d decisions ==\n", kIters);
    std::printf("  p50 = %.1f us,  p90 = %.1f us,  max = %.1f us  (budget: 300000 us)\n",
                us[kIters / 2], us[kIters * 9 / 10], us[kIters - 1]);

    std::printf("\n%s\n", g_failures == 0 ? "ALL CHECKS PASSED"
                                          : "SOME CHECKS FAILED — see [FAIL] lines above");
    return g_failures == 0 ? 0 : 1;
}

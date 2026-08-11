// sim/dummy_player.cpp — trivial smoke-test bot for the local simulator.
//
// Fixed k=3 (unit0 runs actions[0..2], unit1 runs actions[3..5]), order=0,
// vp=0. Each unit walks toward the nearest gold pile it can see, or toward
// the board centre when none is visible, dodging obstacle cells it can see
// (fog is optimistically treated as walkable; the engine skips any step
// that turns out to be illegal).
#include "../src/game_api.h"

namespace {

int absi(int x) { return x < 0 ? -x : x; }

// one greedy step from (r,c) toward (tr,tc); returns the action and
// advances r,c as if the step succeeded
int stepToward(const GameInput* in, int& r, int& c, int tr, int tc) {
    static const int DR[4] = {-1, 1, 0, 0}, DC[4] = {0, 0, -1, 1};
    int dr = tr - r, dc = tc - c;
    int rowAct = dr > 0 ? 1 : 0;  // down : up
    int colAct = dc > 0 ? 3 : 2;  // right : left
    int pref[2];
    int n = 0;
    if (absi(dr) >= absi(dc)) {   // longer axis first
        if (dr != 0) pref[n++] = rowAct;
        if (dc != 0) pref[n++] = colAct;
    } else {
        if (dc != 0) pref[n++] = colAct;
        if (dr != 0) pref[n++] = rowAct;
    }
    for (int i = 0; i < n; ++i) {
        int nr = r + DR[pref[i]], nc = c + DC[pref[i]];
        if (nr < 0 || nr >= GRID_SIZE || nc < 0 || nc >= GRID_SIZE) continue;
        if (in->grid[nr][nc] == -1) continue;  // visible obstacle
        r = nr;
        c = nc;
        return pref[i];
    }
    return 4;  // stay
}

}  // namespace

extern "C" GameOutput moveDecision(const GameInput* in) {
    GameOutput out;
    for (int i = 0; i < S; ++i) out.actions[i] = 4;
    out.k = 3;
    out.order = 0;
    out.vp = 0;
    for (int u = 0; u < 2; ++u) {
        int r = in->my_units[u].row, c = in->my_units[u].col;
        // target: nearest visible gold (vision is at most a 9x9 box, so a
        // radius-4 scan suffices), else the board centre
        int tr = 8, tc = 8, best = 1 << 30;
        for (int i = (r > 4 ? r - 4 : 0); i <= (r + 4 < GRID_SIZE ? r + 4 : GRID_SIZE - 1); ++i)
            for (int j = (c > 4 ? c - 4 : 0); j <= (c + 4 < GRID_SIZE ? c + 4 : GRID_SIZE - 1); ++j)
                if (in->grid[i][j] >= 1) {
                    int d = absi(i - r) + absi(j - c);
                    if (d < best) { best = d; tr = i; tc = j; }
                }
        for (int s = 0; s < 3; ++s)
            out.actions[u * 3 + s] = stepToward(in, r, c, tr, tc);
    }
    return out;
}

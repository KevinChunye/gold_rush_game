// game_api.h — C++ ABI for GoldRush 2.0 (编程掘金争夺赛)
//
// The structs below are copied field-for-field from the official rules
// document. The engine dlopen()s our player.so and calls moveDecision()
// once per round, so the memory layout of these structs MUST match the
// official header exactly.
//
// NOTE: the official reference code sent to contestants also contains a
// game_api.h. If that file ever differs from this one, replace this file
// with the official one and rebuild.

#ifndef GOLD_RUSH_GAME_API_H
#define GOLD_RUSH_GAME_API_H

// ---- Contest parameters (public-beta values, see rules doc) ----
#define GRID_N 17      // board is 17x17
#define MAX_NPCS 7     // A: number of NPCs
#define MAX_MOVES 6    // S: moves per round, shared by our two units
#define NUM_REGIONS 5  // regions in the periodic global snapshot

// grid[][] cell values
#define CELL_FOG (-5)       // outside our vision
#define CELL_BOMB (-3)      // bomb: entering costs 10% of held gold
#define CELL_OBSTACLE (-1)  // impassable
#define CELL_EMPTY 0        // walkable, no gold
                            // >= 1 : walkable, value = gold on the cell

// action codes for GameOutput.actions[]
#define ACT_UP 0     // row - 1
#define ACT_DOWN 1   // row + 1
#define ACT_LEFT 2   // col - 1
#define ACT_RIGHT 3  // col + 1
#define ACT_STAY 4

struct Position {
    int row;
    int col;
};

struct NpcInfo {
    int id;        // stable across rounds; 0 = empty slot
    Position pos;  // (-1,-1) when not visible
};

struct RegionStat {
    int id;              // region id 1-5
    int enter;           // unit entries into the region during the window
    int leave;           // unit exits from the region during the window
    int gold_generated;  // gold spawned in the region during the window
    int gold_collected;  // gold picked up in the region during the window
    int gold_remaining;  // gold currently on the ground in the region
    int occupants;       // units currently inside the region
};

struct Snapshot {
    int window_begin;  // first round of the stat window; -1 if no snapshot
    int window_end;    // last round of the stat window
    RegionStat regions[NUM_REGIONS];
};

struct GameInput {
    int round;                        // current round, starts at 0
    int grid[GRID_N][GRID_N];         // see CELL_* above
    Position my_units[2];             // our unit 0 and unit 1
    int my_units_gold[2];             // gold held by each of our units
    int gold_opp;                     // opponent's two units' gold, summed
    Position visible_enemies[2];      // (-1,-1) when not visible
    int num_visible_npcs;             // valid entries in visible_npcs
    NpcInfo visible_npcs[MAX_NPCS];   // visible NPCs
    int snapshot_valid;               // 1 = fresh snapshot this round
    Snapshot snapshot;                // global region stats, every D rounds
};

struct GameOutput {
    int actions[MAX_MOVES];  // each in [0,4], see ACT_*
    int k;                   // unit 0 runs actions[0:k], unit 1 runs actions[k:6]
    int order;               // 0 = unit 0 moves first, 1 = unit 1 first
    int vp;                  // vision purchase: 0 none, 1 = 7x7 (2 gold), 2 = 9x9 (3 gold)
};

#ifdef __cplusplus
extern "C" {
#endif

// The one entry point the game engine calls. Must return within 300 ms
// (overruns are billed to a 60 s per-match pool; empty pool = loss).
GameOutput moveDecision(const GameInput* input);

#ifdef __cplusplus
}
#endif

#endif  // GOLD_RUSH_GAME_API_H

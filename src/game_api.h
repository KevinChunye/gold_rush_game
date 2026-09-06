// game_api.h — GoldRush 2.0 contest ABI, transcribed from the published
// competition rules. The engine dlopen()s the bot and calls moveDecision()
// once per round, so struct layout must match the engine exactly
// (plain ints, 4-byte aligned throughout).
#pragma once

constexpr int GRID_SIZE = 17;    // the board is 17x17
constexpr int MAX_NPCS = 7;      // NPCs on the map
constexpr int S = 6;             // moves per round, shared by our two units
constexpr int REGION_COUNT = 5;  // regions in the periodic global snapshot

struct Position {
    int row;
    int col;
};

struct NpcInfo {
    int id;        // stable across rounds; 0 = empty slot
    Position pos;  // (-1,-1) when not visible
};

struct RegionStat {
    int id;              // region id, 1..5
    int enter;           // actor entries into the region during the window
    int leave;           // actor exits during the window
    int gold_generated;  // gold spawned in the region during the window
    int gold_collected;  // gold picked up during the window
    int gold_remaining;  // gold currently on the ground in the region
    int occupants;       // actors currently inside
};

struct Snapshot {
    int window_begin;  // first round of the stat window; -1 if no snapshot
    int window_end;    // last round of the stat window
    RegionStat regions[REGION_COUNT];
};

struct GameInput {
    int round;                       // current round, starts at 0
    int grid[GRID_SIZE][GRID_SIZE];  // fogged terrain: -5 fog, -3 bomb,
                                     // -1 obstacle, 0 empty, >=1 gold amount
                                     // (actors are NOT marked in the grid)
    Position my_units[2];            // our two units
    int my_units_gold[2];            // gold held by each of our units
    int gold_opp;                    // opponent's two units' gold, summed
    Position visible_enemies[2];     // packed from index 0; empty slots (-1,-1)
    int num_visible_npcs;            // valid entries in visible_npcs
    NpcInfo visible_npcs[MAX_NPCS];  // visible NPCs; tail slots id=0/(-1,-1)
    int snapshot_valid;              // 1 = fresh snapshot this round
    Snapshot snapshot;               // global region stats, every 5 rounds
};

struct GameOutput {
    int actions[S];  // each in [0,4]: 0 up, 1 down, 2 left, 3 right, 4 stay
    int k;           // split: unit 0 runs actions[0..k), unit 1 actions[k..S)
    int order;       // 0 = unit 0 executes first, 1 = unit 1 first
    int vp;          // vision purchase: 0 none, 1 = 7x7, 2 = 9x9 (for next round)
};

extern "C" GameOutput moveDecision(const GameInput* input);

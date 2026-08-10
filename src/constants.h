// constants.h — rule constants shared by the bot and the local tests.
//
// Kept out of game_api.h so that src/game_api.h stays byte-identical to
// the official reference/game_api.h (run `diff` on the pair any time the
// organizers ship an update — they must never drift apart).
#pragma once

#include "game_api.h"

// grid[][] cell values (the official header documents these only in a comment)
constexpr int CELL_FOG = -5;       // outside our vision
constexpr int CELL_BOMB = -3;      // entering costs 10% of held gold
constexpr int CELL_OBSTACLE = -1;  // impassable
constexpr int CELL_EMPTY = 0;      // walkable; values >= 1 are gold amounts

// action codes for GameOutput.actions[]
constexpr int ACT_UP = 0;     // row - 1
constexpr int ACT_DOWN = 1;   // row + 1
constexpr int ACT_LEFT = 2;   // col - 1
constexpr int ACT_RIGHT = 3;  // col + 1
constexpr int ACT_STAY = 4;

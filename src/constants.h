// constants.h — grid cell values and action codes from the competition
// rules, shared by the bot and the offline test harness.
#pragma once

#include "game_api.h"

// grid[][] cell values
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

# Official beta log analysis (logs/data.tar.gz, 197 games)

Corpus: 8 organizer bots (Intel-codename names) round-robin on ONE map;
full 500-round records from player 1's fogged perspective, plus global
snapshot truth every 5 rounds. Analysis script: tools/analyze_logs.py.

## Log format (per file)
- line 1: {"player1": name, "player2": name}
- line 2: 17x17 map as strings: "0" empty, "1" obstacle, "2" = OUTER GOLD
  HOTSPOT (see below)
- lines 3..502: per-round JSON: {round, start:{grid (P1-fogged), players
  [gold, vision_spent, cost(ns), order, units[position, gold, actions
  (executed), pickup]], npcs (visible, ids NEGATIVE -1..-7)}, end:{same +
  dispatch_order, trample_events, burned}, snapshot?}

## Confirmed engine facts
- Snapshots at rounds 5,10,...,495 labeled window [S-5, S-1], BUT round
  S's own spawn phase is included in snapshot S (delivery happens after
  spawning); movement/pickup of round S counts toward snapshot S+5.
  occupants are measured before round-S moves and always sum to 11
  (4 units + 7 NPCs).
- NPC ids are negative (-1..-7); 7 NPCs.
- Bomb waves land exactly at rounds ≡ 0 (mod 20).

## REGION GEOMETRY (solved by constraint propagation, 0 contradictions)
Pinwheel, 180-degree symmetric:
- region 1 = center 9x9 (rows 4-12, cols 4-12), 81 cells
- region 2 = top band: rows 0-3, cols 0-12 (52 cells)
- region 5 = right band: cols 13-16, rows 0-12 (52 cells)
- region 4 = bottom band: rows 13-16, cols 4-16 (52 cells)
- region 3 = left band: cols 0-3, rows 4-16 (52 cells)
P1 spawns (0,0)/(16,16) sit in regions 2/4; P2 (0,16)/(16,0) in 5/3.

## Spawn economics
- Center: ~9.5 gold/round (~47 per 5-round window), pile sizes ~1-10
  (mean 5.5).
- Outer: a wave every ~10 rounds, ~120 gold per wave split across the 4
  bands (~12.4/window/band in expectation), piles mean 12.5, max 53 —
  outer piles are 2-4x richer than center piles.
- The map's 20 "2" cells receive ~half of ALL outer gold (~10x per-cell
  hotspot rate). Hotspots are map data — learnable in-match by watching
  where outer piles appear.

## Strategy implications (v2.6 backlog)
1. Snapshot macro layer is now unblocked: with the pinwheel geometry,
   gold_remaining per region directly names WHERE uncollected gold sits.
2. Outer-wave farming: every ~10 rounds ~120 mostly-uncontested gold
   drops in the bands while everyone camps the center; a unit positioned
   at band hotspots at wave time collects piles worth 8-35 each.
3. Learn hotspot cells in-match from observed outer spawns (map line is
   not available at runtime; hotspots differ per map).
4. Simulator recalibration: center ~2 piles/round of 1-10; outer wave
   every 10 rounds ~120 gold in big piles biased to hotspots; bomb waves
   at rounds % 20 == 0 (bigger than the old guess); snapshot phase
   rounds % 5 == 0 with the inclusion rule above.

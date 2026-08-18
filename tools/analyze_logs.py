#!/usr/bin/env python3
"""Analyze official GoldRush beta logs (logs/data.tar.gz, extracted).

Reproduces the findings in docs/LOG_FINDINGS.md:
  1. region geometry via constraint propagation over snapshot truth
  2. spawn economics (center vs outer bands, hotspot cells)
  3. bomb wave timing
Usage: python3 tools/analyze_logs.py <dir-with-g*.txt> [max_games]
"""
import json, glob, sys, statistics as S
from collections import Counter

def region_of(r, c):  # solved pinwheel geometry
    if 4 <= r <= 12 and 4 <= c <= 12: return 1
    if r <= 3 and c <= 12: return 2
    if c >= 13 and r <= 12: return 5
    if r >= 13 and c >= 4: return 4
    return 3

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    files = sorted(glob.glob(f"{d}/g*.txt"))[:lim]

    possible = {(r, c): set(range(1, 6)) for r in range(17) for c in range(17)}
    gen = {rid: [] for rid in range(1, 6)}
    spawn_c, spawn_o = [], []
    bomb_mod = Counter()
    viol = 0

    for fp in files:
        with open(fp) as f:
            f.readline(); f.readline()
            prev_end, snaps, spawn_ev, occ_ev = None, {}, [], []
            for line in f:
                rec = json.loads(line)
                t, sg = rec['round'], rec['start']['grid']
                if prev_end is not None:
                    for r in range(17):
                        for c in range(17):
                            a, b = prev_end[r][c], sg[r][c]
                            if a >= 0 and b > a:
                                spawn_ev.append((((t + 4)//5)*5, (r, c), b - a))
                            if a not in (-3, -5) and b == -3:
                                bomb_mod[t % 20] += 1
                prev_end = rec['end']['grid']
                if rec.get('snapshot'):
                    Sr = rec['snapshot']['round']
                    snaps[Sr] = {x['id']: x for x in rec['snapshot']['regions']}
                    for rid in range(1, 6):
                        gen[rid].append(snaps[Sr][rid]['gold_generated'])
                    for u in rec['start']['players'][0]['units']:
                        if u.get('position'):
                            occ_ev.append((Sr, tuple(u['position'])))
        for w, cell, amt in spawn_ev:
            (spawn_c if region_of(*cell) == 1 else spawn_o).append(amt)
            if w in snaps:
                gens = {rid for rid, x in snaps[w].items() if x['gold_generated'] > 0}
                possible[cell] &= gens
                if not possible[cell]: viol += 1
        for w, cell in occ_ev:
            if w in snaps:
                possible[cell] -= {rid for rid, x in snaps[w].items() if x['occupants'] == 0}
                if not possible[cell]: viol += 1

    bad = sum(1 for cell, v in possible.items()
              if v and region_of(*cell) not in v)
    print(f"games: {len(files)}; geometry violations: {bad} cells disagree, "
          f"{viol} contradictions (both should be 0)")
    print(f"center gen/window mean {S.mean(gen[1]):.1f}; outer bands "
          f"{[round(S.mean(gen[r]),1) for r in (2,3,4,5)]}")
    print(f"pile sizes: center mean {S.mean(spawn_c):.1f} max {max(spawn_c)}; "
          f"outer mean {S.mean(spawn_o):.1f} max {max(spawn_o)}")
    print(f"bomb sightings by round%20: {dict(sorted(bomb_mod.items()))}")

if __name__ == "__main__":
    main()

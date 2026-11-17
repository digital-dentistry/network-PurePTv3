"""Train / test lists for the fine-tune (by patient, seeded).  Asset ids only: no scan data.

usage: python make_split.py <assets file> <painted stages jsonl> <public eval list> <inhouse eval list> <out dir>
  assets file: "asset institution patient jaw has_color fully_labelled has_saved" per line (psql -At).
  painted stages jsonl: ../2026-11-17_autolabel-incomplete-teeth/logs/stages100.jsonl (per-tooth IoU of the
  deployed model against the colour map); scans with a tooth below 0.85 go to the test side, because they hold
  every known colour-map error (two teeth in one colour, implants, Case 322's sheets) and the model errors we
  want to see move.

Writes (one "asset source" per line, the input of export_cache.py):
  test_inhouse.txt   half of the inhouse patients with a completed label (saved)
  test_painted.txt   50 painted Teeth3DS scans (colour)
  test_public.txt    the 100 public eval scans of 2026-11-17_autolabel-incomplete-teeth (src)
  train_ddh.txt      the other inhouse half (saved) + 50 painted (auto: the person's label if saved, else colour)
                     + public scans a person completed (saved)
  train_replay.txt   public scans with shipped labels (src), patients of every test list excluded
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPLAY = {"teeth3ds": 420, "poseidon3d": 120, "3d_iosseg": 120, "cbct_ios_fusion_liu2023": 60}


def main():
    assets_f, stages_f, public_f, inhouse_f, out = map(Path, sys.argv[1:6])
    rows = {}
    for line in assets_f.read_text().splitlines():
        a, inst, pid, jaw, colour, full, saved = line.split()
        rows[a] = dict(inst=inst, pid=pid, jaw=jaw, colour=colour == "t", full=full == "t", saved=saved == "t")
    rng = np.random.default_rng(20261117)

    # inhouse: patients with a completed label, half for test
    inhouse = [a for a in inhouse_f.read_text().split() if a in rows]
    by_pid = defaultdict(list)
    for a in inhouse:
        by_pid[rows[a]["pid"]].append(a)
    pids = sorted(by_pid)
    rng.shuffle(pids)
    test_in = [a for p in pids[:len(pids) // 2] for a in by_pid[p]]
    train_in = [a for p in pids[len(pids) // 2:] for a in by_pid[p]]

    # painted: scans with a tooth below 0.85 first into test, then random to 50
    painted = sorted(a for a, r in rows.items() if r["inst"] == "teeth3ds" and r["colour"])
    flagged = set()
    for line in stages_f.read_text().splitlines():
        r = json.loads(line)
        if any(t["iou"] < 0.85 for t in r["teeth"]):
            flagged.add(r["asset_id"])
    rest = [a for a in painted if a not in flagged]
    rng.shuffle(rest)
    test_paint = sorted(flagged) + rest[:50 - len(flagged)]
    test_pids = {rows[a]["pid"] for a in test_paint}
    train_paint = [a for a in rest[50 - len(flagged):] if rows[a]["pid"] not in test_pids]

    test_pub = [a for a in public_f.read_text().split() if a in rows]
    held = {rows[a]["pid"] for a in test_in + test_paint + test_pub}

    saved_pub = sorted(a for a, r in rows.items() if r["saved"] and r["inst"] not in ("inhouse",) and not r["colour"]
                       and r["pid"] not in held)
    replay = []
    for inst, n in REPLAY.items():
        pool = sorted(a for a, r in rows.items() if r["inst"] == inst and r["full"] and not r["colour"]
                      and r["pid"] not in held and a not in saved_pub)
        rng.shuffle(pool)
        replay += pool[:n]

    out.mkdir(parents=True, exist_ok=True)
    write = lambda name, items, src: (out / name).write_text("".join(f"{a} {src}\n" for a in items))  # noqa: E731
    write("test_inhouse.txt", test_in, "saved")
    write("test_painted.txt", test_paint, "colour")
    write("test_public.txt", test_pub, "src")
    (out / "train_ddh.txt").write_text("".join(f"{a} saved\n" for a in train_in)
                                       + "".join(f"{a} auto\n" for a in train_paint)
                                       + "".join(f"{a} saved\n" for a in saved_pub))
    write("train_replay.txt", replay, "src")
    print(f"inhouse test {len(test_in)} / train {len(train_in)}; painted test {len(test_paint)} "
          f"({len(flagged)} flagged) / train {len(train_paint)}; public test {len(test_pub)}; "
          f"public saved (train) {len(saved_pub)}; replay {len(replay)}")


if __name__ == "__main__":
    main()

# 2026-11-17 Fine-tune the ToothSegNet instance model on DDH-convention labels

**Question (user, 2026-11-17):** "그런 매뉴얼한 방식보다, 딥러닝 모델 레벨에서 개선은 어려워?" and then "1,2,3 다
진행해줘". Step 3 is to rebuild the training cache and run a fine-tune experiment.

**What it targets:** the misses that post-processing cannot fix
(`../2026-11-17_autolabel-incomplete-teeth`):
- the 1–3 mm band at the gum line;
- the back of the last molar;
- Case 341 tooth 21's labial cervical patch;
- small third molars.

There, all 8 TTA poses stop at the same place. Thresholds (E1) and geometry snapping (E5) did not help.

**Idea:** start from the deployed weights (`models/toothsegnet-ios/1/instance.pt`, epoch 290) and train a little
more on the labels DDH wants, with public data mixed in so nothing is forgotten.

## Data: `scripts/make_split.py`, `scripts/export_cache.py` → `lists/`

The training cache was rebuilt because `Desktop/AI/data` is gone.
- It was exported read-only from the dev DDH: mesh.ply from ddh-curated, labels from ios_labels or mesh.ply.
- It is in the Docker volume `ddh-train-cache`, outside the repo and outside OneDrive, because it holds patient
  scans.
- No DDH release was built: releases cannot be removed.
- Format: dental_ai ToothSegLoader npz (`xyz`, `nrm`, `cls`, `ins`, `jaw`, `rgb` only for real scanner colour).

Labels:
- 1 tooth and 2 prep-tooth are teeth. 3 scanbody and 4 peek are not teeth.
- Scans with unlabelled vertices are skipped (none were).
- Painted Teeth3DS teeth get class 17 (position unknown). The position loss ignores it.
- **No Teeth3DS cache has rgb.** Teeth3DS colour is the painted label. 6 caches first got it through the saved-label
  path; they were rebuilt and checked (0 left).

Splits are by patient:

| set | train | test |
|---|---|---|
| inhouse with a completed label (saved) | 57 scans | 44 scans |
| painted Teeth3DS (person's saved label if any, else colour map) | 33 | 50: all 18 scans with a tooth below IoU 0.85 in E1 (every known colour-map error and the model misses), plus 32 random |
| public scans a person completed (saved: 12 fusion, 2 poseidon3d) | 14 | — |
| public with shipped labels (replay: 420 teeth3ds, 120 poseidon3d, 120 3d_iosseg, 60 fusion) | 675 (+45 validation) | the 100 public eval scans of E6/E8; their patients are excluded from replay |

The training list (`lists/ft1_train_names.txt`) repeats the 104 DDH-convention scans 4 times: 416 + 675 = 1,091 per
epoch. Validation is 45 replay scans, to watch for forgetting.

## Run ft1: data only, same recipe

`training/train.py --stage instance` was run in `hdl-gpu:latest` (container `ddh-ft1`), with output in the Docker
volume `ddh-train-runs:/runs/ft1`.
- `--init-from /base/instance.pt` (398/398 tensors);
- `--epochs 20 --lr 2e-4 --warmup 1 --val-every 5 --workers 6`;
- the rest is the instance defaults: batch 6×2, crop 0.35, trim 0.5, colour 0.6, so3 rotation.
- One epoch takes about 70 s.

Validation on the 45 replay scans stays level: grey mAP 92.0 → 92.7, colour 96.8 → 97.6. Logs: `logs/ft1_train.log`.

**Test: `scripts/tooth_compare.py` → `logs/ft1_tooth_compare.txt`.**
- Base and ft1 run through the same `eval_fix.py`: the network once, then the deployed clean-up (absorb +
  `join_close_pieces`).
- Per-tooth IoU against GT teeth of at least 5 mm². The base runs are the E8 runs on the same scans.

| test set | base IoU mean | ft1 IoU mean | teeth < 0.85 base → ft1 | lost (< 0.5) |
|---|---|---|---|---|
| painted Teeth3DS 50 (colour GT, no colour sent) | 0.9615 | 0.9610 | 22 → 24 | 7 → 7 |
| inhouse 44 (saved GT, colour sent) | 0.9859 | 0.9838 | 10 → 7 | 4 → 3 |
| inhouse 44 (saved GT, no colour) | 0.9744 | 0.9720 | 9 → 9 | 6 → 7 |
| public 100 (src GT, colour sent) | 0.9512 | 0.9512 | 34 → 37 | 15 → 15 |

- Third molars in the painted test (26 teeth): IoU 0.8886 → 0.8886.
- The "model stops short" teeth move both ways:
  - 1ff3 47: 0.806 → 0.784;
  - d639 46: 0.778 → 0.760;
  - 5c70 37: 0.776 → 0.795;
  - 27ec 47: 0.896 → 0.944.
- Inhouse 663a lost two teeth to a merge: 0.954 → 0.154 and 0.968 → 0.603.
- **ft1 does not help.**

**Why, probably:** the inhouse saved labels were completed on top of auto-label. Their gum-line boundary is mostly
the model's own (base IoU against them is 0.986, against the independent colour map 0.962). Of the 104 DDH-convention
training scans, only the 27 colour-map scans carry a boundary the model did not draw.

## Run ft2: weight the independent labels

Same recipe as ft1, with three changes:
- the 27 colour-map scans repeated 12 times;
- every other DDH-convention scan once;
- `--lr 3e-4 --colour-p 0.3`.

That gives 324 + 77 + 675 = 1,076 per epoch (`lists/ft2_train_names.txt`). Validation grey mAP ends at 92.6 and
colour at 97.6. Logs: `logs/ft2_train.log`, `logs/ft2_tooth_compare.txt`.

| test set | base IoU mean | ft2 IoU mean | teeth < 0.85 base → ft2 | lost (< 0.5) |
|---|---|---|---|---|
| painted Teeth3DS 50 (no colour) | 0.9615 | 0.9611 | 22 → 25 | 7 → 7 |
| inhouse 44 (colour) | 0.9859 | 0.9844 | 10 → 9 | 4 → 2 |
| inhouse 44 (no colour) | 0.9744 | 0.9744 | 9 → 9 | 6 → 7 |
| public 100 (colour) | 0.9512 | 0.9516 | 34 → 37 | 15 → 15 |

- Third molars: 0.8886 → 0.8886.
- Case 341 tooth 21: 0.881 → 0.896. The other targets move both ways: 1ff3 47 0.806 → 0.757, d639 46 0.778 → 0.792,
  5c70 37 0.776 → 0.790.
- Inhouse 663a merges again (0.954 → 0.596).
- **ft2 does not help either.**

## Conclusion (2026-11-17)

Neither fine-tune moves the gum-line band or the third molars on held-out scans. Both add regressions, so neither is
packaged or deployed. The deployed model stays `toothsegnet-ios/1`, with the post-processing fixes of
`../2026-11-17_autolabel-incomplete-teeth`.

What a model-level change would need:
1. **Boundaries the model did not draw.** Labels completed from auto-label inherit its gum line, so training on them
   teaches the model its own boundary back. Only 27 such scans were available here.
   - It needs scans labelled from scratch (or gum lines reviewed on their own), in the hundreds.
   - The hard cases come first: recession, crowns, the end of the arch, partly erupted third molars.
2. **Knowing how much of the band is real.** The painted colour map, the Teeth3DS labels and the inhouse labels put
   the gum line in different places.
   - Two people labelling the same 10–20 scans would show how wide that disagreement is.
   - If it is about as wide as the 1–3 mm band, more training will not close it.
3. **Then architecture.** The mask is predicted on 0.25 mm voxels and spread to vertices by kNN. A mesh-level
   boundary refinement head is a larger project, worth it only after 1 and 2.

The cache volume `ddh-train-cache` (1.5 GB, patient scans) and the runs volume `ddh-train-runs` (ft1, ft2) were
deleted on 2026-11-17 at the user's request. A re-run starts again from the export step below.

## Commands

```bash
# lists (asset ids only)
python scripts/make_split.py <assets.txt from psql> ../2026-11-17_autolabel-incomplete-teeth/logs/stages100.jsonl \
  <list_eval_public.txt> <list_eval_inhouse.txt> lists
# export (read-only, dashboard-server image, env from the running dashboard-server; 4 workers on split lists)
docker run --rm -i --network dental-data-hub_ddh-net --env-file <env> -v <this>/scripts:/s:ro -v ddh-train-cache:/out \
  --entrypoint python ddh-dashboard-server:dev /s/export_cache.py < train_ddh.txt+train_replay.txt
# train
docker run -d --name ddh-ft1 --gpus all --shm-size 8g -e PYTHONDONTWRITEBYTECODE=1 \
  -v <repo>/ios-models/segmentation3d:/work/segmentation3d:ro -v <repo>/dental-ai/dental_ai:/work/dental_ai:ro \
  -v ddh-train-cache:/work/data:ro -v ddh-train-runs:/runs -v <repo>/models/toothsegnet-ios/1:/base:ro \
  -v <this>/lists:/lists:ro -w /work/segmentation3d --entrypoint python hdl-gpu:latest training/train.py \
  --stage instance --init-from /base/instance.pt --out /runs/ft1 --cache /work/data/cache \
  --names /lists/ft1_train_names.txt --val-names /lists/ft1_val_names.txt --epochs 20 --lr 2e-4 --warmup 1 \
  --val-every 5 --workers 6
```

# 2026-11-17 Auto-label sometimes misses part of a tooth

**Question (user, 2026-11-17):** "ㅇㅋ. 컬러 문제인거 확인했어. 그런데, 거의 잘 나오긴 하는데 가끔가다 치아 형상을 다
못잡는 경우가 있어. 뭐가 문제인지 분석하고, 문제 찾았으면 개선 진행해줘."

The question is about `toothsegnet-ios/1` with **컬러 미사용** (no colour) on the 100 Teeth3DS scans without labels.

Ground truth: the painted colour map of those 100 scans (2026-11-16_teeth3ds-unlabeled). The most common colour is
gingiva and every other colour is one tooth, giving 1,389 teeth.

## Answer

Of 1,389 teeth, 23 (in 18 scans) came out below IoU 0.85. They fall into these groups:

| group | teeth | what happens | fixed? |
|---|---|---|---|
| **TTA fusion anchored on pose 0** | 1 (whole molar) | Pose 0 merged molar 46 into its neighbours. Poses 1–6 each found it cleanly, but every pose is matched to pose 0 only, so their 46 queries landed in different empty slots. The mean came out at 0.48, below 0.5, and the whole tooth became gingiva. The value sits on the 0.5 line: the live server lost this tooth on one run and kept it on the next. That fits "가끔가다" (occasionally). | **yes**: `fuse_consensus` |
| **Model stops short** | 6 | The back of the last molar where the scan ends (1ff3 47, 5c70), and a 1–3 mm band at the gum line (64cd 14 with recession, d639 crowned 46, 57aa 25, d639 48). All 8 poses agree, and each covers only about 80% of the tooth. | no (needs training data; see below) |
| Model merge | 2 | A small third molar merged into the second (6bac 28 into 27). | no |
| Not a model error: implants, healing caps, small objects painted as teeth | 5 | 02d6, 5d0e, 9525, e7a6 (implant/healing abutment), 0f20 48 (31 mm² piece) | — |
| Not a model error: two teeth painted in one colour | 8 | colour regions of 182–408 mm²; the prediction splits them into the two real teeth (aa34 and 1ff3 checked by picture, E8) | — |
| **Clean-up splits one tooth on overlapping scan sheets** (found in E8; first counted in the row above) | 1 (+1 whose second sheet is painted gingiva) | Case 322 (de81): the network gives the molar one query in all 8 poses, but the mesh holds two unwelded sheets 0.26 mm apart. `components` makes each sheet a tooth, so numbering gives 16 + 17 (and 18 + 28 on the next tooth) | not yet; see E8 |

A second bug turned up along the way. `absorb_fragments` stopped after 4 rounds. On tiny loose mesh pieces
(0.04–0.17 mm²), the tooth and gingiva labels swap every round and never settle. Specks of 0.0 mm² survived as
"teeth": invisible on screen, but each took a tooth number. They cost F1@75 0.983 → 0.976 on the 100 scans.
**Fixed**: tooth pieces still under 5 mm² after the rounds become gingiva.

Most of the remaining area the model calls gingiva is not visible as a missing piece. 65% of it lies within 0.3 mm
of the predicted tooth, i.e. one or two vertex rings at the bottom of the gum-line groove (E4).

## E1 Stage-by-stage diagnosis: `scripts/stages.py` → `logs/stages100.jsonl`, `logs/stages_summary.txt`

The script runs the deployed instance pipeline step by step on the same network output, in a throwaway container of
the image before the change. It records where tooth area turns into gingiva and scores post-processing variants.
Its v0 matches the running server: same partition on at least 99.5% of the area, mean 99.98%.

| variant (same network output) | F1@75 | scans F1=1 | IoU mean | tooth→gum mm²/scan | gum→tooth share |
|---|---|---|---|---|---|
| v0 deployed | 0.9760 | 72 | 0.9672 | 25.0 | 0.0102 |
| v0 without clean-up (`absorb_fragments` + `components`) | 0.9830 | 80 | 0.9668 | 26.8 | 0.0107 |
| v1 mask extent from sigmoid alone (P(tooth) only picks queries) | 0.9728 | 71 | 0.9674 | 24.0 | 0.0098 |
| v2 average each query over the poses that found it | 0.9600 | 59 | 0.9662 | 23.9 | 0.0102 |
| v3 threshold 0.4 | 0.9724 | 71 | 0.9670 | 17.5 | 0.0141 |
| v3 threshold 0.3 | 0.9719 | 66 | 0.9652 | 10.3 | 0.0189 |
| v4 pose 0 only (no TTA) | 0.9724 | 68 | 0.9624 | 27.7 | 0.0125 |

- A lower threshold only trades tooth area called gingiva for gingiva called tooth, and mean IoU does not move.
  The 0.3–0.5 band is genuinely uncertain. Of the missed area, 49% has confidence 0.3–0.5 and 23% has less than 0.1.
- The clean-up step lowering F1 led to the speck bug above (`scripts/specks.py` → `logs/specks.txt`: the specks sit
  on mesh shells of 0.04–0.17 mm² that hold both labels).
- Weakest position: 8 (third molars), IoU mean 0.902. The others are 0.96–0.97.

## E2 Pose by pose: `scripts/pose_debug.py` → `logs/pose_debug.txt`

For molar 46 of 4e325c26, pose 0 claimed 13% of the tooth, inside queries that are mostly other teeth. Poses 1–6
claimed 97–100% with P(tooth) ≈ 1, and were aligned to pose-0 slots q9, q21, q21, q21, q21. The fused confidence
was 0.48, so 2% of the tooth was claimed. An alignment-free foreground (mean over poses of the max query) calls 100%
of it tooth.

For the "model stops short" teeth (1ff3 47, 5c70 last molar, 64cd 14), every pose uses the same query and claims
66–91%. Fusion is not the cause there.

## E3 What the user changed after auto-label: `scripts/saved_vs_auto.py` → `logs/saved_vs_auto.txt`

Of 19 saved labels on these scans, most changed less than 5 mm². The largest edits were 85cad8a8 (+45 / −29 mm²),
b60ceea8 (+19 / −17), 64cd551f (+18 / −4) and e7a6c49e (−48, an implant). The saved labels match the colour map
about as well as auto-label does.

## E4 Is the boundary a groove the prediction stops above? `scripts/boundary_geometry.py` → `logs/boundary_geometry.txt`

| | signed curvature 1/mm, median | concave share |
|---|---|---|
| true boundary (colour map) | −0.94 | 0.93 |
| predicted boundary | −0.87 | 0.94 |
| missed band (tooth called gingiva) | −0.67 | 0.75 |
| tooth inside | +0.15 | 0.25 |

Missed tooth area by mesh distance from the predicted tooth: 0–0.3 mm 65%, 0.3–1 mm 15%, 1–2 mm 8%, 2–4 mm 10%,
more than 4 mm 2%.

## E5 Snapping the boundary to the groove (marker watershed): negative. `scripts/refine_watershed.py`

Markers came from the prediction: tooth interior and gingiva more than R mm from the boundary. The band in between
was flooded in order of −curvature, so the regions meet where the surface is most concave. Results on the first 28
painted scans:

| | IoU mean | tooth→gum mm²/scan | gum→tooth mm²/scan |
|---|---|---|---|
| prediction | 0.9654 | 22.8 | 20.1 |
| watershed R 0.4 / 0.6 / 1.0 mm | 0.952 / 0.952 / 0.951 | 49.3 / 48.5 / 47.8 | 11.0 / 11.4 / 12.3 |
| same, +curvature (sign flipped), R 0.6 | 0.816 | 169.5 | 51.0 |

The network's boundary is better than snapping to geometry, so this is not used.

## E6 The change against the deployed code, paired: `scripts/eval_fix.py`, `scripts/compare.py` → `logs/pair_*.jsonl`, `logs/compare_*.txt`

The network runs once per scan. The old post-processing (deployed: fuse_poses + old absorb) and the new one
(fuse_consensus + new absorb) are applied to that same output, plus each change alone. The live server is not used:
two runs of the same code differ by GPU non-determinism (the server lost molar 46 of 4e325c26 on one run and kept
it on the next).

**Partial (completed in E8, all three sets).** At 17:2x UTC Claude Code stopped both runs because the PC was low on memory: the CBCT run uses about 24
GB of the 30 GB WSL VM. Their leftover containers were removed. Done so far:
- 51 of 100 painted scans (3 runs each);
- 26 of 101 inhouse scans;
- public sets: not started.
Logs: `logs/pair_*_partial.jsonl`, `logs/compare_*_partial.txt`.

| set (GT) | n | | F1@75 | IoU mean | teeth < 0.85 | teeth lost (< 0.5) | tooth→gum mm²/scan |
|---|---|---|---|---|---|---|---|
| painted Teeth3DS (colour), no colour sent | 51 | old | 0.9751 | 0.9642 | 14 | 6 | 27.6 |
| | | fusion only | 0.9759 | 0.9656 | 13 | 5 | 25.0 |
| | | speck only | 0.9808 | 0.9642 | 14 | 6 | 27.6 |
| | | **new** | **0.9816** | **0.9656** | **13** | **5** | **25.0** |
| inhouse (saved labels), colour as default | 26 | old = new | 0.9847 | 0.9805 | 6 | 3 | 13.6 |

- Painted: 8 scans changed, and all for the better. 4e325c26 regains molar 46 in all 3 runs (IoU 0 → found). The
  rest lose 1–3 specks (n_pred down to the GT count).
- Inhouse: identical on all 26 (no pose-0 merge and no specks there).
- No run got worse.

## E7 Case 452 / 453 (user, 2026-11-17): not reproduced. `scripts/score_src.py`, `scripts/islands.py` → `logs/case452_453.txt`, `logs/pair_case452_453.jsonl`

**Question:** "Case 452, 453 이거 컬러없이 추론 돌리면 일부 영역이 치아로 안나오거나, 같은치아인데 여러 치아로
인식되는 경우가 있어."

- Case 452 and 453 are Teeth3DS patients. Their 4 scans are e9478b72 / 55139e20 (452 lower / upper) and c90d9c5d /
  e5ec207e (453 lower / upper). No other institution has 452 or more patients. The scans carry no colour: rgb is
  (0,0,0) everywhere, so 컬러 사용 and 컬러 미사용 send the same thing. Ground truth is the Teeth3DS label/fdi in
  mesh.ply.
- The live server ran 7 times on each scan through probe.py, which sends what the dashboard sends. Every run gave
  F1@75 1.0, IoU mean 0.971–0.977, no merges and no splits, and the same numbers to 0.001.
  - Each predicted tooth is a single mesh piece, with no gingiva holes inside teeth.
  - The lowest tooth is 453 upper 27 at IoU 0.923, with 4.6% of it (about 10 mm²) called gingiva at the back of the
    last molar, the known "model stops short" pattern.
- The paired old/new post-processing gave identical results over 3 noisy repeats (`changed_share` 0).
- The inference-server log since 2026-11-10 has one request for these scans before this check: 452 upper at
  2026-11-16 22:04 KST, with the same output. None for 453.
  - The user's requests this morning (08:09–08:13 KST) were Teeth3DS Case 321, 322 and 341 (or 868) upper.
  - 322 upper (de81ba7c) gave 16 teeth for 14. This was first read as two specks; E8 shows it is two teeth each
    split across overlapping scan sheets.
  - 341 / 868 upper gave 15 for 14: one speck.
- The labeling screen keeps the vertex order (`parsePly`) and only skips locked labels (`autoLabelCodes`), so it
  shows the server's output unless a label was locked.
- Open: the user to say which jaw and tooth, or send a screenshot.

## E8 Case 322 / 341: the cases the user meant ("322, 341이다 미안")

Both are Teeth3DS upper scans: 322 = de81ba7c, 341 = 77fe8405. Both are among the painted 100. Logs:
`logs/pose_debug_case322_341.txt`, `logs/case322_341_mesh.txt`, `logs/split_pieces_paint.txt`,
`logs/merge_close_paint.txt`. Scripts: `closeup.py` (pictures), `split_pieces.py`, `merge_close.py`,
`sheet_check.py`, `missed_region.py`.

**Case 322: "같은 치아인데 여러 치아로". The model is right; the clean-up step splits the tooth.**
- In all 8 poses one query covers 98% of the molar (colour tooth 13, 296 mm²), and the raw instance holds 99% of it.
- The mesh carries two scan sheets over this molar that are not welded together.
  - Where they overlap, they are 0.26 mm apart along the normal, with the same normal direction.
  - Both have open borders inside the tooth, with 93 mm of open border in the predicted teeth.
- `components()` gives each mesh-connected piece its own tooth: 141 + 156 mm². Numbering then gives them 16 and
  17, which show speckled where the sheets overlap.
- The next molar is split the same way (107 + 119 mm²) into 18 and 28. The colour map paints only one of its
  sheets.
- So 16 teeth come out for 14. The earlier note that this was "two specks" was wrong; it was put in the "painted
  in one colour" group by its area.

**Case 341: "일부 영역이 치아로 안나옴". This is the model.**
- 21 loses 16.7 mm² (11%) of its labial cervical surface, reaching up to 2.3 mm into the crown.
- The mesh there is clean: one piece and no open borders.
- All 8 poses cover only 85–93% with the same query.
- The 15th instance is a speck, fixed by the speck rule.

**How often does the clean-up split one instance (painted 100, raw instances, new absorb)?**

Splits happen in 4 scans:
- de81: two wrong splits, the sheets above.
- 3c50 and 89bb: a tooth plus a 5–13 mm² piece of gingiva more than 5 mm away. The split turns that piece into an
  extra "tooth".
- 02d6: two pieces at an implant, both gingiva.

| clean-up variant | F1@75 | scans F1=1 | IoU mean | teeth < 0.85 | predicted teeth |
|---|---|---|---|---|---|
| absorb + components (deployed and current change) | 0.9814 | 78 | 0.9672 | 23 | 1407 |
| join pieces of one instance closer than 0.5 mm (0.1–1.0 mm all the same) | 0.9820 | 78 | 0.9672 | 23 | 1405 |
| no mesh split: keep the network's instance as it is after absorb | **0.9830** | **80** | 0.9671 | 23 | 1402 |

- Keeping the network's instance is best here: it fixes de81 and removes the 2 extra "teeth".
- `components` dates from the PurePTv3 adapter, which had no instances.

**All three sets.** The user said "1,2,3 다 진행해줘": check on inhouse and public, deploy, then fine-tune.
- `eval_fix.py` now also saves the raw consensus instances and a `nosplit` side.
- `rescore.py` adds `gap0.5` (join the pieces of one instance that come within 0.5 mm) and drops GT "teeth" under
  5 mm².
- Logs: `logs/e8_pair_*.jsonl`, `logs/e8_compare.txt`, `logs/e8_rescore.txt`.

Why drop small GT teeth: the inhouse saved labels were completed on top of the old auto-label.
- Some keep its 0–1 mm² specks as numbered "teeth": c673 GT tooth 8 is 14 vertices, 0.0 mm²; 7055 GT teeth 8 and 15
  are 0.1–0.2 mm².
- The old code's specks match them exactly. With small GT teeth kept, the speck rule looks like 3 lost teeth on
  inhouse.

| F1@75 (GT teeth ≥ 5 mm²) | old (deployed before) | new (fusion + speck) | nosplit | **gap0.5 (deployed)** |
|---|---|---|---|---|
| painted Teeth3DS 100 (colour GT, no colour sent, run 1 of 3) | 0.9763 | 0.9822 | 0.9835 | 0.9828 |
| inhouse 101 (saved labels, colour sent) | 0.9763 | 0.9770 | 0.9835, but 1 merge (52e3: two separate teeth joined) | 0.9797, 0 merges, teeth < 0.85: 16 → 15 |
| public 3d_iosseg / fusion / poseidon3d / teeth3ds (25 each) | 0.991 / 0.995 / 0.930 / 0.986 | 0.991 / 0.995 / 0.934 / 0.986 | IoU min drops on 30e0, f2cc (far gingiva pieces join a tooth) | 0.991 / 0.995 / 0.935 / 0.988 |

- On the painted set, lost teeth over 3 runs drop from 24 (old) to 21 (new and gap0.5). No scan varies between
  runs any more.
- `gap0.5` is never below `new` on any scan of the three sets. It fixes Case 322 and makes no merge. `nosplit` scores
  higher F1 but joins far pieces: one merge and lower IoU on 3 scans. So `gap0.5` was deployed
  (`toothseg.join_close_pieces`, `JOIN_GAP_MM = 0.5`).
- Checked after deploying, through the dashboard API with colour off:
  - Case 322 gives 14 teeth: the molar 97% one number (16), the next 96% 17.
  - Case 341 gives 14.
  - 4e325c26 keeps 46.

## Third molars (user, 2026-11-17: "사랑니까지 고려한다면 모델을 어떻게 개선하는게 좋을까?")

From `logs/stages100.jsonl` (painted 100). FDI comes from the palette, so a few may be off.

| position | teeth | IoU mean | IoU median | below 0.85 | lost (< 0.5) | area median mm² |
|---|---|---|---|---|---|---|
| 8 | 33 (in 20 scans) | 0.902 | 0.968 | 3 | 2 | 138 |
| 7 | 174 | 0.972 | 0.982 | 3 | 0 | 189 |

- A fully erupted 8 scores like a 7.
- The losses are small 8s:
  - 6bac 28 (48 mm²) merged into 27;
  - 0f20 48 (a 31 mm² piece) called gingiva;
  - d639 48 and 85ca 48 lose 10–13% at the gum line.
- Suggested order:
  1. A third-molar test list.
  2. Training data with small or partly erupted 8s, with oversampling and distal-end crops.
  3. A semantic-guided 7/8 split as a post-processing experiment.
  4. Numbering last.

## The change (inference-server)

- `toothsegnet.py`:
  - The adapter now runs segmentation3d's `PrepareStage`, `InstanceNetwork` and `FinishStage` itself, instead of
    `Segmenter`.
  - The poses are merged by `fuse_consensus`: each pose's queries are Hungarian-matched (soft IoU) to the mean of the
    poses before it, not to pose 0. When all poses line up the same way, the result equals the old one (test).
  - Scores are the matched poses' mean P(tooth), not pose 0's alone.
- `toothseg.py` `absorb_fragments`: tooth pieces still under 5 mm² after the rounds become gingiva.
- Tests:
  - `test_fusion_keeps_a_tooth_that_pose_0_merged_into_its_neighbour`: the old fusion gives that tooth 0.475.
  - `test_fusion_equals_the_plain_mean_when_poses_agree`.
  - `test_absorb_fragments_clears_tooth_specks_on_loose_mesh_bits`.
- `toothseg.py` `join_close_pieces` (E8), used by `toothsegnet.py` instead of `components`.
  - Test: `test_join_close_pieces_keeps_a_tooth_on_two_overlapping_sheets_as_one`.
- **Status: deployed on dev, 2026-11-17 ~09:07 KST.** The image was rebuilt; 31 tests pass in the build.
  - The image before the change is tagged `ddh-inference-server:2026-11-10`, for rollback and for `stages.py` and
    `pose_debug.py`, which need the old `tsn.instance`.
  - Rollback: `docker tag ddh-inference-server:2026-11-10 ddh-inference-server:dev`, then
    `docker compose --profile autolabel up -d --no-deps inference-server`.
  - Not committed.
- `docker-compose.yml`: the `SEG3D_SRC` default is now `./ios-models/segmentation3d`. The old `../AI/segmentation3d`
  is no longer on disk. The repo copy differs from the code in the old image only in line breaks and one comment.

## What is left (the user decides)

- **Model stops short** at the back of the last molar and on a 1–3 mm band at the gum line (recession, crowns).
  This is the network itself: every pose agrees, and geometry snapping (E5) and lower thresholds (E1) do not help.
  It needs training examples of these cases. The 100 painted scans are exactly such labels (2026-11-16 option 2),
  but using them for training means they can no longer serve as this test set.
- Implants and healing caps are outside what the model was trained on.

## Commands

```bash
# scratch: S=<session scratch>/t3probe.  Mesh exports and renders stay there (patient-derived).
# probe export (read-only, dashboard-server image): ../2026-11-16_teeth3ds-unlabeled/scripts/probe.py, PROBE_MODES=grey|stats
# stage diagnosis (image before the change):
docker run --rm -i --gpus all -v $R/models:/models:ro -v <this>/scripts:/s:ro -v $S/out100:/in:ro -v $S/stages:/out \
  --entrypoint python ddh-inference-server:2026-11-10 /s/stages.py /in /out < list_t3all.txt > stages100.jsonl
python scripts/summarise.py stages100.jsonl --teeth
# paired old/new (the image's /app is the old code, /new the changed inference-server folder):
docker run --rm -i --gpus all -v $R/models:/models:ro -v $R/inference-server:/new:ro -v <this>/scripts:/s:ro \
  -v $S/out100:/in:ro -v $S/pair_paint:/out --entrypoint python ddh-inference-server:2026-11-10 \
  /s/eval_fix.py /in /out --gt colour --rgb grey --repeats 3 < list_t3all.txt > pair_paint.jsonl
python scripts/compare.py pair_paint.jsonl
python scripts/render.py $S/out100 $S/stages <asset> out.png --tooth <colour id> v0_deployed   # picture, keep outside the repo
```

# 2026-11-16 Teeth3DS scans without labels: why auto-label is poor

**Question (user, 2026-11-16):** "지금 teeth3ds에서 라벨링 데이터 없는거 자동 라벨 돌려보고있는데, 성능이 안좋게 나오는
경우가 많네. 치아를 이상하게 잡는 케이스가 많아. 왜 그런지 원인 분석 진행해줘"

Model on dev: `toothsegnet-ios/1` (instance in_ch 10 = xyz, normal, jaw, **colour**; semantic in_ch 7, no colour).

## Answer

The 100 Teeth3DS scans without labels are the only Teeth3DS scans with vertex colour, and that colour is not
scanner colour. It is a painted label map: gingiva (232,232,232), and one flat colour per tooth from a fixed
palette (upper 1x = red ramp (255,k,k), 2x = blue ramp (k,k,255); lower either green (0,k,0) / purple, or the
same red/blue ramps), k = 0, 56, 85, 113, 141, 170, 198, 226 for positions 1..8 (lower green 255..28).
`has_color` is true for them, so the dashboard sends that colour to the instance model.

The instance model learnt colour from `synth_colour` (ivory teeth, red-pink gingiva) after per-scan
standardisation (`standardise_colour`). In standardised terms the painted map is close to inverted: light grey
gingiva looks like tooth, and the most saturated red teeth (central and lateral incisors) look like gingiva.
They are dropped or eaten by the gum, and the rest get extra splits.

Removing only the colour fixes it (same scans, same server, same TTA).

## E1 probe: with colour vs without colour (20 scans)

Scripts:
- `scripts/probe.py`: runs in a throwaway `ddh-dashboard-server:dev` container. It sends the same arrays as the
  dashboard (`_scan_arrays`) to `POST /v1/ios/segment`, once with rgb and once without. Read-only: SELECT plus S3
  GET only.
- `scripts/colour_regions.py`: area-weighted F1@75 against the colour map, where most common colour = gingiva
  and every other colour = one tooth.
- `scripts/per_colour.py`: the same scoring per palette colour.

Scans: the 5 coloured Teeth3DS scans the user saved labels for today, plus 15 random others (`md5(asset_id)`
order); 9 upper, 11 lower.

```
# in Git Bash, scratch dir as cwd; keys copied from the running dashboard-server, never printed
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY PGHOST PGPASSWORD PGDATABASE PGUSER DDH_INFERENCE_URL; do
  export $v="$(docker exec dental-data-hub-dashboard-server-1 printenv $v)"; done
MSYS_NO_PATHCONV=1 docker run --rm -i --network dental-data-hub_ddh-net -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -e S3_ENDPOINT=rustfs:9000 -e PGHOST -e PGPASSWORD -e PGDATABASE -e PGUSER -e DDH_INFERENCE_URL \
  -e PROBE_MODES=colour,grey -v "<this folder>\scripts:/s:ro" -v "<scratch>\out:/out" \
  --entrypoint python ddh-dashboard-server:dev /s/probe.py < list_colour.txt > colour.jsonl
python colour_regions.py out colour.jsonl --palette ; python per_colour.py out colour.jsonl
```
The mesh exports (`/out/*.npz`) stay in session scratch. Re-create them with the command above.

| 20 scans | colour sent (= dashboard today) | colour removed |
|---|---|---|
| tooth F1@75 vs colour map, mean | **0.909** | **0.983** |
| scans at F1 1.0 | 1 / 20 | 16 / 20 |
| tooth/gingiva area agreement | 0.908 – 0.981 | 0.971 – 0.995 |
| upper scans missing 11 or 12 | 5 / 9 | 0 / 9 |

Per colour: the share of the tooth called gingiva, colour sent / removed. The worst colours, with numbers from
`per_colour.py`:

| colour | tooth | n | IoU sent / removed | called gingiva sent / removed |
|---|---|---|---|---|
| (255,0,0) upper | 11 | 9 | 0.28 / 0.97 | 0.71 / 0.01 |
| (255,56,56) upper | 12 | 9 | 0.40 / 0.97 | 0.57 / 0.01 |
| (255,0,0), (255,56,56) lower | 31, 32 (red/blue palette) | 1 | 0.00 / 0.97 | 1.00 / 0.00 |
| (128,0,255) lower | 48 | 2 | 0.00 / 0.93 | 1.00 / 0.05 |
| (0,28,0) lower | 38 | 3 | 0.59 / 0.94 | 0.34 / 0.01 |
| (255,85,85) upper | 13 | 9 | 0.88 / 0.98 | 0.10 / 0.01 |
| pale colours (k >= 113) | the rest | | 0.93 – 0.99 both | <= 0.03 both |

**The colour map is the real tooth boundary.**
- The prediction without colour matches it at F1 1.0 on 16 of 20 scans.
- The 5 labels the user saved today match it at F1 1.0, 1.0, 1.0, 1.0 and 0.938 (one tooth at IoU 0.69).
- Each tooth colour is one mesh-connected piece.

**The palette is fixed.** The no-colour prediction gives the same FDI to the same colour in 9–10 of 9–10 scans
(the palette table is in `colour_regions.py --palette` output). Exception: one lower scan (d378d121) uses the
red/blue palette. On it the colour says 31–37 + 41–47, while the model says 31–33, 35–38 (an off-by-one on a
premolar gap). Numbering from the palette is therefore close, but not proven.

## E2 all coloured scans in DDH: distinct colours (no inference)

`PROBE_MODES=stats` on every asset with `has_color`:

| institution | scans | distinct colours, min / median / max | grey (r=g=b) share |
|---|---|---|---|
| teeth3ds | 100 | 12 / 15 / 17 | 0.26 – 0.60 |
| inhouse (real scanner colour) | 123 | 3,135 / 49,886 / 124,662 | 0.00 – 0.08 |

All 100 Teeth3DS colour scans are painted maps, and no real scan has fewer than 3,000 colours.

## Follow-up (2026-11-17, at the user's request)

The user asked for two things: a colour view in the labeling screen, and auto-label split into "컬러 사용" and
"컬러 미사용".
- Dashboard: a `컬러` toggle (shortcut C) paints the mesh base with the scan colour, under the label overlay.
- Two auto-label buttons call `POST .../auto-label?colour=true|false`; the response carries `colour`.
- Deployed on dev (`ddh-dashboard-server:dev` eed14fdd). 119 unit tests and tsc pass.
- Browser check (Playwright, writes blocked):
  - Case 302 upper (painted): 컬러 미사용 gave 14 teeth, 컬러 사용 gave 12 (11 and 12 missing), the same as E1.
  - Case 683 upper (no colour): the colour toggle and 컬러 사용 are disabled.
- Not committed.

Options 1–3 below are still open. Option 1 (ignore painted colour automatically) is now a manual choice: the
user presses 컬러 미사용.

## Conclusion and options (not applied; the user decides)

1. **Ignore painted colour for auto-label.** If a scan has fewer than about 256 distinct colours, send no rgb.
   This is one line in `dashboard-server` `_scan_arrays` or in the inference server. Expected result: these
   scans go from F1 0.91 to 0.98, and inhouse is unchanged (all inhouse scans have more than 3,000 colours).
2. **Use the colour map as the label.** These 100 scans are already labelled in colour, and the boundaries
   match. A colour → (label 1, FDI) conversion would label all 100 without model or human work; numbering
   from the palette should be spot-checked. This writes `ios_labels`, so the user decides.
3. Releases use `use_color = false` (ios-teethseg, ios-teethseg-full), so no label leak today. If colour is
   ever turned on for training, these 100 scans would put the answer in the input.

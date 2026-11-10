"""E16b 대시보드 자동 라벨(배포한 서버)을 인하우스 정답 12개로 잰다.  E1과 같은 경로이고, E10과 같은 instance 지표다.

dashboard_fetch.py가 받은 결과(/fetch/autolabel.npz)를 GPU 없이 채점한다.  면은 다른 실험처럼 원본 PLY에서 읽는다.
대시보드는 색이 있는 스캔에 색을 보내므로, 기대값은 E16의 "+colour DDH"다.

    docker run --rm --entrypoint sh (E16과 같은 마운트, --gpus 없이) -v <fetch 폴더>:/fetch:ro hdl-gpu:latest \
        -c "python /exp/scripts/dashboard_e2e_eval.py /gt"
"""
import json
import sys
from pathlib import Path

sys.path[:0] = ["/src", "/seg3d"]
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402,F401
from scipy.sparse import coo_matrix, diags  # noqa: E402,F401
from scipy.sparse.csgraph import connected_components  # noqa: E402,F401
from scipy.spatial import cKDTree  # noqa: E402,F401

MIN_AREA = 5.0  # instance_eval.py의 기본값 (cleanup의 기본 인자)
_src = Path("/exp/scripts/instance_eval.py").read_text(encoding="utf-8")
for start, end in (("# ------------------------------------------------------------------ mesh",
                    "# ------------------------------------------------------------- PurePTv3"),
                   ("# ------------------------------------------------------------------ metrics",
                    "# ------------------------------------------------------------------ run")):
    exec(_src[_src.index(start):_src.index(end)])

from pipeline.config import Placement, SegmentationConfig  # noqa: E402
from pipeline.stages import LoadStage  # noqa: E402

GT = Path(sys.argv[1] if len(sys.argv) > 1 else "/gt")
RAW = Path("/data/ios/inhouse/raw")
_cfg = SegmentationConfig(device="cpu") if "device" in SegmentationConfig.__dataclass_fields__ else SegmentationConfig()
LOAD = LoadStage(_cfg, Placement.resolve(_cfg))
fetched = np.load("/fetch/autolabel.npz")

rows = []
for f in sorted(GT.glob("INHOUSE*.npz")):
    name = f.stem.split("_")[0]
    d = np.load(f)
    faces = LOAD.run(RAW / f"{f.stem}.ply").faces.cpu().numpy().astype(np.int64)
    edges = edges_of(faces)
    xyz = d["xyz"].astype(np.float64)
    area = vertex_area(xyz, faces)
    fdi = d["fdi"].astype(np.int64)
    gt = np.where(fdi > 0, np.unique(fdi, return_inverse=True)[1], 0)
    pf = fetched[f"{name}__fdi"].astype(np.int64)
    meta = json.loads(str(fetched[f"{name}__meta"]))
    assert len(pf) == len(fdi)
    ddh = np.where(pf > 0, np.unique(pf, return_inverse=True)[1], 0)
    r = score(gt, ddh, area, xyz, edges)
    gtooth, ptooth = fdi > 0, pf > 0
    r["tooth_f1"] = 2 * (gtooth & ptooth).sum() / max(gtooth.sum() + ptooth.sum(), 1)
    r["elapsed_ms"] = meta["elapsed_ms"]
    rows.append(r)
    print(f"{f.stem}: F50 {r['F50']:.2f} F75 {r['F75']:.2f} mIoU {r['mIoU']:.3f} m{r['merge']}s{r['split']} "
          f"tooth F1 {r['tooth_f1']:.3f} | {meta['elapsed_ms']} ms (wall {meta['wall_s']} s) "
          f"{meta['info'].get('graph')} tta {meta['info'].get('tta')} {meta['info'].get('quadrant_method')} "
          f"instances {meta['info'].get('instances')} | warnings {len(meta['warnings'])}", flush=True)

mean = lambda k: np.mean([r[k] for r in rows])  # noqa: E731
print(f"\n== 대시보드 자동 라벨, 인하우스 {len(rows)}개 (DDH 기준: 같은 FDI = 한 치아)")
print(f"   F50 {mean('F50'):.3f}  F75 {mean('F75'):.3f}  mIoU {mean('mIoU'):.3f}  "
      f"merge {sum(r['merge'] for r in rows)}  split {sum(r['split'] for r in rows)}  "
      f"tooth/gingiva F1 {mean('tooth_f1'):.3f}")
print(f"   대시보드 서버 응답 시간 평균 {mean('elapsed_ms') / 1000:.1f} s, 최대 {max(r['elapsed_ms'] for r in rows) / 1000:.1f} s")

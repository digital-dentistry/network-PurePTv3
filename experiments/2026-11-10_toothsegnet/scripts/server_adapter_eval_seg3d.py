"""E16과 같은 데이터·지표로 다른 어댑터를 잰다.  E16 = 2026-11-10_inhouse/scripts/server_adapter_eval.py.

그 스크립트(와 그것이 부르는 instance_eval.py)는 고치지 않는다.  읽어서 아래 세 곳만 바꿔 실행한다.
  - 어댑터 AD: --adapter toothsegnet이면 이 패키지의 ToothSegNet, pureptv3면 그대로 (toothseg-pureptv3/1)
  - 공개 크롭 모드: --modes (E16은 full quadrant half.  사용자가 브릿지도 나쁘다고 해서 bridge를 더할 수 있다)
  - --no-inhouse면 인하우스를 건너뛴다 (PurePTv3의 bridge만 더 잴 때)
지표는 번호를 무시한 instance 지표다 (F50/F75/mIoU/merge/split, 정의는 2026-11-10_inhouse/README.md "지표").

    python /exp/scripts/server_adapter_eval_seg3d.py --adapter toothsegnet --package /pkgs/toothsegnet-ios/1 --n 40
    마운트는 2026-11-10_inhouse/README.md와 같고, 그 폴더를 /inhouse_exp에, 이 폴더를 /exp에 더 붙인다.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", choices=("toothsegnet", "pureptv3"), required=True)
ap.add_argument("--package", type=Path, default=None, help="toothsegnet 패키지 폴더 (.../name/version)")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--modes", nargs="+", default=["full", "quadrant", "half"])
ap.add_argument("--no-inhouse", action="store_true")
a = ap.parse_args()
os.environ.setdefault("DDH_SEG3D_SRC", "/seg3d")

SRC = Path("/inhouse_exp/scripts")
instance_src = (SRC / "instance_eval.py").read_text(encoding="utf-8")
server_src = (SRC / "server_adapter_eval.py").read_text(encoding="utf-8")


def patch(src: str, old: str, new: str) -> str:
    assert src.count(old) >= 1, f"E16 스크립트가 바뀌었다: {old!r}를 찾지 못했다"
    return src.replace(old, new)


if a.adapter == "toothsegnet":
    instance_src = patch(instance_src, "AD = ToothSeg17(PKG, spec, DEV)",
                         "from toothsegnet import ToothSegNet\n"
                         "AD = ToothSegNet(SEG3D_PKG, json.loads((SEG3D_PKG / 'model.json').read_text()), DEV)")
server_src = patch(server_src, '_src = Path("/exp/scripts/instance_eval.py").read_text(encoding="utf-8")',
                   "_src = INSTANCE_SRC")
server_src = patch(server_src, 'for mode in ("full", "quadrant", "half"):', "for mode in MODES:")
server_src = patch(server_src, 'pr.info["quadrant_method"]', 'pr.info.get("quadrant_method", "-")')
server_src = patch(server_src, 'for f in sorted(GT.glob("INHOUSE*.npz")):',
                   'for f in ([] if NO_INHOUSE else sorted(GT.glob("INHOUSE*.npz"))):')

sys.argv = [sys.argv[0], str(a.n)]
print(f"adapter {a.adapter} package {a.package} modes {a.modes} inhouse {not a.no_inhouse}", flush=True)
exec(compile(server_src, str(SRC / "server_adapter_eval.py"), "exec"),
     {"__name__": "__main__", "INSTANCE_SRC": instance_src, "SEG3D_PKG": a.package, "MODES": tuple(a.modes),
      "NO_INHOUSE": a.no_inhouse, "json": json})

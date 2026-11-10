"""치아 분할 어댑터: PurePTv3 17-class semantic 모델 -> DDH 정점 라벨 (label, fdi).

모델 class (dental_ai ToothSegLoader와 같다)
    0       치은
    1..8    첫 사분면의 위치 1(중절치) .. 8(제3대구치)   상악 1x, 하악 3x
    9..16   둘째 사분면                               상악 2x, 하악 4x

학습 augmentation(SO(3) 회전 + 라벨을 바꾸는 좌우 반전) 때문에 이 모델은 좌우(사분면)를 구별하지 못하고 위치만 맞힌다
(ios-models/PurePTv3/README.md "첫 학습 결과와 추론").  그래서 사분면은 모델 출력 대신 기하로 정한다.
예측한 위치와 normal로 교합면 frame을 만들면 위·아래턱 모두 첫 사분면이 정중선의 왼쪽(x < 정중선)에 온다
(dental_ai ToothSegLoader._view_frame과 같은 계산).  정중선은 예측한 중절치가 있어야 잡히므로, 정중선을 넘지
않는 부분 스캔(quadrant, bridge)은 사분면을 하나로 몰고 경고를 낸다.

전처리는 학습과 같은 코드(dental_ai.dataset.ToothSegLoader.point_features, voxel_grid_cpu)를 쓴다.
voxel 대표점이 학습의 max_points보다 많으면 겹치지 않는 무작위 묶음 여러 개로 나눠 각각 추론한다 (학습에서
보던 밀도와 같게).  정점의 라벨은 그 정점이 속한 voxel의 대표점 예측이다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

NUM_CLASSES = 17
# class -> 위치 (0 치은, 1..8)와 사분면 (0 첫째, 1 둘째)
POSITION = np.array([0] + list(range(1, 9)) * 2)
SECOND = np.array([False] + [False] * 8 + [True] * 8)
QUADRANT_DIGIT = {"upper": (1, 2), "lower": (3, 4)}  # (첫 사분면, 둘째 사분면)의 FDI 십의 자리
MIN_TOOTH_POINTS = 30          # 이보다 적은 위치는 없는 것으로 본다 (대표점 수)
MIN_INCISOR_POINTS = 20


@dataclass
class Scan:
    xyz: np.ndarray                  # [V,3] float32 mm
    normals: np.ndarray              # [V,3] float32
    rgb: np.ndarray | None           # [V,3] uint8 또는 None (색 없음)
    jaw: str                         # "upper" | "lower"


@dataclass
class Prediction:
    label: np.ndarray                # [V] int8: 0 치은, 1 치아
    fdi: np.ndarray                  # [V] uint8: FDI, 치은 0
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ 사분면
def view_frame(coord: np.ndarray, normals: np.ndarray, position: np.ndarray) -> np.ndarray:
    """[3,3] 행이 x(오른쪽), y(전치 쪽), z(치관 쪽)인 회전.  dental_ai ToothSegLoader._view_frame과 같다."""
    pca = np.linalg.svd(coord - coord.mean(0), full_matrices=False)[2]
    tooth = position > 0
    z = normals[tooth].mean(0) if tooth.any() else pca[2]
    z = z / max(np.linalg.norm(z), 1e-9)
    if tooth.any() and np.ptp(position[tooth]) > 0:
        p, c = position[tooth], coord[tooth]
        y = -((p - p.mean())[:, None] * (c - c.mean(0))).sum(0)
    else:
        y = pca[0]
    y = y - (y @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = pca[1] - (pca[1] @ z) * z
    y = y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z])


def decide_quadrant(coord: np.ndarray, normals: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, str, list[str]]:
    """대표점마다 둘째 사분면인지 (bool), 판정 방법, 경고.

    "midline": 정중선 양쪽에 위치가 2개 이상씩 있다.  점마다 정중선의 어느 쪽인지로 정한다.
    "single": 정중선을 못 잡았다.  모든 치아를 한 사분면으로 둔다 (모델 출력의 다수결).
    """
    position = POSITION[pred]
    tooth = position > 0
    if not tooth.any():
        return np.zeros(len(pred), bool), "none", []
    x = (coord @ view_frame(coord, normals, position).T)[:, 0]
    incisor = position == 1
    if incisor.sum() >= MIN_INCISOR_POINTS:
        mid = float(np.median(x[incisor]))

        def seats(side: np.ndarray) -> set[int]:
            found, counts = np.unique(position[tooth & side], return_counts=True)
            return {int(p) for p, n in zip(found, counts) if n >= MIN_TOOTH_POINTS}

        left, right = seats(x < mid), seats(x >= mid)
        if len(left) >= 2 and len(right) >= 2:
            warnings = []
            if min(len(left), len(right)) < 4:
                warnings.append("정중선 한쪽에 치아가 적은 부분 스캔으로 보입니다. 정중선 가까운 치아의 좌우(사분면)를 확인하세요.")
            return x >= mid, "midline", warnings
    second = bool(SECOND[pred[tooth]].mean() >= 0.5)
    return np.full(len(pred), second), "single", [
        "정중선을 찾지 못한 부분 스캔입니다. 좌우(사분면)는 정하지 못해 한쪽으로 두었습니다. "
        "번호 바꾸기로 사분면을 확인하세요."]


def to_fdi(pred: np.ndarray, second: np.ndarray, jaw: str) -> tuple[np.ndarray, np.ndarray]:
    """class + 사분면 -> DDH (label, fdi)."""
    position = POSITION[pred]
    first_digit, second_digit = QUADRANT_DIGIT[jaw]
    quadrant = np.where(second, second_digit, first_digit)
    label = (position > 0).astype(np.int8)
    fdi = np.where(position > 0, quadrant * 10 + position, 0).astype(np.uint8)
    return label, fdi


# ------------------------------------------------------------------ 어댑터
class ToothSeg17:
    """model.json의 "adapter": "toothseg17"."""

    def __init__(self, package: Path, spec: dict, device: torch.device):
        from pureptv3 import PurePTv3Segmentor

        arch = spec["architecture"]
        self.spec = spec
        self.device = device
        self.pre = spec["preprocessing"]
        self.features = tuple(self.pre["features"])
        model = PurePTv3Segmentor(arch["num_classes"], arch["backbone_out_channels"], backbone=arch["backbone"])
        state = torch.load(package / spec["weights"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(device).eval()
        if arch["num_classes"] != NUM_CLASSES:
            raise ValueError(f"toothseg17 어댑터는 {NUM_CLASSES}-class 모델만 받는다 ({arch['num_classes']})")

    @torch.no_grad()
    def _logits(self, positions: torch.Tensor, normals: torch.Tensor, colour: torch.Tensor | None,
                cells: torch.Tensor, jaw_upper: bool) -> torch.Tensor:
        """대표점 한 묶음 -> [n, 17] 확률.  학습의 ToothSegDataset.__getitem__과 같은 입력을 만든다."""
        from dental_ai.dataset.ToothSegLoader import point_features

        coord = positions - positions.double().mean(0).float()
        feat = point_features(self.features, coord, normals, jaw_upper, colour)
        inputs = self.model.prepare(cells.to(self.device))
        autocast = self.device.type == "cuda"
        with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
            logits = self.model(feat.to(self.device), inputs)
        return logits.float().softmax(1).cpu()

    def predict(self, scan: Scan) -> Prediction:
        from dental_ai.dataset.algorithm.voxel_downsample.voxel_downsample import (VoxelDownsampleInput,
                                                                                    VoxelDownsampleOutput)
        from dental_ai.dataset.algorithm.voxel_downsample.voxel_grid_cpu import voxel_grid_cpu
        from dental_ai.dataset.ToothSegLoader import unit_normals

        voxel = float(self.pre["voxel_mm"])
        positions = torch.from_numpy(np.ascontiguousarray(scan.xyz, dtype=np.float32))
        normals = unit_normals(torch.from_numpy(np.ascontiguousarray(scan.normals, dtype=np.float32)))
        colour = None if scan.rgb is None else torch.from_numpy(scan.rgb.astype(np.float32) / 255.0)

        ds = VoxelDownsampleOutput()
        if not voxel_grid_cpu(VoxelDownsampleInput(positions, voxel), ds):
            raise ValueError("정점이 없습니다.")
        rep = ds.representative
        # 정점 -> voxel 번호.  voxel_grid_cpu와 같은 key라 대표점 순서(key 오름차순)와 번호가 같다
        cell = torch.floor((positions - positions.amin(0)) / voxel).long()
        dims = cell.amax(0) + 1
        key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
        inverse = torch.unique(key, sorted=True, return_inverse=True)[1]

        n = rep.numel()
        cap = int(self.pre.get("max_points") or 0) or n
        chunks = max(1, math.ceil(n / cap))
        order = torch.randperm(n, generator=torch.Generator().manual_seed(0)) if chunks > 1 else torch.arange(n)
        prob = torch.empty(n, NUM_CLASSES)
        for idx in torch.tensor_split(order, chunks):
            r = rep[idx]
            prob[idx] = self._logits(positions[r], normals[r], None if colour is None else colour[r],
                                     ds.cells[idx], scan.jaw == "upper")

        pred = prob.argmax(1).numpy()
        coord = positions[rep].double().numpy()
        second, method, warnings = decide_quadrant(coord - coord.mean(0), normals[rep].double().numpy(), pred)
        label, fdi = to_fdi(pred, second, scan.jaw)
        inv = inverse.numpy()
        teeth = sorted({int(f) for f in np.unique(fdi) if f})
        return Prediction(
            label=label[inv], fdi=fdi[inv], warnings=warnings,
            info={"points": n, "chunks": chunks, "quadrant_method": method, "teeth": teeth,
                  "mean_confidence": round(float(prob.max(1).values.mean()), 4),
                  "colour": scan.rgb is not None},
        )

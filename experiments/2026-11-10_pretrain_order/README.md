# 2026-11-10 사전 가중치와 직렬화 순서

사용자 요청은 두 가지다.
1. PTv3를 학습할 때 사전 가중치를 쓴다.
2. 점을 넣기 전에 직렬화 순서(point order)가 들어갔는지 확인하고, Hilbert 순서를 쓴다.

스크립트는 `scripts/`, 출력은 `results/`에 있다.

## 한눈에

| 결론 | 근거 |
|---|---|
| 직렬화는 원래 들어가 있다. 순서 4가지(z, z-trans, hilbert, hilbert-trans)를 쓰고, 학습 때는 stage마다 섞는다 | `train_toothseg.py` `backbone_config`, `preprocess.build_inputs` |
| 평가·추론에서는 블록 i가 `i % 4`번째 순서를 쓴다. 블록이 2개인 stage(인코더 0·1·2·4, 디코더 전부)는 z와 z-trans만 썼고, hilbert는 인코더 stage 3에서만 쓰였다 | `model.py` `order_index=i % n_orders` |
| 학습한 모델은 평가 순서에 둔감하다. hilbert를 먼저 두거나 Pointcept처럼 무작위로 섞어도 mIoU9 차이가 ±0.4 안이다 | E1 |
| 새 학습은 hilbert를 먼저 둔다(`--order`, 기본 `hilbert hilbert-trans z z-trans`). 효과는 측정되지 않았고, 사용자 요청이라 기본값으로 했다 | E1 |
| PT-v3m1에 그대로 들어가는 공개 사전 가중치는 Pointcept 공식 ScanNet/ScanNet200/S3DIS/nuScenes 학습 가중치뿐이다 (HF `Pointcept/PointTransformerV3`, MIT 표기) | 아래 "가중치 후보" |
| ScanNet 가중치로 학습 중 (`AI/runs/pureptv3_semantic_scannet`) | E2 |

## 가중치 후보 (2026-11-10 확인)

| 가중치 | 구조 | 학습 방식 | 라이선스 | PurePTv3 |
|---|---|---|---|---|
| Pointcept `scannet-semseg-pt-v3m1-0-base` | PT-v3m1 base (46 M) | ScanNet 지도학습, 입력 color(0..1)+normal, 2 cm voxel | HF 저장소 MIT 표기. 학습 데이터 ScanNet의 이용 약관은 비상업 연구용이다 | 그대로 읽힌다 |
| Sonata (`facebook/sonata`) | PT-v3m2 (LayerScale, Linear stem, LN, GridPooling), 인코더만 | 자기지도 140k 장면 | 가중치 CC-BY-NC 4.0 | m2 구현이 필요하다 |
| Concerto (`Pointcept/Concerto`) | PTv3 base/large | 2D-3D 자기지도 | CC-BY-NC 4.0 | m2/m3 구현이 필요하다 |
| Utonia (`Pointcept/Utonia`) | PT-v3m3 (RoPE) | 물체·실내·실외 통합 자기지도 | CC-BY-NC 4.0 | m3 구현이 필요하다 |

자기지도 모델(Sonata, Concerto, Utonia)이 일반적으로 더 좋은 초기값이다. 하지만 가중치가 비상업 라이선스이고, PurePTv3에 m2/m3 구조를 더해야 한다.
그래서 이번에는 구조를 바꾸지 않아도 되는 ScanNet 가중치를 썼다.

가중치 파일: `Desktop/AI/checkpoints/pointcept/scannet-semseg-pt-v3m1-0-base/model_best.pth`
(HF `Pointcept/PointTransformerV3`의 `scannet-semseg-pt-v3m1-0-base/model/model_best.pth`, sha256 `40206376ff2f83f48e4d1bc27d5c5d96be7c87c5d11eb45fe7be501959040e7f`).
pickle 안의 전역 이름을 실행하지 않고 먼저 확인했다(`pickletools`). 텐서, OrderedDict, OneCycleLR 상태, numpy 스칼라와 dtype뿐이었다.
그래서 이것만 허용하고 `weights_only=True`로 읽는다(`checkpoint._pointcept_safe_globals`).

## 실험

### E1 평가 순서만 바꾸기 — `scripts/eval_order.py` → `results/eval_order.{log,json}`

첫 학습 `AI/runs/pureptv3_semantic/best.pt`(epoch 85, 학습 순서 z 먼저)로 검증 158 스캔을 다시 쟀다.
가중치는 그대로이고 평가 때 순서 배치만 바꿨다.
colour 경로는 첫 학습 검증과 같은 legacy 합성 색이다.

| 평가 순서 | grey mIoU17 | grey mIoU9 | colour mIoU17 | colour mIoU9 | TSA (grey/colour) |
|---|---|---|---|---|---|
| z 먼저 (학습 설정) | 31.97 | 88.67 | 33.75 | 91.03 | 98.43 / 99.58 |
| hilbert 먼저 | 32.01 | 88.31 | 33.60 | 90.59 | 98.43 / 99.59 |
| 무작위 섞기 seed 0 (Pointcept 기본 평가) | 32.16 | 88.40 | 33.57 | 90.56 | 98.45 / 99.58 |
| 무작위 섞기 seed 1 | 32.05 | 88.52 | 34.11 | 91.04 | 98.43 / 99.58 |
| 무작위 섞기 seed 2 | 32.23 | 88.62 | 33.51 | 90.82 | 98.44 / 99.59 |

stage별 평균 점 수(스캔 하나)는 52229 / 18833 / 5734 / 1652 / 477이다. 패치(1024점) 수는 51.6 / 18.9 / 6.1 / 2.1 / 1.0이다.
순서는 stage 0~2의 패치 경계를 정한다.

결론: 학습 때 순서를 섞었기 때문에 평가 순서는 점수에 영향이 없다(차이는 무작위 섞기끼리의 흔들림과 같은 크기다).
hilbert를 먼저 둬도 좋아지지 않는다. 좌우(mIoU17)도 그대로다.

```bash
# hdl-gpu 일회용 컨테이너. D=C:/Users/wodyd/OneDrive/Desktop
docker run --rm --gpus all --shm-size 8g -v "$D/DentalDataHub/dental-ai:/app:ro" -v "$D/AI/data:/app/data:ro" \
  -v "$D/AI/runs:/runs:ro" -v "$D/DentalDataHub/ios-models/PurePTv3/experiments/2026-11-10_pretrain_order:/exp" \
  -v "$D/DentalDataHub/ios-models/PurePTv3:/pureptv3:ro" -e PYTHONPATH=/app/dental_ai/src -w /exp \
  --entrypoint python hdl-gpu:latest scripts/eval_order.py --ckpt /runs/pureptv3_semantic/best.pt --out /exp/results/eval_order.json
```

### E2 ScanNet 사전 가중치로 학습 — `AI/runs/pureptv3_semantic_scannet` (진행 중)

질문: 사전 가중치에서 시작하면 첫 학습보다 나은가? 특히 첫 학습이 끝내 배우지 못한 좌우(mIoU17)를 배우는가?
segmentation3d semantic은 같은 SO(3)+mirror 조건에서 epoch 67 무렵 좌우를 배웠다(ce 0.85 → 0.49).

- 초기화(`init_from_pretrained`):
  - backbone 텐서 485개를 그대로 복사했다. BN 통계도 포함한다.
  - `seg_head`는 class 수가 20에서 17로 달라 새로 시작한다.
  - stem conv는 입력 채널만 다르다. normal과 colour 채널은 체크포인트의 normal과 color 가중치를 가져오고, xyz와 jaw 채널은 0에서 시작한다.
  - 색 범위는 다르다(ScanNet은 0..1, 우리는 스캔별 표준화). 이 차이는 stem 뒤의 BN과 fine-tune이 흡수한다고 보고 따로 보정하지 않았다.
- 설정: `train_toothseg.py` 기본값.
  - 100 epoch, lr 0.002 (block ×0.1). Pointcept의 Sonata fine-tune 레시피와 같은 값이다.
  - 순서는 hilbert 먼저.
  - crop 0.5, 색 0.5 clinical. 같은 날 다른 세션이 바꾼 값이다.
- **비교 주의**: 첫 학습(crop 0.35, 색 0.6 legacy, z 먼저)과 증강 설정도 달라서, 차이가 모두 사전 가중치 효과라고 볼 수 없다.
  좌우를 배우는지(mIoU17이 크게 뛰는지)가 이번에 가장 볼 신호다.
- GPU는 다른 실험과 함께 쓴다. 0.42 s/it, epoch당 약 2.5분, peak 6.6 GB.

```bash
docker run -d --gpus all --shm-size 16g --label com.docker.compose.project=pureptv3 --name pureptv3-train-pre \
  -v "$D/DentalDataHub/dental-ai:/app:ro" -v "$D/AI/data:/app/data:ro" -v "$D/AI/runs:/app/runs" \
  -v "$D/AI/checkpoints:/app/checkpoints:ro" -v "$D/DentalDataHub/ios-models/PurePTv3:/pureptv3:ro" \
  -e PYTHONPATH=/app/dental_ai/src -w /pureptv3 --entrypoint python hdl-gpu:latest \
  train_toothseg.py --out /app/runs/pureptv3_semantic_scannet \
  --pretrained /app/checkpoints/pointcept/scannet-semseg-pt-v3m1-0-base/model_best.pth
```

결과: (학습이 끝나면 채운다)

## 코드 변경

- `pureptv3/checkpoint.py`:
  - `init_from_pretrained` 추가. 같은 모양은 복사하고, stem 입력 채널은 다시 매핑하고, head는 새로 시작한다. 그 밖의 불일치는 에러로 막는다.
  - Pointcept 학습 체크포인트를 `weights_only=True`로 읽는다.
- `train_toothseg.py`:
  - `--pretrained`, `--pretrained-features`(기본 `colour normal`), `--order`(기본 hilbert 먼저)를 추가했다.
  - 사전 가중치 경로, sha256, 채널 매핑을 체크포인트 `meta["pretrained"]`에 남긴다.
- `tests/test_native.py`: `test_init_from_pretrained_adapts_stem_and_head`. 13/13 통과.

# PurePTv3

Pointcept의 Point Transformer V3(`PT-v3m1`)를 **순수 PyTorch API**로 다시 쓴 구현입니다.
spconv, FlashAttention, torch_scatter, timm, addict 없이 돌아갑니다.

- 학습된 Pointcept 체크포인트를 **변환 없이** 그대로 읽습니다. 파라미터 이름과 모양이 원본과 같고, sparse conv 가중치도 spconv 2.x 레이아웃 `[Cout, kx, ky, kz, Cin]`을 그대로 씁니다.
- 데이터에 따라 정해지는 인덱스 계산(직렬화, 이웃 탐색, 패치 패딩, 풀링 묶음)은 모두 그래프 밖 전처리(`build_inputs`)로 뺐습니다. 그래프 안에는 Gather, MatMul, Softmax, ReduceMax 같은 표준 연산만 남습니다.
- 그래서 ONNX로 내보낼 수 있고, 점 개수는 동적 축(dynamic axis)으로 처리됩니다.

## 구성

```
pureptv3/
  model.py          PurePTv3 (백본), PurePTv3Segmentor (DefaultSegmentorV2와 같은 키)
  preprocess.py     build_inputs: 그래프 밖 인덱스 계산 (torch, CPU/GPU 모두 가능)
  serialization.py  Z-order / Hilbert 인코딩 (Pointcept 코드 그대로)
  checkpoint.py     Pointcept 체크포인트 로더 (module./backbone. 접두사 처리, 모양 검사), 사전 가중치 초기화
  export.py         ONNX 내보내기 (입력 이름, 동적 축)
train_toothseg.py   dental_ai ToothSegLoader 위에서 17-class 치아 semantic segmentation 학습
infer_toothseg.py   학습한 체크포인트로 검증 스캔 추론: 점수, confusion, 정답/예측/오류 그림
tests/
  test_native.py            torch만으로 도는 단위 테스트
  test_pointcept_parity.py  원본 Pointcept PT-v3m1과 같은 가중치·같은 입력으로 출력 비교
```

## 모듈 대응

| Pointcept | PurePTv3 | 원본과 같은가 |
|---|---|---|
| `spconv.SubMConv3d` (stem k=5, 블록마다 있는 CPE k=3) | `kmap`으로 이웃을 Gather한 뒤 Linear(MatMul) 한 번 | 같음 (spconv 2.3.8로 실측) |
| FlashAttention varlen | `[P, K]` 패치로 reshape한 뒤 `F.scaled_dot_product_attention` | 같음 (패딩 규칙 동일) |
| `torch_scatter.segment_csr(max)` (풀링) | 부모마다 자식 `[M, 8]`을 Gather한 뒤 `amax` | 같음 (비트 단위) |
| `Point.serialization`, 풀링 그룹핑 | 전처리 `build_inputs` | 같음 |
| Unpooling, LayerNorm, BN, GELU, Linear | 그대로 | 같음 |

## 사용법

```python
import torch
from pureptv3 import PurePTv3Segmentor, load_pointcept_checkpoint

# Pointcept config의 model 부분을 그대로 넘기면 됩니다 (type, criteria 키는 무시).
model = PurePTv3Segmentor.from_config(dict(
    type="DefaultSegmentorV2",
    num_classes=17,
    backbone_out_channels=64,
    backbone=dict(type="PT-v3m1", in_channels=6, order=("z", "z-trans", "hilbert", "hilbert-trans")),
)).eval()
load_pointcept_checkpoint(model, "exp/.../model/model_best.pth")

# grid_coord: GridSample로 만든 [N, 3] 정수 좌표 (샘플 안에서 중복이 없어야 함)
# feat:       [N, in_channels] 입력 특징 (학습 때와 같은 전처리)
inputs = model.prepare(grid_coord, offset=offset)  # 그래프 밖 전처리
logits = model(feat, inputs)                         # [N, num_classes]
```

백본만 쓸 때는 `PurePTv3(**backbone_cfg)`를 만들고 `load_pointcept_checkpoint`가 `backbone.` 접두사를 알아서 고릅니다.
학습(파인튜닝)도 됩니다. 아래 "학습" 절을 보세요.

### ONNX 내보내기

```python
from pureptv3.export import export_onnx, onnx_feed

names = export_onnx(model, feat, inputs, "pureptv3.onnx")      # 기본은 dynamo 경로 (onnxscript 필요)
# 추론 앱: 매번 전처리 후 feed
import onnxruntime as ort
sess = ort.InferenceSession("pureptv3.onnx")
(logits,) = sess.run(None, onnx_feed(feat, model.prepare(grid_coord), names))
```

입력 이름과 축 이름은 `export.dynamic_axes_for`에 정의되어 있습니다.
`n{s}`는 stage s의 점 개수, `l{s}`는 패딩된 패치 슬롯 수(= `p{s}` × K), `p{s}`는 패치 수입니다.
TensorRT optimization profile을 잡을 때 이 축들을 쓰면 됩니다.
TensorRT용으로 인덱스를 int32로 만들려면 `prepare(..., index_dtype=torch.int32)`를 쓰세요.
dynamo 경로는 인덱싱을 `GatherND`로, `dynamo=False`(TorchScript, deprecated) 경로는 `Gather`로 내보냅니다. 둘 다 onnxruntime에서 같은 결과를 냈습니다.

## 학습

### 역전파

gradient가 필요할 때만 쓰는 backward 두 가지를 [model.py](pureptv3/model.py)에 따로 두었습니다.
추론과 ONNX 내보내기는 no-grad라서 원래 연산을 그대로 씁니다.

- **conv (`_SubMConvFunction`)**: submanifold conv에서는 `kmap[i, k] == m`이면 `kmap[m, K-1-k] == i`입니다.
  그래서 입력 gradient는 같은 `kmap`에 커널을 뒤집어 gather하면 구해지고, scatter가 필요 없습니다.
  순방향 gather 결과 `[N, K·Cin]`은 저장하지 않고 backward에서 다시 만듭니다.
- **나머지 gather (`gather_rows`)**: 어텐션 패치, 풀링, 언풀링의 `x[idx]`입니다. backward를 fp32 `index_add_`로 계산합니다.

autograd가 기본으로 쓰는 `x[idx]` backward는 정렬 기반 `index_put_`입니다. 이 방식은 같은 인덱스가 반복되면 그 자리를 하나씩 차례로 누적합니다.
`kmap`의 빈 이웃 인덱스 `N`은 수백만 번 반복되기 때문에, 이것만으로 학습 step의 97%를 썼습니다.
RTX 5080, 기본 크기 모델, bf16 autocast, 4스캔 × 4만 점에서 비교한 결과는 다음과 같습니다.

| | step 시간 (전처리 + forward + backward) | 메모리 피크 |
|---|---|---|
| autograd 기본 | 7.0 s | 7.25 GB |
| 지금 | 0.25 s | 3.7 GB |

4스캔 × 7만 점에서는 0.34 s, 6.0 GB입니다.
`model.FAST_BACKWARD = False`로 바꾸면 기본 autograd로 돌아갑니다.
`test_fast_backward_*` 테스트는 float64 gradcheck와 모델 전체 gradient 214개가 기본 autograd와 같은지(상대 오차 6e-7)를 확인합니다.
backward는 atomic add를 쓰므로 실행마다 아주 작은 비결정성이 있습니다.

### dental_ai 데이터로 학습

`train_toothseg.py`는 `dental_ai.dataset.ToothSegLoader`로 data/ios 캐시를 읽습니다.
loader가 augmentation, 0.25 mm voxel downsample, feature, target을 만들고, PurePTv3 전처리는 batch마다 GPU에서 `model.prepare`가 만듭니다.
loader의 `grid`는 스캔마다 0부터 시작하고 voxel마다 점이 하나라서 그대로 넣을 수 있습니다.

| 항목 | 값 |
|---|---|
| target | 17-class (0 치은, 1..16 FDI 위치). 위치 모르는 치아(class 17)는 제외. segmentation3d semantic 단계와 같음 |
| loss | 가중 CE(빈도의 역제곱근, 평균 1, [0.2, 6]) + class soft dice |
| 데이터 | teeth3ds, 3d_iosseg, poseidon3d, cbct_ios_fusion_liu2023의 공식 train split. 환자 단위로 10%를 검증으로 분리 (1426 / 158 스캔) |
| 입력 | xyz, normal, jaw, colour (in_ch 10) |
| augmentation | ToothSegLoader 기본값: SO(3) 회전(매 샘플), mirror 0.5(라벨도 바꿈), gingiva trim 0.5, crop 뒤 원점 중심 정렬. 두 확률은 인자로 정한다. `--crop-p`(기본 0.5): 샘플을 읽을 때마다 이 확률로 bridge/quadrant/half(0.4/0.35/0.25) 중 하나, 그중 25%는 부채꼴(wedge)로 자른다. `--colour-p`(기본 0.5): 이 확률로 색을 쓴다. 실제 색이 없으면 in-house 실측 컬러맵으로 칠하고(`synth_colour_cpu`), 안 쓰면 0으로 넣어 색 없는 스캔도 배운다. `--synth-colour clinical`(기본): 칠한 색이 실제 스캔처럼 경계에서 서서히 바뀌고 덜 깨끗하다(`CLINICAL_COLOUR`, 라벨 있는 in-house 컬러 스캔 10개에 맞춤). 첫 학습(`pureptv3_semantic`)은 crop 0.35, 색 0.6, legacy 색이었다 |
| 모델 | Pointcept semseg-pt-v3m1-0-base 백본 (46.2 M), patch 1024 |
| 사전 가중치 | `--pretrained`로 Pointcept PT-v3m1 체크포인트에서 시작한다(`init_from_pretrained`). backbone은 전부 복사하고, 입력 채널이 다른 stem은 같은 feature 채널(normal, colour)만 가져오고 나머지(xyz, jaw)는 0에서 시작한다. `seg_head`는 새로 만든다. 첫 학습은 사전 가중치 없이 처음부터 학습했다. 후보와 라이선스는 [experiments/2026-11-10_pretrain_order](experiments/2026-11-10_pretrain_order/README.md)에 있다 |
| 직렬화 순서 | `--order`, 기본 `hilbert hilbert-trans z z-trans`. 학습 때는 stage마다 섞고, 평가·추론에서는 블록 i가 i % 4번째를 쓴다. 첫 학습은 z 먼저였다. 평가 순서를 바꿔도 점수 차이는 없었다(같은 실험 E1) |
| 최적화 | AdamW lr 0.002 (Pointcept 0.006 @ batch 12를 batch 4로 비례 축소), "block" 파라미터는 ×0.1, wd 0.05, OneCycle 모양 스케줄, clip 3.0, bf16 autocast |
| 검증 | grey(색 없음)와 colour(칠한 색) 두 경로의 mIoU, acc, TSA. 데이터셋별 mIoU도 기록. 두 경로 mIoU 평균이 가장 높을 때 `best.pt` 저장 |

```bash
# hdl-gpu 이미지, dental-ai 클론을 /app, data를 /app/data에 마운트한 상태에서
PYTHONPATH=/app/dental_ai/src python train_toothseg.py --out /app/runs/pureptv3_semantic
PYTHONPATH=/app/dental_ai/src python train_toothseg.py --out /app/runs/pureptv3_semantic --resume
# 사전 가중치 (AI/checkpoints를 /app/checkpoints에 마운트)
PYTHONPATH=/app/dental_ai/src python train_toothseg.py --out /app/runs/pureptv3_semantic_scannet     --pretrained /app/checkpoints/pointcept/scannet-semseg-pt-v3m1-0-base/model_best.pth
```

체크포인트의 `state_dict`는 Pointcept `DefaultSegmentorV2`와 키가 같습니다. 그래서 `load_pointcept_checkpoint`로 바로 읽힙니다.
`meta`에는 백본 설정, feature, voxel 크기가 들어 있습니다.

### 첫 학습 결과와 추론 (2026-11-09)

100 epoch, best.pt = epoch 85.
검증은 158 스캔이고, 아래는 `infer_toothseg.py` 결과입니다.
학습 중 검증과 같은 split, 같은 전처리를 쓰고, 점수도 학습 로그와 같습니다.

```bash
PYTHONPATH=/app/dental_ai/src python infer_toothseg.py --ckpt /app/runs/pureptv3_semantic/best.pt \
    --out /app/runs/pureptv3_semantic/infer_best
```

| 입력 | mIoU 17-class | mIoU 9-class (좌우 합침) | TSA | 위치는 맞고 좌우만 틀린 치아 점 | 17-class + 기하 좌우 규칙 |
|---|---|---|---|---|---|
| grey | 32.0 | 88.7 | 98.4 | 47.1% | 87.5 |
| colour | 33.8 | 91.0 | 99.6 | 47.4% | 89.9 |

모델은 치아와 치은, 악궁 위치(중절치 .. 제3대구치)는 배웠지만 사분면(좌우)은 배우지 못했습니다.
confusion을 보면 위치마다 정답 좌우와 상관없이 한쪽 사분면을 고정으로 고릅니다.
ToothSegLoader의 SO(3) 회전과 mirror(라벨도 바꿈) 아래에서는 좌우가 손잡이(chirality)로만 구별되기 때문입니다.
loader 주석에도 "17-class target은 noise"라고 적혀 있고, 그래서 loader는 9-class `arch_positions`를 따로 줍니다.

**기하 좌우 규칙**(`side_by_geometry`): 예측한 위치만 씁니다.
예측 위치와 normal로 교합면 frame을 만들고(loader의 `_view_frame`), 예측 중절치의 x 중앙값을 정중선으로 잡아 사분면을 정합니다.
정답은 쓰지 않습니다. 이 규칙을 쓰면 좌우 오류가 0.4%로 줄어듭니다.
남은 오류는 대부분 대구치가 한 칸 밀리는 것(1-7→1-6, 1-8→1-7)과, 치아가 많이 빠졌거나 crowding이 심한 스캔입니다.
이 규칙은 정중선이 보여야 하므로 **full-arch에서만** 맞습니다.
`--crop`으로 부분 스캔을 흉내 내서 colour 입력으로 쟀습니다.
3d_iosseg는 점이 성겨 crop이 적용되지 않으므로 뺐습니다.

| 입력 | 위치 mIoU9 | 좌우 오류 (모델 그대로) | 좌우 오류 (기하 규칙) | mIoU17 (기하 규칙) |
|---|---|---|---|---|
| full arch | 91.0 | 47% | 0.4% | 89.9 |
| half | 92.0 | 48% | 13.6% | 65.7 |
| quadrant | 82.9 | 48% | 25.1% | 50.9 |
| bridge (2~5 치아) | 81.6 | 49% | 37.2% | 42.0 |

부분 스캔에서는 위치는 맞히지만 사분면을 정할 방법이 아직 없습니다.

인하우스 스캔 분석(번호, 색, 치아 instance 품질, 개선 포인트)은 [experiments/2026-11-10_inhouse/README.md](experiments/2026-11-10_inhouse/README.md)에 있습니다.
DDH 추론 서버(`inference-server/toothseg.py`)는 2026-11-10부터 이 규칙을 점마다가 아니라 치아 instance마다 적용합니다.
TTA와 조각 정리도 함께 합니다(실험 E14~E16).

## 전처리 출력 (`build_inputs`)

`s` = stage, `j` = 직렬화 순서 슬롯. 블록 i는 슬롯 `i % len(order)`를 씁니다.

| 키 | 모양 | 의미 |
|---|---|---|
| `stem_kmap` | `[N0, 125]` | stem conv(5×5×5) 이웃 인덱스. 빈 칸은 `N0` |
| `s{s}_kmap` | `[Ns, 27]` | CPE conv(3×3×3) 이웃 인덱스. 빈 칸은 `Ns` |
| `s{s}_order{j}` | `[Ps*Ks]` | 패치 슬롯마다 들어갈 점 인덱스 |
| `s{s}_inverse{j}` | `[Ns]` | 각 점의 결과를 읽어올 패치 슬롯 |
| `s{s}_attn_bias` | `[Ps, 1, 1, Ks]` | 키 마스크(0 또는 -inf). K보다 작은 샘플에만 -inf가 들어감 |
| `s{s}_pool` (s≥1) | `[Ns, 8]` | 부모마다 자식 인덱스(stage s-1). 빈 칸은 첫 자식을 반복 |
| `s{s}_unpool` (s≥1) | `[N(s-1)]` | 자식마다 부모 인덱스(stage s) |

C++/C# 앱에서 추론하려면 이 표의 계산을 앱 쪽에 다시 구현해야 합니다.
알고리즘은 [preprocess.py](pureptv3/preprocess.py)에 단계별로 있습니다(정렬, 이진 탐색, 비트 시프트뿐입니다).

## 원본과 똑같이 맞추려고 일부러 남긴 동작

학습된 가중치와 결과를 맞추려고 Pointcept의 실제 동작을 그대로 따랐습니다. 코드만 보고 "고치면" 결과가 달라집니다.

1. **디코더 각 stage의 첫 블록 CPE는 skip 가지만 입력으로 받습니다.**
   Pointcept의 `SerializedUnpooling`은 `point.feat`(skip + upsampled)만 갱신하고 `sparse_conv_feat`은 갱신하지 않습니다.
   그래서 첫 블록의 spconv CPE는 `proj_skip(parent)`만 봅니다.
   `dec_cpe_from_skip=False`로 바꾸면 합친 특징을 쓰는데, 테스트에서 이렇게 하면 원본과 상대 오차가 0.42로 벌어졌습니다. 새로 학습할 때만 바꾸세요.
2. **디코더 어텐션은 같은 stage 인코더의 패치 구성을 그대로 씁니다.**
   Pointcept는 패딩 결과를 Point 객체에 캐시해서 디코더가 재사용하므로 `dec_patch_size`가 사실상 무시됩니다. 두 값이 다르면 경고를 냅니다.
3. **K보다 점이 적은 샘플은 FlashAttention처럼 처리합니다.**
   flash는 그런 샘플을 길이 n짜리 패치 하나로 계산합니다. 여기서는 K로 패딩하고 패딩 키를 마스크합니다(같은 계산).
   Pointcept의 non-flash 경로는 배치 안에 작은 샘플이 하나라도 있으면 모든 샘플의 K를 줄이므로, 그 경우에만 non-flash 원본과 다릅니다.
   샘플이 하나면 둘은 같습니다.

## 원본과 다른 점

- **직렬화 순서 셔플**: Pointcept는 `shuffle_orders=True`면 추론 때도 순서를 무작위로 섞습니다(그래서 원본은 실행마다 결과가 조금씩 다릅니다).
  `prepare()`는 학습 모드에서만 섞고, eval에서는 순서를 고정합니다.
  고정 순서와 무작위 섞기의 검증 점수는 같은 수준입니다(mIoU9 차이 ±0.4, [E1](experiments/2026-11-10_pretrain_order/README.md)).
- **어텐션 정밀도**: 원본 flash 경로는 bf16/fp16으로 계산합니다. PurePTv3는 입력 dtype(보통 fp32)으로 SDPA를 돌리므로 원본 학습 때보다 조금 더 정확합니다.
- **미지원**: `enable_rpe=True`, PDNorm(`pdnorm_bn`, `pdnorm_ln`), `max`/`min` 외의 풀링. 넣으면 에러가 납니다.
  `enable_flash`, `upcast_attention`, `upcast_softmax`는 받기만 하고 무시합니다.
- **입력 조건**: 샘플 안에서 `grid_coord`가 중복되면 안 됩니다(GridSample 필요). 중복이 있으면 `kernel_map`이 에러를 냅니다.
  원본 spconv도 이 경우 결과가 정의되지 않습니다.

## 검증 결과

`test_pointcept_parity.py`: 원본 `point_transformer_v3m1_base.py`를 그대로 불러와 같은 가중치(무작위로 흔든 값, BN 통계 포함)와 같은 입력으로 비교했습니다.
환경은 torch 2.11 CPU, spconv 2.3.8, fp32입니다.

| 테스트 | 조건 | 상대 오차 (최대 절대 오차 / 출력 최대값) |
|---|---|---|
| 배치 2개, 직렬화 4종, 셔플 켬 | 점 개수 stage별 10691 / 7889 / 3543 / 1046 / 265 | 1.1e-6 |
| 샘플 1개, patch 1024 (ScanNet 설정) | stage 3, 4는 K보다 작음 (마스크 경로) | 9.7e-7 |
| 디코더 CPE 특이 동작 | 그대로 따름 / 따르지 않음 | 9.4e-7 / 4.2e-1 |
| DDP 체크포인트 → `PurePTv3Segmentor` | `module.backbone.*`, `module.seg_head.*` | logits 최대 절대 오차 7.9e-6 |

`test_native.py`: dense `conv3d`와 gather conv 비교, Pointcept 패딩 코드와 인덱스 일치, 패치 규칙을 반복문으로 직접 쓴 어텐션과 비교, 풀링과 scatter max 비교,
입력 점 순서를 섞어도 결과가 같이 섞이기만 하는지, 배치 실행과 개별 실행이 같은지, int32 인덱스, 역전파,
그리고 ONNX 왕복(두 내보내기 경로 모두, 내보낼 때와 다른 점 개수)까지 확인합니다.
기본 크기 모델(32~512 채널)의 ONNX 그래프는 입력 40개, 약 185 MB이고, onnxruntime 결과가 PyTorch와 상대 오차 1.1e-6으로 맞습니다.

**아직 확인하지 않은 것**: GPU에서 원본 Pointcept와의 비교(GPU에서는 PurePTv3 학습만 돌려 봤습니다), FP16, TensorRT 엔진 빌드, Pointcept로 학습한 실제 체크포인트를 PurePTv3에서 돌려 검증셋 mIoU 비교.

### 테스트 실행

```bash
python tests/test_native.py                 # torch만 있으면 됨 (ONNX 테스트는 onnxruntime이 있을 때)
python tests/test_pointcept_parity.py       # spconv + Pointcept 소스 필요 (POINTCEPT_ROOT, 기본 ../Pointcept)
```

parity 테스트는 Pointcept를 설치하지 않고 소스 파일만 불러옵니다. torch_scatter, timm, 모델 레지스트리 등은 없으면 간단한 대체 구현으로 바꿉니다.
spconv의 CPU 커널은 멀티스레드에서 실행마다 값이 달라지는 경쟁 조건이 있어서 테스트는 torch를 1스레드로 고정합니다.
CPU spconv는 bias가 있는 conv를 지원하지 않아서, 원본 쪽 CPE conv의 bias만 conv 뒤에 더하도록 바꿔서 비교합니다(같은 계산).

## 메모리

Gather 기반 conv는 `[N, K·Cin]` 행렬을 만들기 때문에 spconv보다 메모리를 더 씁니다.
24k 점 기준 stage 0 CPE(64채널, fp32)가 약 170 MB이고, FP16이면 절반입니다.
추론에서는 이 행렬이 잠깐만 있다가 사라집니다. 학습에서도 저장하지 않고 backward에서 다시 만듭니다.
patch 1024 설정에서 점이 K보다 적은 stage도 K로 패딩하므로 어텐션 행렬은 `[1, H, 1024, 1024]`가 됩니다.

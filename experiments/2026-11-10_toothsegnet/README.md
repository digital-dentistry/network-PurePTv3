# 2026-11-10 자동 라벨을 ToothSegNet(segmentation3d)으로 바꾸기

사용자 요청: "ToothSegNet 사용해서 학습하고, 그걸로 자동라벨링 만들게". 사용자의 말을 더 옮기면 다음과 같다.
- "ToothSegNet으로 치아 instance 뽑아낼땐 치아 경계가 거의 모든경우에서 들어맞았음."
- "치아 번호맞추는건 별로 중요하지 않아."

그래서 **치아 경계(instance)를 먼저 보고, 번호는 그다음에 본다.**

스크립트는 `scripts/`, 출력은 `results/`에 있다. 추론 서버 쪽 코드는 `inference-server/toothsegnet.py`(어댑터)와 `package_toothsegnet.py`(패키지)다.

## 무엇을 비교했나

| 이름 | 치아 경계 | 번호 |
|---|---|---|
| P = `toothseg-pureptv3/1` | PurePTv3 17-class 정점 분류 → 다른 세션의 후처리 (2026-11-10_inhouse E16: TTA 8, 조각 정리, instance 먼저, 정중선 규칙으로 사분면) | 위치는 모델, 좌우는 정중선 규칙 |
| S0 = `toothsegnet-ios/1` | segmentation3d `runs/instance_ios/best.pt` (mask transformer, query 32, 250 µm, 공개 4개 데이터셋, epoch 290) → TTA 8 Hungarian 합침 → 5 mm² 조각 정리 | segmentation3d `runs/semantic/best.pt` (17-class, Teeth3DS만, 300 µm, 색 없음, epoch 117)를 치아마다 면적 가중 평균하고, 번호가 겹치지 않게 Hungarian |

S0의 semantic은 예전 학습이다. 번호를 위해 새로 학습한 `runs/semantic_ios`(공개 4개 데이터셋, 색 없음, 150 epoch)는 S1로 비교했다(아래 "S1").
사용자가 번호는 덜 중요하다고 했으므로 S0을 먼저 올린다.

## E1 번호까지 — `scripts/eval_models.py` → `results/eval_baseline.{log,json}`

추론 서버 어댑터를 그대로 불렀다(`registry.load`).
- **공개 데이터:** 학습 4개 데이터셋 patient_split(seed 0) 검증셋에서 3d_iosseg를 빼고 60 스캔을 골랐다.
  - 전악을 쓰고, `segment_crop_cpu`로 half, quadrant, bridge 크롭을 만들었다(같은 이름이면 같은 크롭).
  - 면이 없으므로 어댑터는 점 kNN 그래프로 조각을 정리한다.
- **인하우스:** 완료 라벨 12개. 이 평가에서는 면을 넘기지 않았다.

지표:
- tooth F1: 치아/치은 정점 F1.
- tooth FDI: 정답 치아마다 예측 번호의 다수결이 맞은 비율.
- ok75: 정답 치아와 같은 번호로 예측된 영역의 IoU가 0.75 이상인 비율. 영역과 번호가 모두 맞아야 한다.
- side wrong: 위치는 맞고 사분면만 틀린 비율.

| | P tooth FDI | S0 tooth FDI | P ok75 | S0 ok75 | P side wrong | S0 side wrong |
|---|---|---|---|---|---|---|
| 전악 (60, 치아 821) | 0.954 | 0.949 | 0.938 | 0.948 | 0.001 | 0.001 |
| half (60, 471) | 0.824 | **0.919** | 0.699 | **0.919** | 0.140 | 0.009 |
| quadrant (60, 331) | 0.725 | **0.888** | 0.613 | **0.888** | 0.211 | 0.021 |
| bridge (59, 196) | 0.597 | **0.867** | 0.531 | **0.867** | 0.357 | 0.056 |
| 인하우스 (12, 88) | 0.500 | 0.398 | 0.455 | 0.386 | 0.159 | 0.273 |

치아/치은 F1은 두 모델 모두 0.96~0.98로 같다.

- **부분 스캔:** S0이 크게 낫다. P의 오류는 대부분 좌우다. 정중선이 없으면 규칙이 사분면을 정하지 못한다. S0의 semantic은 SO(3)와 mirror 증강으로 학습해 좌우를 배웠다.
- **인하우스 번호:** 두 모델 모두 같은 5개 스캔(0002, 0004, 0005, 0008, 0012)에서 0점이다. 인하우스 사람 라벨의 번호가 밀리거나 거울상으로 보인다는 진단(2026-11-10_inhouse)과 맞는다. 그래서 인하우스 번호 점수로는 모델을 고르지 않는다.
  - 두 모델이 다른 곳은 0010(S0이 좌우를 반대로 냄)과 0011(부분 스캔)이다.
- **시간 (GPU, TTA 8, 다른 학습과 GPU를 같이 썼다):** S0 전악은 중앙값 1.5초(P 2.9초), 부분 크롭은 0.7~0.9초(P 1.4~1.7초).
  - S0 half의 평균 75초는 측정 중 컨테이너를 1시간 넘게 일시정지한 탓이다. 그 한 스캔이 4445초로 찍혔다. 중앙값은 0.90초다.

```bash
# D=C:/Users/wodyd/OneDrive/Desktop, E=이 폴더, G=export_gt.py 출력(인하우스 정답 npz)
docker run --rm --gpus all --shm-size 4g -v "$D/AI/data:/data:ro" -v "$D/DentalDataHub/models:/models:ro" \
  -v "$D/AI/segmentation3d/runs:/runs:ro" -v "$D/DentalDataHub/ios-models/PurePTv3:/pureptv3:ro" -v "$E:/exp" -v "$G:/gt:ro" \
  --entrypoint sh ddh-inference-server:dev -c "python package_toothsegnet.py --out /tmp/pkgs --name s0-sem-teeth3ds --version 1 \
    --instance /runs/instance_ios/best.pt --semantic /runs/semantic/best.pt >/dev/null && \
  python /exp/scripts/eval_models.py --packages /models/toothseg-pureptv3/1 /tmp/pkgs/s0-sem-teeth3ds/1 --n 60 --out /exp/results/eval_baseline.json"
```

## E2 치아 경계, E16과 같은 조건 — `scripts/server_adapter_eval_seg3d.py` → `results/server_adapter_eval_*.log`

사용자가 가장 중요하게 보는 것은 치아 경계다. 그래서 다른 세션이 PurePTv3 후처리를 고를 때 쓴 E16을 그대로 썼다(`2026-11-10_inhouse/scripts/server_adapter_eval.py`).
래퍼는 그 스크립트를 고치지 않고 읽어서 세 가지만 바꾼다.
- 어댑터를 바꾼다.
- 공개 크롭에 bridge를 더한다. 사용자가 bridge도 나쁘다고 했기 때문이다.
- PurePTv3 bridge만 잴 때는 인하우스를 건너뛴다.

E16의 방법은 다음과 같다.
- 번호는 무시한다.
- IoU 0.75로 1:1 매칭해 F1을 낸다.
- 인하우스는 mesh 면을 넘기고 면적 가중으로 잰다. 공개는 면 없이 kNN(8) 그래프로 정점 수를 잰다.
- DDH 기준에서는 같은 FDI 하나를 치아 하나로 본다.

| F75 (병합/분할) | P (E16, 다른 세션) | **S0** |
|---|---|---|
| 인하우스 12 | 0.962 (3/2) | **0.980 (0/0)** |
| 인하우스 12 + 색 | 0.950 (3/2) | **0.973 (0/0)** |
| 공개 전악 40 | 0.969 (4/5) | **0.991 (0/0)** |
| 공개 quadrant 40 | 0.802 (18/5) | **0.993 (1/0)** |
| 공개 half 40 | 0.827 (24/4) | **0.995 (1/0)** |
| 공개 bridge 40 | 0.939 (4/4) | **1.000 (0/0)** |

P bridge는 이번에 같은 래퍼로 쟀다(`server_adapter_eval_pureptv3_bridge.log`).

- **경계:** S0은 모든 조건에서 P보다 낫다. 부분 스캔에서 차이가 가장 크다(0.80 → 0.99).
  - P가 부분 크롭에서 낮은 이유는 번호다. 떨어진 이웃 치아가 같은 번호를 받으면 DDH 기준에서 한 치아로 합쳐진다.
  - S0은 instance가 번호와 따로 나온다. 번호도 Hungarian으로 겹치지 않게 붙이므로 이런 병합이 없다. 그래서 DDH와 CC 점수가 같다.
- **인하우스에서 남은 오류:**
  - 0007(0.83, 색이 있으면 0.77): 치아 하나가 IoU 0.75를 넘지 못했다.
  - 0012(0.92, 색이 있으면 1.00).
  - 0003은 색이 있을 때만 0.91이다.
- **경고:** 좌우 확률이 0.7보다 낮은 치아가 있으면 "좌우(사분면)가 불확실한 치아" 경고가 나온다. 인하우스 12개 중 8개에서 나왔다. 대부분 한쪽 사분면 스캔이다. 번호를 확인하라는 뜻이고, 경계와는 상관없다.
- **시간 (GPU, TTA 8):** 인하우스 평균 1.6초, 최대 3.4초. 공개 전악 평균 1.5초다. P는 2.2초와 3.5초였다.

```bash
# E16 마운트(2026-11-10_inhouse/README.md) + 그 폴더(/inhouse_exp), 이 폴더(/exp), 패키지(/pkgs)
docker run --rm --gpus all --shm-size 2g --entrypoint sh -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp \
  -v $R/dental-ai/dental_ai/src:/src:ro -v $R/ios-models/PurePTv3:/pureptv3:ro -v $R/inference-server:/inference:ro \
  -v $R/models:/models:ro -v $D/AI/segmentation3d:/seg3d:ro -v $D/AI/data:/data:ro \
  -v $R/ios-models/PurePTv3/experiments/2026-11-10_inhouse:/inhouse_exp:ro -v $E:/exp -v $G:/gt:ro -v $R/models:/pkgs:ro \
  hdl-gpu:latest -c "python /exp/scripts/server_adapter_eval_seg3d.py --adapter toothsegnet --package /pkgs/toothsegnet-ios/1 \
    --n 40 --modes full quadrant half bridge; python /exp/scripts/server_adapter_eval_seg3d.py --adapter pureptv3 --n 40 --modes bridge --no-inhouse"
```

`$G`는 인하우스 정답 npz다. 환자 메시이므로 OneDrive와 저장소 밖에 둔다.
없으면 `2026-11-10_inhouse/scripts/export_gt.py`로 다시 만든다. dashboard-server 컨테이너 안에서 `python - <asset_id> <jaw>`로 돌린다.
asset id는 다음과 같다.
INHOUSE0001=01f49b17593f8eab 0002=810645145c00b48e 0003=f90a3cb1b3bd5739 0004=375901438def3b3c 0005=c5cc9c3acce30869
0006=e3512dff2c9c5777 0007=2833a1f6e227aff3 0008=38c40e8fdfc42af4 0009=ddd32762f63091e3 0010=0620d36dfc2a4982
0011=4854b87a583fdca8 0012=7629a758363b13a9.

## E3 배포 후 대시보드 API로 — `results/dashboard_fetch.log`, `results/dashboard_e2e_eval.log`

절차는 다음과 같다.
1. `models/toothsegnet-ios/1`을 만든다(아래 "배포").
2. `.env`에 `DDH_AUTOLABEL_MODEL=toothsegnet-ios/1`을 적는다.
3. dev inference-server만 다시 띄운다.
4. E16b와 같은 스크립트로 인하우스 12개를 대시보드 API(`POST /api/assets/{id}/auto-label`, 읽기만 한다)로 받아 채점한다.
   - 받기: `2026-11-10_inhouse/scripts/dashboard_fetch.py`.
   - 채점: `dashboard_e2e_eval.py`.
   - 받은 결과 npz는 세션 scratch에 두었다(환자 데이터).

| | F50 | F75 | mIoU | 병합 | 분할 | 치아/잇몸 F1 | 응답 평균 / 최대 |
|---|---|---|---|---|---|---|---|
| P 배포 (E16b) | 0.987 | 0.950 | 0.924 | 3 | 2 | 0.959 | 5.3 s / 9.8 s |
| **S0 배포** | **0.986** | **0.973** | **0.949** | **0** | **0** | **0.964** | **4.5 s / 7.8 s** |

- 12개 모두 `info.graph = "mesh"`이고 TTA는 8이다.
- 대시보드는 색을 보내므로 기대값은 E2의 "인하우스 + 색"(0.973)이고, 그대로 나왔다.

## 배포 (2026-11-10)

```bash
docker run --rm --entrypoint python -v "$D/AI/segmentation3d/runs:/in:ro" -v "$R/models:/out" -v "$E/results:/m:ro" \
  ddh-inference-server:dev package_toothsegnet.py --out /out --name toothsegnet-ios --version 1 \
  --instance /in/instance_ios/best.pt --semantic /in/semantic/best.pt --metrics /m/metrics_toothsegnet-ios-1.json \
  --source instance_run=segmentation3d/runs/instance_ios --source semantic_run=segmentation3d/runs/semantic \
  --source segmentation3d_git=none --source evaluation=ios-models/PurePTv3/experiments/2026-11-10_toothsegnet \
  --description "..." --limitation "..." --limitation "..."
# .env: DDH_AUTOLABEL_MODEL=toothsegnet-ios/1
docker compose --profile autolabel up -d --no-deps inference-server
```

되돌리기: `.env`를 `DDH_AUTOLABEL_MODEL=toothseg-pureptv3/1`로 바꾸고 같은 명령으로 다시 띄운다.

## S1 번호용 semantic 새 학습 — 올리지 않는다

`segmentation3d/runs/semantic_ios`는 150 epoch를 마쳤다.
- 설정: `--data dental_ai`, 공개 4개 데이터셋, 250 µm, crop 0.5, trim 0.5, 색 없음(in_ch 7).
- 결과: 검증 mIoU17이 가장 좋을 때 29.59다(epoch 145, ce 0.97). S0의 semantic(`runs/semantic`)은 83.25였다.
- **좌우를 배우지 못했다.** PurePTv3 첫 학습(mIoU17 34, mIoU9 91)과 같은 모양이다.
  - E4에서 S1은 어느 데이터셋에서나 정답 치아의 약 절반이 사분면만 틀린다. 동전 던지기와 같다.
- 그래서 `toothsegnet-ios/2`는 만들지 않았다. S0이 그대로 배포된다.

### E4 데이터셋마다 좌우 — `scripts/side_by_dataset.py` → `results/side_by_dataset.{log,json}`

**가설:** 어떤 데이터셋의 메시가 거울상(det = -1로 변환)이면, 4개를 섞어 학습할 때 좌우 라벨이 서로 모순된다.

**방법:** 좌우를 배운 S0으로 데이터셋마다 patient_split 검증 전악을 잰다.

| 데이터셋 (스캔) | S0 tooth FDI | S0 side wrong | S1 tooth FDI | S1 side wrong |
|---|---|---|---|---|
| 3d_iosseg (12) | 0.986 | 0.000 | 0.493 | 0.487 |
| cbct_ios_fusion_liu2023 (6) | 0.978 | 0.000 | 0.476 | 0.501 |
| poseidon3d (15) | 0.869 | 0.021 | 0.408 | 0.469 |
| teeth3ds (20) | 0.917 | 0.004 | 0.598 | 0.333 |

**결론: 가설은 틀렸다.** 4개 데이터셋 모두 S0에서 좌우가 맞는다. 캐시의 메시와 라벨은 같은 좌우 규약이다.
S1은 데이터셋과 상관없이 좌우를 모른다.

### E5 dental_ai 증강이 좌우 라벨을 지키는지 — `scripts/mirror_check.py` → `results/mirror_check.{log,json}`

S0으로 세 가지를 쟀다. 데이터셋마다 검증 전악 8개를 썼다.
- 원본.
- `mirror_cpu`: x → -x, class 1..8 ↔ 9..16.
- `train_recipe_cpu`: S1과 같은 crop 0.5, trim 0.5. seed는 3개다.

| side wrong | 원본 | mirror | recipe |
|---|---|---|---|
| 3d_iosseg | 0.000 | 0.000 | 0.006 |
| cbct_ios_fusion_liu2023 | 0.000 | 0.000 | 0.111 |
| poseidon3d | 0.018 | 0.008 | 0.132 |
| teeth3ds | 0.000 | 0.000 | 0.039 |

**결론:** 증강은 좌우 라벨을 지킨다. recipe에서 오류가 조금 오르는 것은 부분 크롭에 정중선이 없어서다.
- 어느 평면으로 뒤집든 반사끼리는 회전 하나 차이다. 그래서 SO(3) 증강 위에서는 x축 반사에 사분면을 바꾸는 것이 맞다.
- `ToothSegDataset.__getitem__`도 증강한 같은 샘플에서 좌표, 법선, class를 함께 뽑는다. 좌우가 어긋날 곳이 없다.

### E6 예전 S0 레시피에 로더만 dental_ai — `segmentation3d/runs/semantic_dai_teeth3ds_c015` (진행 중)

**관찰:** 예전 `runs/semantic`도 epoch 40~50에는 ce 0.96~1.02, mIoU 24~26에 머물렀다. S1이 끝날 때와 같은 지점이다. 그 뒤 epoch 67 무렵 좌우를 배우면서 점수가 뛰었다.
그래서 S1은 그 점프에 닿지 못한 것으로 보인다. 원인 후보는 두 가지다.
1. dental_ai 학습 경로(로더, feature, target).
2. 더 어려운 학습 조건(4개 데이터셋, crop 0.5, trim 0.5, 250 µm).

**방법:** `runs/semantic`과 같은 설정으로, 로더만 `--data dental_ai`로 바꿨다.
- 설정: teeth3ds만, 300 µm, max 60000점, crop 0.15, trim 0, in_ch 7, 120 epoch.
- 학습/검증은 1080/120 스캔으로 같다.

**읽는 법:**
- 이것도 좌우를 배우면(mIoU17이 80대) 로더는 문제가 아니다. S1을 바꿀 곳은 학습 조건이다.
- 배우지 못하면 dental_ai 학습 경로에 원인이 있다. PurePTv3 첫 학습도 같은 로더를 썼다.

```bash
docker run -d --gpus all --shm-size 16g --name seg3d-sem-ablation -e PYTHONPATH=/app/segmentation3d -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$D/DentalDataHub/dental-ai:/app:ro" -v "$D/AI/data:/app/data:ro" -v "$D/AI/segmentation3d:/app/segmentation3d" \
  -w /app/segmentation3d --entrypoint python hdl-gpu:latest training/train.py --stage semantic --data dental_ai --ios teeth3ds \
  --res 0.3 --max-points 60000 --in-ch 7 --crop-p 0.15 --trim-p 0.0 --epochs 120 --out runs/semantic_dai_teeth3ds_c015
```

결과: **로더는 문제가 아니다. dental_ai 로더로도 좌우를 배웠다.**

| epoch | 9 | 19 | 29 | 39 | 49 | 59 | 69 | 73 |
|---|---|---|---|---|---|---|---|---|
| train ce | 1.42 | 1.24 | 1.10 | 1.01 | 0.94 | 0.90 | **0.43** | 0.37 |
| 검증 mIoU17 | 20.6 | 24.7 | 26.1 | 22.2 | 28.2 | 28.5 | **72.2** | 72.9 |

- 예전 `runs/semantic`과 같은 모양이다. 40~60 epoch 동안 ce 0.9~1.0에 머물다가, epoch 60~69 사이에 좌우를 배우면서 뛰었다.
- 그래서 S1이 실패한 원인은 학습 조건이다(4개 데이터셋, crop 0.5, trim 0.5, 250 µm). 이 조건에서는 150 epoch 안에 좌우를 배우는 점프가 오지 않았다.
  - 부분 크롭에는 정중선이 없다. 좌우를 가를 단서가 치아 하나하나의 좌우 비대칭(chirality)뿐이라서 신호가 약하다고 본다.
  - 조건 중 무엇이 결정적인지는 따로 재지 않았다.
- PurePTv3 첫 학습(crop 0.35, 100 epoch)도 이 점프 전에 끝났을 수 있다.
- epoch 73까지 돌았을 때 13:22 WSL 자동 업데이트로 Docker 엔진이 멈췄다(아래 "사고"). `--resume`으로 이어서 120 epoch까지 돈다.

마지막 결과 (120 epoch): 검증 mIoU17 최고 **82.53** (예전 `runs/semantic` 83.25), acc 95.1, TSA 97.9. dental_ai 로더로 예전 결과를 그대로 재현한다.

## 사고: WSL 업데이트로 Docker 엔진 정지 (2026-11-10 13:22 KST)

- **원인:** WSL이 3.0.5.0으로 자동 업데이트되면서 WSL VM이 내려갔다. 그때 `wsl --version`이 `Wsl/CallMsi/Install/ERROR_NOT_FOUND`로 실패했다. 그래서 Docker Desktop의 자동 복구도 실패했고, 엔진이 멈춘 채 "Quit / Restart" 대화상자를 띄웠다.
  - 근거: `%LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log`.
- **영향:** 모든 컨테이너가 멈췄다(dev 스택, thinkflow, 학습).
- **복구:**
  - 1~2분 뒤 `wsl --version`이 정상인 것을 확인하고 `docker desktop restart`를 했다.
  - dev 스택은 재시작 정책으로 돌아왔다. 모두 healthy이고 inference-server는 `toothsegnet-ios/1`을 다시 읽었다.
  - thinkflow 두 컨테이너는 재시작 정책이 없어서 `docker start`로 켰다.
  - 볼륨과 컨테이너는 지우지 않았다.

## 남은 것

- 번호를 더 좋게 하려면 4개 데이터셋으로 semantic을 다시 학습한다.
  - crop/trim을 0.15/0으로 낮추거나 epoch를 늘린다.
  - 사용자가 번호는 덜 중요하다고 했으므로 지금은 하지 않는다. 경계(instance)는 같은 모델이므로 번호용 semantic을 바꿔도 E2·E3은 그대로다.
- **segmentation3d에 git이 없다.** 패키지에는 코드 지문(`code.segmentation3d`, 어댑터가 쓰는 `.py`의 sha256)만 남는다. 지우거나 덮어쓰면 되살릴 수 없다.
- 크롭 축: E17(2026-11-10_inhouse)에 따르면 wedge 크롭은 xy에서 자른다. 그런데 3d_iosseg와 poseidon3d 일부는 교합 축이 y다.
  - instance_ios도 dental_ai 로더로 학습했으므로 같은 크롭을 봤다.
  - 이 실험의 공개 크롭도 같은 코드로 만들었다. 그래서 공개 크롭 점수만으로는 이 영향을 알 수 없다.
  - 인하우스 실제 부분 스캔(10개)에서는 0.98이다.

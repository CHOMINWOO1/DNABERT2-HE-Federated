# DNABERT-2 연합학습·CKKS 본 실험 프로토콜

고정일: 2026-08-18  
상태: 결과 확인 전 분석 규칙 고정

## 1. 연구 범위

### Main benchmark

- GUE_v2 `EMP/H3K4me3`
- 공식 전체 split: train 29,439 / dev 3,680 / test 3,680
- 이 test는 이전 pilot에서 이미 조회되었으므로 완전히 untouched한 confirmatory test로 표현하지 않는다.
- 본 결과는 `exploratory pilot 이후 설정을 고정한 full-data main benchmark`로 보고한다.

### Independent confirmatory replication

- GUE_v2 `EMP/H3K36me3`
- 공식 전체 split: train 27,904 / dev 3,488 / test 3,488
- 모델 성능을 보기 전에 task와 설정을 고정한다.
- 사전 감사 결과 exact duplicate 및 split 간 exact sequence overlap은 모두 0건이다.
- SHA-256:
  - train: `645B973DFE3DBF73CA3ACD274CA4490708366BCC2E4D37DFBEBC0D85E954AAEB`
  - dev: `BD95DED07A1DC6A7B9BCB4E946ED845ED5181918301B3509A307EED5DFADB868`
  - test: `FF0AFB0DB1E45AC845D4B26C388530104CA4D2F1F4284BA78C1E9A160FFC87BF`

## 2. Main benchmark 실험군

- `Local-only LoRA`: A/B/C 기관별 독립 모델
- `Centralized LoRA`: pooled plaintext data reference
- `Centralized Full-FT`: DNABERT-2 전체 encoder + 동일 mean-pooling classifier 성능 ceiling
- `Plain FedAvg LoRA`
- `FedProx LoRA`, μ=0.01
- `CKKS HE-FedAvg LoRA`

Full-FT federated/HE는 본 실험에서 제외한다. 실제 model audit 기준 전체 trainable state는 LoRA보다 약 393배 크며, 측정된 LoRA CKKS payload를 선형 환산하면 약 6.3 GiB/client/round 및 18.9 GiB/3 clients/round이다. 이 값은 실측이 아니라 분석적 추정치로 표시한다.

## 3. 데이터 분할과 반복

- Paired seeds: 42, 43, 44, 45, 46
- Main scenarios: IID 및 controlled label-skew non-IID
- Primary scenario: non-IID
- non-IID allocation weights:
  - label 0 → A/B/C = 0.15/0.35/0.50
  - label 1 → A/B/C = 0.60/0.30/0.10
- 공식 train/dev/test 경계는 유지한다.
- 기관은 실제 cohort가 아니라 synthetic cross-silo shard이다.

## 4. 학습 budget

- LoRA: fused attention `Wqkv` only, rank 8, alpha 16, dropout 0.05
- Max length 128
- LoRA effective batch 16
- Central/Local: 5 epochs
- Federated: 5 rounds × 1 local epoch
- 모든 핵심 방법은 최대 5 complete data passes
- AdamW, weight decay 0.01, gradient clipping 1.0
- LoRA LR: 2e-4
- Full-FT LR: 2e-5
- Full-FT: BF16 physical/effective batch 16, gradient accumulation 1
- Custom DNABERT-2 backbone의 gradient checkpointing은 실제 smoke에서 지원되지 않아 사용하지 않는다. Batch 16 smoke의 peak allocated VRAM은 약 2.97 GB로 8 GB GPU에서 안전함을 확인했다.
- Full-FT와 LoRA는 같은 pretrained checkpoint, mean-pooling head 구조, 복사된 동일 classifier 초기 텐서를 사용한다.

Primary matched-compute 결과는 epoch/round 5 checkpoint이다. Round별 pooled-dev curve를 저장한다. Plain FedAvg와 HE-FedAvg의 deployment-style secondary checkpoint를 제시할 경우 Plain FedAvg dev AUPRC가 선택한 같은 round를 두 방법에 공동 적용한다.

## 5. Replication

H3K36me3에서는 H3K4me3에서 고정된 hyperparameter를 변경하지 않고 다음만 실행한다.

- non-IID
- Centralized LoRA
- Plain FedAvg LoRA
- CKKS HE-FedAvg LoRA
- seeds 42–46
- 5 passes

## 6. 평가 지표

Primary endpoint:

- non-IID pooled-test AUPRC의 paired difference
- `Δ = AUPRC_HE − AUPRC_Plain`

Secondary utility:

- AUROC, MCC, F1, accuracy
- Brier score, 15-bin ECE
- 각 기관 A/B/C, site macro mean, weighted mean, worst-site, best−worst gap
- AUPRC는 기관 prevalence와 함께 보고한다.
- non-IID Local pooled 값은 site-matched personalized 결과이며 single global model과 직접 비교하지 않는다.

System/crypto:

- train/eval/encrypt/server-add/decrypt wall time
- peak allocated/reserved GPU memory
- bytes/client/round, total bytes/round, ciphertext expansion
- plaintext aggregate 대비 MAE, max absolute error, relative L2, cosine similarity

## 7. HE 비열등성 규칙

- Primary margin: absolute AUPRC −0.005
- Sensitivity margin: −0.002
- 분석 단위: paired seed
- Primary 판정: paired one-sided 95% lower confidence bound가 −0.005보다 크면 비열등성 기준 충족
- 보조 분석: two-sided 95% paired t interval, individual seed points/range, per-example prediction 기반 hierarchical bootstrap
- seeds가 5개뿐이므로 강한 일반적 비열등성 확정이 아니라 제한된 근거로 해석한다.

## 8. 보안 주장 범위

- CKKS는 LoRA + classifier delta의 가중합에만 적용한다.
- honest-but-curious server와 non-colluding logical Key Authority를 가정한다.
- 현재 구현의 server/KA는 같은 프로세스 안에서 논리적으로만 분리된다.
- threshold decryption, malicious client, poisoning, DP, membership inference, 실제 network security는 해결하지 않는다.
- 권장 표현: `confidential aggregation prototype under the stated threat model`.

## 9. 해석상 제한

- label-skew simulation은 실제 기관의 covariate/domain shift가 아니다.
- LoRA factor-wise FedAvg에서 mean(A)·mean(B)는 mean(BA)와 동일하지 않다.
- Central AdamW optimizer state는 epoch 간 유지되며, local optimizer는 각 FL round에서 재초기화된다.
- 네트워크 latency는 포함하지 않는다.
- sample count와 client weight는 공개된다고 가정한다.

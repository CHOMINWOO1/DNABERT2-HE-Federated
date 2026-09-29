# DNABERT-2 Full-FT 연합학습·CKKS 추가 실험 프로토콜

고정일: 2026-08-18  
상태: 추가 실험 결과 확인 전 고정

## 1. 목적

기존 LoRA 기반 연합학습 결과에 다음 두 비교군을 추가한다.

- `Plain Full-FT FedAvg`: DNABERT-2 전체 trainable state의 평문 가중 평균
- `CKKS HE Full-FT FedAvg`: 동일 전체-state delta의 CKKS 암호화 가중합

이 추가 실험은 “LoRA를 사용할 때만 HE utility가 보존되는가?”와 “전체 파라미터 집계에서도 Plain/HE 차이가 사전 한도 안에 있는가?”를 평가한다.

## 2. 데이터와 반복

- GUE_v2 `EMP/H3K4me3` 공식 전체 split
  - train 29,439 / dev 3,680 / test 3,680
- 기관 A/B/C는 실제 기관이 아닌 synthetic cross-silo shard
- 시나리오: IID 및 controlled label-skew non-IID
- paired seeds: 42, 43, 44, 45, 46
- 각 방법: 5 global rounds × 1 local epoch
- 공식 train/dev/test 경계와 기존 기관 partition 알고리즘을 그대로 사용

H3K4me3 test는 기존 pilot과 main study에서 이미 조회되었으므로, 본 추가 실험을 untouched confirmatory test로 표현하지 않는다. H3K36me3 confirmatory 결과는 LoRA 경로의 독립 재현으로 유지하며 Full-FT federated arm을 추가하지 않는다.

## 3. 모델과 공정성 통제

- pretrained checkpoint, tokenizer, max length 128, mean-pooling 2-class head는 기존 본 실험과 동일
- 전체 trainable parameters: 116,479,490
- BF16, effective batch 16, learning rate 2e-5, AdamW, weight decay 0.01, gradient clipping 1.0
- Central Full-FT와 동일한 pretrained backbone 및 복사된 classifier 초기 tensor 사용
- 각 round에서 모든 기관이 동일 global full state로 시작
- client optimizer는 기존 FL 구현과 동일하게 round마다 재초기화
- FedAvg는 sample count 가중치 `n_i / N` 사용
- Plain/HE는 seed, partition, client order, local training budget, state manifest를 동일하게 유지하고 집계 방식만 다르게 한다.

## 4. CKKS 전체-state 집계

- TenSEAL CKKS
- polynomial modulus degree 8192
- coefficient modulus bits `[60, 40, 60]`
- scale `2^40`
- 4,096 slots/ciphertext
- 전체 full-state delta를 deterministic name/shape 순서로 실제 암호화한다.
- 메모리 폭증을 막기 위해 ciphertext chunk를 생성→직렬화 크기 계측→서버 덧셈→복호화한 뒤 즉시 해제하는 streaming 집계를 사용한다.
- streaming은 암호 연산을 생략하거나 일부 파라미터만 표본화하지 않는다.
- 실제 WAN 전송은 수행하지 않으며 serialized upload bytes를 logical traffic으로 보고한다.

## 5. 분석 규칙

Primary contrast:

- non-IID pooled test AUPRC의 paired difference
- `Δ = AUPRC_HE-FullFT − AUPRC_Plain-FullFT`

비열등성 규칙:

- absolute AUPRC margin `−0.005`
- 5개 complete paired seeds
- paired difference의 one-sided 95% t lower confidence bound가 `−0.005`보다 크면 “사전 정의한 추가 실험 기준 충족”으로 표현

Secondary:

- IID paired difference
- AUROC, MCC, F1, accuracy, Brier, ECE
- site macro 및 worst-site AUROC/AUPRC
- Central Full-FT 대비 federated utility gap
- round별 CKKS aggregate MAE, max absolute error, relative L2, cosine
- full-state plaintext bytes, ciphertext upload/client/round, 3-client total/round, expansion ratio
- client train/encrypt/server-add/decrypt 시간, sequential 및 idealized-parallel time, peak VRAM, peak system RAM

## 6. 중단·무결성 기준

- smoke test에서 GPU OOM, state manifest mismatch, non-finite loss/update, 복호화 길이 불일치가 발생하면 본 실행을 시작하지 않는다.
- CKKS aggregate의 max absolute error가 `1e-5`를 넘거나 cosine이 `0.99999` 미만이면 수치/packing 오류로 간주하고 원인 수정 전 결과를 사용하지 않는다.
- method별 `DONE.json`과 artifact SHA-256가 유효한 실행만 집계한다.
- 완료된 5 paired seeds 전에 비열등성 판정을 내리지 않는다.

## 7. 보안 및 해석 한계

- 전체 모델 학습·추론 자체가 암호화된 것이 아니라, client full-state delta의 집계 구간만 CKKS로 보호한다.
- honest-but-curious server와 non-colluding logical Key Authority를 가정한다.
- 현재 KA/server 분리는 동일 Python process 안의 논리적 역할 분리이며 threshold HE 또는 physical key isolation이 아니다.
- malicious client, poisoning, membership inference, traffic metadata, 실제 TLS/WAN은 보호·측정 범위 밖이다.
- 본 추가 실험은 결과 확인 후 요청된 protocol amendment이므로 최초 main study의 사전등록 결과와 분리해 보고한다.

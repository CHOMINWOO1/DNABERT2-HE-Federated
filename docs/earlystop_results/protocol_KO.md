# H3K4me3 non-IID LoRA-FL 수렴 민감도 프로토콜 v3

## 지위와 목적

- 기존 5-round 결과는 동일 5 data-pass의 주 분석으로 유지한다.
- 본 실험은 5-round Central–FedAvg 격차가 제한된 통신 round 때문인지 확인하는 **보강 수렴 민감도 분석**이다.
- H3K4me3 controlled non-IID에서 Plain FedAvg LoRA의 DEV 기반 checkpoint를 먼저 선택하고, 같은 seed의 CKKS HE-FedAvg LoRA를 동일한 선택 round까지 학습한다.

## 고정 범위

- 데이터: `GUE_v2/EMP/H3K4me3` 공식 train/dev/test 파일
- 시나리오: controlled non-IID 3-silo
  - class 0: A/B/C = 15/35/50%
  - class 1: A/B/C = 60/30/10%
- seeds: 42, 43, 44, 45, 46
- 방법: Plain FedAvg LoRA, CKKS HE-FedAvg LoRA
- DNABERT-2: 로컬 고정 snapshot revision `7bce263b15377fc15361f52cfab88f8b586abda0`
- LoRA: fused Wqkv, rank 8, alpha 16, dropout 0.05 + classifier head
- optimizer: client마다/round마다 AdamW 재생성, LR 2e-4, weight decay 0.01, gradient clip 1.0
- batch: train 16, eval 32
- local training: 모든 A/B/C client 참여, round당 local epoch 1
- max token length: 128
- deterministic PyTorch/CUDA 학습; CKKS 암호화 randomness는 고정하지 않는다.

## Plain checkpoint 선택 규칙

- 모니터: **pooled DEV AUPRC**
- 최소 round: 10
- 최대 round: 40
- patience: 5
- 최소 의미 개선량: absolute AUPRC 0.001
- 의미 개선은 이전 patience anchor 대비 누적 0.001 이상일 때 성립한다.
- 5개 연속 round에서 의미 개선이 없으면 중단한다.
- 반환 checkpoint는 관찰된 pooled DEV AUPRC가 가장 높은 round이다.
- 동률이면 더 이른 round를 선택한다.
- 각 seed는 독립적으로 round를 선택한다.

## HE 대응 규칙

- HE는 자체 DEV 결과로 round를 선택하지 않는다.
- 동일 seed의 Plain이 선택한 round 수를 그대로 사용한다.
- 초기 LoRA/head state, partition, client ordering, local optimizer seed와 학습 예산을 Plain과 대응시킨다.
- Plain/HE 모두 TEST는 checkpoint 선택이 완료된 뒤 정확히 1회 평가한다.

## 중단·재개

- round가 끝날 때 global trainable state, Plain best state, DEV 원장 및 CKKS secret/public context를 원자적으로 checkpoint한다.
- 재개 시 runner/config/data/partition/model snapshot hash가 다르면 실패한다.
- CKKS 재개는 기존 secret context를 복원하여 한 method 내 key를 유지한다.

## 사전 고정 분석

- 각 seed에서 최종 pooled TEST AUPRC를 보고한다.
- Plain 및 HE의 5-seed mean, sample SD, two-sided 95% t-CI를 보고한다.
- HE−Plain paired delta와 one-sided 95% LCB를 계산한다.
- 비열등성 margin은 기존과 동일한 absolute AUPRC −0.005이다.
- Central LoRA와의 격차는 paired seed 기술통계로 보고하되, 본 추가실험은 더 많은 data pass를 사용하므로 기존 5-pass Central과의 인과적·동일예산 우월성 검정으로 해석하지 않는다.
- 기존 5-round 및 20-round 결과를 함께 제시한다.

## 금지 해석

- early-stopped 결과를 기존 5-round 주 분석으로 소급 교체하지 않는다.
- TEST를 early stopping 또는 hyperparameter 선택에 사용하지 않는다.
- HE 비열등성을 Central 비열등성으로 표현하지 않는다.
- synthetic non-IID 결과를 실제 기관 분포의 일반적 결과로 표현하지 않는다.

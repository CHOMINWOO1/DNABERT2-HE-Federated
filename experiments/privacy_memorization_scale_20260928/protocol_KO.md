# 최종 유전체 MLM의 암기·조건부 추출 확장 실험: 실행 전 설계

2026-09-28. 기존 긍정/부정 파일럿을 본 후 설계한 확장 연구다. 이번 확장 실행의 결과를 보기 전에 설정과 본 문서를 해시 고정한다. 외부 공개 사전등록을 했다고 부르지 않는다. 결과를 미리 정하지 않으며 실패한 seed도 포함한다.

## 연구 범위

공개 DNABERT-2-117M의 일반 MLM FullFT 최종 가중치와 공개 tokenizer를 이용해 학습 서열이 추출되는지 평가한다. gradient, 기관 업데이트, 연합학습 및 HE는 사용하지 않는다. 하나의 모델 계열에서 수행하는 통제된 합성 canary 연구이며, 실제 환자 유전체·민감 변이·여러 모델 계열 일반화의 증거와 구분한다.

검증할 명제는 특정 학습 노출과 문맥 조건에서 모델 출력이 일반 DNA 통계를 넘어 학습한 개별 DNA 내용에 의존한다는 것이다. 수학적 가중치 역행렬 계산 대신, 저장된 최종 모델의 masked-token 생성으로 추출한다. 성공한 조건만 선택해 일반적인 유출률로 주장하지 않는다.

## 데이터와 반복

- 학습 seed 42–46의 5개 독립 초기 난수 실행. 모든 모델은 같은 공개 사전학습 가중치에서 시작한다. seed마다 공개 GUE H3K4me3 train의 중앙 96 bp 2,048개를 배경으로 사용한다.
- seed마다 random/genomic 두 문맥 유형 × epoch당 반복 0/1/4/16 × 32개 = **256개 문맥**. 각 문맥에 두 합성 비밀 A/B를 부여해 512개 96 bp 서열, 전체 5 seed에서 2,560개 서열을 만든다.
- 공통 prefix는 84 bp, secret은 12 bp. random prefix는 iid A/C/G/T; genomic prefix는 배경과 다른 공개 train 행의 중앙 96 bp 중 앞 84 bp. A/B secret은 서로 다른 무작위 DNA이고 원래 공개 source의 12 bp suffix와도 다르다.
- 토큰 경계 정보로 비밀 염기가 prefix에 섞이는 것을 막기 위해 `encode(prefix+secret)==encode(prefix)+encode(secret)` 및 secret이 정확히 3 BPE인 경우만 채택한다. A/B는 모두 이 조건을 충족한다. prefix는 최소12 BPE이고 전체 입력은 special token 포함64 이하인 경우로 제한한다. 최대 1,024회 secret construction으로 유효 pair를 못 만들면 다음 prefix를 사용하고 실패 수를 기록한다. 이 조건부 생성 분포를 순수 iid DNA 또는 실제 biological variant 분포라고 부르지 않는다.
- 같은 84 bp prefix와 3-token 길이를 가진 A/B twin은 **동일한 공격 입력**을 가진다. 모든 secret 및 전체 입력의 exact/RC 중복과 train/validation 중복을 검사하고 source/hash/생성 시도/GC/토큰 길이 정보를 저장한다. 근접 상동성 전수 제거를 했다고 주장하지 않는다.
- A arm은 각 노출 문맥의 A만, B arm은 B만 학습한다. 같은 반복 횟수, 길이, 순서, masking RNG를 사용한다. 0회 문맥은 양쪽 secret 모두 미학습이다. 한 epoch는 2,048+2×32×(1+4+16)=3,392개 예시. 마지막 epoch까지 0/20/80/320회 노출이다.
- 이 두 arm은 여러 secret을 동시에 교체하므로 그룹 단위 counterfactual이다. 별도로 seed42/43/44의 사전에 정한 genomic·repeat16 첫 문맥 하나만 A→B로 바꾸는 single_flip arm을 학습한다. 다른 입력과 순서는 A arm과 동일하다. 10개 주 모델+3개 단일 문맥 교환 모델=13개.
- 공개 dev 중앙 crop 512개를 고정된 mask 검증에 사용한다. 모델 선택에 사용하지 않으며 마지막 epoch20이 주 평가 checkpoint다. 검증 표본은 배경과 canary 전체의 exact/RC 중복을 배제한다.

## 학습

로컬 snapshot 7bce263b15377fc15361f52cfab88f8b586abda0, 원 MLM head 포함, 누락/불일치 weight 금지. FullFT, lr5e-5, AdamW WD.01, batch32, 20 epochs, clip1, BF16. 일반 dynamic MLM 15%에서 80%MASK/10%DNA 어휘 random/10%그대로. suffix를 특별히 목표로 하는 학습은 하지 않는다. attention dropout .1로 PyTorch 경로 사용. optimizer/RNG·epoch 순서 및 checkpoint1/5/10/20 저장. seed별 고정 공개 validation masked CE/accuracy를 기록한다. nonfinite는 실패로 보존하고 설정을 바꿔 성공 실행으로 대체하지 않는다.

학습 padding도64로 고정해 A/B와 single_flip에서 RNG 소비량과 mask 위치를 맞춘다. validation은 별도 generator를 사용해 학습 RNG를 바꾸지 않는다.

## 공격자가 받는 것과 고정 예산

공격 함수는 가중치, tokenizer, 별도 challenge의 공개 prefix token과 hidden token 수만 받는다. truth, A/B 정답, 전체 target DNA, exposure, model membership은 전달하지 않는다. decoder에서 정답 후보 목록을 열거하거나 teacher forcing하지 않는다. 모델 점수로 선택한 생성 top1만 정확 복구로 센다. FP32/TF32 off/autocast off, 최대 padding64, DNA-only vocabulary.

1. **주 endpoint**: epoch20, 숨긴 마지막3 BPE=전체12 bp, 전체84 bp prefix, greedy. trained secret 정확 일치율과 같은 prefix의 untrained twin 일치율 차이. repeat16에서 random/genomic을 별도 표와 통합 수치로 보고하고 나머지 dose를 함께 제시한다.
2. 문맥·비밀 길이: 숨긴 마지막3/6/12 BPE 각각에 대해 남은 prefix token의 마지막100%/50%/25%/0%만 제공. 비밀을 늘리면서 가려진 known-prefix 부분도 평가 대상에 포함한다. 실제 hidden/known bp를 기록한다. primary와 같은 secret3에서는 모든 문맥량 조건의 정답이 동일하다. hidden6/12도 A/B간 경계/길이는 같지만 비밀 문자열의 앞부분은 공통 prefix에서 온다. 0%도 숨긴 token 수는 알려주는 조건이다.
3. 탐색 보조: k3에서 전체/절반 문맥만 beam8. step당 1+8+8=17개의 시퀀스 forward. 다른 grid는 greedy만 실행하며 beam 성공을 주 결과로 바꿔 쓰지 않는다.
4. 학습 시점: checkpoint5/10에 대해서는 k3/전체 문맥/greedy만 재평가한다. checkpoint1은 저장 및 validation 진단용으로 둔다.
5. 문맥 없는 전체 서열 추출: 최종 A/B 각 모델과 사전학습 모델에서 독립적으로 2,048개 생성. target-specific prefix, target 길이, target ID 없이 공개 고정 seed로 token 길이12–40를 선택한다. 전부 MASK에서 시작해 12단계 confidence-first fill, top50 temperature1 sampling을 수행한다. 모든 생성 문자열을 저장한다. 전체96 bp canary의 exact-output 및 연속 substring 일치를 별도 지표로 평가한다. 12 bp 조각만 우연히 나타나는 것을 전체 복구로 세지 않는다. 비밀 suffix12 bp 출현은 실시하더라도 보조 발견 지표로만 보고하며 target 연관 복구로 계산하지 않는다.
6. single_flip arm: k3/전체 문맥/greedy 및 beam8, 전체256문맥 평가. 주 A와 비교해 바꾼 한 secret의 예측이 바뀌는지와 다른 문맥의 변화를 함께 기록한다.

## 기준선 및 통계

동일 사전학습 모델을 모든 challenge에 적용한다. twin secret과 미노출 문맥을 별도 대조로 둔다. 별도 dev에서 배운 order2 DNA Markov greedy 기준선도 제공하지만, 임의 BPE 길이만 아는 모델 공격과 bp 길이를 아는 Markov의 보조정보가 다르므로 이를 주 검정 기준선으로 쓰지 않는다. 주 k3는 secret이12 bp인 공개 실험 설계이므로 Markov는12 bp를 생성한다. k6/12는 Markov 비교를 생략한다.

각 seed·arm·유형·반복·문맥·비밀 길이에 대한 n/정확일치/타 twin 일치를 저장한다. 통합 분모는 독립 모델 수가 아니며 같은 문맥 A/B와 여러 공격 조건을 중복 독립 표본으로 취급하지 않는다. 주 추출 차이는 seed를 재표집하고 그 안에서 같은 문맥 pair를 묶어 재표집하는 hierarchical bootstrap 5,000회로 기술적95% CI를 계산한다. 5 seed의 per-seed 결과도 항상 제시한다. 숨긴 token accuracy/edit similarity는 보조이며 DNA exact와 혼용하지 않는다. 다중 grid 결과는 탐색적 기술 통계이고 유리한 조합만 선택한 유의성 주장을 하지 않는다.

single_flip은 3개 대상의 사례 검증이므로 모집단 회복률을 추정하지 않는다. 포함/제외 효과를 다른 학습 데이터가 변하지 않는 조건에서 확인하려는 기전 대조다. 임상 개인의 변이 추출로 부르지 않는다.

## 재현·실패 처리

새 디렉터리에만 결과를 저장하고 기존 실험을 보존한다. dataset construction 및 primary train 코드, config/protocol을 실행 전 해시 고정한다. checkpoint 해시와 optimizer/RNG 저장, 공격 replay, A/B challenge 동일성, single_flip 데이터 차이 하나, target 비밀과 입력 분리를 검증한다. 작은 독립 smoke seed를 코드 점검에 쓸 수 있으나 main 결과에 포함하지 않는다. 평가·보고 코드의 후속 수정이나 버그는 amendments.json에 이유와 영향을 남긴다. 임계값·선택 표본·primary endpoint를 결과에 따라 바꾸지 않는다.

## 선행연구와 주장 범위

canary 기반 암기 검증과 유전체 모델의 암기 연구는 이미 존재한다. 이번 범위의 차별화 후보는 같은 genomic context에 대한 무작위 secret 교체, 단일 문맥 교환 대조, 문맥 지식 감소 및 직접 생성 exact-match를 결합한 검증이다. 신규성이 확정됐다고 주장하지 않는다.

- Carlini et al., The Secret Sharer (USENIX Security2019): https://www.usenix.org/conference/usenixsecurity19/presentation/carlini
- Carlini et al., Extracting Training Data from Large Language Models (USENIX Security2021): https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
- Quantifying Memorization and Privacy Risks in Genomic Language Models: https://arxiv.org/html/2603.08913v1 ; ACM publication landing https://doi.org/10.1145/3807503.3819470

논문의 결론은 실제 결과에 맞춰 조건부 추출, 무문맥 추출, 혹은 해당 공격의 실패로 구분한다. HE의 방어 효과나 일반적인 유전체 보호 주장을 이 실험에 연결하지 않는다.

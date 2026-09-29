# FullFT DNA 복구 후속 문헌 검토와 실험 설계

검토일: 2026-09-28. 목적은 H3K4me3 분류 FullFT의 낮은 복구율을 설명할 후보 원인을 구분하고, 다른 공격의 실행 우선순위를 정하는 것이다. 관련 1차 자료의 방법·평가·학습 설정을 선별 확인했다. 체계적 문헌고찰이나 신규성 보증은 아니다. 이번 작업에서는 추가 학습이나 공격을 실행하지 않았다.

## 현재 결과에서 말할 수 있는 것

기존 결과는 `experiments/privacy_fullft_20260928/results_KO.md`와 개별 metrics JSON에 있다. 5 epoch에서 train/test 분류 AUROC는 0.9624/0.7898, 평가 시 평균 CE는 0.2903/0.6168이다. 학습 포함 여부 추정 AUROC는 0.6374이다. 이미 학습·평가 성능 차이와 membership 신호가 있으므로, 복구 실패를 단순히 학습 부족으로 설명할 근거는 없다. 반대로 이 지표만으로 원문 복구에 충분히 암기했다고 확정할 수도 없다.

기존 공격은 분류 encoder에 고정된 원 MLM head를 붙여 1회씩 greedy prediction을 수행했다. FullFT 후 학습 서열의 1-token 복구는 26/512에서 6/512로, 비학습 서열은 15/512에서 7/512로 낮아졌다. 양쪽에서 감소한 점은 decoder와 encoder 사이의 불일치 가설과 부합하지만, 이를 직접 입증한 것은 아니다. 분류 학습은 DNA를 빈칸에서 재생성하도록 직접 최적화하지 않는다.

## 문헌별 근거와 적용 범위

### 1. Memorization of Named Entities in Fine-tuned BERT Models

[Diera et al., arXiv v3, 2024](https://arxiv.org/pdf/2212.03749), §4.2–4.3, §6.1을 확인했다. 분류용으로 미세조정한 BERT에서 분류 head를 제거하고 사전학습 MLM head를 붙여 샘플링했다. 최종 선택된 학습량은 데이터에 따라 5 또는 10 epoch였고, 미세조정 전 모델에 비해 데이터 고유 개체명 추출이 일관되게 개선되지 않았다. 이번 방식과 가까운 선례다. 자연어 개체명 실험이므로 DNA의 정량적 복구율을 예측하지는 않는다.

### 2. Quantifying Memorization and Privacy Risks in Genomic Language Models

[Nemecek et al., arXiv v1, 2026](https://arxiv.org/html/2603.08913v1), §4–6 및 Appendix A/Table 9를 확인했다. DNABERT-2 설정은 **최대 50 epoch**, validation loss 기반 early stopping을 포함한다. 모든 실행이 실제로 50 epoch를 소화했다는 뜻은 아니다. 1,000개 기본 학습 서열에 반복 canary를 넣고 언어모델 기반 추출·loss·MIA를 비교했다. 우리 29,439개 분류 표본의 5 epoch와는 데이터 크기, 노출 횟수, 학습 목표가 함께 다르다. 따라서 이 논문을 근거로 에포크만 늘리면 복구된다고 예측할 수 없다.

### 3. The Secret Sharer

[Carlini et al., USENIX Security 2019](https://www.usenix.org/system/files/sec19-carlini.pdf), §7.1–7.2, Fig. 7–8을 확인했다. Canary 노출 신호가 초기 학습부터 생기는 사례와, 학습을 더 진행해도 exposure가 계속 증가하지 않는 사례가 있다. §7.2의 실험은 10 epoch에서 exposure가 정점에 이르렀고, 이후에도 해당 canary의 후보 순위는 1이었다. **에포크, 암기 점수, 실제 추출 가능성은 같은 양이 아니다.**

### 4. Quantifying Memorization Across Neural Language Models

[Carlini et al.](https://arxiv.org/html/2202.07646v3), §5.1 및 초록을 확인했다. 모델 크기, 데이터 반복, 주어진 문맥과 추출의 관계를 조사한다. Masked objective의 추출 정의와 평가를 causal objective에 맞게 그대로 옮길 수 없다는 점도 설명한다. 중복 횟수를 늘리는 실험과 전체 데이터의 epoch를 늘리는 실험을 분리해야 한다는 설계 근거로 사용한다.

### 5. Text Revealer

[Zhang et al., 2022](https://arxiv.org/pdf/2209.10505), §2–3.2, Table 1, Appendix A.3을 확인했다. 공개 자료로 학습한 GPT-2 생성기를 대상 분류 모델의 피드백으로 유도한다. 대상 BERT는 5 epoch 학습이다. 그러나 주요 Recovery Rate는 필터링한 단어의 회수율이고 Attack Accuracy는 별도 분류기가 생성문을 목표 클래스로 판정하는 비율이다. 높은 수치를 전체 원문 exact match로 읽으면 안 된다. DNA 생성기와 분류기 피드백을 결합할 아이디어는 얻을 수 있지만, 실제 학습 DNA의 추가 복구는 따로 입증해야 한다.

### 6. Are Your Sensitive Attributes Private?

[Mehnaz et al., USENIX Security 2022](https://www.usenix.org/system/files/sec22-mehnaz.pdf), §1 및 공격 가정을 확인했다. 알려진 속성은 고정하고 모르는 민감 속성의 값을 분류 모델의 confidence 또는 label로 추정한다. 대표적인 클래스 예제 생성과 특정 기록의 정확한 속성 추정을 구분하고, 모델을 쓰지 않는 기준선도 비교한다. 표 형식 데이터에서 검증한 방법이므로 DNA 적용은 별도 연구다. 우리의 짧은 비밀 염기 구간을 후보 값으로 바꿔 평가하는 데 적합한 출발점이다.

### 7. Reconstructing Training Data from Trained Neural Networks

[Haim et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/906927370cbeb537781100623cca6fa6-Paper-Conference.pdf), §3, §5.1–5.2를 확인했다. 학습 완료 가중치만으로 분류 학습 데이터를 재구성하는 실제 선례다. 다만 주 실험은 작은 영상 집합, 특정 ReLU MLP, 특수 초기화 및 매우 긴 full-batch 학습 조건이다. 기본 설정은 10^6 epochs와 거의 0인 학습 손실이다. 이론의 동차성 가정 등을 DNABERT-2 Transformer에 그대로 적용할 수 없으므로 직접 이식의 우선순위는 낮다.

### 8. 추가 접근 권한이 필요한 공격

- [DAGER, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ff1577a1f8308df1ccea6b4f64a103f-Paper-Conference.pdf): self-attention gradient의 구조를 이용한 입력 복구다. 초록과 위협모델을 확인했다. 관찰한 학습 gradient가 필요하며 최종 가중치만 제공된 이번 조건과 다르다. 공개 모델에서 공격자가 스스로 계산한 임의 gradient는 관찰된 비밀 batch gradient를 대신하지 않는다.
- [TIGER, 2026](https://arxiv.org/abs/2606.18312): gradient의 embedding-subspace 정보를 연속 최적화에 활용하는 후속 방법이다. 이번에는 초록만 확인했으며 정량 성능은 비교하지 않는다. 역시 gradient 접근이 필요하다.
- [How Private Are DNA Embeddings?, 2026](https://arxiv.org/html/2603.06950v1): §III–IV에서 실제 서열의 token별 embedding을 받은 뒤 복구하는 조건을 확인했다. 서열별 embedding 공유 위험을 연구할 때 적합하며, 최종 가중치만으로 같은 결과를 얻었다는 근거가 아니다.
- [Reconstructing Training Data with Informed Adversaries, 2022](https://arxiv.org/abs/2201.04845): 초록에서 다른 학습 기록을 모두 알고 하나만 모르는 강한 가정과 가중치→기록 reconstructor를 확인했다. 현재 데이터·권한 조건으로 바로 적용하는 공격과 구분한다.

## 다음 실험의 우선순위

아래는 문헌의 방법을 바탕으로 한 **우리 환경에 대한 제안**이며, DNABERT-2에서 이미 효과가 입증된 방법 목록이 아니다.

| 순서 | 고정할 것 | 바꿀 것 | 판별할 질문 |
|---|---|---|---|
| 1 | 현재 5-epoch FullFT 가중치, 평가 표본 | 짧은 염기 후보 검색, 반복 infilling/beam 탐색 | greedy 한 번만 시도해서 놓쳤는가? |
| 2 | 같은 가중치와 기존 BPE 복구 입력 | 고정 head 대 공개 보조 데이터로 보정한 head | encoder 표현 변화 때문에 원 MLM head가 읽지 못했는가? |
| 3 | 데이터·초기화·공격·query 예산·학습률 | 1/5/10/20 epoch 체크포인트 | 학습량에 따라 노출이 실제로 변하는가? |
| 4 | 위에서 선택한 조건과 학습량 | 평가 비밀의 포함/제외·대체 학습 | 분포적 예측 능력보다 학습 포함으로 인한 추가 복구인가? |
| 별도 연구 | 명시적인 FL 또는 embedding 공유 환경 | 공개되는 중간 정보 | gradient 또는 embedding 누출이 있는가? |

### A. 저장된 5-epoch 모델에서 할 수 있는 평가

**짧은 염기 후보 검색.** 알려진 문맥 사이의 1/2/4 bp를 숨기고 가능한 A/C/G/T 조합 4/16/256개를 모두 생성한다. 정답을 별도로 후보에 끼워 넣지 않고 전체 가능한 공간을 열거한다. 각 완성 서열을 새로 tokenize하여 분류기의 confidence 또는 label 반응을 사용한다. 분류 정답을 아는 공격과 모르는 공격은 분리한다. 원 학습 서열과 무관한 클래스 전형도 높은 점수를 받을 수 있으므로, 1위 후보가 숨긴 염기와 정확히 같은지 별도 평가한다. 비학습 표본·모델 미사용 prior·사전학습 및 학습 제외 모델을 비교한다. 정답 순위와 top-k는 별도 보조 지표이며 top-1 복구와 합치지 않는다. 이 실험은 정확한 bp 길이를 알려주는 새로운 보조정보 조건이므로 기존 BPE 개수만 알려준 실험과 수치를 직접 비교하지 않는다.

**탐색 예산 증가.** 기존 BPE challenge는 그대로 두고 fixed head에 반복 infilling, beam 또는 best-of-N sampling을 적용한다. 예산과 선택 규칙을 먼저 정한다. 여러 후보 중 평가 정답을 보고 고른 성공은 단일 최종 출력의 성공으로 세지 않는다. 이 방법은 head 불일치를 고치지는 못한다.

**Decoder 보정.** Target encoder는 고정하고 공개 보조 서열만으로 작은 MLM head 또는 decoder를 맞춘다. 공개 보조 자료는 private 평가 대상과 exact/RC/근접 중복을 배제하고, 보정용과 공격 검증용으로 분할한다. 새 decoder에 private 원문·숨긴 정답을 제공하면 안 된다. 공개 검증 자료에서 복구 기능이 작동하는지 먼저 확인한다. 각 checkpoint에 같은 보조 자료와 같은 보정 예산을 사용한다. 보정 head가 private 구간을 더 맞혀도 일반적인 DNA 패턴 학습인지와 학습 데이터 암기인지는 포함/제외 대조로 구분한다.

### B. Epoch 효과만 분리하는 실험

새 학습 궤적 하나에서 epoch 1/5/10/20을 저장하고 모든 checkpoint에 동일한 공격을 적용한다. 처음에는 seed 42로 실행 경로를 확인하고, 본 주장에는 여러 seed를 사용한다. 20 epoch에서도 목적을 달성하지 못했다는 이유만으로 성공할 때까지 epoch를 계속 늘리지 않는다. 추가 50-epoch 실험은 별도의 stress condition으로 명시한다.

현재 저장된 파일에는 모델 가중치만 있고 AdamW optimizer 상태는 없다. 따라서 기존 5-epoch 가중치에서 optimizer를 초기화하고 15 epoch 더 돌리는 것을 중단 없이 학습한 20-epoch 결과로 부르면 안 된다. 학습량 비교의 주 실험은 사전학습 상태에서 동일한 궤적으로 다시 학습하며 optimizer·RNG 상태와 데이터 순서를 함께 저장해야 한다. 기존 5-epoch 기록을 재현 점검에 이용한다.

분류 AUROC/CE, MIA AUROC 및 낮은 FPR의 TPR, 숨긴 구간 exact match를 함께 추적한다. 정답을 모르는 공격자 기준의 최종 출력 규칙, 단일 서열의 query 예산, 제공 문맥, 숨긴 bp/token 수를 고정한다. 선택은 독립 보조 검증 자료에서 하고 최종 평가 집합을 공격 튜닝에 반복 사용하지 않는다.

### C. 결과에 따른 해석

- 같은 5-epoch 가중치에서 보정 decoder 또는 탐색 개선만으로 복구가 증가하면 기존 공격의 제한이 있었다는 근거다.
- 같은 공격에서 epoch를 늘릴 때 학습 표본의 복구가 증가하고, 비학습 및 포함/제외 대조보다 증가폭이 크면 학습량에 따른 누출을 주장할 근거가 된다.
- 학습·비학습이 비슷하게 증가하면 일반적인 복구 능력의 개선일 수 있다.
- MIA만 증가하고 exact 복구는 증가하지 않으면 membership 노출과 재구성 가능한 정보가 다르게 변한 것이다.
- 모든 시도가 실패해도 정해진 공격·보조정보·예산에서 실패한 결과로 보고한다. 최악의 공격에 대한 불가능성 증명은 아니다.

## 현재 판단

에포크 부족은 검증할 가설이다. 그러나 현재 자료에서 원인이라고 단정할 수 없다. 우선 저장된 모델에 적용하는 공격과 decoder를 개선하고, 별도 epoch sweep으로 학습량 효과를 분리하는 것이 해석 가능한 다음 단계다. 기존 합성 MLM 실험, 실제 분류 FullFT 실험, 향후 gradient/embedding 공격은 서로 다른 위협모델로 보고해야 한다.

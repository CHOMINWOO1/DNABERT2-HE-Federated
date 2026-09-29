# 기관별 부분복호화 CKKS-FL 추가 실험 계획

## 현재 상태

- 코드 작성: 완료
- 실제 모델 학습: 미실행
- 실제 OpenFHE 암호 연산: 미실행
- GPU 사용: 없음
- 결과/PPT 반영: 아직 없음

실행 스크립트에는 `--execute` 잠금이 있다. 이 옵션을 명시하지 않으면 계획만 출력하고 학습·GPU·출력 디렉터리 생성을 시작하지 않는다.

## 실험 질문

기존 실험은 서버 측에서 최종 복호화를 수행했다. 추가 실험은 세 기관이 동일한 공동 공개키로 업데이트를 암호화하고, 암호문 합산 후 각 기관이 부분복호화 값을 하나씩 제공하도록 바꾼다. 서버 단독으로는 평문 집계값을 복원할 수 없고 세 기관의 부분복호화가 모두 모였을 때만 최종 집계값을 복원하는 all-party 방식이다.

비교하려는 핵심은 다음과 같다.

- Plain FedAvg LoRA와 기관별 부분복호화 CKKS-FedAvg LoRA의 성능 차이
- CKKS 근사 오차: MAE, 최대 절대오차, 상대 L2 오차, cosine similarity
- 기관별 암호화·부분복호화 시간
- 라운드별 논리 통신량: 암호문 업로드, 집계 암호문 기관 배포, 부분복호화 업로드
- 기존 Plain 조기종료 지점을 그대로 사용할 때의 paired 성능

## 고정된 추가 arm

H3K4me3 non-IID, LoRA, seed 42–46의 5개 arm만 추가한다. 새 HE arm에서 최적 라운드를 다시 고르면 비교 편향이 생기므로 기존 Plain FedAvg가 dev에서 선택한 라운드를 고정 사용한다.

| seed | 고정 학습 라운드 | 비교 기준 |
|---:|---:|---|
| 42 | 32 | 기존 Plain FedAvg seed 42 |
| 43 | 27 | 기존 Plain FedAvg seed 43 |
| 44 | 34 | 기존 Plain FedAvg seed 44 |
| 45 | 28 | 기존 Plain FedAvg seed 45 |
| 46 | 27 | 기존 Plain FedAvg seed 46 |

따라서 기존 234 arms에 결과가 검증된 뒤 5개를 추가하면 총 239 arms로 기술할 수 있다. 실행 전이나 실패한 arm은 완료 arm 수에 포함하지 않는다.

## 암호 프로토콜

1. 세 기관이 순차적 multiparty key generation으로 각자 비밀키 조각과 최종 공동 공개키를 만든다.
2. 각 기관은 표본 수 가중치가 적용된 LoRA 업데이트 벡터를 공동 공개키로 CKKS 암호화한다.
3. 서버는 비밀키 없이 세 암호문을 덧셈 집계한다.
4. 서버는 집계 암호문을 세 기관에 배포한다.
5. 첫 기관은 lead partial decryption, 나머지 기관은 main partial decryption을 계산한다.
6. 세 부분복호화 결과를 fusion하여 가중합을 복원하고 전역 LoRA 상태를 갱신한다.

현재 구현은 **한 프로세스 안에서 기관 역할을 분리해 재현하는 시뮬레이션**이다. 완전한 비밀키는 만들거나 저장하지 않지만, 실행 중에는 세 비밀키 조각이 같은 프로세스 메모리에 존재한다. 따라서 논문에는 “OpenFHE multiparty/threshold CKKS의 in-process simulation”으로 써야 하며, 실제 기관 간 물리적 격리나 운영 환경의 보안을 입증했다고 표현하면 안 된다. 실제 기관 격리를 주장하려면 각 역할을 별도 프로세스 또는 별도 호스트로 분리하고 인증된 통신을 추가해야 한다.

## 실행 전 조건

- CUDA가 활성화된 PyTorch 환경
- 기존 고정 DNABERT-2 로컬 스냅샷
- H3K4me3 train/dev/test CSV
- `h3k4me3_fl_earlystop_v3`의 seed 42–46 Plain 완료 산출물
- OpenFHE-Python 1.5.1 계열과 필요한 multiparty API

공식 PyPI wheel은 Ubuntu LTS용이다. 현재 Windows 네이티브 환경에서는 OpenFHE-Python을 소스 빌드하거나, Ubuntu 22.04 WSL/리눅스 환경에서 별도 의존성 파일을 사용해야 한다.

## 나중에 실행할 명령 — 이번 작업에서는 실행하지 않음

준비 상태 확인:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_threshold_ckks.py
```

OpenFHE만 설치한 뒤 작은 암호 덧셈 확인:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_threshold_ckks.py --crypto-smoke
```

학습 없이 고정 계획 출력:

```powershell
.\.venv\Scripts\python.exe scripts\run_fedhe_threshold_ckks_v1.py
```

실제 단일 seed 학습은 명시적 잠금 해제 후에만 가능하다:

```powershell
.\.venv\Scripts\python.exe scripts\run_fedhe_threshold_ckks_v1.py --execute --seed 42
```

나머지 seed도 각각 별도로 실행한다. 한 seed가 완료되면 `DONE.json`과 해시가 있는 산출물만 유효한 arm으로 센다.

전체 seed를 순차 실행하는 PowerShell 런처도 기본값은 계획 출력만 한다. 실제 실행에는 역시 `-Execute`가 필요하다.

```powershell
.\scripts\launch_threshold_ckks_v1.ps1
# 실제 실행 시에만: .\scripts\launch_threshold_ckks_v1.ps1 -Execute
```

## 재시작과 비밀키 처리

- 매 라운드 종료 후 모델 상태와 진행 ledger를 저장한다.
- 비밀키 조각은 체크포인트에 저장하지 않는다.
- 중단 후 재시작하면 새로운 multiparty key epoch를 만들고 다음 라운드부터 계속한다.
- 기존 완료 산출물이나 Plain 비교 결과를 덮어쓰지 않고 새 출력 루트에 저장한다.

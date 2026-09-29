# DNABERT-2 연합학습 실험의 TenSEAL 및 CKKS 재현성 감사

감사 시점은 2026-08-19 02:32:46 UTC이다. 이 문서는 현재 `.venv`를 읽기 전용으로 조사한 결과를 정리한다. 패키지 설치, 다운로드, 업그레이드, 삭제는 수행하지 않았다. H3K9 계열 결과 파일도 열지 않았다. 기계 판독 가능한 전체 증적은 [`paper_extension_crypto_environment_audit.json`](../experiments/paper_extension_crypto_environment_audit.json)에 있다.

## 1. 결론

현재 암호 환경은 CPython 3.11.9, TenSEAL 0.3.17, Windows x86-64 네이티브 wheel 조합이다. 설치된 SEAL 공개 API의 직렬화 헤더는 버전 4.3을 반환했다. 따라서 연결된 Microsoft SEAL 버전은 `4.3.x`로 기록할 수 있다. patch 버전은 wheel 메타데이터, 공개 API, 네이티브 문자열에서 확인되지 않았으므로 추정하지 않는다.

실험 파라미터 `N=8192`, coefficient-modulus bit sizes `[60,40,60]`, scale `2^40`을 설치된 SEAL 검증기에 입력했다. 총 coefficient-modulus 길이 160 bits는 TC128 한도 218 bits 이내였고 `parameters_set=True` 판정을 받았다. TC192 한도 152 bits와 TC256 한도 118 bits는 초과했으며 두 검증 모두 `invalid_parameters_insecure`를 반환했다. 따라서 논문에서는 다음 범위로 기술한다.

> 본 구현의 CKKS 파라미터(N=8192, coefficient-modulus bit sizes=[60,40,60], 총 160 bits)는 설치된 Microsoft SEAL 4.3.x의 HomomorphicEncryption.org TC128 파라미터 검증을 통과했다.

이 문장은 해당 파라미터 집합과 설치된 SEAL 빌드에 관한 적합성 진술이다. 시스템 전체에 대한 보안 증명, 독립 암호분석, side-channel 평가, 192-bit 이상 보안, 실제 Key Authority 격리를 뜻하지 않는다.

현재 집계기는 단일 Python 프로세스 안에 비밀키 context와 공개 context를 함께 둔다. 같은 프로세스가 키를 생성하고, 모의 client update를 암호화하고, 암호문을 합산하고, cohort aggregate를 복호화한다. 따라서 이 구현은 `single-key logical KA prototype`으로 분류한다. 암호문 집계의 기능성, 수치 충실도, 직렬화 비용을 평가할 수 있지만, 서버로부터 비밀키가 물리적으로 격리됐다는 근거는 제공하지 않는다.

## 2. 설치 provenance

| 항목 | 감사 결과 |
|---|---|
| Python | CPython 3.11.9, MSC v.1938, 64-bit AMD64 |
| 운영체제 표기 | Windows-10-10.0.26200-SP0 |
| TenSEAL | 0.3.17 |
| 설치 방식 | `pip`, `REQUESTED` marker 존재 |
| wheel tag | `cp311-cp311-win_amd64` |
| wheel generator | setuptools 83.0.0 |
| wheel 유형 | `Root-Is-Purelib: false` |
| PEP 610 origin | `direct_url.json` 없음 |
| 원본 wheel archive | pip cache에 남아 있지 않아 archive SHA-256 복구 불가 |
| RECORD | 72 rows, 누락 0, hash/size 불일치 0 |
| 설치 파일 | 72개, 이 중 runtime 생성 `pyc` 12개 |
| wheel payload aggregate SHA-256 | `5ae13f8a1a4009df6d5818fbabbe539b6d6ae62a95af8e027f866b9151683fbf` |
| 현재 설치 tree aggregate SHA-256 | `18ae1d6b4cd90ddd6a4694a33570c32508d894229d6024dca19a7fd37ab711d3` |

원본 wheel archive가 없으므로 archive 자체의 SHA-256은 복구할 수 없다. 대신 설치된 모든 관련 파일의 bytes와 SHA-256, `METADATA`, `WHEEL`, `RECORD`, `INSTALLER`, `REQUESTED`를 JSON에 기록했다. 이 증적은 현재 실행 환경을 식별하지만, PyPI의 어느 archive에서 설치됐는지까지 입증하지는 않는다. 후속 재현 패키지에서는 wheel 파일 또는 불변 URL과 archive SHA-256을 별도로 보존해야 한다.

주요 dist-info 해시는 다음과 같다.

| 파일 | bytes | SHA-256 |
|---|---:|---|
| `METADATA` | 8,871 | `cd986274129e094bf3820dbffbeca8f1ae049388f290a25c776eb025ed044b00` |
| `WHEEL` | 101 | `0b81650649feb673233bd9f4ddac78c902452a1d8bf79cf7362558225eedf1c6` |
| `RECORD` | 5,976 | `7f4794bb7934d7b33afc4fb7fe57d777f13f51e486442d7aa999cb1765e7e797` |
| `INSTALLER` | 4 | `ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508` |

## 3. 네이티브 바이너리와 Microsoft SEAL 버전

| 네이티브 모듈 | bytes | SHA-256 | PE import |
|---|---:|---|---|
| `_sealapi_cpp.cp311-win_amd64.pyd` | 1,980,416 | `28ebfe29def3e844d66d7ac41aa2c430a663a6cf57fd6b524d89face6fd270bb` | `bcrypt.dll`, `KERNEL32.dll`, `python311.dll` |
| `_tenseal_cpp.cp311-win_amd64.pyd` | 3,468,288 | `78299513ef9eb36ba9f4921818bc8ccfa07cb47978bb12009adfb0407e984722` | `bcrypt.dll`, `KERNEL32.dll`, `python311.dll` |

두 PE import table에는 SEAL 이름을 가진 DLL이 없었고, site-packages 최상위에도 인접한 SEAL DLL이 없었다. 두 바이너리에는 SEAL C++ symbol 및 RTTI 문자열이 존재한다. 이 조합은 Microsoft SEAL 코드가 Python 확장 모듈에 static link 또는 bundle된 상태와 일치한다. 이는 PE 및 문자열 증거에 따른 구현 수준의 판단이다.

연결 버전은 다음 세 단계로 확인했다.

1. TenSEAL의 설치 source `sealapi_helpers.cpp`는 `Serialization.SEALHeader.version_major`와 `version_minor`를 Python 공개 API에 연결한다.
2. 현재 `_sealapi_cpp`에서 `Serialization.SEALHeader()`를 생성했을 때 magic `41310`, header size `16`, major `4`, minor `3`이 반환됐다.
3. 네이티브 printable-string 감사에서는 SEAL symbol을 확인했지만 명확한 SEAL patch 버전은 찾지 못했다. 관찰된 `1.3.2`와 `3.9.1` 문자열은 압축 또는 protobuf 계열 의존성 흔적이며 SEAL 버전 근거로 사용하지 않았다.

따라서 논문 및 재현성 표에는 `Microsoft SEAL 4.3.x, major/minor observed via Serialization.SEALHeader; patch unavailable`로 기록한다. 정확한 patch를 임의로 `4.3.0` 등으로 표기해서는 안 된다.

## 4. CKKS 파라미터 보안 판정

SEAL `CoeffModulus.Create(8192,[60,40,60])`가 생성한 세 prime의 실제 bit 길이는 `[60,40,60]`이었다. 값은 각각 `1152921504606748673`, `1099511480321`, `1152921504606830593`이다. 총 bit 길이는 160이다.

| SEAL 검증 수준 | N=8192 허용 최대 총 bits | 사용 총 bits | 여유 또는 초과 | 결과 |
|---|---:|---:|---:|---|
| TC128 | 218 | 160 | 58 bits 여유 | 통과, `success` |
| TC192 | 152 | 160 | 8 bits 초과 | 실패, `invalid_parameters_insecure` |
| TC256 | 118 | 160 | 42 bits 초과 | 실패, `invalid_parameters_insecure` |

설치된 low-level utility가 반환한 TQ 참고 한도는 128/192/256 수준에서 각각 202/141/109 bits였다. 사용값 160 bits는 128 수준 참고 한도 이내지만, TenSEAL의 `SEALContext` 보안 enum과 본 명시적 pass/fail 검증은 TC128을 사용했다. 본 연구에서는 이를 근거로 별도의 post-quantum 보안 보증을 주장하지 않는다.

Scale `2^40`은 CKKS approximate-number encoding의 정밀도와 rescaling 동작에 관련된 값이다. Scale을 40 bits로 선택했다는 사실이 40-bit 보안 또는 보안 수준 자체를 의미하지 않는다.

[Microsoft SEAL `modulus.h`](https://github.com/microsoft/SEAL/blob/main/native/src/seal/modulus.h)는 `sec_level_type::tc128`을 HomomorphicEncryption.org 표준에 따른 128-bit 수준으로 정의하고 `CoeffModulus::MaxBitCount`를 제공한다. [Microsoft SEAL의 공식 예제](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp)는 N=8192에서 TC128 최대 coefficient-modulus 길이를 218 bits로 설명한다. 표준의 파라미터 표와 공격 비용 모형은 [Homomorphic Encryption Standard v1.1](https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf)에 제시돼 있다.

## 5. 동적 기능 확인

감사 스크립트는 파일에 키나 암호문을 저장하지 않고 메모리에서만 다음 검사를 수행했다.

- 비밀 context: `is_private=True`, secret key 보유, public key 보유
- 공개 context: `is_public=True`, secret key 미보유, public key 보유
- 공개 context 직렬화 크기: 363,826 bytes
- 비밀 context 직렬화 크기: 545,031 bytes
- 동일 평문을 두 번 암호화한 직렬화 결과: 서로 다름
- 세 원소 벡터 두 개의 암호문 덧셈 후 최대 절대 복원 오차: `5.599003927159174e-09`

동일 평문의 두 암호문이 달랐다는 관찰은 현재 경로가 randomized encryption을 사용한다는 기능적 확인이다. 단일 1회 검사는 난수 생성기의 통계적 품질이나 암호 안전성을 평가하지 않는다. 복원 오차 또한 이 감사용 벡터에 대한 1회 측정이며 본 실험 전체의 오차 범위로 확대 해석하지 않는다.

## 6. 구현된 key topology

감사 대상 runner의 소스 해시는 JSON에 포함했다. 해당 snapshot에서 CKKS 설정은 `run_fedhe_progress_v2.py` 207-209행에 있고, 값은 N=8192, `[60,40,60]`, scale bits=40이다.

`run_fedhe_experiment.py`의 제어 흐름은 다음과 같다.

1. 465행: `self.secret_context = ts.context(...)`로 비밀 context를 생성한다.
2. 470-476행: public key를 포함하되 secret, Galois, relinearization key를 제외해 public blob을 만든다.
3. 477행: 같은 Python 프로세스에서 public context를 복원한다.
4. 491행: public context로 client vector를 암호화한다.
5. 509행: 같은 객체가 보유한 secret context로 aggregate chunk를 복호화한다.

Full-FT 경로도 `run_fedhe_main_experiment.py` 899행에서 같은 `CKKSAdditiveAggregator` backend를 사용한다. 따라서 LoRA와 Full-FT 모두 아래 위협 모형 한계를 공유한다.

| 항목 | 현재 구현 | 논문에서 가능한 해석 |
|---|---|---|
| client upload 표현 | randomized CKKS ciphertext | 모의 upload/aggregation path에서 update가 평문 대신 암호문으로 표현됨 |
| server aggregation | secret key가 없는 public context로 덧셈 | 암호문 가중합의 기능성과 비용 평가 |
| aggregate decryption | cohort aggregate만 secret context에 전달 | 구현 제어 흐름에서 개별 client ciphertext를 직접 복호화하지 않음 |
| key holder | 집계기와 같은 Python 프로세스 | 논리적 역할 분리만 존재 |
| secret-key isolation | 별도 process, host, HSM 없음 | 물리적 또는 조직적 KA 격리를 주장할 수 없음 |
| decryption model | 단일 secret key | threshold 또는 multiparty CKKS가 아님 |
| released aggregate | key holder가 평문 aggregate를 획득 | small-cohort 또는 differencing leakage는 HE만으로 방지되지 않음 |
| malicious behavior | 인증, attestation, poisoning 방어 없음 | honest execution prototype 범위로 제한 |
| privacy layer | DP guarantee 없음 | client update 암호화와 aggregate privacy를 구분해야 함 |
| network | 실제 기관 간 채널 및 traffic 미모사 | network latency, metadata leakage, replay를 평가하지 않음 |

[Microsoft SEAL의 Correct Use 지침](https://github.com/microsoft/SEAL/security)은 복호화 결과를 secret-key owner에게만 제공해야 한다고 설명하고, plaintext-independent execution time을 보장하지 않는다고 명시한다. 본 실험은 process isolation과 side-channel을 평가하지 않으므로 이 한계를 배포 보안 주장에 포함해야 한다.

## 7. 논문 Methods에 사용할 수 있는 문구

다음 문구는 확인된 구현과 감사 증거 범위에 맞춘 것이다.

> 암호 집계는 CPython 3.11.9에서 TenSEAL 0.3.17과 static-bundled Microsoft SEAL 4.3.x를 사용해 구현했다. 연결된 SEAL major/minor는 `Serialization.SEALHeader` 공개 API로 확인했으며, patch 버전은 설치 metadata에서 확인되지 않았다. CKKS는 polynomial modulus degree 8192, coefficient-modulus bit sizes `[60,40,60]`, global scale `2^40`으로 설정했다. 총 coefficient-modulus 길이 160 bits는 설치된 SEAL의 HomomorphicEncryption.org TC128 검증을 통과했다. 각 client update는 secret key를 제외한 public context로 암호화됐고, 서버 역할은 암호문 가중합만 수행했다. cohort aggregate는 secret context로 복호화했다. 다만 두 context가 동일 Python 프로세스에 존재하므로 본 구현은 single-key logical KA prototype이며, 독립 KA 또는 threshold decryption 배포를 구현한 것은 아니다.

Results 또는 Discussion에는 다음과 같이 범위를 제한한다.

> 본 비교는 논리적으로 분리된 public/secret context에서 수행한 암호문 집계의 utility, 수치 충실도, 직렬화 및 계산 비용을 평가한다. 동일 프로세스가 비밀키를 보유하므로 실제 기관 간 key custody, coordinator memory compromise, collusion, side-channel에 대한 보호 효과는 평가하지 않았다.

영문 핵심 보안 문구는 다음과 같다.

> The CKKS parameter set (N=8192; coefficient-modulus bit sizes [60,40,60], 160 bits in total) passes the HomomorphicEncryption.org TC128 parameter validator in the installed Microsoft SEAL 4.3.x build. This is a parameter-compliance statement, not an end-to-end security guarantee.

## 8. 더 강한 보안 주장을 위한 요구사항

실제 기관 간 배포에서 서버와 key holder를 분리하려면 최소한 다음 항목이 추가돼야 한다.

1. 독립 관리 주체가 운영하는 별도 KA process 또는 service에서 key generation과 decryption을 수행한다.
2. 인증된 key distribution, encrypted transport, access control, key rotation, audit log를 정의한다.
3. 단일 조직이 완전한 secret key를 가져서는 안 되는 위협 모형이라면 threshold 또는 multiparty CKKS를 사용한다.
4. minimum cohort size, dropout, collusion, aggregate release 정책을 사전에 정의한다.
5. 복호화된 aggregate 자체의 민감도가 문제라면 secure aggregation과 differential privacy를 별도로 검토한다.
6. 배포 대상 binary, serialization, memory handling, timing behavior에 대해 위협 모형에 맞는 보안 검토를 수행한다.

이 항목이 구현되기 전에는 `privacy-preserving federated learning system`보다 `CKKS-encrypted aggregation prototype` 또는 `logical-KA simulation`이 증거에 맞는 표현이다.

## 9. 재현 명령

읽기 전용 검증은 다음 명령으로 반복할 수 있다.

```powershell
& .\.venv\Scripts\python.exe .\scripts\audit_crypto_environment.py --verify-only
```

기존 증적을 덮어쓰지 않고 별도 JSON을 만들려면 다음과 같이 실행한다.

```powershell
& .\.venv\Scripts\python.exe .\scripts\audit_crypto_environment.py `
  --output .\experiments\paper_extension_crypto_environment_audit_rerun.json
```

현재 감사 script와 JSON의 SHA-256은 다음과 같다.

| artifact | SHA-256 |
|---|---|
| `scripts/audit_crypto_environment.py` | `dd0324006ee9011a9b8d95dc6b1709dd33a61eff533f266905bc4818532831a7` |
| `experiments/paper_extension_crypto_environment_audit.json` | `744954aeaf150938c94a318ea8b08e79cdf433e2946741d6dd1241647b9b6738` |

감사 JSON에는 난수 기반 ephemeral encryption check와 생성 시각이 포함되므로 재실행한 JSON 전체 hash는 달라질 수 있다. 설치 파일별 SHA-256, RECORD 판정, wheel payload aggregate hash를 환경 동일성 비교의 기준으로 사용한다.

## 10. 인용 및 artifact 지침

논문 본문에는 TenSEAL version, SEAL major/minor, CKKS 파라미터, TC128 검증 범위, logical KA 한계를 함께 적는다. Supplementary artifact에는 감사 JSON과 script를 포함한다. 원본 wheel archive hash와 정확한 SEAL patch를 복구할 수 없었다는 사실도 숨기지 않는다.

주요 외부 근거는 다음과 같다.

- [TenSEAL repository](https://github.com/OpenMined/TenSEAL): TenSEAL의 Microsoft SEAL 기반 구조와 공개 API
- [Microsoft SEAL `modulus.h`](https://github.com/microsoft/SEAL/blob/main/native/src/seal/modulus.h): 표준 보안 수준과 `MaxBitCount`
- [Microsoft SEAL BFV basics](https://github.com/microsoft/SEAL/blob/main/native/examples/1_bfv_basics.cpp): N과 coefficient-modulus 총 bit 길이의 관계
- [Homomorphic Encryption Standard v1.1](https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf): 권고 파라미터 표와 공격 비용 모형
- [Correct use of Microsoft SEAL](https://github.com/microsoft/SEAL/security): 복호화 정보, timing, 배포 보안 주의사항

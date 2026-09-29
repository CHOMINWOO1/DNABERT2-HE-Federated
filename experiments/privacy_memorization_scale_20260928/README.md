# 최종 가중치의 DNA 암기·복구 통제 실험

2026-09-28에 수행한 DNABERT-2 MLM FullFT 연구 자료다. 5 seed, 13개 모델, 1,280개 문맥, 2,560개 합성 canary를 포함한다. 기존 실험 폴더는 보존했다.

- 결과와 해석: `results_KO.md`
- 원고용 방법·결과: `manuscript_methods_results_KO.md`
- 수치 표: `paper_tables.md`
- 논문 그림 PNG/PDF/SVG: `figures/`
- 설계: `protocol_KO.md`, `protocol_frozen.json`, `config.json`
- 실행 명령: `REPRODUCE.md`
- 선행연구 비교: `literature_positioning_KO.md`
- 검증: `verification.json`, `runtime_inventory.json`, `visual_review.json`
- 전체 산출물 해시: `completion_manifest.json`; 완료 상태: `DONE.json`

주 결과는 84 bp 문맥과 숨긴 3-token 길이를 제공했을 때 학습된 12 bp 비밀의 정확 복구다. 320회 노출 조건에서 616/640개(96.25%)였으며, 사전학습 모델과 같은 문맥의 미학습 비밀 대조는 각각 0/640개였다. 이는 공개 효모 문맥 또는 무작위 문맥에 삽입한 합성 비밀에 대한 결과다. 실제 환자 유전체 유출이나 무문맥 전체 서열 복원을 입증한 자료로 해석하지 않는다.

# 포트폴리오 감사 기록

## 재현 가능한 V6 authority

- 평가 범위: 회고적 GT0 Nested Outer-5 개발 benchmark
- V1 parent: `600/820 = 73.17%`
- V6: `618/820 = 75.37%`
- 변경 감사: `65 = 25 W→C + 7 C→W + 33 W→W`
- 순개선: `25 - 7 = 18`, 따라서 `600 + 18 = 618`
- fresh standalone replay: `PASS`
- 동결 prediction SHA256: `E9502521DB1FFC315DB6DA9DB86FF9B54970379CF45B6A028F006EA0B433CC64`
- 최종 result SHA256: `317F1371624CE95696A5ABCBDBACB452A15AD2F81E57E1145C648C054613203D`

이 결과는 blind/unseen/external 성능이 아닙니다. production 정확도, 공식 대회 점수 또는 real-world generalization 입증으로 해석하지 않습니다.

## 동결 source 감사

production-style 시스템, 과거 T2 연구, C# 프로젝트의 code와 JSON은 기존 SHA manifest 기준으로 동결했습니다. 정적 검증기는 다음 경로의 비문서 source SHA를 확인합니다.

- `01_CCTV_LPR_System/`
- `02_T2_Accuracy_Research/`
- `03_CSharp_ImageProcessing/`

실행 관계는 `run_detection.py → gpu_pipeline2.GPUPipeline → gpu_pipeline.GPUPipeline`입니다.

## 과거 연구 계보

- T2는 별도 과거 연구 snapshot이며 `541/820 = 65.98%`입니다.
- 2026-10-08 `622/820 = 75.85%` Core-Light 기록은 별도 `PARTIAL_REPRODUCTION` snapshot으로, 현재 재현 기준 headline이 아닙니다.

## 공개 안전성

- model weight, raw CCTV, crop, GT, credential, private path를 공개하지 않습니다.
- root `LICENSE`를 임의로 만들지 않았습니다.
- Ultralytics 상태는 `LICENSE_REVIEW_REQUIRED`입니다.
- Qwen3-VL은 candidate source이며 최종 scorer가 아닙니다.
- 공개 V6 자료는 `AUDIT_SUMMARY_ONLY`입니다.

배포 전 `python tools/verify_portfolio_static.py`를 실행합니다. 하나라도 `FAIL`이면 commit과 push를 중단합니다.

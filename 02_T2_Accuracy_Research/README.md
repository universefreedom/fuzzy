# T2 Accuracy Research

## 목적

T2는 메인 CCTV 실행 파이프라인과 분리된 정확도 개선 연구 트랙입니다. 번호판 후보의 특징을 구성하고, group-disjoint cross-fit selector로 parent와 challenger를 비교합니다.

## 포함한 frozen producer code

- `build_c3_a4_selector_features.py`: C3/A4 후보의 selector feature 생성
- `run_c3_a4_selector_crossfit_pre_gt.py`: GT 결합 전 cross-fit prediction 생성
- `run_real822v3_c3_a4_crossfit_selector_v1.py`: Real822v3 selector 실행
- `score_real822v3_c3_a4_selector_postfreeze.py`: prediction freeze 이후 평가
- `run_real822_cross_group_confusion_transfer_v1.py`: cross-group confusion transfer 실험

## 검증된 수치

`evaluation/METHOD_CARD.json`과 `REPRO_RESULT.json` 기준:

| 항목 | 결과 | 해석 |
|---|---:|---|
| C3 baseline | 481/820, 58.66% | frozen automatic Top-1 |
| T2 | 541/820, 65.98% | leakage-free 5-fold OOF automatic Top-1 |
| paired transition | W→C 81, C→W 21, net +60 | T2 대 C3 |
| A4 | 583/820, 71.10% | candidate material ceiling, automatic accuracy 아님 |

과거 문서의 `542/822`는 이 frozen bundle과 분모·정답 수가 다르므로 포트폴리오 수치로 사용하지 않았습니다.

## 적용 경계

이 bundle은 Real822v3 820-group 실험 재현 근거입니다. 임의 CCTV 영상에 대한 production 성능이나 일반화 성능을 주장하지 않습니다. 실패하거나 승격되지 않은 실험은 원본 연구 artifact에 남아 있으며 이 디렉터리에서는 champion 코드와 직접 검증 파일만 보존했습니다.


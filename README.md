# Software & Vision Portfolio

실제 입력 환경에서 동작하는 영상처리 시스템을 구축하고, 측정·로그 분석·회귀 검증을 통해 문제 원인을 추적해 온 프로젝트 모음입니다.

이 저장소는 실제 구현과 검증 가능한 실험 근거만 골라 만든 포트폴리오 사본입니다. 원본 프로젝트의 실행 경로와 연구 artifact는 변경하지 않았습니다.

## 1. CCTV LPR System

실제 CCTV 입력에서 번호판 검출, ROI 전처리, OCR, 시간축 그룹화와 결과 기록까지 연결한 Python GPU 파이프라인입니다.

- `gpu_pipeline.py`: 공통 GPU 처리 엔진과 비동기 OCR worker, event/temporal 처리, 계측을 구현합니다.
- `gpu_pipeline2.py`: 위 클래스를 상속해 Real822용 PP-OCRv5 연결과 opt-in 연구 통합을 추가합니다.
- `run_detection.py`: 명령행 실행 진입점입니다.

두 pipeline 파일은 한 시스템의 기반 구현과 확장 계층입니다. 자세한 내용은 [01_CCTV_LPR_System](01_CCTV_LPR_System/README.md)을 참고하십시오.

## 2. T2 Accuracy Research

실제 CCTV 번호판 후보의 특징 생성, cross-fit selector, 후보 재평가와 leakage-free 평가를 비교한 별도 연구 트랙입니다. 이 코드는 production inference라고 주장하지 않습니다.

검증 bundle의 frozen 결과는 T2 `541/820 = 65.98%`, C3 `481/820 = 58.66%`입니다. A4 `583/820`은 후보 material ceiling이며 자동 Top-1 정확도가 아닙니다.

## 3. C# Image Processing

.NET Framework 4.7.2 WinForms로 작성한 두 영상처리 프로젝트입니다.

- `da`: 삼각형/사다리꼴 퍼지 소속함수, 동적 α-cut, 블록 분할 이진화
- `ho_pi`: FAM 유사도, softmax 가중 복원, 퍼지 전처리, 마스크 및 잔차 결합

PPT의 향후 아이디어와 실제 코드 구현을 구분한 감사 결과는 [PORTFOLIO_AUDIT.md](PORTFOLIO_AUDIT.md)에 기록했습니다.

## Reproduction note

학습 weight, 원본 CCTV 데이터와 개인정보 가능성이 있는 영상은 포함하지 않았습니다. Python 파일은 원본 import 구조를 유지하므로 전체 저장소의 `pipeline/`, `ocr/`, `decision/` 모듈과 외부 model이 필요합니다. C# 프로젝트는 Windows의 .NET Framework 4.7.2 및 WinForms 개발 도구가 필요합니다.


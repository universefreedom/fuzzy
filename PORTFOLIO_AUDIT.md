# Portfolio Audit

## A. Main Pipeline

### gpu_pipeline.py

공통 `GPUPipeline`과 OCR queue worker를 정의합니다. 코드에는 CUDA frame/crop 처리, detector 결과 소비, ROI 및 OCR variant 생성, EasyOCR/FastPlate/ONNX Runtime 연결, queue priority와 backpressure, event·track history, CUDA event 계측이 있습니다.

### gpu_pipeline2.py

`import gpu_pipeline as _legacy` 후 `class GPUPipeline(_legacy.GPUPipeline)`으로 확장합니다. PP-OCRv5 GPU recognizer, physical Hangul geometry, frozen grouping 이후 OCR 결합, live dual-OCR shadow hook을 추가합니다. 독립 detector나 별도 전체 파이프라인이 아닙니다.

### 관계와 데이터 흐름

```text
video input
  -> frame decode and detector
  -> plate ROI and GPU preprocessing
  -> asynchronous OCR queues
  -> OCR history and temporal/event grouping
  -> optional gpu_pipeline2 PP-OCRv5 adapter
  -> result rows, logs and timing metrics
```

실행 entry point는 `run_detection.py:main()`입니다. 외부 dependency에는 NumPy, OpenCV, PyTorch/CUDA, ONNX Runtime GPU, detector/OCR package와 model weight가 포함됩니다.

현재 검증: 세 Python 파일은 `py_compile`을 통과했습니다. 전체 raw-video smoke는 model weight와 원본 입력이 없어 수행하지 않았습니다.

## B. T2 Lineage

### 관련 파일

`02_T2_Accuracy_Research/`의 5개 producer script와 `evaluation/METHOD_CARD.json`, `REPRO_RESULT.json`을 frozen reproduction bundle에서 복사했습니다.

### Baseline과 champion

- C3: 481/820, 58.66%
- T2: 541/820, 65.98%
- W→C 81, C→W 21, net +60
- A4 583/820은 후보 ceiling이며 자동 Top-1이 아님

T2는 group-disjoint 5-fold OOF selector 결과이며 main production pipeline으로 표기하지 않습니다. bundle 설명도 T2를 historical research-shadow integration으로 제한합니다.

### 실패 및 미승격

원본 저장소에는 IFAM, fuzzy, distance metric과 selector 변형이 다수 존재합니다. 포트폴리오에는 frozen T2 재현 bundle만 포함했습니다. 현재 별도 lattice 연구의 L0 Fold0는 VALID134에서 `U_new=0`으로 `STOP_NO_GAIN`이며 T2 champion을 대체하지 않습니다.

## C. C# Projects

### da

- 프로젝트: WinForms, .NET Framework 4.7.2
- 구현: 평균/삼각형/사다리꼴 퍼지 이진화, 블록 분할, 동적 α-cut
- PPT: 분할 및 동적 α 적용 사다리꼴 퍼지 이진화
- 일치 여부: 핵심 구조 일치. 정량 성능 근거는 없음.
- 포트폴리오 문장: “WinForms에서 사다리꼴 퍼지 소속함수와 동적 α-cut을 구현하고, 영역 분할에 따른 이진화 결과를 비교했습니다.”

### ho_pi

- 프로젝트: WinForms, .NET Framework 4.7.2
- 구현: FAM similarity, softmax weighting, fuzzy preprocessing, mask reconstruction, residual blending
- PPT: IFAM에 잔차 연결, softmax 유사도, 동적 마스크와 퍼지 전처리 적용
- 일치 여부: 핵심 복원 흐름 일치. 유전자 알고리즘·자동 압축·규칙 기반 학습 조절은 발표의 개선 아이디어이며 코드에 없음.
- 포트폴리오 문장: “손상 영상 복원을 위해 FAM 유사도를 softmax로 가중하고, 퍼지 전처리와 마스크·잔차 결합을 WinForms 애플리케이션으로 구현했습니다.”

## D. GitHub Safety

- 제외: CCTV 원본, label, model weight, 대형 artifact, `bin/`, `obj/`, `.vs/`, cache
- 민감정보 검사: 복사한 소스에서 API key, password, 사용자 절대경로 패턴을 발견하지 못함
- 원본 보존: 기존 source와 artifact를 이동하거나 수정하지 않고 별도 사본만 생성
- 외부 데이터: detector/OCR weight, 입력 영상, 전체 원본 package 구조가 필요
- 실행 제약: Python 사본은 문법 검증 완료, end-to-end는 외부 weight 부재로 미실행. C# 두 solution은 실제 build PASS 후 clean을 수행함.

## PORTFOLIO_REPO_CHECK

```text
Main gpu_pipeline pair preserved: PASS
T2 lineage separated: PASS
C# source verified against PPT: PASS
No invented project claims: PASS
No secrets/private paths: PASS
Build artifacts excluded: PASS
README complete: PASS
Executable structure preserved: PASS_WITH_EXTERNAL_DEPENDENCIES
```

`Executable structure preserved`는 source topology를 보존했다는 의미입니다. 포함하지 않은 model/data와 전체 원본 import tree 없이는 독립 실행되지 않습니다.


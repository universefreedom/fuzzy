# CCTV LPR System

## 역할과 관계

`gpu_pipeline.py`와 `gpu_pipeline2.py`는 하나의 CCTV 번호판 처리 시스템입니다. `gpu_pipeline2.GPUPipeline`은 `gpu_pipeline.GPUPipeline`을 상속하며, 기존 public API를 그대로 다시 export합니다.

```text
run_detection.py
  -> gpu_pipeline2.GPUPipeline
       -> gpu_pipeline.GPUPipeline
            video/frame input
            detector and plate ROI
            GPU preprocessing
            async OCR queues
            temporal/event grouping
            logging and metrics
       -> optional PP-OCRv5 and Real822 adapters
```

## 실제 코드에서 확인한 책임

### gpu_pipeline.py

- CUDA availability를 확인하고 GPU-resident frame과 crop 처리를 관리합니다.
- detector 결과에서 번호판 ROI와 OCR crop을 생성합니다.
- EasyOCR, FastPlate ONNX Runtime CUDA, middle-slot recognizer 연결을 제공합니다.
- 우선순위 queue, batch flush, backpressure, 중복 억제를 포함한 비동기 OCR worker를 구현합니다.
- track/event 단위 history와 temporal candidate 상태를 유지합니다.
- CUDA event 기반 단계별 시간 측정과 OCR queue 지표를 기록합니다.

### gpu_pipeline2.py

- legacy `GPUPipeline`을 상속하는 opt-in 확장 계층입니다.
- OCR 또는 GT를 읽지 않는 물리적 한글 slot neutralization을 제공합니다.
- frozen legacy grouping 이후 PP-OCRv5 결과를 결합합니다.
- PP-OCRv5 GPU 실행과 fixed-slot plate 조립을 지원합니다.
- 기본 동작을 바꾸지 않는 live dual-OCR shadow hook을 제공합니다.

## 실행 진입점

```powershell
python run_detection.py --help
```

실제 실행에는 전체 원본 저장소 모듈, CUDA 환경, detector/OCR weight와 입력 영상이 필요합니다. 이 사본만으로 end-to-end 실행을 보장하지 않습니다.

## Dependencies

주요 dependency 범주는 `requirements.txt`에 정리했습니다. 실제 버전 pin은 원본 환경 receipt를 따라야 합니다.


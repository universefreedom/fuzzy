# Third-Party Software Notices

This repository declares or imports the components below. Their copyrights remain with the respective authors. License links point to upstream authority; installed versions are not pinned in `01_CCTV_LPR_System/requirements.txt`, so version-specific terms still need verification before distribution or deployment.

| Component | Official upstream / license evidence | Role in this snapshot | Upstream license | Source bundled here? |
|---|---|---|---|---|
| NumPy | [numpy/numpy LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Array operations in Python pipelines and T2 scripts | BSD 3-Clause-style | No |
| OpenCV / `opencv-python` | [opencv/opencv LICENSE, 4.x](https://github.com/opencv/opencv/blob/4.x/LICENSE) | Image processing | Apache-2.0 for current 4.x; installed version unpinned | No |
| PyTorch / `torch` | [pytorch/pytorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) | GPU tensor execution | BSD 3-Clause-style with additional notices | No |
| torchvision | [pytorch/vision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE) | Imported by `gpu_pipeline.py`; not listed separately in requirements | BSD-3-Clause | No |
| ONNX Runtime / `onnxruntime-gpu` | [microsoft/onnxruntime LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | ONNX OCR runtime | MIT | No |
| Ultralytics | [ultralytics license page](https://www.ultralytics.com/license), [upstream LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | Detector dependency in requirements and runner | AGPL-3.0 or separate Enterprise license | No package source found in this snapshot; model provenance unresolved |
| EasyOCR | [JaidedAI/EasyOCR LICENSE](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE) | OCR dependency/integration | Apache-2.0 | No |
| PaddleOCR | [PaddlePaddle/PaddleOCR LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE) | OCR dependency; PP-OCRv5 adapter | Apache-2.0 | No |
| scikit-learn | [scikit-learn COPYING](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | T2 selector scripts | BSD-3-Clause | No |
| LightGBM | [lightgbm-org/LightGBM LICENSE](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) | T2 ranking experiment | MIT | No |
| FastPlateOCR / `fast-plate-ocr` | [ankandrew/fast-plate-ocr package metadata](https://github.com/ankandrew/fast-plate-ocr/blob/master/pyproject.toml) | Custom FastPlate-compatible ONNX adapter in `gpu_pipeline.py` | MIT for that upstream package | No package source; exact custom ONNX model origin `NEEDS_SOURCE_CONFIRMATION` |

The upstream packages are not vendored in this repository. Users must comply with the applicable upstream licenses. This statement does **not** establish that any separately obtained model weights can be redistributed; this snapshot contains no detector/OCR weights.

Upstream notices name the NumPy Developers; Microsoft Corporation for ONNX Runtime; PaddlePaddle Authors for PaddleOCR; Microsoft Corporation and LightGBM developers for LightGBM; and multiple contributors in PyTorch's license. EasyOCR, OpenCV, Ultralytics, torchvision, scikit-learn and FastPlateOCR copyright details should be taken from the exact distributed package/version, not inferred from a repository name.

## License review required

- `LICENSE_REVIEW_REQUIRED`: Ultralytics' AGPL-3.0/Enterprise terms require a separate review of the actual use, detector weights, distribution and any combined work. A dependency declaration alone does not establish that source was copied here, nor does it settle the obligations for this project. No compatibility conclusion is made here.
- `NEEDS_SOURCE_CONFIRMATION`: exact installed versions, detector/OCR model weight sources and licenses, and the custom FastPlate ONNX model lineage.
- No root `LICENSE` or copied upstream license text was created by this audit. If a later source comparison finds copied or modified third-party code, retain the notices, license text and modification information required by that source before publishing.

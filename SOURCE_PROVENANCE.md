# Source Provenance

Scope: the tracked portfolio snapshot, not the full private research tree. “Project implementation” describes what the files implement; it is **not** a verified claim of sole authorship or clean-room origin. Text search cannot prove that no external code was copied. Unresolved source ancestry is marked `NEEDS_SOURCE_CONFIRMATION`.

## 01_CCTV_LPR_System

| File | Classification and role | External boundary | Open provenance question |
|---|---|---|---|
| `gpu_pipeline.py` | Mixed project integration: GPU preprocessing, OCR adapters, queues, temporal/event logic | Imports OpenCV, NumPy, PyTorch, torchvision and internal modules omitted from the portfolio; calls OCR runtimes | `NEEDS_SOURCE_CONFIRMATION`: compare against the original project and any external examples; identify detector/OCR model assets and custom FastPlate ONNX origin. |
| `gpu_pipeline2.py` | Project extension of `gpu_pipeline.GPUPipeline`; PP-OCRv5 and opt-in research integration | Uses the same internal and external runtime boundary | `NEEDS_SOURCE_CONFIRMATION`: exact PP-OCRv5 package/model version and any adapted snippets. |
| `run_detection.py` | Command-line entry point and detector runner | Ultralytics dependency declared in requirements | `LICENSE_REVIEW_REQUIRED`: exact YOLO model/weights and deployment/distribution conditions. |
| `requirements.txt` | Dependency declaration, not vendored source | Seven unpinned packages | Version and transitive/model licenses remain to be checked. |

## 02_T2_Accuracy_Research

The five included producer/evaluation scripts are a separate experimental snapshot, not the production OCR system. `build_c3_a4_selector_features.py` constructs frozen candidate features; `run_c3_a4_selector_crossfit_pre_gt.py` and `run_real822v3_c3_a4_crossfit_selector_v1.py` implement cross-fit selection; `run_real822_cross_group_confusion_transfer_v1.py` and `score_real822v3_c3_a4_selector_postfreeze.py` handle transfer/scoring. Their source imports are Python standard library plus NumPy, scikit-learn and LightGBM where used. No copied third-party source block or explicit upstream code attribution was identified by the repository text audit; that is not a definitive originality finding. `NEEDS_SOURCE_CONFIRMATION`: compare the original research tree and notebooks if a clean-room authorship claim is needed.

## 03_CSharp_ImageProcessing

- `da/da/Form1.cs`: WinForms coursework implementation of fuzzy membership, alpha-cut and image binarization. Coursework implementation based on course materials. Original bibliographic source not yet confirmed.
- `ho_pi/ho_pi/Form1.cs`: WinForms coursework implementation of FAM-like similarity, softmax weighting and reconstruction. Coursework implementation based on course materials. Original bibliographic source not yet confirmed.
- `*.Designer.cs`, `Properties/*`, `*.resx`: Visual Studio / WinForms-generated project scaffolding and resources; not claimed as original algorithmic code. No embedded external photograph was identified in the tracked resource templates.

No evidence in this snapshot is sufficient to name a copied/adapted external source file. Do not turn “not found” into “none exists”: inspect course materials, old project history and the original full repository before assigning an unqualified original-work or license claim.

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

## 04_V6_75p37_Latest_Audit

This directory is an `AUDIT_SUMMARY_ONLY` publication derived from a certified fresh standalone replay. It records counts, claim boundaries, and cryptographic result identifiers; it does not publish the private candidate registry, predictions, GT, model weights, datasets, or absolute execution paths. Qwen3-VL with fold-local LoRA is one candidate-string source, not the final scorer. The reproducible authority is the frozen V6 decision/evaluation lineage at `618/820 = 75.37%`.

## 03_CSharp_ImageProcessing

- `da/da/Form1.cs`: WinForms coursework implementation of fuzzy membership, alpha-cut and image binarization. The reviewed course bibliography names the 2016 Woo–Kim dynamic alpha-cut paper recorded in `REFERENCES.md`; this is a concept-reference boundary, not proof that every code line derives from that paper.
- `ho_pi/ho_pi/Form1.cs`: WinForms coursework implementation of FAM-like similarity, softmax weighting and reconstruction. The reviewed course bibliography includes the 2020 Silla University IFAM conference paper recorded in `REFERENCES.md`; exact line-level ancestry remains `NEEDS_SOURCE_CONFIRMATION`.
- `*.Designer.cs`, `Properties/*`, `*.resx`: Visual Studio / WinForms-generated project scaffolding and resources; not claimed as original algorithmic code. No embedded external photograph was identified in the tracked resource templates.

## Course material and public-asset boundary

The reviewed PPT extracts are private/course evidence, not redistributable public assets. They are not tracked in this repository. Their bibliographies may support a formal citation after authority verification, but their slide design, screenshots, paper figures, and embedded example images remain `PUBLICATION_PERMISSION_UNCONFIRMED` and outside this GitHub audit.

No raw CCTV, license-plate crop, face image, location-bearing frame, ground-truth file, course PPT, or paper PDF is tracked in the public Git snapshot or appears under a media/document path in the reviewed Git history. This finding is limited to the audited repository and does not reclassify private research archives.

No evidence in this snapshot is sufficient to name a copied/adapted external source file. Do not turn “not found” into “none exists”: inspect course materials, old project history and the original full repository before assigning an unqualified original-work or license claim.

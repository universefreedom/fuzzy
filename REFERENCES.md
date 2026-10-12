# Research References

This portfolio contains coursework, experimental research, and engineering integrations. A technique named in code is not, by itself, evidence that a specific paper was used. A citation does not imply endorsement by its authors.

## CCTV LPR System

| ID | Technique visible in this snapshot | Usage type | Project location | Bibliographic status |
|---|---|---|---|---|
| R1 | YOLO-based detection | External model/tool integration | `01_CCTV_LPR_System/run_detection.py`, `gpu_pipeline.py` | `NEEDS_SOURCE_CONFIRMATION`: exact detector model, weights, version, training provenance and any cited paper are absent from this portfolio snapshot. |
| R2 | OCR variants, temporal grouping and fuzzy evidence | Engineering integration / reimplementation concepts | `01_CCTV_LPR_System/gpu_pipeline.py`, `gpu_pipeline2.py` | `NEEDS_SOURCE_CONFIRMATION`: original research citations, if any, are not identified in the included files. |

## T2 Accuracy Research

| ID | Technique visible in this snapshot | Usage type | Project location | Bibliographic status |
|---|---|---|---|---|
| R3 | Cross-fit, out-of-fold evaluation and candidate re-ranking | Evaluation methodology / project implementation | `02_T2_Accuracy_Research/*.py` | `NEEDS_SOURCE_CONFIRMATION`: no particular paper, author, venue or DOI is named in the included source or project README. |

## Reproducible V6 Audit Snapshot

| ID | Technique visible in the public audit | Usage type | Project location | Bibliographic status |
|---|---|---|---|---|
| R6 | Qwen3-VL with fold-local LoRA candidate generation | External model/tool integration; candidate source only | `04_V6_75p37_Latest_Audit/` | Upstream project: [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). Exact model card and weight license must be checked before obtaining weights. No weights are included. |
| R7 | Frozen provenance registry, parent-protected Top-5 and freeze-before-GT selector replay | Project evaluation methodology | `04_V6_75p37_Latest_Audit/` | Public material is an audit summary and does not assign an external paper lineage without matching evidence. |

## C# Image Processing

| ID | Technique visible in this snapshot | Usage type | Project location | Bibliographic status |
|---|---|---|---|---|
| R4 | Triangular/trapezoidal fuzzy membership and dynamic alpha-cut | Coursework implementation | `03_CSharp_ImageProcessing/da/da/Form1.cs` | Coursework implementation based on course materials. Original bibliographic source not yet confirmed. |
| R5 | FAM-style similarity, softmax weighting and residual reconstruction | Coursework implementation | `03_CSharp_ImageProcessing/ho_pi/ho_pi/Form1.cs` | Coursework implementation based on course materials. Original bibliographic source not yet confirmed. |

No paper title, authors, year, venue or DOI has been assigned without a matching source in the reviewed snapshot. The original slides, reports and course bibliography must be checked before converting any row into a formal citation.

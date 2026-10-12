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
| R4 | Triangular/trapezoidal fuzzy membership and dynamic alpha-cut | Coursework implementation / concept reference | `03_CSharp_ImageProcessing/da/da/Form1.cs` | 우현수·김광백, 「사다리꼴 형태의 소속 함수와 동적 α_cut을 이용한 개선된 퍼지 이진화」, 한국정보통신학회논문지 20(10), 1852–1859, 2016. KCI 권호 및 학술 데이터베이스에서 제목·저자·권호·쪽을 교차확인했습니다. 논문 인용은 강의 PPT나 그림의 재배포 허가를 의미하지 않습니다. |
| R5 | FAM/IFAM-style image restoration | Coursework implementation / concept reference | `03_CSharp_ImageProcessing/ho_pi/ho_pi/Form1.cs` | 이효진·고예림·강민지·권희영·김광백, 「IFAM을 이용한 훼손된 그레이 영상 복원」, 한국정보통신학회 종합학술대회 논문집, 2020. 학회 공개 PDF와 목차에서 제목·저자·신라대학교 소속을 확인했습니다. 현재 코드의 세부 구현 계보 전체를 이 논문과 동일하다고 단정하지 않습니다. |

위 두 문헌은 실제 수업 PPT의 인용 목록과 공개 학술 authority가 일치한 범위에서만 연결했습니다. 그 밖의 논문·강의자료·이미지 출처는 `NEEDS_SOURCE_CONFIRMATION`이며, 인용 사실을 PPT 원본·논문 그림·예제 이미지의 재배포 허가로 해석하지 않습니다.

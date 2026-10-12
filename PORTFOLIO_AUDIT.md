# Portfolio Audit

## Reproducible V6 authority

- Scope: retrospective GT0 Nested Outer-5 development benchmark
- V1 parent: `600/820 = 73.17%`
- V6: `618/820 = 75.37%`
- Switch audit: `65 = 25 W→C + 7 C→W + 33 W→W`
- Net gain: `25 - 7 = 18`; therefore `600 + 18 = 618`
- Fresh standalone replay: PASS
- Frozen prediction SHA-256: `E9502521DB1FFC315DB6DA9DB86FF9B54970379CF45B6A028F006EA0B433CC64`
- Final result SHA-256: `317F1371624CE95696A5ABCBDBACB452A15AD2F81E57E1145C648C054613203D`

This is not blind/unseen/external performance. It is not production accuracy, an official competition score, or proof of real-world generalization.

## Frozen source audit

The existing production-style system, historical T2 research, and C# projects are unchanged from the pre-publication SHA manifest. The static verifier checks every frozen file under:

- `01_CCTV_LPR_System/`
- `02_T2_Accuracy_Research/`
- `03_CSharp_ImageProcessing/`

The production call chain remains `run_detection.py → gpu_pipeline2.GPUPipeline → gpu_pipeline.GPUPipeline`.

## Historical lineage

- T2 remains a separate historical research snapshot: `541/820 = 65.98%`.
- The 2026-10-08 `622/820 = 75.85%` Core-Light record remains a separate historical `PARTIAL_REPRODUCTION` snapshot, not the reproducible V6 headline.

## Publication safety

- No model weights, raw CCTV, crops, GT, credentials, or private paths are published.
- No root license is asserted.
- Ultralytics remains `LICENSE_REVIEW_REQUIRED`.
- Qwen3-VL is described only as a candidate generator/source; it is not the final scorer.
- Public V6 material is an `AUDIT_SUMMARY_ONLY` reproduction record.

Run `python tools/verify_portfolio_static.py` before publication. Any failed gate blocks commit and push.


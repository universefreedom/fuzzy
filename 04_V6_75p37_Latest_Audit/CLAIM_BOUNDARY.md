# 주장 범위

이 결과는 동결된 연구 artifact와 config를 재생한 **회고적 GT0 Nested Outer-5 개발 평가**입니다.

V6의 최종 prediction과 config는 평가 전에 동결했고, freeze 이전 GT read는 `0`, freeze 이후 exact-once GT 결합은 `820`개입니다. 다만 동일 GT0가 이전 연구 과정에서 반복적으로 사용되었으므로 독립적인 외부 일반화 평가가 아닙니다.

따라서 blind, unseen, external benchmark, production 정확도, SOTA, 공식 대회 점수 또는 real-world generalization이 입증되었다고 주장하지 않습니다.

Qwen3-VL + fold-local LoRA는 candidate source 중 하나로 문자열 후보를 제공합니다. 최종 scorer가 아니며 `75.37%`는 Qwen 단독 인식 정확도가 아닙니다.

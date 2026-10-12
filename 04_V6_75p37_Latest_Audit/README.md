# 재현 가능한 V6 연구 결과

현재 재현 기준 authority는 **618/820 = 75.37%**입니다.

- 직접 비교 기준 V1: **600/820 = 73.17%**
- 변경 출력: `65개`
- 오답 → 정답: `25개`
- 정답 → 오답: `7개`
- 오답 → 다른 오답: `33개`
- 순개선: `+18개`

이 디렉터리는 공개 가능한 감사 요약만 제공합니다. 동결된 연구 artifact와 config에서 시작해 최종 selector와 평가 경로를 재생하며, 원본 CCTV부터 재학습하거나 full pixel-to-final inference를 다시 수행했다고 주장하지 않습니다.

> 이 결과는 **회고적 GT0 Nested Outer-5 개발 평가**입니다. blind / unseen / external benchmark / production 정확도 / 공식 대회 성능을 의미하지 않습니다.

- [지표 authority](METRIC_AUTHORITY.json)
- [replay receipt 요약](REPLAY_RECEIPT.json)
- [재현 범위](REPRODUCTION_NOTES.md)
- [주장 범위](CLAIM_BOUNDARY.md)
- [평가 결과](evaluation/REPRO_RESULT.json)

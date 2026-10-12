# V6 618 재현 범위

## 동결된 목표

- V1: `600/820 = 73.17%`
- V6: `618/820 = 75.37%`
- W→C: `25`
- C→W: `7`
- W→W: `33`
- 순개선: `+18`

## 재현 대상

fresh standalone extraction에서 동결된 연구 artifact/configuration을 읽고 최종 decision/evaluation 경로를 재생했습니다. 원본 CCTV부터의 재학습, private candidate evidence 재생성, full pixel-to-final inference를 수행했다는 의미는 아닙니다.

공개 저장소에는 `AUDIT_SUMMARY_ONLY` receipt만 포함합니다. model weight, dataset, raw GT, prediction, credential, private filesystem path는 공개하지 않습니다.

## 평가 해석

이 결과는 회고적 GT0 Nested Outer-5 개발 평가입니다. blind, unseen, external benchmark, production 정확도 또는 공식 대회 성능으로 해석하지 않습니다.

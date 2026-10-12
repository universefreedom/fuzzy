# 공개 포트폴리오 개인정보·자산 감사

## 1. Audit Scope

감사 범위는 공개 GitHub에 올라가는 Git tracked snapshot과 전체 Git object path 이력입니다. 로컬 PPT·DOCX·PDF·추출 이미지·private 연구 archive는 GitHub에 올라가지 않으므로 최종 privacy gate와 자산 수 집계에서 제외합니다. 원본 연구자료와 문서 바이너리는 변경하지 않았습니다.

개인정보, 저작권·이용허락, 학술 출처, 공개 재배포 권한을 서로 독립적으로 판정했습니다. 비식별화는 저작권 허가를 대신하지 않습니다.

| 파일/범위 | 자료 유형 | 실제 CCTV | 번호판 | 얼굴 | 위치정보 | 외부저작물 | 출처 확인 | 공개판 필요 | 조치 |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| 공개 Git tracked snapshot | 코드·Markdown·JSON | 아니오 | 아니오 | 아니오 | 아니오 | 독립 미디어 없음 | Git index 확인 | 아니오 | KEEP |
| 전체 Git object path 이력 | Git 경로 이력 | 아니오 | 아니오 | 아니오 | 아니오 | 미디어/문서 경로 없음 | `git rev-list --objects --all` 확인 | 아니오 | KEEP |

## 2. Real CCTV Assets

- 공개 Git 현재 상태: 실제 CCTV 이미지·영상 `0`.
- 전체 Git object path 이력: 감사 대상 확장자에 해당하는 이미지·영상·PPT·PDF·Word `0`.
- 이 결과는 공개 Git 저장소에 한정하며, private 연구 archive에 원자료가 없다는 뜻이 아닙니다.

## 3. License Plate Exposure

검사한 공개 파일과 채용 문서에서 실제 번호판 이미지, OCR prediction, GT 문자열, 번호판형 문자열을 찾지 못했습니다. 공개 Git에 GT CSV/TSV나 독립 JSON dataset도 없습니다. 이후 실제 crop을 추가하려면 synthetic 우선, 강한 마스킹, OCR/GT 문자열 동시 마스킹, 시각 검토를 모두 거쳐야 합니다.

## 4. Face / Location Exposure

Git tracked 파일과 Git 이력에는 raster/video 자산이 없으므로 사람 얼굴·도로 주소·건물·GPS·카메라 ID·EXIF 노출도 없습니다. 로컬 산출물은 이 판정에 포함하지 않습니다.

## 5. Kim Kwangbaek / Silla University Materials

수업 PPT 인용 목록과 공개 학술 authority가 일치하는 문헌 두 건을 확인했습니다.

1. 우현수·김광백 (2016), 「사다리꼴 형태의 소속 함수와 동적 α_cut을 이용한 개선된 퍼지 이진화」.
2. 이효진·고예림·강민지·권희영·김광백 (2020), 「IFAM을 이용한 훼손된 그레이 영상 복원」.

이 확인은 bibliographic citation을 허용하는 근거이지, 강의 PPT·논문 PDF·그림·예제 이미지의 공개 재배포 허가가 아닙니다. 강의자료 자체는 `COURSE_MATERIAL`, 공개 권한은 `PUBLICATION_PERMISSION_UNCONFIRMED`입니다.

## 6. Copyright & Permission Boundary

- 공개 논문 인용과 논문/PPT 이미지 재사용 권한은 별개입니다.
- 원본 연구자료와 PPT는 수정·삭제하지 않았습니다.
- 로컬 외부 이미지는 이번 GitHub 감사의 대상이 아니며 Git에 추가하지 않았습니다.

## 7. Public-safe Replacements

이번 감사에서 새 공개용 이미지는 만들지 않았습니다. 공개 Git은 이미 Mermaid와 텍스트 중심이며 외부 사진을 필요로 하지 않습니다.

## 8. Remaining Open Items

- 수업 PPT 원본의 공개 재배포 권한은 확인되지 않았습니다.
- 과거 OCR OFF / ON FPS 값은 현재 공개 코드와 보존된 공개 로그에서 독립 검증되지 않았습니다. 공개 README에서는 삭제했으며, 동일 입력·환경의 `wall_clock_effective_fps` receipt를 확보하기 전까지 확정 수치로 재사용하지 않습니다.
- 현재 감사 결과와 PASS 판정은 공개 GitHub tracked 내용에만 적용됩니다.

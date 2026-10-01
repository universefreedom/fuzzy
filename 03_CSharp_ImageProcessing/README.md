# C# Image Processing

두 프로젝트는 Visual Studio의 .NET Framework 4.7.2 WinForms 애플리케이션입니다. `bin/`, `obj/`, `.vs/`는 포함하지 않았습니다.

## da

실제 `Form1.cs`에서 확인한 기능:

- 영상 열기와 BMP 저장
- 평균 임계값 이진화
- 삼각형 퍼지 소속함수 기반 이진화
- 사다리꼴 퍼지 이진화
- 이미지 블록 분할과 동적 α-cut
- 이진화 픽셀 비율에 따른 반복 종료

PPT의 분할·동적 α-cut 설명은 코드와 일치합니다. 다만 정량 성능이나 일반화 주장은 별도 평가 근거가 없습니다.

## ho_pi

실제 `Form1.cs`에서 확인한 기능:

- 입력/메모리 패턴 유사도 계산
- softmax 가중 FAM 복원
- `alpha=0.7` 잔차 결합
- 손실 영역 마스크 기반 재구성
- 밝기 조건 퍼지 스트레칭과 동적 α-cut 전처리
- 2D triangular membership 기반 FAM recall

PPT의 softmax 유사도, 잔차 연결, 동적 마스크와 퍼지 전처리는 코드와 일치합니다. PPT의 유전자 알고리즘, 자동 압축 저장, 규칙 기반 학습 조절은 코드에서 확인되지 않았으므로 향후 개선 아이디어로 분류합니다.

## Build

Windows의 .NET SDK 10 MSBuild 호환 경로에서 두 `.sln`을 실제로 빌드했습니다. `da`는 미사용 필드 경고 1건과 오류 0건, `ho_pi`는 경고·오류 0건으로 통과했습니다. 최종 사본에는 build 출력물을 남기지 않았습니다.


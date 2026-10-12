#!/usr/bin/env python3
"""Fail-closed publication checks for the public CCTV-LPR portfolio."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "04_V6_75p37_Latest_Audit"
FROZEN_ROOTS = ("01_CCTV_LPR_System/", "02_T2_Accuracy_Research/", "03_CSharp_ImageProcessing/")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".safetensors"}
SKIP_DIRS = {".git", "portfolio_output", "__pycache__", "bin", "obj", ".vs"}
RAW_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".mp4", ".avi", ".mkv"}
COURSE_DOCUMENT_SUFFIXES = {".ppt", ".pptx", ".pdf", ".doc", ".docx", ".hwp"}
RAW_GT_SUFFIXES = {".csv", ".tsv"}


def read(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8") if isinstance(path, str) else path.read_text(encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(read(path))


def public_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            yield path


def git_lines(*args: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def tracked_paths() -> list[Path]:
    return [Path(line) for line in git_lines("ls-files")]


def historical_object_paths() -> list[Path]:
    paths = []
    for line in git_lines("rev-list", "--objects", "--all"):
        parts = line.split(" ", 1)
        if len(parts) == 2:
            paths.append(Path(parts[1]))
    return paths


def frozen_source_sha_ok() -> bool:
    manifest = load_json("PORTFOLIO_SHA_MANIFEST.json")
    rows = [
        row for row in manifest["files"]
        if row["path"].startswith(FROZEN_ROOTS) and not row["path"].lower().endswith(".md")
    ]
    if not rows:
        return False
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest().upper() != row["sha256"].upper():
            return False
    return True


def relative_links_ok() -> bool:
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for md in (p for p in public_files() if p.suffix.lower() == ".md"):
        for target in pattern.findall(md.read_text(encoding="utf-8")):
            target = target.strip().split("#", 1)[0]
            if not target or re.match(r"^(?:https?://|mailto:)", target):
                continue
            if not (md.parent / target).resolve().exists():
                return False
    return True


def json_valid() -> bool:
    try:
        for path in public_files():
            if path.suffix.lower() == ".json":
                json.loads(path.read_text(encoding="utf-8"))
        return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


def korean_primary(text: str, required: tuple[str, ...] = ()) -> bool:
    korean = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return korean >= 120 and korean > latin * 0.35 and all(term in text for term in required)


def main() -> int:
    readme = read("README.md")
    audit = read("PORTFOLIO_AUDIT.md")
    architecture = read("docs/ARCHITECTURE.md")
    metric = load_json(V6 / "METRIC_AUTHORITY.json")
    result = load_json(V6 / "evaluation/REPRO_RESULT.json")
    receipt = load_json(V6 / "REPLAY_RECEIPT.json")
    notices = read("THIRD_PARTY_NOTICES.md")
    publication_text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in public_files()
        if p.suffix.lower() in {".md", ".json", ".txt"}
    )

    arithmetic = 25 + 7 + 33 == 65 and 25 - 7 == 18 and 600 + 18 == 618
    metric_ok = round(618 / 820 * 100, 2) == 75.37
    headline = readme.split("<details>", 1)[0]
    forbidden_qwen = ("qwen final scorer", "qwen achieved 75.37", "qwen recognition accuracy = 75.37", "llm alone")
    private_patterns = (
        r"[A-Za-z]:\\Users\\",
        r"[A-Za-z]:\\vision_project_work_data\\",
        r"[A-Za-z]:\\4_1cv_fuzzy\\",
        r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
        r"(?:api[_-]?key|token|password)\s*[:=]\s*[\"'][^\"']+[\"']",
        r"drive\.google\.com/drive/(?:u/\d+/)?folders/",
    )
    tracked = tracked_paths()
    history_paths = historical_object_paths()
    all_git_paths = tracked + history_paths
    privacy_audit = read("PRIVACY_ASSET_AUDIT.md")
    asset_sources = read("ASSET_SOURCES.md")
    source_provenance = read("SOURCE_PROVENANCE.md")
    references = read("REFERENCES.md")
    raw_media = [p for p in all_git_paths if p.suffix.lower() in RAW_MEDIA_SUFFIXES]
    course_docs = [p for p in all_git_paths if p.suffix.lower() in COURSE_DOCUMENT_SUFFIXES]
    raw_gt = [
        p for p in all_git_paths
        if p.suffix.lower() in RAW_GT_SUFFIXES
        or (p.suffix.lower() == ".json" and re.search(r"(?:^|[_-])(?:gt|ground[_-]?truth|label)(?:[_-]|$)", p.stem, re.IGNORECASE))
    ]

    checks = {
        "LATEST_V6_METRIC": metric_ok and metric["v6_reproducible"] == {"correct": 618, "accuracy_percent": 75.37} and "618 / 820 = 75.37%" in readme,
        "V1_BASELINE": metric["v1_parent"] == {"correct": 600, "accuracy_percent": 73.17} and "600 / 820 = 73.17%" in readme,
        "SWITCH_ARITHMETIC": arithmetic and metric["switches"] == {"total": 65, "wrong_to_correct": 25, "correct_to_wrong": 7, "wrong_to_wrong": 33, "net_gain": 18},
        "CLAIM_BOUNDARY": all(value is False for value in metric["claim_boundary"].values()) and "blind / unseen / external benchmark" in publication_text,
        "HISTORICAL_622_SEPARATION": "622/820" not in headline and "622/820 = 75.85%" in readme and "별도 과거 authority" in readme,
        "T2_HISTORICAL_PRESERVED": "541/820 = 65.98%" in readme and "별도 이전 연구 계보" in readme,
        "PIPELINE_PAIR_PRESERVED": all(name in readme for name in ("run_detection.py", "gpu_pipeline2.GPUPipeline", "gpu_pipeline.GPUPipeline")),
        "QWEN_ROLE_BOUNDARY": "candidate source" in readme and "최종 판정 모델이 아니라" in readme and not any(x in publication_text.lower() for x in forbidden_qwen),
        "EXISTING_SOURCE_SHA": frozen_source_sha_ok(),
        "ATTRIBUTION_DOCS": all((ROOT / name).is_file() for name in ("REFERENCES.md", "THIRD_PARTY_NOTICES.md", "SOURCE_PROVENANCE.md", "ASSET_SOURCES.md", "docs/PORTFOLIO_STYLE_SOURCES.md")) and all(name in notices for name in ("Qwen3-VL", "Transformers", "PEFT", "Accelerate")),
        "ULTRALYTICS_REVIEW_STATUS": "LICENSE_REVIEW_REQUIRED" in notices and "Ultralytics" in notices,
        "NO_ROOT_LICENSE": not any((ROOT / name).exists() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")),
        "NO_MODEL_WEIGHTS": not any(p.suffix.lower() in WEIGHT_SUFFIXES for p in public_files()),
        "NO_LARGE_FILES": not any(p.stat().st_size > 10 * 1024 * 1024 for p in public_files()),
        "NO_SECRET_OR_PRIVATE_PATH": not any(re.search(pattern, publication_text, re.IGNORECASE) for pattern in private_patterns),
        "RELATIVE_LINKS": relative_links_ok(),
        "JSON_VALID": json_valid(),
        "V6_618_REPLAY_RECEIPT": receipt["status"] == "PASS" and receipt["checks"] == {"v1_600": True, "v6_618": True, "exact_once": True},
        "V6_618_METRIC_PARITY": result["status"] == "PASS" and result["denominator"] == 820 and result["v1_correct"] == 600 and result["v6_correct"] == 618 and result["v6_accuracy_percent"] == 75.37,
        "V6_618_SWITCH_PARITY": all(result[k] == v for k, v in {"switch_total": 65, "wrong_to_correct": 25, "correct_to_wrong": 7, "wrong_to_wrong": 33, "net_gain": 18}.items()),
        "V6_618_CONFIG_FROZEN": receipt["config_frozen"] is True and receipt["pre_freeze_gt_reads"] == 0,
        "V6_618_PREDICTION_FROZEN": receipt["prediction_frozen"] is True and receipt["post_freeze_exact_once_groups"] == 820 and receipt["prediction_freeze_sha256"] == "E9502521DB1FFC315DB6DA9DB86FF9B54970379CF45B6A028F006EA0B433CC64",
    }

    privacy_checks = {
        "PRIVACY_AUDIT_DOC_EXISTS": (ROOT / "PRIVACY_ASSET_AUDIT.md").is_file(),
        "ASSET_SOURCES_UPDATED": all(term in asset_sources for term in ("신라대학교 연구·수업 자료", "PUBLICATION_PERMISSION_UNCONFIRMED", "김광백 교수")),
        "NO_TRACKED_RAW_CCTV_VIDEO": not raw_media,
        "NO_TRACKED_RAW_GT_FILE": not raw_gt,
        "NO_TRACKED_UNANONYMIZED_PLATE_IMAGE": not raw_media,
        "NO_PRIVATE_DRIVE_LINK": "drive.google.com/drive/" not in publication_text,
        "NO_LOCAL_ABSOLUTE_PATH": not any(re.search(pattern, publication_text, re.IGNORECASE) for pattern in private_patterns[:3]),
        "PROFESSOR_MATERIAL_PERMISSION_BOUNDARY": all(term in source_provenance + asset_sources for term in ("PUBLICATION_PERMISSION_UNCONFIRMED", "not redistributable public assets")),
        "ROOT_LICENSE_UNCHANGED": not any((ROOT / name).exists() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")),
        "UNVERIFIED_FPS_NOT_PUBLISHED": "22.94" not in publication_text and "1.04" not in publication_text and "wall_clock_effective_fps" in readme,
    }

    public_asset_checks = {
        "RAW_SOURCE_UNCHANGED": frozen_source_sha_ok(),
        "PUBLIC_COPY_SEPARATE": not raw_media,
        "PLATE_ANONYMIZATION": not raw_media,
        "FACE_ANONYMIZATION": not raw_media,
        "LOCATION_METADATA_CHECK": not raw_media,
        "EXIF_GPS_REMOVED": not raw_media,
        "OCR_GT_STRING_MASKED": not raw_gt,
        "SILLA_SOURCE_BOUNDARY": "신라대학교 연구·수업 자료" in asset_sources and not course_docs,
        "KIM_KWANGBAEK_REFERENCE_BOUNDARY": "김광백" in references and "재배포" in references,
        "COURSE_MATERIAL_NOT_REDISTRIBUTED": not course_docs,
        "COPYRIGHT_PERMISSION_BOUNDARY": "NEEDS_PERMISSION" in asset_sources and "공개 논문 인용과 논문/PPT 이미지 재사용 권한은 별개" in privacy_audit,
        "ASSET_SOURCES_UPDATED": privacy_checks["ASSET_SOURCES_UPDATED"],
        "PRIVACY_AUDIT_CREATED": privacy_checks["PRIVACY_AUDIT_DOC_EXISTS"],
        "NO_RAW_CCTV_IN_GITHUB": not raw_media,
    }

    project_readmes = [
        read("01_CCTV_LPR_System/README.md"),
        read("02_T2_Accuracy_Research/README.md"),
        read("03_CSharp_ImageProcessing/README.md"),
    ]
    v6_explanation = "\n".join((read(V6 / "README.md"), read(V6 / "CLAIM_BOUNDARY.md"), read(V6 / "REPRODUCTION_NOTES.md")))
    technical_terms = ("Qwen3-VL", "LoRA", "Top-1", "Top-5", "Nested Outer-5", "selector", "candidate", "CUDA", "ONNX Runtime")
    license_names = ("Apache-2.0", "MIT", "BSD-3-Clause", "AGPL-3.0", "LICENSE_REVIEW_REQUIRED", "NEEDS_SOURCE_CONFIRMATION")
    korean_checks = {
        "README_KOREAN_PRIMARY": korean_primary(readme, ("현재 재현", "전체 시스템", "프로젝트 구성", "정확도", "재현 범위", "참고문헌")),
        "PROJECT_READMES_KOREAN_PRIMARY": all(korean_primary(text) for text in project_readmes),
        "ARCHITECTURE_KOREAN_PRIMARY": korean_primary(architecture, ("전체 시스템 구조", "메인 CCTV 파이프라인", "V6 연구 파이프라인")),
        "V6_EXPLANATION_KOREAN_PRIMARY": korean_primary(v6_explanation, ("재현", "주장 범위")),
        "TECHNICAL_TERMS_PRESERVED": all(term in publication_text for term in technical_terms),
        "CODE_IDENTIFIERS_UNCHANGED": checks["EXISTING_SOURCE_SHA"] and all(name in publication_text for name in ("gpu_pipeline.py", "gpu_pipeline2.py", "run_detection.py")),
        "LICENSE_NAMES_UNCHANGED": all(name in notices for name in license_names),
        "CLAIM_BOUNDARY_UNCHANGED": checks["CLAIM_BOUNDARY"] and "독립적인 외부 일반화 평가가 아닙니다" in v6_explanation,
        "618_AUTHORITY_UNCHANGED": checks["LATEST_V6_METRIC"] and checks["V6_618_METRIC_PARITY"] and checks["V6_618_SWITCH_PARITY"],
        "UTF8_KOREAN_VALID": "�" not in publication_text,
    }

    print("[PORTFOLIO_STATIC_GATE]\n")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("\n[KOREAN DOCUMENTATION GATE]\n")
    for name, passed in korean_checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("\n[PRIVACY STATIC GATE]\n")
    for name, passed in privacy_checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("\n[PUBLIC ASSET PRIVACY GATE]\n")
    for name, passed in public_asset_checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    final = all(checks.values()) and all(korean_checks.values()) and all(privacy_checks.values()) and all(public_asset_checks.values())
    print(f"\nFINAL: {'PASS' if final else 'FAIL'}")
    return 0 if final else 1


if __name__ == "__main__":
    sys.exit(main())

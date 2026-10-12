#!/usr/bin/env python3
"""Fail-closed publication checks for the public CCTV-LPR portfolio."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "04_V6_75p37_Latest_Audit"
FROZEN_ROOTS = ("01_CCTV_LPR_System/", "02_T2_Accuracy_Research/", "03_CSharp_ImageProcessing/")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".safetensors"}
SKIP_DIRS = {".git", "portfolio_output", "__pycache__", "bin", "obj", ".vs"}


def read(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8") if isinstance(path, str) else path.read_text(encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(read(path))


def public_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            yield path


def frozen_source_sha_ok() -> bool:
    manifest = load_json("PORTFOLIO_SHA_MANIFEST.json")
    rows = [row for row in manifest["files"] if row["path"].startswith(FROZEN_ROOTS)]
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
    )

    checks = {
        "LATEST_V6_METRIC": metric_ok and metric["v6_reproducible"] == {"correct": 618, "accuracy_percent": 75.37} and "618 / 820 = 75.37%" in readme,
        "V1_BASELINE": metric["v1_parent"] == {"correct": 600, "accuracy_percent": 73.17} and "600 / 820 = 73.17%" in readme,
        "SWITCH_ARITHMETIC": arithmetic and metric["switches"] == {"total": 65, "wrong_to_correct": 25, "correct_to_wrong": 7, "wrong_to_wrong": 33, "net_gain": 18},
        "CLAIM_BOUNDARY": all(value is False for value in metric["claim_boundary"].values()) and "This is not blind/unseen/external performance" in publication_text,
        "HISTORICAL_622_SEPARATION": "622/820" not in headline and "622/820 = 75.85%" in readme and "separate historical" in readme,
        "T2_HISTORICAL_PRESERVED": "541/820 = 65.98%" in readme and "Historical T2" in readme,
        "PIPELINE_PAIR_PRESERVED": all(name in readme for name in ("run_detection.py", "gpu_pipeline2.GPUPipeline", "gpu_pipeline.GPUPipeline")),
        "QWEN_ROLE_BOUNDARY": "candidate generator/source, not the final scorer" in readme and not any(x in publication_text.lower() for x in forbidden_qwen),
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

    print("[PORTFOLIO_STATIC_GATE]\n")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    final = all(checks.values())
    print(f"\nFINAL: {'PASS' if final else 'FAIL'}")
    return 0 if final else 1


if __name__ == "__main__":
    sys.exit(main())

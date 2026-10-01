#!/usr/bin/env python3
"""Build leakage-safe selector authority/features from frozen C3 A4 candidates.

This program performs no OCR, restoration, candidate generation, or GT reads.
Candidate identity is kept in 00_AUTHORITY and is never emitted as a model
feature. 01_FEATURES contains only an opaque row key and numeric evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "artifacts/real822v3_c3_ifam_ocr_aware_restoration_gate_v1"
OUT = ROOT / "artifacts/real822v3_c3_ifam_a4_crossfit_top1_selector_v1"
SPLIT = ROOT / "artifacts/real822v3_nested5_cluster_isolated_parallel_v2/00_split/OUTER5_FINAL_MANIFEST.csv"

FROZEN = [
    SOURCE_ROOT / "06_POSTTRAIN_PRE_GT_INFERENCE/PREDICTION_FREEZE_V1/A0_A4_CANDIDATES_PRE_GT.csv",
    SOURCE_ROOT / "11_OUTER1_POSTTRAIN_PRE_GT/PREDICTION_FREEZE_V1/A0_A4_CANDIDATES_PRE_GT.csv",
    SOURCE_ROOT / "12_OUTER2_POSTTRAIN_PRE_GT/PREDICTION_FREEZE_V1/A0_A4_CANDIDATES_PRE_GT.csv",
    SOURCE_ROOT / "13_OUTER3_POSTTRAIN_PRE_GT_FREEZE/freeze/A0_A4_CANDIDATES_PRE_GT.csv",
    SOURCE_ROOT / "13_OUTER4_POSTTRAIN_PRE_GT_FREEZE/freeze/A0_A4_CANDIDATES_PRE_GT.csv",
]

SOURCE_NAMES = ("BASE_C3", "OCR_RAW", "OCR_IFAM", "OCR_IFAM_SHARP")
FEATURES = (
    "is_c3_anchor",
    "present_raw",
    "present_ifam",
    "present_sharp",
    "view_source_count",
    "confidence_raw",
    "confidence_raw_missing",
    "confidence_ifam",
    "confidence_ifam_missing",
    "confidence_sharp",
    "confidence_sharp_missing",
    "confidence_max",
    "confidence_mean",
    "confidence_count",
    "c3_agreement",
    "raw_ifam_agree",
    "raw_sharp_agree",
    "ifam_sharp_agree",
    "string_length",
    "group_N",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    for path in [*FROZEN, SPLIT]:
        if not path.is_file():
            raise FileNotFoundError(path)

    split_rows = read_csv(SPLIT)
    split = {r["group_id"]: r for r in split_rows}
    if len(split) != 820 or sorted(int(r["outer_fold"]) for r in split_rows).count(0) != 164:
        raise RuntimeError("unexpected split authority")

    all_a4: list[dict[str, str]] = []
    source_receipts = []
    fold_groups: dict[int, set[str]] = {}
    for fold, path in enumerate(FROZEN):
        rows = read_csv(path)
        cols = list(rows[0]) if rows else []
        expected = ["arm_id", "group_id", "source_arm", "candidate_name", "prediction", "confidence", "source_image_sha256"]
        if cols != expected:
            raise RuntimeError(f"schema mismatch: {path}: {cols}")
        a4 = [r for r in rows if r["arm_id"] == "A4"]
        groups = {r["group_id"] for r in a4}
        if len(groups) != 164 or len(a4) != 656:
            raise RuntimeError(f"fold {fold}: expected 164 groups and 656 A4 rows")
        if any(int(split[g]["outer_fold"]) != fold for g in groups):
            raise RuntimeError(f"fold assignment mismatch: {fold}")
        fold_groups[fold] = groups
        all_a4.extend(a4)
        source_receipts.append({
            "outer_fold": fold,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
            "rows_total": len(rows),
            "rows_A4": len(a4),
            "groups": len(groups),
        })

    union = set().union(*fold_groups.values())
    pairwise_overlap = sum(len(fold_groups[a] & fold_groups[b]) for a in range(5) for b in range(a + 1, 5))
    if len(union) != 820 or pairwise_overlap:
        raise RuntimeError("fold disjointness/coverage failure")

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_a4:
        by_group[row["group_id"]].append(row)

    identity_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    source_sha_cross_group: dict[str, set[str]] = defaultdict(set)
    duplicate_source_rows = 0
    next_row_id = 0
    for gid in sorted(by_group):
        rows = by_group[gid]
        if {r["candidate_name"] for r in rows} != set(SOURCE_NAMES):
            raise RuntimeError(f"missing/extra source view: {gid}")
        source_to_row = {r["candidate_name"]: r for r in rows}
        if len(source_to_row) != len(rows):
            duplicate_source_rows += len(rows) - len(source_to_row)
        c3 = source_to_row["BASE_C3"]["prediction"]
        by_prediction: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_prediction[row["prediction"]].append(row)
            if row["source_image_sha256"]:
                source_sha_cross_group[row["source_image_sha256"]].add(gid)

        for ordinal, prediction in enumerate(sorted(by_prediction)):
            members = by_prediction[prediction]
            names = {m["candidate_name"] for m in members}
            # Opaque sequential join key: it carries neither group nor string identity.
            row_id = f"CAND_{next_row_id:07d}"
            next_row_id += 1
            conf = {}
            for name in ("OCR_RAW", "OCR_IFAM", "OCR_IFAM_SHARP"):
                vals = [float(m["confidence"]) for m in members if m["candidate_name"] == name and m["confidence"]]
                conf[name] = vals[0] if vals else None
            conf_values = [v for v in conf.values() if v is not None]
            fold = int(split[gid]["outer_fold"])
            identity_rows.append({
                "candidate_row_id": row_id,
                "outer_fold": fold,
                "group_id": gid,
                "candidate_string": prediction,
                "candidate_ordinal": ordinal,
                "provenance_views": "|".join(sorted(names)),
                "source_image_sha256s": "|".join(sorted(m["source_image_sha256"] for m in members if m["source_image_sha256"])),
            })
            feature_rows.append({
                "candidate_row_id": row_id,
                "is_c3_anchor": int("BASE_C3" in names),
                "present_raw": int("OCR_RAW" in names),
                "present_ifam": int("OCR_IFAM" in names),
                "present_sharp": int("OCR_IFAM_SHARP" in names),
                "view_source_count": len(names),
                "confidence_raw": conf["OCR_RAW"] or 0.0,
                "confidence_raw_missing": int(conf["OCR_RAW"] is None),
                "confidence_ifam": conf["OCR_IFAM"] or 0.0,
                "confidence_ifam_missing": int(conf["OCR_IFAM"] is None),
                "confidence_sharp": conf["OCR_IFAM_SHARP"] or 0.0,
                "confidence_sharp_missing": int(conf["OCR_IFAM_SHARP"] is None),
                "confidence_max": max(conf_values) if conf_values else 0.0,
                "confidence_mean": sum(conf_values) / len(conf_values) if conf_values else 0.0,
                "confidence_count": len(conf_values),
                "c3_agreement": int(prediction == c3),
                "raw_ifam_agree": int("OCR_RAW" in names and "OCR_IFAM" in names),
                "raw_sharp_agree": int("OCR_RAW" in names and "OCR_IFAM_SHARP" in names),
                "ifam_sharp_agree": int("OCR_IFAM" in names and "OCR_IFAM_SHARP" in names),
                "string_length": len(prediction),
                "group_N": int(split[gid]["image_count"]),
            })

    auth = OUT / "00_AUTHORITY"
    feats = OUT / "01_FEATURES"
    auth.mkdir(parents=True, exist_ok=True)
    feats.mkdir(parents=True, exist_ok=True)
    identity_path = auth / "CANDIDATE_IDENTITY_MAP.csv"
    with identity_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(identity_rows[0]))
        w.writeheader(); w.writerows(identity_rows)
    feature_path = feats / "CANDIDATE_FEATURES.csv"
    with feature_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_row_id", *FEATURES])
        w.writeheader(); w.writerows(feature_rows)

    repeated_sha = {k: sorted(v) for k, v in source_sha_cross_group.items() if len(v) > 1}
    write_json(auth / "A4_FROZEN_CANDIDATE_AUTHORITY.json", {
        "schema": "C3_A4_FROZEN_CANDIDATE_AUTHORITY_V1",
        "status": "FROZEN_CANDIDATES_ONLY_PASS",
        "source_freezes": source_receipts,
        "split_manifest": {"path": str(SPLIT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(SPLIT)},
        "raw_A4_rows": len(all_a4),
        "groups": len(by_group),
        "deduplicated_candidate_rows": len(identity_rows),
        "dedup_key": "exact (group_id, candidate_string); no normalization and no new strings",
        "identity_map": {"path": str(identity_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(identity_path)},
        "new_OCR": 0, "new_restoration": 0, "new_strings": 0, "GT_reads": 0,
    })
    write_json(auth / "C3_BASELINE_AUTHORITY.json", {
        "schema": "C3_BASELINE_AUTHORITY_V1", "frozen_exact": 481, "denominator": 820,
        "ratio": 481 / 820, "role": "T0_FROZEN_BASELINE", "mutation": 0,
    })
    write_json(auth / "OUTER5_SPLIT_AUTHORITY.json", {
        "schema": "OUTER5_SELECTOR_SPLIT_AUTHORITY_V1", "fold_groups": [len(fold_groups[i]) for i in range(5)],
        "pairwise_overlap": pairwise_overlap, "union_groups": len(union), "split_sha256": sha256(SPLIT),
        "selector_train_rule": "other four folds only", "test_GT_feature_reads": 0,
    })
    write_json(feats / "FEATURE_SCHEMA.json", {
        "schema": "C3_A4_SELECTOR_FEATURE_SCHEMA_V1",
        "model_feature_columns": list(FEATURES),
        "join_only_columns": ["candidate_row_id"],
        "forbidden_and_absent": ["GT", "correct", "wrong", "group_id", "filename", "candidate_string", "character_identity", "string_one_hot"],
        "unavailable_and_excluded": {
            "raw_frame_support": "not present in frozen A4 record; not inferred",
            "raw_frame_support_ratio": "not present in frozen A4 record; not inferred",
            "valid_H7": "not present in frozen A4 record; not inferred",
            "valid_H8": "not present in frozen A4 record; not inferred",
        },
        "independence_contract": "RAW/IFAM/SHARP are correlated views; view_source_count is not independent frame support",
        "missing_confidence_contract": "zero fill plus explicit missing indicator",
        "feature_file_sha256": sha256(feature_path),
    })
    bad_feature_names = sorted(set(feature_rows[0]) & {"GT", "gt", "group_id", "candidate_string", "filename", "prediction"})
    receipt = {
        "schema": "C3_A4_AUTHORITY_FEATURE_STATIC_RECEIPT_V1",
        "status": "STATIC_AUTHORITY_AND_FEATURES_PASS" if not bad_feature_names and not repeated_sha and not duplicate_source_rows else "STATIC_BLOCK",
        "counts": {
            "fold_groups": [len(fold_groups[i]) for i in range(5)], "union_groups": len(union),
            "raw_A4_rows": len(all_a4), "deduplicated_candidate_rows": len(identity_rows),
            "candidate_rows_per_fold": {str(i): sum(1 for r in identity_rows if r["outer_fold"] == i) for i in range(5)},
        },
        "checks": {
            "fold_pairwise_disjoint": pairwise_overlap == 0,
            "fold_union_820": len(union) == 820,
            "four_source_rows_per_group": len(all_a4) == 3280,
            "candidate_dedup_unique": len({r["candidate_row_id"] for r in identity_rows}) == len(identity_rows),
            "source_row_duplicates": duplicate_source_rows,
            "source_image_sha_cross_group_collisions": len(repeated_sha),
            "feature_forbidden_column_hits": bad_feature_names,
            "GT_reads": 0, "new_OCR": 0, "new_restoration": 0, "new_strings": 0,
            "independent_support_fabricated": False,
        },
        "outputs": {
            "identity_map_sha256": sha256(identity_path), "candidate_features_sha256": sha256(feature_path),
        },
    }
    write_json(feats / "STATIC_RECEIPT.json", receipt)
    if receipt["status"] != "STATIC_AUTHORITY_AND_FEATURES_PASS":
        raise RuntimeError(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Leakage-safe preparation and fold-local prediction for the frozen C3 A4 bank.

The predictor command can only consume one fold package plus a label file whose
group IDs equal the package's TRAIN IDs.  Test-fold labels, GT-derived features,
and candidate string identity are therefore unavailable to model fitting.
Scoring is intentionally a separate, later program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/real822v3_c3_ifam_a4_crossfit_top1_selector_v1"
DEFAULT_BANKS = (
    ROOT / "artifacts/real822v3_c3_ifam_ocr_aware_restoration_gate_v1/06_POSTTRAIN_PRE_GT_INFERENCE/PREDICTION_FREEZE_V1/A0_A4_CANDIDATES_PRE_GT.csv",
    ROOT / "artifacts/real822v3_c3_ifam_ocr_aware_restoration_gate_v1/11_OUTER1_POSTTRAIN_PRE_GT/PREDICTION_FREEZE_V1/A0_A4_CANDIDATES_PRE_GT.csv",
    ROOT / "artifacts/real822v3_c3_ifam_ocr_aware_restoration_gate_v1/12_OUTER2_POSTTRAIN_PRE_GT/PREDICTION_FREEZE_V1/A0_A4_CANDIDATES_PRE_GT.csv",
    ROOT / "artifacts/real822v3_c3_ifam_ocr_aware_restoration_gate_v1/13_OUTER3_POSTTRAIN_PRE_GT_FREEZE/freeze/A0_A4_CANDIDATES_PRE_GT.csv",
    ROOT / "artifacts/real822v3_c3_ifam_ocr_aware_restoration_gate_v1/13_OUTER4_POSTTRAIN_PRE_GT_FREEZE/freeze/A0_A4_CANDIDATES_PRE_GT.csv",
)
SOURCES = ("BASE_C3", "OCR_RAW", "OCR_IFAM", "OCR_IFAM_SHARP")
FEATURES = (
    "is_c3_anchor", "present_raw", "present_ifam", "present_sharp",
    "source_count", "raw_frame_support", "raw_frame_support_ratio",
    "raw_confidence", "ifam_confidence", "sharp_confidence",
    "raw_confidence_available", "ifam_confidence_available", "sharp_confidence_available",
    "max_confidence", "mean_confidence", "c3_agreement",
    "raw_ifam_agree", "raw_sharp_agree", "ifam_sharp_agree",
    "valid_h7", "valid_h8", "string_length", "group_n",
)
FORBIDDEN_COLUMNS = {"gt", "truth", "label", "correct", "is_correct", "target"}
SEED = 20260910


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or FORBIDDEN_COLUMNS.intersection(x.lower() for x in r.fieldnames):
            raise RuntimeError(f"GT_OR_TARGET_COLUMN_FORBIDDEN:{path}")
        return list(r)


def stable_key(candidate: str) -> str:
    # Used only after all scientific evidence ties; never enters the model.
    return hashlib.sha256(candidate.encode("utf-8")).hexdigest().upper()


def make_features(rows: list[dict], fold: int) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["arm_id"] == "A4":
            by_group[row["group_id"]].append(row)
    out: list[dict] = []
    for gid in sorted(by_group):
        rr = by_group[gid]
        if {r["candidate_name"] for r in rr} != set(SOURCES):
            raise RuntimeError(f"A4_SOURCE_SET_MISMATCH:{gid}")
        source = {r["candidate_name"]: r for r in rr}
        baseline = source["BASE_C3"]["prediction"]
        candidates = sorted({r["prediction"] for r in rr}, key=stable_key)
        # Empty OCR output is a real frozen candidate, not missing data.  Keep it
        # in the bank so the selector cannot silently improve its universe.
        for candidate in candidates:
            present = {name: int(source[name]["prediction"] == candidate) for name in SOURCES}
            confs: dict[str, float] = {}
            available: dict[str, int] = {}
            for name in SOURCES[1:]:
                raw = source[name]["confidence"].strip()
                available[name] = int(bool(raw))
                confs[name] = float(raw) if raw else 0.0
            matching = [confs[name] for name in SOURCES[1:] if present[name] and available[name]]
            length = len(candidate)
            out.append({
                "outer_fold": fold, "group_id": gid, "candidate": candidate,
                "candidate_tie_sha256": stable_key(candidate),
                "is_c3_anchor": int(candidate == baseline),
                "present_raw": present["OCR_RAW"], "present_ifam": present["OCR_IFAM"],
                "present_sharp": present["OCR_IFAM_SHARP"], "source_count": sum(present.values()),
                # No independent raw-frame support exists in the frozen A4 schema.
                "raw_frame_support": 0, "raw_frame_support_ratio": 0.0,
                "raw_confidence": confs["OCR_RAW"] if present["OCR_RAW"] else 0.0,
                "ifam_confidence": confs["OCR_IFAM"] if present["OCR_IFAM"] else 0.0,
                "sharp_confidence": confs["OCR_IFAM_SHARP"] if present["OCR_IFAM_SHARP"] else 0.0,
                "raw_confidence_available": available["OCR_RAW"] * present["OCR_RAW"],
                "ifam_confidence_available": available["OCR_IFAM"] * present["OCR_IFAM"],
                "sharp_confidence_available": available["OCR_IFAM_SHARP"] * present["OCR_IFAM_SHARP"],
                "max_confidence": max(matching) if matching else 0.0,
                "mean_confidence": sum(matching) / len(matching) if matching else 0.0,
                "c3_agreement": int(candidate == baseline),
                "raw_ifam_agree": present["OCR_RAW"] * present["OCR_IFAM"],
                "raw_sharp_agree": present["OCR_RAW"] * present["OCR_IFAM_SHARP"],
                "ifam_sharp_agree": present["OCR_IFAM"] * present["OCR_IFAM_SHARP"],
                "valid_h7": int(length == 7), "valid_h8": int(length == 8),
                "string_length": length, "group_n": len(rr),
            })
    return out


def choose_t1(rows: list[dict]) -> dict:
    # Frozen requested priority: anchor, agreement, raw support, confidence, stable tie.
    return max(rows, key=lambda r: (
        int(r["is_c3_anchor"]), int(r["source_count"]), int(r["raw_frame_support"]),
        float(r["max_confidence"]), r["candidate_tie_sha256"],
    ))


def load_labels(path: Path, expected_train_ids: set[str]) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(reader.fieldnames) != {"group_id", "gt"}:
            raise RuntimeError("TRAIN_LABEL_SCHEMA_MUST_BE_EXACT_GROUP_ID_GT")
        rows = list(reader)
    labels = {r["group_id"]: r["gt"] for r in rows}
    if len(labels) != len(rows) or set(labels) != expected_train_ids:
        raise RuntimeError("TRAIN_LABEL_IDS_NOT_EXACT_OR_TEST_GT_DELIVERED")
    return labels


def fit_predict(package: Path, labels_path: Path, out: Path) -> None:
    contract = json.loads((package / "FOLD_CONTRACT.json").read_text(encoding="utf-8"))
    train = read_csv(package / "TRAIN_FEATURES_PRE_GT.csv")
    test = read_csv(package / "TEST_FEATURES_PRE_GT.csv")
    train_ids = {r["group_id"] for r in train}; test_ids = {r["group_id"] for r in test}
    if train_ids & test_ids or train_ids != set(contract["train_group_ids"]) or test_ids != set(contract["test_group_ids"]):
        raise RuntimeError("FOLD_PACKAGE_IDENTITY_OR_OVERLAP_FAIL")
    labels = load_labels(labels_path, train_ids)
    for r in train:
        r["label"] = int(r["candidate"] == labels[r["group_id"]])
    x = np.asarray([[float(r[k]) for k in FEATURES] for r in train], dtype=np.float64)
    y = np.asarray([r["label"] for r in train], dtype=np.int64)
    group_sizes = {gid: sum(r["group_id"] == gid for r in train) for gid in train_ids}
    weights = np.asarray([1.0 / group_sizes[r["group_id"]] for r in train], dtype=np.float64)
    if set(y) != {0, 1}:
        raise RuntimeError("TRAIN_TARGET_DEGENERATE")
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", random_state=SEED, max_iter=1000)
    model.fit(scaler.transform(x), y, sample_weight=weights)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in test: by[r["group_id"]].append(r)
    t2=[]; t3=[]
    for gid in sorted(by):
        rr=by[gid]; xx=np.asarray([[float(r[k]) for k in FEATURES] for r in rr], dtype=np.float64)
        scores=model.predict_proba(scaler.transform(xx))[:, 1]
        for r,s in zip(rr,scores): r["selector_score"]=float(s)
        winner=max(rr,key=lambda r:(r["selector_score"],r["candidate_tie_sha256"]))
        base=next(r for r in rr if int(r["is_c3_anchor"])==1)
        conservative=winner
        if winner["candidate"] != base["candidate"] and not (
            winner["selector_score"] > base["selector_score"]
            and (int(winner["valid_h7"]) or int(winner["valid_h8"]))
            and int(winner["source_count"]) >= 2
        ):
            conservative=base
        t2.append({"group_id":gid,"prediction":winner["candidate"],"candidate_tie_sha256":winner["candidate_tie_sha256"]})
        t3.append({"group_id":gid,"prediction":conservative["candidate"],"candidate_tie_sha256":conservative["candidate_tie_sha256"]})
    out.mkdir(parents=True,exist_ok=True)
    model_path=out/"SELECTOR_MODEL_PRE_GT.json"
    atomic_json(model_path,{"schema":"C3_A4_L2_LOGISTIC_V1","outer_fold":contract["outer_fold"],
        "features":list(FEATURES),"string_identity_feature_count":0,"seed":SEED,"C":1.0,
        "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),
        "coefficient":model.coef_[0].tolist(),"intercept":float(model.intercept_[0]),
        "group_weight_rule":"1/|C_g|","group_weight_total_min":min(sum(weights[i] for i,r in enumerate(train) if r['group_id']==g) for g in train_ids),
        "group_weight_total_max":max(sum(weights[i] for i,r in enumerate(train) if r['group_id']==g) for g in train_ids),
        "test_gt_delivered":False,"test_gt_reads":0})
    atomic_json(out/"T2_PRE_GT.json",{"outer_fold":contract["outer_fold"],"predictions":t2,"test_gt_reads":0})
    atomic_json(out/"T3_PRE_GT.json",{"outer_fold":contract["outer_fold"],"predictions":t3,"test_gt_reads":0})
    atomic_json(out/"PREDICTION_FREEZE.json",{"status":"PRE_GT_FREEZE_PASS","groups":len(by),
        "model_sha256":sha(model_path),"T2_sha256":sha(out/'T2_PRE_GT.json'),"T3_sha256":sha(out/'T3_PRE_GT.json'),
        "missing":0,"duplicates":0,"new_strings":0,"test_gt_reads":0})


def prepare(banks: list[Path], out: Path) -> None:
    if len(banks) != 5: raise RuntimeError("EXACTLY_FIVE_FOLD_BANKS_REQUIRED")
    features=[]; authority=[]
    for fold,path in enumerate(banks):
        rows=read_csv(path); ff=make_features(rows,fold)
        gids={r["group_id"] for r in ff}
        if len(gids)!=164: raise RuntimeError(f"FOLD_GROUP_COUNT_FAIL:{fold}:{len(gids)}")
        features.extend(ff);authority.append({"outer_fold":fold,"path":str(path),"sha256":sha(path),"groups":len(gids)})
    all_ids={r["group_id"] for r in features}
    if len(all_ids)!=820: raise RuntimeError(f"OOF_UNION_OR_OVERLAP_FAIL:{len(all_ids)}")
    fields=["outer_fold","group_id","candidate","candidate_tie_sha256",*FEATURES]
    atomic_csv(out/"01_FEATURES/CANDIDATE_FEATURES_PRE_GT.csv",features,fields)
    atomic_json(out/"00_AUTHORITY/A4_FROZEN_CANDIDATE_AUTHORITY.json",{"status":"PASS","banks":authority,"groups":820,"new_ocr":0,"new_restoration":0,"new_strings":0})
    atomic_json(out/"00_AUTHORITY/SELECTOR_EXECUTION_CONTRACT.json",{
        "experiment_id":"REAL822V3_C3_IFAM_A4_CROSSFIT_TOP1_SELECTOR_V1",
        "mode":"FROZEN_CANDIDATES_ONLY","folds":5,"seed":SEED,
        "classifier":"L2 LogisticRegression(C=1.0, solver=liblinear)",
        "group_weight":"1/|C_g|","feature_order":list(FEATURES),
        "test_gt_delivery":"FORBIDDEN; fit-predict rejects any label IDs outside the 656 TRAIN group IDs",
        "prediction_before_score":True,"scoring_program":"SEPARATE_NOT_IMPLEMENTED_IN_THIS_PRE_GT_NODE",
        "no_ocr_rerun":True,"no_restoration_rerun":True,"no_new_strings":True,
        "no_string_identity_feature":True,"deterministic_tie_break":"SHA256(candidate) after score tie",
        "t3_gate":"alternative score strictly greater than C3 AND H7/H8 valid AND source_count>=2; tie retains C3",
        "one_shot_no_retuning":True,"production_mutation":0})
    atomic_json(out/"01_FEATURES/FEATURE_SCHEMA.json",{"status":"FROZEN","feature_order":list(FEATURES),
        "string_identity_features":[],"unavailable_features":{"raw_frame_support":"0; absent from frozen A4","raw_frame_support_ratio":"0; absent from frozen A4"},
        "tie_break":"SHA256(candidate), only after evidence score tie"})
    t1=[]
    grouped: dict[str,list[dict]]=defaultdict(list)
    for r in features: grouped[r["group_id"]].append(r)
    for gid in sorted(grouped):
        w=choose_t1(grouped[gid]);t1.append({"group_id":gid,"prediction":w["candidate"],"candidate_tie_sha256":w["candidate_tie_sha256"]})
    atomic_json(out/"03_PRE_GT/T1_PRE_GT.json",{"predictions":t1,"groups":820,"gt_reads":0})
    for fold in range(5):
        package=out/f"02_SELECTORS/fold{fold}/package";train=[r for r in features if int(r["outer_fold"])!=fold];test=[r for r in features if int(r["outer_fold"])==fold]
        atomic_csv(package/"TRAIN_FEATURES_PRE_GT.csv",train,fields);atomic_csv(package/"TEST_FEATURES_PRE_GT.csv",test,fields)
        atomic_json(package/"FOLD_CONTRACT.json",{"outer_fold":fold,"train_group_ids":sorted({r['group_id'] for r in train}),
            "test_group_ids":sorted({r['group_id'] for r in test}),"train_groups":656,"test_groups":164,
            "intersection":0,"test_gt_file_in_package":False,"test_gt_reads":0})
    atomic_json(out/"02_SELECTORS/SELECTOR_LEAKAGE_AUDIT_PREPARED.json",{"status":"PASS","folds":5,
        "train_groups_each":656,"test_groups_each":164,"test_gt_files":0,"feature_gt_columns":0,"production_mutation":0})


def main() -> None:
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare");q.add_argument("--banks",type=Path,nargs=5,default=list(DEFAULT_BANKS));q.add_argument("--output",type=Path,default=EXP)
    q=sub.add_parser("fit-predict");q.add_argument("--package",type=Path,required=True);q.add_argument("--train-labels",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.command=="prepare": prepare([x.resolve() for x in a.banks],a.output.resolve())
    else: fit_predict(a.package.resolve(),a.train_labels.resolve(),a.output.resolve())


if __name__=="__main__": main()

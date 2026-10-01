#!/usr/bin/env python3
"""Post-freeze scorer for REAL822V3 C3/A4 cross-fitted selectors.

This module deliberately has no prediction or training code.  It opens GT only
after a complete, SHA-verified 03_PRE_GT authority has been validated.

Time: O(N * L^2) for Levenshtein scoring.  Space: O(N + L^2).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable


class PostFreezeGateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _declared_files(audit: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("prediction_files", "files", "artifacts"):
        value = audit.get(key)
        if isinstance(value, list):
            return value
    return []


def validate_pregt(root: Path) -> tuple[dict[str, Any], list[Path]]:
    """Fail closed unless the full prediction universe was frozen first."""
    pregt = root / "03_PRE_GT"
    audit_path = pregt / "PRE_GT_SHA_AUDIT.json"
    if not audit_path.is_file():
        raise PostFreezeGateError("PRE_GT_SHA_AUDIT_MISSING")
    audit = read_json(audit_path)
    status = str(audit.get("status", audit.get("verdict", ""))).upper()
    if status not in {"PASS", "PRE_GT_SHA_FREEZE_PASS", "PRE_GT_FREEZE_PASS", "PRE_GT_OOF820_FREEZE_PASS"}:
        raise PostFreezeGateError(f"PRE_GT_NOT_FROZEN:{status or 'EMPTY'}")
    if audit.get("gt_reads_before_freeze", audit.get("test_gt_reads", 0)) != 0:
        raise PostFreezeGateError("GT_READ_BEFORE_FREEZE_NONZERO")
    if audit.get("new_strings", 0) != 0:
        raise PostFreezeGateError("NEW_STRINGS_NONZERO")
    declared = _declared_files(audit)
    if not declared and status == "PRE_GT_OOF820_FREEZE_PASS":
        canonical = {"T1": "T1_OOF820_PRE_GT.json", "T2": "T2_OOF820_PRE_GT.json", "T3": "T3_OOF820_PRE_GT.json"}
        declared = [{"path": name, "sha256": audit.get(f"{arm}_sha256", "")} for arm, name in canonical.items()]
    if not declared:
        raise PostFreezeGateError("PRE_GT_FILE_MANIFEST_EMPTY")
    paths: list[Path] = []
    for item in declared:
        rel = item.get("path") or item.get("file")
        expected = str(item.get("sha256", "")).upper()
        if not rel or len(expected) != 64:
            raise PostFreezeGateError("PRE_GT_FILE_DECLARATION_INVALID")
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
            if not path.exists():
                path = pregt / rel
        if not path.is_file() or sha256(path) != expected:
            raise PostFreezeGateError(f"PRE_GT_SHA_MISMATCH:{rel}")
        paths.append(path)
    return audit, paths


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(x) for x in value]
    if isinstance(value, dict):
        for key in ("predictions", "rows", "records"):
            if isinstance(value.get(key), list):
                return [dict(x) for x in value[key]]
        return [{"group_id": k, "prediction": v} for k, v in value.items() if isinstance(v, str)]
    raise PostFreezeGateError("PREDICTION_SCHEMA_INVALID")


def load_predictions(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    arms: dict[str, dict[str, str]] = {}
    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        for row in _rows(read_json(path)):
            gid = str(row.get("group_id", ""))
            arm = str(row.get("arm") or row.get("selector") or row.get("method") or path.stem.split("_")[0]).upper()
            pred = row.get("prediction", row.get("selected_string", row.get("pred")))
            if not gid or pred is None:
                continue
            if gid in arms.setdefault(arm, {}):
                raise PostFreezeGateError(f"DUPLICATE_PREDICTION:{arm}:{gid}")
            arms[arm][gid] = str(pred)
    if not arms:
        raise PostFreezeGateError("NO_PREDICTIONS_IN_FROZEN_FILES")
    return arms


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_gt(paths: list[Path], expected_sha: dict[str, str]) -> dict[str, str]:
    gt: dict[str, str] = {}
    for path in paths:
        expected = expected_sha.get(str(path.resolve()).lower())
        if not expected:
            raise PostFreezeGateError(f"GT_NOT_IN_AUTHORITY:{path}")
        if sha256(path) != expected.upper():
            raise PostFreezeGateError(f"GT_SHA_MISMATCH:{path}")
        for row in load_jsonl(path):
            gid = str(row.get("group_id", row.get("id", "")))
            label = row.get("gt", row.get("label", row.get("text")))
            if not gid or label is None:
                raise PostFreezeGateError(f"GT_SCHEMA_INVALID:{path}")
            if gid in gt and gt[gid] != str(label):
                raise PostFreezeGateError(f"GT_CONFLICT:{gid}")
            gt[gid] = str(label)
    return gt


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return csv_rows(path)
    value = read_json(path) if path.suffix.lower() == ".json" else load_jsonl(path)
    return _rows(value)


def baseline_from_a4_authority(path: Path, repo_root: Path) -> dict[str, str]:
    authority = read_json(path)
    baseline: dict[str, str] = {}
    for source in authority.get("source_freezes", []):
        src = Path(source["path"])
        if not src.is_absolute():
            src = repo_root / src
        if sha256(src) != str(source["sha256"]).upper():
            raise PostFreezeGateError(f"BASELINE_SOURCE_SHA_MISMATCH:{src}")
        for row in csv_rows(src):
            if row.get("arm_id") != "A0":
                continue
            gid, pred = row["group_id"], row["prediction"]
            if gid in baseline and baseline[gid] != pred:
                raise PostFreezeGateError(f"BASELINE_CONFLICT:{gid}")
            baseline[gid] = pred
    if len(baseline) != 820:
        raise PostFreezeGateError(f"BASELINE_UNIVERSE_NOT_820:{len(baseline)}")
    return baseline


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(pred: dict[str, str], gt: dict[str, str]) -> dict[str, Any]:
    ids = sorted(gt)
    if set(pred) != set(ids):
        raise PostFreezeGateError(f"PREDICTION_UNIVERSE_MISMATCH:missing={len(set(ids)-set(pred))}:extra={len(set(pred)-set(ids))}")
    ed = [edit_distance(pred[g], gt[g]) for g in ids]
    exact = sum(pred[g] == gt[g] for g in ids)
    chars = sum(len(gt[g]) for g in ids)
    return {"exact": exact, "denominator": len(ids), "ratio": exact / len(ids),
            "sum_ed": sum(ed), "mean_ed": sum(ed) / len(ids),
            "cer": sum(ed) / chars if chars else 0.0, "gt_characters": chars}


def paired(base: dict[str, str], pred: dict[str, str], gt: dict[str, str]) -> dict[str, int]:
    wc = sum(base[g] != gt[g] and pred[g] == gt[g] for g in gt)
    cw = sum(base[g] == gt[g] and pred[g] != gt[g] for g in gt)
    changes = [edit_distance(pred[g], gt[g]) - edit_distance(base[g], gt[g]) for g in gt]
    return {"W_to_C": wc, "C_to_W": cw, "Net": wc - cw,
            "ED_improve": sum(x < 0 for x in changes), "ED_same": sum(x == 0 for x in changes),
            "ED_worse": sum(x > 0 for x in changes)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise PostFreezeGateError(f"EMPTY_OUTPUT:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def render_viewer(summary: dict[str, Any]) -> str:
    cards = "".join(f"<article><h2>{html.escape(a)}</h2><b>{m['exact']}/{m['denominator']}</b><p>{m['ratio']:.2%}</p><p>Mean ED {m['mean_ed']:.4f} · CER {m['cer']:.4f}</p></article>" for a, m in summary["metrics"].items())
    folds = "".join(f"<tr><td>{r['fold']}</td><td>{r['arm']}</td><td>{r['exact']}/{r['N']}</td><td>{r['Net']}</td></tr>" for r in summary["folds"])
    return f"""<!doctype html><meta charset=utf-8><title>C3 A4 Cross-fit Top-1</title><style>body{{font:16px system-ui;background:#10131a;color:#eef;margin:2rem}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem}}article,section{{background:#1c2230;padding:1rem;border-radius:12px}}b{{font-size:2rem;color:#78e6b2}}table{{width:100%;border-collapse:collapse}}td,th{{padding:.45rem;border-bottom:1px solid #445;text-align:left}}</style><h1>REAL822v3 C3/A4 Cross-fitted Top-1</h1><p>POST-FREEZE only · frozen candidates · no new strings</p><main>{cards}</main><section><h2>Fold comparison</h2><table><tr><th>Fold</th><th>Arm</th><th>Exact</th><th>Net vs C3</th></tr>{folds}</table></section><section><h2>Bottleneck</h2><pre>{html.escape(json.dumps(summary['bottleneck'], ensure_ascii=False, indent=2))}</pre></section>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gt", type=Path, action="append", required=True)
    ap.add_argument("--gt-sha-authority", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, help="Frozen C3 predictions JSON")
    ap.add_argument("--candidate-authority", type=Path, help="A4 authority used to recover SHA-frozen A0")
    ap.add_argument("--candidates", type=Path, required=True, help="Frozen A4 candidate rows JSON/JSONL")
    args = ap.parse_args()

    audit, frozen_paths = validate_pregt(args.root)  # GT must not be opened before this line.
    predictions = load_predictions(frozen_paths)
    if args.baseline:
        base_rows = _rows(read_json(args.baseline))
        baseline = {str(r["group_id"]): str(r.get("prediction", r.get("pred"))) for r in base_rows}
    elif args.candidate_authority:
        baseline = baseline_from_a4_authority(args.candidate_authority, Path.cwd())
    else:
        raise PostFreezeGateError("BASELINE_AUTHORITY_REQUIRED")
    authority = read_json(args.gt_sha_authority)
    gt_shas = {}
    for item in authority.get("GT_authority", authority.get("files", [])):
        p = Path(item.get("path", item.get("file", "")))
        if not p.is_absolute():
            p = Path.cwd() / p
        gt_shas[str(p.resolve()).lower()] = str(item["sha256"]).upper()
    gt = load_gt(args.gt, gt_shas)  # first GT open, after frozen prediction verification
    if len(gt) != 820:
        raise PostFreezeGateError(f"GT_UNIVERSE_NOT_820:{len(gt)}")

    all_pred = {"T0": baseline, **predictions}
    scored = {arm: metrics(pred, gt) for arm, pred in sorted(all_pred.items())}
    transitions = {arm: paired(baseline, pred, gt) for arm, pred in sorted(all_pred.items())}
    candidate_rows = load_candidate_rows(args.candidates)
    candidate_sets: dict[str, set[str]] = {}
    for r in candidate_rows:
        candidate_sets.setdefault(str(r["group_id"]), set()).add(str(r.get("candidate_string", r.get("candidate"))))
    if set(candidate_sets) != set(gt):
        raise PostFreezeGateError("CANDIDATE_UNIVERSE_MISMATCH")
    ceiling = sum(gt[g] in candidate_sets[g] for g in gt)

    fold_of = {str(r["group_id"]): int(r["outer_fold"]) for r in candidate_rows}
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fg = {g: y for g, y in gt.items() if fold_of[g] == fold}
        if len(fg) != 164:
            raise PostFreezeGateError(f"FOLD_NOT_164:{fold}:{len(fg)}")
        for arm, pred in sorted(all_pred.items()):
            fm = metrics({g: pred[g] for g in fg}, fg)
            fp = paired({g: baseline[g] for g in fg}, {g: pred[g] for g in fg}, fg)
            fold_rows.append({"fold": fold, "arm": arm, "N": 164, "exact": fm["exact"], "ratio": fm["ratio"], "mean_ed": fm["mean_ed"], "cer": fm["cer"], **fp})

    bottleneck = []
    # The preregistered PRIMARY is T2; T3 is the C3-preserving safety arm.
    primary = "T2" if "T2" in all_pred else ("T3" if "T3" in all_pred else sorted(predictions)[0])
    for g, truth in gt.items():
        if baseline[g] == truth:
            cls = "C3_ALREADY_CORRECT"
        elif truth not in candidate_sets[g]:
            cls = "GENERATION_LIMIT"
        elif all_pred[primary][g] == truth:
            cls = "RESCUE_SUCCESS"
        else:
            cls = "DECISION_MISS"
        bottleneck.append({"group_id": g, "outer_fold": fold_of[g], "class": cls})
    counts = {k: sum(r["class"] == k for r in bottleneck) for k in ("C3_ALREADY_CORRECT", "GENERATION_LIMIT", "RESCUE_SUCCESS", "DECISION_MISS")}
    gain = scored[primary]["exact"] - scored["T0"]["exact"]
    headroom = ceiling - scored["T0"]["exact"]
    efficiency = gain / headroom if headroom else 0.0
    exact_primary = scored[primary]["exact"]
    decision = ("REAL822_STRICT_TOP1_70P_REACHED" if exact_primary >= 574 else
                "TOP1_SELECTOR_STRONG_GAIN" if exact_primary >= 533 else
                "TOP1_SELECTOR_OOF820_PASS" if exact_primary > scored["T0"]["exact"] else
                "TOP1_SELECTOR_NO_GAIN")
    summary = {"schema": "REAL822V3_C3_A4_TOP1_POSTFREEZE_SCORE_V1", "status": "FINAL_EXPERIMENT_REPORT_PASS", "decision": decision,
               "freeze_audit_sha256": sha256(args.root / "03_PRE_GT" / "PRE_GT_SHA_AUDIT.json"),
               "metrics": scored, "paired_vs_C3": transitions, "folds": fold_rows,
               "candidate_ceiling": ceiling, "decision_gap": ceiling - scored[primary]["exact"],
               "ceiling_efficiency": efficiency, "primary_arm": primary, "bottleneck": counts,
               "gt_opened_after_all_freezes": True, "new_strings": 0, "production_mutation": 0}
    score = args.root / "04_SCORE"; final = args.root / "05_FINAL"
    atomic_text(score / "TOP1_SCORE_COMPARISON.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_csv(score / "PAIRED_TRANSITIONS.csv", [{"arm": a, **v} for a, v in transitions.items()])
    write_csv(score / "FOLD_COMPARISON.csv", fold_rows)
    write_csv(score / "BOTTLENECK_DECOMPOSITION.csv", bottleneck)
    atomic_text(score / "CEILING_EFFICIENCY.json", json.dumps({"baseline": scored["T0"]["exact"], "ceiling": ceiling, "top1": scored[primary]["exact"], "efficiency": efficiency}, indent=2) + "\n")
    atomic_text(final / "FINAL_GATE.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    atomic_text(final / "FINAL_REPORT.md", f"# C3/A4 Top-1 Selector\n\nStatus: `FINAL_EXPERIMENT_REPORT_PASS`\n\nPrimary {primary}: {scored[primary]['exact']}/820 ({scored[primary]['ratio']:.2%})\n\nCandidate ceiling: {ceiling}/820; decision gap: {summary['decision_gap']}; efficiency: {efficiency:.2%}.\n")
    atomic_text(final / "index.html", render_viewer(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

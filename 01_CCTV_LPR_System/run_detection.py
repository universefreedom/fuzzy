from __future__ import annotations

import argparse
import json
from collections import deque
import time
from datetime import datetime
from pathlib import Path
import sys
import cv2

from pipeline.factory import PipelineFactory
from pipeline.middle_slot_presets import (
    apply_middle_slot_preset_to_args,
    available_middle_slot_presets,
)
from visual_web.gpu_runtime_bridge import GPURuntimeViewerSink


class StageLogger:
    def __init__(self, console: bool = False, log_txt: str | None = None):
        self.console = console
        self.log_txt = log_txt
        self._file = None
        if self.log_txt:
            Path(self.log_txt).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_txt, "a", encoding="utf-8")
            self._count = 0

        sinks = []
        if self._file is not None:
            def file_logger(msg: str):
                print(msg, file=self._file, flush=False)
                self._count += 1
                if self._count % 50 == 0:
                    self._file.flush()
            sinks.append(file_logger)

        if self.console:
            def console_logger(msg: str):
                print(msg)
            sinks.append(console_logger)

        if len(sinks) == 0:
            self._logger = None
        elif len(sinks) == 1:
            self._logger = sinks[0]
        else:
            def composite_logger(msg: str):
                for fn in sinks:
                    fn(msg)
            self._logger = composite_logger

    def log(self, msg: str) -> None:
        ts = datetime.now().isoformat(timespec="milliseconds")
        line = f"{ts} {msg}"
        if self._logger is not None:
            self._logger(line)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None


def write_webapp_result_json(
    result_path,
    *,
    frame_idx=None,
    track_id=None,
    state="WAITING",
    plate="-",
    candidate="-",
    fps=None,
    source="-",
    confidence=None,
    recent_candidates=None,
    extra=None,
):
    """
    모바일 웹앱이 읽을 수 있는 result.json을 안전하게 저장한다.
    이 함수에서 오류가 나도 메인 파이프라인은 중단되면 안 된다.
    """
    if not result_path:
        return

    try:
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "frame": frame_idx,
            "track_id": track_id,
            "state": state,
            "plate": plate if plate is not None else "-",
            "candidate": candidate if candidate is not None else "-",
            "fps": fps,
            "source": source if source is not None else "-",
            "confidence": confidence,
            "recent_candidates": recent_candidates or [],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if extra and isinstance(extra, dict):
            payload.update(extra)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        tmp_path.replace(path)

    except Exception:
        # 웹앱 출력 실패는 메인 파이프라인 실패로 취급하지 않는다.
        return


def _webapp_float_or_none(value):
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def _webapp_int_or_none(value):
    try:
        if value in {"", None}:
            return None
        return int(float(value))
    except Exception:
        return None


def _webapp_latest_history_item(pipe, track_id=None):
    try:
        history = getattr(pipe, "ocr_history", {}) or {}
        if track_id is not None and int(track_id) in history and history.get(int(track_id)):
            return dict(history.get(int(track_id), [])[-1] or {})
        latest = None
        latest_frame = -1
        for history_track_id, items in history.items():
            if not items:
                continue
            item = dict(items[-1] or {})
            frame_value = _webapp_int_or_none(item.get("frame_idx"))
            if frame_value is None:
                frame_value = -1
            if latest is None or frame_value >= latest_frame:
                latest = item
                latest_frame = frame_value
        return latest or {}
    except Exception:
        return {}


def _webapp_paired_history_item(pipe):
    """Newest OCR observation that owns its text, track, frame and bbox."""
    try:
        history = getattr(pipe, "ocr_history", {}) or {}
        best = None
        best_frame = -1
        for history_track_id, items in history.items():
            for raw_item in reversed(items or []):
                item = dict(raw_item or {})
                text = (
                    str(item.get("raw_text", "") or "")
                    or str(item.get("text", "") or "")
                    or str(item.get("ocr_text", "") or "")
                    or str(item.get("ocr_plate_text_final", "") or "")
                    or str(item.get("normalized_plate_candidate", "") or "")
                )
                frame_value = _webapp_int_or_none(item.get("frame_idx"))
                track_value = _webapp_int_or_none(item.get("track_id"))
                if track_value is None:
                    track_value = _webapp_int_or_none(history_track_id)
                runtime_bbox = item.get("bbox")
                if isinstance(runtime_bbox, (list, tuple)) and len(runtime_bbox) == 4:
                    bbox = [_webapp_float_or_none(v) for v in runtime_bbox]
                else:
                    bbox = [_webapp_float_or_none(item.get(k)) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
                if not text or frame_value is None or track_value is None or any(v is None for v in bbox):
                    continue
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                item["_viewer_text"] = text
                item["_viewer_bbox_xyxy"] = bbox
                item["track_id"] = track_value
                if best is None or frame_value > best_frame:
                    best = item
                    best_frame = frame_value
                break
        return best or {}
    except Exception:
        return {}


def _webapp_source_label(raw_source, *, with_febam=False):
    source_text = str(raw_source or "")
    if "fusion" in source_text:
        return "fusion+febam" if with_febam else "fusion"
    if with_febam:
        return "ocr+febam"
    return "ocr" if source_text else "waiting"


def _webapp_best_string_state(pipe):
    states = getattr(pipe, "string_febam_states", {}) or {}
    best_track_id = None
    best_state = None
    best_score = float("-inf")
    best_rank = -1
    for track_key, state in states.items():
        if not isinstance(state, dict):
            continue
        score = _webapp_float_or_none(state.get("activation"))
        if score is None:
            score = _webapp_float_or_none(state.get("score"))
        if score is None:
            score = 0.0
        rank = 1 if str(state.get("state", state.get("string_febam_state", ""))) == "COMMIT" else 0
        if best_state is None or rank > best_rank or (rank == best_rank and score > best_score):
            best_track_id = _webapp_int_or_none(track_key)
            best_state = state
            best_score = score
            best_rank = rank
    return best_track_id, best_state


def build_webapp_result_payload(pipe, times, frame_idx):
    times = times if isinstance(times, dict) else {}
    current_fps = _webapp_float_or_none(times.get("fps"))
    # Live display follows the newest completed OCR observation. Selecting the
    # globally strongest FEBAM track here pins the Viewer to one old frame even
    # while newer OCR jobs finish on other tracks.
    latest_history = _webapp_paired_history_item(pipe)
    ocr_observation_frame_idx = _webapp_int_or_none(latest_history.get("frame_idx"))
    history_track_id = _webapp_int_or_none(latest_history.get("track_id"))
    track_id = history_track_id
    states = getattr(pipe, "string_febam_states", {}) or {}
    string_state = states.get(track_id) if track_id is not None else None
    if string_state is None and track_id is not None:
        string_state = states.get(str(track_id))
    if track_id is None:
        track_id = _webapp_int_or_none(times.get("track_id"))
        if track_id is not None and track_id < 0:
            track_id = None

    raw_candidate = str(latest_history.get("_viewer_text", "") or "")
    ocr_bbox_xyxy = latest_history.get("_viewer_bbox_xyxy")
    raw_source = (
        latest_history.get("source")
        or latest_history.get("ocr_source")
        or latest_history.get("ocr_trigger_reason")
        or latest_history.get("ocr_sampling_level")
        or ""
    )
    history_conf = _webapp_float_or_none(latest_history.get("conf"))
    if history_conf is None:
        history_conf = _webapp_float_or_none(latest_history.get("raw_conf"))

    if isinstance(string_state, dict) and str(string_state.get("state", string_state.get("string_febam_state", ""))) == "COMMIT":
        plate = (
            str(string_state.get("final_observed_plate", "") or "")
            or str(string_state.get("committed_text", "") or "")
            or str(string_state.get("final_consensus_plate", "") or "")
            or str(getattr(pipe, "_last_final_plate", "") or "")
            or "-"
        )
        candidate = raw_candidate or str(string_state.get("best_text", "") or plate or "-")
        confidence = _webapp_float_or_none(string_state.get("activation"))
        if confidence is None:
            confidence = _webapp_float_or_none(string_state.get("score"))
        if confidence is None:
            confidence = history_conf
        return {
            "frame_idx": frame_idx,
            "ocr_observation_frame_idx": ocr_observation_frame_idx,
            "ocr_bbox_xyxy": ocr_bbox_xyxy,
            "track_id": track_id,
            "state": "COMMIT",
            "plate": plate,
            "candidate": candidate or "-",
            "fps": current_fps,
            "source": _webapp_source_label(raw_source, with_febam=True),
            "confidence": confidence,
        }

    best_text = ""
    confidence = history_conf
    if isinstance(string_state, dict):
        best_text = (
            str(string_state.get("best_text", "") or "")
            or str(string_state.get("final_observed_plate", "") or "")
            or str(string_state.get("final_consensus_plate", "") or "")
        )
        state_conf = _webapp_float_or_none(string_state.get("activation"))
        if state_conf is None:
            state_conf = _webapp_float_or_none(string_state.get("score"))
        if state_conf is not None:
            confidence = state_conf

    candidate = raw_candidate or best_text
    candidates_count = _webapp_float_or_none(times.get("candidates_count")) or 0.0
    tracks_count = _webapp_float_or_none(times.get("tracks_count")) or 0.0
    has_candidate = bool(candidate or best_text) or candidates_count > 0.0 or tracks_count > 0.0
    if has_candidate:
        return {
            "frame_idx": frame_idx,
            "ocr_observation_frame_idx": ocr_observation_frame_idx,
            "ocr_bbox_xyxy": ocr_bbox_xyxy,
            "track_id": track_id,
            "state": "HOLD",
            "plate": best_text or candidate or "-",
            "candidate": candidate or best_text or "-",
            "fps": current_fps,
            "source": _webapp_source_label(raw_source, with_febam=True),
            "confidence": confidence,
        }

    return {
        "frame_idx": frame_idx,
        "ocr_observation_frame_idx": None,
        "ocr_bbox_xyxy": None,
        "track_id": None,
        "state": "WAITING",
        "plate": "-",
        "candidate": "-",
        "fps": current_fps,
        "source": "waiting",
        "confidence": None,
    }


def _webapp_candidate_text(payload):
    if not isinstance(payload, dict):
        return ""
    state = str(payload.get("state", "") or "")
    if state == "COMMIT":
        return str(payload.get("plate", "") or payload.get("candidate", "") or "")
    return str(payload.get("candidate", "") or payload.get("plate", "") or "")


def _webapp_candidate_state(payload):
    state = str(payload.get("state", "") or "")
    source = str(payload.get("source", "") or "")
    if state == "COMMIT":
        return "COMMIT"
    if "fusion" in source:
        return "FUSION"
    if state == "HOLD":
        return "HOLD"
    return "OCR"


def _append_webapp_recent_candidate(recent_candidates, payload):
    if recent_candidates is None or not isinstance(payload, dict):
        return
    try:
        state = str(payload.get("state", "") or "")
        if state == "WAITING":
            return
        text = _webapp_candidate_text(payload).strip()
        if not text or text == "-":
            return
        item = {
            "frame": _webapp_int_or_none(payload.get("frame_idx")),
            "track_id": payload.get("track_id"),
            "text": text,
            "state": _webapp_candidate_state(payload),
            "source": str(payload.get("source", "") or "-"),
            "confidence": payload.get("confidence"),
        }
        if recent_candidates:
            last = recent_candidates[-1]
            if (
                last.get("frame") == item["frame"]
                and last.get("text") == item["text"]
                and last.get("source") == item["source"]
            ):
                return
        recent_candidates.append(item)
    except Exception:
        return


def maybe_write_webapp_result(args, pipe, times, frame_idx, last_commit_key=None, recent_candidates=None):
    if not (getattr(args, "webapp_enable", False) and getattr(args, "webapp_result_json", None)):
        return last_commit_key

    try:
        payload = build_webapp_result_payload(pipe, times, frame_idx)
        state = str(payload.get("state", "WAITING"))
        _append_webapp_recent_candidate(recent_candidates, payload)
        payload_recent_candidates = [] if state == "WAITING" else list(recent_candidates or [])
        if state == "COMMIT":
            commit_key = (payload.get("track_id"), payload.get("plate"))
            if commit_key != last_commit_key:
                write_webapp_result_json(args.webapp_result_json, recent_candidates=payload_recent_candidates, **payload)
                return commit_key
            return last_commit_key

        update_every = max(1, int(getattr(args, "webapp_update_every_frames", 10) or 10))
        frame_value = _webapp_int_or_none(frame_idx)
        if frame_value is not None and frame_value % update_every == 0:
            write_webapp_result_json(args.webapp_result_json, recent_candidates=payload_recent_candidates, **payload)
    except Exception:
        return last_commit_key
    return last_commit_key


def _webapp_ascii_text(value, default="-"):
    text = str(default if value is None or value == "" else value)
    return text.encode("ascii", errors="replace").decode("ascii")


def _webapp_frame_to_bgr(frame_u8):
    try:
        frame = frame_u8
        if hasattr(frame, "detach"):
            frame = frame.detach()
            if getattr(frame, "ndim", 0) == 4:
                frame = frame[0]
            if getattr(frame, "ndim", 0) == 3 and int(frame.shape[0]) in {1, 3, 4}:
                frame = frame.permute(1, 2, 0)
            frame = frame.to("cpu")
            if str(getattr(frame, "dtype", "")) != "torch.uint8":
                frame = frame.clamp(0, 255).to(dtype=__import__("torch").uint8)
            image = frame.numpy()
        else:
            image = frame

        if image is None or getattr(image, "size", 0) == 0:
            return None
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if len(image.shape) == 3 and image.shape[2] >= 3:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
        return None
    except Exception:
        return None


def _webapp_boxes_to_list(boxes):
    if boxes is None:
        return []
    try:
        value = boxes
        if isinstance(value, tuple) and value:
            value = value[0]
        if hasattr(value, "detach"):
            value = value.detach().to("cpu")
            if getattr(value, "ndim", 0) == 1:
                value = value.reshape(1, -1)
            rows = value[:, :4].tolist()
        else:
            rows = value
        out = []
        for row in list(rows)[:20]:
            if len(row) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in row[:4]]
            out.append((x1, y1, x2, y2))
        return out
    except Exception:
        return []


def _normalize_webapp_preview_path(preview_path, preview_format: str | None = None) -> Path:
    path = Path(preview_path)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        return path

    normalized_format = str(preview_format or "jpg").strip().lower()
    suffix = ".png" if normalized_format == "png" else ".jpg"
    return path.with_suffix(suffix)


def _resize_webapp_preview_if_needed(image_bgr, max_width: int | None):
    try:
        limit = int(max_width or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0 or image_bgr is None:
        return image_bgr

    height, width = image_bgr.shape[:2]
    if width <= limit:
        return image_bgr

    scale = float(limit) / float(max(width, 1))
    target_size = (limit, max(1, int(round(height * scale))))
    return cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)


def write_webapp_preview_image(preview_path, image_bgr, *, preview_format: str | None = None, max_width: int | None = 960):
    """
    모바일 웹앱용 preview 이미지를 단일 파일로 안전하게 교체 저장한다.
    실패해도 메인 파이프라인은 중단하지 않는다.
    """
    if not preview_path or image_bgr is None:
        return

    try:
        path = _normalize_webapp_preview_path(preview_path, preview_format)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = _resize_webapp_preview_if_needed(image_bgr, max_width)
        tmp_path = path.parent / f"{path.stem}.tmp{path.suffix}"
        if cv2.imwrite(str(tmp_path), output):
            tmp_path.replace(path)
        elif tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    except Exception:
        return


def write_webapp_preview_jpg(
    preview_path,
    frame_bgr,
    *,
    frame_idx=None,
    state=None,
    plate=None,
    track_id=None,
    fps=None,
    boxes=None,
    preview_format: str | None = None,
    max_width: int | None = 960,
):
    """
    모바일 웹앱용 preview 이미지를 안전하게 저장한다.
    기존 함수명 호환성을 유지하면서 jpg/png 저장을 모두 지원한다.
    실패해도 메인 파이프라인은 중단하지 않는다.
    """
    if not preview_path or frame_bgr is None:
        return

    try:
        vis = frame_bgr.copy()

        for x1, y1, x2, y2 in _webapp_boxes_to_list(boxes):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lines = [
            f"FRAME: {frame_idx if frame_idx is not None else '-'}",
            f"STATE: {_webapp_ascii_text(state)}",
            f"PLATE: {_webapp_ascii_text(plate)}",
            f"TRACK: {track_id if track_id is not None else '-'}",
            f"FPS: {float(fps):.1f}" if fps is not None else "FPS: -",
        ]
        x, y = 16, 30
        for line in lines:
            cv2.putText(vis, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y += 34

        write_webapp_preview_image(
            preview_path,
            vis,
            preview_format=preview_format,
            max_width=max_width,
        )
    except Exception:
        return


def maybe_write_webapp_preview(args, pipe, times, frame_idx, frame_u8, step_out):
    if not (getattr(args, "webapp_enable", False) and getattr(args, "webapp_preview_jpg", None)):
        return
    try:
        update_every = max(1, int(getattr(args, "webapp_preview_every_frames", 10) or 10))
        frame_value = _webapp_int_or_none(frame_idx)
        if frame_value is None or frame_value % update_every != 0:
            return
        payload = build_webapp_result_payload(pipe, times, frame_idx)
        boxes = step_out[0] if isinstance(step_out, tuple) and len(step_out) >= 1 else None
        frame_bgr = _webapp_frame_to_bgr(frame_u8)
        write_webapp_preview_jpg(
            getattr(args, "webapp_preview_jpg", None),
            frame_bgr,
            frame_idx=frame_idx,
            state=payload.get("state"),
            plate=payload.get("plate") or payload.get("candidate"),
            track_id=payload.get("track_id"),
            fps=payload.get("fps"),
            boxes=boxes,
            preview_format=getattr(args, "webapp_preview_format", "jpg"),
            max_width=getattr(args, "webapp_preview_max_width", 960),
        )
    except Exception:
        return


def maybe_write_webapp_preview_early(args, frame_idx, frame_u8):
    """
    frame decode 직후 raw preview.jpg를 먼저 저장한다.
    OCR/FEBAM/fusion 상태를 조회하지 않아 heavy pipeline 지연과 독립적으로 동작한다.
    """
    if not (
        getattr(args, "webapp_enable", False)
        and getattr(args, "webapp_preview_jpg", None)
        and getattr(args, "webapp_preview_early", True)
    ):
        return
    try:
        update_every = max(1, int(getattr(args, "webapp_preview_every_frames", 10) or 10))
        frame_value = _webapp_int_or_none(frame_idx)
        if frame_value is None or frame_value % update_every != 0:
            return
        frame_bgr = _webapp_frame_to_bgr(frame_u8)
        write_webapp_preview_jpg(
            getattr(args, "webapp_preview_jpg", None),
            frame_bgr,
            frame_idx=frame_idx,
            state=None,
            plate=None,
            track_id=None,
            fps=None,
            boxes=None,
            preview_format=getattr(args, "webapp_preview_format", "jpg"),
            max_width=getattr(args, "webapp_preview_max_width", 960),
        )
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU-only license plate detection runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
String FEBAM 추가 옵션:
  --string-febam-sim-center
  --string-febam-sim-k
  --string-febam-min-len
  --string-febam-len-k
  --string-febam-release-thr
  --string-febam-switch-margin-thr
  --string-febam-switch-min-segments
  --string-febam-segment-slope
  --string-febam-segment-cap
  --string-febam-source-weight-weak
  --string-febam-source-weight-near
  --string-febam-source-weight-confirmed
  --string-febam-strict-korean-commit
  --no-string-febam-strict-korean-commit

출력 파일:
  --ocr-csv ../data/processed/debug/ocr_history.csv
    온라인 OCR/String FEBAM observation 및 final row 기록 CSV.
""",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--c3-mode", choices=["off", "shadow", "replay"], default="off", help="Research-only C3 observer; never changes production output")
    p.add_argument("--c3-shadow-output", default="outputs/c3_shadow/C3_DECISION_TRACE.json", help="C3 research shadow trace path")
    p.add_argument("--c3-replay-predictions", default="", help="Frozen OOF prediction CSV required only by --c3-mode replay")
    p.add_argument("--c3-live-dual-ocr", action="store_true", help="Shadow-only exact G6+PPv5 canonical crop observer")
    p.add_argument("--c3-live-g6", default="")
    p.add_argument("--c3-live-g6-config", default="")
    p.add_argument("--c3-live-ppv5", default="")
    p.add_argument("--c3-live-legal", default="")
    p.add_argument("--c3-live-output", default="outputs/c3_live_dual_ocr")
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu_only"])
    p.add_argument("--gpu-dummy-frame", action="store_true")
    p.add_argument("--gpu-test-image", default=None)
    p.add_argument("--gpu-test-image-repeat", type=int, default=1)
    p.add_argument("--gpu-profile", action="store_true")
    p.add_argument("--gpu-stage-profile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gpu-stage-profile-warmup-frames", type=int, default=8)
    p.add_argument("--gpu-stage-profile-max-frames", type=int, default=256)
    p.add_argument("--gpu-stage-profile-output", default="")
    p.add_argument(
        "--preprocess-routing-mode",
        choices=["baseline_full_frame", "yolo_primary_roi"],
        default="baseline_full_frame",
        help="Preprocess routing. yolo_primary_roi is rejected unless YOLO consumes raw/stabilized RGB.",
    )
    p.add_argument(
        "--yolo-input-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit one scalar-only tensor-lineage record for the first YOLO input.",
    )
    p.add_argument("--gpu-profile-frames", type=int, default=100)
    p.add_argument("--gpu-debug-stage-log", action="store_true")
    p.add_argument("--gpu-debug-log-txt", default=None)
    p.add_argument("--debug-start-sec", type=float, default=None)
    p.add_argument("--debug-end-sec", type=float, default=None)
    p.add_argument("--debug-dump-seconds", default="")
    p.add_argument("--debug-save-video", default=None)
    p.add_argument("--debug-save-frames-dir", default=None)
    p.add_argument("--debug-event-dir", default=None)
    p.add_argument("--debug-save-every-sec", type=float, default=1.0)
    p.add_argument("--debug-max-events", type=int, default=200)
    p.add_argument("--use-yolo-detector", action="store_true")
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--yolo-imgsz", type=int, default=960)
    p.add_argument("--yolo-conf", type=float, default=0.10)
    p.add_argument("--yolo-iou", type=float, default=0.45)
    p.add_argument("--yolo-half", action="store_true", default=False, help="Run YOLO detector inference in FP16")
    p.add_argument(
        "--yolo-batch-mode",
        choices=["disabled", "microbatch"],
        default="disabled",
        help="YOLO detector batching mode. microbatch batches consecutive frames before tracker/FEBAM replay.",
    )
    p.add_argument(
        "--yolo-batch-size",
        type=int,
        default=1,
        help="Run YOLO detector in temporal micro-batches. 1 keeps existing per-frame behavior.",
    )
    p.add_argument("--use-mlp-updater", action="store_true")
    p.add_argument("--mlp-weights", default=None)
    p.add_argument("--use-sigmoid-febam", action="store_true", default=True)
    p.add_argument("--no-gray-stretched-ocr", action="store_true")
    p.add_argument(
        "--scientific-g0",
        action="store_true",
        help="Use the evidence-frozen canonical source crop as the sole main OCR image; auxiliary transforms remain diagnostic only.",
    )
    p.add_argument("--ocr-ignore-febam", action="store_true", help="Debug OCR by attempting tracked candidates without FEBAM confirmation")
    p.add_argument("--ocr-small-sharpen-fallback", action="store_true", help="Enable optional small/blurred ROI upscaled-sharpened OCR fallback")
    p.add_argument("--ocr-save-debug-crops", action=argparse.BooleanOptionalAction, default=False, help="Save OCR ROI debug crop images (disabled by default to avoid I/O bottlenecks)")
    p.add_argument("--ocr-backend", choices=["easyocr", "fastplate", "ppocrv5", "wise_korean", "both", "none"], default="easyocr", help="OCR backend: EasyOCR, FastPlateOCR, Korean PP-OCRv5, final WiSE Korean Student, both, or skip OCR")
    p.add_argument("--ppocrv5-model-dir", default="", help="Optional local korean_PP-OCRv5_mobile_rec directory; empty uses official Paddle cache/download")
    p.add_argument("--bio-adaptive-ocr", action="store_true", help="Enable active Bio Route A after FastPlate digit-anchor recognition")
    p.add_argument("--bio-adaptive-shadow", action="store_true", help="Record bio quality/router decisions without changing OCR output")
    p.add_argument("--bio-router-mode", choices=["rule", "learned_shadow", "learned_active"], default="rule")
    p.add_argument("--bio-clear-backend", choices=["hog_lbp_svm", "fastplate_only", "disabled"], default="hog_lbp_svm")
    p.add_argument("--bio-medium-backend", choices=["dual_axis_svtr", "disabled"], default="disabled")
    p.add_argument("--bio-severe-backend", choices=["temporal_dual_axis_svtr", "disabled"], default="disabled")
    p.add_argument("--bio-fallback-backend", choices=["hold", "residual_sr", "onnx_sr"], default="hold")
    p.add_argument("--bio-svm-weights", default="")
    p.add_argument("--bio-svtr-model", default="")
    p.add_argument("--bio-svtr-config", default="")
    p.add_argument("--bio-sr-model", default="")
    p.add_argument("--bio-temporal-mode", choices=["causal", "symmetric"], default="causal")
    p.add_argument("--bio-temporal-window", type=int, default=5)
    p.add_argument("--bio-temporal-min-frames", type=int, default=3)
    p.add_argument("--bio-temporal-min-segments", type=int, default=2)
    p.add_argument("--bio-alignment-mode", choices=["bbox_affine", "phase_corr_gpu", "homography_plugin", "optical_flow_plugin"], default="bbox_affine")
    p.add_argument("--bio-save-debug", action="store_true")
    p.add_argument("--bio-debug-dir", default="")
    p.add_argument("--bio-no-cpu-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bio-clear-sharpness-thr", type=float, default=0.55)
    p.add_argument("--bio-clear-edge-thr", type=float, default=0.45)
    p.add_argument("--bio-clear-max-blur", type=float, default=0.45)
    p.add_argument("--bio-clear-max-ghost", type=float, default=0.35)
    p.add_argument("--bio-clear-min-bbox-stability", type=float, default=0.60)
    p.add_argument("--bio-clear-min-crop-height", type=int, default=24)
    p.add_argument("--bio-severe-blur-thr", type=float, default=0.68)
    p.add_argument("--bio-severe-ghost-thr", type=float, default=0.62)
    p.add_argument("--bio-route-a-min-conf", type=float, default=0.70)
    p.add_argument("--bio-route-a-min-margin", type=float, default=0.15)
    p.add_argument("--wise-korean-model",default=str(Path(__file__).resolve().parent/"models"/"final_wise_korean"/"final_wise_best.onnx"),help="Final WiSE Korean Student ONNX path")
    p.add_argument("--wise-korean-plate-config",default=str(Path(__file__).resolve().parent/"models"/"final_wise_korean"/"plate_config.yaml"),help="Final WiSE Korean Student plate config")
    p.add_argument("--wise-korean-device",choices=["cuda"],default="cuda",help="Final WiSE Student is GPU-only")
    p.add_argument("--fastplate-model", default="cct-s-v2-global-model", help="fast-plate-ocr hub OCR model name")
    p.add_argument("--fastplate-device", choices=["cuda", "cpu", "auto"], default="cuda", help="fast-plate-ocr runtime device")
    p.add_argument("--fastplate-batch-size", type=int, default=32, help="fast-plate-ocr batch size")
    p.add_argument("--fastplate-preload-torch-cuda-dlls", action=argparse.BooleanOptionalAction, default=True, help="Prepend torch/lib to PATH before fast-plate-ocr/onnxruntime import")
    p.add_argument("--fastplate-min-text-len", type=int, default=4, help="Minimum normalized FastPlateOCR text length to keep")
    p.add_argument("--fastplate-async", action=argparse.BooleanOptionalAction, default=True, help="Run FastPlateOCR through an async batch queue")
    p.add_argument("--fastplate-min-batch-size", type=int, default=4, help="Minimum preferred FastPlateOCR async batch size")
    p.add_argument("--fastplate-flush-timeout-ms", type=float, default=150.0, help="FastPlateOCR async batch flush timeout in milliseconds")
    p.add_argument("--fastplate-queue-max", type=int, default=512, help="Maximum queued FastPlateOCR crops")
    p.add_argument("--fastplate-save-debug-crops", action=argparse.BooleanOptionalAction, default=False, help="Keep FastPlateOCR async debug crop saving enabled")
    p.add_argument("--fastplate-rate-limit-relaxed", action=argparse.BooleanOptionalAction, default=False, help="Relax FastPlateOCR async per-track/variant rate limits for experimental fusion text evidence")
    p.add_argument("--fastplate-large-batch-mode", action=argparse.BooleanOptionalAction, default=False, help="Hold FastPlateOCR async crops longer to improve large-batch throughput")
    p.add_argument("--fastplate-target-batch-size", type=int, default=64, help="Preferred FastPlateOCR async target batch size before timeout flush")
    p.add_argument("--fastplate-max-flush-timeout-ms", type=float, default=1500.0, help="Maximum FastPlateOCR async wait before flushing at min batch size")
    p.add_argument(
        "--fastplate-runtime-preset",
        choices=["default", "single_camera_microbatch", "offline_throughput_b4", "frozen_lovo_b3"],
        default="default",
        help="Runtime preset. frozen_lovo_b3 replays FastPlate -> position evidence -> StringFEBAM in fail-closed shadow mode.",
    )
    p.add_argument("--fastplate-gpu-crop-batch", action="store_true", help="Prepare FastPlateOCR crop batches through the GPU-first tensor path when available")
    p.add_argument("--fastplate-tensor-runner", action="store_true", help="Enable the FastPlateOCR tensor runner scaffold")
    p.add_argument("--fastplate-tensor-runner-mode", choices=["disabled", "shadow", "active"], default="disabled", help="FastPlateOCR tensor runner mode")
    p.add_argument("--fastplate-ocr-input-h", type=int, default=96, help="FastPlateOCR tensor runner input height")
    p.add_argument("--fastplate-ocr-input-w", type=int, default=384, help="FastPlateOCR tensor runner input width")
    p.add_argument("--fastplate-parity-sample-limit", type=int, default=256, help="Maximum FastPlateOCR old/new parity rows to write")
    p.add_argument("--fastplate-parity-log-csv", default="", help="Optional CSV path for FastPlateOCR tensor parity diagnostics")
    p.add_argument("--fastplate-tensor-profile", action="store_true", help="Include FastPlateOCR tensor runner profile metrics")
    p.add_argument("--fastplate-direct-ort", action="store_true", help="Use direct ONNXRuntime CUDA runner for FastPlateOCR when ONNX model path and decoder are available.")
    p.add_argument("--fastplate-direct-ort-iobinding", action=argparse.BooleanOptionalAction, default=True, help="Enable ORT CUDA I/O binding for direct FastPlateOCR runner")
    p.add_argument("--fastplate-direct-ort-debug", action="store_true", help="Enable ORT profiling/debug for direct FastPlateOCR runner")
    p.add_argument("--fastplate-direct-ort-model-path", default="", help="Explicit FastPlateOCR ONNX model path or model alias for direct ORT runner")
    p.add_argument("--fastplate-direct-ort-input-h", type=int, default=0, help="Direct ORT FastPlateOCR input height override")
    p.add_argument("--fastplate-direct-ort-input-w", type=int, default=0, help="Direct ORT FastPlateOCR input width override")
    p.add_argument("--fastplate-direct-ort-compare-wrapper", action="store_true", help="Compare direct ORT output with FastPlateOCR wrapper in diagnostics")
    p.add_argument("--fastplate-preserve-crop-geometry", action=argparse.BooleanOptionalAction, default=False, help="Preserve OCR crop aspect ratio with symmetric letterbox padding after slicing the expanded bbox from the original frame")
    p.add_argument("--fastplate-custom-onnx", type=str, default=None, help="Custom FastPlateOCR ONNX model path. If provided, use this instead of named fastplate model.")
    p.add_argument("--fastplate-custom-plate-config", type=str, default=None, help="Plate config YAML for custom FastPlateOCR model.")
    p.add_argument("--fastplate-custom-input-width", type=int, default=256, help="Custom FastPlateOCR ONNX input width")
    p.add_argument("--fastplate-custom-input-height", type=int, default=64, help="Custom FastPlateOCR ONNX input height")
    p.add_argument("--dual-branch-ocr", action=argparse.BooleanOptionalAction, default=False, help="Use official global digits plus Korean student slot logits through two CUDA IOBinding sessions")
    p.add_argument("--dual-branch-global-onnx", default=str(Path(__file__).resolve().parent / "models" / "dual_branch" / "cct_xs_v2_global.onnx"), help="Official global teacher ONNX")
    p.add_argument("--dual-branch-korean-onnx", default=str(Path(__file__).resolve().parent / "models" / "dual_branch" / "korean_best.onnx"), help="Exported Korean student ONNX")
    p.add_argument("--dual-branch-hangul-weight", type=float, default=1.2, help="Hangul log-probability weight for H7/H8 hypothesis scoring")
    p.add_argument("--dual-branch-digit-alignment-mode", choices=["fixed_legacy", "monotonic_dp"], default="monotonic_dp")
    p.add_argument("--dual-branch-digit-mass-thr", type=float, default=0.50)
    p.add_argument("--dual-branch-digit-conf-thr", type=float, default=0.50)
    p.add_argument("--dual-branch-digit-margin-thr", type=float, default=0.10)
    p.add_argument("--dual-branch-global-input-variant", choices=["raw", "gray", "both"], default="raw")
    p.add_argument("--dual-branch-gray-secondary", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dual-branch-global-crop-mode", choices=["gpu_crop", "cpu_crop", "cpu_crop_official", "cpu_crop_y27", "cpu_crop_y27_exact"], default="gpu_crop")
    p.add_argument("--dual-branch-global-result-mode", choices=["digits_only", "shadow_only"], default="digits_only")
    p.add_argument("--dual-branch-global-cpu-crop", action="store_true")
    p.add_argument("--dual-branch-global-decode-mode", choices=["official_full_sequence", "constrained_legacy"], default="official_full_sequence")
    p.add_argument("--dual-branch-korean-combine-mode", choices=["positional_insert_replace", "h7_h8_legacy"], default="positional_insert_replace")
    p.add_argument("--dual-branch-temporal-mode", choices=["whole_string_legacy", "slot_probability"], default="slot_probability")
    p.add_argument("--dual-branch-global-length-policy", choices=["unrestricted", "diagnostic_only", "strict_legacy"], default="unrestricted")
    p.add_argument("--dual-branch-korean-mode", choices=["training_exact", "full_sequence_primary", "full_sequence_native_canvas", "disabled"], default="training_exact")
    p.add_argument("--dual-branch-korean-slot-mode", choices=["independent", "english_length_legacy"], default="independent")
    p.add_argument("--dual-branch-korean-mass-thr", type=float, default=0.50)
    p.add_argument("--dual-branch-korean-conf-thr", type=float, default=0.50)
    p.add_argument("--dual-branch-korean-margin-thr", type=float, default=0.10)
    p.add_argument("--dual-branch-length-margin-thr", type=float, default=0.15)
    p.add_argument("--dual-branch-length-hold", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--non-generative-restoration-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default-OFF canonical-event restoration observer; never changes final OCR/COMMIT",
    )
    p.add_argument(
        "--restoration-shadow-output",
        default="./outputs/non_generative_restoration_shadow.jsonl",
        help="Separate JSONL manifest for observation-only restoration results",
    )
    p.add_argument("--restoration-shadow-max-events", type=int, default=64)
    p.add_argument("--restoration-shadow-max-frames", type=int, default=5)
    p.add_argument("--restoration-shadow-min-frames", type=int, default=2)
    p.add_argument("--restoration-shadow-schedule-frames", type=int, default=3)
    p.add_argument("--fastplate-custom-fullplate-onnx", type=str, default=None, help="Alias for the 8-slot custom full-plate ONNX model")
    p.add_argument("--fastplate-custom-fullplate-plate-config", type=str, default=None, help="Plate config for the 8-slot full-plate model")
    p.add_argument("--fastplate-custom-fullplate-diag-only", action="store_true", help="Run custom full-plate OCR as diagnostic evidence only")
    p.add_argument("--middle-slot-preset", default="none", choices=available_middle_slot_presets(), help="Middle-slot preset. Use 'hog_lbp_v4_aux' to enable the final HOG/LBP v4 auxiliary Korean-slot branch.")
    p.add_argument("--middle-slot-upl", action="store_true", help="Enable Korean middle-slot UPL OCR post-adapter using GPU batch inference")
    p.add_argument("--middle-slot-device", default="cuda", help="MiddleSlotUPL torch device; realtime path defaults to cuda")
    p.add_argument("--middle-slot-batch-size", type=int, default=64, help="MiddleSlotUPL GPU batch size")
    p.add_argument("--middle-slot-min-batch-size", type=int, default=4, help="MiddleSlotUPL minimum preferred batch size")
    p.add_argument("--middle-slot-flush-timeout-ms", type=float, default=100.0, help="MiddleSlotUPL async batch flush timeout in milliseconds")
    p.add_argument("--middle-slot-queue-max", type=int, default=512, help="Maximum queued MiddleSlotUPL crops")
    p.add_argument("--middle-slot-input-size", type=int, default=48, help="MiddleSlotUPL square input size")
    p.add_argument("--middle-slot-max-crops-per-group", type=int, default=3, help="Maximum MiddleSlotUPL crops per pseudo vehicle/group")
    p.add_argument("--middle-slot-backend", default="upl", choices=["none", "easyocr", "upl", "hog_lbp_gpu"], help="Middle-slot OCR backend: upl keeps existing prototype/HOG path, easyocr uses EasyOCR Korean GPU, hog_lbp_gpu uses the v1 HOG/LBP GPU diagnostic branch.")
    p.add_argument("--middle-slot-cropper-mode", default="gpu_aspect_prior", choices=["gpu_aspect_prior", "gpu_aspect_prior_wide_core", "gpu_projection", "gpu_anchor_projection_refine"], help="MiddleSlot crop candidate generator mode")
    p.add_argument("--middle-slot-easyocr-single-crop", dest="middle_slot_easyocr_single_crop", action="store_true", default=True, help="Use only one middle-slot crop for EasyOCR backend.")
    p.add_argument("--no-middle-slot-easyocr-single-crop", dest="middle_slot_easyocr_single_crop", action="store_false", help="Allow multiple middle-slot crop candidates for EasyOCR backend.")
    p.add_argument("--middle-slot-wide-core-x-pad-ratio", type=float, default=0.16, help="MiddleSlot wide-core crop horizontal pad ratio")
    p.add_argument("--middle-slot-wide-core-y-pad-ratio", type=float, default=0.15, help="MiddleSlot wide-core crop vertical pad ratio")
    p.add_argument("--middle-slot-wide-core-min-width-ratio", type=float, default=0.14, help="Minimum plate-width ratio for wide-core crop")
    p.add_argument("--middle-slot-wide-core-max-width-ratio", type=float, default=0.34, help="Maximum plate-width ratio for wide-core crop")
    p.add_argument("--middle-slot-easyocr-gpu", action=argparse.BooleanOptionalAction, default=True, help="Use GPU for MiddleSlot EasyOCR backend")
    p.add_argument("--middle-slot-easyocr-batch-size", type=int, default=16, help="MiddleSlot EasyOCR recognition batch size")
    p.add_argument("--middle-slot-easyocr-min-conf", type=float, default=0.30, help="Minimum EasyOCR confidence required to append middle-slot evidence.")
    p.add_argument("--middle-slot-easyocr-allowlist", type=str, default="가거고구나너노누다더도두라러로루마머모무바버보부사서소수아어오우자저조주하허호배", help="Allowed Korean plate middle-slot characters for EasyOCR.")
    p.add_argument("--middle-slot-easyocr-resize-scale", type=int, default=6, help="Resize scale for small middle-slot crop before EasyOCR.")
    p.add_argument("--middle-slot-easyocr-border", type=int, default=50, help="White border size added around middle-slot crop before EasyOCR.")
    p.add_argument("--middle-slot-encoder-mode", default="hog_torch", choices=["tinycnn", "hog_torch"], help="MiddleSlotUPL encoder mode; must match the prototype bank encoder_mode")
    p.add_argument("--middle-slot-model-path", default=None, help="Optional MiddleSlotUPL encoder checkpoint path")
    p.add_argument("--middle-slot-prototype-path", default=None, help="Optional MiddleSlotUPL prototype bank path")
    p.add_argument("--middle-slot-source-weight", type=float, default=0.70, help="Base StringFEBAM source weight for MiddleSlotUPL evidence")
    p.add_argument("--middle-slot-hog-lbp-model", default=None, help="Path to v1 HOG/LBP GPU middle-slot checkpoint, e.g. ./runs/middle_slot/hog_lbp_gpu_middle_slot.pt")
    p.add_argument("--middle-slot-hog-lbp-topk", type=int, default=3, help="Top-k Korean middle-slot candidates to emit for the HOG/LBP GPU backend")
    p.add_argument("--middle-slot-hog-lbp-source-weight", type=float, default=0.10, help="Low soft-evidence source weight for HOG/LBP GPU middle-slot candidates")
    p.add_argument("--middle-slot-hog-lbp-min-conf", type=float, default=0.07, help="Minimum HOG/LBP top-1 confidence before adding soft evidence")
    p.add_argument("--middle-slot-hog-lbp-min-margin", type=float, default=0.005, help="Minimum HOG/LBP top1-top2 margin before adding soft evidence")
    p.add_argument("--middle-slot-hog-lbp-min-crop-score", type=float, default=0.40, help="Minimum middle-slot crop score before adding HOG/LBP soft evidence")
    p.add_argument("--middle-slot-v32", action="store_true", help="Enable gabor_scatter_lite_structured_v32 as a middle-slot soft evidence generator")
    p.add_argument("--middle-slot-v32-model", default=r".\runs\middle_slot\gabor_scatter_lite_structured_v32_runtime_fuzzyres_cache_v5.pt", help="Path to v3.2 middle-slot runtime checkpoint")
    p.add_argument("--middle-slot-v32-device", default="cuda", help="Middle-slot v3.2 torch device")
    p.add_argument("--middle-slot-v32-batch-size", type=int, default=512, help="Middle-slot v3.2 batch size")
    p.add_argument("--middle-slot-v32-min-batch-size", type=int, default=8, help="Minimum preferred v3.2 middle-slot batch size before flushing")
    p.add_argument("--middle-slot-v32-flush-every-frames", type=int, default=5, help="Flush pending v3.2 middle-slot crops every N frames")
    p.add_argument("--middle-slot-v32-topk", type=int, default=3, help="Middle-slot v3.2 top-k soft evidence count")
    p.add_argument("--middle-slot-v32-source-weight", type=float, default=0.04, help="Maximum String-FEBAM source weight for v3.2 middle-slot evidence")
    p.add_argument("--middle-slot-v32-min-conf", type=float, default=0.20, help="Minimum v3.2 top1 confidence for stronger soft evidence")
    p.add_argument("--middle-slot-v32-min-margin", type=float, default=0.03, help="Minimum v3.2 top1-top2 margin for stronger soft evidence")
    p.add_argument("--middle-slot-v32-max-crops-per-event", type=int, default=8, help="Maximum v3.2 middle-slot crops per event")
    p.add_argument("--middle-slot-v32-evidence-mode", choices=["topk_soft"], default="topk_soft", help="v3.2 evidence mode; hard commit is intentionally unsupported")
    p.add_argument("--middle-slot-v32-profile", action="store_true", help="Include middle-slot v3.2 profile counters in runtime summaries")
    p.add_argument("--middle-slot-conf-thr", type=float, default=0.70, help="MiddleSlotUPL evidence confidence threshold")
    p.add_argument("--middle-slot-margin-thr", type=float, default=0.12, help="MiddleSlotUPL top1-top2 margin threshold")
    p.add_argument("--middle-slot-sim-center", type=float, default=0.65, help="MiddleSlotUPL prototype similarity center")
    p.add_argument("--middle-slot-sim-k", type=float, default=12.0, help="MiddleSlotUPL prototype similarity gate steepness")
    p.add_argument("--middle-slot-alpha-min", type=float, default=0.85, help="MiddleSlotUPL prototype update alpha lower bound")
    p.add_argument("--middle-slot-alpha-max", type=float, default=0.995, help="MiddleSlotUPL prototype update alpha upper bound")
    p.add_argument("--middle-slot-debug-dir", default="../data/processed/debug/middle_slot", help="Directory for optional MiddleSlotUPL diagnostics")
    p.add_argument("--middle-slot-save-debug-crops", action="store_true", help="Save MiddleSlotUPL debug crops")
    p.add_argument("--middle-slot-disable-prototype-update", action="store_true", help="Disable online MiddleSlotUPL prototype updates")
    p.add_argument("--middle-slot-diagnostic-only", action="store_true", help="Run MiddleSlotUPL crop/prototype diagnostics without adding evidence rows or StringFEBAM observations")
    p.add_argument(
        "--middle-slot-gt-anchor-confused-fair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GT plate list only as teacher calibration to reweight middle-slot confused fair candidates. Enabled by default; use --no-middle-slot-gt-anchor-confused-fair to disable.",
    )
    p.add_argument(
        "--middle-slot-gt-plates-path",
        default=None,
        help="Path to txt/csv file containing GT plate strings for teacher calibration.",
    )
    p.add_argument(
        "--middle-slot-gt-anchor-boost",
        type=float,
        default=0.40,
        help="Score boost for GT-matched middle character when prefix/suffix digit anchor uniquely matches GT plate.",
    )
    p.add_argument(
        "--middle-slot-gt-confused-min-score",
        type=float,
        default=0.05,
        help="Minimum score for GT-anchor confused fair candidate.",
    )
    p.add_argument(
        "--middle-slot-gt-confused-min-support",
        type=int,
        default=1,
        help="Minimum support count before appending GT-anchor confused fair weak evidence. For today's demo default can be 1.",
    )
    p.add_argument("--event-roi-fusion", action=argparse.BooleanOptionalAction, default=False, help="Enable in-process event-level OCR ROI fusion bridge")
    p.add_argument("--event-roi-fusion-mode", choices=["ours_all_roi_febam", "b5_yolo_high_quality_fusion"], default="ours_all_roi_febam", help="Event ROI fusion comparison mode")
    p.add_argument("--b5-fusion-top-k", type=int, default=8, help="B5 YOLO high-quality crop fusion top-k")
    p.add_argument("--b5-fusion-min-quality-score", type=float, default=0.0, help="Minimum B5 YOLO crop quality score")
    p.add_argument("--b5-fusion-min-crops", type=int, default=2, help="Minimum B5 YOLO crops per event fusion")
    p.add_argument("--b5-fusion-output-tag", default="b5_yolo_high_quality_fusion", help="Output tag for B5 YOLO high-quality fusion rows")
    p.add_argument("--event-roi-fusion-preset", choices=["none", "trial020", "trial013", "trial013_fixed", "trial013_fixed_motion", "trial013_fixed_motion_color"], default="none", help="Event ROI fusion preset")
    p.add_argument("--event-roi-fusion-debug-dir", default="../data/processed/debug/event_fusion", help="Directory for event ROI fusion debug images and manifest")
    p.add_argument("--event-roi-fusion-save-debug", action=argparse.BooleanOptionalAction, default=False, help="Save fused event ROI images")
    p.add_argument("--event-roi-fusion-save-manifest", action=argparse.BooleanOptionalAction, default=False, help="Save event ROI fusion manifest CSV")
    p.add_argument("--event-roi-fusion-source-weight", type=float, default=0.55, help="String-FEBAM source_weight for trial020 fused observations")
    p.add_argument("--fusion-center-pad-ratio", type=float, default=0.15, help="Trial020 center alignment pad ratio; kept at 0.15 for compatibility")
    p.add_argument("--event-roi-fusion-ocr", action=argparse.BooleanOptionalAction, default=False, help="OCR every event fusion image with a dedicated FastPlate batch buffer")
    p.add_argument(
        "--event-roi-fusion-ocr-worker-mode",
        choices=["shared_async", "legacy_separate"],
        default="shared_async",
        help="shared_async reuses the primary Final WiSE queue/worker/ORT session; legacy_separate keeps the old second recognizer.",
    )
    p.add_argument("--event-roi-fusion-ocr-backend", choices=["fastplate"], default="fastplate", help="OCR backend for event fusion images")
    p.add_argument("--event-roi-fusion-ocr-source-weight", type=float, default=0.65, help="String-FEBAM source_weight for event fusion image OCR observations")
    p.add_argument("--event-roi-fusion-ocr-batch-size", type=int, default=32, help="FastPlate batch size for event fusion image OCR")
    p.add_argument("--event-roi-fusion-ocr-flush-every-frames", type=int, default=5, help="Flush event fusion OCR buffer at this frame interval")
    p.add_argument("--export-group-review", action="store_true", help="Export existing Motion-FEBAM group crops for offline human labeling")
    p.add_argument("--group-review-output", default="./outputs/group_review")
    p.add_argument("--group-review-video-id", default="", help="Stable video_id; defaults to input filename stem")
    p.add_argument("--group-review-max-crops", type=int, default=6)
    p.add_argument("--group-review-min-frame-gap", type=int, default=5)
    p.add_argument("--group-review-dedup-threshold", type=float, default=0.95)
    p.add_argument("--group-review-save-raw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--group-review-save-normalized", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--group-review-save-context", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--group-review-use-existing-ocr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--group-review-debug", action="store_true")
    p.add_argument("--labeling-febam-sink-dir", default="", help="Default-OFF metadata-only YOLO/pre-OCR FEBAM observation sink")
    p.add_argument("--easyocr-recognize-only", action="store_true", default=True, help="Prefer EasyOCR recognize() on the full plate crop as one text box")
    p.add_argument("--no-easyocr-recognize-only", action="store_false", dest="easyocr_recognize_only", help="Use legacy EasyOCR readtext() path first")
    p.add_argument("--easyocr-batch-size", type=int, default=16, help="EasyOCR recognition batch_size for recognize/readtext calls")
    p.add_argument("--easyocr-workers", type=int, default=0, help="EasyOCR workers for recognize/readtext calls")
    p.add_argument("--easyocr-readtext-fallback", action="store_true", default=True, help="Fallback to EasyOCR readtext() when recognize-only is empty or low confidence")
    p.add_argument("--no-easyocr-readtext-fallback", action="store_false", dest="easyocr_readtext_fallback", help="Disable readtext fallback after recognize-only failure")
    p.add_argument("--ocr-expand-x", type=float, default=0.20, help="Default horizontal OCR crop expansion ratio")
    p.add_argument("--ocr-expand-top-strong", type=float, default=0.30, help="Strong top OCR crop expansion ratio for small/blur/confirmed plates")
    p.add_argument("--ocr-variant-tiered", action=argparse.BooleanOptionalAction, default=True, help="Run OCR variants in tiered early-stop mode instead of always trying every variant")
    p.add_argument("--ocr-max-variants-per-candidate", type=int, default=3, help="Default max OCR variants per candidate in tiered mode")
    p.add_argument("--ocr-max-fallback-variants", type=int, default=5, help="Max OCR variants for confirmed/repeated-failure fallback candidates")
    p.add_argument("--ocr-max-readtext-fallbacks-per-candidate", type=int, default=1, help="Max recognize-only to readtext fallbacks per candidate")
    p.add_argument("--ocr-disable-heavy-variants", action=argparse.BooleanOptionalAction, default=True, help="Disable mild CLAHE/unsharp/binary heavy OCR variants by default")
    p.add_argument("--ocr-heavy-variants-only-confirmed", action=argparse.BooleanOptionalAction, default=True, help="Use heavy OCR variants only for near-confirmed/FEBAM-confirmed or repeated-failure tracks")
    p.add_argument("--easyocr-mode-policy", choices=["recognize_first", "readtext_first", "readtext_only", "auto"], default="auto", help="EasyOCR mode policy per OCR variant")
    p.add_argument("--string-febam", action="store_true", help="Enable Temporal Segment-aware String FEBAM after OCR")
    p.add_argument("--string-febam-group-key-mode", choices=["track", "event_fusion_group"], default="track")
    p.add_argument("--string-febam-alpha", type=float, default=0.30, help="Soft MRS frame-weight floor alpha for String FEBAM")
    p.add_argument("--string-febam-sim-thr", type=float, default=0.60, help="Temporal segment grouping similarity threshold")
    p.add_argument("--string-febam-node-merge-thr", type=float, default=0.65, help="String FEBAM node merge threshold")
    p.add_argument("--string-febam-cluster-thr", type=float, default=0.60, help="String FEBAM cluster similarity threshold")
    p.add_argument("--string-febam-commit-thr", type=float, default=0.70, help="String FEBAM COMMIT activation threshold")
    p.add_argument("--string-febam-margin-thr", type=float, default=0.15, help="String FEBAM score margin threshold")
    p.add_argument("--string-febam-debug", action="store_true", help="Print String FEBAM node/segment debug logs")
    p.add_argument("--string-febam-skip-after-commit", action="store_true", help="After COMMIT, OCR only every N frames for verification")
    p.add_argument("--string-febam-sim-center", type=float, default=0.65, help="Sigmoid center for string similarity evidence shaping.")
    p.add_argument("--string-febam-sim-k", type=float, default=10.0, help="Sigmoid steepness for string similarity evidence shaping.")
    p.add_argument("--string-febam-min-len", type=float, default=6.0, help="Minimum length center for partial text length sigmoid penalty.")
    p.add_argument("--string-febam-len-k", type=float, default=2.0, help="Sigmoid steepness for partial text length penalty.")
    p.add_argument("--string-febam-release-thr", type=float, default=0.55, help="Activation threshold for keeping an already committed plate.")
    p.add_argument("--string-febam-switch-margin-thr", type=float, default=0.25, help="Required margin to switch from an already committed plate to a new one.")
    p.add_argument("--string-febam-switch-min-segments", type=int, default=2, help="Minimum segment support required to switch committed plate.")
    p.add_argument("--string-febam-segment-slope", type=float, default=0.25, help="Effective count slope: 1 + slope * (segment_len - 1).")
    p.add_argument("--string-febam-segment-cap", type=float, default=1.75, help="Maximum segment effective count.")
    p.add_argument("--string-febam-source-weight-weak", type=float, default=0.35, help="Source weight for weak_plate_sample OCR observations.")
    p.add_argument("--string-febam-source-weight-near", type=float, default=0.70, help="Source weight for near_confirmed OCR observations.")
    p.add_argument("--string-febam-source-weight-confirmed", type=float, default=1.00, help="Source weight for febam_confirmed OCR observations.")
    p.add_argument("--string-febam-strict-korean-commit", action="store_true", default=True, help=r"Require Korean plate format for COMMIT: \d{2,3}[가-힣]\d{4}, with controlled fallback for korean_center_reocr.")
    p.add_argument("--no-string-febam-strict-korean-commit", action="store_false", dest="string_febam_strict_korean_commit")
    p.add_argument(
        "--ocr-csv",
        type=str,
        default=None,
        help="Save per-track OCR history and final voting result to CSV",
    )
    p.add_argument(
        "--webapp-result-json",
        type=str,
        default=None,
        help='모바일 웹앱이 polling할 result.json 저장 경로. 예: "../data/processed/webapp/result.json"',
    )
    p.add_argument(
        "--webapp-update-every-frames",
        type=int,
        default=10,
        help="WAITING/HOLD 상태를 몇 프레임마다 JSON으로 갱신할지 지정. COMMIT 상태는 즉시 저장",
    )
    p.add_argument(
        "--webapp-enable",
        action="store_true",
        help="이 옵션과 --webapp-result-json이 함께 지정된 경우에만 웹앱 JSON 저장 활성화",
    )
    p.add_argument(
        "--webapp-recent-candidates",
        type=int,
        default=5,
        help="웹앱 result.json에 포함할 최근 OCR 후보 개수. 0이면 recent_candidates를 비움",
    )
    p.add_argument(
        "--webapp-preview-jpg",
        type=str,
        default=None,
        help='모바일 웹앱에서 표시할 preview.jpg 저장 경로. 예: "../data/processed/webapp/preview.jpg"',
    )
    p.add_argument(
        "--webapp-preview-from-debug-frame",
        action="store_true",
        default=False,
        help="debug frame 저장 시점에 동일 이미지를 webapp preview 파일로도 갱신한다",
    )
    p.add_argument(
        "--webapp-preview-format",
        choices=["jpg", "png"],
        default="jpg",
        help="웹앱 preview 저장 포맷. 속도 우선은 jpg, 품질/무손실은 png",
    )
    p.add_argument(
        "--webapp-preview-max-width",
        type=int,
        default=960,
        help="웹앱 preview 이미지 최대 폭. 원본이 더 크면 비율 유지 축소 후 저장",
    )
    p.add_argument(
        "--webapp-preview-every-frames",
        type=int,
        default=1,
        help="몇 프레임마다 preview.jpg를 저장할지 지정. MJPEG 스트리밍용 기본값은 매 프레임 저장",
    )
    p.add_argument(
        "--webapp-preview-early",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="frame decode 직후 raw preview.jpg를 먼저 저장한다",
    )
    p.add_argument(
        "--webapp-preview-overlay-after-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pipe.step() 이후 bbox/state overlay preview를 추가 저장할지 여부",
    )
    p.add_argument(
        "--viewer-runtime-mode",
        choices=["off", "snapshot", "jsonl", "snapshot_jsonl"],
        default="off",
        help="External GPU-runtime Viewer observer. Explicit opt-in; never changes OCR/COMMIT.",
    )
    p.add_argument(
        "--viewer-runtime-snapshot",
        default="outputs/viewer_runtime/latest.json",
        help="Atomic latest-state JSON produced from gpu_pipeline runtime state.",
    )
    p.add_argument(
        "--viewer-runtime-jsonl",
        default="outputs/viewer_runtime/events.jsonl",
        help="Append-only scalar runtime event stream for the external Viewer.",
    )
    p.add_argument("--viewer-runtime-assets", default="outputs/viewer_runtime/assets", help="At most five synchronized preview frames per runtime vehicle group")
    args = p.parse_args()
    if args.ocr_backend == "wise_korean":
        args.ocr_backend = "fastplate"
        args.fastplate_custom_onnx = args.wise_korean_model
        args.fastplate_custom_plate_config = args.wise_korean_plate_config
        args.fastplate_device = args.wise_korean_device
        args.fastplate_custom_input_width = 256
        args.fastplate_custom_input_height = 64
        args.fastplate_custom_fullplate_diag_only = False
        model_display = str(args.wise_korean_model).encode("ascii", "backslashreplace").decode("ascii")
        print(f"[WISE_KOREAN_REQUESTED] model={model_display} runtime_mode=student_only_temporal")
    if getattr(args, "fastplate_custom_fullplate_onnx", None):
        if args.fastplate_custom_onnx and args.fastplate_custom_onnx != args.fastplate_custom_fullplate_onnx:
            p.error("use only one custom ONNX path")
        args.fastplate_custom_onnx = args.fastplate_custom_fullplate_onnx
    if getattr(args, "fastplate_custom_fullplate_plate_config", None):
        args.fastplate_custom_plate_config = args.fastplate_custom_fullplate_plate_config
    if getattr(args, "fastplate_custom_onnx", None):
        args.fastplate_tensor_runner = False
        args.fastplate_tensor_runner_mode = "disabled"
        args.fastplate_direct_ort = False
        onnx_display = str(args.fastplate_custom_onnx).encode("ascii", "backslashreplace").decode("ascii")
        config_display = str(getattr(args, "fastplate_custom_plate_config", None)).encode("ascii", "backslashreplace").decode("ascii")
        print(f"[CUSTOM_FASTPLATE_REQUESTED] onnx={onnx_display}")
        print(f"[CUSTOM_FASTPLATE_REQUESTED] plate_config={config_display}")
    if getattr(args, "fastplate_direct_ort", False) and not getattr(args, "fastplate_custom_onnx", None):
        args.fastplate_tensor_runner = True
        if getattr(args, "fastplate_tensor_runner_mode", "disabled") == "disabled":
            args.fastplate_tensor_runner_mode = "active"
        if getattr(args, "fastplate_direct_ort_model_path", "") and not getattr(args, "fastplate_custom_onnx", None):
            args.fastplate_model = args.fastplate_direct_ort_model_path
        if getattr(args, "fastplate_direct_ort_input_h", 0):
            args.fastplate_ocr_input_h = args.fastplate_direct_ort_input_h
        if getattr(args, "fastplate_direct_ort_input_w", 0):
            args.fastplate_ocr_input_w = args.fastplate_direct_ort_input_w
    return args




def _read_input_fps(input_path: str, logger: StageLogger, default: float = 25.0) -> float:
    cap = cv2.VideoCapture(input_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    finally:
        cap.release()
    if fps <= 1e-3:
        logger.log(f"WARNING: input_fps_unavailable fallback_fps={default:.3f}")
        return default
    logger.log(f"video info fps={fps:.3f}")
    return fps


def _safe_float_or_none(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _print_profile_summary(rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    keys = [
        "stage0_decode_ms",
        "stage1_stabilize_ms",
        "stage2_preprocess_ms",
        "stage3_structure_tensor_ms",
        "stage4_morphology_ms",
        "stage5_ccl_ms",
        "stage6_9_head_ms",
        "stage10_tracker_ms",
        "stage11_febam_ms",
        "stage12_ocr_trigger_ms",
        "stage13_writer_enqueue_ms",
        "total_frame_ms",
        "fps",
        "stage2_gray_ms",
        "stage2_fuzzy_clahe_ms",
        "stage2_sobel_ms",
        "stage2_blackhat_ms",
        "stage4_reconstruction_iter",
        "stage4_early_stop",
        "stage4_changed_count",
        "stage4_orientation_bins",
        "stage4_overconnect_risk",
        "stage4_component_evidence",
        "stage5_component_count",
        "stage5_merge_iterations",
        "stage5_attrs_rows",
        "stage5_attrs_cols",
        "stage5_threshold_levels",
        "stage6_raw_component_count",
        "stage6_filter_pass_count",
        "stage6_nms_pre_count",
        "stage6_nms_post_count",
        "stage6_topk_count",
        "stage6_yolo_used",
        "stage6_yolo_count",
        "yolo_batch_enabled",
        "yolo_batch_size",
        "yolo_batch_call_count",
        "yolo_batch_item_count",
        "yolo_batch_avg_size",
        "yolo_batch_last_size",
        "yolo_batch_fallback_single_count",
        "stage7_mlp_used",
        "stage7_mlp_top_score",
        "stage11_febam_mode_id",
        "stage12_gray_stretched_ocr_used",
        "fastplate_async_enqueued",
        "fastplate_async_processed",
        "fastplate_batch_call_count",
        "fastplate_avg_batch_size",
        "fastplate_batch_size",
        "fastplate_min_batch_size",
        "fastplate_large_batch_mode",
        "fastplate_target_batch_size",
        "fastplate_flush_timeout_ms",
        "fastplate_max_flush_timeout_ms",
        "fastplate_batch_efficiency_ratio",
        "fastplate_total_ocr_ms",
        "fastplate_avg_ocr_ms_per_crop",
        "fastplate_queue_dropped",
        "fastplate_queue_drop_weak",
        "fastplate_queue_drop_duplicate",
        "fastplate_queue_drop_rate_limited",
        "fastplate_queue_drop_backpressure",
        "fastplate_result_delay_avg_frames",
        "fastplate_result_delay_avg_ms",
        "fastplate_queue_max_seen",
        "fastplate_queue_depth",
        "fastplate_oldest_age_ms",
        "fastplate_gpu_crop_batch_enabled",
        "fastplate_custom_onnx_enabled",
        "fastplate_custom_onnx_path",
        "fastplate_custom_provider",
        "fastplate_custom_cuda_used",
        "fastplate_custom_input_name",
        "fastplate_custom_output_names",
        "fastplate_custom_input_shape",
        "fastplate_custom_output_shapes",
        "fastplate_custom_input_layout",
        "fastplate_custom_alphabet_len",
        "fastplate_custom_batch_call_count",
        "fastplate_custom_item_count",
        "fastplate_custom_avg_batch_size",
        "fastplate_custom_total_ms",
        "fastplate_custom_avg_ms_per_crop",
        "fastplate_custom_preprocess_ms",
        "fastplate_custom_infer_ms",
        "fastplate_custom_decode_ms",
        "fastplate_custom_error_count",
        "dual_branch_enabled",
        "dual_branch_session_count",
        "dual_branch_global_infer_ms",
        "dual_branch_korean_infer_ms",
        "dual_branch_iobinding_used",
        "dual_branch_h7_selected",
        "dual_branch_h8_selected",
        "dual_global_digits_valid_count",
        "dual_global_digits_rejected_count",
        "dual_branch_alignment_rejected",
        "dual_branch_digit_alignment_candidate_rejected",
        "dual_branch_hangul_candidate_rejected",
        "dual_branch_low_digit_mass_rejected",
        "dual_branch_forced_digit_count",
        "dual_branch_avg_digit_mass",
        "dual_branch_avg_digit_margin",
        "dual_branch_length_ambiguity_hold_count",
        "dual_branch_cpu_fallback_count",
        "dual_branch_warm_item_count",
        "dual_branch_warm_avg_ms_per_plate",
        "dual_branch_warm_call_p95_ms",
        "dual_branch_warm_p95_ms_per_plate",
        "dual_branch_length_margin_thr_configured",
        "dual_branch_length_margin_thr_effective",
        "dual_branch_length_hold_enabled",
        "fastplate_tensor_runner_enabled",
        "fastplate_tensor_runner_mode",
        "fastplate_tensor_runner_mode_id",
        "fastplate_tensor_runner_active_used",
        "fastplate_tensor_runner_shadow_count",
        "fastplate_gpu_crop_count",
        "fastplate_cpu_crop_count",
        "fastplate_gpu_crop_ms",
        "fastplate_cpu_crop_ms",
        "fastplate_crop_cpu_copy_count",
        "fastplate_crop_gpu_tensor_count",
        "fastplate_tensor_batch_call_count",
        "fastplate_tensor_avg_batch_size",
        "fastplate_tensor_total_ms",
        "fastplate_tensor_avg_ms_per_crop",
        "fastplate_tensor_preprocess_ms",
        "fastplate_tensor_infer_ms",
        "fastplate_tensor_decode_ms",
        "fastplate_tensor_iobinding_used",
        "fastplate_tensor_cpu_fallback_count",
        "fastplate_tensor_error_count",
        "fastplate_tensor_direct_backend_available",
        "fastplate_tensor_backend_kind",
        "fastplate_tensor_gpu_only_claim_valid",
        "fastplate_tensor_active_requested",
        "fastplate_tensor_active_success_count",
        "fastplate_tensor_active_blocked_count",
        "fastplate_tensor_active_fallback_used",
        "fastplate_tensor_active_fallback_reason",
        "final_vehicle_sparse_route_count",
        "final_vehicle_posterior_route_count",
        "final_vehicle_identity_hold_count",
        "final_vehicle_last_n_valid",
        "final_vehicle_last_posterior_active",
        "fastplate_tensor_last_error_type",
        "fastplate_tensor_last_error_message",
        "fastplate_tensor_last_error_stage",
        "fastplate_tensor_error_sample_input_shape",
        "fastplate_tensor_error_sample_input_dtype",
        "fastplate_tensor_error_sample_output_shapes",
        "fastplate_tensor_error_sample_batch_len",
        "fastplate_tensor_provider",
        "fastplate_tensor_input_name",
        "fastplate_tensor_output_names",
        "fastplate_tensor_preprocess_mode",
        "fastplate_tensor_decoder_mode",
        "fastplate_tensor_last_input_shape",
        "fastplate_tensor_last_input_dtype",
        "fastplate_tensor_prepare_valid_count",
        "fastplate_tensor_prepare_failed_count",
        "fastplate_tensor_prepare_failed_examples",
        "fastplate_tensor_preprocess_resize_count",
        "fastplate_parity_sample_count",
        "fastplate_parity_success_count",
        "fastplate_parity_error_count",
        "fastplate_parity_skipped_count",
        "fastplate_parity_text_match_count",
        "fastplate_parity_text_match_rate",
        "fastplate_parity_normalized_match_rate",
        "fastplate_parity_korean_slot_match_rate",
        "fastplate_parity_conf_abs_diff_mean",
        "fastplate_parity_effective_match_rate",
        "middle_slot_queue_dropped",
        "middle_slot_avg_ms_per_crop",
        "middle_slot_total_infer_ms",
        "middle_slot_avg_batch_size",
        "middle_slot_batch_call_count",
        "middle_slot_processed",
        "middle_slot_enqueued",
        "middle_slot_update_skipped_count",
        "middle_slot_update_applied_count",
        "middle_slot_skip_no_prototype",
        "middle_slot_cuda_used",
        "middle_slot_enabled",
        "middle_slot_backend",
        "middle_slot_hog_lbp_model",
        "middle_slot_hog_lbp_topk",
        "middle_slot_hog_lbp_source_weight",
        "middle_slot_hog_lbp_min_conf",
        "middle_slot_hog_lbp_min_margin",
        "middle_slot_hog_lbp_min_crop_score",
        "middle_slot_prototype_label_count",
        "middle_slot_easyocr_single_crop",
        "middle_slot_easyocr_processed",
        "middle_slot_easyocr_batch_call_count",
        "middle_slot_easyocr_avg_batch_size",
        "middle_slot_easyocr_total_ms",
        "middle_slot_easyocr_avg_ms_per_crop",
        "middle_slot_easyocr_error_count",
        "middle_slot_wide_core_x_pad_ratio",
        "middle_slot_wide_core_y_pad_ratio",
        "middle_slot_wide_core_crop_count",
        "middle_slot_evidence_added",
        "middle_slot_evidence_skipped",
        "middle_slot_gt_anchor_confused_fair",
        "middle_slot_gt_plate_count",
        "middle_slot_gt_anchor_boost",
        "middle_slot_gt_confused_evidence_added",
        "middle_slot_gt_confused_evidence_skipped",
        "middle_slot_results_drained",
        "middle_slot_queue_depth",
        "middle_slot_diagnostic_only",
        "middle_slot_evidence_diagnostic_only_skipped",
        "middle_slot_trigger_seen",
        "middle_slot_trigger_crop_present",
        "middle_slot_trigger_text_present",
        "middle_slot_trigger_fallback_text_used",
        "middle_slot_trigger_skipped_no_crop",
        "middle_slot_trigger_skipped_no_text",
        "middle_slot_trigger_worker_none",
        "middle_slot_trigger_enqueue_success",
        "middle_slot_trigger_enqueue_failed",
        "middle_slot_trigger_worker_init_error",
        "middle_slot_v32_enabled",
        "middle_slot_v32_processed",
        "middle_slot_v32_batch_call_count",
        "middle_slot_v32_avg_batch_size",
        "middle_slot_v32_min_batch_size",
        "middle_slot_v32_flush_every_frames",
        "middle_slot_v32_pending_final_flush_count",
        "middle_slot_v32_total_infer_ms",
        "middle_slot_v32_avg_ms_per_crop",
        "middle_slot_v32_added_evidence",
        "middle_slot_v32_skipped_low_conf",
        "middle_slot_v32_skipped_low_margin",
        "middle_slot_v32_skipped_no_event_id",
        "middle_slot_v32_skipped_max_crops_per_event",
        "middle_slot_v32_skipped_empty_crop",
        "middle_slot_v32_skipped_duplicate",
        "middle_slot_v32_skipped_pending_not_flushed",
        "middle_slot_v32_skipped_model_error",
        "middle_slot_v32_skipped_before_infer_unknown",
        "middle_slot_v32_skipped_other",
        "wall_clock_elapsed_sec",
        "wall_clock_effective_fps",
        "wall_clock_ms_per_frame",
        "debug_video_duration_sec",
        "processing_speed_x",
        "ocr_session_create_count",
        "ocr_worker_create_count",
        "ocr_shared_queue_enabled",
        "fusion_ocr_shared_worker_enabled",
        "fusion_ocr_separate_session_create_count",
        "fusion_ocr_enqueued",
        "fusion_ocr_processed",
        "fusion_ocr_dropped",
        "fusion_ocr_duplicate_blocked",
        "fusion_ocr_commit_blocked",
        "fusion_ocr_result_applied",
        "fusion_ocr_result_stale",
        "fusion_ocr_avg_queue_delay_ms",
        "fusion_ocr_avg_infer_ms",
        "fusion_ocr_max_per_event_seen",
        "fusion_ocr_cpu_input_count",
    ]

    avg = {}
    numeric_counts = {}
    skipped_counts = {}
    for k in keys:
        vals = []
        skipped = 0
        for r in rows:
            if not isinstance(r, dict):
                skipped += 1
                continue
            val = _safe_float_or_none(r.get(k))
            if val is None:
                skipped += 1
                continue
            vals.append(val)
        if vals:
            avg[k] = sum(vals) / len(vals)
            numeric_counts[k] = len(vals)
            skipped_counts[k] = skipped

    if not avg:
        sys.stdout.write("[PROFILE_SUMMARY] no numeric profile rows; skipped\n")
        return

    latest = rows[-1] if isinstance(rows[-1], dict) else {}
    latest_num = {k: _safe_float_or_none(latest.get(k)) for k in keys}

    def latest_float(k: str, default: float = 0.0) -> float:
        val = latest_num.get(k)
        if val is None:
            val = _safe_float_or_none(latest.get(k))
        return default if val is None else float(val)

    gpu_pipeline_ms = (
        avg.get("stage1_stabilize_ms", 0.0)
        + avg.get("stage2_preprocess_ms", 0.0)
        + avg.get("stage3_structure_tensor_ms", 0.0)
        + avg.get("stage4_morphology_ms", 0.0)
        + avg.get("stage5_ccl_ms", 0.0)
        + avg.get("stage6_9_head_ms", 0.0)
        + avg.get("stage10_tracker_ms", 0.0)
        + avg.get("stage11_febam_ms", 0.0)
        + avg.get("stage12_ocr_trigger_ms", 0.0)
    )
    ms_keys = [k for k in keys if k.endswith("_ms") and k in avg]
    bottleneck_stage = max(ms_keys, key=lambda k: avg[k]) if ms_keys else ""

    sys.stdout.write(
        "PROFILE_SUMMARY "
        + str(
            {
                "frames_processed": len(rows),
                "avg_total_ms": round(avg.get("total_frame_ms", 0.0), 3),
                "avg_stage0_decode_ms": round(avg.get("stage0_decode_ms", 0.0), 3),
                "avg_gpu_pipeline_ms": round(gpu_pipeline_ms, 3),
                "bottleneck_stage": bottleneck_stage,
                "effective_fps": round(avg.get("fps", 0.0), 3),
                "wall_clock_elapsed_sec": round(latest_float("wall_clock_elapsed_sec"), 3),
                "wall_clock_effective_fps": round(latest_float("wall_clock_effective_fps"), 3),
                "wall_clock_ms_per_frame": round(latest_float("wall_clock_ms_per_frame"), 3),
                "debug_video_duration_sec": round(latest_float("debug_video_duration_sec"), 3),
                "processing_speed_x": round(latest_float("processing_speed_x"), 3),
                "ocr_session_create_count": round(latest_float("ocr_session_create_count"), 3),
                "ocr_worker_create_count": round(latest_float("ocr_worker_create_count"), 3),
                "ocr_shared_queue_enabled": round(latest_float("ocr_shared_queue_enabled"), 3),
                "fusion_ocr_shared_worker_enabled": round(latest_float("fusion_ocr_shared_worker_enabled"), 3),
                "fusion_ocr_separate_session_create_count": round(latest_float("fusion_ocr_separate_session_create_count"), 3),
                "fusion_ocr_enqueued": round(latest_float("fusion_ocr_enqueued"), 3),
                "fusion_ocr_processed": round(latest_float("fusion_ocr_processed"), 3),
                "fusion_ocr_dropped": round(latest_float("fusion_ocr_dropped"), 3),
                "fusion_ocr_duplicate_blocked": round(latest_float("fusion_ocr_duplicate_blocked"), 3),
                "fusion_ocr_commit_blocked": round(latest_float("fusion_ocr_commit_blocked"), 3),
                "fusion_ocr_result_applied": round(latest_float("fusion_ocr_result_applied"), 3),
                "fusion_ocr_result_stale": round(latest_float("fusion_ocr_result_stale"), 3),
                "fusion_ocr_avg_queue_delay_ms": round(latest_float("fusion_ocr_avg_queue_delay_ms"), 3),
                "fusion_ocr_avg_infer_ms": round(latest_float("fusion_ocr_avg_infer_ms"), 3),
                "fusion_ocr_max_per_event_seen": round(latest_float("fusion_ocr_max_per_event_seen"), 3),
                "fusion_ocr_cpu_input_count": round(latest_float("fusion_ocr_cpu_input_count"), 3),
                "preprocess_routing_mode": latest.get("preprocess_routing_mode", "baseline_full_frame"),
                "yolo_input_source": latest.get("yolo_input_source", ""),
                "yolo_input_preprocess_source": latest.get("yolo_input_preprocess_source", ""),
                "yolo_input_shape": latest.get("yolo_input_shape", ""),
                "yolo_input_dtype": latest.get("yolo_input_dtype", ""),
                "yolo_input_device": latest.get("yolo_input_device", ""),
                "yolo_input_stride": latest.get("yolo_input_stride", ""),
                "yolo_input_channels": round(latest_float("yolo_input_channels"), 3),
                "yolo_input_min": round(latest_float("yolo_input_min"), 6),
                "yolo_input_max": round(latest_float("yolo_input_max"), 6),
                "gpu_stage_profile_status": latest.get("gpu_stage_profile_status", ""),
                "gpu_stage_profile_output": latest.get("gpu_stage_profile_output", ""),
                "gpu_stage_profiled_frames": round(latest_float("gpu_stage_profiled_frames"), 3),
                "gpu_stage_warmup_skipped": round(latest_float("gpu_stage_warmup_skipped"), 3),
                "gpu_stage_event_count": round(latest_float("gpu_stage_event_count"), 3),
                "gpu_stage_total_avg_ms": round(latest_float("gpu_stage_total_avg_ms"), 6),
                "gpu_stage_largest_name": latest.get("gpu_stage_largest_name", ""),
                "gpu_stage_largest_avg_ms": round(latest_float("gpu_stage_largest_avg_ms"), 6),
                "gpu_stage_stabilize_avg_ms": round(latest_float("gpu_stage_stabilize_avg_ms"), 6),
                "gpu_stage_preprocess_avg_ms": round(latest_float("gpu_stage_preprocess_avg_ms"), 6),
                "gpu_stage_yolo_wrapper_avg_ms": round(latest_float("gpu_stage_yolo_wrapper_avg_ms"), 6),
                "gpu_stage_tracker_avg_ms": round(latest_float("gpu_stage_tracker_avg_ms"), 6),
                "gpu_stage_febam_avg_ms": round(latest_float("gpu_stage_febam_avg_ms"), 6),
                "gpu_stage_ocr_trigger_avg_ms": round(latest_float("gpu_stage_ocr_trigger_avg_ms"), 6),
                "preprocess_parent_avg_ms": round(latest_float("preprocess_parent_avg_ms"), 6),
                "preprocess_substage_sum_avg_ms": round(latest_float("preprocess_substage_sum_avg_ms"), 6),
                "preprocess_unattributed_avg_ms": round(latest_float("preprocess_unattributed_avg_ms"), 6),
                "preprocess_largest_substage": latest.get("preprocess_largest_substage", ""),
                "preprocess_largest_substage_avg_ms": round(latest_float("preprocess_largest_substage_avg_ms"), 6),
                "fastplate_async_enqueued": round(latest_float("fastplate_async_enqueued"), 3),
                "fastplate_async_processed": round(latest_float("fastplate_async_processed"), 3),
                "fastplate_batch_call_count": round(latest_float("fastplate_batch_call_count"), 3),
                "fastplate_avg_batch_size": round(latest_float("fastplate_avg_batch_size"), 3),
                "fastplate_batch_size": round(latest_float("fastplate_batch_size"), 3),
                "fastplate_min_batch_size": round(latest_float("fastplate_min_batch_size"), 3),
                "fastplate_large_batch_mode": round(latest_float("fastplate_large_batch_mode"), 3),
                "fastplate_target_batch_size": round(latest_float("fastplate_target_batch_size"), 3),
                "fastplate_flush_timeout_ms": round(latest_float("fastplate_flush_timeout_ms"), 3),
                "fastplate_max_flush_timeout_ms": round(latest_float("fastplate_max_flush_timeout_ms"), 3),
                "fastplate_batch_efficiency_ratio": round(latest_float("fastplate_batch_efficiency_ratio"), 3),
                "fastplate_total_ocr_ms": round(latest_float("fastplate_total_ocr_ms"), 3),
                "fastplate_avg_ocr_ms_per_crop": round(latest_float("fastplate_avg_ocr_ms_per_crop"), 3),
                "fastplate_queue_dropped": round(latest_float("fastplate_queue_dropped"), 3),
                "fastplate_queue_drop_weak": round(latest_float("fastplate_queue_drop_weak"), 3),
                "fastplate_queue_drop_duplicate": round(latest_float("fastplate_queue_drop_duplicate"), 3),
                "fastplate_queue_drop_rate_limited": round(latest_float("fastplate_queue_drop_rate_limited"), 3),
                "fastplate_queue_drop_backpressure": round(latest_float("fastplate_queue_drop_backpressure"), 3),
                "fastplate_result_delay_avg_frames": round(latest_float("fastplate_result_delay_avg_frames"), 3),
                "fastplate_result_delay_avg_ms": round(latest_float("fastplate_result_delay_avg_ms"), 3),
                "fastplate_queue_max_seen": round(latest_float("fastplate_queue_max_seen"), 3),
                "fastplate_queue_depth": round(latest_float("fastplate_queue_depth"), 3),
                "fastplate_oldest_age_ms": round(latest_float("fastplate_oldest_age_ms"), 3),
                "fastplate_gpu_crop_batch_enabled": round(latest_float("fastplate_gpu_crop_batch_enabled"), 3),
                "fastplate_custom_onnx_enabled": round(latest_float("fastplate_custom_onnx_enabled"), 3),
                "fastplate_custom_onnx_path": latest.get("fastplate_custom_onnx_path", ""),
                "fastplate_custom_provider": latest.get("fastplate_custom_provider", ""),
                "fastplate_custom_cuda_used": round(latest_float("fastplate_custom_cuda_used"), 3),
                "fastplate_custom_input_name": latest.get("fastplate_custom_input_name", ""),
                "fastplate_custom_output_names": latest.get("fastplate_custom_output_names", ""),
                "fastplate_custom_input_shape": latest.get("fastplate_custom_input_shape", ""),
                "fastplate_custom_output_shapes": latest.get("fastplate_custom_output_shapes", ""),
                "fastplate_custom_input_layout": latest.get("fastplate_custom_input_layout", ""),
                "fastplate_custom_alphabet_len": round(latest_float("fastplate_custom_alphabet_len"), 3),
                "fastplate_custom_batch_call_count": round(latest_float("fastplate_custom_batch_call_count"), 3),
                "fastplate_custom_item_count": round(latest_float("fastplate_custom_item_count"), 3),
                "fastplate_custom_avg_batch_size": round(latest_float("fastplate_custom_avg_batch_size"), 3),
                "fastplate_custom_total_ms": round(latest_float("fastplate_custom_total_ms"), 3),
                "fastplate_custom_avg_ms_per_crop": round(latest_float("fastplate_custom_avg_ms_per_crop"), 3),
                "fastplate_custom_preprocess_ms": round(latest_float("fastplate_custom_preprocess_ms"), 3),
                "fastplate_custom_infer_ms": round(latest_float("fastplate_custom_infer_ms"), 3),
                "fastplate_custom_decode_ms": round(latest_float("fastplate_custom_decode_ms"), 3),
                "fastplate_custom_error_count": round(latest_float("fastplate_custom_error_count"), 3),
                "dual_branch_enabled": round(latest_float("dual_branch_enabled"), 3),
                "dual_branch_session_count": round(latest_float("dual_branch_session_count"), 3),
                "dual_branch_global_infer_ms": round(latest_float("dual_branch_global_infer_ms"), 3),
                "dual_branch_korean_infer_ms": round(latest_float("dual_branch_korean_infer_ms"), 3),
                "dual_branch_iobinding_used": round(latest_float("dual_branch_iobinding_used"), 3),
                "dual_branch_h7_selected": round(latest_float("dual_branch_h7_selected"), 3),
                "dual_branch_h8_selected": round(latest_float("dual_branch_h8_selected"), 3),
                "dual_global_digits_valid_count": round(latest_float("dual_global_digits_valid_count"), 3),
                "dual_global_digits_rejected_count": round(latest_float("dual_global_digits_rejected_count"), 3),
                "dual_branch_alignment_rejected": round(latest_float("dual_branch_alignment_rejected"), 3),
                "dual_branch_digit_alignment_candidate_rejected": round(latest_float("dual_branch_digit_alignment_candidate_rejected"), 3),
                "dual_branch_hangul_candidate_rejected": round(latest_float("dual_branch_hangul_candidate_rejected"), 3),
                "dual_branch_low_digit_mass_rejected": round(latest_float("dual_branch_low_digit_mass_rejected"), 3),
                "dual_branch_forced_digit_count": round(latest_float("dual_branch_forced_digit_count"), 3),
                "dual_branch_avg_digit_mass": round(latest_float("dual_branch_avg_digit_mass"), 6),
                "dual_branch_avg_digit_margin": round(latest_float("dual_branch_avg_digit_margin"), 6),
                "dual_branch_length_ambiguity_hold_count": round(latest_float("dual_branch_length_ambiguity_hold_count"), 3),
                "dual_branch_cpu_fallback_count": round(latest_float("dual_branch_cpu_fallback_count"), 3),
                "dual_branch_warm_item_count": round(latest_float("dual_branch_warm_item_count"), 3),
                "dual_branch_warm_avg_ms_per_plate": round(latest_float("dual_branch_warm_avg_ms_per_plate"), 3),
                "dual_branch_warm_call_p95_ms": round(latest_float("dual_branch_warm_call_p95_ms"), 3),
                "dual_branch_warm_p95_ms_per_plate": round(latest_float("dual_branch_warm_p95_ms_per_plate"), 3),
                "dual_branch_length_margin_thr_configured": round(latest_float("dual_branch_length_margin_thr_configured"), 3),
                "dual_branch_length_margin_thr_effective": round(latest_float("dual_branch_length_margin_thr_effective"), 3),
                "dual_branch_length_hold_enabled": round(latest_float("dual_branch_length_hold_enabled"), 3),
                "fastplate_tensor_runner_enabled": round(latest_float("fastplate_tensor_runner_enabled"), 3),
                "fastplate_tensor_runner_mode_id": round(latest_float("fastplate_tensor_runner_mode_id"), 3),
                "fastplate_tensor_runner_active_used": round(latest_float("fastplate_tensor_runner_active_used"), 3),
                "fastplate_tensor_runner_shadow_count": round(latest_float("fastplate_tensor_runner_shadow_count"), 3),
                "fastplate_gpu_crop_count": round(latest_float("fastplate_gpu_crop_count"), 3),
                "fastplate_cpu_crop_count": round(latest_float("fastplate_cpu_crop_count"), 3),
                "fastplate_gpu_crop_ms": round(latest_float("fastplate_gpu_crop_ms"), 3),
                "fastplate_cpu_crop_ms": round(latest_float("fastplate_cpu_crop_ms"), 3),
                "fastplate_crop_cpu_copy_count": round(latest_float("fastplate_crop_cpu_copy_count"), 3),
                "fastplate_crop_gpu_tensor_count": round(latest_float("fastplate_crop_gpu_tensor_count"), 3),
                "fastplate_tensor_batch_call_count": round(latest_float("fastplate_tensor_batch_call_count"), 3),
                "fastplate_tensor_avg_batch_size": round(latest_float("fastplate_tensor_avg_batch_size"), 3),
                "fastplate_tensor_total_ms": round(latest_float("fastplate_tensor_total_ms"), 3),
                "fastplate_tensor_avg_ms_per_crop": round(latest_float("fastplate_tensor_avg_ms_per_crop"), 3),
                "fastplate_tensor_preprocess_ms": round(latest_float("fastplate_tensor_preprocess_ms"), 3),
                "fastplate_tensor_infer_ms": round(latest_float("fastplate_tensor_infer_ms"), 3),
                "fastplate_tensor_decode_ms": round(latest_float("fastplate_tensor_decode_ms"), 3),
                "fastplate_tensor_iobinding_used": round(latest_float("fastplate_tensor_iobinding_used"), 3),
                "fastplate_tensor_cpu_fallback_count": round(latest_float("fastplate_tensor_cpu_fallback_count"), 3),
                "fastplate_tensor_error_count": round(latest_float("fastplate_tensor_error_count"), 3),
                "fastplate_tensor_direct_backend_available": round(latest_float("fastplate_tensor_direct_backend_available"), 3),
                "fastplate_tensor_backend_kind": latest.get("fastplate_tensor_backend_kind", ""),
                "fastplate_tensor_gpu_only_claim_valid": round(latest_float("fastplate_tensor_gpu_only_claim_valid"), 3),
                "fastplate_tensor_active_requested": round(latest_float("fastplate_tensor_active_requested"), 3),
                "fastplate_tensor_active_success_count": round(latest_float("fastplate_tensor_active_success_count"), 3),
                "fastplate_tensor_active_blocked_count": round(latest_float("fastplate_tensor_active_blocked_count"), 3),
                "fastplate_tensor_active_fallback_used": round(latest_float("fastplate_tensor_active_fallback_used"), 3),
                "fastplate_tensor_active_fallback_reason": latest.get("fastplate_tensor_active_fallback_reason", ""),
                "final_vehicle_sparse_route_count": round(latest_float("final_vehicle_sparse_route_count"), 3),
                "final_vehicle_posterior_route_count": round(latest_float("final_vehicle_posterior_route_count"), 3),
                "final_vehicle_identity_hold_count": round(latest_float("final_vehicle_identity_hold_count"), 3),
                "final_vehicle_last_n_valid": round(latest_float("final_vehicle_last_n_valid"), 3),
                "final_vehicle_last_posterior_active": round(latest_float("final_vehicle_last_posterior_active"), 3),
                "fastplate_tensor_last_error_type": latest.get("fastplate_tensor_last_error_type", ""),
                "fastplate_tensor_last_error_message": latest.get("fastplate_tensor_last_error_message", ""),
                "fastplate_tensor_last_error_stage": latest.get("fastplate_tensor_last_error_stage", ""),
                "fastplate_tensor_error_sample_input_shape": latest.get("fastplate_tensor_error_sample_input_shape", ""),
                "fastplate_tensor_error_sample_input_dtype": latest.get("fastplate_tensor_error_sample_input_dtype", ""),
                "fastplate_tensor_error_sample_output_shapes": latest.get("fastplate_tensor_error_sample_output_shapes", ""),
                "fastplate_tensor_error_sample_batch_len": round(latest_float("fastplate_tensor_error_sample_batch_len"), 3),
                "fastplate_tensor_provider": latest.get("fastplate_tensor_provider", ""),
                "fastplate_tensor_input_name": latest.get("fastplate_tensor_input_name", ""),
                "fastplate_tensor_output_names": latest.get("fastplate_tensor_output_names", ""),
                "fastplate_tensor_preprocess_mode": latest.get("fastplate_tensor_preprocess_mode", ""),
                "fastplate_tensor_decoder_mode": latest.get("fastplate_tensor_decoder_mode", ""),
                "fastplate_tensor_last_input_shape": latest.get("fastplate_tensor_last_input_shape", ""),
                "fastplate_tensor_last_input_dtype": latest.get("fastplate_tensor_last_input_dtype", ""),
                "fastplate_tensor_prepare_valid_count": round(latest_float("fastplate_tensor_prepare_valid_count"), 3),
                "fastplate_tensor_prepare_failed_count": round(latest_float("fastplate_tensor_prepare_failed_count"), 3),
                "fastplate_tensor_prepare_failed_examples": latest.get("fastplate_tensor_prepare_failed_examples", ""),
                "fastplate_tensor_preprocess_resize_count": round(latest_float("fastplate_tensor_preprocess_resize_count"), 3),
                "fastplate_parity_sample_count": round(latest_float("fastplate_parity_sample_count"), 3),
                "fastplate_parity_success_count": round(latest_float("fastplate_parity_success_count"), 3),
                "fastplate_parity_error_count": round(latest_float("fastplate_parity_error_count"), 3),
                "fastplate_parity_skipped_count": round(latest_float("fastplate_parity_skipped_count"), 3),
                "fastplate_parity_text_match_count": round(latest_float("fastplate_parity_text_match_count"), 3),
                "fastplate_parity_text_match_rate": round(latest_float("fastplate_parity_text_match_rate"), 3),
                "fastplate_parity_normalized_match_rate": round(latest_float("fastplate_parity_normalized_match_rate"), 3),
                "fastplate_parity_korean_slot_match_rate": round(latest_float("fastplate_parity_korean_slot_match_rate"), 3),
                "fastplate_parity_conf_abs_diff_mean": round(latest_float("fastplate_parity_conf_abs_diff_mean"), 3),
                "fastplate_parity_effective_match_rate": round(latest_float("fastplate_parity_effective_match_rate"), 3),
                "middle_slot_enqueued": round(latest_float("middle_slot_enqueued"), 3),
                "middle_slot_processed": round(latest_float("middle_slot_processed"), 3),
                "middle_slot_batch_call_count": round(latest_float("middle_slot_batch_call_count"), 3),
                "middle_slot_avg_batch_size": round(latest_float("middle_slot_avg_batch_size"), 3),
                "middle_slot_total_infer_ms": round(latest_float("middle_slot_total_infer_ms"), 3),
                "middle_slot_avg_ms_per_crop": round(latest_float("middle_slot_avg_ms_per_crop"), 3),
                "middle_slot_queue_dropped": round(latest_float("middle_slot_queue_dropped"), 3),
                "middle_slot_enabled": round(latest_float("middle_slot_enabled"), 3),
                "middle_slot_cuda_used": round(latest_float("middle_slot_cuda_used"), 3),
                "middle_slot_encoder_mode": latest.get("middle_slot_encoder_mode", ""),
                "middle_slot_prototype_encoder_mode": latest.get("middle_slot_prototype_encoder_mode", ""),
                "middle_slot_prototype_path": latest.get("middle_slot_prototype_path", ""),
                "middle_slot_prototype_label_count": round(latest_float("middle_slot_prototype_label_count"), 3),
                "middle_slot_easyocr_single_crop": round(latest_float("middle_slot_easyocr_single_crop"), 3),
                "middle_slot_wide_core_x_pad_ratio": round(latest_float("middle_slot_wide_core_x_pad_ratio"), 3),
                "middle_slot_wide_core_y_pad_ratio": round(latest_float("middle_slot_wide_core_y_pad_ratio"), 3),
                "middle_slot_wide_core_crop_count": round(latest_float("middle_slot_wide_core_crop_count"), 3),
                "middle_slot_skip_no_prototype": round(latest_float("middle_slot_skip_no_prototype"), 3),
                "middle_slot_update_applied_count": round(latest_float("middle_slot_update_applied_count"), 3),
                "middle_slot_update_skipped_count": round(latest_float("middle_slot_update_skipped_count"), 3),
                "middle_slot_evidence_added": round(latest_float("middle_slot_evidence_added"), 3),
                "middle_slot_evidence_skipped": round(latest_float("middle_slot_evidence_skipped"), 3),
                "middle_slot_gt_anchor_confused_fair": round(latest_float("middle_slot_gt_anchor_confused_fair"), 3),
                "middle_slot_gt_plate_count": round(latest_float("middle_slot_gt_plate_count"), 3),
                "middle_slot_gt_anchor_boost": round(latest_float("middle_slot_gt_anchor_boost"), 3),
                "middle_slot_gt_confused_evidence_added": round(latest_float("middle_slot_gt_confused_evidence_added"), 3),
                "middle_slot_gt_confused_evidence_skipped": round(latest_float("middle_slot_gt_confused_evidence_skipped"), 3),
                "middle_slot_results_drained": round(latest_float("middle_slot_results_drained"), 3),
                "middle_slot_queue_depth": round(latest_float("middle_slot_queue_depth"), 3),
                "middle_slot_diagnostic_only": round(latest_float("middle_slot_diagnostic_only"), 3),
                "middle_slot_evidence_diagnostic_only_skipped": round(latest_float("middle_slot_evidence_diagnostic_only_skipped"), 3),
                "middle_slot_trigger_seen": round(latest_float("middle_slot_trigger_seen"), 3),
                "middle_slot_trigger_crop_present": round(latest_float("middle_slot_trigger_crop_present"), 3),
                "middle_slot_trigger_text_present": round(latest_float("middle_slot_trigger_text_present"), 3),
                "middle_slot_trigger_fallback_text_used": round(latest_float("middle_slot_trigger_fallback_text_used"), 3),
                "middle_slot_trigger_skipped_no_crop": round(latest_float("middle_slot_trigger_skipped_no_crop"), 3),
                "middle_slot_trigger_skipped_no_text": round(latest_float("middle_slot_trigger_skipped_no_text"), 3),
                "middle_slot_trigger_worker_none": round(latest_float("middle_slot_trigger_worker_none"), 3),
                "middle_slot_trigger_enqueue_success": round(latest_float("middle_slot_trigger_enqueue_success"), 3),
                "middle_slot_trigger_enqueue_failed": round(latest_float("middle_slot_trigger_enqueue_failed"), 3),
                "middle_slot_trigger_worker_init_error": latest.get("middle_slot_trigger_worker_init_error", ""),
                "middle_slot_v32_enabled": round(latest_float("middle_slot_v32_enabled"), 3),
                "middle_slot_v32_processed": round(latest_float("middle_slot_v32_processed"), 3),
                "middle_slot_v32_batch_call_count": round(latest_float("middle_slot_v32_batch_call_count"), 3),
                "middle_slot_v32_avg_batch_size": round(latest_float("middle_slot_v32_avg_batch_size"), 3),
                "middle_slot_v32_min_batch_size": round(latest_float("middle_slot_v32_min_batch_size"), 3),
                "middle_slot_v32_flush_every_frames": round(latest_float("middle_slot_v32_flush_every_frames"), 3),
                "middle_slot_v32_pending_final_flush_count": round(latest_float("middle_slot_v32_pending_final_flush_count"), 3),
                "middle_slot_v32_batch_efficiency_note": "runtime v32 should use batched inference; avg_batch_size=1 indicates inefficient synchronous per-crop execution",
                "middle_slot_v32_total_infer_ms": round(latest_float("middle_slot_v32_total_infer_ms"), 3),
                "middle_slot_v32_avg_ms_per_crop": round(latest_float("middle_slot_v32_avg_ms_per_crop"), 3),
                "middle_slot_v32_added_evidence": round(latest_float("middle_slot_v32_added_evidence"), 3),
                "middle_slot_v32_skipped_low_conf": round(latest_float("middle_slot_v32_skipped_low_conf"), 3),
                "middle_slot_v32_skipped_low_margin": round(latest_float("middle_slot_v32_skipped_low_margin"), 3),
                "middle_slot_v32_skipped_no_event_id": round(latest_float("middle_slot_v32_skipped_no_event_id"), 3),
                "middle_slot_v32_skipped_max_crops_per_event": round(latest_float("middle_slot_v32_skipped_max_crops_per_event"), 3),
                "middle_slot_v32_skipped_empty_crop": round(latest_float("middle_slot_v32_skipped_empty_crop"), 3),
                "middle_slot_v32_skipped_duplicate": round(latest_float("middle_slot_v32_skipped_duplicate"), 3),
                "middle_slot_v32_skipped_pending_not_flushed": round(latest_float("middle_slot_v32_skipped_pending_not_flushed"), 3),
                "middle_slot_v32_skipped_model_error": round(latest_float("middle_slot_v32_skipped_model_error"), 3),
                "middle_slot_v32_skipped_before_infer_unknown": round(latest_float("middle_slot_v32_skipped_before_infer_unknown"), 3),
                "middle_slot_v32_skipped_other": round(latest_float("middle_slot_v32_skipped_other"), 3),
                "profile_numeric_counts": numeric_counts,
                "profile_skipped_non_numeric": skipped_counts,
            }
        )
        + "\n"
    )


def _log_profile_rows(rows: list[dict[str, float]], logger: StageLogger, interval: int = 30) -> None:
    if not rows:
        return
    interval = max(1, interval)
    keys = [
        "stage0_decode_ms",
        "stage1_stabilize_ms",
        "stage2_preprocess_ms",
        "stage6_9_head_ms",
        "stage10_tracker_ms",
        "stage11_febam_ms",
        "stage12_ocr_trigger_ms",
        "stage13_writer_enqueue_ms",
        "total_frame_ms",
        "fps",
    ]
    for start in range(0, len(rows), interval):
        chunk = rows[start:start + interval]
        if not chunk:
            continue
        avg = {k: sum(r.get(k, 0.0) for r in chunk) / len(chunk) for k in keys}
        pipeline_ms = (
            avg["stage1_stabilize_ms"]
            + avg["stage2_preprocess_ms"]
            + avg["stage6_9_head_ms"]
            + avg["stage10_tracker_ms"]
            + avg["stage11_febam_ms"]
            + avg["stage12_ocr_trigger_ms"]
        )
        logger.log(
            "PROFILE "
            f"frame={start + len(chunk) - 1} "
            f"decode={avg['stage0_decode_ms']:.3f}ms "
            f"pipeline={pipeline_ms:.3f}ms "
            f"overlay={avg['stage13_writer_enqueue_ms']:.3f}ms "
            "writer=async "
            f"total={avg['total_frame_ms']:.3f}ms "
            f"fps={avg['fps']:.3f}"
        )


def _run_gpu_test_image(
    pipe: object,
    image_path: str,
    repeat: int,
    profile: bool,
    logger: StageLogger,
) -> list[dict[str, float]]:
    if repeat < 1:
        raise RuntimeError("gpu-test-image-repeat must be >= 1")
    import cv2
    import torch

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"gpu-test-image read failed: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_u8 = (
        torch.from_numpy(img_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device="cuda", dtype=torch.uint8)
    )
    if not frame_u8.is_cuda:
        raise RuntimeError("gpu-test-image produced CPU tensor")

    out = []
    for i in range(repeat):
        step_out = pipe.step(frame_u8, profile=profile, frame_idx=i)
        if isinstance(step_out, tuple):
            times = step_out[1] if len(step_out) >= 2 else {}
        else:
            times = {}
        total_ms = float(times.get("total_frame_ms", 0.0))
        if total_ms <= 0.0:
            total_ms = 1e-6
            times["total_frame_ms"] = total_ms
        times["stage0_decode_ms"] = 0.0
        times["fps"] = 1000.0 / max(total_ms, 1e-6)
        out.append(times)
    return out


def apply_event_roi_fusion_mode_preset(args: argparse.Namespace) -> None:
    mode = getattr(args, "event_roi_fusion_mode", "ours_all_roi_febam")

    if mode == "b5_yolo_high_quality_fusion":
        print("[B5_BASELINE] mode=b5_yolo_high_quality_fusion")
        print("[B5_BASELINE] YOLO high-quality crop fusion baseline")
        print("[B5_BASELINE] String-FEBAM disabled")
        print("[B5_BASELINE] Input = high-quality YOLO plate crops only")
        print("[B5_BASELINE] Output = fused image OCR rows")

        if hasattr(args, "event_roi_fusion"):
            args.event_roi_fusion = True
        if hasattr(args, "event_roi_fusion_ocr"):
            args.event_roi_fusion_ocr = True
        if hasattr(args, "event_roi_fusion_ocr_backend"):
            args.event_roi_fusion_ocr_backend = "fastplate"
        if hasattr(args, "event_roi_fusion_preset"):
            args.event_roi_fusion_preset = "trial013_fixed"
        if hasattr(args, "event_roi_fusion_save_debug"):
            args.event_roi_fusion_save_debug = True
        if hasattr(args, "event_roi_fusion_save_manifest"):
            args.event_roi_fusion_save_manifest = True
        if hasattr(args, "fusion_center_pad_ratio"):
            args.fusion_center_pad_ratio = 0.15

        for name in ["string_febam", "use_sigmoid_febam", "sigmoid_febam"]:
            if hasattr(args, name):
                setattr(args, name, False)

        if hasattr(args, "ocr_csv") and not getattr(args, "ocr_csv", None):
            args.ocr_csv = "../data/processed/debug/ocr_history_b5_yolo_high_quality.csv"

        default_debug_dir = "../data/processed/debug/event_fusion"
        if hasattr(args, "event_roi_fusion_debug_dir") and (
            not getattr(args, "event_roi_fusion_debug_dir", None)
            or getattr(args, "event_roi_fusion_debug_dir", None) == default_debug_dir
        ):
            args.event_roi_fusion_debug_dir = "../data/processed/debug/b5_yolo_high_quality_fusion"

    elif mode == "ours_all_roi_febam":
        print("[OURS] mode=ours_all_roi_febam")
        print("[OURS] existing all-ROI event fusion + String-FEBAM behavior is preserved")
        preset = str(getattr(args, "event_roi_fusion_preset", "none") or "none").lower().strip()
        if preset == "trial013_fixed_motion":
            print("[OURS] event_roi_fusion_preset=trial013_fixed_motion (trial013_fixed + Motion-FEBAM soft filter)")
        elif preset == "trial013_fixed_motion_color":
            print("[OURS] event_roi_fusion_preset=trial013_fixed_motion_color (trial013_fixed + Motion-FEBAM + color soft weight)")


def apply_fastplate_runtime_preset(args: argparse.Namespace) -> None:
    """Apply the low-latency single-camera policy after all fusion presets.

    The async worker already owns exactly one recognizer/session.  This preset
    intentionally changes queue timing only; it never creates another ORT
    session or enables a CPU OCR path.
    """
    preset = getattr(args, "fastplate_runtime_preset", "default")
    if preset == "offline_throughput_b4":
        # Dataset/replay throughput policy.  The frozen dual-branch runner has
        # physical B4 buffers, so accumulate sparse candidates across frames
        # instead of flushing B1 after the low-latency 150 ms deadline.
        # This changes scheduling only: crop, geometry, OCR and fusion meaning
        # remain identical, and final drain still processes every queued item.
        args.fastplate_batch_size = 4
        args.fastplate_min_batch_size = 4
        args.fastplate_target_batch_size = 4
        args.fastplate_flush_timeout_ms = 5000.0
        args.fastplate_max_flush_timeout_ms = 5000.0
        args.fastplate_large_batch_mode = True
        args.fastplate_gpu_crop_batch = True
        print(
            "[FASTPLATE_RUNTIME_PRESET] offline_throughput_b4 "
            "physical_batch=4 min_batch=4 target_batch=4 latency_cap_ms=5000 "
            "cross_frame_accumulation=1"
        )
        return
    if preset not in {"single_camera_microbatch", "frozen_lovo_b3"}:
        return

    args.fastplate_batch_size = 4
    args.fastplate_min_batch_size = 2
    args.fastplate_target_batch_size = 4
    args.fastplate_flush_timeout_ms = 60.0
    args.fastplate_max_flush_timeout_ms = 150.0
    args.fastplate_large_batch_mode = False
    args.fastplate_rate_limit_relaxed = False
    # Final WiSE custom ONNX receives CUDA crops directly.  This flag records
    # the production dataflow; it does not create a second worker/session.
    args.fastplate_gpu_crop_batch = True

    # Fusion OCR creates an additional delayed OCR path.  The runtime preset
    # keeps raw OCR -> String-FEBAM -> HOLD/COMMIT as the default path.
    args.event_roi_fusion = False
    args.event_roi_fusion_ocr = False
    args.event_roi_fusion_preset = "none"
    args.event_roi_fusion_save_debug = False
    args.event_roi_fusion_save_manifest = False
    if preset == "frozen_lovo_b3":
        # Frozen method-selection result.  These switches prevent accidentally
        # re-enabling rejected/shadow evidence paths in the primary runtime.
        # The grouping bridge is structural identity plumbing, not a second
        # OCR evidence source.  Keep fusion OCR OFF but retain trial020 so a
        # canonical event key can be formed before final-vehicle routing.
        args.event_roi_fusion = False
        args.event_roi_fusion_preset = "none"
        args.event_roi_fusion_ocr = False
        args.event_roi_fusion_save_debug = False
        args.event_roi_fusion_save_manifest = False
        # Core sparse-evidence branch: with at most three unique frames the
        # non-generative EPC_SR path may collect one weak OCR observation.
        # It is identity-locked, redegradation-guarded and never has direct
        # final/COMMIT authority. Four-or-more frames stay on position
        # posterior; restoration remains auxiliary evidence only.
        args.non_generative_restoration_shadow = True
        args.restoration_shadow_max_frames = 3
        args.restoration_shadow_min_frames = 2
        args.restoration_shadow_schedule_frames = 2
        args.restoration_shadow_output = "outputs/lovo_final/sparse_epc_sr_evidence.jsonl"
        args.string_febam = True
        args.string_febam_group_key_mode = "event_fusion_group"
        # The held-out paired evaluation found 57 wrong COMMITs among 161
        # B3 COMMITs.  Test-set evidence must not be used to tune a new
        # threshold, so the frozen B3 replay is fail-closed until an
        # independently calibrated rule passes the wrong-COMMIT gate.
        args.string_febam_commit_thr = 1.01
        args.string_febam_skip_after_commit = False
        # Korean plates require the official global digit sequence plus the
        # frozen Korean-slot head. This is one primary recognizer decode path,
        # not the rejected independent secondary-OCR method.
        args.dual_branch_ocr = True
        # Use the exact Korean training-loader contract. The expanded bbox is
        # cropped once; only the model-required 64x256 resize is performed.
        args.dual_branch_korean_mode = "full_sequence_primary"
        args.middle_slot_upl = False
        args.middle_slot_v32 = False
        args.bio_adaptive_ocr = False
        args.ocr_disable_heavy_variants = True
        args.fastplate_preserve_crop_geometry = True
        args.gpu_final_vehicle_grouping = True
        args.fastplate_tensor_runner = True
        args.fastplate_tensor_runner_mode = "active"
        args.fastplate_direct_ort = True
        args.no_gray_stretched_ocr = True
        args.fastplate_rate_limit_relaxed = False
        print(
            "[FROZEN_LOVO_B3] FastPlateOCR -> position evidence -> "
            "StringFEBAM SHADOW/HOLD (paired wrong-COMMIT gate failed); "
            "rejected auxiliary paths=OFF"
        )
    print(
        "[FASTPLATE_RUNTIME_PRESET] single_camera_microbatch "
        "gpu_workers=1 batch=4 min_batch=2 flush_ms=60 max_flush_ms=150 "
        "fusion_ocr=disabled"
    )


def apply_dual_branch_fixed_batch_contract(args: argparse.Namespace) -> None:
    """Keep the fixed physical B4 dual-branch ONNX models within contract.

    This is independent of the single-camera latency preset: canonical event
    fusion remains enabled when requested, while only OCR queue batch limits
    are clamped.
    """
    if not bool(getattr(args, "dual_branch_ocr", False)):
        return
    args.fastplate_batch_size = min(
        4, max(1, int(getattr(args, "fastplate_batch_size", 4)))
    )
    args.fastplate_min_batch_size = min(
        args.fastplate_batch_size,
        max(1, int(getattr(args, "fastplate_min_batch_size", 2))),
    )
    args.fastplate_target_batch_size = min(
        args.fastplate_batch_size,
        max(1, int(getattr(args, "fastplate_target_batch_size", 4))),
    )
    print(
        "[DUAL_BRANCH_BATCH_CONTRACT] physical_batch=4 "
        f"batch={args.fastplate_batch_size} "
        f"min_batch={args.fastplate_min_batch_size} "
        f"target_batch={args.fastplate_target_batch_size}"
    )


def run_gpu_only(args: argparse.Namespace):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 없어서 gpu_only 실행 불가")
    if args.preprocess_routing_mode != "baseline_full_frame":
        raise RuntimeError(
            "yolo_primary_roi is unsafe: YOLO consumes fused_preprocess LAST_CAN ([B,1,H,W]), "
            "not raw/stabilized RGB. Use --preprocess-routing-mode baseline_full_frame."
        )

    logger = StageLogger(console=args.gpu_debug_stage_log, log_txt=args.gpu_debug_log_txt)
    logger.log("using GPU pipeline")
    input_fps = 25.0 if (args.gpu_test_image or args.gpu_dummy_frame) else _read_input_fps(args.input, logger)
    debug_dump_seconds = set()
    if args.debug_dump_seconds:
        debug_dump_seconds = {
            int(float(x.strip())) for x in args.debug_dump_seconds.split(",") if x.strip()
        }

    requested_ppocrv5 = str(getattr(args, "ocr_backend", "")) == "ppocrv5"
    if requested_ppocrv5:
        args.ocr_backend = "fastplate"
        args.fastplate_async = False
    pipeline_config = PipelineFactory.config_from_args(
        args,
        fps=input_fps,
        debug_dump_seconds=debug_dump_seconds,
    )
    for key in (
        "fastplate_custom_onnx",
        "fastplate_custom_plate_config",
        "fastplate_custom_input_width",
        "fastplate_custom_input_height",
    ):
        if hasattr(args, key):
            try:
                setattr(pipeline_config, key, getattr(args, key))
            except Exception:
                pass
    requested_c3_shadow = str(getattr(args, "c3_mode", "off")) != "off"
    requested_live_dual = bool(getattr(args, "c3_live_dual_ocr", False))
    if requested_ppocrv5 or requested_c3_shadow or requested_live_dual:
        from gpu_pipeline2 import GPUPipeline as FinalGPUPipeline
        pipe = PipelineFactory.build_gpu_pipeline(pipeline_config, pipeline_cls=FinalGPUPipeline)
        if requested_ppocrv5:
            pipe.configure_ppocrv5_live(model_dir=str(getattr(args, "ppocrv5_model_dir", "") or ""), device="gpu:0")
    else:
        pipe = PipelineFactory.build_gpu_pipeline(pipeline_config)
    if requested_c3_shadow:
        pipe.configure_c3_shadow(
            mode=str(args.c3_mode),
            output_path=str(args.c3_shadow_output),
            replay_predictions=str(args.c3_replay_predictions or ""),
        )
    if requested_live_dual:
        pipe.configure_live_dual_ocr_shadow(
            g6=str(args.c3_live_g6), g6_config=str(args.c3_live_g6_config),
            ppv5=str(args.c3_live_ppv5), output=str(args.c3_live_output), legal=str(args.c3_live_legal),
        )
    if hasattr(pipe, "configure_gpu_stage_profile_from_args"):
        pipe.configure_gpu_stage_profile_from_args(args)
    if hasattr(pipe, "configure_yolo_input_trace_from_args"):
        pipe.configure_yolo_input_trace_from_args(args)
    if getattr(args, "fastplate_direct_ort", False) and not getattr(args, "fastplate_custom_onnx", None):
        args.fastplate_tensor_runner = True
        if getattr(args, "fastplate_tensor_runner_mode", "disabled") == "disabled":
            args.fastplate_tensor_runner_mode = "active"
    if hasattr(pipe, "configure_fastplate_direct_ort_from_args"):
        pipe.configure_fastplate_direct_ort_from_args(args)
    if hasattr(pipe, "configure_bio_adaptive_ocr_from_args"):
        pipe.configure_bio_adaptive_ocr_from_args(args)
    if hasattr(pipe, "configure_middle_slot_v32_from_args"):
        pipe.configure_middle_slot_v32_from_args(args)
    if hasattr(pipe, "configure_webapp_preview_from_args"):
        pipe.configure_webapp_preview_from_args(args)
    else:
        setattr(pipe, "webapp_preview_from_debug_frame", bool(getattr(args, "webapp_enable", False) and getattr(args, "webapp_preview_from_debug_frame", False)))
        setattr(pipe, "webapp_preview_path", getattr(args, "webapp_preview_jpg", None))
        setattr(pipe, "webapp_preview_format", getattr(args, "webapp_preview_format", "jpg"))
        setattr(pipe, "webapp_preview_max_width", getattr(args, "webapp_preview_max_width", 960))
    if hasattr(pipe, "configure_event_roi_fusion_mode_from_args"):
        pipe.configure_event_roi_fusion_mode_from_args(args)
    if hasattr(pipe, "configure_group_review_from_args"):
        pipe.configure_group_review_from_args(args)
    if hasattr(pipe, "configure_labeling_febam_sink_from_args"):
        pipe.configure_labeling_febam_sink_from_args(args)
    if hasattr(pipe, "configure_middle_slot_upl_from_args"):
        pipe.configure_middle_slot_upl_from_args(args)
    if hasattr(pipe, "configure_non_generative_restoration_shadow_from_args"):
        pipe.configure_non_generative_restoration_shadow_from_args(args)

    rows: list[dict[str, float]] = []
    viewer_sink = GPURuntimeViewerSink(
        mode=getattr(args, "viewer_runtime_mode", "off"),
        snapshot_path=getattr(args, "viewer_runtime_snapshot", None),
        jsonl_path=getattr(args, "viewer_runtime_jsonl", None),
        preview_source=getattr(args, "webapp_preview_jpg", None),
        asset_dir=getattr(args, "viewer_runtime_assets", None),
    )
    last_webapp_commit_key = None
    webapp_recent_candidates = deque(maxlen=max(0, int(getattr(args, "webapp_recent_candidates", 5))))
    if getattr(args, "webapp_enable", False) and getattr(args, "webapp_result_json", None):
        write_webapp_result_json(
            args.webapp_result_json,
            frame_idx=None,
            track_id=None,
            state="WAITING",
            plate="-",
            candidate="-",
            fps=None,
            source="startup",
            recent_candidates=[],
        )
    total_candidates = 0
    total_tracks = 0
    total_confirmed = 0
    ocr_trigger_count = 0
    wall_start_t = time.perf_counter()

    def _debug_video_duration_sec() -> float:
        if args.debug_start_sec is not None and args.debug_end_sec is not None:
            return max(0.0, float(args.debug_end_sec) - float(args.debug_start_sec))
        return float(len(rows)) / max(float(input_fps), 1e-6) if rows else 0.0

    def _update_wall_clock_profile(final: bool = False) -> None:
        if not rows:
            return
        wall_elapsed_sec = time.perf_counter() - wall_start_t
        frames_processed = len(rows)
        wall_clock_effective_fps = float(frames_processed) / max(1e-6, wall_elapsed_sec)
        debug_video_duration_sec = _debug_video_duration_sec()
        rows[-1].update({
            "wall_clock_elapsed_sec": wall_elapsed_sec,
            "wall_clock_effective_fps": wall_clock_effective_fps,
            "wall_clock_ms_per_frame": 1000.0 / max(1e-6, wall_clock_effective_fps),
            "debug_video_duration_sec": debug_video_duration_sec,
            "processing_speed_x": debug_video_duration_sec / max(1e-6, wall_elapsed_sec),
            "wall_clock_finalized": 1.0 if final else 0.0,
        })

    def _record_step_out(step_out, frame_u8, frame_idx: int, step_wall_ms: float):
        nonlocal total_candidates, total_tracks, total_confirmed, ocr_trigger_count, last_webapp_commit_key
        if isinstance(step_out, tuple):
            times = step_out[1] if len(step_out) >= 2 else {}
        else:
            times = {}
        total_ms = float(times.get("total_frame_ms", step_wall_ms))
        if total_ms <= 0.0:
            total_ms = step_wall_ms
        times["total_frame_ms"] = total_ms
        times["stage0_decode_ms"] = step_wall_ms - total_ms
        times["fps"] = 1000.0 / max(total_ms, 1e-6)
        times["preprocess_routing_mode"] = args.preprocess_routing_mode
        trace = getattr(pipe, "_yolo_input_trace", None)
        if trace:
            times.update(trace)
        rows.append(times)
        total_candidates += int(times.get("candidates_count", 0))
        total_tracks += int(times.get("tracks_count", 0))
        total_confirmed += int(times.get("confirmed_count", 0))
        ocr_trigger_count += int(times.get("ocr_triggered", 0))
        last_webapp_commit_key = maybe_write_webapp_result(args, pipe, times, frame_idx, last_webapp_commit_key, webapp_recent_candidates)
        if getattr(args, "webapp_preview_overlay_after_step", False):
            maybe_write_webapp_preview(args, pipe, times, frame_idx, frame_u8, step_out)
        if viewer_sink.enabled:
            viewer_payload = build_webapp_result_payload(pipe, times, frame_idx)
            viewer_sink.publish(
                payload=viewer_payload,
                pipe=pipe,
                times=times,
                frame_idx=frame_idx,
                preview_image=frame_u8,
            )

    try:
        if args.gpu_test_image:
            rows = _run_gpu_test_image(
                pipe,
                args.gpu_test_image,
                args.gpu_test_image_repeat,
                args.gpu_profile,
                logger,
            )
            for t in rows:
                total_candidates += int(t.get("candidates_count", 0))
                total_tracks += int(t.get("tracks_count", 0))
                total_confirmed += int(t.get("confirmed_count", 0))
                ocr_trigger_count += int(t.get("ocr_triggered", 0))
            if args.gpu_profile and args.gpu_debug_log_txt:
                _log_profile_rows(rows, logger)
            if args.gpu_profile:
                _update_wall_clock_profile(final=False)
                _print_profile_summary(rows)
            logger.log(
                f"summary total_candidates={total_candidates} total_tracks={total_tracks} total_confirmed={total_confirmed} ocr_trigger_count={ocr_trigger_count}"
            )
            return

        if args.gpu_dummy_frame:
            n = args.gpu_profile_frames if args.gpu_profile else 10
            for i in range(n):
                frame_u8 = torch.randint(0, 256, (1, 3, 720, 1280), device="cuda", dtype=torch.uint8)
                maybe_write_webapp_preview_early(args, i, frame_u8)
                step_out = pipe.step(frame_u8, profile=args.gpu_profile, frame_idx=i)
                if isinstance(step_out, tuple):
                    times = step_out[1] if len(step_out) >= 2 else {}
                else:
                    times = {}
                total_ms = float(times.get("total_frame_ms", 0.0))
                if total_ms <= 0.0:
                    total_ms = 1e-6
                    times["total_frame_ms"] = total_ms
                times["stage0_decode_ms"] = 0.0
                times["fps"] = 1000.0 / max(total_ms, 1e-6)
                rows.append(times)
                total_candidates += int(times.get("candidates_count", 0))
                total_tracks += int(times.get("tracks_count", 0))
                total_confirmed += int(times.get("confirmed_count", 0))
                ocr_trigger_count += int(times.get("ocr_triggered", 0))
                last_webapp_commit_key = maybe_write_webapp_result(args, pipe, times, i, last_webapp_commit_key, webapp_recent_candidates)
                if getattr(args, "webapp_preview_overlay_after_step", False):
                    maybe_write_webapp_preview(args, pipe, times, i, frame_u8, step_out)
            if args.gpu_profile and args.gpu_debug_log_txt:
                _log_profile_rows(rows, logger)
            if args.gpu_profile:
                _update_wall_clock_profile(final=False)
                _print_profile_summary(rows)
            logger.log(
                f"summary total_candidates={total_candidates} total_tracks={total_tracks} total_confirmed={total_confirmed} ocr_trigger_count={ocr_trigger_count}"
            )
            return

        fps = input_fps
        pipe.fps = fps
        start_frame = int(args.debug_start_sec * fps) if args.debug_start_sec is not None else 0
        end_frame = int(args.debug_end_sec * fps) if args.debug_end_sec is not None else None
        limit = args.gpu_profile_frames if args.gpu_profile else None
        if (
            args.gpu_profile
            and args.gpu_profile_frames is not None
            and int(args.gpu_profile_frames) > 0
            and start_frame > 0
            and int(args.gpu_profile_frames) <= start_frame
        ):
            recommended = int(end_frame) + 1 if end_frame is not None else int(start_frame) + 1
            msg = (
                f"[PROFILE_WINDOW_WARNING] gpu_profile_frames={int(args.gpu_profile_frames)} "
                f"stops before debug_start_frame={start_frame} for debug_start_sec={args.debug_start_sec}. "
                f"fps={fps:.6g}. Increase --gpu-profile-frames to at least {recommended} "
                f"or remove --gpu-profile-frames."
            )
            print(msg, flush=True)
            logger.log(msg)

        from gpu_pipeline import Stage0GPUDecoder

        decoder = Stage0GPUDecoder(
            args.input,
            debug_stage_log=args.gpu_debug_stage_log or bool(args.gpu_debug_log_txt),
            profile_frame_limit=args.gpu_profile_frames if args.gpu_profile else None,
            logger=logger.log,
            fps=input_fps,
            debug_dump_seconds=debug_dump_seconds,
            start_frame=start_frame,
        )

        yolo_batch_mode = str(getattr(args, "yolo_batch_mode", "disabled") or "disabled")
        yolo_batch_size = int(max(1, getattr(args, "yolo_batch_size", 1) or 1))
        use_yolo_microbatch = (
            yolo_batch_mode == "microbatch"
            and yolo_batch_size > 1
            and hasattr(pipe, "step_yolo_batch")
        )
        pending_frames = []
        pending_indices = []

        def _flush_yolo_microbatch() -> None:
            nonlocal pending_frames, pending_indices
            if not pending_frames:
                return
            t0 = time.perf_counter()
            step_outs = pipe.step_yolo_batch(
                pending_frames,
                pending_indices,
                profile=args.gpu_profile,
            )
            t1 = time.perf_counter()
            per_frame_wall_ms = ((t1 - t0) * 1000.0) / float(max(1, len(step_outs)))
            for frame_u8_b, frame_idx_b, step_out_b in zip(pending_frames, pending_indices, step_outs):
                _record_step_out(step_out_b, frame_u8_b, frame_idx_b, per_frame_wall_ms)
            pending_frames = []
            pending_indices = []

        # Stage0 may have sought directly to the debug window.  Its
        # ``start_frame`` remains zero when the installed NVDEC decoder cannot
        # seek, so the existing sequential behavior is retained in that case.
        for i, frame_u8 in enumerate(decoder, start=int(getattr(decoder, "start_frame", 0))):
            if i < start_frame:
                continue
            if end_frame is not None and i > end_frame:
                break
            if limit is not None and len(rows) >= limit:
                break
            if not frame_u8.is_cuda:
                raise RuntimeError("Decoder produced CPU tensor. GPU-only violation")

            maybe_write_webapp_preview_early(args, i, frame_u8)

            if use_yolo_microbatch:
                pending_frames.append(frame_u8)
                pending_indices.append(i)
                if len(pending_frames) >= yolo_batch_size or (limit is not None and len(rows) + len(pending_frames) >= limit):
                    _flush_yolo_microbatch()
            else:
                t0 = time.perf_counter()
                step_out = pipe.step(frame_u8, profile=args.gpu_profile, frame_idx=i)
                t1 = time.perf_counter()
                _record_step_out(step_out, frame_u8, i, (t1 - t0) * 1000.0)

        if use_yolo_microbatch:
            _flush_yolo_microbatch()

        if args.gpu_profile and args.gpu_debug_log_txt:
            _log_profile_rows(rows, logger)
        if args.gpu_profile and not getattr(args, "gpu_stage_profile", False):
            _update_wall_clock_profile(final=False)
            _print_profile_summary(rows)
        logger.log(
            f"summary total_candidates={total_candidates} total_tracks={total_tracks} total_confirmed={total_confirmed} ocr_trigger_count={ocr_trigger_count}"
        )
    finally:
        close_error = None
        try:
            pipe.close()
        except Exception as exc:
            close_error = exc
            logger.log(f"pipe_close_failed error={exc}")
        if getattr(args, "gpu_stage_profile", False) and hasattr(pipe, "finalize_gpu_stage_profile"):
            try:
                gpu_stage_summary = pipe.finalize_gpu_stage_profile()
                print("[GPU_STAGE_PROFILE]", gpu_stage_summary, flush=True)
                if rows and gpu_stage_summary:
                    stages = gpu_stage_summary.get("stages", {})
                    merged = {
                        "gpu_stage_profile_status": gpu_stage_summary.get("status", ""),
                        "gpu_stage_profile_output": gpu_stage_summary.get("profile_output_path", ""),
                        "gpu_stage_profiled_frames": gpu_stage_summary.get("profiled_frames", 0),
                        "gpu_stage_warmup_skipped": gpu_stage_summary.get("warmup_skipped", 0),
                        "gpu_stage_event_count": gpu_stage_summary.get("event_count", 0),
                        "gpu_stage_stream_count": gpu_stage_summary.get("stream_count", 0),
                        "gpu_stage_largest_name": gpu_stage_summary.get("largest_gpu_stage", ""),
                        "gpu_stage_largest_avg_ms": gpu_stage_summary.get("largest_gpu_stage_avg_ms", 0.0),
                        "preprocess_parent_avg_ms": gpu_stage_summary.get("preprocess_parent_avg_ms", 0.0),
                        "preprocess_substage_sum_avg_ms": gpu_stage_summary.get("preprocess_substage_sum_avg_ms", 0.0),
                        "preprocess_unattributed_avg_ms": gpu_stage_summary.get("preprocess_unattributed_avg_ms", 0.0),
                        "preprocess_largest_substage": gpu_stage_summary.get("preprocess_largest_substage", ""),
                        "preprocess_largest_substage_avg_ms": gpu_stage_summary.get("preprocess_largest_substage_avg_ms", 0.0),
                    }
                    merged["gpu_stage_total_avg_ms"] = gpu_stage_summary.get(
                        "gpu_stage_total_avg_ms",
                        sum(float(v.get("avg_ms", 0.0)) for name, v in stages.items() if not name.startswith("preprocess_")),
                    )
                    for stage_name, values in stages.items():
                        for field in ("calls", "avg_ms", "p50_ms", "p95_ms", "max_ms"):
                            merged[f"gpu_stage_{stage_name}_{field}"] = values.get(field, 0.0)
                    rows[-1].update(merged)
            except Exception as exc:
                logger.log(f"gpu_stage_profile_finalize_failed error={exc}")
        if args.gpu_profile and rows:
            try:
                rows[-1].update(pipe._fastplate_async_stats())
                rows[-1].update(pipe._middle_slot_stats())
                if hasattr(pipe, "_yolo_batch_stats"):
                    rows[-1].update(pipe._yolo_batch_stats(enabled=1.0 if getattr(args, "yolo_batch_mode", "disabled") == "microbatch" else 0.0))
                _update_wall_clock_profile(final=True)
                _print_profile_summary(rows)
                # Keep the final merged row beside the profiler JSON so the
                # validation result is inspectable without scraping stdout.
                profile_output = str(getattr(args, "gpu_stage_profile_output", "") or "")
                if profile_output:
                    runtime_summary_path = Path(profile_output).with_name("runtime_summary.json")
                    runtime_summary_path.write_text(
                        json.dumps(rows[-1], ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    print(f"[RUNTIME_SUMMARY] {runtime_summary_path}", flush=True)
            except Exception as exc:
                logger.log(f"final_profile_summary_failed error={exc}")
        if args.ocr_csv:
            try:
                pipe.save_ocr_csv(args.ocr_csv)
                logger.log(f"ocr_csv_saved path={args.ocr_csv}")
            except Exception as exc:
                logger.log(f"ocr_csv_save_failed path={args.ocr_csv} error={exc}")
        logger.close()
        if close_error is not None:
            raise close_error


def main():
    args = parse_args()
    args = apply_middle_slot_preset_to_args(args)
    if getattr(args, "middle_slot_backend", "") == "hog_lbp_gpu":
        args.middle_slot_upl = True
    if getattr(args, "export_group_review", False):
        args.event_roi_fusion = True
        if getattr(args, "event_roi_fusion_preset", "none") == "none":
            args.event_roi_fusion_preset = "trial013_fixed_motion"
        args.event_roi_fusion_save_manifest = True
    apply_event_roi_fusion_mode_preset(args)
    apply_fastplate_runtime_preset(args)
    apply_dual_branch_fixed_batch_contract(args)
    if args.backend != "gpu_only":
        raise RuntimeError("이 버전은 gpu_only 전용입니다. --backend gpu_only 사용")
    run_gpu_only(args)


if __name__ == "__main__":
    main()

"""Opt-in Real822 research integration layered over :mod:`gpu_pipeline`.

The legacy pipeline is deliberately imported, rather than edited.  With the new
features unused ``GPUPipeline`` has the exact legacy constructor and execution
path.  This module only adds deterministic H7/H8 image-space neutralisation and
the frozen Hangul authority adapters required by the Real822 replay.

No GT, OCR string position, or predicted string length is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import gpu_pipeline as _legacy
from gpu_pipeline import *  # noqa: F401,F403 - preserve the legacy public API


HANGUL_LIGHTWEIGHT_AUTHORITY = "PPV5_CONSTRAINED_HCD_493_OF_820"
HANGUL_DECISION_MAX_AUTHORITY = "ALLFRAME_ONEHOT_OOF_498_OF_820"


@dataclass(frozen=True, slots=True)
class FinalGPUIntegrationConfig:
    """Configuration for the additive Real822 integration.

    ``enabled=False`` is the parity/default-off state.  The lightweight 493
    authority is the only production default.  The 498 authority is offline
    OOF shadow evidence and can never silently become a deployment model.
    """

    enabled: bool = False
    hangul_mode: str = "lightweight"
    enable_h7_neutralized_mgp: bool = False
    enable_h8_neutralized_mgp: bool = False
    enable_decision_max_shadow: bool = False
    production_ocr_backend: str = "ppocrv5"
    enable_mgp_shadow: bool = False
    enable_easyocr_shadow: bool = False
    enable_tesseract_shadow: bool = False
    enable_real822_a4_shadow: bool = False

    def __post_init__(self) -> None:
        if self.hangul_mode not in {"lightweight", "max_shadow"}:
            raise ValueError("hangul_mode must be lightweight|max_shadow")
        if self.hangul_mode == "max_shadow" and not self.enable_decision_max_shadow:
            raise ValueError("max_shadow requires enable_decision_max_shadow=True")
        if self.production_ocr_backend != "ppocrv5":
            raise ValueError("Real822 production OCR backend is fixed to ppocrv5")


@dataclass(frozen=True, slots=True)
class PPV5ProductionDecision:
    """One PP-OCRv5 observation reused for digits and the Hangul slot."""

    raw_text: str
    digits: str
    hangul: str
    layout: str
    assembled_plate: str
    confidence: float
    state: str
    reason: str
    inference_count: int = 1


@dataclass(frozen=True, slots=True)
class PhysicalSlotGeometry:
    layout: str
    slot_count: int
    hangul_slot: int
    x1: int
    y1: int
    x2: int
    y2: int


def _sha256_array(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest().upper()


def physical_hangul_geometry(image: np.ndarray, layout: str) -> PhysicalSlotGeometry:
    """Return a deterministic physical H7/H8 Hangul slot.

    H7 is ``DD K DDDD`` and H8 is ``DDD K DDDD``.  The geometry is based only
    on image width; it never reads OCR text, string length, candidates, or GT.
    """

    if image.ndim not in {2, 3} or image.shape[0] < 1 or image.shape[1] < 7:
        raise ValueError("expected a non-empty plate image")
    name = str(layout).upper()
    if name == "H7":
        slots, hangul = 7, 2
    elif name == "H8":
        slots, hangul = 8, 3
    else:
        raise ValueError("layout must be H7 or H8")
    height, width = image.shape[:2]
    bounds = np.rint(np.linspace(0, width, slots + 1)).astype(np.int64)
    x1, x2 = int(bounds[hangul]), int(bounds[hangul + 1])
    if not (0 <= x1 < x2 <= width):
        raise AssertionError("invalid physical slot geometry")
    return PhysicalSlotGeometry(name, slots, hangul, x1, 0, x2, height)


def neutralize_physical_hangul_slot(
    original: np.ndarray, layout: str
) -> tuple[np.ndarray, PhysicalSlotGeometry, dict[str, Any]]:
    """Create a conservative copy with the physical Hangul slot neutralised.

    Each row/channel is filled with the median of pixels outside the slot. This
    preserves horizontal illumination and border structure without generating a
    glyph, blurring adjacent digits, or modifying ``original``.
    """

    source = np.asarray(original)
    geometry = physical_hangul_geometry(source, layout)
    output = source.copy()
    outside = np.concatenate(
        (source[:, : geometry.x1, ...], source[:, geometry.x2 :, ...]), axis=1
    )
    if outside.shape[1] == 0:
        raise ValueError("neutralisation has no background support")
    fill = np.median(outside, axis=1, keepdims=True)
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        fill = np.rint(fill).clip(info.min, info.max).astype(source.dtype)
    else:
        fill = fill.astype(source.dtype, copy=False)
    output[:, geometry.x1 : geometry.x2, ...] = fill
    audit = {
        "layout_hypothesis": geometry.layout,
        "slot_count": geometry.slot_count,
        "hangul_slot": geometry.hangul_slot,
        "roi_xyxy": [geometry.x1, geometry.y1, geometry.x2, geometry.y2],
        "method": "per_row_per_channel_outside_slot_median",
        "ocr_string_reads": 0,
        "gt_reads": 0,
        "input_hash": _sha256_array(source),
        "output_hash": _sha256_array(output),
        "original_unchanged": bool(np.array_equal(source, original)),
    }
    return output, geometry, audit


def prepare_h7_h8_variants(original: np.ndarray) -> dict[str, Any]:
    """Produce both hypotheses; preprocessing never hard-selects a layout."""

    started = time.perf_counter()
    h7, h7_geometry, h7_audit = neutralize_physical_hangul_slot(original, "H7")
    h8, h8_geometry, h8_audit = neutralize_physical_hangul_slot(original, "H8")
    return {
        "ORIGINAL": original,
        "H7_NEUT": h7,
        "H8_NEUT": h8,
        "geometry": {"H7": h7_geometry, "H8": h8_geometry},
        "audit": {"H7": h7_audit, "H8": h8_audit},
        "geometry_neutralization_ms": (time.perf_counter() - started) * 1000.0,
    }


def prepare_original_only(original: np.ndarray) -> dict[str, Any]:
    """Explicit no-neutralisation comparator and mandatory Hangul input.

    The returned array is the caller's original object by design: this function
    performs no pixel operation.  Consumers that mutate inputs must copy it.
    """

    source = np.asarray(original)
    return {
        "ORIGINAL": original,
        "source_variant": "ORIGINAL_NO_NEUTRALIZATION",
        "input_hash": _sha256_array(source),
        "pixel_transform": "NONE",
        "hangul_input_allowed": True,
        "gt_reads": 0,
        "ocr_string_reads": 0,
    }


def count_independent_frame_support(rows: Iterable[Mapping[str, Any]]) -> int:
    """Count physical frames, never RAW/H7/H8 variants as separate votes."""

    keys: set[tuple[str, str]] = set()
    for row in rows:
        track = str(row.get("track_id", row.get("group_id", "")))
        frame = str(row.get("parent_frame_id", row.get("frame_id", "")))
        if track or frame:
            keys.add((track, frame))
    return len(keys)


def assemble_fixed_slot_plate(digits: str, hangul: str, layout: str) -> str:
    """Assemble digits and Hangul using fixed physical layout semantics."""

    layout = str(layout).upper()
    expected_digits, prefix_len = (6, 2) if layout == "H7" else (7, 3) if layout == "H8" else (-1, -1)
    if expected_digits < 0:
        raise ValueError("layout must be H7 or H8")
    if len(digits) != expected_digits or not digits.isdigit():
        raise ValueError(f"{layout} requires exactly {expected_digits} digits")
    if len(hangul) != 1:
        raise ValueError("Hangul decision must be one slot character")
    return f"{digits[:prefix_len]}{hangul}{digits[prefix_len:]}"


def _strict_single_hangul(text: str) -> tuple[str, str]:
    """Collapse identical repeats; mixed Hangul must abstain."""

    chars = re.findall(r"[가-힣]", str(text or ""))
    if not chars:
        return "", "NO_HANGUL"
    unique = set(chars)
    if len(unique) != 1:
        return "", "MIXED_HANGUL_ABSTAIN"
    return chars[0], "IDENTICAL_REPEAT_COLLAPSED" if len(chars) > 1 else "SINGLE_HANGUL"


def ppv5_single_pass_decision(raw_text: str, confidence: float = 0.0) -> PPV5ProductionDecision:
    """Build a fixed-slot plate from exactly one PP-OCRv5 observation.

    This adapter performs no image preprocessing and does not call another OCR
    family.  The same decoder output supplies both the numeric sequence and the
    constrained Hangul slot.  Ambiguous/mixed Hangul or non-H7/H8 digit counts
    fail closed for String-FEBAM.
    """

    raw = str(raw_text or "")
    number = "".join(re.findall(r"[0-9]", raw))
    hangul, hangul_reason = _strict_single_hangul(raw)
    layout = "H7" if len(number) == 6 else "H8" if len(number) == 7 else ""
    if not raw:
        return PPV5ProductionDecision(raw, number, hangul, layout, "", float(confidence), "HOLD", "EMPTY_PPV5")
    if not layout:
        return PPV5ProductionDecision(raw, number, hangul, layout, "", float(confidence), "HOLD", "INVALID_DIGIT_COUNT")
    if not hangul:
        return PPV5ProductionDecision(raw, number, hangul, layout, "", float(confidence), "HOLD", hangul_reason)
    assembled = assemble_fixed_slot_plate(number, hangul, layout)
    return PPV5ProductionDecision(raw, number, hangul, layout, assembled, float(confidence), "OBSERVED", hangul_reason)


def attach_ppv5_after_legacy_grouping(
    upstream_rows: Iterable[Mapping[str, Any]],
    ppv5_rows: Iterable[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...] = ("group_id", "observation_id"),
) -> list[dict[str, Any]]:
    """Attach PPv5 only after the legacy pipeline freezes canonical groups.

    This is deliberately a left attachment to the legacy upstream authority,
    not a grouping join. YOLO/tracking/event grouping, color filtering,
    representative selection, fusion and Q2 remain owned by ``gpu_pipeline``.
    An OCR-only key can never create, merge, split or promote a production
    group. Missing or invalid OCR remains an auditable HOLD.
    """

    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        value = tuple(str(row.get(field, "") or "") for field in key_fields)
        if not any(value):
            raise ValueError("outer-join row has no canonical key")
        return value

    upstream: dict[tuple[str, ...], dict[str, Any]] = {}
    ocr: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in upstream_rows:
        row_key = key(row)
        if row_key in upstream:
            raise ValueError(f"duplicate upstream key: {row_key}")
        upstream[row_key] = dict(row)
    for row in ppv5_rows:
        row_key = key(row)
        if row_key in ocr:
            raise ValueError(f"duplicate PPv5 key: {row_key}")
        ocr[row_key] = dict(row)

    orphan_ocr_keys = set(ocr) - set(upstream)
    if orphan_ocr_keys:
        raise ValueError(f"PPv5 rows outside frozen legacy groups: {len(orphan_ocr_keys)}")
    joined = []
    for row_key in upstream:
        visual = upstream[row_key]
        prediction = ocr.get(row_key, {})
        raw_text = str(prediction.get("raw_text", prediction.get("text", "")) or "")
        confidence = float(prediction.get("confidence", prediction.get("conf", 0.0)) or 0.0)
        decision = ppv5_single_pass_decision(raw_text, confidence)
        record = dict(visual)
        for index, field in enumerate(key_fields):
            record[field] = row_key[index]
        record.update(
            {
                    "upstream_present": 1,
                "ppv5_present": int(bool(prediction)),
                "ppv5_raw_text": decision.raw_text,
                "ppv5_confidence": decision.confidence,
                "ppv5_digits": decision.digits,
                "ppv5_hangul": decision.hangul,
                "ppv5_layout": decision.layout,
                "ppv5_fixed_slot_plate": decision.assembled_plate,
                "production_decision_state": decision.state,
                "production_decision_reason": "PPV5_MISSING_HOLD" if not prediction else decision.reason,
                "grouping_authority": "LEGACY_GPU_PIPELINE_FROZEN",
                "group_mutation": 0,
                "production_ocr_authority": "PP-OCRv5_SINGLE_PASS",
                "mgp_production_reads": 0,
                "easyocr_production_reads": 0,
                "tesseract_production_reads": 0,
                "real822_a4_production_invocations": 0,
            }
        )
        joined.append(record)
    return joined


class GPUPipeline(_legacy.GPUPipeline):
    """Legacy GPUPipeline plus opt-in Real822 integration helpers.

    The constructor and ``process`` implementation are inherited unchanged,
    guaranteeing the old path when the caller does not use the new helpers.
    """

    final_gpu_integration_config = FinalGPUIntegrationConfig()

    def configure_c3_shadow(
        self,
        *,
        mode: str = "off",
        output_path: str = "outputs/c3_shadow/C3_DECISION_TRACE.json",
        replay_predictions: str = "",
    ) -> None:
        """Attach a GT-free observer; never change the production decision."""
        self._c3_shadow_collector = None
        self._c3_runtime_reconstruction = None
        self._c3_shadow_closed = False
        if mode == "off":
            return
        from decision.c3_shadow import C3ShadowCollector
        from decision.c3_runtime_reconstruction import C3RuntimeReconstruction
        self._c3_shadow_collector = C3ShadowCollector(
            output_path,
            mode=mode,
            replay_predictions=replay_predictions or None,
        )
        self._c3_runtime_reconstruction = C3RuntimeReconstruction()

    def configure_live_dual_ocr_shadow(self, *, g6: str, g6_config: str, ppv5: str, output: str, legal: str) -> None:
        from decision.live_dual_ocr_shadow import LiveDualOCRShadow
        self._live_dual_ocr_shadow = LiveDualOCRShadow(g6=g6, g6_config=g6_config, ppv5=ppv5, output=output, legal=legal)
        # Capture before OCR at the inherited async enqueue boundary.  This
        # preserves crops even when the legacy recognizer returns no text and
        # permits the existing batch worker to retain its production semantics.
        self.fastplate_async = True

    def _enqueue_fastplate_async(self, image_rgb, meta):
        observer = getattr(self, "_live_dual_ocr_shadow", None)
        if observer is not None:
            rgb = image_rgb if isinstance(image_rgb, np.ndarray) else self._ocr_crop_to_rgb_np(image_rgb)
            track_id = int(meta.get("track_id", -1))
            group_id, group_key = self._resolve_string_febam_group_id(meta, track_id)
            observer.capture_bound(
                rgb,
                variant=str(meta.get("variant_name", meta.get("variant", "unknown"))),
                group_id=str(group_key or group_id or track_id),
                track_id=track_id,
                frame_idx=meta.get("frame_idx"),
                candidate_idx=meta.get("candidate_idx"),
            )
        return super()._enqueue_fastplate_async(image_rgb, meta)

    def _append_ocr_history(self, track_id, text, conf, frame_idx, bbox, mrs, **kwargs):
        dual = getattr(self, "_live_dual_ocr_shadow", None)
        if dual is not None:
            group_key = str(kwargs.get("string_group_key", "") or kwargs.get("string_group_id", "") or track_id)
            dual.bind(group_id=group_key, track_id=int(track_id), frame_idx=frame_idx, candidate_idx=kwargs.get("candidate_idx"), legacy_raw_text=str(kwargs.get("raw_text", "") or ""))
        reconstruction = getattr(self, "_c3_runtime_reconstruction", None)
        if reconstruction is not None:
            group_key = str(kwargs.get("string_group_key", "") or "")
            group_id = group_key or kwargs.get("string_group_id", track_id)
            reconstruction.observe_passthrough(
                text,
                {
                    "group_id": group_id,
                    "track_id": int(track_id),
                    "frame_idx": frame_idx,
                    "physical_frame_id": f"track={int(track_id)}|frame={frame_idx}",
                    "candidate_idx": kwargs.get("candidate_idx"),
                    "source": str(kwargs.get("source", "raw")),
                    "text": str(text or ""),
                    "confidence": float(conf),
                },
            )
        collector = getattr(self, "_c3_shadow_collector", None)
        if collector is not None:
            try:
                group_key = str(kwargs.get("string_group_key", "") or "")
                group_id = group_key or str(kwargs.get("string_group_id", "") or track_id)
                collector.observe(
                    group_id=group_id,
                    track_id=int(track_id),
                    text=str(text or ""),
                    confidence=float(conf),
                    frame_idx=frame_idx,
                    source=str(kwargs.get("source", "raw")),
                )
            except Exception as exc:
                collector.errors.append(f"{type(exc).__name__}:{exc}")
        return super()._append_ocr_history(track_id, text, conf, frame_idx, bbox, mrs, **kwargs)

    def close(self):
        try:
            return super().close()
        finally:
            collector = getattr(self, "_c3_shadow_collector", None)
            if collector is not None and not bool(getattr(self, "_c3_shadow_closed", False)):
                try:
                    collector.close()
                except Exception as exc:
                    self._log(f"[C3_SHADOW] close failed error={type(exc).__name__}:{exc}")
                self._c3_shadow_closed = True
            dual = getattr(self, "_live_dual_ocr_shadow", None)
            if dual is not None:
                dual.close()

    def configure_ppocrv5_live(self, *, model_dir: str = "", device: str = "gpu:0") -> None:
        """Enable the opt-in Korean PP-OCRv5 live recognizer.

        Detection, tracking, grouping, representative selection, color-aware
        fusion and String-FEBAM remain inherited.  Only the recognizer called
        by the legacy synchronous FastPlate slot is replaced.  Initialization
        is lazy so importing this module never downloads a model.
        """
        self._ppocrv5_live_enabled = True
        self._ppocrv5_model_dir = str(model_dir or "")
        self._ppocrv5_device = str(device or "gpu:0")
        self._ppocrv5_model = None
        self.fastplate_async = False
        self.ocr_backend = "fastplate"
        self.final_gpu_integration_config = FinalGPUIntegrationConfig(enabled=True)

    def _get_ppocrv5_model(self):
        model = getattr(self, "_ppocrv5_model", None)
        if model is not None:
            return model
        import paddle
        from paddleocr import TextRecognition
        device = str(getattr(self, "_ppocrv5_device", "gpu:0"))
        if not device.startswith("gpu") or not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("PPOCRV5_GPU_REQUIRED_CPU_FALLBACK_FORBIDDEN")
        paddle.device.set_device(device)
        kwargs = {"model_name": "korean_PP-OCRv5_mobile_rec", "device": device, "enable_hpi": False}
        model_dir = str(getattr(self, "_ppocrv5_model_dir", "") or "")
        if model_dir:
            if not Path(model_dir).is_dir():
                raise FileNotFoundError(f"PP-OCRv5 model dir not found: {model_dir}")
            kwargs["model_dir"] = model_dir
        model = TextRecognition(**kwargs)
        self._ppocrv5_model = model
        return model

    def _run_fastplate_once(self, image_rgb: np.ndarray, variant_name: str = "unknown"):
        observer = getattr(self, "_live_dual_ocr_shadow", None)
        if observer is not None:
            observer.infer(image_rgb, variant_name)
        if not bool(getattr(self, "_ppocrv5_live_enabled", False)):
            result = super()._run_fastplate_once(image_rgb, variant_name=variant_name)
            if observer is not None:
                observer.tag_latest(str(result[0] or ""))
            return result
        image_bgr = np.ascontiguousarray(np.asarray(image_rgb)[..., ::-1])
        values = list(self._get_ppocrv5_model().predict(input=[image_bgr], batch_size=1))
        if len(values) != 1:
            raise RuntimeError(f"PPOCRV5_BATCH_ACCOUNTING:{len(values)}")
        payload = values[0].json
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        inner = payload.get("res", payload)
        raw_text = str(inner.get("rec_text", "") or "")
        confidence = float(inner.get("rec_score", 0.0) or 0.0)
        decision = ppv5_single_pass_decision(raw_text, confidence)
        pattern = "DDDKDDDD" if decision.layout == "H8" else "DDKDDDD" if decision.layout == "H7" else "empty"
        self._last_fastplate_result_meta = {
            "source": "ppocrv5_live_fixed_slot", "raw_result": inner,
            "variant_name": variant_name, "decision_state": decision.state,
            "decision_reason": decision.reason,
        }
        result = (raw_text, decision.assembled_plate, decision.assembled_plate,
                  confidence, pattern, int(bool(decision.assembled_plate)),
                  decision.reason, decision.hangul, decision.hangul)
        if observer is not None:
            observer.tag_latest(raw_text)
        return result

    @staticmethod
    def prepare_real822_h7_h8(original: np.ndarray) -> dict[str, Any]:
        return prepare_h7_h8_variants(original)

    @staticmethod
    def prepare_real822_original_only(original: np.ndarray) -> dict[str, Any]:
        return prepare_original_only(original)

    @staticmethod
    def real822_independent_frame_support(rows: Iterable[Mapping[str, Any]]) -> int:
        return count_independent_frame_support(rows)

    @staticmethod
    def assemble_real822_plate(digits: str, hangul: str, layout: str) -> str:
        return assemble_fixed_slot_plate(digits, hangul, layout)

    @staticmethod
    def real822_ppv5_single_pass(raw_text: str, confidence: float = 0.0) -> PPV5ProductionDecision:
        """OCR-only replacement hook; YOLO/tracking/fusion remain inherited."""

        return ppv5_single_pass_decision(raw_text, confidence)

    @staticmethod
    def attach_real822_ppv5_after_legacy_grouping(
        upstream_rows: Iterable[Mapping[str, Any]],
        ppv5_rows: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return attach_ppv5_after_legacy_grouping(upstream_rows, ppv5_rows)

    def submit_real822_ppv5_to_string_febam(
        self,
        *,
        track_id: int,
        raw_text: str,
        confidence: float,
        frame_idx: int | None,
        bbox: Any,
        febam_reliability: float,
        string_group_id: int | None = None,
        string_group_key: str = "",
    ) -> dict[str, Any]:
        """Submit the PPv5-only fixed-slot observation to legacy String-FEBAM.

        Detection, tracking, crop selection, image fusion and FEBAM reliability
        are inputs from the inherited pipeline.  This method replaces only the
        OCR decision source and disables legacy fallback voting for this path.
        """

        decision = ppv5_single_pass_decision(raw_text, confidence)
        committed = ""
        if decision.state == "OBSERVED":
            committed = self._append_ocr_history(
                int(track_id),
                decision.assembled_plate,
                float(decision.confidence),
                frame_idx,
                bbox,
                float(febam_reliability),
                raw_text=decision.raw_text,
                corrected_text=decision.assembled_plate,
                source="ppocrv5_real822_fixed_slot",
                source_weight=1.0,
                string_group_id=string_group_id,
                string_group_key=string_group_key,
                allow_legacy_vote=False,
            )
        temporal_id = int(string_group_id) if string_group_id is not None else int(track_id)
        state = dict(getattr(self, "string_febam_states", {}).get(temporal_id, {}))
        return {
            "decision_layer": {
                "stage": "PPV5_FIXED_SLOT_OBSERVED_DECISION",
                "ocr": decision,
                "observed_plate": decision.assembled_plate,
                "observed_state": decision.state,
                "observed_reason": decision.reason,
            },
            "stabilization_layer": {
                "stage": "STRING_FEBAM_TEMPORAL_STABILIZATION",
                "state": state.get("state", "HOLD"),
                "committed_plate": committed,
                "activation": state.get("activation", 0.0),
                "margin": state.get("margin", 0.0),
                "support_segments": (
                    state.get("nodes", [])[0].support_segments
                    if state.get("nodes") else 0
                ),
                "changes_observed_decision": int(bool(committed) and committed != decision.assembled_plate),
            },
            "production_sources": ["PP-OCRv5"],
            "shadow_sources_enabled": {
                "MGP": bool(self.final_gpu_integration_config.enable_mgp_shadow),
                "EasyOCR": bool(self.final_gpu_integration_config.enable_easyocr_shadow),
                "Tesseract": bool(self.final_gpu_integration_config.enable_tesseract_shadow),
                "Real822_A4": bool(self.final_gpu_integration_config.enable_real822_a4_shadow),
            },
        }


LEGACY_GPU_PIPELINE_CLASS = _legacy.GPUPipeline

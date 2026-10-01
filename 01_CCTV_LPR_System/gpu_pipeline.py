from __future__ import annotations

import importlib
import json
import math
import importlib.util
import inspect
import csv
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from detection.gpu_preprocessing import fused_preprocess
from detection.gpu_morphology import orientation_morphology
from detection.gpu_ccl import ccl_label_propagation
import detection.gpu_preprocessing as gpu_pre_mod
import detection.gpu_ccl as gpu_ccl_mod
from tracking.gpu_febam import GPUFEBAM
from detection.classifier import PlateMLPUpdater
from string_febam_core import StringFEBAMConfig, StringFEBAMEngine, StringFEBAMObservation
from pipeline.config import OCRConfig
from pipeline.ocr_bbox_geometry import resolve_ocr_crop_expand_ratios
from fusion.trial020_fusion_bridge import Trial020FusionBridge
from ocr.english_mixed_recovery import (
    EnglishMixedRecoveryDecoder,
    VehicleFEBAMRouter,
)
from pipeline.gpu_final_vehicle_grouping import GPUFinalVehicleGrouper
from pipeline.evidence import (
    EXT30EvidenceOrchestrator,
    PairEvidence,
    build_pair_evidence,
    route_by_observed_n,
    select_s4,
    SupportConstrainedFusionResult,
    apply_support_constrained_f4,
)
from pipeline.policies.ocr_policy import DefaultOCRPolicy
from pipeline.types import CandidateBatch as PipelineCandidateBatch, TrackBatch as PipelineTrackBatch
from pipeline.lovo_runtime import (
    DROP_METHODS,
    FinalVehiclePosteriorGate,
    RuntimeEvent,
    position_posterior,
)
from ocr.ocr_easy_batch import EasyOCRBatchRecognizer
from ocr.middle_slot_batch_worker import MiddleSlotBatchQueueWorker, VALID_KOR
from ocr.middle_slot_types import MiddleSlotResult
from ocr.middle_slot_upl import topk_to_jsonable
from ocr.middle_slot_gabor_scatter_gpu import MiddleSlotV32Runtime
from ocr.grammar_decoder import (
    clean_ocr_text,
    choose_final_plate,
    decode_korean_plate_candidates,
    decode_split_ocr_candidates,
    estimate_plate_layout,
    is_valid_final_plate,
    select_best_plate_candidate,
)

DEBUG_DRAW_CANDIDATE_STAGES = False
DEBUG_VERBOSE_DEBUG_IMAGES = False

BIO_CSV_FIELDS = (
    "bio_route",
    "bio_route_confidence",
    "bio_route_reason",
    "bio_sharpness",
    "bio_edge_completeness",
    "bio_local_contrast",
    "bio_blur_strength",
    "bio_ghost_strength",
    "bio_temporal_frames",
    "bio_temporal_segments",
    "bio_digit_locked",
    "bio_early_exit",
    "bio_fallback_reason",
    "bio_hog_conf",
    "bio_lbp_feature_norm",
    "bio_svm_top1",
    "bio_svm_top2",
    "bio_svm_margin",
    "bio_middle_crop_score",
    "bio_middle_crop_reason",
    "bio_latency_ms",
)


_KO_PLATE_RE = re.compile(r"^(\d{2,3})([가-힣])(\d{4})$")
_HOG_LBP_CONFUSED_GROUPS = (
    frozenset({"가", "거", "고", "구"}),
    frozenset({"나", "너", "노", "누"}),
    frozenset({"다", "더", "도", "두"}),
    frozenset({"라", "러", "로", "루"}),
    frozenset({"마", "머", "모", "무"}),
    frozenset({"바", "버", "보", "부"}),
    frozenset({"사", "서", "소", "수"}),
    frozenset({"아", "어", "오", "우"}),
    frozenset({"자", "저", "조", "주"}),
    frozenset({"하", "허", "호"}),
)


def _split_korean_plate_candidate(text: str) -> tuple[str, str, str]:
    s = str(text or "").strip()
    m = _KO_PLATE_RE.match(s)
    if not m:
        return "", "", ""
    return m.group(1), m.group(2), m.group(3)


def _digit_skeleton_of_plate(text: str) -> str:
    prefix, _ko, suffix = _split_korean_plate_candidate(text)
    if not prefix or not suffix:
        return ""
    return f"{prefix}?{suffix}"


def _simple_yaml_scalar_map(path: str | Path) -> dict[str, object]:
    data: dict[str, object] = {}
    if not path:
        return data
    cfg_path = Path(path)
    if not cfg_path.exists():
        return data
    for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        else:
            try:
                data[key] = int(value)
            except ValueError:
                data[key] = value
    return data


class CustomFastPlateONNXRecognizer:
    DEFAULT_ALPHABET = "0123456789가거고구나너노누다더도두라러로루마머모무바버보부사서소수아어오우자저조주하허호배_"

    def __init__(self, *, onnx_path: str, plate_config: str | None = None, input_width: int = 256, input_height: int = 64, device: str = "cuda", logger=None) -> None:
        self.logger = logger
        import onnxruntime as ort

        try:
            import torch  # noqa: F401
            self._log("[CUSTOM_FASTPLATE] torch imported for CUDA DLL preload")
        except Exception as e:
            self._log(f"[CUSTOM_FASTPLATE] torch preload skipped: {e}")

        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
                self._log("[CUSTOM_FASTPLATE] ort.preload_dlls() done")
        except Exception as e:
            self._log(f"[CUSTOM_FASTPLATE] ort.preload_dlls warning: {e}")

        self.onnx_path = str(onnx_path or "")
        self.plate_config = str(plate_config or "")
        self.input_width = int(max(1, input_width))
        self.input_height = int(max(1, input_height))
        self.device = str(device or "cuda")
        if not self.onnx_path:
            raise RuntimeError("fastplate custom ONNX path is empty")
        if not Path(self.onnx_path).exists():
            raise RuntimeError(f"fastplate custom ONNX not found: {self.onnx_path}")
        cfg = _simple_yaml_scalar_map(self.plate_config)
        self.alphabet = str(cfg.get("alphabet", self.DEFAULT_ALPHABET) or self.DEFAULT_ALPHABET)
        self.pad_char = str(cfg.get("pad_char", "_") or "_")
        self.max_plate_slots = int(cfg.get("max_plate_slots", 10) or 10)
        self.input_height = int(cfg.get("img_height", self.input_height) or self.input_height)
        self.input_width = int(cfg.get("img_width", self.input_width) or self.input_width)
        self.preprocess = str(cfg.get("preprocess", "norm01") or "norm01").lower()
        if self.preprocess not in {"norm01", "raw255"}:
            raise RuntimeError(f"unsupported custom FastPlate preprocessing: {self.preprocess}")
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.startswith("cuda") else ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [provider for provider in requested if provider in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.providers = list(self.session.get_providers())
        if self.device.startswith("cuda") and (not self.providers or self.providers[0] != "CUDAExecutionProvider"):
            raise RuntimeError(f"CUDAExecutionProvider must be the first provider, got {self.providers}")
        self.inputs = list(self.session.get_inputs())
        self.outputs = list(self.session.get_outputs())
        if not self.inputs:
            raise RuntimeError("custom FastPlate ONNX has no inputs")
        self.input = self.inputs[0]
        self.input_name = self.input.name
        self.input_shape = list(self.input.shape)
        self.output_names = [out.name for out in self.outputs]
        self.output_shapes = [list(out.shape) for out in self.outputs]
        self.input_layout = self._infer_input_layout(self.input_shape)
        self._log(f"[CUSTOM_FASTPLATE] onnx={self.onnx_path}")
        self._log(f"[CUSTOM_FASTPLATE] providers={self.providers}")
        self._log(f"[CUSTOM_FASTPLATE] alphabet_len={len(self.alphabet)}")
        self._log(f"[CUSTOM_FASTPLATE] input_shape={self.input_shape}")
        self._log(f"[CUSTOM_FASTPLATE] output_shape={self.output_shapes}")
        self.batch_call_count = 0
        self.batch_item_count = 0
        self.total_ms = 0.0
        self.preprocess_ms = 0.0
        self.infer_ms = 0.0
        self.decode_ms = 0.0
        self.error_count = 0
        self.iobinding_used = 0
        self.gpu_tensor_item_count = 0
        self._gpu_io_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, Any]] = {}

    def _log(self, message: str) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger(message)
        else:
            print(message, flush=True)

    @staticmethod
    def _infer_input_layout(shape: list[object]) -> str:
        if len(shape) == 4:
            if shape[1] in (1, 3):
                return "NCHW"
            if shape[-1] in (1, 3):
                return "NHWC"
        return "NHWC"

    def _prepare_batch(self, crops: list[np.ndarray], *, input_color: str = "bgr") -> np.ndarray:
        batch = []
        for crop in crops:
            arr = np.asarray(crop)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif str(input_color).lower() == "bgr":
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                arr = arr[:, :, :3]
            arr = cv2.resize(arr, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            arr = arr.astype(np.float32)
            if self.preprocess == "norm01":
                arr = arr / 255.0
            batch.append(arr)
        if not batch:
            return np.empty((0, self.input_height, self.input_width, 3), dtype=np.float32)
        prepared = np.ascontiguousarray(np.stack(batch, axis=0).astype(np.float32))
        if self.input_layout == "NCHW":
            prepared = np.ascontiguousarray(np.transpose(prepared, (0, 3, 1, 2)))
        return prepared

    @staticmethod
    def _softmax_if_needed(logits: np.ndarray) -> np.ndarray:
        sums = np.sum(logits, axis=-1)
        if np.all(np.isfinite(sums)) and float(np.nanmean(np.abs(sums - 1.0))) < 0.05:
            return logits
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-9)

    def _select_plate_output(self, outputs: list[np.ndarray], batch_size: int) -> np.ndarray:
        vocab = len(self.alphabet)
        for output in outputs:
            arr = np.asarray(output)
            if arr.ndim == 3 and arr.shape[0] == batch_size and arr.shape[-1] == vocab:
                return arr
        for output in outputs:
            arr = np.asarray(output)
            if arr.ndim == 2 and arr.shape[0] == batch_size and arr.shape[1] % vocab == 0:
                return arr.reshape(batch_size, arr.shape[1] // vocab, vocab)
        for output in outputs:
            arr = np.asarray(output)
            if arr.ndim == 3 and arr.shape[0] == batch_size:
                return arr
        shapes = [list(np.asarray(output).shape) for output in outputs]
        raise RuntimeError(f"no plate output compatible with alphabet_len={vocab} output_shapes={shapes}")

    def _decode_plate_output(self, plate: np.ndarray) -> list[dict[str, object]]:
        probs = self._softmax_if_needed(plate.astype(np.float32))
        ids = np.argmax(probs, axis=-1)
        slot_confs = np.max(probs, axis=-1)
        top2_confs = np.partition(probs, -2, axis=-1)[..., -2]
        slot_entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-9)), axis=-1)
        pad_idx = self.alphabet.find(self.pad_char)
        rows = []
        for row_ids, row_confs, row_top2, row_entropy in zip(ids, slot_confs, top2_confs, slot_entropy):
            chars = []
            keep_confs = []
            for idx, conf in zip(row_ids.tolist(), row_confs.tolist()):
                if idx < 0 or idx >= len(self.alphabet):
                    continue
                if idx == pad_idx or self.alphabet[idx] == self.pad_char:
                    continue
                chars.append(self.alphabet[idx])
                keep_confs.append(float(conf))
            text = "".join(chars)
            conf = float(np.mean(keep_confs)) if keep_confs else float(np.mean(row_confs)) if len(row_confs) else 0.0
            rows.append({
                "text": text,
                "conf": conf,
                "score": conf,
                "source": "fastplate_custom_onnx",
                "backend": "fastplate_custom_onnx",
                "ocr_source": "fastplate_custom_onnx",
                "raw_result": {
                    "plate_ids": row_ids.tolist(),
                    "slot_confs": [float(v) for v in row_confs.tolist()],
                    "slot_top2_confs": [float(v) for v in row_top2.tolist()],
                    "slot_margins": [float(v) for v in (row_confs - row_top2).tolist()],
                    "slot_entropy": [float(v) for v in row_entropy.tolist()],
                    "alphabet": self.alphabet,
                    "pad_char": self.pad_char,
                },
            })
        return rows

    def _gpu_io_buffer(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Return stable CUDA buffers for the only supported micro-batches."""
        if batch_size not in {1, 2, 4}:
            raise RuntimeError(f"GPU resident OCR only permits B1/B2/B4, got B{batch_size}")
        buffers = self._gpu_io_buffers.get(batch_size)
        if buffers is None:
            input_shape = ((batch_size, 3, self.input_height, self.input_width)
                           if self.input_layout == "NCHW"
                           else (batch_size, self.input_height, self.input_width, 3))
            input_buf = torch.empty(input_shape, dtype=torch.float32, device="cuda")
            output_buf = torch.empty((batch_size, 8, len(self.alphabet)), dtype=torch.float32, device="cuda")
            # Addresses and shapes are fixed per B1/B2/B4 pool, so bind once
            # and reuse.  Rebinding per request adds Python/CUDA API overhead.
            binding = self.session.io_binding()
            binding.bind_input(self.input_name, "cuda", 0, np.float32, tuple(input_buf.shape), input_buf.data_ptr())
            binding.bind_output(self.output_names[0], "cuda", 0, np.float32, tuple(output_buf.shape), output_buf.data_ptr())
            buffers = (input_buf, output_buf, binding)
            self._gpu_io_buffers[batch_size] = buffers
        return buffers

    def recognize_tensor_batch(self, crops: list[torch.Tensor], metas=None, variant_names=None) -> list[dict[str, object]]:
        """GPU crop -> fixed CUDA input -> ORT IOBinding -> small CPU logits.

        Input crops remain CUDA tensors until the 8x51 logits are decoded.
        This intentionally has no CPU fallback: callers must opt into the
        legacy NumPy recognizer explicitly if this path cannot be used.
        """
        if not crops:
            return []
        batch_size = len(crops)
        if any(not torch.is_tensor(crop) or not crop.is_cuda for crop in crops):
            raise RuntimeError("GPU resident OCR received a non-CUDA crop")
        try:
            t0 = time.perf_counter()
            prepared = []
            for crop in crops:
                x = crop[:3].unsqueeze(0)
                x = F.interpolate(x.float(), size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
                if self.preprocess == "norm01":
                    x.mul_(1.0 / 255.0)
                prepared.append(x)
            nchw = torch.cat(prepared, dim=0).contiguous()
            input_buf, output_buf, binding = self._gpu_io_buffer(batch_size)
            if self.input_layout == "NHWC":
                input_buf.copy_(nchw.permute(0, 2, 3, 1).contiguous(), non_blocking=True)
            else:
                input_buf.copy_(nchw, non_blocking=True)
            t1 = time.perf_counter()
            self.session.run_with_iobinding(binding)
            t2 = time.perf_counter()
            plate = output_buf.detach().cpu().numpy()
            rows = self._decode_plate_output(plate)
            t3 = time.perf_counter()
            self.batch_call_count += 1
            self.batch_item_count += batch_size
            self.gpu_tensor_item_count += batch_size
            self.iobinding_used += 1
            self.preprocess_ms += (t1 - t0) * 1000.0
            self.infer_ms += (t2 - t1) * 1000.0
            self.decode_ms += (t3 - t2) * 1000.0
            self.total_ms += (t3 - t0) * 1000.0
            for idx, row in enumerate(rows):
                if metas is not None and idx < len(metas): row["meta"] = dict(metas[idx] or {})
                if variant_names is not None and idx < len(variant_names): row["variant_name"] = str(variant_names[idx])
            return rows
        except Exception:
            self.error_count += 1
            raise

    def recognize_batch(self, crops: list[np.ndarray], metas=None, variant_names=None, *, input_color: str = "bgr") -> list[dict[str, object]]:
        if not crops:
            return []
        try:
            t0 = time.perf_counter()
            batch = self._prepare_batch(crops, input_color=input_color)
            t1 = time.perf_counter()
            outputs = self.session.run(self.output_names, {self.input_name: batch})
            t2 = time.perf_counter()
            plate = self._select_plate_output([np.asarray(output) for output in outputs], len(crops))
            rows = self._decode_plate_output(plate)
            t3 = time.perf_counter()
            self.batch_call_count += 1
            self.batch_item_count += len(crops)
            self.preprocess_ms += (t1 - t0) * 1000.0
            self.infer_ms += (t2 - t1) * 1000.0
            self.decode_ms += (t3 - t2) * 1000.0
            self.total_ms += (t3 - t0) * 1000.0
            for idx, row in enumerate(rows):
                if metas is not None and idx < len(metas):
                    row["meta"] = dict(metas[idx] or {})
                if variant_names is not None and idx < len(variant_names):
                    row["variant_name"] = str(variant_names[idx])
            return rows
        except Exception as exc:
            self.error_count += 1
            message = f"fastplate_custom_onnx_error {type(exc).__name__}: {exc}"
            self._log(message)
            return [{
                "text": "",
                "conf": 0.0,
                "score": 0.0,
                "source": "fastplate_custom_onnx",
                "backend": "fastplate_custom_onnx",
                "ocr_source": "fastplate_custom_onnx",
                "error": "fastplate_custom_onnx_error",
                "error_message": str(exc),
            } for _ in crops]

    def stats(self) -> dict[str, object]:
        calls = max(1, int(self.batch_call_count))
        items = max(1, int(self.batch_item_count))
        return {
            "fastplate_custom_provider": ",".join(self.providers),
            "fastplate_custom_cuda_used": 1.0 if "CUDAExecutionProvider" in self.providers else 0.0,
            "fastplate_custom_input_name": self.input_name,
            "fastplate_custom_output_names": ",".join(self.output_names),
            "fastplate_custom_input_shape": str(self.input_shape),
            "fastplate_custom_output_shapes": str(self.output_shapes),
            "fastplate_custom_input_layout": self.input_layout,
            "fastplate_custom_preprocess": self.preprocess,
            "fastplate_custom_alphabet_len": float(len(self.alphabet)),
            "fastplate_custom_batch_call_count": float(self.batch_call_count),
            "fastplate_custom_item_count": float(self.batch_item_count),
            "fastplate_custom_avg_batch_size": float(self.batch_item_count) / calls,
            "fastplate_custom_total_ms": float(self.total_ms),
            "fastplate_custom_avg_ms_per_crop": float(self.total_ms) / items,
            "fastplate_custom_preprocess_ms": float(self.preprocess_ms),
            "fastplate_custom_infer_ms": float(self.infer_ms),
            "fastplate_custom_decode_ms": float(self.decode_ms),
            "fastplate_custom_error_count": float(self.error_count),
            "fastplate_tensor_iobinding_used": float(self.iobinding_used),
            "fastplate_crop_gpu_tensor_count": float(self.gpu_tensor_item_count),
        }

    def recognize_one_box(self, image_rgb: np.ndarray, meta=None, variant_name: str = "unknown") -> dict[str, object]:
        rows = self.recognize_batch([image_rgb], metas=[meta or {}], variant_names=[variant_name], input_color="rgb")
        return rows[0] if rows else {"text": "", "conf": 0.0, "source": "fastplate_custom_onnx", "backend": "fastplate_custom_onnx"}



def _is_placeholder_digit_skeleton(skeleton: str) -> bool:
    s = str(skeleton or "")
    if "?" not in s:
        return True
    prefix, suffix = s.split("?", 1)
    if prefix in {"00", "000"}:
        return True
    if suffix in {"0000", "1111"}:
        return True
    if len(suffix) == 4 and len(set(suffix)) == 1:
        return True
    return not (prefix and suffix and len(suffix) == 4)


def _collapse_repeated_digits(s: str) -> str:
    out: list[str] = []
    for ch in str(s or ""):
        if not ch.isdigit():
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


def _generate_digit_skeleton_candidates(text: str) -> list[dict[str, object]]:
    raw = str(text or "").strip()
    digits = re.sub(r"\D", "", raw)
    out: list[dict[str, object]] = []

    def add(prefix: str, suffix: str, score: float, reason: str) -> None:
        if not prefix or not suffix or len(suffix) != 4:
            return
        skeleton = f"{prefix}?{suffix}"
        if _is_placeholder_digit_skeleton(skeleton):
            return
        out.append({"skeleton": skeleton, "score": float(score), "reason": reason, "source_text": raw})

    prefix, _ko, suffix = _split_korean_plate_candidate(raw)
    if prefix and suffix:
        add(prefix, suffix, 1.0, "valid_korean_plate")
        return out

    if len(digits) == 7:
        add(digits[:2], digits[-4:], 0.90, "digit7_slot_noise")
    elif len(digits) == 8:
        add(digits[:3], digits[-4:], 0.85, "digit8_slot_noise")
        collapsed = _collapse_repeated_digits(digits)
        if collapsed != digits and len(collapsed) in {7, 8}:
            for row in _generate_digit_skeleton_candidates(collapsed):
                row = dict(row)
                row["score"] = float(row.get("score", 0.0)) * 0.70
                row["reason"] = "repeat_collapse_" + str(row.get("reason", ""))
                row["source_text"] = raw
                out.append(row)

    best: dict[str, dict[str, object]] = {}
    for row in out:
        skeleton = str(row.get("skeleton", ""))
        if skeleton and (skeleton not in best or float(row.get("score", 0.0)) > float(best[skeleton].get("score", 0.0))):
            best[skeleton] = row
    return sorted(best.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)


def _hog_lbp_confused_neighbors(ko: str) -> set[str]:
    ch = str(ko or "")[:1]
    for group in _HOG_LBP_CONFUSED_GROUPS:
        if ch in group:
            return {v for v in group if v != ch}
    return set()


@dataclass
class StringFEBAMNode:
    text: str
    score: float = 0.0
    energy: float = 0.0
    memory: float = 0.0
    activation: float = 0.0
    last_frame: int = -1
    support_segments: int = 0
    total_weight: float = 0.0
    cluster_id: int = -1
    format_score: float = 0.0
    first_frame: int = -1




def _bottom_motion_prior(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    corrected bottom motion prior

    Intended plate paths in ROI-local normalized coordinates:

    1) left-middle  -> right-bottom
    2) right-middle -> left-bottom
    3) left-middle  -> center-bottom
    4) right-middle -> center-bottom

    Important:
    This is not an X-shaped symmetric prior.
    Top area must be weak.
    Bottom corridor must be strong.
    """
    yy = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1)
    xx = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W)

    # Lower-half activation.
    # Suppress top and middle-high areas.
    bottom_weight = ((yy - 0.42) / 0.45).clamp(0.0, 1.0)

    # Path 1: left-middle -> right-bottom
    # x=0: y≈0.52, x=1: y≈0.88
    y_lr = 0.52 + 0.36 * xx

    # Path 2: right-middle -> left-bottom
    # x=1: y≈0.52, x=0: y≈0.88
    y_rl = 0.52 + 0.36 * (1.0 - xx)

    # Path 3/4: both sides -> center-bottom
    # x=0 or 1: y≈0.54, x=0.5: y≈0.86
    y_mid = 0.54 + 0.32 * (1.0 - (xx - 0.5).abs() * 2.0).clamp(0.0, 1.0)

    sigma = 0.075

    p_lr = torch.exp(-((yy - y_lr).abs() / sigma))
    p_rl = torch.exp(-((yy - y_rl).abs() / sigma))
    p_mid = torch.exp(-((yy - y_mid).abs() / sigma))

    # Center-bottom path is more important than pure crossing diagonals.
    prior = torch.maximum(
        0.65 * p_mid,
        torch.maximum(0.45 * p_lr, 0.45 * p_rl),
    )

    # Enforce bottom-dominant prior.
    prior = prior * bottom_weight

    return prior.clamp(0.0, 1.0)


@dataclass
class CandidateBatch:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


@dataclass(slots=True)
class OCRTask:
    priority: int
    frame_idx: int
    track_id: int
    candidate_idx: int
    variant: str
    crop_bgr: Any
    meta: dict[str, Any]
    enqueue_time: float
    key: tuple


@dataclass(slots=True)
class OCRResultItem:
    frame_idx: int
    track_id: int
    candidate_idx: int
    variant: str
    text: str
    conf: float
    source: str
    meta: dict[str, Any]
    delay_ms: float
    crop_bgr: Any = None


class DeferredCudaBoundaryProfiler:
    """Deferred CUDA-event profiler: no host timing reads in frame processing."""
    def __init__(self, enabled=False, warmup_frames=8, max_frames=256):
        self.enabled = bool(enabled); self.warmup_frames=max(0,int(warmup_frames)); self.max_frames=max(1,int(max_frames))
        self.records=[]; self.profiled_frames=0; self.warmup_skipped=0; self.dropped_frames=0; self.last_events={}
    def begin_frame(self, frame_idx):
        if not self.enabled: return None
        if self.warmup_skipped < self.warmup_frames: self.warmup_skipped+=1; return None
        if self.profiled_frames >= self.max_frames: self.dropped_frames+=1; return None
        self.profiled_frames+=1; return {"frame_idx":frame_idx,"items":[]}
    def measure(self, rec, name, fn):
        if rec is None: return fn()
        stream=torch.cuda.current_stream(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(stream); out=fn(); end.record(stream)
        rec["items"].append((name,start,end,id(stream))); self.last_events[id(stream)]=end
        return out
    def end_frame(self, rec):
        if rec is not None: self.records.append(rec)
    def finalize(self):
        for event in self.last_events.values(): event.synchronize()
        values={}
        for rec in self.records:
            for name,start,end,_stream in rec["items"]: values.setdefault(name,[]).append(float(start.elapsed_time(end)))
        stages={}; total=sum(sum(v) for v in values.values())
        for name,v in values.items():
            s=sorted(v); n=len(s); stages[name]={"calls":n,"total_ms":sum(v),"avg_ms":sum(v)/n,"min_ms":s[0],"p50_ms":s[n//2],"p95_ms":s[min(n-1,int(n*.95))],"max_ms":s[-1],"percent_of_profiled_gpu_time":(sum(v)/total*100 if total else 0)}
        largest=max(stages,key=lambda k:stages[k]["avg_ms"],default="")
        return {"status":"ok" if stages else "no_profile_samples","profiled_frames":self.profiled_frames,"warmup_skipped":self.warmup_skipped,"dropped_profile_frames":self.dropped_frames,"event_count":sum(len(r['items'])*2 for r in self.records),"stream_count":len(self.last_events),"stages":stages,"largest_gpu_stage":largest,"largest_gpu_stage_avg_ms":stages.get(largest,{}).get("avg_ms",0.0)}
    def write_json(self,path):
        summary=self.finalize(); Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); summary["profile_output_path"]=str(path); return summary


class Stage0GPUDecoder:
    def __init__(self, input_uri: str, debug_stage_log: bool = False, profile_frame_limit: int | None = None, logger=None, fps: float = 25.0, debug_dump_seconds: set[int] | None = None, start_frame: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: GPU-only pipeline requires NVIDIA CUDA")
        self.input_uri = input_uri
        self.debug_stage_log = debug_stage_log
        self.logger = logger
        self.profile_frame_limit = profile_frame_limit
        self.total_frames = None
        self.fps = fps
        self.debug_dump_seconds = debug_dump_seconds or set()
        # Keep the pipeline's frame IDs global when a bounded debug window is
        # requested.  Without a decoder seek, a 145 s smoke test first decoded
        # and discarded about 4,350 frames before any profile row was produced.
        self.start_frame = 0

        spec = importlib.util.find_spec("PyNvVideoCodec")
        if spec is None:
            raise RuntimeError("PyNvVideoCodec 모듈을 찾을 수 없습니다. GPU-only 경로는 NVDEC 필수")
        self.pynv = importlib.import_module("PyNvVideoCodec")
        self.simple_decoder = self._create_simple_or_threaded_decoder()
        self.demuxer = None
        self.decoder = None
        if self.simple_decoder is None:
            self.demuxer = self._create_demuxer(self.input_uri)
            self.decoder = self._create_decoder_from_demuxer()
            if self.logger is not None:
                self.logger("using PyNvVideoCodec demuxer/decoder")
        elif self.logger is not None:
            self.logger("using SimpleDecoder")
        self._seek_to_start_frame(int(max(0, start_frame)))

    def _seek_to_start_frame(self, start_frame: int) -> None:
        """Seek only when the selected NVDEC implementation exposes it.

        The installed SimpleDecoder's seek_to_index() advances twice (for
        example, index 300 starts at source frame 600). Keep global frame IDs
        exact by using the existing sequential GPU decode path.
        """
        if start_frame > 0 and self.logger is not None:
            self.logger(f"Stage0 exact sequential decode to frame={start_frame}")

    def _create_simple_or_threaded_decoder(self) -> Any | None:
        nvc = self.pynv
        if hasattr(nvc, "CreateSimpleDecoder") and callable(getattr(nvc, "CreateSimpleDecoder")):
            try:
                return nvc.CreateSimpleDecoder(encSource=self.input_uri, gpuid=0, useDeviceMemory=True, outputColorType=nvc.OutputColorType.RGB)
            except Exception:
                return None
        return None

    def _create_demuxer(self, uri: str) -> Any:
        if hasattr(self.pynv, "CreateDemuxer"):
            return self.pynv.CreateDemuxer(uri)
        return self.pynv.PyNvDemuxer(uri)

    def _create_decoder_from_demuxer(self) -> Any:
        nvc = self.pynv
        codec = getattr(getattr(nvc, "cudaVideoCodec"), "H264")
        return nvc.CreateDecoder(gpuid=0, codec=codec, usedevicememory=True, outputColorType=nvc.OutputColorType.NATIVE)

    def _frame_to_torch_cuda(self, frame: Any) -> torch.Tensor:
        t = torch.utils.dlpack.from_dlpack(frame) if hasattr(frame, "__dlpack__") else frame
        if not isinstance(t, torch.Tensor):
            raise RuntimeError("지원하지 않는 디코더 프레임 타입")
        if not t.is_cuda:
            raise RuntimeError("Stage0 decoder produced CPU tensor")
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1).contiguous().unsqueeze(0)
        elif t.ndim == 3 and t.shape[0] == 3:
            t = t.contiguous().unsqueeze(0)
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        return t

    def _iter_from_decoder_object(self, dec: Any):
        frame_idx = 0
        while True:
            batch = dec.get_batch_frames(1)
            if not batch:
                break
            frames = batch if isinstance(batch, (list, tuple)) else [batch]
            for frame in frames:
                yield self._frame_to_torch_cuda(frame)
                frame_idx += 1
                if self.profile_frame_limit is not None and frame_idx >= self.profile_frame_limit:
                    return

    def __iter__(self):
        if self.simple_decoder is not None:
            yield from self._iter_from_decoder_object(self.simple_decoder)
            return
        demux_fn = getattr(self.demuxer, "DemuxSinglePacket", None) or getattr(self.demuxer, "demux")
        decode_fn = getattr(self.decoder, "Decode", None) or getattr(self.decoder, "decode")
        while True:
            pkt = demux_fn()
            if pkt is None:
                break
            out = decode_fn(pkt)
            if out is None:
                continue
            frames = out if isinstance(out, (list, tuple)) else [out]
            for frame in frames:
                yield self._frame_to_torch_cuda(frame)


class FastPlateOCRQueueWorker:
    PRIORITY_FEBAM_CONFIRMED = 0
    PRIORITY_NEAR_CONFIRMED = 1
    PRIORITY_CURRENT_BEST = 2
    PRIORITY_WEAK_PLATE_SAMPLE = 3
    PRIORITY_PERIODIC_VERIFY = 4
    PRIORITIES = (0, 1, 2, 3, 4)
    DEFAULT_MIN_GAP_BY_PRIORITY = {0: 3, 1: 3, 2: 5, 3: 10, 4: 15}
    RELAXED_MIN_GAP_BY_PRIORITY = {0: 1, 1: 1, 2: 2, 3: 3, 4: 5}

    def __init__(
        self,
        *,
        model: str,
        device: str,
        batch_size: int = 32,
        min_batch_size: int = 4,
        preload_torch_cuda_dlls: bool = True,
        min_text_len: int = 4,
        flush_timeout_ms: float = 150.0,
        queue_max: int = 512,
        rate_limit_relaxed: bool = False,
        large_batch_mode: bool = False,
        target_batch_size: int = 64,
        max_flush_timeout_ms: float = 1500.0,
        gpu_crop_batch: bool = False,
        tensor_runner_enabled: bool = False,
        tensor_runner_mode: str = "disabled",
        tensor_input_h: int = 96,
        tensor_input_w: int = 384,
        parity_sample_limit: int = 256,
        parity_log_csv: str = "",
        tensor_profile: bool = False,
        direct_ort_enabled: bool = False,
        direct_ort_iobinding: bool = True,
        direct_ort_debug: bool = False,
        direct_ort_model_path: str = "",
        direct_ort_input_h: int = 0,
        direct_ort_input_w: int = 0,
        direct_ort_compare_wrapper: bool = False,
        tensor_iobinding: bool = True,
        tensor_enable_direct_ort: bool = False,
        tensor_model_path_override: str = "",
        tensor_direct_input_h: int = 0,
        tensor_direct_input_w: int = 0,
        tensor_direct_debug: bool = False,
        tensor_compare_wrapper: bool = False,
        preserve_aspect_ratio: bool = False,
        fastplate_custom_onnx: str | None = None,
        fastplate_custom_plate_config: str | None = None,
        fastplate_custom_input_width: int = 256,
        fastplate_custom_input_height: int = 64,
        dual_branch_ocr: bool = False,
        dual_branch_global_onnx: str = "",
        dual_branch_korean_onnx: str = "",
        dual_branch_hangul_weight: float = 1.2,
        dual_branch_digit_alignment_mode: str = "monotonic_dp",
        dual_branch_digit_mass_thr: float = 0.50,
        dual_branch_digit_conf_thr: float = 0.50,
        dual_branch_digit_margin_thr: float = 0.10,
        dual_branch_global_input_variant: str = "raw",
        dual_branch_gray_secondary: bool = False,
        dual_branch_global_crop_mode: str = "gpu_crop",
        dual_branch_global_result_mode: str = "digits_only",
        dual_branch_korean_mode: str = "training_exact",
        dual_branch_korean_slot_mode: str = "independent",
        dual_branch_korean_mass_thr: float = 0.50,
        dual_branch_korean_conf_thr: float = 0.50,
        dual_branch_korean_margin_thr: float = 0.10,
        dual_branch_length_margin_thr: float = 0.15,
        dual_branch_length_hold: bool = True,
        logger=None,
    ) -> None:
        self.model = str(model or "cct-s-v2-global-model")
        self.fastplate_custom_onnx = str(fastplate_custom_onnx or "")
        self.fastplate_custom_plate_config = str(fastplate_custom_plate_config or "")
        self.fastplate_custom_input_width = int(max(1, fastplate_custom_input_width))
        self.fastplate_custom_input_height = int(max(1, fastplate_custom_input_height))
        self.dual_branch_ocr = bool(dual_branch_ocr)
        self.dual_branch_global_onnx = str(dual_branch_global_onnx or "")
        self.dual_branch_korean_onnx = str(dual_branch_korean_onnx or "")
        self.dual_branch_hangul_weight = float(dual_branch_hangul_weight)
        self.dual_branch_digit_alignment_mode = str(dual_branch_digit_alignment_mode)
        self.dual_branch_digit_mass_thr = float(dual_branch_digit_mass_thr)
        self.dual_branch_digit_conf_thr = float(dual_branch_digit_conf_thr)
        self.dual_branch_digit_margin_thr = float(dual_branch_digit_margin_thr)
        self.dual_branch_global_input_variant = str(dual_branch_global_input_variant or "raw")
        self.dual_branch_gray_secondary = bool(dual_branch_gray_secondary)
        self.dual_branch_global_crop_mode = str(dual_branch_global_crop_mode or "gpu_crop")
        self.dual_branch_global_result_mode = str(dual_branch_global_result_mode or "digits_only")
        self.dual_branch_korean_mode = str(dual_branch_korean_mode or "training_exact")
        self.dual_branch_korean_slot_mode = str(dual_branch_korean_slot_mode or "independent")
        self.dual_branch_korean_mass_thr = float(dual_branch_korean_mass_thr)
        self.dual_branch_korean_conf_thr = float(dual_branch_korean_conf_thr)
        self.dual_branch_korean_margin_thr = float(dual_branch_korean_margin_thr)
        self.dual_branch_length_margin_thr = float(dual_branch_length_margin_thr)
        self.dual_branch_length_hold = bool(dual_branch_length_hold)
        self.device = str(device or "cuda")
        self.batch_size = int(max(1, batch_size))
        self.min_batch_size = int(max(1, min_batch_size))
        self.preload_torch_cuda_dlls = bool(preload_torch_cuda_dlls)
        self.min_text_len = int(max(0, min_text_len))
        self.flush_timeout_ms = float(max(1.0, flush_timeout_ms))
        self.queue_max = int(max(1, queue_max))
        self.rate_limit_relaxed = bool(rate_limit_relaxed)
        self.large_batch_mode = bool(large_batch_mode)
        self.target_batch_size = int(max(1, min(target_batch_size, self.batch_size)))
        self.max_flush_timeout_ms = float(max(self.flush_timeout_ms, max_flush_timeout_ms))
        self.gpu_crop_batch = bool(gpu_crop_batch)
        self.tensor_runner_enabled = bool(tensor_runner_enabled)
        self.tensor_runner_mode = str(tensor_runner_mode or "disabled").lower().strip()
        if self.tensor_runner_mode not in {"disabled", "shadow", "active"}:
            self.tensor_runner_mode = "disabled"
        if not self.tensor_runner_enabled:
            self.tensor_runner_mode = "disabled"
        self.tensor_input_h = int(max(1, tensor_input_h))
        self.tensor_input_w = int(max(1, tensor_input_w))
        self.parity_sample_limit = int(max(0, parity_sample_limit))
        self.parity_log_csv = str(parity_log_csv or "")
        self.tensor_profile = bool(tensor_profile)
        self.direct_ort_enabled = bool(direct_ort_enabled or tensor_enable_direct_ort)
        self.direct_ort_iobinding = bool(direct_ort_iobinding if direct_ort_enabled else tensor_iobinding)
        self.direct_ort_debug = bool(direct_ort_debug or tensor_direct_debug)
        self.direct_ort_model_path = str(direct_ort_model_path or tensor_model_path_override or "")
        self.direct_ort_input_h = int(max(0, direct_ort_input_h or tensor_direct_input_h))
        self.direct_ort_input_w = int(max(0, direct_ort_input_w or tensor_direct_input_w))
        self.direct_ort_compare_wrapper = bool(direct_ort_compare_wrapper or tensor_compare_wrapper)
        if self.direct_ort_enabled:
            self.tensor_runner_enabled = True
            if self.tensor_runner_mode == "disabled":
                self.tensor_runner_mode = "active"
        if self.fastplate_custom_onnx or self.dual_branch_ocr:
            self.tensor_runner_enabled = False
            self.tensor_runner_mode = "disabled"
            self.direct_ort_enabled = False
        self.tensor_iobinding = bool(self.direct_ort_iobinding)
        self.tensor_enable_direct_ort = bool(self.direct_ort_enabled)
        self.tensor_model_path_override = self.direct_ort_model_path
        self.tensor_direct_input_h = self.direct_ort_input_h
        self.tensor_direct_input_w = self.direct_ort_input_w
        self.tensor_direct_debug = self.direct_ort_debug
        self.tensor_compare_wrapper = self.direct_ort_compare_wrapper
        self.preserve_aspect_ratio = bool(preserve_aspect_ratio)
        self.tensor_runner = None
        self.min_gap_by_priority = dict(
            self.RELAXED_MIN_GAP_BY_PRIORITY if self.rate_limit_relaxed else self.DEFAULT_MIN_GAP_BY_PRIORITY
        )
        self.buckets: dict[int, deque[OCRTask]] = {priority: deque() for priority in self.PRIORITIES}
        self.output_queue: queue.Queue[OCRResultItem] = queue.Queue()
        self.seen_keys: set[tuple] = set()
        self.last_enqueue_frame_by_track_variant: dict[tuple[int, str], int] = {}
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.logger = logger
        self.recognizer = None
        self.ocr_session_create_count = 0
        self.ocr_worker_create_count = 1
        self.enqueued = 0
        self.processed = 0
        self.dual_branch_raw_enqueued = 0
        self.dual_branch_gray_enqueued = 0
        self.dual_branch_raw_processed = 0
        self.dual_branch_gray_processed = 0
        self.batch_call_count = 0
        self.batch_size_total = 0
        self.fastplate_batch_size_hist = Counter()
        self.fastplate_tensor_batch_size_hist = Counter()
        self.total_ocr_ms = 0.0
        self.queue_dropped = 0
        self.queue_drop_weak = 0
        self.queue_drop_duplicate = 0
        self.queue_drop_rate_limited = 0
        self.queue_drop_backpressure = 0
        self.queue_max_seen = 0
        self.fastplate_gpu_crop_count = 0
        self.fastplate_cpu_crop_count = 0
        self.fastplate_gpu_crop_ms = 0.0
        self.fastplate_cpu_crop_ms = 0.0
        self.fastplate_crop_cpu_copy_count = 0
        self.fastplate_crop_gpu_tensor_count = 0
        self.fastplate_tensor_batch_call_count = 0
        self.fastplate_tensor_batch_size_total = 0
        self.fastplate_tensor_total_ms = 0.0
        self.fastplate_tensor_preprocess_ms = 0.0
        self.fastplate_tensor_infer_ms = 0.0
        self.fastplate_tensor_decode_ms = 0.0
        self.fastplate_tensor_iobinding_used = 0
        self.fastplate_tensor_cpu_fallback_count = 0
        self.fastplate_tensor_error_count = 0
        self.fastplate_tensor_runner_shadow_count = 0
        self.fastplate_tensor_runner_active_used = 0
        self.fastplate_tensor_direct_backend_available = 0
        self.fastplate_tensor_backend_kind = ""
        self.fastplate_tensor_gpu_only_claim_valid = 0
        self.fastplate_tensor_active_requested = 0
        self.fastplate_tensor_active_success_count = 0
        self.fastplate_tensor_active_blocked_count = 0
        self.fastplate_tensor_active_fallback_used = 0
        self.fastplate_tensor_active_fallback_reason = ""
        self.fastplate_tensor_active_blocked_log_count = 0
        self.fastplate_direct_ort_enabled = 0
        self.fastplate_direct_ort_iobinding = 0
        self.fastplate_direct_ort_compare_wrapper = 0
        self.fastplate_tensor_direct_backend_init_error = ""
        self.fastplate_tensor_direct_model_path_found = 0.0
        self.fastplate_tensor_direct_session_created = 0.0
        self.fastplate_tensor_direct_decode_ready = 0.0
        self.fastplate_tensor_model_path = ""
        self.fastplate_tensor_input_layout = ""
        self.fastplate_tensor_output_shapes = ""
        self.fastplate_parity_sample_count = 0
        self.fastplate_parity_success_count = 0
        self.fastplate_parity_error_count = 0
        self.fastplate_parity_skipped_count = 0
        self.fastplate_parity_text_match_count = 0
        self.fastplate_parity_normalized_match_count = 0
        self.fastplate_parity_korean_slot_match_count = 0
        self.fastplate_parity_conf_abs_diff_sum = 0.0
        self.fastplate_tensor_last_error_type = ""
        self.fastplate_tensor_last_error_message = ""
        self.fastplate_tensor_last_error_stage = ""
        self.fastplate_tensor_error_sample_input_shape = ""
        self.fastplate_tensor_error_sample_input_dtype = ""
        self.fastplate_tensor_error_sample_batch_len = 0.0
        self.fastplate_tensor_provider = ""
        self.fastplate_tensor_input_name = ""
        self.fastplate_tensor_output_names = ""
        self.fastplate_tensor_error_sample_output_shapes = ""
        self.fastplate_tensor_preprocess_mode = ""
        self.fastplate_tensor_decoder_mode = ""
        self.fastplate_tensor_last_input_shape = ""
        self.fastplate_tensor_last_input_dtype = ""
        self.fastplate_tensor_prepare_valid_count = 0.0
        self.fastplate_tensor_prepare_failed_count = 0.0
        self.fastplate_tensor_prepare_failed_examples = ""
        self.fastplate_tensor_preprocess_resize_count = 0.0
        self.fastplate_tensor_error_log_count = 0
        if self.fastplate_custom_onnx:
            msg = f"[CUSTOM_FASTPLATE_WORKER_REQUESTED] onnx={self.fastplate_custom_onnx} plate_config={self.fastplate_custom_plate_config}"
            print(msg, flush=True)
            self._log(msg)
            try:
                self._get_recognizer()
            except Exception as exc:
                self.fastplate_tensor_error_count += 1
                self.fastplate_tensor_last_error_type = type(exc).__name__
                self.fastplate_tensor_last_error_message = str(exc)[:500]
                self.fastplate_tensor_last_error_stage = "custom_fastplate_eager_init"
                msg = f"[CUSTOM_FASTPLATE_WORKER_INIT_FAILED] {type(exc).__name__}: {exc}"
                print(msg, flush=True)
                self._log(msg)

        self.thread = threading.Thread(target=self._run, name="fastplate-ocr-worker", daemon=True)
        self.thread.start()

    def _log(self, message: str) -> None:
        if self.logger is not None:
            try:
                self.logger(message)
            except Exception:
                pass

    def _qsize_locked(self) -> int:
        return sum(len(bucket) for bucket in self.buckets.values())

    def _oldest_enqueue_time_locked(self) -> float | None:
        oldest: float | None = None
        for bucket in self.buckets.values():
            if bucket:
                value = float(bucket[0].enqueue_time)
                oldest = value if oldest is None else min(oldest, value)
        return oldest

    def _oldest_age_ms_locked(self) -> float:
        oldest = self._oldest_enqueue_time_locked()
        if oldest is None:
            return 0.0
        return max(0.0, (time.perf_counter() - oldest) * 1000.0)

    def _priority_from_meta(self, meta: dict[str, object]) -> int:
        if str(meta.get("source_type", "") or "") == "event_fused_group_key":
            return self.PRIORITY_FEBAM_CONFIRMED
        source_level = str(meta.get("source_level", "") or "")
        reason = str(meta.get("ocr_sampling_reason", meta.get("trigger_reason", "")) or "")
        if bool(meta.get("febam_confirmed", False)) or source_level == "febam_confirmed":
            return self.PRIORITY_FEBAM_CONFIRMED
        if bool(meta.get("near_confirmed_large_roi", False)) or bool(meta.get("near_confirmed_ocr_sample", False)) or source_level == "near_confirmed":
            return self.PRIORITY_NEAR_CONFIRMED
        if reason == "periodic_track_sample" or source_level == "periodic_verify":
            return self.PRIORITY_PERIODIC_VERIFY
        if bool(meta.get("weak_plate_sample", False)) or source_level == "weak_plate_sample":
            return self.PRIORITY_WEAK_PLATE_SAMPLE
        return self.PRIORITY_CURRENT_BEST

    def _is_allowed_variant_for_priority(self, priority: int, variant: str, qsize: int) -> bool:
        if variant == "event_fusion_shared":
            return True
        if variant not in {"raw_expanded", "gray_stretched_norm"}:
            return False
        # The official global CCT model is trained/inferred from the raw RGB
        # plate image. Do not replace that input with the contrast-stretched
        # grayscale compatibility variant in dual-branch mode.
        if bool(getattr(self, "dual_branch_ocr", False)):
            mode = str(getattr(self, "dual_branch_global_input_variant", "raw"))
            allowed = {"raw_expanded"} if mode == "raw" else (
                {"gray_stretched_norm"} if mode == "gray" else {"raw_expanded", "gray_stretched_norm"}
            )
            return variant in allowed
        if priority in {self.PRIORITY_CURRENT_BEST, self.PRIORITY_WEAK_PLATE_SAMPLE}:
            return variant == "gray_stretched_norm"
        if variant == "raw_expanded" and qsize >= int(0.70 * self.queue_max):
            return False
        return True

    def _drop_lowest_priority_locked(self) -> bool:
        for priority in sorted(self.PRIORITIES, reverse=True):
            bucket = self.buckets[priority]
            if bucket:
                dropped = bucket.popleft()
                self.seen_keys.discard(dropped.key)
                self.queue_dropped += 1
                self.queue_drop_backpressure += 1
                if priority == self.PRIORITY_WEAK_PLATE_SAMPLE:
                    self.queue_drop_weak += 1
                return True
        return False

    def enqueue(self, crop_bgr: Any, meta: dict[str, object]) -> bool:
        meta = dict(meta or {})
        frame_idx = int(meta.get("frame_idx", -1) if meta.get("frame_idx", -1) is not None else -1)
        track_id = int(meta.get("track_id", -1) if meta.get("track_id", -1) is not None else -1)
        candidate_idx = int(meta.get("candidate_idx", -1) if meta.get("candidate_idx", -1) is not None else -1)
        variant = str(meta.get("variant_name", meta.get("variant", "unknown")) or "unknown")
        source_type = str(meta.get("source_type", "frame_roi") or "frame_roi")
        priority = self._priority_from_meta(meta)
        event_fusion_group_key = str(
            meta.get("event_fusion_group_key", "") or ""
        ).strip()
        if source_type == "event_fused_group_key" and not event_fusion_group_key:
            self.queue_dropped += 1
            self.queue_drop_duplicate += 1
            return False
        restoration_shadow_kind = str(meta.get("restoration_shadow_kind", "") or "")
        key = (
            source_type,
            event_fusion_group_key,
            restoration_shadow_kind,
            track_id,
            frame_idx,
            variant,
        )
        with self.condition:
            qsize = self._qsize_locked()
            if key in self.seen_keys:
                self.queue_dropped += 1
                self.queue_drop_duplicate += 1
                return False
            if not self._is_allowed_variant_for_priority(priority, variant, qsize):
                self.queue_dropped += 1
                self.queue_drop_backpressure += 1
                if priority == self.PRIORITY_WEAK_PLATE_SAMPLE:
                    self.queue_drop_weak += 1
                return False
            track_variant_key = (track_id, variant)
            last_frame = self.last_enqueue_frame_by_track_variant.get(track_variant_key)
            min_gap = int(self.min_gap_by_priority.get(priority, 5))
            allow_priority0_bypass = priority == self.PRIORITY_FEBAM_CONFIRMED and qsize == 0
            if source_type != "event_fused_group_key" and last_frame is not None and frame_idx >= 0 and not allow_priority0_bypass and frame_idx - int(last_frame) < min_gap:
                self.queue_dropped += 1
                self.queue_drop_rate_limited += 1
                return False
            if qsize >= int(0.70 * self.queue_max) and priority == self.PRIORITY_WEAK_PLATE_SAMPLE:
                self.queue_dropped += 1
                self.queue_drop_weak += 1
                self.queue_drop_backpressure += 1
                return False
            if qsize >= int(0.90 * self.queue_max) and priority not in {self.PRIORITY_FEBAM_CONFIRMED, self.PRIORITY_NEAR_CONFIRMED}:
                self.queue_dropped += 1
                self.queue_drop_backpressure += 1
                return False
            if qsize >= self.queue_max and not self._drop_lowest_priority_locked():
                self.queue_dropped += 1
                self.queue_drop_backpressure += 1
                return False
            task = OCRTask(
                priority=priority,
                frame_idx=frame_idx,
                track_id=track_id,
                candidate_idx=candidate_idx,
                variant=variant,
                crop_bgr=(crop_bgr if torch.is_tensor(crop_bgr) else np.ascontiguousarray(crop_bgr)),
                meta=meta,
                enqueue_time=time.perf_counter(),
                key=key,
            )
            self.buckets[priority].append(task)
            self.seen_keys.add(key)
            self.last_enqueue_frame_by_track_variant[track_variant_key] = frame_idx
            self.enqueued += 1
            if self.dual_branch_ocr and variant == "raw_expanded":
                self.dual_branch_raw_enqueued += 1
            elif self.dual_branch_ocr and variant == "gray_stretched_norm":
                self.dual_branch_gray_enqueued += 1
            self.queue_max_seen = max(self.queue_max_seen, self._qsize_locked())
            self.condition.notify()
            return True

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=timeout)

    def stats(self) -> dict[str, float]:
        avg_batch = float(self.batch_size_total) / float(self.batch_call_count) if self.batch_call_count else 0.0
        avg_ms_per_crop = float(self.total_ocr_ms) / float(self.processed) if self.processed else 0.0
        tensor_avg_batch = float(self.fastplate_tensor_batch_size_total) / float(self.fastplate_tensor_batch_call_count) if self.fastplate_tensor_batch_call_count else 0.0
        tensor_avg_ms = float(self.fastplate_tensor_total_ms) / float(self.fastplate_tensor_batch_size_total) if self.fastplate_tensor_batch_size_total else 0.0
        parity_count = float(self.fastplate_parity_sample_count)
        parity_success_count = float(self.fastplate_parity_success_count)
        with self.condition:
            qsize = self._qsize_locked()
            oldest_age_ms = self._oldest_age_ms_locked()
        custom_stats = {}
        if (self.fastplate_custom_onnx or self.dual_branch_ocr) and self.recognizer is not None and hasattr(self.recognizer, "stats"):
            try:
                custom_stats = self.recognizer.stats()
            except Exception:
                custom_stats = {}
        stats = {
            "ocr_session_create_count": float(self.ocr_session_create_count),
            "ocr_worker_create_count": float(self.ocr_worker_create_count),
            "fastplate_async_enqueued": float(self.enqueued),
            "fastplate_async_processed": float(self.processed),
            "fastplate_batch_call_count": float(self.batch_call_count),
            "fastplate_avg_batch_size": avg_batch,
            "fastplate_batch_size": float(self.batch_size),
            "fastplate_min_batch_size": float(self.min_batch_size),
            "fastplate_large_batch_mode": 1.0 if self.large_batch_mode else 0.0,
            "fastplate_target_batch_size": float(self.target_batch_size),
            "fastplate_flush_timeout_ms": float(self.flush_timeout_ms),
            "fastplate_max_flush_timeout_ms": float(self.max_flush_timeout_ms),
            "fastplate_batch_efficiency_ratio": avg_batch / float(max(1, self.batch_size)),
            "fastplate_batch_size_hist": json.dumps(dict(sorted(self.fastplate_batch_size_hist.items())), ensure_ascii=False),
            "fastplate_tensor_batch_size_hist": json.dumps(dict(sorted(self.fastplate_tensor_batch_size_hist.items())), ensure_ascii=False),
            "fastplate_total_ocr_ms": float(self.total_ocr_ms),
            "fastplate_avg_ocr_ms_per_crop": avg_ms_per_crop,
            "fastplate_queue_dropped": float(self.queue_dropped),
            "fastplate_queue_drop_weak": float(self.queue_drop_weak),
            "fastplate_queue_drop_duplicate": float(self.queue_drop_duplicate),
            "fastplate_queue_drop_rate_limited": float(self.queue_drop_rate_limited),
            "fastplate_queue_drop_backpressure": float(self.queue_drop_backpressure),
            "fastplate_queue_max_seen": float(self.queue_max_seen),
            "fastplate_queue_depth": float(qsize),
            "fastplate_oldest_age_ms": float(oldest_age_ms),
            "fastplate_gpu_crop_batch_enabled": 1.0 if self.gpu_crop_batch else 0.0,
            "fastplate_custom_onnx_enabled": 1.0 if (self.fastplate_custom_onnx or self.dual_branch_ocr) else 0.0,
            "fastplate_custom_onnx_path": self.fastplate_custom_onnx or self.dual_branch_global_onnx,
            "dual_branch_enabled": 1.0 if self.dual_branch_ocr else 0.0,
            "dual_branch_raw_enqueued": float(self.dual_branch_raw_enqueued),
            "dual_branch_gray_enqueued": float(self.dual_branch_gray_enqueued),
            "dual_branch_raw_processed": float(self.dual_branch_raw_processed),
            "dual_branch_gray_processed": float(self.dual_branch_gray_processed),
            "dual_branch_global_raw_primary_count": float(
                self.dual_branch_raw_processed if self.dual_branch_global_input_variant in {"raw", "both"} else 0
            ),
            "dual_branch_global_gray_primary_count": float(
                self.dual_branch_gray_processed if self.dual_branch_global_input_variant == "gray" else 0
            ),
            "dual_branch_session_count": 0.0,
            "fastplate_tensor_runner_enabled": 1.0 if self.tensor_runner_mode != "disabled" else 0.0,
            "fastplate_tensor_runner_mode": self.tensor_runner_mode,
            "fastplate_tensor_runner_mode_id": float({"disabled": 0, "shadow": 1, "active": 2}.get(self.tensor_runner_mode, 0)),
            "fastplate_tensor_runner_active_used": float(self.fastplate_tensor_runner_active_used),
            "fastplate_tensor_runner_shadow_count": float(self.fastplate_tensor_runner_shadow_count),
            "fastplate_tensor_direct_backend_available": float(self.fastplate_tensor_direct_backend_available),
            "fastplate_tensor_backend_kind": self.fastplate_tensor_backend_kind,
            "fastplate_tensor_gpu_only_claim_valid": float(self.fastplate_tensor_gpu_only_claim_valid),
            "fastplate_tensor_active_requested": float(self.fastplate_tensor_active_requested),
            "fastplate_tensor_active_success_count": float(self.fastplate_tensor_active_success_count),
            "fastplate_tensor_active_blocked_count": float(self.fastplate_tensor_active_blocked_count),
            "fastplate_tensor_active_fallback_used": float(self.fastplate_tensor_active_fallback_used),
            "fastplate_tensor_active_fallback_reason": self.fastplate_tensor_active_fallback_reason,
            "fastplate_direct_ort_enabled": 1.0 if self.direct_ort_enabled else 0.0,
            "fastplate_direct_ort_iobinding": 1.0 if self.direct_ort_iobinding else 0.0,
            "fastplate_direct_ort_compare_wrapper": 1.0 if self.direct_ort_compare_wrapper else 0.0,
            "fastplate_tensor_direct_backend_init_error": self.fastplate_tensor_direct_backend_init_error,
            "fastplate_tensor_direct_model_path_found": float(self.fastplate_tensor_direct_model_path_found),
            "fastplate_tensor_direct_session_created": float(self.fastplate_tensor_direct_session_created),
            "fastplate_tensor_direct_decode_ready": float(self.fastplate_tensor_direct_decode_ready),
            "fastplate_tensor_model_path": self.fastplate_tensor_model_path,
            "fastplate_tensor_input_layout": self.fastplate_tensor_input_layout,
            "fastplate_tensor_output_shapes": self.fastplate_tensor_output_shapes,
            "fastplate_gpu_crop_count": float(self.fastplate_gpu_crop_count),
            "fastplate_cpu_crop_count": float(self.fastplate_cpu_crop_count),
            "fastplate_gpu_crop_ms": float(self.fastplate_gpu_crop_ms),
            "fastplate_cpu_crop_ms": float(self.fastplate_cpu_crop_ms),
            "fastplate_crop_cpu_copy_count": float(self.fastplate_crop_cpu_copy_count),
            "fastplate_crop_gpu_tensor_count": float(self.fastplate_crop_gpu_tensor_count),
            "fastplate_tensor_batch_call_count": float(self.fastplate_tensor_batch_call_count),
            "fastplate_tensor_avg_batch_size": tensor_avg_batch,
            "fastplate_tensor_total_ms": float(self.fastplate_tensor_total_ms),
            "fastplate_tensor_avg_ms_per_crop": tensor_avg_ms,
            "fastplate_tensor_preprocess_ms": float(self.fastplate_tensor_preprocess_ms),
            "fastplate_tensor_infer_ms": float(self.fastplate_tensor_infer_ms),
            "fastplate_tensor_decode_ms": float(self.fastplate_tensor_decode_ms),
            "fastplate_tensor_iobinding_used": float(self.fastplate_tensor_iobinding_used),
            "fastplate_tensor_cpu_fallback_count": float(self.fastplate_tensor_cpu_fallback_count),
            "fastplate_tensor_error_count": float(self.fastplate_tensor_error_count),
            "fastplate_parity_sample_count": parity_count,
            "fastplate_parity_success_count": parity_success_count,
            "fastplate_parity_error_count": float(self.fastplate_parity_error_count),
            "fastplate_parity_skipped_count": float(self.fastplate_parity_skipped_count),
            "fastplate_parity_text_match_count": float(self.fastplate_parity_text_match_count),
            "fastplate_parity_text_match_rate": float(self.fastplate_parity_text_match_count) / max(1.0, parity_success_count),
            "fastplate_parity_normalized_match_rate": float(self.fastplate_parity_normalized_match_count) / max(1.0, parity_success_count),
            "fastplate_parity_korean_slot_match_rate": float(self.fastplate_parity_korean_slot_match_count) / max(1.0, parity_success_count),
            "fastplate_parity_conf_abs_diff_mean": float(self.fastplate_parity_conf_abs_diff_sum) / max(1.0, parity_success_count),
            "fastplate_parity_effective_match_rate": float(self.fastplate_parity_text_match_count) / max(1.0, parity_success_count),
            "fastplate_tensor_last_error_type": self.fastplate_tensor_last_error_type,
            "fastplate_tensor_last_error_message": self.fastplate_tensor_last_error_message,
            "fastplate_tensor_last_error_stage": self.fastplate_tensor_last_error_stage,
            "fastplate_tensor_error_sample_input_shape": self.fastplate_tensor_error_sample_input_shape,
            "fastplate_tensor_error_sample_input_dtype": self.fastplate_tensor_error_sample_input_dtype,
            "fastplate_tensor_error_sample_output_shapes": self.fastplate_tensor_error_sample_output_shapes,
            "fastplate_tensor_error_sample_batch_len": float(self.fastplate_tensor_error_sample_batch_len),
            "fastplate_tensor_provider": self.fastplate_tensor_provider,
            "fastplate_tensor_input_name": self.fastplate_tensor_input_name,
            "fastplate_tensor_output_names": self.fastplate_tensor_output_names,
            "fastplate_tensor_preprocess_mode": self.fastplate_tensor_preprocess_mode,
            "fastplate_tensor_decoder_mode": self.fastplate_tensor_decoder_mode,
            "fastplate_tensor_last_input_shape": self.fastplate_tensor_last_input_shape,
            "fastplate_tensor_last_input_dtype": self.fastplate_tensor_last_input_dtype,
            "fastplate_tensor_prepare_valid_count": float(self.fastplate_tensor_prepare_valid_count),
            "fastplate_tensor_prepare_failed_count": float(self.fastplate_tensor_prepare_failed_count),
            "fastplate_tensor_prepare_failed_examples": self.fastplate_tensor_prepare_failed_examples,
            "fastplate_tensor_preprocess_resize_count": float(self.fastplate_tensor_preprocess_resize_count),
            "fastplate_custom_provider": "",
            "fastplate_custom_cuda_used": 0.0,
            "fastplate_custom_input_name": "",
            "fastplate_custom_output_names": "",
            "fastplate_custom_input_shape": "",
            "fastplate_custom_output_shapes": "",
            "fastplate_custom_input_layout": "",
            "fastplate_custom_alphabet_len": 0.0,
            "fastplate_custom_batch_call_count": 0.0,
            "fastplate_custom_item_count": 0.0,
            "fastplate_custom_avg_batch_size": 0.0,
            "fastplate_custom_total_ms": 0.0,
            "fastplate_custom_avg_ms_per_crop": 0.0,
            "fastplate_custom_preprocess_ms": 0.0,
            "fastplate_custom_infer_ms": 0.0,
            "fastplate_custom_decode_ms": 0.0,
            "fastplate_custom_error_count": 0.0,
        }
        stats.update(custom_stats)
        return stats

    def _get_recognizer(self):
        if self.recognizer is None:
            if self.dual_branch_ocr:
                # Keep the recovery implementation outside the pipeline.  The
                # subclass preserves the normal dual-branch contract and only
                # emits low-weight FEBAM_REQUIRED evidence for collapsed OCR.
                self.recognizer = EnglishMixedRecoveryDecoder(
                    global_onnx=self.dual_branch_global_onnx,
                    korean_onnx=self.dual_branch_korean_onnx,
                    device=self.device,
                    hangul_weight=self.dual_branch_hangul_weight,
                    digit_alignment_mode=self.dual_branch_digit_alignment_mode,
                    digit_mass_thr=self.dual_branch_digit_mass_thr,
                    digit_conf_thr=self.dual_branch_digit_conf_thr,
                    digit_margin_thr=self.dual_branch_digit_margin_thr,
                    global_crop_mode=self.dual_branch_global_crop_mode,
                    global_result_mode=self.dual_branch_global_result_mode,
                    global_fastplate_model="cct-xs-v2-global-model",
                    korean_mode=self.dual_branch_korean_mode,
                    korean_slot_mode=self.dual_branch_korean_slot_mode,
                    korean_mass_thr=self.dual_branch_korean_mass_thr,
                    korean_conf_thr=self.dual_branch_korean_conf_thr,
                    korean_margin_thr=self.dual_branch_korean_margin_thr,
                    length_margin_thr=self.dual_branch_length_margin_thr,
                    length_hold=self.dual_branch_length_hold,
                    logger=self.logger,
                )
                self._log(
                    f"[OCR] Dual-branch async worker initialized global={self.dual_branch_global_onnx} "
                    f"korean={self.dual_branch_korean_onnx} device={self.device} batch_size={self.batch_size}"
                )
            elif self.fastplate_custom_onnx:
                self.recognizer = CustomFastPlateONNXRecognizer(
                    onnx_path=self.fastplate_custom_onnx,
                    plate_config=self.fastplate_custom_plate_config,
                    input_width=self.fastplate_custom_input_width,
                    input_height=self.fastplate_custom_input_height,
                    device=self.device,
                    logger=self.logger,
                )
                self._log(f"[OCR] Custom FastPlateONNX async worker initialized onnx={self.fastplate_custom_onnx} device={self.device} batch_size={self.batch_size}")
            else:
                from ocr.ocr_fastplate_batch import FastPlateOCRBatchRecognizer

                self.recognizer = FastPlateOCRBatchRecognizer(
                    model=self.model,
                    device=self.device,
                    batch_size=self.batch_size,
                    preload_torch_cuda_dlls=self.preload_torch_cuda_dlls,
                    min_text_len=self.min_text_len,
                )
                self._log(f"[OCR] FastPlateOCR async worker initialized model={self.model} device={self.device} batch_size={self.batch_size}")
            self.ocr_session_create_count += int(getattr(self.recognizer, "session_count", 1))
        return self.recognizer

    def _get_tensor_runner(self):
        if self.tensor_runner_mode == "disabled":
            return None
        if self.tensor_runner is None:
            from ocr.fastplate_tensor_runner import FastPlateTensorRunner

            direct_input_hw = None
            if self.direct_ort_input_h > 0 and self.direct_ort_input_w > 0:
                direct_input_hw = (self.direct_ort_input_h, self.direct_ort_input_w)

            self.tensor_runner = FastPlateTensorRunner(
                model=self.model,
                device=self.device,
                input_h=self.tensor_input_h,
                input_w=self.tensor_input_w,
                batch_size=self.batch_size,
                min_text_len=self.min_text_len,
                preload_torch_cuda_dlls=self.preload_torch_cuda_dlls,
                enable_direct_ort=self.direct_ort_enabled,
                enable_iobinding=self.direct_ort_iobinding,
                model_path_override=self.direct_ort_model_path,
                input_hw=direct_input_hw,
                debug=self.direct_ort_debug,
                compare_wrapper=self.direct_ort_compare_wrapper,
                preserve_aspect_ratio=self.preserve_aspect_ratio,
                logger=self.logger,
            )
            self._log(
                f"[OCR] FastPlate tensor runner initialized mode={self.tensor_runner_mode} "
                f"direct_ort={int(self.direct_ort_enabled)} "
                f"iobinding={int(self.direct_ort_iobinding)} "
                f"input={self.tensor_input_h}x{self.tensor_input_w} batch_size={self.batch_size}"
            )
        return self.tensor_runner

    def _update_tensor_runner_profile_from_stats(self, runner_stats: dict[str, object]) -> None:
        self.fastplate_direct_ort_enabled = int(float(runner_stats.get("fastplate_direct_ort_enabled", 0.0) or 0.0) > 0.0)
        self.fastplate_direct_ort_iobinding = int(self.tensor_iobinding)
        self.fastplate_direct_ort_compare_wrapper = int(self.tensor_compare_wrapper)
        self.fastplate_tensor_preprocess_ms = float(runner_stats.get("fastplate_tensor_preprocess_ms", self.fastplate_tensor_preprocess_ms) or 0.0)
        self.fastplate_tensor_infer_ms = float(runner_stats.get("fastplate_tensor_infer_ms", self.fastplate_tensor_infer_ms) or 0.0)
        self.fastplate_tensor_decode_ms = float(runner_stats.get("fastplate_tensor_decode_ms", self.fastplate_tensor_decode_ms) or 0.0)
        self.fastplate_tensor_iobinding_used = int(float(runner_stats.get("fastplate_tensor_iobinding_used", 0.0) or 0.0) > 0.0)
        self.fastplate_tensor_provider = str(runner_stats.get("fastplate_tensor_provider", runner_stats.get("fastplate_tensor_onnx_provider", "")) or "")
        self.fastplate_tensor_direct_backend_available = int(float(runner_stats.get("fastplate_tensor_direct_backend_available", 0.0) or 0.0) > 0.0)
        self.fastplate_tensor_backend_kind = str(runner_stats.get("fastplate_tensor_backend_kind", "") or "")
        self.fastplate_tensor_gpu_only_claim_valid = int(float(runner_stats.get("fastplate_tensor_gpu_only_claim_valid", 0.0) or 0.0) > 0.0)
        self.fastplate_tensor_input_name = str(runner_stats.get("fastplate_tensor_input_name", "") or "")
        self.fastplate_tensor_output_names = str(runner_stats.get("fastplate_tensor_output_names", "") or "")
        self.fastplate_tensor_last_error_type = str(runner_stats.get("fastplate_tensor_last_error_type", "") or "")
        self.fastplate_tensor_last_error_message = str(runner_stats.get("fastplate_tensor_last_error_message", "") or "")
        self.fastplate_tensor_last_error_stage = str(runner_stats.get("fastplate_tensor_last_error_stage", "") or "")
        self.fastplate_tensor_error_sample_input_shape = str(runner_stats.get("fastplate_tensor_error_sample_input_shape", "") or "")
        self.fastplate_tensor_error_sample_input_dtype = str(runner_stats.get("fastplate_tensor_error_sample_input_dtype", "") or "")
        self.fastplate_tensor_error_sample_output_shapes = str(runner_stats.get("fastplate_tensor_error_sample_output_shapes", "") or "")
        self.fastplate_tensor_direct_backend_init_error = str(runner_stats.get("fastplate_tensor_direct_backend_init_error", "") or "")
        self.fastplate_tensor_direct_model_path_found = float(runner_stats.get("fastplate_tensor_direct_model_path_found", 0.0) or 0.0)
        self.fastplate_tensor_direct_session_created = float(runner_stats.get("fastplate_tensor_direct_session_created", 0.0) or 0.0)
        self.fastplate_tensor_direct_decode_ready = float(runner_stats.get("fastplate_tensor_direct_decode_ready", runner_stats.get("fastplate_tensor_decode_ready", 0.0)) or 0.0)
        self.fastplate_tensor_model_path = str(runner_stats.get("fastplate_tensor_model_path", "") or "")
        self.fastplate_tensor_input_layout = str(runner_stats.get("fastplate_tensor_input_layout", "") or "")
        self.fastplate_tensor_output_shapes = str(runner_stats.get("fastplate_tensor_output_shapes", self.fastplate_tensor_error_sample_output_shapes) or "")
        self.fastplate_tensor_error_sample_batch_len = float(runner_stats.get("fastplate_tensor_error_sample_batch_len", 0.0) or 0.0)
        self.fastplate_tensor_preprocess_mode = str(runner_stats.get("fastplate_tensor_preprocess_mode", runner_stats.get("fastplate_tensor_preprocessing_mode", "")) or "")
        self.fastplate_tensor_decoder_mode = str(runner_stats.get("fastplate_tensor_decoder_mode", "") or "")
        self.fastplate_tensor_last_input_shape = str(runner_stats.get("fastplate_tensor_last_input_shape", "") or "")
        self.fastplate_tensor_last_input_dtype = str(runner_stats.get("fastplate_tensor_last_input_dtype", "") or "")
        self.fastplate_tensor_prepare_valid_count = float(runner_stats.get("fastplate_tensor_prepare_valid_count", 0.0) or 0.0)
        self.fastplate_tensor_prepare_failed_count = float(runner_stats.get("fastplate_tensor_prepare_failed_count", 0.0) or 0.0)
        self.fastplate_tensor_prepare_failed_examples = str(runner_stats.get("fastplate_tensor_prepare_failed_examples", "") or "")
        self.fastplate_tensor_preprocess_resize_count = float(runner_stats.get("fastplate_tensor_preprocess_resize_count", 0.0) or 0.0)

    def _make_fastplate_active_blocked_results(self, batch: list[OCRTask], reason: str) -> list[dict[str, object]]:
        return [
            {
                "text": "",
                "conf": 0.0,
                "source": "fastplate_tensor_active_blocked",
                "variant": batch[idx].variant,
                "fallback_used": False,
                "raw_result": "",
                "error": "FastPlateTensorActiveBlocked",
                "error_message": reason,
                "ocr_skip_reason": f"fastplate_tensor_active_blocked_{reason}",
                "fastplate_tensor_active_blocked": 1,
                "fastplate_tensor_fallback_used": 0,
                "fastplate_tensor_fallback_reason": reason,
            }
            for idx in range(len(batch))
        ]

    @staticmethod
    def _normalize_parity_text(value: object) -> str:
        text = re.sub(r"[^0-9A-Za-z가-힣]", "", str(value or "")).upper()
        return text

    @staticmethod
    def _korean_slot(value: object) -> str:
        match = re.search(r"[가-힣]", str(value or ""))
        return match.group(0) if match else ""

    @staticmethod
    def _digit_skeleton(value: object) -> str:
        text = str(value or "")
        digits = re.sub(r"\D", "", text)
        if len(digits) == 7:
            return f"{digits[:2]}?{digits[-4:]}"
        if len(digits) >= 8:
            return f"{digits[:3]}?{digits[-4:]}"
        return ""

    def _append_parity_rows(self, batch: list[OCRTask], old_results: list[dict[str, Any]], new_results: list[dict[str, Any]], error: Exception | None = None) -> None:
        if not self.parity_log_csv or self.fastplate_parity_sample_count >= self.parity_sample_limit:
            return
        path = Path(self.parity_log_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "fastplate_parity_enabled", "fastplate_parity_mode", "fastplate_parity_crop_id",
            "fastplate_parity_frame_idx", "fastplate_parity_track_id", "fastplate_parity_candidate_idx",
            "fastplate_parity_variant_name", "old_fastplate_text", "old_fastplate_conf",
            "old_fastplate_raw_result", "old_fastplate_source", "new_tensor_text", "new_tensor_conf",
            "new_tensor_raw_result", "new_tensor_source", "fastplate_parity_text_match",
            "fastplate_parity_normalized_match", "fastplate_parity_conf_abs_diff",
            "fastplate_parity_korean_slot_match", "fastplate_parity_digit_skeleton_match",
            "fastplate_parity_error_type", "fastplate_parity_error_message",
        ]
        write_header = not path.exists()
        with path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for idx, task in enumerate(batch):
                if self.fastplate_parity_sample_count >= self.parity_sample_limit:
                    break
                old = dict(old_results[idx] if idx < len(old_results) else {})
                new = dict(new_results[idx] if idx < len(new_results) else {})
                new_error = str(new.get("error", "") or new.get("ocr_skip_reason", "") or "")
                has_error = bool(error is not None or new_error)
                is_success = bool(new) and not has_error and "text" in new and "conf" in new
                old_text = str(old.get("text", "") or "")
                new_text = str(new.get("text", "") or "")
                old_norm = self._normalize_parity_text(old_text)
                new_norm = self._normalize_parity_text(new_text)
                old_conf = float(old.get("conf", 0.0) or 0.0)
                new_conf = float(new.get("conf", 0.0) or 0.0)
                text_match = int(old_text == new_text)
                norm_match = int(old_norm == new_norm)
                korean_match = int(self._korean_slot(old_text) == self._korean_slot(new_text))
                skeleton_match = int(self._digit_skeleton(old_text) == self._digit_skeleton(new_text))
                conf_diff = abs(old_conf - new_conf)
                self.fastplate_parity_sample_count += 1
                if is_success:
                    self.fastplate_parity_success_count += 1
                    self.fastplate_parity_text_match_count += text_match
                    self.fastplate_parity_normalized_match_count += norm_match
                    self.fastplate_parity_korean_slot_match_count += korean_match
                    self.fastplate_parity_conf_abs_diff_sum += conf_diff
                elif has_error:
                    self.fastplate_parity_error_count += 1
                else:
                    self.fastplate_parity_skipped_count += 1
                writer.writerow({
                    "fastplate_parity_enabled": 1,
                    "fastplate_parity_mode": self.tensor_runner_mode,
                    "fastplate_parity_crop_id": task.key,
                    "fastplate_parity_frame_idx": task.frame_idx,
                    "fastplate_parity_track_id": task.track_id,
                    "fastplate_parity_candidate_idx": task.candidate_idx,
                    "fastplate_parity_variant_name": task.variant,
                    "old_fastplate_text": old_text,
                    "old_fastplate_conf": old_conf,
                    "old_fastplate_raw_result": str(old.get("raw_result", "") or "")[:500],
                    "old_fastplate_source": str(old.get("source", "") or ""),
                    "new_tensor_text": new_text,
                    "new_tensor_conf": new_conf,
                    "new_tensor_raw_result": str(new.get("raw_result", "") or "")[:500],
                    "new_tensor_source": str(new.get("source", "") or ""),
                    "fastplate_parity_text_match": text_match,
                    "fastplate_parity_normalized_match": norm_match,
                    "fastplate_parity_conf_abs_diff": conf_diff,
                    "fastplate_parity_korean_slot_match": korean_match,
                    "fastplate_parity_digit_skeleton_match": skeleton_match,
                    "fastplate_parity_error_type": type(error).__name__ if error else str(new.get("error", "") or ""),
                    "fastplate_parity_error_message": str(error or new.get("error_message", "") or "")[:500],
                })

    def _take_batch_locked(self, force_all: bool = False) -> list[OCRTask]:
        limit = self.batch_size if not force_all else self._qsize_locked()
        # The Final WiSE IOBinding buffers are fixed B1/B2/B4.  Do not create
        # an accidental B3 at a timeout or final drain; drain it as B2 then B1.
        available = self._qsize_locked()
        if (self.fastplate_custom_onnx or self.dual_branch_ocr) and min(limit, available) == 3:
            limit = 2
        batch: list[OCRTask] = []
        while len(batch) < limit:
            selected: OCRTask | None = None
            for priority in self.PRIORITIES:
                if self.buckets[priority]:
                    selected = self.buckets[priority].popleft()
                    break
            if selected is None:
                break
            self.seen_keys.discard(selected.key)
            batch.append(selected)
        return batch

    def _flush_batch(self, batch: list[OCRTask]) -> None:
        if not batch:
            return
        self.fastplate_batch_size_hist[len(batch)] += 1
        crops = [task.crop_bgr for task in batch]
        metas = [dict(task.meta or {}) for task in batch]
        variant_names = [task.variant for task in batch]
        start = time.perf_counter()
        try:
            gpu_custom_batch = bool(
                (self.fastplate_custom_onnx or self.dual_branch_ocr)
                and all(torch.is_tensor(crop) and crop.is_cuda for crop in crops)
            )
            if gpu_custom_batch:
                results = self._get_recognizer().recognize_tensor_batch(crops, metas=metas, variant_names=variant_names)
                self.fastplate_gpu_crop_count += len(crops)
                self.fastplate_crop_gpu_tensor_count += len(crops)
                self.fastplate_tensor_batch_call_count += 1
                self.fastplate_tensor_batch_size_total += len(crops)
                self.fastplate_tensor_batch_size_hist[len(batch)] += 1
                self.fastplate_tensor_iobinding_used += 1
            elif self.tensor_runner_mode == "active":
                self.fastplate_tensor_active_requested = 1
                runner = self._get_tensor_runner()
                runner_stats = runner.stats() if runner is not None else {}
                if runner_stats:
                    self._update_tensor_runner_profile_from_stats(runner_stats)
                backend_kind = str(runner_stats.get("fastplate_tensor_backend_kind", "") or self.fastplate_tensor_backend_kind)
                direct_available = bool(float(runner_stats.get("fastplate_tensor_direct_backend_available", 0) or 0))
                direct_ok = bool(runner is not None and backend_kind == "direct_onnxruntime_cuda" and direct_available)
                if not direct_ok:
                    reason = "fallback_fastplate_wrapper_not_gpu_only" if backend_kind == "fallback_fastplate_wrapper" else "direct_onnxruntime_cuda_not_available"
                    self.fastplate_tensor_active_blocked_count += len(batch)
                    self.fastplate_tensor_active_fallback_used = 0
                    self.fastplate_tensor_active_fallback_reason = reason
                    if self.fastplate_tensor_active_blocked_log_count < 5:
                        self.fastplate_tensor_active_blocked_log_count += 1
                        msg = (
                            f"[FASTPLATE_ACTIVE_BLOCKED] backend_kind={self.fastplate_tensor_backend_kind or 'unknown'} "
                            f"gpu_only_claim_valid={int(self.fastplate_tensor_gpu_only_claim_valid)}"
                        )
                        print(msg, flush=True)
                        self._log(msg)
                    results = self._make_fastplate_active_blocked_results(batch, reason)
                else:
                    tensor_start = time.perf_counter()
                    if all(torch.is_tensor(crop) and crop.is_cuda for crop in crops):
                        new_results = runner.recognize_cuda_crop_list(crops, metas=metas, variant_names=variant_names)
                        self.fastplate_gpu_crop_count += len(crops)
                        self.fastplate_crop_gpu_tensor_count += len(crops)
                    else:
                        new_results = self._make_fastplate_active_blocked_results(batch, "non_cuda_crop_in_gpu_only_active")
                    tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
                    self.fastplate_tensor_batch_call_count += 1
                    self.fastplate_tensor_batch_size_total += len(crops)
                    self.fastplate_tensor_total_ms += tensor_ms
                    self._update_tensor_runner_profile_from_stats(runner.stats())
                    result_error_count = sum(1 for row in new_results if str(row.get("error", "") or ""))
                    self.fastplate_tensor_error_count += result_error_count
                    valid_active = (
                        result_error_count == 0
                        and len(new_results) == len(batch)
                        and all("text" in row and "conf" in row and row.get("fastplate_tensor_backend_kind") == "direct_onnxruntime_cuda" for row in new_results)
                    )
                    if valid_active:
                        self.fastplate_tensor_batch_size_hist[len(batch)] += 1
                        self.fastplate_tensor_runner_active_used += len(new_results)
                        self.fastplate_tensor_active_success_count += len(new_results)
                        accept_msg = f"[FASTPLATE_ACTIVE_DIRECT] accepted batch={len(new_results)} backend=direct_onnxruntime_cuda iobinding={int(self.fastplate_tensor_iobinding_used)}"
                        print(accept_msg, flush=True)
                        self._log(accept_msg)
                        results = new_results
                    else:
                        reason = "active_guard_failed"
                        self.fastplate_tensor_active_blocked_count += len(batch)
                        self.fastplate_tensor_active_fallback_used = 0
                        self.fastplate_tensor_active_fallback_reason = reason
                        results = self._make_fastplate_active_blocked_results(batch, reason)
            else:
                self.fastplate_cpu_crop_count += len(crops)
                self.fastplate_crop_cpu_copy_count += len(crops)
                old_results = self._get_recognizer().recognize_batch(crops, metas=metas, variant_names=variant_names)
                results = old_results
                if self.tensor_runner_mode == "shadow":
                    tensor_start = time.perf_counter()
                    try:
                        runner = self._get_tensor_runner()
                        if runner is not None:
                            new_results = runner.recognize_crop_batch(crops, metas=metas, variant_names=variant_names)
                            tensor_ms = (time.perf_counter() - tensor_start) * 1000.0
                            self.fastplate_tensor_batch_call_count += 1
                            self.fastplate_tensor_batch_size_total += len(crops)
                            self.fastplate_tensor_batch_size_hist[len(batch)] += 1
                            self.fastplate_tensor_total_ms += tensor_ms
                            runner_stats = runner.stats()
                            self._update_tensor_runner_profile_from_stats(runner_stats)
                            self.fastplate_tensor_cpu_fallback_count += sum(1 for row in new_results if bool(row.get("fastplate_tensor_cpu_fallback_used", 0)))
                            result_error_count = sum(1 for row in new_results if str(row.get("error", "") or ""))
                            self.fastplate_tensor_error_count += result_error_count
                            if result_error_count and self.fastplate_tensor_error_log_count < 3:
                                self.fastplate_tensor_error_log_count += 1
                                self._log(
                                    "[FASTPLATE_TENSOR_ERROR]\n"
                                    f"stage={self.fastplate_tensor_last_error_stage}\n"
                                    f"type={self.fastplate_tensor_last_error_type}\n"
                                    f"message={self.fastplate_tensor_last_error_message}\n"
                                    f"input_shape={self.fastplate_tensor_error_sample_input_shape}\n"
                                    f"input_dtype={self.fastplate_tensor_error_sample_input_dtype}\n"
                                    f"provider={self.fastplate_tensor_provider}\n"
                                    f"batch_len={self.fastplate_tensor_error_sample_batch_len}\n"
                                    f"input_name={self.fastplate_tensor_input_name}\n"
                                    f"output_names={self.fastplate_tensor_output_names}\n"
                                    f"output_shapes={self.fastplate_tensor_error_sample_output_shapes}"
                                )
                            self.fastplate_tensor_runner_shadow_count += len(new_results)
                            self._append_parity_rows(batch, old_results, new_results)
                    except Exception as tensor_exc:
                        self.fastplate_tensor_error_count += 1
                        self.fastplate_tensor_last_error_type = type(tensor_exc).__name__
                        self.fastplate_tensor_last_error_message = str(tensor_exc)[:500]
                        self.fastplate_tensor_last_error_stage = "worker_tensor_runner"
                        self.fastplate_tensor_error_sample_batch_len = float(len(crops))
                        if self.fastplate_tensor_error_log_count < 3:
                            self.fastplate_tensor_error_log_count += 1
                            self._log(
                                "[FASTPLATE_TENSOR_ERROR]\n"
                                f"stage={self.fastplate_tensor_last_error_stage}\n"
                                f"type={self.fastplate_tensor_last_error_type}\n"
                                f"message={self.fastplate_tensor_last_error_message}\n"
                                f"input_shape={self.fastplate_tensor_error_sample_input_shape}\n"
                                f"input_dtype={self.fastplate_tensor_error_sample_input_dtype}\n"
                                f"provider={self.fastplate_tensor_provider}\n"
                                f"batch_len={self.fastplate_tensor_error_sample_batch_len}\n"
                                f"input_name={self.fastplate_tensor_input_name}\n"
                                f"output_names={self.fastplate_tensor_output_names}\n"
                                f"output_shapes={self.fastplate_tensor_error_sample_output_shapes}"
                            )
                        self._append_parity_rows(batch, old_results, [], error=tensor_exc)
            error = ""
        except Exception as exc:
            self._log(f"[FASTPLATE_BATCH_ERROR] batch={len(batch)} type={type(exc).__name__} message={str(exc)[:500]}")
            results = [
                {
                    "text": "",
                    "conf": 0.0,
                    "source": "fast_plate_ocr_exception",
                    "variant": variant_names[idx],
                    "fallback_used": False,
                    "raw_result": "",
                    "error": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "ocr_skip_reason": f"fastplate_exception_{type(exc).__name__}",
                }
                for idx in range(len(batch))
            ]
            error = type(exc).__name__
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.batch_call_count += 1
        self.batch_size_total += len(batch)
        self.total_ocr_ms += elapsed_ms
        self.processed += len(batch)
        if self.dual_branch_ocr:
            self.dual_branch_raw_processed += sum(task.variant == "raw_expanded" for task in batch)
            self.dual_branch_gray_processed += sum(task.variant == "gray_stretched_norm" for task in batch)
        now = time.perf_counter()
        for idx, result in enumerate(results):
            result_dict = dict(result or {})
            meta = metas[idx]
            meta["raw_result"] = str(result_dict.get("raw_result", "") or "")[:500]
            for key in (
                "evidence_text",
                "evidence_conf",
                "english_mixed_recovery",
                "english_mixed_recovery_reason",
                "english_mixed_latin_slots",
                "recovery_candidate_text",
                "recovery_source_weight",
                "recovery_requires_string_febam",
                "dual_branch_digit_conf",
                "dual_branch_hangul_conf",
                "dual_branch_decision_state",
                "dual_branch_length_margin",
                "dual_branch_evidence_hypotheses",
                "length_top1_text", "length_top1_log_score", "length_top1_length",
                "length_top2_text", "length_top2_log_score", "length_top2_length",
                "length_score_margin", "length_decision_state", "length_score_contract",
                "posterior_serialization",
                "logical_batch_size", "physical_batch_size", "padding_count",
                "padding_policy", "valid_output_count",
                "padded_output_discarded_count", "discarded_padding_rows",
            ):
                if key in result_dict:
                    meta[key] = result_dict[key]
            result_error = str(result_dict.get("error", "") or error or "")
            if result_error:
                meta["fastplate_error"] = result_error
                meta["fastplate_error_message"] = str(result_dict.get("error_message", "") or "")[:500]
                meta["ocr_skip_reason"] = str(result_dict.get("ocr_skip_reason", f"fastplate_exception_{result_error}"))
            meta["worker_error"] = error
            meta["ocr_batch_size"] = len(batch)
            meta["ocr_batch_ms"] = elapsed_ms
            self.output_queue.put(OCRResultItem(
                frame_idx=batch[idx].frame_idx,
                track_id=batch[idx].track_id,
                candidate_idx=batch[idx].candidate_idx,
                variant=batch[idx].variant,
                text=str(result_dict.get("text", "") or ""),
                conf=float(result_dict.get("conf", 0.0) or 0.0),
                source=str(result_dict.get("source", "fast_plate_ocr") or "fast_plate_ocr"),
                meta=meta,
                delay_ms=(now - batch[idx].enqueue_time) * 1000.0,
                crop_bgr=batch[idx].crop_bgr,
            ))

    def _should_flush_locked(self) -> bool:
        qsize = self._qsize_locked()
        if qsize <= 0:
            return False
        if qsize >= self.batch_size:
            return True
        oldest = self._oldest_enqueue_time_locked()
        if oldest is None:
            return False
        age_ms = (time.perf_counter() - oldest) * 1000.0
        if self.large_batch_mode:
            if qsize >= self.target_batch_size:
                return True
            if qsize >= self.min_batch_size and age_ms >= self.flush_timeout_ms:
                return True
            if age_ms >= self.max_flush_timeout_ms:
                return True
            return False
        return age_ms >= self.flush_timeout_ms and qsize >= min(self.min_batch_size, qsize)

    def _run(self) -> None:
        while True:
            with self.condition:
                while not self.stop_event.is_set() and not self._should_flush_locked():
                    oldest = self._oldest_enqueue_time_locked()
                    if oldest is None:
                        wait = (self.max_flush_timeout_ms if self.large_batch_mode else self.flush_timeout_ms) / 1000.0
                    else:
                        if self.large_batch_mode and self._qsize_locked() >= self.target_batch_size:
                            timeout_ms = self.flush_timeout_ms
                        else:
                            timeout_ms = self.max_flush_timeout_ms if self.large_batch_mode else self.flush_timeout_ms
                        wait = max(0.001, timeout_ms / 1000.0 - (time.perf_counter() - oldest))
                    self.condition.wait(timeout=wait)
                if self.stop_event.is_set() and self._qsize_locked() == 0:
                    break
                batch = self._take_batch_locked(force_all=self.stop_event.is_set())
            self._flush_batch(batch)

class GPUPipeline:
    @staticmethod
    def ext30_route_by_observed_n(observations: list[object]):
        """Opt-in capability router; it does not wait for or synthesize frames."""
        return route_by_observed_n(observations)

    @staticmethod
    def ext30_sparse_pair_evidence(**kwargs) -> PairEvidence:
        """Build immutable pair metadata without changing the hard OCR candidate."""
        return build_pair_evidence(**kwargs)

    @staticmethod
    def ext30_sparse_pair_prediction(pair: PairEvidence) -> str:
        """Recover the original S0 candidate; pair conflict remains HOLD metadata."""
        return select_s4(pair)

    @staticmethod
    def ext30_support_constrained_f4(**kwargs) -> SupportConstrainedFusionResult:
        """Build an opt-in R1C shadow candidate; production routing is unchanged."""
        return apply_support_constrained_f4(**kwargs)

    @staticmethod
    def ext30_r1d_shadow_route(*, baseline_prediction: str, observations: list[str],
                               frame_ids: list[str], f4_prediction: str | None = None,
                               event_id: str = "") -> dict:
        """Fail-isolated R1D shadow router; never mutates the production candidate."""
        baseline = str(baseline_prediction)
        try:
            route = route_by_observed_n(observations)
            if len(observations) == 1:
                shadow, reason = baseline, "SINGLE_VIEW_BASELINE"
            elif len(observations) == 2:
                pair = build_pair_evidence(
                    event_id=event_id, observation_ids=frame_ids,
                    predictions=observations, posterior_refs=frame_ids,
                    provenance="EXT30_R1D_FROZEN_H2",
                )
                shadow, reason = select_s4(pair), pair.fallback_reason
            else:
                if f4_prediction is None:
                    raise ValueError("R1D_F4_EVIDENCE_REQUIRED")
                result = apply_support_constrained_f4(
                    event_id=event_id, baseline_prediction=baseline,
                    f4_prediction=str(f4_prediction),
                    observed_predictions=observations, frame_ids=frame_ids,
                    provenance="EXT30_R1D_FROZEN_H2",
                )
                shadow, reason = result.constrained_prediction, "SUPPORT_CONSTRAINED_F4"
            return {"baseline_prediction": baseline, "shadow_prediction": shadow,
                    "route": str(route.value), "status": "PASS", "reason": reason}
        except Exception as exc:
            return {"baseline_prediction": baseline, "shadow_prediction": baseline,
                    "route": "SHADOW_ERROR", "status": "FAIL_ISOLATED",
                    "reason": f"{type(exc).__name__}:{exc}"}

    def __init__(
        self,
        debug_stage_log: bool = False,
        debug_log_txt: str | None = None,
        debug_save_video: str | None = None,
        debug_save_frames_dir: str | None = None,
        debug_event_dir: str | None = None,
        debug_dump_seconds: set[int] | None = None,
        debug_save_every_sec: float = 1.0,
        debug_max_events: int = 200,
        fps: float = 25.0,
        scientific_g0: bool = False,
        ext30_modular_evidence: bool = False,
        ext30_external_anchor_correction: bool = False,
        use_yolo_detector: bool = False,
        yolo_weights: str | None = None,
        yolo_imgsz: int = 960,
        yolo_conf: float = 0.10,
        yolo_iou: float = 0.45,
        yolo_half: bool = False,
        use_mlp_updater: bool = False,
        mlp_weights: str | None = None,
        use_sigmoid_febam: bool = True,
        use_gray_stretched_ocr: bool = True,
        ocr_ignore_febam: bool = False,
        string_febam: bool = False,
        string_febam_alpha: float = 0.30,
        string_febam_sim_thr: float = 0.60,
        string_febam_node_merge_thr: float = 0.65,
        string_febam_cluster_thr: float = 0.60,
        string_febam_commit_thr: float = 0.70,
        string_febam_margin_thr: float = 0.15,
        string_febam_debug: bool = False,
        string_febam_skip_after_commit: bool = False,
        string_febam_sim_center: float = 0.65,
        string_febam_sim_k: float = 10.0,
        string_febam_min_len: float = 6.0,
        string_febam_len_k: float = 2.0,
        string_febam_release_thr: float = 0.55,
        string_febam_switch_margin_thr: float = 0.25,
        string_febam_switch_min_segments: int = 2,
        string_febam_segment_slope: float = 0.25,
        string_febam_segment_cap: float = 1.75,
        string_febam_source_weight_weak: float = 0.35,
        string_febam_source_weight_near: float = 0.70,
        string_febam_source_weight_confirmed: float = 1.00,
        string_febam_strict_korean_commit: bool = True,
        string_febam_group_key_mode: str = "track",
        ocr_small_sharpen_fallback: bool = False,
        ocr_save_debug_crops: bool = False,
        easyocr_recognize_only: bool = True,
        easyocr_batch_size: int = 16,
        easyocr_workers: int = 0,
        easyocr_readtext_fallback: bool = True,
        ocr_expand_x: float = 0.20,
        ocr_expand_top_strong: float = 0.30,
        ocr_variant_tiered: bool = True,
        ocr_max_variants_per_candidate: int = 3,
        ocr_max_fallback_variants: int = 5,
        ocr_max_readtext_fallbacks_per_candidate: int = 1,
        ocr_disable_heavy_variants: bool = True,
        ocr_heavy_variants_only_confirmed: bool = True,
        easyocr_mode_policy: str = "auto",
        ocr_backend: str = "easyocr",
        fastplate_model: str = "cct-s-v2-global-model",
        fastplate_device: str = "cuda",
        fastplate_batch_size: int = 32,
        fastplate_preload_torch_cuda_dlls: bool = True,
        fastplate_min_text_len: int = 4,
        fastplate_async: bool = True,
        fastplate_min_batch_size: int = 4,
        fastplate_flush_timeout_ms: float = 150.0,
        fastplate_queue_max: int = 512,
        fastplate_save_debug_crops: bool = False,
        fastplate_rate_limit_relaxed: bool = False,
        fastplate_large_batch_mode: bool = False,
        fastplate_target_batch_size: int = 64,
        fastplate_max_flush_timeout_ms: float = 1500.0,
        fastplate_gpu_crop_batch: bool = False,
        fastplate_tensor_runner: bool = False,
        fastplate_tensor_runner_mode: str = "disabled",
        fastplate_ocr_input_h: int = 96,
        fastplate_ocr_input_w: int = 384,
        fastplate_parity_sample_limit: int = 256,
        fastplate_parity_log_csv: str = "",
        fastplate_tensor_profile: bool = False,
        fastplate_direct_ort: bool = False,
        fastplate_direct_ort_iobinding: bool = True,
        fastplate_direct_ort_debug: bool = False,
        fastplate_direct_ort_model_path: str | None = None,
        fastplate_direct_ort_input_h: int = 0,
        fastplate_direct_ort_input_w: int = 0,
        fastplate_direct_ort_compare_wrapper: bool = False,
        fastplate_custom_onnx: str | None = None,
        fastplate_custom_plate_config: str | None = None,
        fastplate_custom_input_width: int = 256,
        fastplate_custom_input_height: int = 64,
        dual_branch_ocr: bool = False,
        dual_branch_global_onnx: str = "",
        dual_branch_korean_onnx: str = "",
        dual_branch_hangul_weight: float = 1.2,
        dual_branch_digit_alignment_mode: str = "monotonic_dp",
        dual_branch_digit_mass_thr: float = 0.50,
        dual_branch_digit_conf_thr: float = 0.50,
        dual_branch_digit_margin_thr: float = 0.10,
        dual_branch_global_input_variant: str = "raw",
        dual_branch_gray_secondary: bool = False,
        dual_branch_global_crop_mode: str = "gpu_crop",
        dual_branch_global_result_mode: str = "digits_only",
        dual_branch_korean_mode: str = "training_exact",
        dual_branch_korean_slot_mode: str = "independent",
        dual_branch_korean_mass_thr: float = 0.50,
        dual_branch_korean_conf_thr: float = 0.50,
        dual_branch_korean_margin_thr: float = 0.10,
        dual_branch_length_margin_thr: float = 0.15,
        dual_branch_length_hold: bool = True,
        event_roi_fusion: bool = False,
        event_roi_fusion_preset: str = "none",
        event_roi_fusion_debug_dir: str = "../data/processed/debug/event_fusion",
        event_roi_fusion_save_debug: bool = False,
        event_roi_fusion_save_manifest: bool = False,
        event_roi_fusion_source_weight: float = 0.55,
        fusion_center_pad_ratio: float = 0.15,
        event_roi_fusion_mode: str = "ours_all_roi_febam",
        b5_fusion_top_k: int = 8,
        b5_fusion_min_quality_score: float = 0.0,
        b5_fusion_min_crops: int = 2,
        b5_fusion_output_tag: str = "b5_yolo_high_quality_fusion",
        event_roi_fusion_ocr: bool = False,
        event_roi_fusion_ocr_worker_mode: str = "shared_async",
        event_roi_fusion_ocr_backend: str = "fastplate",
        event_roi_fusion_ocr_source_weight: float = 0.65,
        event_roi_fusion_ocr_batch_size: int = 32,
        event_roi_fusion_ocr_flush_every_frames: int = 5,
        middle_slot_upl: bool = False,
        middle_slot_device: str = "cuda",
        middle_slot_batch_size: int = 64,
        middle_slot_min_batch_size: int = 4,
        middle_slot_flush_timeout_ms: float = 100.0,
        middle_slot_queue_max: int = 512,
        middle_slot_input_size: int = 48,
        middle_slot_max_crops_per_group: int = 3,
        middle_slot_backend: str = "upl",
        middle_slot_cropper_mode: str = "gpu_aspect_prior",
        middle_slot_encoder_mode: str = "hog_torch",
        middle_slot_easyocr_single_crop: bool = True,
        middle_slot_easyocr_gpu: bool = True,
        middle_slot_easyocr_batch_size: int = 16,
        middle_slot_easyocr_min_conf: float = 0.30,
        middle_slot_easyocr_allowlist: str = VALID_KOR,
        middle_slot_easyocr_resize_scale: int = 6,
        middle_slot_easyocr_border: int = 50,
        middle_slot_wide_core_x_pad_ratio: float = 0.16,
        middle_slot_wide_core_y_pad_ratio: float = 0.15,
        middle_slot_wide_core_min_width_ratio: float = 0.14,
        middle_slot_wide_core_max_width_ratio: float = 0.34,
        middle_slot_model_path: str | None = None,
        middle_slot_prototype_path: str | None = None,
        middle_slot_source_weight: float = 0.70,
        middle_slot_hog_lbp_model: str | None = None,
        middle_slot_hog_lbp_topk: int = 3,
        middle_slot_hog_lbp_source_weight: float = 0.10,
        middle_slot_hog_lbp_min_conf: float = 0.07,
        middle_slot_hog_lbp_min_margin: float = 0.005,
        middle_slot_hog_lbp_min_crop_score: float = 0.40,
        middle_slot_conf_thr: float = 0.70,
        middle_slot_margin_thr: float = 0.12,
        middle_slot_sim_center: float = 0.65,
        middle_slot_sim_k: float = 12.0,
        middle_slot_alpha_min: float = 0.85,
        middle_slot_alpha_max: float = 0.995,
        middle_slot_debug_dir: str = "../data/processed/debug/middle_slot",
        middle_slot_save_debug_crops: bool = False,
        middle_slot_disable_prototype_update: bool = False,
        middle_slot_diagnostic_only: bool = False,
        middle_slot_preset: str = "none",
        middle_slot_gt_anchor_confused_fair: bool = True,
        middle_slot_gt_plates_path: str | None = None,
        middle_slot_gt_anchor_boost: float = 0.40,
        middle_slot_gt_confused_min_score: float = 0.05,
        middle_slot_gt_confused_min_support: int = 1,
        middle_slot_v32: bool = False,
        middle_slot_v32_model: str | None = None,
        middle_slot_v32_device: str = "cuda",
        middle_slot_v32_batch_size: int = 512,
        middle_slot_v32_topk: int = 3,
        middle_slot_v32_source_weight: float = 0.04,
        middle_slot_v32_min_conf: float = 0.20,
        middle_slot_v32_min_margin: float = 0.03,
        middle_slot_v32_max_crops_per_event: int = 8,
        middle_slot_v32_evidence_mode: str = "topk_soft",
        middle_slot_v32_profile: bool = False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.device = torch.device("cuda")
        self.debug_stage_log = debug_stage_log
        self.debug_log_txt = debug_log_txt
        self.debug_save_video = debug_save_video
        self.debug_save_frames_dir = debug_save_frames_dir
        self.debug_event_dir = debug_event_dir
        self.debug_dump_seconds = debug_dump_seconds or set()
        self.debug_save_every_sec = debug_save_every_sec
        self.debug_max_events = debug_max_events
        self.webapp_preview_from_debug_frame = False
        self.webapp_preview_path: str | None = None
        self.webapp_preview_format = "jpg"
        self.webapp_preview_max_width = 960
        self.fps = fps
        self.scientific_g0 = bool(scientific_g0)
        self.ext30_modular_evidence = bool(ext30_modular_evidence) and self.scientific_g0
        self.ext30_external_anchor_correction = (
            bool(ext30_external_anchor_correction) and self.ext30_modular_evidence
        )
        self.ext30_evidence_orchestrator = (
            EXT30EvidenceOrchestrator(
                external_anchor_correction=self.ext30_external_anchor_correction
            )
            if self.ext30_modular_evidence
            else None
        )
        self._event_count = 0
        self.track_state = torch.zeros((256, 13), device=self.device)
        self.use_mlp_updater = use_mlp_updater
        self.use_gray_stretched_ocr = use_gray_stretched_ocr
        self.ocr_ignore_febam = bool(ocr_ignore_febam)
        self.ocr_small_sharpen_fallback = bool(ocr_small_sharpen_fallback)
        self.ocr_save_debug_crops = bool(ocr_save_debug_crops)
        self.easyocr_recognize_only = bool(easyocr_recognize_only)
        self.easyocr_batch_size = int(max(1, easyocr_batch_size))
        self.easyocr_workers = int(max(0, easyocr_workers))
        self.easyocr_readtext_fallback = bool(easyocr_readtext_fallback)
        self.ocr_expand_x = float(max(0.0, ocr_expand_x))
        self.ocr_expand_top_strong = float(max(0.0, ocr_expand_top_strong))
        self.ocr_expand_left_ratio = min(0.25, max(0.0, self.ocr_expand_x * 0.75))
        self.ocr_expand_right_ratio = min(0.25, max(0.0, self.ocr_expand_x))
        self.ocr_expand_bottom_ratio = 0.12
        self.ocr_variant_tiered = bool(ocr_variant_tiered)
        self.ocr_max_variants_per_candidate = int(max(1, ocr_max_variants_per_candidate))
        self.ocr_max_fallback_variants = int(max(self.ocr_max_variants_per_candidate, ocr_max_fallback_variants))
        self.ocr_max_readtext_fallbacks_per_candidate = int(max(0, ocr_max_readtext_fallbacks_per_candidate))
        self.ocr_disable_heavy_variants = bool(ocr_disable_heavy_variants)
        self.ocr_heavy_variants_only_confirmed = bool(ocr_heavy_variants_only_confirmed)
        self.easyocr_mode_policy = str(easyocr_mode_policy or "auto")
        if self.easyocr_mode_policy not in {"recognize_first", "readtext_first", "readtext_only", "auto"}:
            self.easyocr_mode_policy = "auto"
        self.ocr_backend = str(ocr_backend or "easyocr").lower().strip()
        if self.ocr_backend not in {"easyocr", "fastplate", "both", "none"}:
            self._log(f"[OCR] invalid backend={ocr_backend!r}; fallback to easyocr")
            self.ocr_backend = "easyocr"
        self.fastplate_model = str(fastplate_model or "cct-s-v2-global-model")
        self.fastplate_device = str(fastplate_device or "cuda")
        self.fastplate_batch_size = int(max(1, fastplate_batch_size))
        self.fastplate_preload_torch_cuda_dlls = bool(fastplate_preload_torch_cuda_dlls)
        self.fastplate_min_text_len = int(max(0, fastplate_min_text_len))
        self.fastplate_async = bool(fastplate_async)
        self.fastplate_min_batch_size = int(max(1, fastplate_min_batch_size))
        self.fastplate_flush_timeout_ms = float(max(1.0, fastplate_flush_timeout_ms))
        self.fastplate_queue_max = int(max(1, fastplate_queue_max))
        self.fastplate_save_debug_crops = bool(fastplate_save_debug_crops)
        self.fastplate_rate_limit_relaxed = bool(fastplate_rate_limit_relaxed)
        self.fastplate_large_batch_mode = bool(fastplate_large_batch_mode)
        self.fastplate_target_batch_size = int(max(1, min(fastplate_target_batch_size, self.fastplate_batch_size)))
        self.fastplate_max_flush_timeout_ms = float(max(self.fastplate_flush_timeout_ms, fastplate_max_flush_timeout_ms))
        self.fastplate_gpu_crop_batch = bool(fastplate_gpu_crop_batch)
        self.fastplate_tensor_runner = bool(fastplate_tensor_runner)
        self.fastplate_tensor_runner_mode = str(fastplate_tensor_runner_mode or "disabled").lower().strip()
        if self.fastplate_tensor_runner_mode not in {"disabled", "shadow", "active"}:
            self.fastplate_tensor_runner_mode = "disabled"
        if not self.fastplate_tensor_runner:
            self.fastplate_tensor_runner_mode = "disabled"
        self.fastplate_ocr_input_h = int(max(1, fastplate_ocr_input_h))
        self.fastplate_ocr_input_w = int(max(1, fastplate_ocr_input_w))
        self.fastplate_parity_sample_limit = int(max(0, fastplate_parity_sample_limit))
        self.fastplate_parity_log_csv = str(fastplate_parity_log_csv or "")
        self.fastplate_tensor_profile = bool(fastplate_tensor_profile)
        self.fastplate_direct_ort = bool(fastplate_direct_ort)
        self.fastplate_direct_ort_iobinding = bool(fastplate_direct_ort_iobinding)
        self.fastplate_direct_ort_debug = bool(fastplate_direct_ort_debug)
        self.fastplate_direct_ort_model_path = str(fastplate_direct_ort_model_path or "")
        self.fastplate_direct_ort_input_h = int(max(0, fastplate_direct_ort_input_h))
        self.fastplate_direct_ort_input_w = int(max(0, fastplate_direct_ort_input_w))
        self.fastplate_direct_ort_compare_wrapper = bool(fastplate_direct_ort_compare_wrapper)
        self.fastplate_custom_onnx = str(fastplate_custom_onnx or "")
        self.fastplate_custom_plate_config = str(fastplate_custom_plate_config or "")
        self.fastplate_custom_input_width = int(max(1, fastplate_custom_input_width))
        self.fastplate_custom_input_height = int(max(1, fastplate_custom_input_height))
        self.dual_branch_ocr = bool(dual_branch_ocr)
        self.dual_branch_global_onnx = str(dual_branch_global_onnx or "")
        self.dual_branch_korean_onnx = str(dual_branch_korean_onnx or "")
        self.dual_branch_hangul_weight = float(dual_branch_hangul_weight)
        self.dual_branch_digit_alignment_mode = str(dual_branch_digit_alignment_mode)
        self.dual_branch_digit_mass_thr = float(dual_branch_digit_mass_thr)
        self.dual_branch_digit_conf_thr = float(dual_branch_digit_conf_thr)
        self.dual_branch_digit_margin_thr = float(dual_branch_digit_margin_thr)
        self.dual_branch_global_input_variant = str(dual_branch_global_input_variant or "raw")
        self.dual_branch_gray_secondary = bool(dual_branch_gray_secondary)
        self.dual_branch_global_crop_mode = str(dual_branch_global_crop_mode or "gpu_crop")
        self.dual_branch_global_result_mode = str(dual_branch_global_result_mode or "digits_only")
        self.dual_branch_korean_mode = str(dual_branch_korean_mode or "training_exact")
        self.dual_branch_korean_slot_mode = str(dual_branch_korean_slot_mode or "independent")
        self.dual_branch_korean_mass_thr = float(dual_branch_korean_mass_thr)
        self.dual_branch_korean_conf_thr = float(dual_branch_korean_conf_thr)
        self.dual_branch_korean_margin_thr = float(dual_branch_korean_margin_thr)
        self.dual_branch_length_margin_thr = float(dual_branch_length_margin_thr)
        self.dual_branch_length_hold = bool(dual_branch_length_hold)
        if self.fastplate_custom_onnx or self.dual_branch_ocr:
            self.fastplate_tensor_runner = False
            self.fastplate_tensor_runner_mode = "disabled"
            self.fastplate_direct_ort = False
        if self.fastplate_direct_ort and not self.fastplate_custom_onnx:
            self.fastplate_tensor_runner = True
            if self.fastplate_tensor_runner_mode == "disabled":
                self.fastplate_tensor_runner_mode = "active"
            if self.fastplate_direct_ort_model_path and not self.fastplate_custom_onnx:
                self.fastplate_model = self.fastplate_direct_ort_model_path
        self.event_roi_fusion = bool(event_roi_fusion)
        self.event_roi_fusion_preset = str(event_roi_fusion_preset or "none").lower().strip()
        self.event_roi_fusion_debug_dir = str(event_roi_fusion_debug_dir)
        self.event_roi_fusion_save_debug = bool(event_roi_fusion_save_debug)
        self.event_roi_fusion_save_manifest = bool(event_roi_fusion_save_manifest)
        self.event_roi_fusion_source_weight = float(max(0.0, min(1.0, event_roi_fusion_source_weight)))
        self.fusion_center_pad_ratio = 0.15 if abs(float(fusion_center_pad_ratio) - 0.15) > 1e-9 else float(fusion_center_pad_ratio)
        self.event_roi_fusion_mode = str(event_roi_fusion_mode or "ours_all_roi_febam")
        self.is_b5_yolo_high_quality_fusion_mode = self.event_roi_fusion_mode == "b5_yolo_high_quality_fusion"
        self.b5_fusion_top_k = int(max(1, b5_fusion_top_k))
        self.b5_fusion_min_quality_score = float(b5_fusion_min_quality_score)
        self.b5_fusion_min_crops = int(max(2, b5_fusion_min_crops))
        self.b5_fusion_output_tag = str(b5_fusion_output_tag or "b5_yolo_high_quality_fusion")
        self.b5_fusion_log_count = 0
        self.b5_fusion_log_limit = 100
        self.b5_fusion_groups: dict[str, list[dict[str, object]]] = {}
        self.b5_fusion_input_index: dict[tuple[int, int, int, str], dict[str, object]] = {}
        self.b5_fusion_finalized = False
        if self.is_b5_yolo_high_quality_fusion_mode:
            self._log("[B5_BASELINE] gpu_pipeline B5 YOLO high-quality crop fusion mode enabled")
            self._log(
                f"[B5_BASELINE] top_k={self.b5_fusion_top_k} "
                f"min_quality={self.b5_fusion_min_quality_score} min_crops={self.b5_fusion_min_crops}"
            )
        self.event_roi_fusion_ocr = bool(event_roi_fusion_ocr)
        self.event_roi_fusion_ocr_worker_mode = str(event_roi_fusion_ocr_worker_mode or "shared_async").lower().strip()
        if self.event_roi_fusion_ocr_worker_mode not in {"shared_async", "legacy_separate"}:
            self.event_roi_fusion_ocr_worker_mode = "shared_async"
        self.event_roi_fusion_ocr_backend = str(event_roi_fusion_ocr_backend or "fastplate").lower().strip()
        if self.event_roi_fusion_ocr_backend != "fastplate":
            self._log(f"[EVENT_FUSION_OCR] unsupported backend={event_roi_fusion_ocr_backend!r}; disabling fusion OCR")
            self.event_roi_fusion_ocr = False
        self.event_roi_fusion_ocr_source_weight = float(max(0.0, min(1.0, event_roi_fusion_ocr_source_weight)))
        self.event_roi_fusion_ocr_batch_size = int(max(1, event_roi_fusion_ocr_batch_size))
        self.event_roi_fusion_ocr_flush_every_frames = int(max(1, event_roi_fusion_ocr_flush_every_frames))
        self.event_fusion_ocr_buffer = []
        self.event_fusion_ocr_seen_paths = set()
        self.event_fusion_fastplate_recognizer = None
        self.fusion_ocr_enqueued_events: set[str] = set()
        self.fusion_ocr_enqueued = 0
        self.fusion_ocr_processed = 0
        self.fusion_ocr_dropped = 0
        self.fusion_ocr_duplicate_blocked = 0
        self.fusion_ocr_commit_blocked = 0
        self.fusion_ocr_result_applied = 0
        self.fusion_ocr_result_stale = 0
        self.fusion_ocr_csv_row_written = 0
        self.fusion_ocr_queue_delay_total_ms = 0.0
        self.fusion_ocr_infer_total_ms = 0.0
        self.fusion_ocr_max_per_event_seen = 0
        self.fusion_ocr_cpu_input_count = 0
        self._last_event_fusion_ocr_flush_frame = -1
        if self.event_roi_fusion and self.event_roi_fusion_preset in {"trial020", "trial013", "trial013_fixed", "trial013_fixed_motion", "trial013_fixed_motion_color", "trial013_fixed_motion_filter"}:
            self.trial020_fusion_bridge = Trial020FusionBridge(
                enabled=True,
                debug_dir=self.event_roi_fusion_debug_dir,
                save_debug=self.event_roi_fusion_save_debug or self.event_roi_fusion_ocr,
                save_manifest=self.event_roi_fusion_save_manifest,
                source_weight=self.event_roi_fusion_source_weight,
                min_crops=(self.b5_fusion_min_crops if self.is_b5_yolo_high_quality_fusion_mode else 2),
                top_k=(self.b5_fusion_top_k if self.is_b5_yolo_high_quality_fusion_mode else 16),
                logger=self._log,
                center_pad_ratio=self.fusion_center_pad_ratio,
                preset_name=self.event_roi_fusion_preset,
                use_motion_fusion_filter=self.event_roi_fusion_preset in {"trial013_fixed_motion", "trial013_fixed_motion_color", "trial013_fixed_motion_filter"},
                use_color_soft_weight=self.event_roi_fusion_preset == "trial013_fixed_motion_color",
            )
        else:
            self.trial020_fusion_bridge = None
        self.fastplate_async_worker: FastPlateOCRQueueWorker | None = None
        self.non_generative_restoration_shadow = None
        self.middle_slot_upl = bool(middle_slot_upl)
        self.middle_slot_device = str(middle_slot_device or "cuda")
        self.middle_slot_batch_size = int(max(1, middle_slot_batch_size))
        self.middle_slot_min_batch_size = int(max(1, middle_slot_min_batch_size))
        self.middle_slot_flush_timeout_ms = float(max(1.0, middle_slot_flush_timeout_ms))
        self.middle_slot_queue_max = int(max(1, middle_slot_queue_max))
        self.middle_slot_input_size = int(max(16, middle_slot_input_size))
        self.middle_slot_max_crops_per_group = int(max(1, middle_slot_max_crops_per_group))
        self.middle_slot_backend = str(middle_slot_backend or "upl").lower().strip()
        if self.middle_slot_backend not in {"upl", "easyocr", "hog_lbp_gpu", "none"}:
            self.middle_slot_backend = "upl"
        if self.middle_slot_backend == "hog_lbp_gpu":
            self.middle_slot_upl = True
        self.middle_slot_easyocr_single_crop = bool(middle_slot_easyocr_single_crop)
        self.middle_slot_easyocr_gpu = bool(middle_slot_easyocr_gpu)
        self.middle_slot_easyocr_batch_size = int(max(1, middle_slot_easyocr_batch_size))
        self.middle_slot_easyocr_min_conf = float(max(0.0, middle_slot_easyocr_min_conf))
        self.middle_slot_easyocr_allowlist = str(middle_slot_easyocr_allowlist or VALID_KOR)
        self.middle_slot_easyocr_resize_scale = int(max(1, middle_slot_easyocr_resize_scale))
        self.middle_slot_easyocr_border = int(max(0, middle_slot_easyocr_border))
        self.middle_slot_wide_core_x_pad_ratio = float(max(0.0, middle_slot_wide_core_x_pad_ratio))
        self.middle_slot_wide_core_y_pad_ratio = float(max(0.0, middle_slot_wide_core_y_pad_ratio))
        self.middle_slot_wide_core_min_width_ratio = float(max(0.01, middle_slot_wide_core_min_width_ratio))
        self.middle_slot_wide_core_max_width_ratio = float(max(self.middle_slot_wide_core_min_width_ratio, middle_slot_wide_core_max_width_ratio))
        self.middle_slot_cropper_mode = str(middle_slot_cropper_mode or "gpu_aspect_prior")
        if self.middle_slot_backend == "easyocr" and self.middle_slot_easyocr_single_crop and self.middle_slot_cropper_mode == "gpu_aspect_prior":
            self.middle_slot_cropper_mode = "gpu_aspect_prior_wide_core"
        self.middle_slot_encoder_mode = str(middle_slot_encoder_mode or "hog_torch")
        self.middle_slot_model_path = middle_slot_model_path
        self.middle_slot_prototype_path = middle_slot_prototype_path
        self.middle_slot_source_weight = float(max(0.0, min(1.0, middle_slot_source_weight)))
        self.middle_slot_hog_lbp_model = middle_slot_hog_lbp_model
        self.middle_slot_hog_lbp_topk = int(max(1, middle_slot_hog_lbp_topk))
        self.middle_slot_hog_lbp_source_weight = float(max(0.0, min(1.0, middle_slot_hog_lbp_source_weight)))
        self.middle_slot_hog_lbp_min_conf = float(max(0.0, middle_slot_hog_lbp_min_conf))
        self.middle_slot_hog_lbp_min_margin = float(max(0.0, middle_slot_hog_lbp_min_margin))
        self.middle_slot_hog_lbp_min_crop_score = float(max(0.0, middle_slot_hog_lbp_min_crop_score))
        self.middle_slot_conf_thr = float(middle_slot_conf_thr)
        self.middle_slot_margin_thr = float(middle_slot_margin_thr)
        self.middle_slot_sim_center = float(middle_slot_sim_center)
        self.middle_slot_sim_k = float(middle_slot_sim_k)
        self.middle_slot_alpha_min = float(middle_slot_alpha_min)
        self.middle_slot_alpha_max = float(middle_slot_alpha_max)
        self.middle_slot_debug_dir = str(middle_slot_debug_dir)
        self.middle_slot_diagnostic_only = bool(middle_slot_diagnostic_only)
        self.middle_slot_preset = str(middle_slot_preset or "none")
        self.middle_slot_save_debug_crops = bool(middle_slot_save_debug_crops or self.middle_slot_diagnostic_only)
        self.middle_slot_disable_prototype_update = bool(middle_slot_disable_prototype_update)
        self.middle_slot_gt_anchor_confused_fair = bool(middle_slot_gt_anchor_confused_fair)
        self.middle_slot_gt_plates_path = middle_slot_gt_plates_path
        self.middle_slot_gt_anchor_boost = float(middle_slot_gt_anchor_boost)
        self.middle_slot_gt_confused_min_score = float(middle_slot_gt_confused_min_score)
        self.middle_slot_trigger_seen = 0
        self.middle_slot_trigger_crop_present = 0
        self.middle_slot_trigger_text_present = 0
        self.middle_slot_trigger_fallback_text_used = 0
        self.middle_slot_trigger_skipped_no_crop = 0
        self.middle_slot_trigger_skipped_no_text = 0
        self.middle_slot_trigger_worker_none = 0
        self.middle_slot_trigger_enqueue_success = 0
        self.middle_slot_trigger_enqueue_failed = 0
        self.middle_slot_trigger_worker_init_error = ""
        self.middle_slot_gt_confused_min_support = int(max(1, middle_slot_gt_confused_min_support))
        self.middle_slot_v32_enabled = bool(middle_slot_v32)
        self.middle_slot_v32_model = str(middle_slot_v32_model or r".\runs\middle_slot\gabor_scatter_lite_structured_v32_runtime_fuzzyres_cache_v5.pt")
        self.middle_slot_v32_model_path = self.middle_slot_v32_model
        self.middle_slot_v32_device = str(middle_slot_v32_device or "cuda")
        self.middle_slot_v32_batch_size = int(max(1, middle_slot_v32_batch_size))
        self.middle_slot_v32_min_batch_size = 8
        self.middle_slot_v32_flush_every_frames = 5
        self.middle_slot_v32_topk = int(max(1, min(3, middle_slot_v32_topk)))
        self.middle_slot_v32_source_weight = float(max(0.0, min(0.05, middle_slot_v32_source_weight)))
        self.middle_slot_v32_min_conf = float(max(0.0, middle_slot_v32_min_conf))
        self.middle_slot_v32_min_margin = float(max(0.0, middle_slot_v32_min_margin))
        self.middle_slot_v32_max_crops_per_event = int(max(1, middle_slot_v32_max_crops_per_event))
        self.middle_slot_v32_evidence_mode = str(middle_slot_v32_evidence_mode or "topk_soft")
        self.middle_slot_v32_profile = bool(middle_slot_v32_profile)
        self.middle_slot_v32_runtime: MiddleSlotV32Runtime | None = None
        self.middle_slot_v32_processed = 0
        self.middle_slot_v32_batch_call_count = 0
        self.middle_slot_v32_total_infer_ms = 0.0
        self.middle_slot_v32_pending_final_flush_count = 0
        self.middle_slot_v32_added_evidence = 0
        self.middle_slot_v32_skipped_low_conf = 0
        self.middle_slot_v32_skipped_low_margin = 0
        self.middle_slot_v32_skipped_other = 0
        self.middle_slot_v32_skipped_no_event_id = 0
        self.middle_slot_v32_skipped_no_track_id = 0
        self.middle_slot_v32_skipped_no_digit_skeleton = 0
        self.middle_slot_v32_skipped_invalid_digit_skeleton = 0
        self.middle_slot_v32_skipped_max_crops_per_event = 0
        self.middle_slot_v32_skipped_empty_crop = 0
        self.middle_slot_v32_skipped_duplicate = 0
        self.middle_slot_v32_skipped_pending_not_flushed = 0
        self.middle_slot_v32_skipped_model_error = 0
        self.middle_slot_v32_skipped_before_infer_unknown = 0
        self.middle_slot_v32_debug_rows: list[dict[str, object]] = []
        self.middle_slot_v32_pending: list[dict[str, object]] = []
        self.middle_slot_v32_pending_keys: set[tuple[object, ...]] = set()
        self.middle_slot_v32_crops_by_event: dict[str, int] = {}
        self._middle_slot_v32_profile_logged = False
        self.middle_slot_gt_items = []
        if self.middle_slot_gt_anchor_confused_fair and self.middle_slot_gt_plates_path:
            from ocr.middle_slot_gt_anchor_confused_fair import load_gt_plates

            self.middle_slot_gt_items = load_gt_plates(self.middle_slot_gt_plates_path)
        self.middle_slot_worker: MiddleSlotBatchQueueWorker | None = None
        self._async_ocr_outputs_finalized = False
        self.middle_slot_results_drained = 0
        self.middle_slot_evidence_added = 0
        self.middle_slot_evidence_skipped = 0
        self.middle_slot_gt_confused_evidence_added = 0
        self.middle_slot_gt_confused_evidence_skipped = 0
        self.middle_slot_evidence_diagnostic_only_skipped = 0
        self.middle_slot_csv_diagnostics: list[dict[str, object]] = []
        self._middle_slot_char_memory: dict[int, dict[str, dict[str, object]]] = {}
        self._middle_slot_group_skeleton_bank: dict[str, dict[str, dict[str, object]]] = {}
        self.fastplate_result_delay_total_frames = 0.0
        self.fastplate_result_delay_total_ms = 0.0
        self.fastplate_result_delay_count = 0
        self.fastplate_async_enqueued = 0
        self.fastplate_async_processed = 0
        self.fastplate_queue_dropped = 0
        self.fastplate_batch_recognizer = None
        self._fastplate_init_failed = False
        self._last_fastplate_result_meta: dict[str, object] = {}
        self.easyocr_batch_recognizer: EasyOCRBatchRecognizer | None = None
        self.easyocr_batch_recognizers: dict[str, EasyOCRBatchRecognizer] = {}
        self._last_easyocr_result_meta: dict[str, object] = {}
        self.ocr_policy = DefaultOCRPolicy(
            OCRConfig(
                use_gray_stretched=bool(use_gray_stretched_ocr),
                ignore_febam=bool(ocr_ignore_febam),
                small_sharpen_fallback=bool(ocr_small_sharpen_fallback),
                save_debug_crops=self.ocr_save_debug_crops,
                mandatory_one_per_frame=True,
                per_frame_budget=1,
            )
        )
        self.yolo_half = bool(yolo_half)
        self.ocr_roi_debug_dir = Path("../data/processed/debug/ocr_roi")
        self.febam_score_thr = 0.35
        self.febam_energy_thr = 0.40
        self.febam_memory_min = 2
        self.near_febam_score_thr = 0.30
        self.near_febam_energy_thr = 0.45
        self.near_confirmed_sample_score_thr = 0.45
        self.near_confirmed_sample_energy_thr = 0.30
        self.near_confirmed_sample_memory_min = 1.0
        self.weak_plate_sample_score_thr = 0.02
        self.weak_plate_sample_min_w = 80
        self.weak_plate_sample_min_h = 18
        self.weak_plate_sample_max_per_frame = 2
        self.periodic_track_sample_interval = 5
        self.yolo_batch_call_count = 0
        self.yolo_batch_item_count = 0
        self.yolo_batch_size_sum = 0
        self.yolo_batch_last_size = 0
        self.yolo_batch_fallback_single_count = 0
        self.yolo_detector = None
        if use_yolo_detector:
            from detection.yolo_plate_detector import YOLOPlateDetector
            self.yolo_detector = YOLOPlateDetector(
                yolo_weights,
                conf=yolo_conf,
                iou=yolo_iou,
                imgsz=yolo_imgsz,
                device="cuda",
                half=yolo_half,
            )
        self.mlp_updater = PlateMLPUpdater(mlp_weights, device="cuda") if use_mlp_updater else None
        febam_mode = "sigmoid_ema" if use_sigmoid_febam else "default"
        self.febam = GPUFEBAM(mode=febam_mode)
        self.febam_mode_id = 1.0 if febam_mode == "sigmoid_ema" else 0.0
        self.tracker_max_age = int(getattr(self.febam, "max_age", 2))
        self.ocr_history: dict[int, list[dict[str, object]]] = {}
        self.final_plates: dict[int, str] = {}
        self.plate_skeleton_memory: dict[int, dict[str, dict[str, object]]] = {}
        self.plate_skeleton_memory_max_age_frames = 60
        self.ocr_csv_rows: list[dict[str, object]] = []
        self.probability_clip_count = 0
        self.invalid_probability_count = 0
        self.invalid_probability_field_examples: list[str] = []
        self.ocr_history_max = 10
        self.ocr_topk = 3
        self.use_string_febam = bool(string_febam)
        self.string_febam_alpha = float(max(0.0, min(1.0, string_febam_alpha)))
        self.string_febam_sim_thr = float(string_febam_sim_thr)
        self.string_febam_node_merge_thr = float(string_febam_node_merge_thr)
        self.string_febam_cluster_thr = float(string_febam_cluster_thr)
        self.string_febam_commit_thr = float(string_febam_commit_thr)
        self.string_febam_margin_thr = float(string_febam_margin_thr)
        self.string_febam_debug = bool(string_febam_debug)
        self.string_febam_skip_after_commit = bool(string_febam_skip_after_commit)
        self.string_febam_skip_interval = 5
        self.string_febam_config = StringFEBAMConfig(
            alpha=self.string_febam_alpha,
            sim_thr=self.string_febam_sim_thr,
            node_merge_thr=self.string_febam_node_merge_thr,
            cluster_thr=self.string_febam_cluster_thr,
            commit_thr=self.string_febam_commit_thr,
            release_thr=float(string_febam_release_thr),
            margin_thr=self.string_febam_margin_thr,
            switch_margin_thr=float(string_febam_switch_margin_thr),
            switch_min_segments=int(string_febam_switch_min_segments),
            sim_center=float(string_febam_sim_center),
            sim_k=float(string_febam_sim_k),
            min_len=float(string_febam_min_len),
            len_k=float(string_febam_len_k),
            segment_slope=float(string_febam_segment_slope),
            segment_cap=float(string_febam_segment_cap),
            source_weight_weak=float(string_febam_source_weight_weak),
            source_weight_near=float(string_febam_source_weight_near),
            source_weight_confirmed=float(string_febam_source_weight_confirmed),
            strict_korean_commit=bool(string_febam_strict_korean_commit),
            debug=self.string_febam_debug,
        )
        self.string_febam_engine = StringFEBAMEngine(self.string_febam_config)
        self.string_febam_min_support_segments = int(self.string_febam_config.min_support_segments)
        self.string_febam_min_segment_frames = int(self.string_febam_config.min_segment_frames)
        self.string_febam_epoch_frame_gap = int(self.string_febam_config.epoch_frame_gap)
        self.string_febam_commit_epoch_gap = int(self.string_febam_config.commit_epoch_gap)
        self.string_febam_epoch_dissim_thr = float(self.string_febam_config.epoch_dissim_thr)
        self.string_febam_observations: dict[int, list[dict[str, object]]] = {}
        self.string_febam_quarantine: dict[int, list[dict[str, object]]] = {}
        self.string_febam_nodes: dict[int, list[StringFEBAMNode]] = {}
        self.string_febam_segments: dict[int, list[dict[str, object]]] = {}
        self.string_febam_states: dict[int, dict[str, object]] = {}
        self.string_febam_commit_frames: dict[int, int] = {}
        self.string_febam_ocr_call_count: dict[int, int] = {}
        self.string_febam_skipped_after_commit: dict[int, int] = {}
        # StringFEBAMEngine uses integer keys internally. Vehicle/event group
        # identifiers may be strings, so retain their semantics through a
        # stable runtime integer mapping instead of falling back to track_id.
        self.string_febam_group_runtime_ids: dict[str, int] = {}
        self.string_febam_runtime_group_keys: dict[int, str] = {}
        self._next_string_febam_group_runtime_id = 1_000_000_000
        self.string_febam_group_migrations: set[tuple[int, int]] = set()
        self.string_febam_group_key_mode = str(string_febam_group_key_mode or "track")
        # Runtime state only; routing policy and ED/space/time clustering stay
        # in the external OCR module.
        self.vehicle_febam_router = VehicleFEBAMRouter()
        self.gpu_final_vehicle_grouper = GPUFinalVehicleGrouper()
        self.final_vehicle_posterior_gate = FinalVehiclePosteriorGate(minimum_valid_frames=4)
        self.final_vehicle_sparse_route_count = 0
        self.final_vehicle_posterior_route_count = 0
        self.final_vehicle_identity_hold_count = 0
        self.final_vehicle_last_route: dict[str, object] = {}
        self.gpu_final_vehicle_grouping = False
        self.runtime_video_id = ""
        self.last_valid_track_bboxes: dict[int, tuple[torch.Tensor, int]] = {}
        self.last_valid_bbox_max_age = 5
        self._last_candidate_track_ids = torch.zeros((0,), device=self.device, dtype=torch.long)
        self._last_gray_stretched_ocr_used = False
        self._last_ocr_text = ""
        self._last_ocr_conf = 0.0
        self._last_final_plate = ""
        self.ocr_reader = None
        self.ocr_ko_reader = None
        self.ocr_reader_langs = ["ko", "en"]
        self.ocr_korean_plate_chars = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호배"
        self.ocr_allowlist_expanded = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" + self.ocr_korean_plate_chars
        self.ocr_allowlist_digits_ko = "0123456789" + self.ocr_korean_plate_chars

        # Roles: YOLO=recall-first candidate generator, MLP=reranker,
        # Tracker=track_id splitter, FEBAM=short-burst per-track stabilizer,
        # OCR=text evidence, OCR voting=per-track final text decision.
        self._log_lock = threading.Lock()
        self._log_fh = None
        if self.debug_log_txt:
            os.makedirs(os.path.dirname(self.debug_log_txt) or ".", exist_ok=True)
            self._log_fh = open(self.debug_log_txt, "a", encoding="utf-8")
        if self.yolo_detector is not None:
            self._log(f"[YOLO] half={self.yolo_half}")
        try:
            middle_slot_source = inspect.getsource(self._handle_middle_slot_result)
            middle_slot_append_patch = (
                "append(diagnostic_row)" in middle_slot_source
                and "append(evidence_row)" in middle_slot_source
                and ("ocr_csv_rows[-1]." + "update(diag)") not in middle_slot_source
                and ("ocr_csv_rows[-1]." + "update(evidence_row_extra)") not in middle_slot_source
            )
        except Exception as exc:
            middle_slot_append_patch = False
            middle_slot_source = f"inspect_failed:{type(exc).__name__}"
        code_id_msg = f"[CODE_ID] gpu_pipeline_file= {__file__}"
        patch_id_msg = f"[CODE_ID] middle_slot_append_patch={middle_slot_append_patch} source_len={len(middle_slot_source)}"
        print(code_id_msg, flush=True)
        print(patch_id_msg, flush=True)
        self._log(code_id_msg)
        self._log(patch_id_msg)

        self.video_writer_q: queue.Queue = queue.Queue(maxsize=128)
        self.image_writer_q: queue.Queue = queue.Queue(maxsize=128)
        self._video_writer = None
        self._video_path = None
        self.debug_video_width: int | None = None
        self.debug_video_height: int | None = None
        self._debug_video_shape_logged = False
        self._video_writer_failed = False
        self._writer_error_count = 0
        self.video_writer = threading.Thread(target=self._video_writer_loop, daemon=True)
        self.image_writer = threading.Thread(target=self._image_writer_loop, daemon=True)
        self.video_writer.start()
        self.image_writer.start()
        self._stage6_debug = {}
        self.debug_draw_candidate_stages = DEBUG_DRAW_CANDIDATE_STAGES
        self.fuzzy_memory = None
        self.memory_decay = 0.90
        self.memory_gain = 0.10
        self._stage6_gx = None
        self._stage6_fuzzy_temporal = None
        self._stage6_direction_prior = None
        self._stage6_lower_prior = None
        self._stage6_fuzzy_source = None
        self._last_fuzzy_soft_map = None
        self._last_local_peak_map = None
        self._last_rect_response = None
        self._last_rect_response_raw = None
        self._last_rect_peaks = None
        self._last_rect_candidates = None
        self._last_center_inner_midscale = None
        self._last_ring_mean_midscale = None
        self._last_vertical_closure_midscale = None
        self._last_rect_response_selectivity = None
        # If ghost is too weak, try 0.55~0.60.
        # If water splash false positives increase, use 0.70~0.75.

    def _log(self, msg: str):
        # Writer thread may still emit late debug logs while the pipeline is closing.
        # Keep logging best-effort so a closed/invalid file handle never kills _writer_loop.
        lock = getattr(self, "_log_lock", None)
        if lock is None:
            return
        with lock:
            fh = self._log_fh
            if fh is None or getattr(fh, "closed", False):
                return
            try:
                fh.write(msg + "\n")
                fh.flush()
            except Exception:
                try:
                    fh.close()
                except Exception:
                    pass
                self._log_fh = None


    def _log_always(self, msg: str) -> None:
        print(str(msg), flush=True)
        try:
            self._log(str(msg))
        except Exception:
            pass

    def _assert_cuda(self, t: torch.Tensor, name: str):
        if not t.is_cuda:
            raise RuntimeError(f"CPU fallback detected: {name}")

    def stage1_stabilize(self, frame_u8: torch.Tensor, gate):
        x, y, w, h = gate
        return frame_u8, frame_u8[:, :, y:y+h, x:x+w]

    def stage3_structure_tensor(self, gx: torch.Tensor, gy: torch.Tensor):
        j11, j22, j12 = gx*gx, gy*gy, gx*gy
        k = torch.tensor([1.0,4.0,6.0,4.0,1.0], device=self.device)
        k = (k / k.sum()).view(1,1,1,5)
        def sep(x):
            x = torch.nn.functional.conv2d(x, k, padding=(0,2))
            x = torch.nn.functional.conv2d(x, k.transpose(-1,-2), padding=(2,0))
            return x
        j11, j22, j12 = sep(j11), sep(j22), sep(j12)
        theta = 0.5 * torch.atan2(2.0*j12, j11-j22+1e-6)
        coherence = torch.sqrt((j11-j22)**2 + 4.0*(j12**2)) / (j11+j22+1e-6)
        return theta, coherence


    def _merge_same_row_plate_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        max_boxes: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0 or boxes.shape[0] <= 1:
            return boxes, scores

        if boxes.shape[0] > max_boxes:
            top = torch.topk(scores, k=max_boxes).indices
            boxes_m = boxes[top]
            scores_m = scores[top]
            boxes_rest_mask = torch.ones((boxes.shape[0],), device=boxes.device, dtype=torch.bool)
            boxes_rest_mask[top] = False
            boxes_rest = boxes[boxes_rest_mask]
            scores_rest = scores[boxes_rest_mask]
        else:
            boxes_m = boxes
            scores_m = scores
            boxes_rest = torch.zeros((0, 4), device=boxes.device, dtype=boxes.dtype)
            scores_rest = torch.zeros((0,), device=scores.device, dtype=scores.dtype)

        x1, y1, x2, y2 = boxes_m[:, 0], boxes_m[:, 1], boxes_m[:, 2], boxes_m[:, 3]
        bw = (x2 - x1 + 1.0).clamp(min=1.0)
        bh = (y2 - y1 + 1.0).clamp(min=1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        h_i = bh[:, None]
        h_j = bh[None, :]
        mean_h = 0.5 * (h_i + h_j)

        cy_i = cy[:, None]
        cy_j = cy[None, :]

        x1_i = x1[:, None]
        x1_j = x1[None, :]
        x2_i = x2[:, None]
        x2_j = x2[None, :]

        horizontal_gap = torch.maximum(
            torch.maximum(x1_i, x1_j) - torch.minimum(x2_i, x2_j) - 1.0,
            torch.zeros_like(mean_h),
        )

        norm_gap = horizontal_gap / (mean_h + 1e-6)
        center_y_diff = torch.abs(cy_i - cy_j) / (mean_h + 1e-6)
        height_ratio = torch.minimum(h_i, h_j) / (torch.maximum(h_i, h_j) + 1e-6)

        same_row_merge = (
            (center_y_diff <= 0.50)
            & (height_ratio >= 0.50)
            & (norm_gap <= 2.00)
        )

        eye = torch.eye(boxes_m.shape[0], device=boxes.device, dtype=torch.bool)
        adj = same_row_merge | eye

        gid = torch.arange(boxes_m.shape[0], device=boxes.device, dtype=torch.long)
        neg = torch.full((boxes_m.shape[0], boxes_m.shape[0]), -1, device=boxes.device, dtype=torch.long)

        for _ in range(16):
            gmat = gid[None, :].expand(boxes_m.shape[0], boxes_m.shape[0])
            gid = torch.where(adj, gmat, neg).max(dim=1).values

        unique_gid, inv = torch.unique(gid, return_inverse=True)
        g = unique_gid.numel()

        ux1 = torch.full((g,), float(1e9), device=boxes.device, dtype=boxes.dtype)
        uy1 = torch.full((g,), float(1e9), device=boxes.device, dtype=boxes.dtype)
        ux2 = torch.zeros((g,), device=boxes.device, dtype=boxes.dtype)
        uy2 = torch.zeros((g,), device=boxes.device, dtype=boxes.dtype)
        uscore = torch.zeros((g,), device=scores.device, dtype=scores.dtype)

        ux1.scatter_reduce_(0, inv, x1, reduce="amin", include_self=True)
        uy1.scatter_reduce_(0, inv, y1, reduce="amin", include_self=True)
        ux2.scatter_reduce_(0, inv, x2, reduce="amax", include_self=True)
        uy2.scatter_reduce_(0, inv, y2, reduce="amax", include_self=True)
        uscore.scatter_reduce_(0, inv, scores_m, reduce="amax", include_self=True)

        merged_boxes = torch.stack([ux1, uy1, ux2, uy2], dim=1)
        out_boxes = torch.cat([merged_boxes, boxes_rest], dim=0)
        out_scores = torch.cat([uscore, scores_rest], dim=0)
        return out_boxes, out_scores


    def _rect_sums(self, integral: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), device=integral.device, dtype=integral.dtype)
        h_max = integral.shape[-2] - 1
        w_max = integral.shape[-1] - 1
        x1 = boxes[:, 0].round().to(torch.long).clamp(0, w_max)
        y1 = boxes[:, 1].round().to(torch.long).clamp(0, h_max)
        x2 = (boxes[:, 2].round().to(torch.long) + 1).clamp(0, w_max)
        y2 = (boxes[:, 3].round().to(torch.long) + 1).clamp(0, h_max)
        return integral[0, 0, y2, x2] - integral[0, 0, y1, x2] - integral[0, 0, y2, x1] + integral[0, 0, y1, x1]

    def _fit_spatial(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x[:, :, :height, :width]
        pad_h = max(0, height - x.shape[-2])
        pad_w = max(0, width - x.shape[-1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def _shift_y_zero(self, x: torch.Tensor, offset: int, direction: str) -> torch.Tensor:
        if offset <= 0:
            return x
        if direction == "from_above":
            return F.pad(x[:, :, :-offset, :], (0, 0, offset, 0))
        if direction == "from_below":
            return F.pad(x[:, :, offset:, :], (0, 0, 0, offset))
        raise ValueError(f"unknown shift direction: {direction}")

    def make_center_surround_rect_response(
        self,
        evidence_source: torch.Tensor,
        height_scales: tuple[float, ...] = (0.040, 0.055, 0.070, 0.090),
        aspect_candidates: tuple[float, ...] = (2.5, 3.5, 4.5),
        surround_scale: float = 1.8,
        use_vertical_closure: bool = False,
    ) -> torch.Tensor:
        # temporal evidence center-surround rectangle proposal gpu-only
        self._assert_cuda(evidence_source, "evidence_source")
        e = torch.nan_to_num(evidence_source, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        roi_h, roi_w = e.shape[-2:]
        responses: list[torch.Tensor] = []
        mid_scale = len(height_scales) // 2
        mid_aspect = len(aspect_candidates) // 2
        self._last_center_inner_midscale = None
        self._last_ring_mean_midscale = None
        self._last_vertical_closure_midscale = None

        for scale_index, frac in enumerate(height_scales):
            plate_h = max(3, int(round(roi_h * frac)))
            if plate_h % 2 == 0:
                plate_h += 1
            if plate_h > roi_h:
                continue

            for aspect_index, aspect in enumerate(aspect_candidates):
                plate_w = max(9, int(round(plate_h * aspect)))
                if plate_w % 2 == 0:
                    plate_w += 1
                if plate_w > roi_w:
                    continue

                outer_h = max(plate_h + 2, int(round(plate_h * surround_scale)))
                outer_w = max(plate_w + 2, int(round(plate_w * surround_scale)))
                if outer_h % 2 == 0:
                    outer_h += 1
                if outer_w % 2 == 0:
                    outer_w += 1
                outer_h = min(outer_h, roi_h if roi_h % 2 == 1 else max(1, roi_h - 1))
                outer_w = min(outer_w, roi_w if roi_w % 2 == 1 else max(1, roi_w - 1))
                if outer_h <= plate_h or outer_w <= plate_w:
                    continue

                inner_mean = F.avg_pool2d(
                    e,
                    kernel_size=(plate_h, plate_w),
                    stride=1,
                    padding=(plate_h // 2, plate_w // 2),
                )
                outer_mean = F.avg_pool2d(
                    e,
                    kernel_size=(outer_h, outer_w),
                    stride=1,
                    padding=(outer_h // 2, outer_w // 2),
                )
                inner_mean = self._fit_spatial(inner_mean, roi_h, roi_w)
                outer_mean = self._fit_spatial(outer_mean, roi_h, roi_w)

                inner_area = float(plate_h * plate_w)
                outer_area = float(outer_h * outer_w)
                ring_area = max(outer_area - inner_area, 1.0)
                ring_sum = outer_mean * outer_area - inner_mean * inner_area
                ring_mean = (ring_sum / (ring_area + 1e-6)).clamp(0.0, 1.0)
                rect_resp = (inner_mean - ring_mean).clamp_min(0.0) * inner_mean

                vertical_closure = torch.ones_like(rect_resp)
                if use_vertical_closure:
                    offset = plate_h // 2
                    top_band = self._shift_y_zero(inner_mean, offset, "from_above")
                    bottom_band = self._shift_y_zero(inner_mean, offset, "from_below")
                    vertical_closure = (inner_mean - 0.5 * (top_band + bottom_band)).clamp_min(0.0)
                    rect_resp = rect_resp * vertical_closure

                responses.append(rect_resp.clamp_min(0.0))

                if scale_index == mid_scale and aspect_index == mid_aspect:
                    self._last_center_inner_midscale = inner_mean
                    self._last_ring_mean_midscale = ring_mean
                    self._last_vertical_closure_midscale = vertical_closure

        if len(responses) == 0:
            raw_rect_response = torch.zeros_like(e)
        else:
            raw_rect_response = torch.stack(responses, dim=0).amax(dim=0)

        raw_max = raw_rect_response.amax(dim=(-2, -1), keepdim=True)
        raw_mean = raw_rect_response.mean(dim=(-2, -1), keepdim=True)
        self._last_rect_response_raw = raw_rect_response
        self._last_rect_response_selectivity = raw_max / (raw_mean + 1e-6)
        rect_response = (raw_rect_response / (raw_max + 1e-6)).clamp(0.0, 1.0)
        self._last_rect_response = rect_response
        return rect_response

    def generate_boxes(
        self,
        evidence_source: torch.Tensor,
        min_peak_abs: float = 0.20,
        peak_pool_kernel: int = 9,
        pre_nms_topk: int = 2048,
        selectivity_thr: float = 3.0,
    ) -> torch.Tensor:
        # temporal evidence center-surround rectangle proposal gpu-only
        self._assert_cuda(evidence_source, "evidence_source")
        roi_h, roi_w = evidence_source.shape[-2:]
        dtype = evidence_source.dtype
        device = evidence_source.device
        rect_response = self.make_center_surround_rect_response(evidence_source)
        selectivity = self._last_rect_response_selectivity
        if selectivity is None or bool((selectivity < selectivity_thr).all()):
            self._last_rect_peaks = torch.zeros_like(evidence_source)
            self._last_local_peak_map = self._last_rect_peaks
            return torch.zeros((0, 4), device=device, dtype=dtype)

        local_max = F.max_pool2d(
            rect_response,
            kernel_size=peak_pool_kernel,
            stride=1,
            padding=peak_pool_kernel // 2,
        )
        peak_score_map = torch.where(rect_response == local_max, rect_response, torch.zeros_like(rect_response))
        self._last_rect_peaks = peak_score_map
        self._last_local_peak_map = peak_score_map

        peak_values = peak_score_map.flatten()
        pre_k = min(pre_nms_topk, int(peak_values.numel()))
        top_values, top_indices = torch.topk(peak_values, k=pre_k)
        keep_peak = top_values > min_peak_abs
        peak_indices = top_indices[keep_peak]
        peak_y = torch.div(peak_indices, roi_w, rounding_mode="floor").to(dtype)
        peak_x = (peak_indices % roi_w).to(dtype)

        height_scales = torch.tensor((0.040, 0.055, 0.070, 0.090), device=device, dtype=dtype)
        aspect_candidates = torch.tensor((2.5, 3.5, 4.5), device=device, dtype=dtype)
        win_h = (height_scales * float(roi_h)).round().clamp(min=3.0)
        win_h = win_h + torch.remainder(win_h + 1.0, 2.0)
        win_w = (win_h[:, None] * aspect_candidates[None, :]).round().clamp(min=9.0)
        win_w = win_w + torch.remainder(win_w + 1.0, 2.0)
        win_h = win_h[:, None].expand_as(win_w).reshape(-1)
        win_w = win_w.reshape(-1)
        area_ratio = (win_h * win_w) / float(max(roi_h * roi_w, 1))
        shape_keep = (win_w <= float(roi_w)) & (win_h <= float(roi_h)) & (area_ratio >= 0.0005) & (area_ratio <= 0.040)
        win_h = win_h[shape_keep]
        win_w = win_w[shape_keep]
        if peak_x.numel() == 0 or win_h.numel() == 0:
            return torch.zeros((0, 4), device=device, dtype=dtype)

        cx = peak_x[:, None].expand(-1, win_h.numel()).reshape(-1)
        cy = peak_y[:, None].expand(-1, win_h.numel()).reshape(-1)
        ww = win_w[None, :].expand(peak_x.numel(), -1).reshape(-1)
        hh = win_h[None, :].expand(peak_x.numel(), -1).reshape(-1)
        x1 = (cx - ww * 0.5).clamp(0.0, float(max(roi_w - 1, 0)))
        y1 = (cy - hh * 0.5).clamp(0.0, float(max(roi_h - 1, 0)))
        x2 = (x1 + ww - 1.0).clamp(0.0, float(max(roi_w - 1, 0)))
        y2 = (y1 + hh - 1.0).clamp(0.0, float(max(roi_h - 1, 0)))
        x1 = (x2 - ww + 1.0).clamp(0.0, float(max(roi_w - 1, 0)))
        y1 = (y2 - hh + 1.0).clamp(0.0, float(max(roi_h - 1, 0)))
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        widths = (boxes[:, 2] - boxes[:, 0] + 1.0).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1] + 1.0).clamp(min=1.0)
        ar = widths / (heights + 1e-6)
        valid = (ar >= 2.3) & (ar <= 5.2) & (boxes[:, 0] >= 0.0) & (boxes[:, 1] >= 0.0) & (boxes[:, 2] < float(roi_w)) & (boxes[:, 3] < float(roi_h))
        return boxes[valid]

    def score_boxes(
        self,
        boxes: torch.Tensor,
        evidence_source: torch.Tensor,
        rect_response: torch.Tensor,
        fuzzy_memory: torch.Tensor,
        lower_prior_map: torch.Tensor,
        diagonal_prior_map: torch.Tensor,
        gx: torch.Tensor,
        w_y: float = 0.08,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_cuda(boxes, "boxes")
        self._assert_cuda(evidence_source, "evidence_source")
        if boxes.numel() == 0:
            z = torch.zeros((0,), device=evidence_source.device, dtype=evidence_source.dtype)
            return z, torch.zeros((0, 6), device=evidence_source.device, dtype=evidence_source.dtype)

        score_source = evidence_source.clamp(0.0, 1.0)
        memory_support = fuzzy_memory.to(device=score_source.device, dtype=score_source.dtype).clamp(0.0, 1.0)
        rect_support = rect_response.to(device=score_source.device, dtype=score_source.dtype).clamp(0.0, 1.0)
        gx_abs = gx.abs()
        gx_abs = gx_abs / (gx_abs.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        source_integral = F.pad(score_source.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))
        memory_integral = F.pad(memory_support.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))
        rect_integral = F.pad(rect_support.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))
        gx_integral = F.pad(gx_abs.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))

        roi_h, roi_w = score_source.shape[-2:]
        widths = (boxes[:, 2] - boxes[:, 0] + 1.0).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1] + 1.0).clamp(min=1.0)
        area = (widths * heights).clamp(min=1.0)
        evidence_mean = (self._rect_sums(source_integral, boxes) / area).clamp(0.0, 1.0)
        response_mean = (self._rect_sums(rect_integral, boxes) / area).clamp(0.0, 1.0)
        rect_region_score = (0.70 * evidence_mean + 0.30 * response_mean).clamp(0.0, 1.0)
        temporal_consistency = (self._rect_sums(memory_integral, boxes) / area).clamp(0.0, 1.0)
        gx_mean = (self._rect_sums(gx_integral, boxes) / area).clamp(0.0, 1.0)

        band_h = torch.clamp((heights * 0.38).round(), min=1.0)
        center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        band_y1 = (center_y - band_h * 0.5).clamp(min=0.0)
        band_y2 = (center_y + band_h * 0.5).clamp(max=float(max(roi_h - 1, 0)))
        band_boxes = torch.stack([boxes[:, 0], band_y1, boxes[:, 2], band_y2], dim=1)
        band_area = (widths * (band_y2 - band_y1 + 1.0).clamp(min=1.0)).clamp(min=1.0)
        band_mean = (self._rect_sums(source_integral, band_boxes) / band_area).clamp(0.0, 1.0)
        band_gx_mean = (self._rect_sums(gx_integral, band_boxes) / band_area).clamp(0.0, 1.0)
        text_band_score = (0.65 * (band_mean - evidence_mean).abs() + 0.35 * band_gx_mean).clamp(0.0, 1.0)
        blob_penalty = ((evidence_mean - 0.60).clamp(0.0, 1.0) * (1.0 - response_mean) * (1.0 - gx_mean)).clamp(0.0, 1.0)

        cx_norm = (((boxes[:, 0] + boxes[:, 2]) * 0.5) / max(float(roi_w), 1.0)).clamp(0.0, 1.0)
        cy_norm = (((boxes[:, 1] + boxes[:, 3]) * 0.5) / max(float(roi_h), 1.0)).clamp(0.0, 1.0)
        lower_prior_score = torch.exp(-torch.abs(cy_norm - 0.80) / 0.16).clamp(0.0, 1.0)
        diagonal_band = torch.exp(-torch.abs((cy_norm - 0.55) - 0.75 * (cx_norm - 0.30)) / 0.22)
        diagonal_motion_prior_score = (0.60 + 0.40 * diagonal_band).clamp(0.0, 1.0)
        y_gate_soft = w_y * torch.tanh((cy_norm - 0.58) / 0.12)
        mlp_score = (
            0.35 * rect_region_score
            + 0.25 * text_band_score
            + 0.20 * temporal_consistency
            + 0.10 * lower_prior_score
            + 0.10 * diagonal_motion_prior_score
            - 0.35 * blob_penalty
        ).clamp(0.0, 1.0)
        proposal_score = (
            0.45 * rect_region_score
            + 0.20 * lower_prior_score
            + 0.15 * diagonal_motion_prior_score
            + 0.10 * temporal_consistency
            + 0.10 * mlp_score
            - blob_penalty
            + y_gate_soft
        ).clamp(0.0, 1.0)
        metrics = torch.stack([rect_region_score, text_band_score, lower_prior_score, diagonal_motion_prior_score, blob_penalty, mlp_score], dim=1)
        return proposal_score, metrics

    def _horizontal_same_row_stroke_merge(
        self,
        morphology_binary: torch.Tensor,
        coherence: torch.Tensor,
        max_components: int = 80,
        max_pairs: int = 128,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Merge sparse same-row stroke components with an x-only bridge."""
        self._assert_cuda(morphology_binary, "morphology_binary")
        self._assert_cuda(coherence, "coherence")
        source = morphology_binary.float().clamp(0.0, 1.0)
        B, _, H, W = source.shape
        device = source.device
        dtype = source.dtype
        zero4 = torch.zeros((0, 4), device=device, dtype=dtype)
        zero1 = torch.zeros((1,), device=device, dtype=dtype)
        zero_map = torch.zeros_like(source)

        motion_prior_map = _bottom_motion_prior(H, W, device=device, dtype=dtype) if H > 0 and W > 0 else zero_map

        debug_empty = {
            "stroke_merge_source_map": source,
            "stroke_components_map": zero_map,
            "motion_prior_map": motion_prior_map,
            "same_row_merge_map": zero_map,
            "stroke_merge_morphology_before": source,
            "stroke_merge_morphology_after": source,
            "stroke_merge_pair_boxes": zero4,
            "stroke_merge_count": zero1,
        }
        if B != 1 or H <= 0 or W <= 0:
            return source, debug_empty

        # First pass: read existing morphology components only. This pass is not
        # candidate generation; final Candidate Generation is run again below on
        # the merged morphology mask.
        ccl_label_propagation(source, coherence)
        raw_boxes = gpu_ccl_mod.LAST_CCL_DEBUG.get("raw_boxes", zero4).to(device=device, dtype=dtype)
        if raw_boxes.numel() == 0 or raw_boxes.shape[0] <= 1:
            return source, debug_empty

        boxes = raw_boxes.clamp_min(0.0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(max(W - 1, 0)))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(max(H - 1, 0)))
        x1_all, y1_all, x2_all, y2_all = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        bw_all = (x2_all - x1_all + 1.0).clamp_min(1.0)
        bh_all = (y2_all - y1_all + 1.0).clamp_min(1.0)
        bbox_area_all = (bw_all * bh_all).clamp_min(1.0)
        aspect_all = bw_all / (bh_all + 1e-6)
        cx_all = (x1_all + x2_all) * 0.5
        cy_all = (y1_all + y2_all) * 0.5
        cx_norm_all = (cx_all / max(float(W - 1), 1.0)).clamp(0.0, 1.0)
        cy_norm_all = (cy_all / max(float(H - 1), 1.0)).clamp(0.0, 1.0)

        integral = F.pad(source.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))
        pixel_area_all = self._rect_sums(integral, boxes).clamp_min(0.0)
        fill_all = (pixel_area_all / (bbox_area_all + 1e-6)).clamp(0.0, 1.0)

        cx_idx_all = cx_all.round().to(torch.long).clamp(0, W - 1)
        cy_idx_all = cy_all.round().to(torch.long).clamp(0, H - 1)
        motion_prior_all = motion_prior_map[0, 0, cy_idx_all, cx_idx_all].clamp(0.0, 1.0)

        height_min = torch.tensor(4.0, device=device, dtype=dtype)
        height_max = torch.tensor(28.0, device=device, dtype=dtype)
        width_min = torch.tensor(2.0, device=device, dtype=dtype)
        width_max = torch.tensor(40.0, device=device, dtype=dtype)
        area_max = torch.tensor(600.0, device=device, dtype=dtype)
        stroke_like = (
            (bh_all >= height_min)
            & (bh_all <= height_max)
            & (bw_all >= width_min)
            & (bw_all <= width_max)
            & (bbox_area_all <= area_max)
            & (pixel_area_all <= area_max)
            & (aspect_all <= 3.2)
            & (fill_all >= 0.01)
            & (fill_all <= 0.95)
        )
        small_area_score_all = torch.exp(-bbox_area_all / area_max).clamp(0.05, 1.0)
        component_score_all = (motion_prior_all * small_area_score_all * stroke_like.to(dtype)).clamp(0.0, 1.0)

        component_map_all = torch.zeros_like(source)
        for comp_idx in range(int(boxes.shape[0])):
            if component_score_all[comp_idx] <= 0.0:
                continue
            bx1 = x1_all[comp_idx].round().to(torch.long).clamp(0, W - 1)
            by1 = y1_all[comp_idx].round().to(torch.long).clamp(0, H - 1)
            bx2 = x2_all[comp_idx].round().to(torch.long).clamp(0, W - 1)
            by2 = y2_all[comp_idx].round().to(torch.long).clamp(0, H - 1)
            component_map_all[:, :, by1:by2 + 1, bx1:bx2 + 1] = torch.maximum(component_map_all[:, :, by1:by2 + 1, bx1:bx2 + 1], component_score_all[comp_idx])

        idx = torch.where(component_score_all > 0.0)[0]
        if idx.numel() <= 1:
            debug_empty.update({"stroke_components_map": component_map_all})
            return source, debug_empty
        if idx.numel() > max_components:
            top = torch.topk(component_score_all[idx], k=int(max_components)).indices
            idx = idx[top]

        boxes = boxes[idx]
        component_score = component_score_all[idx]
        small_area_score = small_area_score_all[idx]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        bw = (x2 - x1 + 1.0).clamp_min(1.0)
        bh = (y2 - y1 + 1.0).clamp_min(1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        n = boxes.shape[0]

        x1_i, x1_j = x1[:, None], x1[None, :]
        x2_i, x2_j = x2[:, None], x2[None, :]
        y1_i, y1_j = y1[:, None], y1[None, :]
        y2_i, y2_j = y2[:, None], y2[None, :]
        bw_i, bw_j = bw[:, None], bw[None, :]
        bh_i, bh_j = bh[:, None], bh[None, :]
        cy_i, cy_j = cy[:, None], cy[None, :]
        small_i, small_j = small_area_score[:, None], small_area_score[None, :]
        max_h = torch.maximum(bh_i, bh_j).clamp_min(1.0)
        min_h = torch.minimum(bh_i, bh_j).clamp_min(1.0)
        mean_w = 0.5 * (bw_i + bw_j)

        x_gap = torch.maximum(torch.maximum(x1_i, x1_j) - torch.minimum(x2_i, x2_j), torch.zeros((n, n), device=device, dtype=dtype))
        overlap_y = (torch.minimum(y2_i, y2_j) - torch.maximum(y1_i, y1_j) + 1.0).clamp_min(0.0)
        dy_norm = torch.abs(cy_i - cy_j) / (max_h + 1e-6)
        height_ratio = (min_h / (max_h + 1e-6)).clamp(0.0, 1.0)
        vertical_overlap_ratio = (overlap_y / (min_h + 1e-6)).clamp(0.0, 1.0)
        merged_x1 = torch.minimum(x1_i, x1_j)
        merged_y1 = torch.minimum(y1_i, y1_j)
        merged_x2 = torch.maximum(x2_i, x2_j)
        merged_y2 = torch.maximum(y2_i, y2_j)
        merged_w = (merged_x2 - merged_x1 + 1.0).clamp_min(1.0)
        merged_h = (merged_y2 - merged_y1 + 1.0).clamp_min(1.0)
        merged_aspect = merged_w / (merged_h + 1e-6)
        merged_cx_norm = ((merged_x1 + merged_x2) * 0.5 / max(float(W - 1), 1.0)).clamp(0.0, 1.0)
        merged_cy_norm = ((merged_y1 + merged_y2) * 0.5 / max(float(H - 1), 1.0)).clamp(0.0, 1.0)
        merged_cx_idx = (merged_cx_norm * max(float(W - 1), 1.0)).round().to(torch.long).clamp(0, W - 1)
        merged_cy_idx = (merged_cy_norm * max(float(H - 1), 1.0)).round().to(torch.long).clamp(0, H - 1)
        motion_prior = motion_prior_map[0, 0, merged_cy_idx, merged_cx_idx].clamp(0.0, 1.0)

        upper = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), diagonal=1)
        same_row = (
            upper
            & (dy_norm < 0.45)
            & (height_ratio > 0.45)
            & (vertical_overlap_ratio > 0.30)
            & (x_gap < 4.0 * max_h)
            & (merged_aspect >= 2.0)
            & (merged_aspect <= 9.5)
            & (motion_prior > 0.25)
        )
        pair_i, pair_j = torch.where(same_row)
        if pair_i.numel() == 0:
            return source, {
                "stroke_merge_source_map": source,
                "stroke_components_map": component_map_all,
                "motion_prior_map": motion_prior_map,
                "same_row_merge_map": zero_map,
                "stroke_merge_morphology_before": source,
                "stroke_merge_morphology_after": source,
                "stroke_merge_pair_boxes": zero4,
                "stroke_merge_count": zero1,
            }

        pair_merged_x1 = merged_x1[pair_i, pair_j]
        pair_merged_y1 = merged_y1[pair_i, pair_j]
        pair_merged_x2 = merged_x2[pair_i, pair_j]
        pair_merged_y2 = merged_y2[pair_i, pair_j]
        pair_merged_h = merged_h[pair_i, pair_j].clamp_min(1.0)
        inside_x = (cx.view(1, -1) >= pair_merged_x1.view(-1, 1)) & (cx.view(1, -1) <= pair_merged_x2.view(-1, 1))
        inside_y = torch.abs(cy.view(1, -1) - ((pair_merged_y1 + pair_merged_y2) * 0.5).view(-1, 1)) <= (pair_merged_h * 0.5).view(-1, 1)
        repeat_count = (inside_x & inside_y).to(dtype).sum(dim=1).clamp_min(2.0)
        stroke_repeat_score = (repeat_count / 4.0).clamp(0.0, 1.0)
        same_row_alignment = torch.exp(-dy_norm[pair_i, pair_j] / 0.30).clamp(0.0, 1.0)
        x_repeat_score = (0.60 * (torch.minimum(bw_i, bw_j) / (torch.maximum(bw_i, bw_j) + 1e-6))[pair_i, pair_j] + 0.40 * torch.exp(-torch.abs(x_gap[pair_i, pair_j] / (mean_w[pair_i, pair_j] + 1e-6) - 2.0) / 2.5)).clamp(0.0, 1.0)
        stroke_repeat_score = torch.maximum(stroke_repeat_score, x_repeat_score * 0.50)
        aspect_score = torch.exp(-torch.abs(merged_aspect[pair_i, pair_j] - 4.0) / 1.5).clamp(0.0, 1.0)
        small_component_score = (0.5 * (small_i[pair_i, pair_j] + small_j[pair_i, pair_j]) * stroke_repeat_score).clamp(0.0, 1.0)
        pair_score = (
            0.40 * motion_prior[pair_i, pair_j]
            + 0.25 * same_row_alignment
            + 0.15 * stroke_repeat_score
            + 0.10 * aspect_score
            + 0.10 * small_component_score
        ).clamp(0.0, 1.0)

        keep_pair_count = min(int(max_pairs), int(pair_score.numel()))
        if keep_pair_count <= 0:
            return source, debug_empty
        order = torch.topk(pair_score, k=keep_pair_count).indices
        pair_i = pair_i[order]
        pair_j = pair_j[order]
        pair_score = pair_score[order]

        bridge = torch.zeros_like(source)
        same_row_merge_map = torch.zeros_like(source)
        pair_boxes = torch.zeros((keep_pair_count, 4), device=device, dtype=dtype)
        merge_count = 0
        for idx_pair in range(keep_pair_count):
            i = pair_i[idx_pair]
            j = pair_j[idx_pair]
            yy1 = torch.maximum(y1[i], y1[j]).round().to(torch.long).clamp(0, H - 1)
            yy2 = torch.minimum(y2[i], y2[j]).round().to(torch.long).clamp(0, H - 1)
            left_x2 = torch.minimum(x2[i], x2[j]).round().to(torch.long).clamp(0, W - 1)
            right_x1 = torch.maximum(x1[i], x1[j]).round().to(torch.long).clamp(0, W - 1)
            xx1 = (left_x2 + 1).clamp(0, W - 1)
            xx2 = (right_x1 - 1).clamp(0, W - 1)
            if bool((yy2 >= yy1) & (xx2 >= xx1)):
                bridge[:, :, yy1:yy2 + 1, xx1:xx2 + 1] = 1.0
                same_row_merge_map[:, :, yy1:yy2 + 1, xx1:xx2 + 1] = torch.maximum(same_row_merge_map[:, :, yy1:yy2 + 1, xx1:xx2 + 1], pair_score[idx_pair])
                pair_boxes[idx_pair] = torch.stack([xx1.to(dtype), yy1.to(dtype), xx2.to(dtype), yy2.to(dtype)])
                merge_count += 1

        if merge_count == 0:
            return source, {
                "stroke_merge_source_map": source,
                "stroke_components_map": component_map_all,
                "motion_prior_map": motion_prior_map,
                "same_row_merge_map": zero_map,
                "stroke_merge_morphology_before": source,
                "stroke_merge_morphology_after": source,
                "stroke_merge_pair_boxes": zero4,
                "stroke_merge_count": zero1,
            }

        merged = torch.maximum(source, bridge).clamp(0.0, 1.0)
        return merged, {
            "stroke_merge_source_map": source,
            "stroke_components_map": component_map_all,
            "motion_prior_map": motion_prior_map,
            "same_row_merge_map": same_row_merge_map,
            "stroke_merge_morphology_before": source,
            "stroke_merge_morphology_after": merged,
            "stroke_merge_pair_boxes": pair_boxes[:merge_count],
            "stroke_merge_count": torch.tensor([float(merge_count)], device=device, dtype=dtype),
        }

    def _search_to_gate(self, box_search: torch.Tensor, sx1: torch.Tensor, sy1: torch.Tensor) -> torch.Tensor:
        # temporal vehicle ROI gate + fuzzy region plate candidates + MLP selector
        offset = torch.stack([sx1, sy1, sx1, sy1], dim=-1).to(device=box_search.device, dtype=box_search.dtype)
        return box_search + offset

    def _gate_to_full(self, box_roi: torch.Tensor, gate_offset: torch.Tensor) -> torch.Tensor:
        # temporal vehicle ROI gate + fuzzy region plate candidates + MLP selector
        return box_roi + gate_offset.to(device=box_roi.device, dtype=box_roi.dtype)

    def _propose_plate_candidates_vehicle_gate_gpu(
        self,
        fuzzy_region: torch.Tensor,
        fuzzy_temporal: torch.Tensor,
        value_map: torch.Tensor | None = None,
        gx_map: torch.Tensor | None = None,
        gy_map: torch.Tensor | None = None,
        gate_offset_xy: tuple[int, int] = (0, 0),
        gt_box_full: torch.Tensor | None = None,
        pre_topk: int = 50,
        final_topk: int = 1,
    ) -> torch.Tensor:
        # gray * uniformity * blackhat candidate source experiment -> ScharrX/morphology/CCL pipeline.
        # B-mode horizontal band experiment is toggled in detection/gpu_preprocessing.py via DEBUG_EXPERIMENT_SCHARR_BAND.
        self._assert_cuda(fuzzy_region, "fuzzy_region")
        self._assert_cuda(fuzzy_temporal, "fuzzy_temporal")
        if fuzzy_region.ndim != 4 or fuzzy_region.shape[1] != 1:
            raise RuntimeError("fuzzy_region must be [B,1,H,W]")
        if fuzzy_temporal.shape != fuzzy_region.shape:
            raise RuntimeError("fuzzy_temporal shape must match fuzzy_region")

        region = torch.nan_to_num(fuzzy_region, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        temporal = torch.nan_to_num(fuzzy_temporal, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        B, _, H, W = region.shape
        device = region.device
        dtype = region.dtype
        if B != 1:
            raise RuntimeError("GPU Canny/ScharrX candidate path currently expects batch size 1")

        zero4 = torch.zeros((0, 4), device=device, dtype=dtype)
        zero6 = torch.zeros((0, 6), device=device, dtype=dtype)
        zero_map = torch.zeros_like(region)
        gate_offset = torch.tensor([gate_offset_xy[0], gate_offset_xy[1], gate_offset_xy[0], gate_offset_xy[1]], device=device, dtype=dtype).view(1, 4)

        gray_map = getattr(gpu_pre_mod, "LAST_GRAY", None)
        if gray_map is None:
            gray_map = value_map if value_map is not None else region
        gray_map = torch.nan_to_num(gray_map.to(device=device, dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if gray_map.shape != region.shape:
            gray_map = F.interpolate(gray_map, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        fuzzy_stretched_map = getattr(gpu_pre_mod, "LAST_FUZZY_STRETCHED", None)
        if fuzzy_stretched_map is None:
            fuzzy_stretched_map = gray_map
        fuzzy_stretched_map = torch.nan_to_num(fuzzy_stretched_map.to(device=device, dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if fuzzy_stretched_map.shape != region.shape:
            fuzzy_stretched_map = F.interpolate(fuzzy_stretched_map, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        candidate_source = getattr(gpu_pre_mod, "LAST_CANDIDATE_SOURCE", None)
        if candidate_source is None:
            candidate_source = getattr(gpu_pre_mod, "LAST_EVIDENCE", None)
        if candidate_source is None:
            candidate_source = getattr(gpu_pre_mod, "LAST_ENHANCED_GRAY_UNIFORM", None)
        if candidate_source is None:
            candidate_source = fuzzy_stretched_map
        candidate_source = torch.nan_to_num(candidate_source.to(device=device, dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if candidate_source.shape != region.shape:
            candidate_source = F.interpolate(candidate_source, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        enhanced_gray = candidate_source

        uniform_mu = getattr(gpu_pre_mod, "LAST_UNIFORM_MU", None)
        if uniform_mu is None:
            uniform_mu = getattr(gpu_pre_mod, "LAST_UNIFORMITY_SCORE", None)
        if uniform_mu is None:
            uniform_mu = torch.ones_like(region)
        uniform_mu = torch.nan_to_num(uniform_mu.to(device=device, dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if uniform_mu.shape != region.shape:
            uniform_mu = F.interpolate(uniform_mu, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        canny = getattr(gpu_pre_mod, "LAST_CANNY", None)
        scharrx = getattr(gpu_pre_mod, "LAST_SCHARRX", None)
        if scharrx is None:
            scharrx = getattr(gpu_pre_mod, "LAST_SOBELX", None)
        scharr_band = getattr(gpu_pre_mod, "LAST_SCHARR_BAND", None)
        blackhat = getattr(gpu_pre_mod, "LAST_BLACKHAT", None)
        blackhat_weight = getattr(gpu_pre_mod, "LAST_BLACKHAT_WEIGHT", None)
        local_contrast = getattr(gpu_pre_mod, "LAST_LOCAL_CONTRAST", None)
        evidence = getattr(gpu_pre_mod, "LAST_EVIDENCE", None)
        binary = getattr(gpu_pre_mod, "LAST_BINARY", None)
        if canny is None:
            canny = zero_map
        if scharrx is None:
            _, scharrx = gpu_pre_mod.scharr_x_response(candidate_source)
        scharr_band_enabled = bool(getattr(gpu_pre_mod, "DEBUG_EXPERIMENT_SCHARR_BAND", False))
        if scharr_band_enabled and scharr_band is None:
            scharr_band = gpu_pre_mod.scharr_horizontal_band(scharrx)
        if blackhat is None:
            close_gray = gpu_pre_mod._morph_close(fuzzy_stretched_map, k=15)
            blackhat = (close_gray - fuzzy_stretched_map).clamp_min(0.0)
            blackhat = (blackhat / (blackhat.amax(dim=(-2, -1), keepdim=True) + 1e-6)).clamp(0.0, 1.0)
        if blackhat_weight is None:
            blackhat_alpha_value = float(getattr(gpu_pre_mod, "LAST_BLACKHAT_ALPHA", 0.0000008))
            blackhat_weight = torch.sigmoid((blackhat - blackhat_alpha_value) * 10.0).clamp(0.0, 1.0)
        if local_contrast is None:
            local_contrast = torch.zeros_like(candidate_source)
        if evidence is None:
            evidence = candidate_source
        if binary is None:
            edge = (scharr_band if scharr_band_enabled and scharr_band is not None else scharrx).to(device=device, dtype=dtype).clamp(0.0, 1.0)
            binary = (edge >= (edge.mean(dim=(-2, -1), keepdim=True) + 0.35 * edge.std(dim=(-2, -1), keepdim=True))).to(dtype)
        canny = canny.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        scharrx = scharrx.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if scharr_band is None:
            scharr_band = zero_map
        scharr_band = scharr_band.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        blackhat = blackhat.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        blackhat_weight = blackhat_weight.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        local_contrast = local_contrast.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        evidence = evidence.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        binary = binary.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if canny.shape != region.shape:
            canny = F.interpolate(canny, size=(H, W), mode="nearest").clamp(0.0, 1.0)
        if scharrx.shape != region.shape:
            scharrx = F.interpolate(scharrx, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if scharr_band.shape != region.shape:
            scharr_band = F.interpolate(scharr_band, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if blackhat.shape != region.shape:
            blackhat = F.interpolate(blackhat, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if blackhat_weight.shape != region.shape:
            blackhat_weight = F.interpolate(blackhat_weight, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if local_contrast.shape != region.shape:
            local_contrast = F.interpolate(local_contrast, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if evidence.shape != region.shape:
            evidence = F.interpolate(evidence, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if binary.shape != region.shape:
            binary = F.interpolate(binary, size=(H, W), mode="nearest").clamp(0.0, 1.0)

        gx_feat = gx_map.to(device=device, dtype=dtype) if gx_map is not None else zero_map
        gy_feat = gy_map.to(device=device, dtype=dtype) if gy_map is not None else zero_map
        theta = torch.atan2(gy_feat, gx_feat + 1e-6)
        morph_before_merge = orientation_morphology(binary, theta)
        coherence = evidence.clamp(0.0, 1.0)
        morph, stroke_merge_debug = self._horizontal_same_row_stroke_merge(morph_before_merge, coherence)
        label_map, attrs = ccl_label_propagation(morph, coherence)
        attrs = attrs.to(device=device, dtype=dtype)

        raw_boxes = gpu_ccl_mod.LAST_CCL_DEBUG.get("raw_boxes", zero4).to(device=device, dtype=dtype)
        filtered_boxes = gpu_ccl_mod.LAST_CCL_DEBUG.get("filtered_boxes", zero4).to(device=device, dtype=dtype)
        merge_char_boxes = gpu_ccl_mod.LAST_CCL_DEBUG.get("merge_char_boxes", zero4).to(device=device, dtype=dtype)
        merge_pair_boxes = gpu_ccl_mod.LAST_CCL_DEBUG.get("merge_pair_boxes", zero4).to(device=device, dtype=dtype)
        merged_boxes = gpu_ccl_mod.LAST_CCL_DEBUG.get("merged_boxes", zero4).to(device=device, dtype=dtype)

        uniform_sigma_sq = torch.tensor([float(getattr(gpu_pre_mod, "LAST_UNIFORMITY_SIGMA_SQ", 0.00009))], device=device, dtype=dtype)
        uniform_weight = torch.tensor([float(getattr(gpu_pre_mod, "LAST_UNIFORM_WEIGHT", 0.30))], device=device, dtype=dtype)
        blackhat_alpha = torch.tensor([float(getattr(gpu_pre_mod, "LAST_BLACKHAT_ALPHA", 0.0000008))], device=device, dtype=dtype)
        candidate_source_mode_id = torch.tensor([float(getattr(gpu_pre_mod, "LAST_CANDIDATE_SOURCE_MODE_ID", 0))], device=device, dtype=dtype)
        default_candidate_source_mode = getattr(gpu_pre_mod, "CANDIDATE_SOURCE_BLEND_NAME", "uniform_blackhat_alpha_cut")
        candidate_source_mode = str(getattr(gpu_pre_mod, "LAST_CANDIDATE_SOURCE_MODE_NAME", default_candidate_source_mode)).strip()
        if not candidate_source_mode:
            candidate_source_mode = default_candidate_source_mode
        uniform_mu_mean = uniform_mu.mean().reshape(1)
        uniform_mu_max = uniform_mu.amax().reshape(1)
        blackhat_mean = blackhat.mean().reshape(1)
        blackhat_max = blackhat.amax().reshape(1)
        blackhat_weight_mean = blackhat_weight.mean().reshape(1)
        blackhat_weight_max = blackhat_weight.amax().reshape(1)
        candidate_source_mean = candidate_source.mean().reshape(1)
        candidate_source_max = candidate_source.amax().reshape(1)
        scharr_mean = scharrx.mean().reshape(1)
        scharr_max = scharrx.amax().reshape(1)

        gt_box_roi = zero4
        if gt_box_full is not None:
            gt = gt_box_full.to(device=device, dtype=dtype).reshape(-1, 4)
            if gt.numel() > 0:
                gt_box_roi = (gt[:1] - gate_offset).clamp_min(0.0)
                gt_box_roi[:, 0::2] = gt_box_roi[:, 0::2].clamp(0.0, float(max(W - 1, 0)))
                gt_box_roi[:, 1::2] = gt_box_roi[:, 1::2].clamp(0.0, float(max(H - 1, 0)))

        if attrs.numel() == 0:
            self._stage6_debug = {
                "stage6_raw_component_count": float(raw_boxes.shape[0]),
                "stage6_filter_pass_count": 0.0,
                "stage6_nms_post_count": 0.0,
                "stage6_topk_count": 0.0,
                "stage6_best_score": torch.tensor(0.0, device=device, dtype=dtype),
                "score_pre_boxes": raw_boxes,
                "nms_boxes": zero4,
                "final_boxes": zero4,
                "merged_score_pre_boxes": filtered_boxes,
                "rect_response_raw": coherence,
                "rect_response": morph,
                "rect_peaks": binary,
                "rect_candidates": morph,
                "final_candidates": zero_map,
                "gray_map": gray_map,
                "fuzzy_stretched_map": fuzzy_stretched_map,
                "uniform_mu": uniform_mu,
                "enhanced_gray_uniform": candidate_source,
                "candidate_source_map": candidate_source,
                "canny_map": canny,
                "scharrx_map": scharrx,
                "scharr_band_map": scharr_band if scharr_band_enabled else zero_map,
                "sobelx_map": scharrx,
                "blackhat_map": blackhat,
                "blackhat_weight_map": blackhat_weight,
                "local_contrast_map": local_contrast,
                "parallel_evidence_map": evidence,
                "binary_map": binary,
                "morphology_map": morph,
                "morphology_before_merge_map": morph_before_merge,
                "stroke_components_map": stroke_merge_debug["stroke_components_map"],
                "motion_prior_map": stroke_merge_debug["motion_prior_map"],
                "same_row_merge_map": stroke_merge_debug["same_row_merge_map"],
                "stroke_merge_pair_boxes": stroke_merge_debug["stroke_merge_pair_boxes"],
                "stroke_merge_count": stroke_merge_debug["stroke_merge_count"],
                "stroke_merge_mode": "same_row_horizontal_merge",
                "candidate_count": torch.tensor([0.0], device=device, dtype=dtype),
                "top_candidate_score": torch.tensor([0.0], device=device, dtype=dtype),
                "uniform_sigma_sq": uniform_sigma_sq,
                "uniform_weight": uniform_weight,
                "blackhat_alpha": blackhat_alpha,
                "candidate_source_mode": candidate_source_mode,
                "candidate_source_mode_id": candidate_source_mode_id,
                "uniform_mu_mean": uniform_mu_mean,
                "uniform_mu_max": uniform_mu_max,
                "blackhat_mean": blackhat_mean,
                "blackhat_max": blackhat_max,
                "blackhat_weight_mean": blackhat_weight_mean,
                "blackhat_weight_max": blackhat_weight_max,
                "candidate_source_mean": candidate_source_mean,
                "candidate_source_max": candidate_source_max,
                "scharr_mean": scharr_mean,
                "scharr_max": scharr_max,
                "plate_candidate_boxes_roi": zero4,
                "plate_candidate_boxes_full": zero4,
                "plate_selected_box_roi": zero4,
                "plate_selected_box_full": zero4,
                "candidate_final_score": torch.zeros((0,), device=device, dtype=dtype),
                "candidate_mlp_score": torch.zeros((0,), device=device, dtype=dtype),
                "candidate_rule_score": torch.zeros((0,), device=device, dtype=dtype),
                "candidate_peak_score": torch.zeros((0,), device=device, dtype=dtype),
                "selected_candidate_idx": torch.zeros((0,), device=device, dtype=torch.long),
                "direct_crop_gt_box_roi": gt_box_roi,
                "direct_crop_iou_pred_gt": torch.zeros((0,), device=device, dtype=dtype),
                "direct_crop_score_map": coherence,
                "direct_crop_peak": morph,
                "direct_crop_pred_mask": zero_map,
                "direct_crop_crop_box_roi": zero4,
                "direct_crop_crop_box_full": zero4,
                "direct_crop_peak_xy_roi": torch.zeros((0, 2), device=device, dtype=dtype),
            }
            return zero6

        if attrs.numel() > 0:
            boxes = attrs[:, :4].clamp_min(0.0)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(max(W - 1, 0)))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(max(H - 1, 0)))
            area = attrs[:, 4].clamp_min(1.0)
            fill_ratio = attrs[:, 5].clamp(0.0, 1.0)
            aspect = attrs[:, 6].clamp_min(1e-6)
            mean_coh = attrs[:, 7].clamp(0.0, 1.0)
            white_plate_score = attrs[:, 9].clamp(0.0, 1.0)
            ghost_score = attrs[:, 10].clamp(0.0, 1.0)
            stroke_density = attrs[:, 11].clamp(0.0, 1.0)
            center_prior = attrs[:, 12].clamp(0.0, 1.0)
            overconnect_risk = attrs[:, 13].clamp(0.0, 1.0)
            area_ratio = attrs[:, 14].clamp(0.0, 1.0)
            cy_norm = attrs[:, 16].clamp(0.0, 1.0)
            inner_brightness = attrs[:, 17].clamp(0.0, 1.0)
            aspect_score = torch.exp(-torch.abs(aspect - 4.0) / 2.2).clamp(0.0, 1.0)
            lower_prior = torch.exp(-torch.abs(cy_norm - 0.72) / 0.25).clamp(0.0, 1.0)
            rule_score = (
                0.30 * white_plate_score
                + 0.20 * ghost_score
                + 0.15 * stroke_density
                + 0.15 * center_prior
                + 0.10 * aspect_score
                + 0.10 * mean_coh
                - 0.20 * overconnect_risk
            ).clamp(0.0, 1.0)
        else:
            boxes = zero4
            area = zero1
            fill_ratio = zero1
            aspect = zero1
            mean_coh = zero1
            stroke_density = zero1
            center_prior = zero1
            area_ratio = zero1
            cy_norm = zero1
            inner_brightness = zero1
            aspect_score = zero1
            lower_prior = zero1
            rule_score = zero1

        candidate_type_id = torch.zeros_like(rule_score)

        final_score = rule_score
        pre_k = min(max(int(pre_topk), 1), int(final_score.numel()))
        top_scores, top_idx = torch.topk(final_score, k=pre_k)
        boxes = boxes[top_idx]
        area = area[top_idx]
        fill_ratio = fill_ratio[top_idx]
        aspect = aspect[top_idx]
        mean_coh = mean_coh[top_idx]
        stroke_density = stroke_density[top_idx]
        lower_prior = lower_prior[top_idx]
        aspect_score = aspect_score[top_idx]
        area_ratio = area_ratio[top_idx]
        inner_brightness = inner_brightness[top_idx]
        rule_score = rule_score[top_idx]
        candidate_type_id = candidate_type_id[top_idx]
        final_score = top_scores
        crop_box_full = self._gate_to_full(boxes, gate_offset)
        out_k = min(max(int(final_topk), 1), int(final_score.numel()))
        selected_score, selected_order = torch.topk(final_score, k=out_k)
        selected_boxes = boxes[selected_order]
        selected_full = crop_box_full[selected_order]
        candidates = torch.cat([selected_boxes, selected_score.unsqueeze(1), torch.zeros((out_k, 1), device=device, dtype=dtype)], dim=1)

        peak_xy_roi = torch.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=1)
        final_candidates = torch.zeros_like(region)
        if out_k > 0 and selected_score[0] > 0:
            b = selected_boxes[0].round().to(torch.long)
            final_candidates[:, :, b[1]:b[3] + 1, b[0]:b[2] + 1] = enhanced_gray[:, :, b[1]:b[3] + 1, b[0]:b[2] + 1]

        iou_pred_gt = torch.zeros((0,), device=device, dtype=dtype)
        if gt_box_roi.numel() > 0 and selected_boxes.numel() > 0:
            gt = gt_box_roi[:1].expand(selected_boxes.shape[0], 4)
            ix1 = torch.maximum(selected_boxes[:, 0], gt[:, 0])
            iy1 = torch.maximum(selected_boxes[:, 1], gt[:, 1])
            ix2 = torch.minimum(selected_boxes[:, 2], gt[:, 2])
            iy2 = torch.minimum(selected_boxes[:, 3], gt[:, 3])
            inter = (ix2 - ix1 + 1.0).clamp_min(0.0) * (iy2 - iy1 + 1.0).clamp_min(0.0)
            pa = ((selected_boxes[:, 2] - selected_boxes[:, 0] + 1.0).clamp_min(1.0) * (selected_boxes[:, 3] - selected_boxes[:, 1] + 1.0).clamp_min(1.0))
            ga = ((gt[:, 2] - gt[:, 0] + 1.0).clamp_min(1.0) * (gt[:, 3] - gt[:, 1] + 1.0).clamp_min(1.0))
            iou_pred_gt = inter / (pa + ga - inter + 1e-6)

        bw = (boxes[:, 2] - boxes[:, 0] + 1.0).clamp_min(1.0)
        bh = (boxes[:, 3] - boxes[:, 1] + 1.0).clamp_min(1.0)
        score_map = morph * coherence
        peak_map = torch.zeros_like(region)
        if peak_xy_roi.numel() > 0:
            px = peak_xy_roi[:, 0].round().to(torch.long).clamp(0, max(W - 1, 0))
            py = peak_xy_roi[:, 1].round().to(torch.long).clamp(0, max(H - 1, 0))
            peak_map[0, 0, py, px] = final_score.clamp(0.0, 1.0)

        self._stage6_debug = {
            "stage6_raw_component_count": float(raw_boxes.shape[0]),
            "stage6_filter_pass_count": float(boxes.shape[0]),
            "stage6_nms_post_count": float(out_k),
            "stage6_topk_count": float(out_k),
            "stage6_best_score": selected_score[0].detach() if out_k > 0 else torch.tensor(0.0, device=device, dtype=dtype),
            "stage6_best_area_ratio": area_ratio[selected_order[0]].detach() if out_k > 0 else torch.tensor(0.0, device=device, dtype=dtype),
            "stage6_best_aspect": aspect[selected_order[0]].detach() if out_k > 0 else torch.tensor(0.0, device=device, dtype=dtype),
            "stage6_best_fill": fill_ratio[selected_order[0]].detach() if out_k > 0 else torch.tensor(0.0, device=device, dtype=dtype),
            "stage6_best_cy_norm": cy_norm[top_idx][selected_order[0]].detach() if out_k > 0 else torch.tensor(0.0, device=device, dtype=dtype),
            "stage6_best_inner_bright": inner_brightness[selected_order[0]].detach() if out_k > 0 else torch.tensor(0.0, device=device, dtype=dtype),
            "score_pre_boxes": boxes,
            "nms_boxes": selected_boxes,
            "final_boxes": selected_boxes,
            "merged_score_pre_boxes": filtered_boxes,
            "raw_candidate_boxes": raw_boxes,
            "merge_char_boxes": merge_char_boxes,
            "merge_pair_boxes": merge_pair_boxes,
            "merged_boxes": merged_boxes,
            "rect_response_raw": coherence,
            "rect_response": score_map,
            "rect_peaks": peak_map,
            "rect_candidates": morph,
            "center_inner_midscale": canny,
            "ring_mean_midscale": scharrx,
            "vertical_closure_midscale": binary,
            "final_candidates": final_candidates,
            "gray_map": gray_map,
            "fuzzy_stretched_map": fuzzy_stretched_map,
            "uniform_mu": uniform_mu,
            "enhanced_gray_uniform": candidate_source,
            "candidate_source_map": candidate_source,
            "canny_map": canny,
            "scharrx_map": scharrx,
            "scharr_band_map": scharr_band if scharr_band_enabled else zero_map,
            "sobelx_map": scharrx,
            "blackhat_map": blackhat,
            "blackhat_weight_map": blackhat_weight,
            "local_contrast_map": local_contrast,
            "parallel_evidence_map": evidence,
            "binary_map": binary,
            "morphology_map": morph,
            "morphology_before_merge_map": morph_before_merge,
            "stroke_components_map": stroke_merge_debug["stroke_components_map"],
            "motion_prior_map": stroke_merge_debug["motion_prior_map"],
            "same_row_merge_map": stroke_merge_debug["same_row_merge_map"],
            "stroke_merge_pair_boxes": stroke_merge_debug["stroke_merge_pair_boxes"],
            "stroke_merge_count": stroke_merge_debug["stroke_merge_count"],
            "stroke_merge_mode": "same_row_horizontal_merge",
            "candidate_count": torch.tensor([float(boxes.shape[0])], device=device, dtype=dtype),
            "top_candidate_score": selected_score[:1].detach() if out_k > 0 else torch.tensor([0.0], device=device, dtype=dtype),
            "uniform_sigma_sq": uniform_sigma_sq,
            "uniform_weight": uniform_weight,
            "blackhat_alpha": blackhat_alpha,
            "candidate_source_mode": candidate_source_mode,
            "candidate_source_mode_id": candidate_source_mode_id,
            "uniform_mu_mean": uniform_mu_mean,
            "uniform_mu_max": uniform_mu_max,
            "blackhat_mean": blackhat_mean,
            "blackhat_max": blackhat_max,
            "blackhat_weight_mean": blackhat_weight_mean,
            "blackhat_weight_max": blackhat_weight_max,
            "candidate_source_mean": candidate_source_mean,
            "candidate_source_max": candidate_source_max,
            "scharr_mean": scharr_mean,
            "scharr_max": scharr_max,
            "plate_score_map": score_map,
            "plate_peak_score_map": peak_map,
            "plate_candidate_boxes_roi": boxes,
            "plate_candidate_boxes_full": crop_box_full,
            "plate_selected_box_roi": selected_boxes,
            "plate_selected_box_full": selected_full,
            "peak_xy_search": peak_xy_roi,
            "crop_box_search": boxes,
            "crop_box_roi": boxes,
            "crop_box_full": crop_box_full,
            "candidate_final_score": final_score,
            "selected_candidate_idx": selected_order.detach(),
            "candidate_peak_score": final_score,
            "candidate_region_mean": inner_brightness,
            "candidate_temporal_mean": mean_coh,
            "candidate_region_contrast": stroke_density,
            "candidate_lower_prior": lower_prior,
            "candidate_diagonal_prior": center_prior[top_idx],
            "candidate_width": bw,
            "candidate_height": bh,
            "candidate_aspect": aspect,
            "candidate_area_ratio": area_ratio,
            "candidate_aspect_score": aspect_score,
            "candidate_size_score": (1.0 - area_ratio / 0.04).clamp(0.0, 1.0),
            "candidate_rectangularity_score": aspect_score,
            "candidate_type_id": candidate_type_id,
            "candidate_mlp_score": final_score,
            "candidate_rule_score": rule_score,
            "candidate_mlp_fallback": torch.ones((1,), device=device, dtype=torch.bool),
            "candidate_gray_included": torch.ones((1,), device=device, dtype=dtype),
            "best_scale_idx": torch.zeros_like(final_score, dtype=torch.long),
            "best_aspect_idx": torch.zeros_like(final_score, dtype=torch.long),
            "peak_xy_roi": peak_xy_roi,
            "top_peak_xy_search": peak_xy_roi,
            "top_peak_score": final_score,
            "vehicle_area": torch.tensor([float(max(H * W, 1))], device=device, dtype=dtype),
            "vehicle_center": torch.tensor([[float(W - 1) * 0.5, float(H - 1) * 0.5]], device=device, dtype=dtype),
            "vehicle_roi_mean": enhanced_gray.mean(dim=(-2, -1)).reshape(-1),
            "vehicle_roi_max": enhanced_gray.amax(dim=(-2, -1)).reshape(-1),
            "plate_search_wh": torch.tensor([[float(W), float(H)]], device=device, dtype=dtype),
            "direct_crop_gt_box_roi": gt_box_roi,
            "direct_crop_iou_pred_gt": iou_pred_gt,
            "direct_crop_score_map": score_map,
            "direct_crop_peak": peak_map,
            "direct_crop_pred_mask": final_candidates,
            "direct_crop_crop_box_roi": selected_boxes,
            "direct_crop_crop_box_full": selected_full,
            "direct_crop_peak_xy_roi": peak_xy_roi[selected_order] if out_k > 0 else torch.zeros((0, 2), device=device, dtype=dtype),
            "direct_crop_best_scale_idx": torch.zeros((out_k,), device=device, dtype=torch.long),
            "direct_crop_peak_score": selected_score,
        }
        return candidates

    def _find_plate_by_region_avg_pool(
        self,
        region_evidence: torch.Tensor,
        roi_box_full=None,
        gt_box_full: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # fuzzy region direct plate crop gpu-only
        self._assert_cuda(region_evidence, "region_evidence")
        if region_evidence.ndim != 4 or region_evidence.shape[1] != 1:
            raise RuntimeError("region_evidence must be [B,1,H,W]")

        E = torch.nan_to_num(region_evidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        B, _, H, W = E.shape
        device = E.device
        dtype = E.dtype
        plate_score_thr = 0.20
        height_scales = (0.035, 0.045, 0.055, 0.070)
        aspect_candidates = (3.2, 4.0, 4.8)
        min_h = 7
        min_w = 15

        yy = torch.arange(H, device=device).view(1, 1, H, 1)
        xx = torch.arange(W, device=device).view(1, 1, 1, W)
        neg_inf = torch.full((B, 1, H, W), -torch.inf, device=device, dtype=dtype)
        scores = []
        scale_hw = []

        for height_scale in height_scales:
            h = max(min_h, int(H * height_scale))
            if h % 2 == 0:
                h += 1
            for aspect in aspect_candidates:
                w = max(min_w, int(h * aspect))
                if w % 2 == 0:
                    w += 1
                if h >= H or w >= W:
                    continue

                half_h = h // 2
                half_w = w // 2
                score_s = F.avg_pool2d(
                    E,
                    kernel_size=(h, w),
                    stride=1,
                    padding=(half_h, half_w),
                    count_include_pad=False,
                )
                valid = (xx >= half_w) & (xx < (W - half_w)) & (yy >= half_h) & (yy < (H - half_h))
                scores.append(torch.where(valid, score_s, neg_inf))
                scale_hw.append((h, w))

        zero_boxes = torch.zeros((B, 4), device=device, dtype=dtype)
        zero_xy = torch.zeros((B, 2), device=device, dtype=dtype)
        zero_score = torch.zeros((B,), device=device, dtype=dtype)
        zero_scale = torch.full((B,), -1, device=device, dtype=torch.long)
        zero_map = torch.zeros_like(E)

        if len(scores) == 0:
            debug = {
                "direct_crop_score_map": zero_map,
                "direct_crop_peak": zero_map,
                "direct_crop_pred_mask": zero_map,
                "direct_crop_crop_box_roi": zero_boxes,
                "direct_crop_crop_box_full": zero_boxes,
                "direct_crop_peak_xy_roi": zero_xy,
                "direct_crop_best_scale_idx": zero_scale,
                "direct_crop_h_best": zero_score,
                "direct_crop_w_best": zero_score,
                "direct_crop_peak_score": zero_score,
                "direct_crop_gt_box_roi": zero_boxes,
                "direct_crop_iou_pred_gt": zero_score,
            }
            return zero_boxes, zero_xy, zero_score, zero_scale, debug

        stacked_scores = torch.stack(scores, dim=0)
        score_map, best_scale_idx_map = stacked_scores.max(dim=0)
        flat = score_map.flatten(start_dim=1)
        peak_score, peak_flat = flat.max(dim=1)
        peak_hw = peak_flat % (H * W)
        peak_y = torch.div(peak_hw, W, rounding_mode="floor").to(torch.long)
        peak_x = (peak_hw - peak_y * W).to(torch.long)
        batch_idx = torch.arange(B, device=device)
        best_s = best_scale_idx_map[batch_idx, 0, peak_y, peak_x].to(torch.long)

        scale_table = torch.tensor(scale_hw, device=device, dtype=dtype)
        hw_best = scale_table[best_s.clamp(min=0)]
        h_best = hw_best[:, 0]
        w_best = hw_best[:, 1]
        half_h_best = torch.div(h_best.to(torch.long), 2, rounding_mode="floor")
        half_w_best = torch.div(w_best.to(torch.long), 2, rounding_mode="floor")

        x1 = (peak_x - half_w_best).clamp(0, max(W - 1, 0))
        y1 = (peak_y - half_h_best).clamp(0, max(H - 1, 0))
        x2 = (peak_x + half_w_best).clamp(0, max(W - 1, 0))
        y2 = (peak_y + half_h_best).clamp(0, max(H - 1, 0))
        valid_peak = torch.isfinite(peak_score) & (peak_score >= plate_score_thr) & (x2 > x1) & (y2 > y1)

        crop_box_roi = torch.stack([x1, y1, x2, y2], dim=1).to(dtype)
        crop_box_roi = torch.where(valid_peak[:, None], crop_box_roi, zero_boxes)
        peak_xy_roi = torch.stack([peak_x, peak_y], dim=1).to(dtype)
        peak_xy_roi = torch.where(valid_peak[:, None], peak_xy_roi, zero_xy)
        peak_score = torch.where(valid_peak, peak_score, zero_score)
        best_s = torch.where(valid_peak, best_s, zero_scale)
        h_best = torch.where(valid_peak, h_best, zero_score)
        w_best = torch.where(valid_peak, w_best, zero_score)

        peak_map_flat = torch.zeros((B, H * W), device=device, dtype=dtype)
        peak_value = torch.where(valid_peak, torch.ones_like(peak_score), zero_score)
        peak_map_flat.scatter_(1, peak_hw.view(B, 1), peak_value.view(B, 1))
        direct_crop_peak = peak_map_flat.view(B, 1, H, W)

        mask_x = (xx >= crop_box_roi[:, 0].view(B, 1, 1, 1)) & (xx <= crop_box_roi[:, 2].view(B, 1, 1, 1))
        mask_y = (yy >= crop_box_roi[:, 1].view(B, 1, 1, 1)) & (yy <= crop_box_roi[:, 3].view(B, 1, 1, 1))
        direct_crop_pred_mask = (mask_x & mask_y & valid_peak.view(B, 1, 1, 1)).to(dtype) * E

        # ROI-local function: no full-frame offset is applied here.
        # The wrapper adds [rx0, ry0, rx0, ry0] exactly once for full-frame debug.
        crop_box_full = zero_boxes
        gt_box_roi = zero_boxes
        iou_pred_gt = zero_score

        debug = {
            "direct_crop_score_map": score_map.clamp_min(0.0),
            "direct_crop_peak": direct_crop_peak,
            "direct_crop_pred_mask": direct_crop_pred_mask,
            "direct_crop_crop_box_roi": crop_box_roi,
            "direct_crop_crop_box_full": crop_box_full,
            "direct_crop_peak_xy_roi": peak_xy_roi,
            "direct_crop_best_scale_idx": best_s,
            "direct_crop_h_best": h_best,
            "direct_crop_w_best": w_best,
            "direct_crop_peak_score": peak_score,
            "direct_crop_gt_box_roi": gt_box_roi,
            "direct_crop_iou_pred_gt": iou_pred_gt,
        }
        return crop_box_roi, peak_xy_roi, peak_score, best_s, debug

    def _direct_plate_crop_candidates_gpu(
        self,
        region_evidence: torch.Tensor,
        roi_box_full=None,
        frame_shape: tuple[int, int] | None = None,
        gt_box_full: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # fuzzy region direct plate crop gpu-only
        crop_box_roi, peak_xy_roi, peak_score, best_scale_idx, direct_debug = self._find_plate_by_region_avg_pool(
            region_evidence,
            roi_box_full=roi_box_full,
            gt_box_full=gt_box_full,
        )
        device = region_evidence.device
        dtype = region_evidence.dtype
        B, _, H, W = region_evidence.shape
        zero4 = torch.zeros((0, 4), device=device, dtype=dtype)
        zero6 = torch.zeros((0, 6), device=device, dtype=dtype)
        zero_map = torch.zeros_like(region_evidence)
        if roi_box_full is None:
            offset = torch.zeros((B, 4), device=device, dtype=dtype)
        elif isinstance(roi_box_full, torch.Tensor):
            roi_full = roi_box_full.to(device=device, dtype=dtype).view(-1, 4)
            offset_one = torch.stack([roi_full[:, 0], roi_full[:, 1], roi_full[:, 0], roi_full[:, 1]], dim=1)
            offset = offset_one.expand(B, 4) if offset_one.shape[0] == 1 and B > 1 else offset_one
        else:
            rx0, ry0, rx1, ry1 = roi_box_full
            offset = torch.tensor([rx0, ry0, rx0, ry0], device=device, dtype=dtype).view(1, 4).expand(B, 4)
        crop_box_full = crop_box_roi + offset
        if frame_shape is not None:
            frame_h, frame_w = frame_shape
            max_xy = torch.tensor([frame_w - 1, frame_h - 1, frame_w - 1, frame_h - 1], device=device, dtype=dtype).view(1, 4)
            crop_box_full = torch.minimum(crop_box_full.clamp_min(0.0), max_xy)
        crop_box_full = torch.where((peak_score > 0.0).view(B, 1), crop_box_full, torch.zeros_like(crop_box_full))
        direct_debug["direct_crop_crop_box_full"] = crop_box_full

        if gt_box_full is not None:
            gt_full = gt_box_full.to(device=device, dtype=dtype).view(-1, 4)
            gt_offset = offset[: gt_full.shape[0]] if offset.shape[0] >= gt_full.shape[0] else offset[:1].expand(gt_full.shape[0], 4)
            gt_box_roi = gt_full - gt_offset
            pred = crop_box_roi[: gt_box_roi.shape[0]]
            ix1 = torch.maximum(pred[:, 0], gt_box_roi[:, 0])
            iy1 = torch.maximum(pred[:, 1], gt_box_roi[:, 1])
            ix2 = torch.minimum(pred[:, 2], gt_box_roi[:, 2])
            iy2 = torch.minimum(pred[:, 3], gt_box_roi[:, 3])
            iw = (ix2 - ix1 + 1.0).clamp_min(0.0)
            ih = (iy2 - iy1 + 1.0).clamp_min(0.0)
            inter = iw * ih
            pa = ((pred[:, 2] - pred[:, 0] + 1.0).clamp_min(1.0) * (pred[:, 3] - pred[:, 1] + 1.0).clamp_min(1.0))
            ga = ((gt_box_roi[:, 2] - gt_box_roi[:, 0] + 1.0).clamp_min(1.0) * (gt_box_roi[:, 3] - gt_box_roi[:, 1] + 1.0).clamp_min(1.0))
            direct_debug["direct_crop_gt_box_roi"] = gt_box_roi
            direct_debug["direct_crop_iou_pred_gt"] = inter / (pa + ga - inter + 1e-6)

        valid = peak_score > 0.0
        valid_idx = torch.where(valid)[0]
        if valid_idx.numel() == 0:
            self._stage6_debug = {
                "stage6_raw_component_count": 0.0,
                "stage6_filter_pass_count": 0.0,
                "stage6_nms_post_count": 0.0,
                "stage6_topk_count": 0.0,
                "stage6_best_score": 0.0,
                "stage6_best_area_ratio": 0.0,
                "stage6_best_aspect": 0.0,
                "stage6_best_fill": 0.0,
                "stage6_best_cy_norm": 0.0,
                "stage6_best_inner_bright": 0.0,
                "score_pre_boxes": zero4,
                "nms_boxes": zero4,
                "final_boxes": zero4,
                "merged_score_pre_boxes": zero4,
                "rect_response_raw": zero_map,
                "rect_response": direct_debug["direct_crop_score_map"],
                "rect_peaks": direct_debug["direct_crop_peak"],
                "rect_candidates": direct_debug["direct_crop_pred_mask"],
                "center_inner_midscale": zero_map,
                "ring_mean_midscale": zero_map,
                "vertical_closure_midscale": zero_map,
                "final_candidates": zero_map,
                **direct_debug,
            }
            return zero6

        best_idx = valid_idx[torch.argmax(peak_score[valid_idx])]
        box_roi = crop_box_roi[best_idx : best_idx + 1]
        score = peak_score[best_idx : best_idx + 1]
        candidates = torch.cat([box_roi, score.unsqueeze(1), torch.zeros((1, 1), device=device, dtype=dtype)], dim=1)

        bw = (box_roi[:, 2] - box_roi[:, 0] + 1.0).clamp_min(1.0)
        bh = (box_roi[:, 3] - box_roi[:, 1] + 1.0).clamp_min(1.0)
        area_ratio = (bw * bh / float(max(H * W, 1))).clamp(0.0, 1.0)
        aspect = bw / (bh + 1e-6)
        cy_norm = (((box_roi[:, 1] + box_roi[:, 3]) * 0.5) / max(float(H), 1.0)).clamp(0.0, 1.0)
        final_candidates = torch.zeros_like(region_evidence)
        bx = box_roi[0].round().to(torch.long)
        final_candidates[:, :, bx[1]:bx[3] + 1, bx[0]:bx[2] + 1] = region_evidence[:, :, bx[1]:bx[3] + 1, bx[0]:bx[2] + 1]

        self._stage6_debug = {
            "stage6_raw_component_count": 1.0,
            "stage6_filter_pass_count": 1.0,
            "stage6_nms_post_count": 1.0,
            "stage6_topk_count": 1.0,
            "stage6_best_score": score[0].detach(),
            "stage6_best_area_ratio": area_ratio[0].detach(),
            "stage6_best_aspect": aspect[0].detach(),
            "stage6_best_fill": score[0].detach(),
            "stage6_best_cy_norm": cy_norm[0].detach(),
            "stage6_best_inner_bright": score[0].detach(),
            "score_pre_boxes": box_roi,
            "nms_boxes": box_roi,
            "final_boxes": box_roi,
            "merged_score_pre_boxes": zero4,
            "rect_response_raw": zero_map,
            "rect_response": direct_debug["direct_crop_score_map"],
            "rect_peaks": direct_debug["direct_crop_peak"],
            "rect_candidates": direct_debug["direct_crop_pred_mask"],
            "center_inner_midscale": zero_map,
            "ring_mean_midscale": zero_map,
            "vertical_closure_midscale": zero_map,
            "final_candidates": final_candidates,
            **direct_debug,
        }
        return candidates

    def _propose_plate_windows_gpu(
        self,
        evidence_source: torch.Tensor,
        fuzzy_evidence: torch.Tensor,
        fuzzy_memory: torch.Tensor,
        lower_prior_map: torch.Tensor,
        diagonal_prior_map: torch.Tensor,
        gx: torch.Tensor,
        final_topk: int = 10,
        pre_nms_topk: int = 2048,
        nms_iou: float = 0.35,
    ) -> torch.Tensor:
        self._assert_cuda(evidence_source, "evidence_source")
        roi_h, roi_w = evidence_source.shape[-2:]
        dtype = evidence_source.dtype
        device = evidence_source.device
        zero4 = torch.zeros((0, 4), device=device, dtype=dtype)
        zero6 = torch.zeros((0, 6), device=device, dtype=dtype)
        zero_map = torch.zeros_like(evidence_source)
        self._stage6_debug = {
            "stage6_raw_component_count": 0.0,
            "stage6_filter_pass_count": 0.0,
            "stage6_nms_post_count": 0.0,
            "stage6_topk_count": 0.0,
            "stage6_best_score": 0.0,
            "stage6_best_area_ratio": 0.0,
            "stage6_best_aspect": 0.0,
            "stage6_best_fill": 0.0,
            "stage6_best_cy_norm": 0.0,
            "stage6_best_inner_bright": 0.0,
            "score_pre_boxes": zero4,
            "nms_boxes": zero4,
            "final_boxes": zero4,
            "merged_score_pre_boxes": zero4,
            "rect_response_raw": zero_map,
            "rect_response": zero_map,
            "rect_peaks": zero_map,
            "rect_candidates": zero_map,
            "center_inner_midscale": zero_map,
            "ring_mean_midscale": zero_map,
            "vertical_closure_midscale": zero_map,
            "final_candidates": zero_map,
        }

        boxes = self.generate_boxes(evidence_source, pre_nms_topk=pre_nms_topk)
        rect_response = self._last_rect_response if self._last_rect_response is not None else zero_map
        rect_peaks = self._last_rect_peaks if self._last_rect_peaks is not None else zero_map
        raw_count = int(boxes.shape[0])
        if raw_count == 0:
            self._stage6_debug["rect_response_raw"] = self._last_rect_response_raw if self._last_rect_response_raw is not None else zero_map
            self._stage6_debug["rect_response"] = rect_response
            self._stage6_debug["rect_peaks"] = rect_peaks
            self._stage6_debug["center_inner_midscale"] = self._last_center_inner_midscale if self._last_center_inner_midscale is not None else zero_map
            self._stage6_debug["ring_mean_midscale"] = self._last_ring_mean_midscale if self._last_ring_mean_midscale is not None else zero_map
            self._stage6_debug["vertical_closure_midscale"] = self._last_vertical_closure_midscale if self._last_vertical_closure_midscale is not None else zero_map
            return zero6

        scores, metrics = self.score_boxes(
            boxes,
            evidence_source,
            rect_response,
            fuzzy_memory,
            lower_prior_map,
            diagonal_prior_map,
            gx,
        )
        pre_k = min(pre_nms_topk, int(scores.numel()))
        pre_score, pre_idx = torch.topk(scores, k=pre_k)
        pre_boxes = boxes[pre_idx]
        pre_metrics = metrics[pre_idx]
        keep = torchvision.ops.nms(pre_boxes, pre_score, iou_threshold=nms_iou)
        nms_boxes_all = pre_boxes[keep]
        nms_scores_all = pre_score[keep]
        nms_metrics_all = pre_metrics[keep]
        nms_count = int(nms_scores_all.numel())
        if nms_count == 0:
            self._stage6_debug["rect_response_raw"] = self._last_rect_response_raw if self._last_rect_response_raw is not None else zero_map
            self._stage6_debug["rect_response"] = rect_response
            self._stage6_debug["rect_peaks"] = rect_peaks
            self._stage6_debug["center_inner_midscale"] = self._last_center_inner_midscale if self._last_center_inner_midscale is not None else zero_map
            self._stage6_debug["ring_mean_midscale"] = self._last_ring_mean_midscale if self._last_ring_mean_midscale is not None else zero_map
            self._stage6_debug["vertical_closure_midscale"] = self._last_vertical_closure_midscale if self._last_vertical_closure_midscale is not None else zero_map
            return zero6

        out_k = min(final_topk, nms_count)
        out_score, out_idx = torch.topk(nms_scores_all, k=out_k)
        boxes_out = nms_boxes_all[out_idx]
        metrics_out = nms_metrics_all[out_idx]
        out = torch.cat([boxes_out, out_score.unsqueeze(1), torch.zeros((out_k, 1), device=device, dtype=dtype)], dim=1)

        center_x = (((pre_boxes[:, 0] + pre_boxes[:, 2]) * 0.5).round().to(torch.long)).clamp(0, roi_w - 1)
        center_y = (((pre_boxes[:, 1] + pre_boxes[:, 3]) * 0.5).round().to(torch.long)).clamp(0, roi_h - 1)
        proposal_flat = torch.zeros((roi_h * roi_w,), device=device, dtype=dtype)
        center_flat = center_y * roi_w + center_x
        proposal_flat.scatter_reduce_(0, center_flat, pre_score.to(dtype), reduce="amax", include_self=True)
        rect_candidates = proposal_flat.view(1, 1, roi_h, roi_w)
        rect_candidates = F.max_pool2d(rect_candidates, kernel_size=15, stride=1, padding=7)
        self._last_rect_candidates = rect_candidates

        final_candidates = torch.zeros_like(evidence_source)
        best = boxes_out[0].round().to(torch.long)
        final_candidates[:, :, best[1]:best[3] + 1, best[0]:best[2] + 1] = evidence_source[:, :, best[1]:best[3] + 1, best[0]:best[2] + 1]

        bw = (boxes_out[:, 2] - boxes_out[:, 0] + 1.0).clamp(min=1.0)
        bh = (boxes_out[:, 3] - boxes_out[:, 1] + 1.0).clamp(min=1.0)
        area_ratio = ((bw * bh) / float(max(roi_h * roi_w, 1))).clamp(0.0, 1.0)
        aspect = bw / (bh + 1e-6)
        cy_norm = (((boxes_out[:, 1] + boxes_out[:, 3]) * 0.5) / max(float(roi_h), 1.0)).clamp(0.0, 1.0)

        need_debug_scalar = self.debug_stage_log or self.debug_log_txt is not None
        if need_debug_scalar:
            best_vals = torch.stack([out_score[0], area_ratio[0], aspect[0], metrics_out[0, 0], cy_norm[0], metrics_out[0, 1]]).detach().tolist()
        else:
            best_vals = [0.0] * 6
        self._stage6_debug = {
            "stage6_raw_component_count": float(raw_count),
            "stage6_filter_pass_count": float(raw_count),
            "stage6_nms_post_count": float(nms_count),
            "stage6_topk_count": float(out_k),
            "stage6_best_score": float(best_vals[0]),
            "stage6_best_area_ratio": float(best_vals[1]),
            "stage6_best_aspect": float(best_vals[2]),
            "stage6_best_fill": float(best_vals[3]),
            "stage6_best_cy_norm": float(best_vals[4]),
            "stage6_best_inner_bright": float(best_vals[5]),
            "score_pre_boxes": pre_boxes[:30],
            "nms_boxes": nms_boxes_all[:15],
            "final_boxes": boxes_out,
            "merged_score_pre_boxes": zero4,
            "rect_response_raw": self._last_rect_response_raw if self._last_rect_response_raw is not None else zero_map,
            "rect_response": rect_response,
            "rect_peaks": rect_peaks,
            "rect_candidates": rect_candidates,
            "center_inner_midscale": self._last_center_inner_midscale if self._last_center_inner_midscale is not None else zero_map,
            "ring_mean_midscale": self._last_ring_mean_midscale if self._last_ring_mean_midscale is not None else zero_map,
            "vertical_closure_midscale": self._last_vertical_closure_midscale if self._last_vertical_closure_midscale is not None else zero_map,
            "final_candidates": final_candidates,
        }
        return out

    def stage10_tracker(self, candidates: torch.Tensor):
        # Tracker = same vehicle/plate track_id splitter.
        # Primary association uses bbox IoU, bbox center distance, and short frame gap.
        active = self.track_state[:, 10] > 0
        self.track_state[active, :4] += self.track_state[active, 4:8]
        self.track_state[active, 11] += 1
        self.track_state[active, 12] += 1

        n_det = int(candidates.shape[0])
        det_boxes = candidates[:, :4] if n_det > 0 else torch.zeros((0, 4), device=self.device)
        det_scores = candidates[:, 4] if n_det > 0 else torch.zeros((0,), device=self.device)
        det_track_ids = torch.full((n_det,), -1, device=self.device, dtype=torch.long)
        used_det = torch.zeros((n_det,), device=self.device, dtype=torch.bool)
        trk_idx = torch.where(active & (self.track_state[:, 12] <= self.tracker_max_age))[0]

        if trk_idx.numel() > 0 and n_det > 0:
            tbox = self.track_state[trk_idx, :4]
            xx1 = torch.maximum(tbox[:, None, 0], det_boxes[None, :, 0])
            yy1 = torch.maximum(tbox[:, None, 1], det_boxes[None, :, 1])
            xx2 = torch.minimum(tbox[:, None, 2], det_boxes[None, :, 2])
            yy2 = torch.minimum(tbox[:, None, 3], det_boxes[None, :, 3])
            iw = (xx2 - xx1 + 1).clamp(min=0)
            ih = (yy2 - yy1 + 1).clamp(min=0)
            inter = iw * ih
            ta = ((tbox[:, 2] - tbox[:, 0] + 1).clamp(min=1) * (tbox[:, 3] - tbox[:, 1] + 1).clamp(min=1))[:, None]
            da = ((det_boxes[:, 2] - det_boxes[:, 0] + 1).clamp(min=1) * (det_boxes[:, 3] - det_boxes[:, 1] + 1).clamp(min=1))[None, :]
            iou = inter / (ta + da - inter + 1e-6)

            tcx = (tbox[:, 0] + tbox[:, 2]) * 0.5
            tcy = (tbox[:, 1] + tbox[:, 3]) * 0.5
            dcx = (det_boxes[:, 0] + det_boxes[:, 2]) * 0.5
            dcy = (det_boxes[:, 1] + det_boxes[:, 3]) * 0.5
            diag = torch.sqrt((da.reshape(1, -1).clamp_min(1.0))) + 1e-6
            center_dist = torch.sqrt((tcx[:, None] - dcx[None, :]) ** 2 + (tcy[:, None] - dcy[None, :]) ** 2)
            center_score = (1.0 - center_dist / diag).clamp(0.0, 1.0)
            gap = self.track_state[trk_idx, 12].float().view(-1, 1)
            gap_ok = gap <= float(self.tracker_max_age)
            assoc_score = 0.70 * iou + 0.30 * center_score
            assoc_score = torch.where(gap_ok, assoc_score, torch.zeros_like(assoc_score))

            best_score, best_det = assoc_score.max(dim=1)
            _, best_trk_per_det = assoc_score.max(dim=0)
            local_ids = torch.arange(assoc_score.shape[0], device=assoc_score.device)
            match_valid = (best_score >= 0.25) | ((center_score[local_ids, best_det] >= 0.72) & gap_ok.reshape(-1))
            mutual = match_valid & (best_trk_per_det[best_det] == local_ids)
            matched_trk_local = torch.where(mutual)[0]
            matched_det = best_det[matched_trk_local]
            if matched_trk_local.numel() > 0:
                ti = trk_idx[matched_trk_local]
                prev = self.track_state[ti, :4].clone()
                new = det_boxes[matched_det]
                self.track_state[ti, 4:8] = new - prev
                self.track_state[ti, :4] = new
                self.track_state[ti, 8] = 0.65 * self.track_state[ti, 8] + 0.35 * det_scores[matched_det]
                self.track_state[ti, 10] = 1
                self.track_state[ti, 11] += 1
                self.track_state[ti, 12] = 0
                used_det[matched_det] = True
                det_track_ids[matched_det] = ti

        free = torch.where(self.track_state[:, 10] <= 0)[0]
        new_idx = torch.where(~used_det)[0]
        k = min(int(free.numel()), int(new_idx.numel()))
        if k > 0:
            ni = free[:k]
            di = new_idx[:k]
            self.track_state[ni, :4] = det_boxes[di]
            self.track_state[ni, 4:8] = 0
            self.track_state[ni, 8] = det_scores[di]
            self.track_state[ni, 10] = 1
            self.track_state[ni, 11] = 1
            self.track_state[ni, 12] = 0
            det_track_ids[di] = ni

        frame_value = int(getattr(self, "_current_frame_idx", -1) or -1)
        gate_w, gate_h = getattr(self, "_current_gate_size", (0, 0))
        for track_id_t in det_track_ids[det_track_ids >= 0].detach().reshape(-1).tolist():
            track_id_i = int(track_id_t)
            self._remember_valid_track_bbox(track_id_i, self.track_state[track_id_i, :4], frame_value, int(gate_w), int(gate_h))

        dead = self.track_state[:, 12] > self.tracker_max_age
        if bool(dead.any().detach().item()):
            self._finalize_dead_tracks(torch.where(dead)[0], getattr(self, "_current_frame_idx", None))
        self.track_state[dead, :] = 0
        self._last_candidate_track_ids = det_track_ids
        valid = det_track_ids >= 0
        return det_track_ids[valid], det_scores[valid]

    def _bbox_to_csv_values(self, bbox) -> tuple[object, object, object, object]:
        if bbox is None:
            return "", "", "", ""
        try:
            if torch.is_tensor(bbox):
                values = bbox.detach().float().cpu().reshape(-1).tolist()
            elif hasattr(bbox, "tolist"):
                values = bbox.tolist()
                if isinstance(values, (int, float)):
                    values = [values]
            else:
                values = list(bbox)
            if len(values) < 4:
                return "", "", "", ""
            return float(values[0]), float(values[1]), float(values[2]), float(values[3])
        except Exception:
            return "", "", "", ""

    def _is_valid_local_bbox(self, bbox: torch.Tensor, roi_w: int, roi_h: int) -> bool:
        try:
            if not torch.is_tensor(bbox) or bbox.numel() < 4:
                return False
            values = bbox.detach().float().reshape(-1)
            x1, y1, x2, y2 = [float(v.item()) for v in values[:4]]
            if not all(np.isfinite([x1, y1, x2, y2])):
                return False
            if x2 <= x1 or y2 <= y1:
                return False
            if x1 < 0 or y1 < 0:
                return False
            if int(roi_w) > 0 and x2 >= int(roi_w):
                return False
            if int(roi_h) > 0 and y2 >= int(roi_h):
                return False
            return True
        except Exception:
            return False

    def _remember_valid_track_bbox(self, track_id: int, bbox: torch.Tensor, frame_idx: int | None, roi_w: int, roi_h: int) -> None:
        if track_id < 0 or not self._is_valid_local_bbox(bbox, roi_w, roi_h):
            return
        frame_value = int(frame_idx) if frame_idx is not None else int(getattr(self, "_current_frame_idx", -1) or -1)
        self.last_valid_track_bboxes[int(track_id)] = (bbox.detach().clone(), frame_value)

    def _resolve_ocr_bbox(self, track_id: int, frame_idx: int | None, roi_w: int, roi_h: int) -> tuple[torch.Tensor | None, str, int]:
        frame_value = int(frame_idx) if frame_idx is not None else int(getattr(self, "_current_frame_idx", -1) or -1)
        current_bbox = self.track_state[track_id, :4]
        if self._is_valid_local_bbox(current_bbox, roi_w, roi_h):
            self._remember_valid_track_bbox(track_id, current_bbox, frame_value, roi_w, roi_h)
            return current_bbox, "", 0
        last = self.last_valid_track_bboxes.get(int(track_id))
        if last is not None:
            last_bbox, last_frame = last
            if frame_value < 0 or last_frame < 0 or frame_value - int(last_frame) <= int(getattr(self, "last_valid_bbox_max_age", 5)):
                if self._is_valid_local_bbox(last_bbox, roi_w, roi_h):
                    return last_bbox.to(device=self.device), "febam_last_valid_bbox", 1
        return None, "roi_invalid_no_valid_box", 0

    def _febam_debug_values(self, track_id: int) -> dict[str, object]:
        values: dict[str, object] = {
            "febam_score": "",
            "febam_energy": "",
            "febam_threshold": "",
            "febam_memory": "",
            "febam_confirmed": 0,
        }
        if track_id < 0 or track_id >= self.track_state.shape[0]:
            return values
        try:
            energy = float(getattr(self.febam, "energy")[track_id].detach().item())
            memory = float(getattr(self.febam, "stable_count")[track_id].detach().item())
            confirmed = bool(getattr(self.febam, "last_confirmed")[track_id].detach().item())
            threshold = self.febam_energy_thr
            confidence = float(self.track_state[track_id, 8].detach().item())
            memory_score = max(0.0, min(1.0, memory / max(float(getattr(self, "febam_memory_min", 2)), 1.0)))
            score = max(0.0, min(1.0, 0.55 * confidence + 0.25 * energy + 0.20 * memory_score))
            values.update({
                "febam_score": score,
                "febam_energy": energy,
                "febam_threshold": threshold,
                "febam_memory": memory,
                "febam_confirmed": int(confirmed),
            })
        except Exception:
            pass
        return values

    def _febam_skip_reason(self, track_id: int) -> str:
        if track_id < 0 or track_id >= self.track_state.shape[0]:
            return "febam_track_id_invalid"
        hits = int(self.track_state[track_id, 11].detach().item())
        stable_count = int(getattr(self.febam, "stable_count", torch.zeros((self.track_state.shape[0],), device=self.device))[track_id].detach().item())
        energy = float(getattr(self.febam, "energy", torch.zeros((self.track_state.shape[0],), device=self.device))[track_id].detach().item())
        delta = float(getattr(self.febam, "last_delta", torch.ones((self.track_state.shape[0],), device=self.device))[track_id].detach().item())
        confidence = float(self.track_state[track_id, 8].detach().item())
        min_hits = int(getattr(self.febam, "min_hits", 2))
        memory_min = int(getattr(self, "febam_memory_min", 2))
        eps = float(getattr(self.febam, "eps", 0.08))
        energy_threshold = float(getattr(self, "febam_energy_thr", 0.40))
        memory_score = max(0.0, min(1.0, stable_count / max(float(memory_min), 1.0)))
        score = max(0.0, min(1.0, 0.55 * confidence + 0.25 * energy + 0.20 * memory_score))
        if hits < min_hits:
            return "febam_track_too_young"
        if score < float(getattr(self, "febam_score_thr", 0.35)):
            return "febam_score_below_threshold"
        if energy < energy_threshold:
            return "febam_energy_below_threshold"
        if stable_count < memory_min or delta >= eps:
            return "febam_memory_below_min"
        return "no_febam_confirmed"

    def _ocr_crop_top_policy(self, y1: int, roi_w: int, roi_h: int) -> tuple[str, int, int, float]:
        roi_w = max(int(roi_w), 0)
        roi_h = max(int(roi_h), 0)
        aspect = float(roi_w) / float(max(roi_h, 1))
        layout = self._estimate_plate_layout(roi_w=roi_w, roi_h=roi_h)
        is_plate_like = 1.4 <= aspect <= 7.5
        if not is_plate_like:
            return "no_expand_not_plate_like", 0, int(y1), aspect
        if layout == "two_line_possible":
            ratio = 0.35
            policy = "expand_top35_two_line_possible"
        elif roi_h < 70 or roi_w < 120:
            ratio = float(getattr(self, "ocr_expand_top_strong", 0.30))
            policy = f"expand_top{int(round(ratio * 100)):02d}_small_or_blur_like"
        else:
            ratio = 0.20
            policy = "expand_top20_default"
        expand_top = int(round(float(roi_h) * ratio))
        return policy, expand_top, max(0, int(y1) - expand_top), aspect

    def _ocr_crop_expand_ratios(self, roi_w: int, roi_h: int, layout: str, sampling_level: str = "") -> tuple[float, float, float, float]:
        return resolve_ocr_crop_expand_ratios(
            roi_w, roi_h, layout, sampling_level,
            expand_x=float(getattr(self, "ocr_expand_x", 0.20)),
            expand_top_strong=float(getattr(self, "ocr_expand_top_strong", 0.30)),
        )

    def _ocr_crop_to_bgr(self, crop_u8: torch.Tensor | np.ndarray) -> np.ndarray:
        """Normalize the OCR crop boundary to contiguous BGR uint8.

        CUDA crops arrive as CHW RGB tensors, while the non-resident OCR path
        already supplies HWC RGB numpy arrays.  The identity bridge accepts
        both paths and must not infer channel layout merely from rank.
        """
        if torch.is_tensor(crop_u8):
            arr = crop_u8.detach().cpu()
            if arr.ndim == 3:
                arr_np = arr.permute(1, 2, 0).contiguous().numpy()
            else:
                arr_np = arr.contiguous().numpy()
        else:
            arr_np = np.asarray(crop_u8)
        arr_np = np.ascontiguousarray(arr_np)
        if arr_np.ndim == 2:
            return cv2.cvtColor(arr_np, cv2.COLOR_GRAY2BGR)
        if arr_np.ndim != 3:
            raise ValueError(f"OCR crop must be HW or HWC/CHW, got shape={arr_np.shape}")
        if arr_np.shape[2] == 4:
            return cv2.cvtColor(arr_np, cv2.COLOR_RGBA2BGR)
        if arr_np.shape[2] == 3:
            return cv2.cvtColor(arr_np, cv2.COLOR_RGB2BGR)
        if arr_np.shape[2] == 1:
            return cv2.cvtColor(arr_np[:, :, 0], cv2.COLOR_GRAY2BGR)
        raise ValueError(f"unsupported OCR crop channel count={arr_np.shape[2]}")

    def _ocr_debug_image_enabled(self) -> bool:
        return bool(self.debug_stage_log or self.debug_log_txt or self.debug_event_dir or self.debug_save_frames_dir)

    def _save_ocr_roi_debug(self, frame_idx: int | None, track_id: int, crop_u8: torch.Tensor, suffix: str = "raw") -> str:
        if not getattr(self, "ocr_save_debug_crops", False):
            return ""
        if not self._ocr_debug_image_enabled():
            return ""
        try:
            self.ocr_roi_debug_dir.mkdir(parents=True, exist_ok=True)
            frame_value = int(frame_idx) if frame_idx is not None else -1
            path = self.ocr_roi_debug_dir / f"frame{frame_value:05d}_track{int(track_id):04d}_{suffix}.png"
            bgr = self._ocr_crop_to_bgr(crop_u8)
            ok = cv2.imwrite(str(path), bgr)
            return str(path) if ok else ""
        except Exception as exc:
            self._log(f"[OCR_DEBUG] roi_save_failed frame={frame_idx} track={track_id} suffix={suffix} error={type(exc).__name__}")
            return ""

    def _rotate_ocr_crop_cuda(self, crop01: torch.Tensor, angle_deg: float) -> torch.Tensor:
        if abs(float(angle_deg)) <= 1e-6:
            return crop01
        theta_rad = torch.tensor(float(angle_deg) * 3.141592653589793 / 180.0, device=crop01.device, dtype=crop01.dtype)
        cos_a = torch.cos(theta_rad)
        sin_a = torch.sin(theta_rad)
        theta = torch.stack(
            [
                torch.stack([cos_a, -sin_a, torch.zeros((), device=crop01.device, dtype=crop01.dtype)]),
                torch.stack([sin_a, cos_a, torch.zeros((), device=crop01.device, dtype=crop01.dtype)]),
            ]
        ).unsqueeze(0)
        grid = F.affine_grid(theta, crop01.shape, align_corners=False)
        return F.grid_sample(crop01, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _ocr_projection_score_cuda(self, crop01: torch.Tensor) -> torch.Tensor:
        gray = (0.299 * crop01[:, 0:1] + 0.587 * crop01[:, 1:2] + 0.114 * crop01[:, 2:3]).clamp(0.0, 1.0)
        grad_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")
        grad_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        mag = (grad_x + 0.5 * grad_y).clamp_min(0.0)
        threshold = (mag.mean(dim=(-2, -1), keepdim=True) + 0.5 * mag.std(dim=(-2, -1), keepdim=True)).clamp_min(0.03)
        text_mask = (mag >= threshold).to(crop01.dtype) * mag
        proj_y = text_mask.mean(dim=-1).squeeze(1)
        return proj_y.max(dim=-1).values / (proj_y.mean(dim=-1) + 1e-6)

    def _rectify_ocr_crop_cuda(self, crop_u8: torch.Tensor, enable_angle_search: bool) -> tuple[torch.Tensor, float, float, int, int]:
        try:
            if str(getattr(self, "dual_branch_global_crop_mode", "")) == "cpu_crop_y27_exact":
                return crop_u8, 0.0, 0.0, 0, int(torch.is_tensor(crop_u8) and crop_u8.is_cuda)
            if not torch.is_tensor(crop_u8):
                return crop_u8, 0.0, 0.0, 0, 0
            crop_cuda = crop_u8.to(device=self.device, dtype=torch.uint8, non_blocking=True)
            if crop_cuda.ndim != 3:
                return crop_cuda, 0.0, 0.0, 0, 0
            h, w = crop_cuda.shape[-2:]
            if h < 4 or w < 8 or not enable_angle_search:
                return crop_cuda, 0.0, 0.0, 0, 1

            crop01 = crop_cuda.unsqueeze(0).to(dtype=torch.float32).clamp(0, 255) / 255.0
            angles = torch.tensor([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0], device=self.device, dtype=torch.float32)
            original_score = self._ocr_projection_score_cuda(crop01)[0]
            best_score = torch.tensor(-1.0, device=self.device)
            best_angle = torch.tensor(0.0, device=self.device)
            best_crop = crop01
            for angle_t in angles:
                angle = float(angle_t.item())
                rotated = self._rotate_ocr_crop_cuda(crop01, angle)
                score = self._ocr_projection_score_cuda(rotated)[0]
                if bool((score > best_score).detach().item()):
                    best_score = score
                    best_angle = angle_t
                    best_crop = rotated

            angle_value = float(best_angle.detach().item())
            score_value = float(best_score.detach().item())
            original_score_value = float(original_score.detach().item())
            if abs(angle_value) < 1.5 or score_value < original_score_value * 1.03:
                return crop_cuda, angle_value, score_value, 0, 1
            out = (best_crop[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            return out, angle_value, score_value, 1, 1
        except Exception as exc:
            self._log(f"[OCR_GPU_RECTIFY] rectification_failed error={type(exc).__name__}")
            return crop_u8, 0.0, 0.0, 0, 0


    def _small_roi_ocr_preprocess_cuda(
        self,
        crop_u8: torch.Tensor,
        roi_w: int,
        roi_h: int,
        roi_aspect: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int, int, float, float]:
        crop_cuda = crop_u8.to(device=self.device, dtype=torch.uint8, non_blocking=True)
        is_plate_like = 2.0 <= float(roi_aspect) <= 7.0
        is_small = (int(roi_w) <= 120 or int(roi_h) <= 35) and is_plate_like
        if not is_small:
            return crop_cuda, None, None, 0, 0, 1.0, 0.0

        if int(roi_h) < 70:
            scale = 4.0
        elif int(roi_h) <= 100:
            scale = 3.0
        else:
            scale = 2.0
        amount = 0.35
        crop01 = crop_cuda.unsqueeze(0).to(dtype=torch.float32).clamp(0, 255) / 255.0
        upscaled = F.interpolate(
            crop01,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        blur = F.avg_pool2d(upscaled, kernel_size=3, stride=1, padding=1)
        sharpened = (upscaled + amount * (upscaled - blur)).clamp(0.0, 1.0)
        upscaled_u8 = (upscaled[0] * 255.0).round().to(torch.uint8)
        sharpened_u8 = (sharpened[0] * 255.0).round().to(torch.uint8)
        return sharpened_u8, upscaled_u8, sharpened_u8, 1, 1, scale, amount

    def _raw_gray_ocr_crop_cuda(self, raw_crop_u8: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(raw_crop_u8):
            return raw_crop_u8
        crop = raw_crop_u8.to(device=self.device, dtype=torch.float32, non_blocking=True).clamp(0, 255) / 255.0
        if crop.ndim != 3:
            return raw_crop_u8.to(device=self.device, dtype=torch.uint8, non_blocking=True)
        if crop.shape[0] >= 3:
            gray = (0.299 * crop[0:1] + 0.587 * crop[1:2] + 0.114 * crop[2:3]).clamp(0.0, 1.0)
        else:
            gray = crop[0:1].clamp(0.0, 1.0)
        return (gray.repeat(3, 1, 1) * 255.0).round().to(torch.uint8)

    def _mild_contrast_ocr_crop_cuda(self, raw_gray_u8: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(raw_gray_u8):
            return raw_gray_u8
        gray = raw_gray_u8.to(device=self.device, dtype=torch.float32, non_blocking=True).clamp(0, 255) / 255.0
        if gray.ndim != 3:
            return raw_gray_u8.to(device=self.device, dtype=torch.uint8, non_blocking=True)
        gray01 = gray[0:1].clamp(0.0, 1.0)
        flat = gray01.flatten()
        if flat.numel() < 2:
            mild = gray01
        else:
            qs = torch.quantile(flat, torch.tensor([0.05, 0.95], device=flat.device, dtype=flat.dtype))
            p5 = qs[0]
            p95 = qs[1]
            mild = ((gray01 - p5) / (p95 - p5 + 1e-6)).clamp(0.0, 1.0)
        return (mild.repeat(3, 1, 1) * 255.0).round().to(torch.uint8)


    def _get_easyocr_reader(self):
        if self.ocr_reader is None:
            import easyocr

            self.ocr_reader = easyocr.Reader(self.ocr_reader_langs, gpu=True)
        return self.ocr_reader

    def _get_easyocr_batch_recognizer(self, allowlist: str | None = None) -> EasyOCRBatchRecognizer:
        key = str(allowlist or "")
        recognizer = self.easyocr_batch_recognizers.get(key)
        if recognizer is None:
            recognizer = EasyOCRBatchRecognizer(
                langs=tuple(self.ocr_reader_langs),
                gpu=True,
                batch_size=self.easyocr_batch_size,
                workers=self.easyocr_workers,
                allowlist=allowlist,
                readtext_fallback=self.easyocr_readtext_fallback,
            )
            self.easyocr_batch_recognizers[key] = recognizer
            if self.ocr_reader is None:
                self.ocr_reader = recognizer.reader
        self.easyocr_batch_recognizer = recognizer
        return recognizer


    def _get_easyocr_ko_reader(self):
        if self.ocr_ko_reader is None:
            import easyocr

            self.ocr_ko_reader = easyocr.Reader(["ko"], gpu=True)
        return self.ocr_ko_reader

    def _get_fastplate_batch_recognizer(self):
        if self._fastplate_init_failed:
            return None
        if self.fastplate_batch_recognizer is None:
            try:
                if self.fastplate_custom_onnx:
                    self.fastplate_batch_recognizer = CustomFastPlateONNXRecognizer(
                        onnx_path=self.fastplate_custom_onnx,
                        plate_config=self.fastplate_custom_plate_config,
                        input_width=self.fastplate_custom_input_width,
                        input_height=self.fastplate_custom_input_height,
                        device=self.fastplate_device,
                        logger=self._log_always,
                    )
                    self._log_always(
                        f"[OCR] Custom FastPlateONNX initialized onnx={self.fastplate_custom_onnx} "
                        f"device={self.fastplate_device} batch_size={self.fastplate_batch_size}"
                    )
                else:
                    from ocr.ocr_fastplate_batch import FastPlateOCRBatchRecognizer

                    self.fastplate_batch_recognizer = FastPlateOCRBatchRecognizer(
                        model=self.fastplate_model,
                        device=self.fastplate_device,
                        batch_size=self.fastplate_batch_size,
                        min_batch_size=self.fastplate_min_batch_size,
                        preload_torch_cuda_dlls=self.fastplate_preload_torch_cuda_dlls,
                        min_text_len=self.fastplate_min_text_len,
                    )
                    self._log(
                        f"[OCR] FastPlateOCR initialized model={self.fastplate_model} "
                        f"device={self.fastplate_device} batch_size={self.fastplate_batch_size}"
                    )
            except Exception as exc:
                self._fastplate_init_failed = True
                self._log(f"[OCR] FastPlateOCR init failed: {type(exc).__name__}: {exc}")
                return None
        return self.fastplate_batch_recognizer

    def _ocr_crop_to_rgb_np(self, crop_u8: torch.Tensor) -> np.ndarray:
        bgr = self._ocr_crop_to_bgr(crop_u8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _normalize_ocr_text(self, text: str) -> str:
        return clean_ocr_text(text)

    def _extract_event_fusion_raw_plate(self, raw_result: object, fallback_text: str = "") -> str:
        raw = str(raw_result or "")
        plate_match = re.search(r"plate\s*=\s*['\"]([^'\"]+)['\"]", raw, flags=re.IGNORECASE)
        if plate_match:
            return self._normalize_ocr_text(plate_match.group(1))
        candidates = []
        forbidden = {"PLATEPREDICTION", "REGIONPROB", "REGION", "REGION_PROB", "CHARPROBS", "CHAR_PROBS", "UNKNOWN", "NONE", "PLATE", "CONF", "BOX", "SCORE"}
        for token in re.findall(r"[0-9A-Za-z가-힣_]+", raw):
            cleaned = self._normalize_ocr_text(token)
            if token.upper() in forbidden or cleaned.upper() in forbidden:
                continue
            if cleaned:
                candidates.append(cleaned)
        if not candidates and fallback_text:
            cleaned = self._normalize_ocr_text(fallback_text)
            if cleaned:
                candidates.append(cleaned)
        return max(candidates, key=len) if candidates else ""

    def _postprocess_event_fusion_ocr_text(self, text: str, *, conf: float = 0.0, source: str = "fast_plate_ocr") -> str:
        cleaned = self._normalize_ocr_text(text)
        candidates = decode_korean_plate_candidates(cleaned, conf=conf, source=source)
        best, _score, _candidate_source, _reason = select_best_plate_candidate(candidates, track_state={})
        return best or cleaned

    def _estimate_plate_layout(self, crop=None, roi_w: int | float | None = None, roi_h: int | float | None = None, ocr_boxes=None) -> str:
        return estimate_plate_layout(crop=crop, roi_w=roi_w, roi_h=roi_h, ocr_boxes=ocr_boxes)

    def _grammar_decode_text(
        self,
        text: str,
        conf: float = 0.0,
        layout: str = "unknown",
        previous_consensus: str = "",
        source: str = "whole_ocr",
        known_plates: list[str] | None = None,
        context_anchor: dict[str, object] | None = None,
        preferred_skeleton: dict[str, object] | str | None = None,
    ):
        return decode_korean_plate_candidates(
            text,
            conf=conf,
            layout=layout,
            source=source,
            previous_consensus=previous_consensus,
            known_plates=known_plates,
            context_anchor=context_anchor,
            preferred_skeleton=preferred_skeleton,
        )

    def _plate_skeleton_memory_key(self, prefix: str, suffix: str, pattern: str) -> str:
        return f"{pattern}:{prefix}:{suffix}"

    def _prune_plate_skeleton_memory(self, track_id: int, frame_idx: int | None) -> None:
        if frame_idx is None:
            return
        memory = self.plate_skeleton_memory.get(int(track_id))
        if not memory:
            return
        max_age = int(getattr(self, "plate_skeleton_memory_max_age_frames", 60))
        for key, item in list(memory.items()):
            try:
                last_frame = int(item.get("last_frame", -1))
            except Exception:
                last_frame = -1
            if last_frame < 0 or int(frame_idx) - last_frame > max_age:
                memory.pop(key, None)
        if not memory:
            self.plate_skeleton_memory.pop(int(track_id), None)

    def _update_plate_skeleton_memory(self, track_id: int, raw_text: str, frame_idx: int | None, source: str = "") -> dict[str, object] | None:
        if track_id is None or int(track_id) < 0:
            return None
        raw = self._normalize_ocr_text(raw_text)
        match = re.fullmatch(r"(\d{2,3})([A-Z0-9가-힣])(\d{4})", raw)
        if not match:
            self._prune_plate_skeleton_memory(int(track_id), frame_idx)
            return None
        prefix, middle, suffix = match.group(1), match.group(2), match.group(3)
        pattern = "DDXDDDD" if len(prefix) == 2 else "DDDXDDDD"
        memory = self.plate_skeleton_memory.setdefault(int(track_id), {})
        key = self._plate_skeleton_memory_key(prefix, suffix, pattern)
        item = dict(memory.get(key, {}))
        middle_candidates = list(item.get("middle_candidates", []))
        if middle and middle not in middle_candidates:
            middle_candidates.append(middle)
        support = int(item.get("support", 0) or 0) + 1
        item.update({
            "pattern": pattern,
            "prefix": prefix,
            "suffix": suffix,
            "middle": middle,
            "middle_candidates": middle_candidates,
            "text": raw,
            "source": str(source or ""),
            "first_frame": item.get("first_frame", frame_idx if frame_idx is not None else -1),
            "last_frame": int(frame_idx) if frame_idx is not None else int(item.get("last_frame", -1)),
            "support": support,
        })
        memory[key] = item
        self._prune_plate_skeleton_memory(int(track_id), frame_idx)
        return item

    def _find_plate_skeleton_context_anchor(self, track_id: int, raw_text: str, frame_idx: int | None) -> dict[str, object] | None:
        if track_id is None or int(track_id) < 0:
            return None
        raw = self._normalize_ocr_text(raw_text)
        if not raw.isdigit() or len(raw) not in {6, 7}:
            return None
        self._prune_plate_skeleton_memory(int(track_id), frame_idx)
        memory = self.plate_skeleton_memory.get(int(track_id), {})
        anchor = None
        if len(raw) == 7:
            candidates = [
                (raw[:2], raw[3:], "DDXDDDD"),
                (raw[:3], raw[3:], "DDDXDDDD"),
            ]
        else:
            candidates = [(raw[:2], raw[2:], "DDXDDDD")]
        for prefix, suffix, pattern in candidates:
            anchor = memory.get(self._plate_skeleton_memory_key(prefix, suffix, pattern))
            if anchor:
                break
        if not anchor:
            return None
        if frame_idx is not None:
            try:
                if int(frame_idx) - int(anchor.get("last_frame", -1)) > int(getattr(self, "plate_skeleton_memory_max_age_frames", 60)):
                    return None
            except Exception:
                return None
        return dict(anchor)

    def _extract_custom_korean_slot(self, text: str) -> str:
        normalized = self._normalize_ocr_text(text)
        chars = [ch for ch in normalized if ch in self.ocr_korean_plate_chars]
        if not chars:
            return ""
        return chars[0]

    def _best_plate_skeleton_memory_anchor(self, track_id: int, frame_idx: int | None) -> dict[str, object] | None:
        if track_id is None or int(track_id) < 0:
            return None
        self._prune_plate_skeleton_memory(int(track_id), frame_idx)
        memory = self.plate_skeleton_memory.get(int(track_id), {})
        if not memory:
            return None
        candidates = []
        for anchor in memory.values():
            prefix = str(anchor.get("prefix", "") or "")
            suffix = str(anchor.get("suffix", "") or "")
            if len(prefix) not in {2, 3} or len(suffix) != 4:
                continue
            skeleton = f"{prefix}?{suffix}"
            if _is_placeholder_digit_skeleton(skeleton):
                continue
            candidates.append(dict(anchor))
        if not candidates:
            return None
        return max(candidates, key=lambda row: (int(row.get("support", 0) or 0), int(row.get("last_frame", -1) or -1)))

    def _plate_from_digit_skeleton(self, skeleton: str, korean_slot: str) -> str:
        skeleton = str(skeleton or "").strip()
        korean_slot = str(korean_slot or "").strip()[:1]
        if not korean_slot or korean_slot not in self.ocr_korean_plate_chars:
            return ""
        if "?" not in skeleton:
            return ""
        prefix, suffix = skeleton.split("?", 1)
        if len(prefix) not in {2, 3} or len(suffix) != 4 or not prefix.isdigit() or not suffix.isdigit():
            return ""
        return f"{prefix}{korean_slot}{suffix}"

    def _resolve_custom_onnx_digit_skeleton(
        self,
        *,
        track_id: int,
        frame_idx: int | None,
        context_anchor: dict[str, object] | None,
        variant_group_selected_skeleton: str,
        candidate_texts: list[str],
    ) -> tuple[str, str]:
        if context_anchor:
            prefix = str(context_anchor.get("prefix", "") or "")
            suffix = str(context_anchor.get("suffix", "") or "")
            skeleton = f"{prefix}?{suffix}"
            if not _is_placeholder_digit_skeleton(skeleton):
                return skeleton, "context_anchor"
        memory_anchor = self._best_plate_skeleton_memory_anchor(track_id, frame_idx)
        if memory_anchor:
            prefix = str(memory_anchor.get("prefix", "") or "")
            suffix = str(memory_anchor.get("suffix", "") or "")
            skeleton = f"{prefix}?{suffix}"
            if not _is_placeholder_digit_skeleton(skeleton):
                return skeleton, "plate_skeleton_memory"
        direct = str(variant_group_selected_skeleton or "")
        if "?" in direct and not _is_placeholder_digit_skeleton(direct):
            return direct, "variant_group_selected_skeleton"
        best: dict[str, object] | None = None
        for text in candidate_texts:
            for row in _generate_digit_skeleton_candidates(text):
                skeleton = str(row.get("skeleton", "") or "")
                if not skeleton or _is_placeholder_digit_skeleton(skeleton):
                    continue
                if best is None or float(row.get("score", 0.0) or 0.0) > float(best.get("score", 0.0) or 0.0):
                    best = row
        if best is not None:
            return str(best.get("skeleton", "") or ""), str(best.get("reason", "generated_digit_skeleton") or "generated_digit_skeleton")
        return "", "no_digit_skeleton"

    def _raw_plate_variant_anchor(self, raw_text: str, variant_name: str = "") -> dict[str, object] | None:
        raw = self._normalize_ocr_text(raw_text)
        match = re.fullmatch(r"(\d{2,3})([A-Z0-9가-힣])(\d{4})", raw)
        if not match:
            return None
        prefix, middle, suffix = match.group(1), match.group(2), match.group(3)
        pattern = "DDXDDDD" if len(prefix) == 2 else "DDDXDDDD"
        return {
            "pattern": pattern,
            "prefix": prefix,
            "suffix": suffix,
            "middle": middle,
            "middle_candidates": [middle] if middle else [],
            "text": raw,
            "variant": str(variant_name or ""),
            "source": "fast_plate_ocr_variant_group",
            "support": 1,
        }

    def _anchor_matches_digit_raw(self, anchor: dict[str, object], raw_text: str) -> bool:
        raw = self._normalize_ocr_text(raw_text)
        if not raw.isdigit() or len(raw) not in {6, 7}:
            return False
        pattern = str(anchor.get("pattern", "") or "")
        prefix = str(anchor.get("prefix", "") or "")
        suffix = str(anchor.get("suffix", "") or "")
        if pattern == "DDXDDDD" and len(prefix) == 2 and len(suffix) == 4:
            return (len(raw) == 6 and raw[:2] == prefix and raw[2:] == suffix) or (len(raw) == 7 and raw[:2] == prefix and raw[3:] == suffix)
        if pattern == "DDDXDDDD" and len(prefix) == 3 and len(suffix) == 4:
            return len(raw) == 7 and raw[:3] == prefix and raw[3:] == suffix
        return False

    def _preferred_skeleton_from_anchor(self, anchor: dict[str, object] | None, *, reason: str = "variant_anchor_missing_middle", score_bonus: float = 0.08) -> dict[str, object] | None:
        if not anchor:
            return None
        prefix = str(anchor.get("prefix", "") or "")
        suffix = str(anchor.get("suffix", "") or "")
        if len(prefix) not in {2, 3} or len(suffix) != 4:
            return None
        middle_candidates = [str(value) for value in anchor.get("middle_candidates", []) if str(value or "")]
        raw_middle = "".join(middle_candidates) or str(anchor.get("middle", "") or "")
        return {
            "skeleton": f"{prefix}?{suffix}",
            "pattern": "DD?DDDD" if len(prefix) == 2 else "DDD?DDDD",
            "raw_middle": raw_middle,
            "reason": reason,
            "score_bonus": float(score_bonus),
            "source_anchor": str(anchor.get("source", "variant_group") or "variant_group"),
        }

    def _merge_fastplate_variant_group(self, track_id: int, frame_idx: int | None, candidate_idx: int, variant_results: list[dict[str, object]]) -> dict[str, object]:
        group_key = f"{int(track_id)}:{int(frame_idx) if frame_idx is not None else -1}:{int(candidate_idx)}"
        merged_raw_texts = []
        anchors: list[dict[str, object]] = []
        digit_raws: list[str] = []
        for item in variant_results or []:
            variant_name = str(item.get("variant_name", item.get("variant", "")) or "")
            raw = self._normalize_ocr_text(str(item.get("raw_text", item.get("text", "")) or ""))
            if not raw:
                continue
            merged_raw_texts.append(f"{variant_name}={raw}" if variant_name else raw)
            anchor = self._raw_plate_variant_anchor(raw, variant_name)
            if anchor:
                anchors.append(anchor)
            if raw.isdigit() and len(raw) in {6, 7}:
                digit_raws.append(raw)

        selected_anchor: dict[str, object] | None = None
        merge_reason = "variant_group_no_anchor"
        for raw in digit_raws:
            dd_anchor = next((anchor for anchor in anchors if str(anchor.get("pattern")) == "DDXDDDD" and self._anchor_matches_digit_raw(anchor, raw)), None)
            if dd_anchor:
                selected_anchor = dd_anchor
                merge_reason = "variant_group_dd_anchor_missing_middle"
                break
            ddd_anchor = next((anchor for anchor in anchors if str(anchor.get("pattern")) == "DDDXDDDD" and self._anchor_matches_digit_raw(anchor, raw)), None)
            if ddd_anchor:
                selected_anchor = ddd_anchor
                merge_reason = "variant_group_ddd_anchor_missing_middle"
                break
        if selected_anchor is None:
            selected_anchor = next((anchor for anchor in anchors if str(anchor.get("pattern")) == "DDXDDDD"), None)
            if selected_anchor is not None:
                merge_reason = "variant_group_dd_anchor"
        if selected_anchor is None:
            selected_anchor = next((anchor for anchor in anchors if str(anchor.get("pattern")) == "DDDXDDDD"), None)
            if selected_anchor is not None:
                merge_reason = "variant_group_ddd_anchor"

        preferred_skeleton = self._preferred_skeleton_from_anchor(
            selected_anchor,
            reason="variant_anchor_missing_middle",
            score_bonus=0.08,
        )
        selected_skeleton = str((preferred_skeleton or {}).get("skeleton", "") or "")
        return {
            "group_key": group_key,
            "merged_raw_texts": "|".join(merged_raw_texts),
            "preferred_skeleton": preferred_skeleton,
            "context_anchor": selected_anchor,
            "preferred_candidates": [],
            "merge_reason": merge_reason,
            "anchor_text": str((selected_anchor or {}).get("text", "") or ""),
            "selected_skeleton": selected_skeleton,
        }

    def _grammar_decode_split_texts(self, *, left: str = "", mid: str = "", right: str = "", top: str = "", bottom: str = "", conf: float = 0.0, layout: str = "unknown", previous_consensus: str = ""):
        return decode_split_ocr_candidates(left=left, mid=mid, right=right, top=top, bottom=bottom, conf=conf, layout=layout, previous_consensus=previous_consensus)

    def _ocr_variant_bonus(self, variant_name: str) -> float:
        return {
            "raw_expanded": 0.06,
            "raw_upscaled": 0.06,
            "mild_clahe_upscaled": 0.05,
            "weak_unsharp_upscaled": 0.04,
            "small_upscaled_sharpened": 0.03,
            "gray_stretched_norm": 0.00,
            "binary_fallback": -0.04,
        }.get(str(variant_name or ""), 0.0)

    def _ocr_variant_tier(self, variant_name: str) -> int:
        return {
            "raw_expanded": 1,
            "gray_stretched_norm": 1,
            "small_upscaled_sharpened": 2,
            "raw_upscaled": 3,
            "mild_clahe_upscaled": 3,
            "weak_unsharp_upscaled": 3,
            "binary_fallback": 4,
        }.get(str(variant_name or ""), 9)

    def _recent_ocr_failure_count(self, track_id: int, window: int = 3) -> int:
        history = self.ocr_history.get(int(track_id), [])[-int(max(1, window)):]
        failures = 0
        for item in history:
            text = str(item.get("text", item.get("corrected_text", "")) or "")
            if not self._valid_final_plate_candidate(text):
                failures += 1
        return failures

    def _select_ocr_variants_for_candidate(
        self,
        roi_w: int,
        roi_h: int,
        roi_aspect: float,
        febam_score: float = 0.0,
        febam_confirmed: bool = False,
        near_confirmed_large_roi: bool = False,
        ocr_trigger_reason: str = "",
        previous_ocr_failed: int = 0,
        committed_plate_locked: bool = False,
        ocr_small_roi: bool = False,
        plate_layout: str = "unknown",
    ) -> list[str]:
        if committed_plate_locked:
            return []
        if self.dual_branch_ocr:
            mode = str(getattr(self, "dual_branch_global_input_variant", "raw"))
            if mode == "gray":
                return ["gray_stretched_norm"]
            if mode == "both":
                return ["raw_expanded", "gray_stretched_norm"]
            return ["raw_expanded"]
        if not self.ocr_variant_tiered:
            variants = [
                "raw_expanded",
                "gray_stretched_norm",
                "small_upscaled_sharpened",
                "raw_upscaled",
                "mild_clahe_upscaled",
                "weak_unsharp_upscaled",
                "binary_fallback",
            ]
            if not self.use_gray_stretched_ocr:
                variants.remove("gray_stretched_norm")
            return variants

        reason = str(ocr_trigger_reason or "")
        is_near_or_confirmed = bool(
            febam_confirmed
            or near_confirmed_large_roi
            or "near_confirmed" in reason
            or "confirmed" in reason
            or "febam" in reason
        )
        small_or_blur = bool(ocr_small_roi or roi_h < 35 or roi_w < 120 or self.ocr_small_sharpen_fallback)
        variants = ["raw_expanded"]
        if self.use_gray_stretched_ocr:
            variants.append("gray_stretched_norm")
        if self.ocr_max_variants_per_candidate >= 3 or small_or_blur or is_near_or_confirmed or previous_ocr_failed > 0:
            variants.append("small_upscaled_sharpened")
        if is_near_or_confirmed or previous_ocr_failed >= 2:
            variants.append("raw_upscaled")
        if previous_ocr_failed >= 2 and is_near_or_confirmed and not self.ocr_disable_heavy_variants:
            variants.extend(["mild_clahe_upscaled", "weak_unsharp_upscaled"])
        if previous_ocr_failed >= 3 and is_near_or_confirmed and not self.ocr_disable_heavy_variants:
            variants.append("binary_fallback")

        if self.ocr_heavy_variants_only_confirmed and not is_near_or_confirmed:
            variants = [v for v in variants if v not in {"mild_clahe_upscaled", "weak_unsharp_upscaled", "binary_fallback"}]

        limit = max(2, self.ocr_max_variants_per_candidate)
        if small_or_blur:
            limit = max(limit, self.ocr_max_variants_per_candidate)
        if is_near_or_confirmed:
            limit = max(limit, min(self.ocr_max_fallback_variants, 4))
        if previous_ocr_failed >= 2 and is_near_or_confirmed:
            limit = self.ocr_max_fallback_variants
        return variants[:max(1, int(limit))]

    def _easyocr_mode_policy_for_candidate(
        self,
        roi_w: int,
        roi_h: int,
        roi_aspect: float,
        plate_layout: str,
        ocr_small_roi: bool,
        previous_ocr_failed: int,
    ) -> str:
        configured = str(getattr(self, "easyocr_mode_policy", "auto") or "auto")
        if configured != "auto":
            return configured
        if (
            ocr_small_roi
            or roi_h < 35
            or roi_w < 120
            or str(plate_layout) == "two_line_possible"
            or previous_ocr_failed >= 2
            or roi_aspect < 2.5
        ):
            return "readtext_first"
        return "recognize_first" if self.easyocr_recognize_only else "readtext_first"

    def _ocr_variant_early_stop_reason(
        self,
        text: str,
        conf: float,
        grammar_best_text: str = "",
        grammar_best_score: float = 0.0,
        previous_consensus: str = "",
    ) -> str:
        normalized = self._normalize_ocr_text(text)
        if self._valid_final_plate_candidate(normalized) and float(conf or 0.0) >= 0.55:
            return "valid_korean_plate_conf"
        if grammar_best_text and self._valid_final_plate_candidate(grammar_best_text) and float(grammar_best_score or 0.0) >= 0.70:
            return "grammar_valid_score"
        previous = self._normalize_ocr_text(previous_consensus)
        if previous and normalized == previous and float(conf or 0.0) >= 0.35:
            return "previous_consensus_match"
        return ""

    def _fix_korean_plate_middle(
        self,
        normalized_text: str,
        conf: float,
    ) -> tuple[str, float, str, int, str, str, str]:
        normalized_text = self._normalize_ocr_text(normalized_text)
        if not normalized_text:
            return "", conf, "empty", 0, "", "", ""

        layout = "unknown"
        candidates = self._grammar_decode_text(normalized_text, conf=conf, layout=layout)
        if candidates:
            best = candidates[0]
            # Grammar is a validator, not an OCR source. A candidate that
            # changes any observed character must remain unresolved/HOLD.
            if best.text != normalized_text:
                return "", float(np.clip(conf, 0.0, 1.0)), "grammar_reject", 0, "grammar_would_modify_observation", "", ""
            prefix_len = 3 if best.pattern == "DDDKDDDD" else 2
            middle_raw = normalized_text[prefix_len:prefix_len + 1]
            middle_fixed = best.text[prefix_len:prefix_len + 1]
            return best.text, float(np.clip(conf, 0.0, 1.0)), "korean_plate_raw", 0, "grammar_validate_only", middle_raw, middle_fixed

        return "", conf, "grammar_invalid", 0, "invalid_final_plate", "", ""

    def _select_ocr_result(self, results) -> tuple[str, str, str, float, str, int, str, str, str]:
        candidates: list[tuple[str, str, str, float, str, int, str, str, str]] = []
        joined_text_parts: list[str] = []
        joined_conf_values: list[float] = []
        for item in results or []:
            raw_text = ""
            conf = 0.0
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    raw_text = str(item[1])
                if len(item) >= 3:
                    try:
                        conf = float(item[2])
                    except Exception:
                        conf = 0.0
            else:
                raw_text = str(item)
            normalized = self._normalize_ocr_text(raw_text)
            final_text, final_conf, pattern_type, applied, reason, middle_raw, middle_fixed = self._fix_korean_plate_middle(normalized, conf)
            if final_text:
                candidates.append((raw_text, normalized, final_text, final_conf, pattern_type, applied, reason, middle_raw, middle_fixed))
                joined_text_parts.append(raw_text)
                joined_conf_values.append(conf)
        if joined_text_parts:
            joined_raw = "".join(joined_text_parts)
            joined_norm = self._normalize_ocr_text(joined_raw)
            joined_conf = max(joined_conf_values) if joined_conf_values else 0.0
            final_text, final_conf, pattern_type, applied, reason, middle_raw, middle_fixed = self._fix_korean_plate_middle(joined_norm, joined_conf)
            if final_text:
                candidates.append((joined_raw, joined_norm, final_text, final_conf, pattern_type, applied, reason, middle_raw, middle_fixed))
        if not candidates:
            return "", "", "", 0.0, "empty", 0, "", "", ""
        for wanted in ("korean_plate_raw", "korean_plate_fixed_middle", "english_alnum_plate"):
            matching = [item for item in candidates if item[4] == wanted]
            if matching:
                return max(matching, key=lambda item: item[3])
        return max(candidates, key=lambda item: item[3])


    def _is_korean_middle_crop_candidate(self, normalized_text: str) -> bool:
        normalized_text = self._normalize_ocr_text(normalized_text)
        if len(normalized_text) not in (7, 8):
            return False
        digit_count = len(re.findall(r"\d", normalized_text))
        korean_or_english_count = len(re.findall(r"[A-Z가-힣]", normalized_text))
        return digit_count >= 5 and korean_or_english_count <= 2

    def _read_korean_middle_crop(self, image_rgb: np.ndarray, normalized_text: str) -> tuple[str, float]:
        if not self._is_korean_middle_crop_candidate(normalized_text):
            return "", 0.0
        h, w = image_rgb.shape[:2]
        if h < 2 or w < 4:
            return "", 0.0
        if len(normalized_text) == 8:
            x1_ratio, x2_ratio = 0.34, 0.55
        else:
            x1_ratio, x2_ratio = 0.28, 0.50
        x1 = max(0, min(w - 1, int(round(w * x1_ratio))))
        x2 = max(x1 + 1, min(w, int(round(w * x2_ratio))))
        y1 = max(0, min(h - 1, int(round(h * 0.05))))
        y2 = max(y1 + 1, min(h, int(round(h * 0.95))))
        middle_crop = image_rgb[y1:y2, x1:x2]
        if middle_crop.size == 0:
            return "", 0.0
        reader = self._get_easyocr_ko_reader()
        results = reader.readtext(
            middle_crop,
            detail=1,
            paragraph=False,
            allowlist=self.ocr_korean_plate_chars,
        )
        best_char = ""
        best_conf = 0.0
        for item in results or []:
            raw_text = ""
            conf = 0.0
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    raw_text = str(item[1])
                if len(item) >= 3:
                    try:
                        conf = float(item[2])
                    except Exception:
                        conf = 0.0
            else:
                raw_text = str(item)
            for ch in self._normalize_ocr_text(raw_text):
                if ch in self.ocr_korean_plate_chars and conf >= best_conf:
                    best_char = ch
                    best_conf = conf
        return best_char, best_conf

    def _apply_korean_middle_crop_fix(
        self,
        image_rgb: np.ndarray,
        raw_text: str,
        normalized_text: str,
        plate_text_final: str,
        conf: float,
        pattern_type: str,
        post_applied: int,
        post_reason: str,
        middle_raw: str,
        middle_fixed: str,
    ) -> tuple[str, str, str, float, str, int, str, str, str]:
        if not self._is_korean_middle_crop_candidate(normalized_text):
            return raw_text, normalized_text, plate_text_final, conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed
        prefix_len = 3 if len(normalized_text) == 8 else 2
        prefix = normalized_text[:prefix_len]
        middle_candidate = normalized_text[prefix_len:prefix_len + 1]
        suffix = normalized_text[prefix_len + 1:]
        if middle_candidate in self.ocr_korean_plate_chars:
            return raw_text, normalized_text, plate_text_final, conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed
        ko_middle, ko_conf = self._read_korean_middle_crop(image_rgb, normalized_text)
        if not ko_middle:
            return raw_text, normalized_text, plate_text_final, conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed
        fixed = f"{prefix}{ko_middle}{suffix}"
        reason = f"middle_crop_{middle_candidate}_to_{ko_middle}"
        return raw_text, normalized_text, fixed, conf, "korean_plate_fixed_middle", 1, reason, middle_candidate, ko_middle

    def _normalize_ocr_input_np(
        self,
        image_rgb: np.ndarray,
        min_w: int = 240,
        min_h: int = 60,
        max_w: int = 384,
        max_h: int = 96,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        if image_rgb is None or image_rgb.size == 0:
            empty = np.full((min_h, min_w, 3), 245, dtype=np.uint8)
            return empty, {"w": min_w, "h": min_h, "scale": 1.0, "pad_x": 0, "pad_y": 0, "aspect_preserved": 1}
        img = np.asarray(image_rgb)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        img = img.astype(np.uint8, copy=False)
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            empty = np.full((min_h, min_w, 3), 245, dtype=np.uint8)
            return empty, {"w": min_w, "h": min_h, "scale": 1.0, "pad_x": 0, "pad_y": 0, "aspect_preserved": 1}

        min_scale = max(float(min_w) / float(w), float(min_h) / float(h), 1.0)
        max_scale = min(float(max_w) / float(w), float(max_h) / float(h))
        if max_scale < 1.0:
            scale = max_scale
        else:
            scale = min(min_scale, max_scale)
        scale = max(scale, 1e-6)
        new_w = max(1, min(max_w, int(round(float(w) * scale))))
        new_h = max(1, min(max_h, int(round(float(h) * scale))))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA)
        canvas_w = max(min_w, new_w)
        canvas_h = max(min_h, new_h)
        canvas_w = min(max_w, canvas_w)
        canvas_h = min(max_h, canvas_h)
        pad_x = max(0, (canvas_w - new_w) // 2)
        pad_y = max(0, (canvas_h - new_h) // 2)
        border_pixels = np.concatenate([
            resized[0:1, :, :].reshape(-1, 3),
            resized[-1:, :, :].reshape(-1, 3),
            resized[:, 0:1, :].reshape(-1, 3),
            resized[:, -1:, :].reshape(-1, 3),
        ], axis=0)
        pad_color = np.maximum(border_pixels.mean(axis=0), np.array([225.0, 225.0, 225.0])).astype(np.uint8)
        out = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
        out[:, :] = pad_color.reshape(1, 1, 3)
        out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return out, {"w": int(canvas_w), "h": int(canvas_h), "scale": float(scale), "pad_x": int(pad_x), "pad_y": int(pad_y), "aspect_preserved": 1}

    def _apply_mild_clahe_rgb(self, image_rgb: np.ndarray, clip_limit: float = 1.5, blend: float = 0.32) -> np.ndarray:
        img = np.asarray(image_rgb).astype(np.uint8, copy=False)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        mixed_l = cv2.addWeighted(l_channel, 1.0 - float(blend), enhanced_l, float(blend), 0.0)
        enhanced = cv2.merge([mixed_l, a_channel, b_channel])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    def _ocr_early_stop_reason(self, normalized_text: str, conf: float, pattern_type: str) -> str:
        normalized = self._normalize_ocr_text(normalized_text)
        if self._valid_final_plate_candidate(normalized):
            return "valid_final_plate_candidate"
        if len(normalized) >= 6 and float(conf) >= 0.30:
            return "good_text_conf"
        return ""

    def _run_easyocr_once(
        self,
        image_rgb: np.ndarray,
        allowlist: str | None,
        variant_name: str = "unknown",
        mode_policy: str = "auto",
        allow_readtext_fallback: bool | None = None,
    ) -> tuple[str, str, str, float, str, int, str, str, str]:
        self._last_easyocr_result_meta = {}
        policy = str(mode_policy or "auto")
        if policy == "auto":
            policy = "recognize_first" if self.easyocr_recognize_only else "readtext_first"
        fallback_allowed = self.easyocr_readtext_fallback if allow_readtext_fallback is None else bool(allow_readtext_fallback)

        def _readtext_result() -> dict[str, object]:
            recognizer = self._get_easyocr_batch_recognizer(allowlist)
            return recognizer.readtext_one_box(
                image_rgb,
                meta={"allowlist_mode": "custom" if allowlist else "none"},
                variant_name=variant_name,
            )

        def _recognize_result(readtext_fallback: bool) -> dict[str, object]:
            recognizer = self._get_easyocr_batch_recognizer(allowlist)
            return recognizer.recognize_one_box(
                image_rgb,
                meta={"allowlist_mode": "custom" if allowlist else "none"},
                variant_name=variant_name,
                readtext_fallback=readtext_fallback,
            )

        if policy == "readtext_only":
            result = _readtext_result()
        elif policy == "readtext_first":
            result = _readtext_result()
            raw_text = str(result.get("text", "") or "")
            conf = float(result.get("conf", 0.0) or 0.0)
            if (not raw_text or conf < 0.10) and self.easyocr_recognize_only:
                recog = _recognize_result(False)
                recog_text = str(recog.get("text", "") or "")
                recog_conf = float(recog.get("conf", 0.0) or 0.0)
                if recog_text and recog_conf >= conf:
                    result = recog
        else:
            result = _recognize_result(fallback_allowed)

        self._last_easyocr_result_meta = dict(result)
        raw_text = str(result.get("text", "") or "")
        conf = float(result.get("conf", 0.0) or 0.0)
        selected = self._select_ocr_result([(None, raw_text, conf)])
        return self._apply_korean_middle_crop_fix(image_rgb, *selected)

    def _run_fastplate_once(
        self,
        image_rgb: np.ndarray,
        variant_name: str = "unknown",
    ) -> tuple[str, str, str, float, str, int, str, str, str]:
        self._last_fastplate_result_meta = {}
        recognizer = self._get_fastplate_batch_recognizer()
        if recognizer is None:
            return "", "", "", 0.0, "empty", 0, "fastplate_unavailable", "", ""
        result = recognizer.recognize_one_box(
            image_rgb,
            meta={
                "fastplate_model": self.fastplate_model,
                "fastplate_device": self.fastplate_device,
                "fastplate_batch_size": self.fastplate_batch_size,
            },
            variant_name=variant_name,
        )
        self._last_fastplate_result_meta = dict(result)
        raw_text = str(result.get("text", "") or "")
        conf = float(result.get("conf", 0.0) or 0.0)
        return self._select_ocr_result([(None, raw_text, conf)])


    def configure_fastplate_direct_ort_from_args(self, args) -> None:
        self.fastplate_custom_onnx = str(getattr(args, "fastplate_custom_onnx", getattr(self, "fastplate_custom_onnx", "")) or "")
        self.fastplate_custom_plate_config = str(getattr(args, "fastplate_custom_plate_config", getattr(self, "fastplate_custom_plate_config", "")) or "")
        self.fastplate_custom_input_width = int(max(1, getattr(args, "fastplate_custom_input_width", getattr(self, "fastplate_custom_input_width", 256)) or 256))
        self.fastplate_custom_input_height = int(max(1, getattr(args, "fastplate_custom_input_height", getattr(self, "fastplate_custom_input_height", 64)) or 64))
        self.dual_branch_ocr = bool(getattr(args, "dual_branch_ocr", getattr(self, "dual_branch_ocr", False)))
        self.dual_branch_global_onnx = str(getattr(args, "dual_branch_global_onnx", getattr(self, "dual_branch_global_onnx", "")) or "")
        self.dual_branch_korean_onnx = str(getattr(args, "dual_branch_korean_onnx", getattr(self, "dual_branch_korean_onnx", "")) or "")
        self.dual_branch_hangul_weight = float(getattr(args, "dual_branch_hangul_weight", getattr(self, "dual_branch_hangul_weight", 1.2)))
        self.dual_branch_digit_alignment_mode = str(getattr(args, "dual_branch_digit_alignment_mode", self.dual_branch_digit_alignment_mode))
        self.dual_branch_digit_mass_thr = float(getattr(args, "dual_branch_digit_mass_thr", self.dual_branch_digit_mass_thr))
        self.dual_branch_digit_conf_thr = float(getattr(args, "dual_branch_digit_conf_thr", self.dual_branch_digit_conf_thr))
        self.dual_branch_digit_margin_thr = float(getattr(args, "dual_branch_digit_margin_thr", self.dual_branch_digit_margin_thr))
        self.dual_branch_global_input_variant = str(getattr(args, "dual_branch_global_input_variant", self.dual_branch_global_input_variant))
        self.dual_branch_gray_secondary = bool(getattr(args, "dual_branch_gray_secondary", self.dual_branch_gray_secondary))
        self.dual_branch_global_crop_mode = str(getattr(args, "dual_branch_global_crop_mode", self.dual_branch_global_crop_mode))
        if self.dual_branch_global_crop_mode == "cpu_crop_y27":
            self.dual_branch_global_crop_mode = "cpu_crop_y27_exact"
        if bool(getattr(args, "dual_branch_global_cpu_crop", False)):
            self.dual_branch_global_crop_mode = "cpu_crop"
        self.dual_branch_global_result_mode = str(getattr(args, "dual_branch_global_result_mode", self.dual_branch_global_result_mode))
        self.dual_branch_korean_mode = str(getattr(args, "dual_branch_korean_mode", self.dual_branch_korean_mode))
        self.dual_branch_korean_slot_mode = str(getattr(args, "dual_branch_korean_slot_mode", self.dual_branch_korean_slot_mode))
        self.dual_branch_korean_mass_thr = float(getattr(args, "dual_branch_korean_mass_thr", self.dual_branch_korean_mass_thr))
        self.dual_branch_korean_conf_thr = float(getattr(args, "dual_branch_korean_conf_thr", self.dual_branch_korean_conf_thr))
        self.dual_branch_korean_margin_thr = float(getattr(args, "dual_branch_korean_margin_thr", self.dual_branch_korean_margin_thr))
        self.dual_branch_length_margin_thr = float(getattr(args, "dual_branch_length_margin_thr", self.dual_branch_length_margin_thr))
        self.dual_branch_length_hold = bool(getattr(args, "dual_branch_length_hold", self.dual_branch_length_hold))
        self.string_febam_group_key_mode = str(getattr(args, "string_febam_group_key_mode", self.string_febam_group_key_mode))
        self.fastplate_custom_fullplate_diag_only = bool(getattr(args, "fastplate_custom_fullplate_diag_only", False))
        self.fastplate_direct_ort = bool(getattr(args, "fastplate_direct_ort", getattr(self, "fastplate_direct_ort", False)))
        self.fastplate_direct_ort_iobinding = bool(getattr(args, "fastplate_direct_ort_iobinding", getattr(self, "fastplate_direct_ort_iobinding", True)))
        self.fastplate_direct_ort_debug = bool(getattr(args, "fastplate_direct_ort_debug", getattr(self, "fastplate_direct_ort_debug", False)))
        self.fastplate_direct_ort_model_path = str(getattr(args, "fastplate_direct_ort_model_path", getattr(self, "fastplate_direct_ort_model_path", "")) or "")
        self.fastplate_direct_ort_input_h = int(max(0, getattr(args, "fastplate_direct_ort_input_h", getattr(self, "fastplate_direct_ort_input_h", 0)) or 0))
        self.fastplate_direct_ort_input_w = int(max(0, getattr(args, "fastplate_direct_ort_input_w", getattr(self, "fastplate_direct_ort_input_w", 0)) or 0))
        self.fastplate_direct_ort_compare_wrapper = bool(getattr(args, "fastplate_direct_ort_compare_wrapper", getattr(self, "fastplate_direct_ort_compare_wrapper", False)))
        self.fastplate_preserve_crop_geometry = bool(getattr(args, "fastplate_preserve_crop_geometry", False))
        self.gpu_final_vehicle_grouping = bool(getattr(args, "gpu_final_vehicle_grouping", False))
        input_path = str(getattr(args, "input", "") or "")
        self.runtime_video_id = str(Path(input_path).stem if input_path else "")

        if self.fastplate_custom_onnx or self.dual_branch_ocr:
            self.fastplate_tensor_runner = False
            self.fastplate_tensor_runner_mode = "disabled"
            self.fastplate_direct_ort = False
        if self.fastplate_direct_ort and not self.fastplate_custom_onnx:
            self.fastplate_tensor_runner = True
            if self.fastplate_tensor_runner_mode == "disabled":
                self.fastplate_tensor_runner_mode = "active"
            if self.fastplate_direct_ort_model_path:
                self.fastplate_model = self.fastplate_direct_ort_model_path

        if self.fastplate_async_worker is not None:
            self._log(
                "[FASTPLATE_DIRECT_ORT] configure called after async worker init; "
                "existing worker will keep its current settings"
            )
        else:
            self._log(
                "[FASTPLATE_DIRECT_ORT] configured "
                f"enabled={int(self.fastplate_direct_ort)} "
                f"iobinding={int(self.fastplate_direct_ort_iobinding)} "
                f"mode={self.fastplate_tensor_runner_mode}"
            )

    def configure_bio_adaptive_ocr_from_args(self, args) -> None:
        self.bio_adaptive_enabled = bool(getattr(args, "bio_adaptive_ocr", False))
        self.bio_adaptive_shadow = bool(getattr(args, "bio_adaptive_shadow", False))
        self.bio_adaptive_processor = None
        if not self.bio_adaptive_enabled and not self.bio_adaptive_shadow:
            return
        router_mode = str(getattr(args, "bio_router_mode", "rule") or "rule")
        if router_mode == "learned_active":
            raise RuntimeError("bio learned_active router is not implemented in Phase 1-2")
        from ocr.bio_adaptive_ocr import BioAdaptiveOCR
        from ocr.bio_quality_router import BioRouterThresholds

        thresholds = BioRouterThresholds(
            clear_sharpness=float(getattr(args, "bio_clear_sharpness_thr", 0.55)),
            clear_edge_completeness=float(getattr(args, "bio_clear_edge_thr", 0.45)),
            clear_max_blur=float(getattr(args, "bio_clear_max_blur", 0.45)),
            clear_max_ghost=float(getattr(args, "bio_clear_max_ghost", 0.35)),
            clear_min_bbox_stability=float(getattr(args, "bio_clear_min_bbox_stability", 0.60)),
            clear_min_crop_height=int(getattr(args, "bio_clear_min_crop_height", 24)),
            severe_blur=float(getattr(args, "bio_severe_blur_thr", 0.68)),
            severe_ghost=float(getattr(args, "bio_severe_ghost_thr", 0.62)),
            temporal_min_frames=int(getattr(args, "bio_temporal_min_frames", 3)),
            temporal_min_segments=int(getattr(args, "bio_temporal_min_segments", 2)),
        )
        active = bool(self.bio_adaptive_enabled and not self.bio_adaptive_shadow)
        self.bio_adaptive_processor = BioAdaptiveOCR(
            active=active,
            svm_weights=str(getattr(args, "bio_svm_weights", "") or ""),
            clear_backend=str(getattr(args, "bio_clear_backend", "hog_lbp_svm") or "hog_lbp_svm"),
            no_cpu_fallback=bool(getattr(args, "bio_no_cpu_fallback", True)),
            svm_min_conf=float(getattr(args, "bio_route_a_min_conf", 0.70)),
            svm_min_margin=float(getattr(args, "bio_route_a_min_margin", 0.15)),
            router_thresholds=thresholds,
        )
        self._log(
            "[BIO_ADAPTIVE_OCR] "
            f"mode={'active' if active else 'shadow'} router={router_mode} "
            f"clear_backend={getattr(args, 'bio_clear_backend', 'hog_lbp_svm')}"
        )

    def configure_non_generative_restoration_shadow_from_args(self, args) -> None:
        enabled = bool(getattr(args, "non_generative_restoration_shadow", False))
        self.non_generative_restoration_shadow = None
        if not enabled:
            self._log("[RESTORATION_SHADOW] disabled")
            return
        if not self.fastplate_async:
            raise RuntimeError(
                "restoration shadow requires the shared FastPlate async OCR queue"
            )
        from ocr.non_generative_restoration_shadow import (
            NonGenerativeRestorationShadow,
        )

        self.non_generative_restoration_shadow = NonGenerativeRestorationShadow(
            enabled=True,
            output_jsonl=str(
                getattr(
                    args,
                    "restoration_shadow_output",
                    "./outputs/non_generative_restoration_shadow.jsonl",
                )
            ),
            submit_fused=self._submit_restoration_shadow_fused,
            max_events=int(getattr(args, "restoration_shadow_max_events", 64)),
            max_frames_per_event=int(
                getattr(args, "restoration_shadow_max_frames", 5)
            ),
            min_frames=int(getattr(args, "restoration_shadow_min_frames", 2)),
            schedule_frames=int(
                getattr(args, "restoration_shadow_schedule_frames", 3)
            ),
        )
        self._log(
            "[RESTORATION_SHADOW] enabled observation_only=1 "
            "final_authority=unchanged shared_ocr_session=1"
        )

    def _submit_restoration_shadow_fused(
        self, kind: str, image_bgr: np.ndarray, meta: dict[str, object]
    ) -> bool:
        shadow = getattr(self, "non_generative_restoration_shadow", None)
        if shadow is None or not getattr(shadow, "enabled", False):
            return False
        worker = self._get_fastplate_async_worker()
        if worker is None:
            return False
        rgb = cv2.cvtColor(
            np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB
        )
        crop = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .to(device="cuda", non_blocking=True)
            .contiguous()
        )
        task_meta = dict(meta)
        task_meta["restoration_shadow_only"] = True
        task_meta["restoration_shadow_kind"] = str(kind)
        task_meta["variant_name"] = f"restoration_shadow_{kind}"
        task_meta["source_type"] = "event_fused_group_key"
        return bool(worker.enqueue(crop, task_meta))

    def _get_fastplate_async_worker(self) -> FastPlateOCRQueueWorker | None:
        if self.ocr_backend not in {"fastplate", "both"} or not self.fastplate_async:
            return None
        if self.fastplate_async_worker is None:
            try:
                self.fastplate_async_worker = FastPlateOCRQueueWorker(
                    model=self.fastplate_model,
                    device=self.fastplate_device,
                    batch_size=self.fastplate_batch_size,
                    min_batch_size=self.fastplate_min_batch_size,
                    preload_torch_cuda_dlls=self.fastplate_preload_torch_cuda_dlls,
                    min_text_len=self.fastplate_min_text_len,
                    flush_timeout_ms=self.fastplate_flush_timeout_ms,
                    queue_max=self.fastplate_queue_max,
                    rate_limit_relaxed=self.fastplate_rate_limit_relaxed,
                    large_batch_mode=self.fastplate_large_batch_mode,
                    target_batch_size=self.fastplate_target_batch_size,
                    max_flush_timeout_ms=self.fastplate_max_flush_timeout_ms,
                    gpu_crop_batch=self.fastplate_gpu_crop_batch,
                    tensor_runner_enabled=self.fastplate_tensor_runner,
                    tensor_runner_mode=self.fastplate_tensor_runner_mode,
                    tensor_input_h=self.fastplate_ocr_input_h,
                    tensor_input_w=self.fastplate_ocr_input_w,
                    parity_sample_limit=self.fastplate_parity_sample_limit,
                    parity_log_csv=self.fastplate_parity_log_csv,
                    tensor_profile=self.fastplate_tensor_profile,
                    direct_ort_enabled=self.fastplate_direct_ort,
                    direct_ort_iobinding=self.fastplate_direct_ort_iobinding,
                    direct_ort_debug=self.fastplate_direct_ort_debug,
                    direct_ort_model_path=self.fastplate_direct_ort_model_path,
                    direct_ort_input_h=self.fastplate_direct_ort_input_h,
                    direct_ort_input_w=self.fastplate_direct_ort_input_w,
                    direct_ort_compare_wrapper=self.fastplate_direct_ort_compare_wrapper,
                    preserve_aspect_ratio=bool(getattr(self, "fastplate_preserve_crop_geometry", False)),
                    fastplate_custom_onnx=self.fastplate_custom_onnx,
                    fastplate_custom_plate_config=self.fastplate_custom_plate_config,
                    fastplate_custom_input_width=self.fastplate_custom_input_width,
                    fastplate_custom_input_height=self.fastplate_custom_input_height,
                    dual_branch_ocr=self.dual_branch_ocr,
                    dual_branch_global_onnx=self.dual_branch_global_onnx,
                    dual_branch_korean_onnx=self.dual_branch_korean_onnx,
                    dual_branch_hangul_weight=self.dual_branch_hangul_weight,
                    dual_branch_digit_alignment_mode=self.dual_branch_digit_alignment_mode,
                    dual_branch_digit_mass_thr=self.dual_branch_digit_mass_thr,
                    dual_branch_digit_conf_thr=self.dual_branch_digit_conf_thr,
                    dual_branch_digit_margin_thr=self.dual_branch_digit_margin_thr,
                    dual_branch_global_input_variant=self.dual_branch_global_input_variant,
                    dual_branch_gray_secondary=self.dual_branch_gray_secondary,
                    dual_branch_global_crop_mode=self.dual_branch_global_crop_mode,
                    dual_branch_global_result_mode=self.dual_branch_global_result_mode,
                    dual_branch_korean_mode=self.dual_branch_korean_mode,
                    dual_branch_korean_slot_mode=self.dual_branch_korean_slot_mode,
                    dual_branch_korean_mass_thr=self.dual_branch_korean_mass_thr,
                    dual_branch_korean_conf_thr=self.dual_branch_korean_conf_thr,
                    dual_branch_korean_margin_thr=self.dual_branch_korean_margin_thr,
                    dual_branch_length_margin_thr=self.dual_branch_length_margin_thr,
                    dual_branch_length_hold=self.dual_branch_length_hold,
                    logger=self._log_always,
                )
            except Exception as exc:
                self._log(f"[OCR] FastPlateOCR async worker init failed: {type(exc).__name__}: {exc}")
                return None
        return self.fastplate_async_worker

    def _enqueue_fastplate_async(self, image_rgb: Any, meta: dict[str, object]) -> bool:
        worker = self._get_fastplate_async_worker()
        if worker is None:
            return False
        ok = worker.enqueue(image_rgb, meta)
        if ok:
            self.fastplate_async_enqueued += 1
        else:
            self.fastplate_queue_dropped += 1
        return ok

    def _get_middle_slot_worker(self) -> MiddleSlotBatchQueueWorker | None:
        if not self.middle_slot_upl:
            return None
        if self.middle_slot_backend == "none":
            return None
        if self.middle_slot_worker is None:
            try:
                self.middle_slot_worker = MiddleSlotBatchQueueWorker(
                    device=self.middle_slot_device,
                    batch_size=self.middle_slot_batch_size,
                    min_batch_size=self.middle_slot_min_batch_size,
                    flush_timeout_ms=self.middle_slot_flush_timeout_ms,
                    queue_max=self.middle_slot_queue_max,
                    input_size=self.middle_slot_input_size,
                    max_crops_per_group=(1 if self.middle_slot_backend == "easyocr" and self.middle_slot_easyocr_single_crop else self.middle_slot_max_crops_per_group),
                    prototype_topk=3,
                    backend=self.middle_slot_backend,
                    middle_slot_hog_lbp_model=self.middle_slot_hog_lbp_model,
                    middle_slot_hog_lbp_topk=self.middle_slot_hog_lbp_topk,
                    middle_slot_hog_lbp_source_weight=self.middle_slot_hog_lbp_source_weight,
                    middle_slot_diagnostic_only=self.middle_slot_diagnostic_only,
                    cropper_mode=self.middle_slot_cropper_mode,
                    encoder_mode=self.middle_slot_encoder_mode,
                    easyocr_gpu=self.middle_slot_easyocr_gpu,
                    easyocr_batch_size=self.middle_slot_easyocr_batch_size,
                    easyocr_min_conf=self.middle_slot_easyocr_min_conf,
                    easyocr_allowlist=self.middle_slot_easyocr_allowlist,
                    easyocr_resize_scale=self.middle_slot_easyocr_resize_scale,
                    easyocr_border=self.middle_slot_easyocr_border,
                    easyocr_single_crop=self.middle_slot_easyocr_single_crop,
                    wide_core_x_pad_ratio=self.middle_slot_wide_core_x_pad_ratio,
                    wide_core_y_pad_ratio=self.middle_slot_wide_core_y_pad_ratio,
                    wide_core_min_width_ratio=self.middle_slot_wide_core_min_width_ratio,
                    wide_core_max_width_ratio=self.middle_slot_wide_core_max_width_ratio,
                    model_path=self.middle_slot_model_path,
                    prototype_path=self.middle_slot_prototype_path,
                    conf_thr=self.middle_slot_conf_thr,
                    margin_thr=self.middle_slot_margin_thr,
                    sim_center=self.middle_slot_sim_center,
                    sim_k=self.middle_slot_sim_k,
                    alpha_min=self.middle_slot_alpha_min,
                    alpha_max=self.middle_slot_alpha_max,
                    disable_prototype_update=self.middle_slot_disable_prototype_update,
                    debug_dir=self.middle_slot_debug_dir,
                    save_debug_crops=self.middle_slot_save_debug_crops,
                    gt_anchor_confused_fair=self.middle_slot_gt_anchor_confused_fair,
                    gt_items=self.middle_slot_gt_items,
                    gt_anchor_boost=self.middle_slot_gt_anchor_boost,
                    gt_confused_min_score=self.middle_slot_gt_confused_min_score,
                    gt_confused_min_support=self.middle_slot_gt_confused_min_support,
                    logger=self._log,
                )
                proto_encoder = getattr(self.middle_slot_worker, "prototype_encoder_mode", "")
                proto_label_count = getattr(self.middle_slot_worker, "prototype_label_count", 0)
                self._log(
                    f"[MIDDLE_SLOT_UPL] worker initialized device={self.middle_slot_device} batch_size={self.middle_slot_batch_size} "
                    f"backend={self.middle_slot_backend} preset={getattr(self, 'middle_slot_preset', 'none')} "
                    f"cropper={self.middle_slot_cropper_mode} diagnostic_only={1 if self.middle_slot_diagnostic_only else 0} "
                    f"input_size={self.middle_slot_input_size} max_crops_per_group={self.middle_slot_max_crops_per_group} "
                    f"encoder_mode={self.middle_slot_encoder_mode} hog_lbp_model={self.middle_slot_hog_lbp_model or ''} "
                    f"topk={self.middle_slot_hog_lbp_topk} source_weight={self.middle_slot_hog_lbp_source_weight} "
                    f"prototype_encoder_mode={proto_encoder or 'unknown'} prototype_path={self.middle_slot_prototype_path or ''} "
                    f"prototype_label_count={proto_label_count}"
                )
                print(
                    f"[MIDDLE_SLOT_UPL] worker initialized backend={self.middle_slot_backend} "
                    f"model={self.middle_slot_hog_lbp_model or self.middle_slot_model_path or ''} "
                    f"cropper={self.middle_slot_cropper_mode}",
                    flush=True,
                )
            except Exception as exc:
                self.middle_slot_trigger_worker_init_error = f"{type(exc).__name__}: {exc}"
                msg = f"[MIDDLE_SLOT_UPL] worker init failed: {type(exc).__name__}: {exc}"
                print(msg, flush=True)
                self._log(msg)
                return None
        return self.middle_slot_worker

    def _get_middle_slot_v32_runtime(self) -> MiddleSlotV32Runtime | None:
        if not getattr(self, "middle_slot_v32_enabled", False):
            return None
        if self.middle_slot_v32_runtime is None:
            try:
                self.middle_slot_v32_runtime = MiddleSlotV32Runtime(
                    self.middle_slot_v32_model,
                    device=self.middle_slot_v32_device,
                    batch_size=self.middle_slot_v32_batch_size,
                    topk=self.middle_slot_v32_topk,
                )
                self._log(
                    f"[MIDDLE_SLOT_V32] runtime initialized model={self.middle_slot_v32_model} "
                    f"device={self.middle_slot_v32_device} batch_size={self.middle_slot_v32_batch_size} topk={self.middle_slot_v32_topk}"
                )
            except Exception as exc:
                self._log(f"[MIDDLE_SLOT_V32] runtime init failed: {type(exc).__name__}: {exc}")
                self.middle_slot_v32_enabled = False
                return None
        return self.middle_slot_v32_runtime

    def _append_middle_slot_v32_diag(
        self,
        *,
        crop_bgr: np.ndarray | None,
        ocr_text: str,
        ocr_conf: float,
        source: str,
        meta: dict[str, object],
        topk: list[dict[str, object]] | None = None,
        margin: float = 0.0,
        added: bool = False,
        skip_reason: str = "unknown",
        infer_ms_per_crop: float = 0.0,
        batch_size: int = 0,
    ) -> None:
        topk_rows = list(topk or [])
        top1 = topk_rows[0] if len(topk_rows) > 0 else {}
        top2 = topk_rows[1] if len(topk_rows) > 1 else {}
        top3 = topk_rows[2] if len(topk_rows) > 2 else {}
        top1_score = float(top1.get("score", 0.0) or 0.0)
        h, w = (crop_bgr.shape[:2] if isinstance(crop_bgr, np.ndarray) and crop_bgr.ndim >= 2 else (0, 0))
        row = {
            "row_type": "middle_slot_v32_diagnostic",
            "ocr_source": "middle_slot_v32",
            "source": "middle_slot_v32",
            "frame_idx": int(meta.get("frame_idx", -1) or -1),
            "event_id": meta.get("event_id", meta.get("variant_group_key", meta.get("event_fusion_group_key", ""))),
            "track_id": int(meta.get("track_id", -1) or -1),
            "candidate_idx": int(meta.get("candidate_idx", -1) or -1),
            "crop_id": meta.get("crop_id", meta.get("candidate_idx", "")),
            "crop_source": source,
            "crop_score": meta.get("crop_score", meta.get("quality_score", meta.get("score", ""))),
            "crop_shape": f"{int(h)}x{int(w)}" if h and w else "",
            "middle_slot_v32_event_crop_count": int(meta.get("middle_slot_v32_event_crop_count", 0) or 0),
            "middle_slot_v32_pending_len": int(meta.get("middle_slot_v32_pending_len", len(getattr(self, "middle_slot_v32_pending", []) or [])) or 0),
            "middle_slot_v32_flush_reason": str(meta.get("middle_slot_v32_flush_reason", "") or ""),
            "middle_slot_v32_frame_idx": int(meta.get("frame_idx", -1) or -1),
            "middle_slot_v32_event_id": meta.get("event_id", meta.get("variant_group_key", meta.get("event_fusion_group_key", ""))),
            "middle_slot_v32_track_id": int(meta.get("track_id", -1) or -1),
            "middle_slot_v32_digit_skeleton": str(meta.get("middle_slot_v32_digit_skeleton", "") or ""),
            "middle_slot_v32_digit_skeleton_source": str(meta.get("middle_slot_v32_digit_skeleton_source", "") or ""),
            "middle_slot_v32_digit_skeleton_reason": str(meta.get("middle_slot_v32_digit_skeleton_reason", "") or ""),
            "ocr_text": str(ocr_text or ""),
            "ocr_conf": float(ocr_conf or 0.0),
            "middle_slot_v32_enabled": 1,
            "middle_slot_v32_model_path": str(getattr(self, "middle_slot_v32_model_path", getattr(self, "middle_slot_v32_model", "")) or ""),
            "middle_slot_v32_topk": int(getattr(self, "middle_slot_v32_topk", 3) or 3),
            "middle_slot_v32_top1": str(top1.get("ko", "") or ""),
            "middle_slot_v32_top1_roman": str(top1.get("roman", "") or ""),
            "middle_slot_v32_top1_ko": str(top1.get("ko", "") or ""),
            "middle_slot_v32_top1_score": top1_score,
            "middle_slot_v32_top2": str(top2.get("ko", "") or ""),
            "middle_slot_v32_top2_roman": str(top2.get("roman", "") or ""),
            "middle_slot_v32_top2_ko": str(top2.get("ko", "") or ""),
            "middle_slot_v32_top2_score": float(top2.get("score", 0.0) or 0.0),
            "middle_slot_v32_top3": str(top3.get("ko", "") or ""),
            "middle_slot_v32_top3_roman": str(top3.get("roman", "") or ""),
            "middle_slot_v32_top3_ko": str(top3.get("ko", "") or ""),
            "middle_slot_v32_top3_score": float(top3.get("score", 0.0) or 0.0),
            "middle_slot_v32_margin": float(margin),
            "middle_slot_v32_min_conf": float(getattr(self, "middle_slot_v32_min_conf", 0.20)),
            "middle_slot_v32_min_margin": float(getattr(self, "middle_slot_v32_min_margin", 0.03)),
            "middle_slot_v32_source_weight": float(getattr(self, "middle_slot_v32_source_weight", 0.04)),
            "middle_slot_v32_evidence_weight_top1": float(getattr(self, "middle_slot_v32_source_weight", 0.04)) * top1_score,
            "middle_slot_v32_added_evidence": int(added),
            "middle_slot_v32_skip_reason": skip_reason,
            "middle_slot_v32_batch_size": int(batch_size or 0),
            "middle_slot_v32_infer_ms": float(infer_ms_per_crop or 0.0),
        }
        self.middle_slot_v32_debug_rows.append(row)
        self.ocr_csv_rows.append(row)

    def _record_middle_slot_v32_skip(self, skip_reason: str) -> None:
        if skip_reason == "added":
            self.middle_slot_v32_added_evidence += 1
        elif skip_reason == "low_conf":
            self.middle_slot_v32_skipped_low_conf += 1
        elif skip_reason == "low_margin":
            self.middle_slot_v32_skipped_low_margin += 1
        elif skip_reason == "no_event_id":
            self.middle_slot_v32_skipped_no_event_id += 1
        elif skip_reason == "no_track_id":
            self.middle_slot_v32_skipped_no_track_id += 1
        elif skip_reason == "no_digit_skeleton":
            self.middle_slot_v32_skipped_no_digit_skeleton += 1
        elif skip_reason == "invalid_digit_skeleton":
            self.middle_slot_v32_skipped_invalid_digit_skeleton += 1
        elif skip_reason == "max_crops_per_event":
            self.middle_slot_v32_skipped_max_crops_per_event += 1
        elif skip_reason == "empty_crop":
            self.middle_slot_v32_skipped_empty_crop += 1
        elif skip_reason == "duplicate":
            self.middle_slot_v32_skipped_duplicate += 1
        elif skip_reason == "pending_not_flushed":
            self.middle_slot_v32_skipped_pending_not_flushed += 1
        elif skip_reason == "model_error":
            self.middle_slot_v32_skipped_model_error += 1
        elif skip_reason == "before_infer_unknown":
            self.middle_slot_v32_skipped_before_infer_unknown += 1
        else:
            self.middle_slot_v32_skipped_other += 1

    def _resolve_middle_slot_v32_digit_skeleton(self, meta: dict[str, object], ocr_text: str) -> dict[str, object]:
        candidates: list[dict[str, object]] = []
        text_sources = [("ocr_text", ocr_text)]
        for key in (
            "ocr_text",
            "raw_text",
            "corrected_text",
            "normalized_plate_candidate",
            "grammar_best_text",
            "final_output_plate",
            "variant_group_selected_skeleton",
            "skeleton_anchor_text",
            "variant_group_anchor_text",
            "ocr_normalized_text",
            "ocr_plate_text_final",
        ):
            text_sources.append((key, meta.get(key, "")))

        seen_texts: set[tuple[str, str]] = set()
        for source_name, raw_value in text_sources:
            text = str(raw_value or "").strip()
            if not text:
                continue
            seen_key = (source_name, text)
            if seen_key in seen_texts:
                continue
            seen_texts.add(seen_key)

            direct = re.sub(r"\s+", "", text)
            if "?" in direct:
                skeleton = direct
                if not _is_placeholder_digit_skeleton(skeleton):
                    candidates.append({
                        "skeleton": skeleton,
                        "score": 1.10,
                        "source": source_name,
                        "reason": "direct_digit_skeleton",
                        "source_text": text,
                    })

            for row in _generate_digit_skeleton_candidates(text):
                skeleton = str(row.get("skeleton", "") or "")
                if not skeleton or _is_placeholder_digit_skeleton(skeleton):
                    continue
                candidates.append({
                    "skeleton": skeleton,
                    "score": float(row.get("score", 0.0) or 0.0),
                    "source": source_name,
                    "reason": str(row.get("reason", "") or "generated_digit_skeleton"),
                    "source_text": str(row.get("source_text", text) or text),
                })

        best_by_skeleton: dict[str, dict[str, object]] = {}
        for row in candidates:
            skeleton = str(row.get("skeleton", "") or "")
            score = float(row.get("score", 0.0) or 0.0)
            if skeleton and (skeleton not in best_by_skeleton or score > float(best_by_skeleton[skeleton].get("score", 0.0) or 0.0)):
                best_by_skeleton[skeleton] = row
        if not best_by_skeleton:
            return {"skeleton": "", "source": "", "reason": "not_found", "score": 0.0}
        return max(best_by_skeleton.values(), key=lambda row: float(row.get("score", 0.0) or 0.0))

    def _handle_middle_slot_v32_result(self, item: dict[str, object], result: dict[str, object], infer_ms_per_crop: float, batch_size: int) -> bool:
        crop_bgr = item.get("crop_bgr")
        crop_arr = crop_bgr if isinstance(crop_bgr, np.ndarray) else None
        ocr_text = str(item.get("ocr_text", "") or "")
        ocr_conf = float(item.get("ocr_conf", 0.0) or 0.0)
        source = str(item.get("source", "middle_slot_v32") or "middle_slot_v32")
        meta = dict(item.get("meta", {}) or {})
        topk = list(result.get("topk") or [])
        top1_score = float(result.get("top1_score", 0.0) or 0.0)
        margin = float(result.get("margin", 0.0) or 0.0)
        track_id = int(meta.get("track_id", -1) or -1)
        frame_idx = int(meta.get("frame_idx", -1) or -1)
        skeleton_info = self._resolve_middle_slot_v32_digit_skeleton(meta, ocr_text)
        skeleton = str(skeleton_info.get("skeleton", "") or "")
        meta["middle_slot_v32_digit_skeleton"] = skeleton
        meta["middle_slot_v32_digit_skeleton_source"] = str(skeleton_info.get("source", "") or "")
        meta["middle_slot_v32_digit_skeleton_reason"] = str(skeleton_info.get("reason", "") or "")
        skip_reason = "added"
        if top1_score < self.middle_slot_v32_min_conf:
            skip_reason = "low_conf"
        elif margin < self.middle_slot_v32_min_margin:
            skip_reason = "low_margin"
        elif track_id < 0:
            skip_reason = "no_track_id"
        elif not skeleton:
            skip_reason = "no_digit_skeleton"
        elif "?" not in skeleton:
            skip_reason = "invalid_digit_skeleton"

        added_count = 0
        if skip_reason == "added":
            prefix, suffix = skeleton.split("?", 1)
            for row in topk:
                ko = str(row.get("ko", "") or "")[:1]
                score = float(row.get("score", 0.0) or 0.0)
                if not ko or ko not in VALID_KOR or score <= 0.0:
                    continue
                candidate = f"{prefix}{ko}{suffix}"
                self._append_string_observation(
                    track_id,
                    frame_idx=frame_idx,
                    raw_text=ocr_text,
                    corrected_text=candidate,
                    text=candidate,
                    ocr_conf=score,
                    febam_reliability=float(meta.get("febam_score", 0.5) or 0.5),
                    source="middle_slot_v32",
                    source_weight=self.middle_slot_v32_source_weight * score,
                )
                added_count += 1
            if added_count <= 0:
                skip_reason = "before_infer_unknown"

        added = skip_reason == "added" and added_count > 0
        self._record_middle_slot_v32_skip("added" if added else skip_reason)
        self._append_middle_slot_v32_diag(
            crop_bgr=crop_arr,
            ocr_text=ocr_text,
            ocr_conf=ocr_conf,
            source=source,
            meta=meta,
            topk=topk,
            margin=margin,
            added=added,
            skip_reason="added" if added else skip_reason,
            infer_ms_per_crop=infer_ms_per_crop,
            batch_size=batch_size,
        )
        return bool(added)

    def _flush_middle_slot_v32_pending(self, final: bool = False, flush_reason: str = "manual") -> int:
        if not getattr(self, "middle_slot_v32_enabled", False):
            return 0
        pending = getattr(self, "middle_slot_v32_pending", [])
        if not pending:
            return 0
        runtime = self._get_middle_slot_v32_runtime()
        batch = pending[: int(max(1, getattr(self, "middle_slot_v32_batch_size", 512)))]
        del pending[:len(batch)]
        for item in batch:
            item_meta = dict(item.get("meta", {}) or {})
            item_meta["middle_slot_v32_flush_reason"] = flush_reason
            item_meta["middle_slot_v32_pending_len"] = len(pending)
            item["meta"] = item_meta
            pending_key = item.get("pending_key")
            if pending_key is not None:
                self.middle_slot_v32_pending_keys.discard(pending_key)
        if final:
            self.middle_slot_v32_pending_final_flush_count += 1
        if runtime is None:
            for item in batch:
                self._record_middle_slot_v32_skip("model_error")
                self._append_middle_slot_v32_diag(
                    crop_bgr=item.get("crop_bgr") if isinstance(item.get("crop_bgr"), np.ndarray) else None,
                    ocr_text=str(item.get("ocr_text", "") or ""),
                    ocr_conf=float(item.get("ocr_conf", 0.0) or 0.0),
                    source=str(item.get("source", "middle_slot_v32") or "middle_slot_v32"),
                    meta=dict(item.get("meta", {}) or {}),
                    skip_reason="model_error",
                    batch_size=len(batch),
                )
            return 0
        valid_items = []
        crops = []
        for item in batch:
            crop = item.get("crop_bgr")
            if isinstance(crop, np.ndarray) and crop.size > 0:
                valid_items.append(item)
                crops.append(crop)
                continue
            self._record_middle_slot_v32_skip("empty_crop")
            self._append_middle_slot_v32_diag(
                crop_bgr=crop if isinstance(crop, np.ndarray) else None,
                ocr_text=str(item.get("ocr_text", "") or ""),
                ocr_conf=float(item.get("ocr_conf", 0.0) or 0.0),
                source=str(item.get("source", "middle_slot_v32") or "middle_slot_v32"),
                meta=dict(item.get("meta", {}) or {}),
                skip_reason="empty_crop",
                batch_size=len(batch),
            )
        if not crops:
            return 0
        try:
            start_t = time.perf_counter()
            results = runtime.predict_batch(crops)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        except Exception as exc:
            self._log(f"[MIDDLE_SLOT_V32] batch predict failed: {type(exc).__name__}: {exc}")
            for item in batch:
                self._record_middle_slot_v32_skip("model_error")
                self._append_middle_slot_v32_diag(
                    crop_bgr=item.get("crop_bgr") if isinstance(item.get("crop_bgr"), np.ndarray) else None,
                    ocr_text=str(item.get("ocr_text", "") or ""),
                    ocr_conf=float(item.get("ocr_conf", 0.0) or 0.0),
                    source=str(item.get("source", "middle_slot_v32") or "middle_slot_v32"),
                    meta=dict(item.get("meta", {}) or {}),
                    skip_reason="model_error",
                    batch_size=len(batch),
                )
            return 0
        self.middle_slot_v32_processed += len(crops)
        self.middle_slot_v32_batch_call_count += 1
        self.middle_slot_v32_total_infer_ms += elapsed_ms
        infer_ms_per_crop = elapsed_ms / max(1, len(crops))
        handled = 0
        for item, result in zip(valid_items, results):
            self._handle_middle_slot_v32_result(item, dict(result), infer_ms_per_crop, len(crops))
            handled += 1
        if handled < len(valid_items):
            for item in valid_items[handled:]:
                self._record_middle_slot_v32_skip("before_infer_unknown")
                self._append_middle_slot_v32_diag(
                    crop_bgr=item.get("crop_bgr") if isinstance(item.get("crop_bgr"), np.ndarray) else None,
                    ocr_text=str(item.get("ocr_text", "") or ""),
                    ocr_conf=float(item.get("ocr_conf", 0.0) or 0.0),
                    source=str(item.get("source", "middle_slot_v32") or "middle_slot_v32"),
                    meta=dict(item.get("meta", {}) or {}),
                    skip_reason="before_infer_unknown",
                    infer_ms_per_crop=infer_ms_per_crop,
                    batch_size=len(crops),
                )
        return handled

    def _maybe_flush_middle_slot_v32_pending(self, frame_idx: int, final: bool = False) -> int:
        pending_len = len(getattr(self, "middle_slot_v32_pending", []) or [])
        if pending_len <= 0:
            return 0
        if final:
            total = 0
            while getattr(self, "middle_slot_v32_pending", []):
                total += self._flush_middle_slot_v32_pending(final=True, flush_reason="final")
            return total
        if pending_len >= int(getattr(self, "middle_slot_v32_batch_size", 512) or 512):
            return self._flush_middle_slot_v32_pending(final=False, flush_reason="batch_size")
        min_batch_size = int(getattr(self, "middle_slot_v32_min_batch_size", 8) or 8)
        if pending_len >= min_batch_size:
            flush_every_frames = int(getattr(self, "middle_slot_v32_flush_every_frames", 5) or 5)
            last_flush = int(getattr(self, "_middle_slot_v32_last_flush_frame", -1) or -1)
            if frame_idx >= 0 and (last_flush < 0 or frame_idx - last_flush >= flush_every_frames):
                self._middle_slot_v32_last_flush_frame = frame_idx
                return self._flush_middle_slot_v32_pending(final=False, flush_reason="frame_interval_min_batch")
        return 0

    def _handle_middle_slot_v32_soft_evidence(self, crop_bgr: np.ndarray | None, *, ocr_text: str, ocr_conf: float, source: str, meta: dict[str, object]) -> bool:
        if not getattr(self, "middle_slot_v32_enabled", False):
            return False
        meta = dict(meta or {})
        if crop_bgr is None:
            self._record_middle_slot_v32_skip("empty_crop")
            self._append_middle_slot_v32_diag(crop_bgr=None, ocr_text=ocr_text, ocr_conf=ocr_conf, source=source, meta=meta, skip_reason="empty_crop")
            return False
        event_key = str(meta.get("event_id", meta.get("variant_group_key", meta.get("event_fusion_group_key", meta.get("track_id", "global")))) or "global")
        frame_idx = int(meta.get("frame_idx", -1) or -1)
        candidate_idx = int(meta.get("candidate_idx", -1) or -1)
        pending_key = (event_key, frame_idx, candidate_idx, str(source or ""), str(ocr_text or ""))
        if pending_key in self.middle_slot_v32_pending_keys:
            meta["middle_slot_v32_event_crop_count"] = int(self.middle_slot_v32_crops_by_event.get(event_key, 0))
            meta["middle_slot_v32_pending_len"] = len(getattr(self, "middle_slot_v32_pending", []) or [])
            self._record_middle_slot_v32_skip("duplicate")
            self._append_middle_slot_v32_diag(crop_bgr=crop_bgr, ocr_text=ocr_text, ocr_conf=ocr_conf, source=source, meta=meta, skip_reason="duplicate")
            return False
        seen_for_event = int(self.middle_slot_v32_crops_by_event.get(event_key, 0))
        if seen_for_event >= int(getattr(self, "middle_slot_v32_max_crops_per_event", 8) or 8):
            meta["middle_slot_v32_event_crop_count"] = seen_for_event
            meta["middle_slot_v32_pending_len"] = len(getattr(self, "middle_slot_v32_pending", []) or [])
            self._record_middle_slot_v32_skip("max_crops_per_event")
            self._append_middle_slot_v32_diag(crop_bgr=crop_bgr, ocr_text=ocr_text, ocr_conf=ocr_conf, source=source, meta=meta, skip_reason="max_crops_per_event")
            return False
        event_crop_count = seen_for_event + 1
        self.middle_slot_v32_crops_by_event[event_key] = event_crop_count
        meta["middle_slot_v32_event_crop_count"] = event_crop_count
        meta["middle_slot_v32_pending_len"] = len(getattr(self, "middle_slot_v32_pending", []) or []) + 1
        self.middle_slot_v32_pending_keys.add(pending_key)
        self.middle_slot_v32_pending.append({
            "crop_bgr": np.ascontiguousarray(crop_bgr),
            "ocr_text": str(ocr_text or ""),
            "ocr_conf": float(ocr_conf or 0.0),
            "source": source,
            "meta": meta,
            "pending_key": pending_key,
        })
        self._maybe_flush_middle_slot_v32_pending(frame_idx=frame_idx, final=False)
        return False

    def _middle_slot_anchor_text_from_meta(self, ocr_text: str, meta: dict[str, object]) -> tuple[str, str]:
        """Resolve a prefix/suffix digit anchor text for HOG+LBP middle-slot crops."""
        candidates: list[tuple[str, object]] = [("ocr_text", ocr_text)]
        for key in (
            "normalized_plate_candidate",
            "grammar_best_text",
            "final_output_plate",
            "ocr_plate_text_final",
            "corrected_text",
            "raw_text",
            "variant_group_anchor_text",
            "skeleton_anchor_text",
            "variant_group_selected_skeleton",
            "middle_slot_digit_skeleton",
            "middle_slot_v32_digit_skeleton",
        ):
            candidates.append((key, (meta or {}).get(key, "")))

        for source, value in candidates:
            raw_text = re.sub(r"\s+", "", str(value or ""))
            if not raw_text:
                continue
            if "?" in raw_text:
                match = re.fullmatch(r"(\d{2,3})\?(\d{4})", raw_text)
                if match:
                    return f"{match.group(1)}X{match.group(2)}", f"{source}_skeleton_x"
            text = self._normalize_ocr_text(raw_text)
            if not text:
                continue
            if re.fullmatch(r"\d{2,3}[A-Z가-힣]?\d{4}", text):
                return text, source
            digits = re.sub(r"\D", "", text)
            if len(digits) in {6, 7, 8}:
                return digits, f"{source}_digits"
        return "", "not_found"

    def _enqueue_middle_slot_upl(self, crop_bgr: np.ndarray | None, *, ocr_text: str, ocr_conf: float, source: str, meta: dict[str, object]) -> bool:
        self.middle_slot_trigger_seen += 1
        meta = dict(meta or {})
        anchor_text = str(ocr_text or "")
        anchor_source = "ocr_text"
        if self.middle_slot_backend == "hog_lbp_gpu":
            fallback_text, fallback_source = self._middle_slot_anchor_text_from_meta(anchor_text, meta)
            if fallback_text:
                if not anchor_text or fallback_text != anchor_text:
                    self.middle_slot_trigger_fallback_text_used += 1
                anchor_text = fallback_text
                anchor_source = fallback_source

        v32_ok = self._handle_middle_slot_v32_soft_evidence(crop_bgr, ocr_text=anchor_text, ocr_conf=ocr_conf, source=source, meta=dict(meta or {}))
        if not self.middle_slot_upl:
            return bool(v32_ok)
        if crop_bgr is None and not str(meta.get("fused_image_path", "") or ""):
            self.middle_slot_trigger_skipped_no_crop += 1
            self.middle_slot_csv_diagnostics.append({
                "middle_slot_attempted": 1,
                "middle_slot_source": source,
                "middle_slot_status": "skipped",
                "middle_slot_skip_reason": "missing_crop_or_path",
                "middle_slot_generated_candidate": "",
            })
            return False
        self.middle_slot_trigger_crop_present += 1
        if not str(anchor_text or ""):
            self.middle_slot_trigger_skipped_no_text += 1
            if self.middle_slot_trigger_skipped_no_text <= 5:
                print(
                    f"[MIDDLE_SLOT_TRIGGER_SKIP] no_text source={source} "
                    f"frame={meta.get('frame_idx')} track={meta.get('track_id')} keys={list(meta.keys())[:20]}",
                    flush=True,
                )
            return False
        self.middle_slot_trigger_text_present += 1
        worker = self._get_middle_slot_worker()
        if worker is None:
            self.middle_slot_trigger_worker_none += 1
            if self.middle_slot_trigger_worker_none <= 5:
                print(
                    f"[MIDDLE_SLOT_TRIGGER_WORKER_NONE] backend={self.middle_slot_backend} "
                    f"model={self.middle_slot_hog_lbp_model} frame={meta.get('frame_idx')} track={meta.get('track_id')}",
                    flush=True,
                )
            return False
        task_meta = dict(meta or {})
        task_meta.update({
            "ocr_text": anchor_text,
            "ocr_conf": float(ocr_conf),
            "middle_slot_anchor_text": anchor_text,
            "middle_slot_anchor_source": anchor_source,
            "middle_slot_source": source,
            "source": source,
            "source_weight": float(task_meta.get("source_weight", self.middle_slot_source_weight) or self.middle_slot_source_weight),
        })
        if self.middle_slot_backend == "hog_lbp_gpu":
            task_meta["source_weight"] = float(self.middle_slot_hog_lbp_source_weight)
            task_meta["middle_slot_diagnostic_only"] = int(self.middle_slot_diagnostic_only)
            task_meta["middle_slot_model"] = str(self.middle_slot_hog_lbp_model or "")
        ok = worker.enqueue(crop_bgr, task_meta)
        if ok:
            self.middle_slot_trigger_enqueue_success += 1
        else:
            self.middle_slot_trigger_enqueue_failed += 1
            if self.middle_slot_trigger_enqueue_failed <= 5:
                print(
                    f"[MIDDLE_SLOT_TRIGGER_ENQUEUE_FAIL] backend={self.middle_slot_backend} "
                    f"anchor={anchor_text} anchor_source={anchor_source} frame={meta.get('frame_idx')} track={meta.get('track_id')}",
                    flush=True,
                )
        return ok


    def _middle_slot_v32_stats(self) -> dict[str, float]:
        processed = float(getattr(self, "middle_slot_v32_processed", 0))
        batch_calls = float(getattr(self, "middle_slot_v32_batch_call_count", 0))
        total_ms = float(getattr(self, "middle_slot_v32_total_infer_ms", 0.0))
        return {
            "middle_slot_v32_enabled": 1.0 if getattr(self, "middle_slot_v32_enabled", False) else 0.0,
            "middle_slot_v32_model_path": str(getattr(self, "middle_slot_v32_model_path", getattr(self, "middle_slot_v32_model", "")) or ""),
            "middle_slot_v32_processed": processed,
            "middle_slot_v32_batch_call_count": batch_calls,
            "middle_slot_v32_avg_batch_size": processed / max(1.0, batch_calls),
            "middle_slot_v32_min_batch_size": float(getattr(self, "middle_slot_v32_min_batch_size", 8)),
            "middle_slot_v32_flush_every_frames": float(getattr(self, "middle_slot_v32_flush_every_frames", 5)),
            "middle_slot_v32_pending_final_flush_count": float(getattr(self, "middle_slot_v32_pending_final_flush_count", 0)),
            "middle_slot_v32_total_infer_ms": total_ms,
            "middle_slot_v32_avg_ms_per_crop": total_ms / max(1.0, processed),
            "middle_slot_v32_added_evidence": float(getattr(self, "middle_slot_v32_added_evidence", 0)),
            "middle_slot_v32_skipped_low_conf": float(getattr(self, "middle_slot_v32_skipped_low_conf", 0)),
            "middle_slot_v32_skipped_low_margin": float(getattr(self, "middle_slot_v32_skipped_low_margin", 0)),
            "middle_slot_v32_skipped_no_event_id": float(getattr(self, "middle_slot_v32_skipped_no_event_id", 0)),
            "middle_slot_v32_skipped_no_track_id": float(getattr(self, "middle_slot_v32_skipped_no_track_id", 0)),
            "middle_slot_v32_skipped_no_digit_skeleton": float(getattr(self, "middle_slot_v32_skipped_no_digit_skeleton", 0)),
            "middle_slot_v32_skipped_invalid_digit_skeleton": float(getattr(self, "middle_slot_v32_skipped_invalid_digit_skeleton", 0)),
            "middle_slot_v32_skipped_max_crops_per_event": float(getattr(self, "middle_slot_v32_skipped_max_crops_per_event", 0)),
            "middle_slot_v32_skipped_empty_crop": float(getattr(self, "middle_slot_v32_skipped_empty_crop", 0)),
            "middle_slot_v32_skipped_duplicate": float(getattr(self, "middle_slot_v32_skipped_duplicate", 0)),
            "middle_slot_v32_skipped_pending_not_flushed": float(getattr(self, "middle_slot_v32_skipped_pending_not_flushed", 0)) + float(len(getattr(self, "middle_slot_v32_pending", []) or [])),
            "middle_slot_v32_skipped_model_error": float(getattr(self, "middle_slot_v32_skipped_model_error", 0)),
            "middle_slot_v32_skipped_before_infer_unknown": float(getattr(self, "middle_slot_v32_skipped_before_infer_unknown", 0)),
            "middle_slot_v32_skipped_other": float(getattr(self, "middle_slot_v32_skipped_other", 0)),
        }

    def _middle_slot_trigger_stats(self) -> dict[str, object]:
        return {
            "middle_slot_trigger_seen": float(getattr(self, "middle_slot_trigger_seen", 0)),
            "middle_slot_trigger_crop_present": float(getattr(self, "middle_slot_trigger_crop_present", 0)),
            "middle_slot_trigger_text_present": float(getattr(self, "middle_slot_trigger_text_present", 0)),
            "middle_slot_trigger_fallback_text_used": float(getattr(self, "middle_slot_trigger_fallback_text_used", 0)),
            "middle_slot_trigger_skipped_no_crop": float(getattr(self, "middle_slot_trigger_skipped_no_crop", 0)),
            "middle_slot_trigger_skipped_no_text": float(getattr(self, "middle_slot_trigger_skipped_no_text", 0)),
            "middle_slot_trigger_worker_none": float(getattr(self, "middle_slot_trigger_worker_none", 0)),
            "middle_slot_trigger_enqueue_success": float(getattr(self, "middle_slot_trigger_enqueue_success", 0)),
            "middle_slot_trigger_enqueue_failed": float(getattr(self, "middle_slot_trigger_enqueue_failed", 0)),
            "middle_slot_trigger_worker_init_error": str(getattr(self, "middle_slot_trigger_worker_init_error", "") or ""),
        }

    def _middle_slot_stats(self) -> dict[str, float]:
        if self.middle_slot_worker is None:
            return {
                "middle_slot_enabled": 1.0 if self.middle_slot_upl else 0.0,
                "middle_slot_cuda_used": 1.0 if str(self.middle_slot_device).startswith("cuda") and torch.cuda.is_available() else 0.0,
                "middle_slot_enqueued": 0.0,
                "middle_slot_processed": 0.0,
                "middle_slot_batch_call_count": 0.0,
                "middle_slot_avg_batch_size": 0.0,
                "middle_slot_total_infer_ms": 0.0,
                "middle_slot_avg_ms_per_crop": 0.0,
                "middle_slot_queue_dropped": 0.0,
                "middle_slot_skip_no_digit_anchor": 0.0,
                "middle_slot_skip_crop_failed": 0.0,
                "middle_slot_skip_no_prototype": 0.0,
                "middle_slot_update_applied_count": 0.0,
                "middle_slot_update_skipped_count": 0.0,
                "middle_slot_results_drained": float(self.middle_slot_results_drained),
                "middle_slot_backend": getattr(self, "middle_slot_backend", "upl"),
                "middle_slot_hog_lbp_model": str(getattr(self, "middle_slot_hog_lbp_model", "") or ""),
                "middle_slot_hog_lbp_topk": float(getattr(self, "middle_slot_hog_lbp_topk", 3)),
                "middle_slot_hog_lbp_source_weight": float(getattr(self, "middle_slot_hog_lbp_source_weight", 0.10)),
                "middle_slot_hog_lbp_min_conf": float(getattr(self, "middle_slot_hog_lbp_min_conf", 0.07)),
                "middle_slot_hog_lbp_min_margin": float(getattr(self, "middle_slot_hog_lbp_min_margin", 0.005)),
                "middle_slot_hog_lbp_min_crop_score": float(getattr(self, "middle_slot_hog_lbp_min_crop_score", 0.40)),
                "middle_slot_easyocr_single_crop": 1.0 if getattr(self, "middle_slot_easyocr_single_crop", False) else 0.0,
                "middle_slot_easyocr_processed": 0.0,
                "middle_slot_easyocr_batch_call_count": 0.0,
                "middle_slot_easyocr_avg_batch_size": 0.0,
                "middle_slot_easyocr_total_ms": 0.0,
                "middle_slot_easyocr_avg_ms_per_crop": 0.0,
                "middle_slot_easyocr_error_count": 0.0,
                "middle_slot_wide_core_x_pad_ratio": float(getattr(self, "middle_slot_wide_core_x_pad_ratio", 0.16)),
                "middle_slot_wide_core_y_pad_ratio": float(getattr(self, "middle_slot_wide_core_y_pad_ratio", 0.15)),
                "middle_slot_wide_core_crop_count": 0.0,
                "middle_slot_encoder_mode": getattr(self, "middle_slot_encoder_mode", "hog_torch"),
                "middle_slot_prototype_encoder_mode": "",
                "middle_slot_prototype_path": str(getattr(self, "middle_slot_prototype_path", "") or ""),
                "middle_slot_prototype_label_count": 0.0,
                "middle_slot_evidence_added": float(self.middle_slot_evidence_added),
                "middle_slot_evidence_skipped": float(self.middle_slot_evidence_skipped),
                "middle_slot_gt_anchor_confused_fair": 1.0 if getattr(self, "middle_slot_gt_anchor_confused_fair", False) else 0.0,
                "middle_slot_gt_plate_count": float(len(getattr(self, "middle_slot_gt_items", []) or [])),
                "middle_slot_gt_anchor_boost": float(getattr(self, "middle_slot_gt_anchor_boost", 0.0)),
                "middle_slot_gt_confused_evidence_added": float(getattr(self, "middle_slot_gt_confused_evidence_added", 0)),
                "middle_slot_gt_confused_evidence_skipped": float(getattr(self, "middle_slot_gt_confused_evidence_skipped", 0)),
                "middle_slot_diagnostic_only": 1.0 if getattr(self, "middle_slot_diagnostic_only", False) else 0.0,
                "middle_slot_evidence_diagnostic_only_skipped": float(getattr(self, "middle_slot_evidence_diagnostic_only_skipped", 0)),
                **self._middle_slot_v32_stats(),
                **self._middle_slot_trigger_stats(),
                "middle_slot_queue_depth": 0.0,
            }
        stats = self.middle_slot_worker.stats()
        try:
            queue_depth = float(self.middle_slot_worker.output_queue.qsize())
        except Exception:
            queue_depth = 0.0
        stats["middle_slot_results_drained"] = float(self.middle_slot_results_drained)
        stats["middle_slot_backend"] = getattr(self, "middle_slot_backend", "upl")
        stats["middle_slot_hog_lbp_model"] = str(getattr(self, "middle_slot_hog_lbp_model", "") or "")
        stats["middle_slot_hog_lbp_topk"] = float(getattr(self, "middle_slot_hog_lbp_topk", 3))
        stats["middle_slot_hog_lbp_source_weight"] = float(getattr(self, "middle_slot_hog_lbp_source_weight", 0.10))
        stats["middle_slot_hog_lbp_min_conf"] = float(getattr(self, "middle_slot_hog_lbp_min_conf", 0.07))
        stats["middle_slot_hog_lbp_min_margin"] = float(getattr(self, "middle_slot_hog_lbp_min_margin", 0.005))
        stats["middle_slot_hog_lbp_min_crop_score"] = float(getattr(self, "middle_slot_hog_lbp_min_crop_score", 0.40))
        stats["middle_slot_easyocr_single_crop"] = 1.0 if getattr(self, "middle_slot_easyocr_single_crop", False) else 0.0
        stats["middle_slot_wide_core_x_pad_ratio"] = float(getattr(self, "middle_slot_wide_core_x_pad_ratio", 0.16))
        stats["middle_slot_wide_core_y_pad_ratio"] = float(getattr(self, "middle_slot_wide_core_y_pad_ratio", 0.15))
        stats["middle_slot_wide_core_crop_count"] = float(stats.get("middle_slot_wide_core_crop_count", 0.0))
        stats["middle_slot_encoder_mode"] = getattr(self, "middle_slot_encoder_mode", "hog_torch") if getattr(self, "middle_slot_backend", "upl") == "upl" else ("hog_lbp_gpu" if getattr(self, "middle_slot_backend", "upl") == "hog_lbp_gpu" else "easyocr")
        stats["middle_slot_prototype_encoder_mode"] = getattr(self.middle_slot_worker, "prototype_encoder_mode", "")
        stats["middle_slot_prototype_path"] = str(getattr(self, "middle_slot_prototype_path", "") or "")
        stats["middle_slot_prototype_label_count"] = float(getattr(self.middle_slot_worker, "prototype_label_count", 0))
        stats["middle_slot_evidence_added"] = float(self.middle_slot_evidence_added)
        stats["middle_slot_evidence_skipped"] = float(self.middle_slot_evidence_skipped)
        stats["middle_slot_gt_anchor_confused_fair"] = 1.0 if getattr(self, "middle_slot_gt_anchor_confused_fair", False) else 0.0
        stats["middle_slot_gt_plate_count"] = float(len(getattr(self, "middle_slot_gt_items", []) or []))
        stats["middle_slot_gt_anchor_boost"] = float(getattr(self, "middle_slot_gt_anchor_boost", 0.0))
        stats["middle_slot_gt_confused_evidence_added"] = float(getattr(self, "middle_slot_gt_confused_evidence_added", 0))
        stats["middle_slot_gt_confused_evidence_skipped"] = float(getattr(self, "middle_slot_gt_confused_evidence_skipped", 0))
        stats["middle_slot_diagnostic_only"] = 1.0 if getattr(self, "middle_slot_diagnostic_only", False) else 0.0
        stats["middle_slot_evidence_diagnostic_only_skipped"] = float(getattr(self, "middle_slot_evidence_diagnostic_only_skipped", 0))
        stats.update(self._middle_slot_v32_stats())
        stats.update(self._middle_slot_trigger_stats())
        stats["middle_slot_queue_depth"] = queue_depth
        return stats

    def _finalize_fastplate_async_for_middle_slot(self) -> None:
        worker = self.fastplate_async_worker
        if worker is None:
            return
        try:
            worker.stop(timeout=5.0)
        except Exception:
            self._log_exception("FastPlate async worker finalize_stop_exception")
        while True:
            drained = self._drain_fastplate_results(max_items=None)
            if drained <= 0:
                break

    def _log_middle_slot_v32_profile_once(self) -> None:
        if not getattr(self, "middle_slot_v32_profile", False):
            return
        if getattr(self, "_middle_slot_v32_profile_logged", False):
            return
        middle_stats = self._middle_slot_stats()
        v32_msg = (
            "[MIDDLE_SLOT_V32_PROFILE]\n"
            f"enabled={int(middle_stats.get('middle_slot_v32_enabled', 0.0))}\n"
            f"processed={int(middle_stats.get('middle_slot_v32_processed', 0.0))}\n"
            f"batch_calls={int(middle_stats.get('middle_slot_v32_batch_call_count', 0.0))}\n"
            f"avg_batch={float(middle_stats.get('middle_slot_v32_avg_batch_size', 0.0)):.3f}\n"
            f"total_infer_ms={float(middle_stats.get('middle_slot_v32_total_infer_ms', 0.0)):.3f}\n"
            f"avg_ms_per_crop={float(middle_stats.get('middle_slot_v32_avg_ms_per_crop', 0.0)):.6f}\n"
            f"added={int(middle_stats.get('middle_slot_v32_added_evidence', 0.0))}\n"
            f"skipped_low_conf={int(middle_stats.get('middle_slot_v32_skipped_low_conf', 0.0))}\n"
            f"skipped_low_margin={int(middle_stats.get('middle_slot_v32_skipped_low_margin', 0.0))}\n"
            f"skipped_other={int(middle_stats.get('middle_slot_v32_skipped_other', 0.0))}\n"
            f"skipped_no_event_id={int(middle_stats.get('middle_slot_v32_skipped_no_event_id', 0.0))}\n"
            f"skipped_no_track_id={int(middle_stats.get('middle_slot_v32_skipped_no_track_id', 0.0))}\n"
            f"skipped_no_digit_skeleton={int(middle_stats.get('middle_slot_v32_skipped_no_digit_skeleton', 0.0))}\n"
            f"skipped_invalid_digit_skeleton={int(middle_stats.get('middle_slot_v32_skipped_invalid_digit_skeleton', 0.0))}\n"
            f"skipped_max_crops_per_event={int(middle_stats.get('middle_slot_v32_skipped_max_crops_per_event', 0.0))}\n"
            f"skipped_empty_crop={int(middle_stats.get('middle_slot_v32_skipped_empty_crop', 0.0))}\n"
            f"skipped_duplicate={int(middle_stats.get('middle_slot_v32_skipped_duplicate', 0.0))}\n"
            f"skipped_pending_not_flushed={int(middle_stats.get('middle_slot_v32_skipped_pending_not_flushed', 0.0))}\n"
            f"skipped_model_error={int(middle_stats.get('middle_slot_v32_skipped_model_error', 0.0))}\n"
            f"skipped_before_infer_unknown={int(middle_stats.get('middle_slot_v32_skipped_before_infer_unknown', 0.0))}"
        )
        print(v32_msg, flush=True)
        self._log(v32_msg)
        self._middle_slot_v32_profile_logged = True

    def finalize_middle_slot_upl(self) -> None:
        worker = self.middle_slot_worker
        if worker is None:
            return
        try:
            worker.stop(timeout=5.0)
        except Exception:
            self._log_exception("MiddleSlotUPL worker finalize_stop_exception")
        while True:
            drained = self._drain_middle_slot_results(max_items=None)
            if drained <= 0:
                break

    def finalize_async_ocr_outputs(self) -> None:
        if getattr(self, "_async_ocr_outputs_finalized", False):
            return
        self._finalize_fastplate_async_for_middle_slot()
        self._maybe_flush_middle_slot_v32_pending(final=True, frame_idx=-1)
        self.finalize_middle_slot_upl()
        self._log_middle_slot_v32_profile_once()
        self._async_ocr_outputs_finalized = True

    def _drain_middle_slot_results(self, max_items: int | None = None) -> int:
        worker = self.middle_slot_worker
        if worker is None:
            return 0
        drained = 0
        while max_items is None or drained < int(max_items):
            try:
                item = worker.output_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_middle_slot_result(item)
            drained += 1
        self.middle_slot_results_drained += drained
        return drained



    def _resolve_middle_slot_group_key(self, meta: dict[str, object], track_id) -> tuple[str, str]:
        for key in [
            "event_fusion_group_key",
            "variant_group_key",
            "group_id",
            "segment_id",
            "event_id",
            "motion_group_id",
            "track_group_id",
        ]:
            value = meta.get(key, "")
            text = str(value or "").strip()
            if text and text.lower() not in {"none", "nan", "-1"}:
                return text, key
        return f"track:{track_id}", "track_id_fallback"

    def _update_middle_slot_group_skeleton_bank(
        self,
        group_key: str,
        frame_idx: int,
        texts: list[tuple[str, str]],
    ) -> dict[str, dict[str, object]]:
        bank = self._middle_slot_group_skeleton_bank.setdefault(str(group_key), {})
        for source, text in texts:
            for candidate in _generate_digit_skeleton_candidates(text):
                skeleton = str(candidate.get("skeleton", ""))
                if not skeleton:
                    continue
                score = float(candidate.get("score", 0.0))
                rec = bank.setdefault(skeleton, {
                    "support": 0.0,
                    "count": 0,
                    "last_frame": int(frame_idx),
                    "sources": set(),
                    "reasons": {},
                })
                rec["support"] = float(rec.get("support", 0.0)) * 0.95 + score
                rec["count"] = int(rec.get("count", 0)) + 1
                rec["last_frame"] = int(frame_idx)
                sources = rec.setdefault("sources", set())
                if not isinstance(sources, set):
                    sources = set(str(sources).split("|")) if sources else set()
                    rec["sources"] = sources
                sources.add(str(source or "runtime_ocr"))
                reasons = rec.setdefault("reasons", {})
                reason = str(candidate.get("reason", ""))
                reasons[reason] = int(reasons.get(reason, 0)) + 1
        return bank

    @staticmethod
    def _serialize_middle_slot_skeleton_bank(bank: dict[str, dict[str, object]]) -> str:
        rows: list[dict[str, object]] = []
        for skeleton, rec in sorted(bank.items(), key=lambda item: float(item[1].get("support", 0.0) or 0.0), reverse=True):
            sources = rec.get("sources", set())
            if isinstance(sources, set):
                sources_text = "|".join(sorted(str(v) for v in sources if str(v)))
            else:
                sources_text = str(sources or "")
            rows.append({
                "skeleton": skeleton,
                "support": round(float(rec.get("support", 0.0) or 0.0), 6),
                "count": int(rec.get("count", 0) or 0),
                "sources": sources_text,
                "last_frame": int(rec.get("last_frame", -1) or -1),
            })
        return json.dumps(rows, ensure_ascii=False)

    def _update_middle_slot_char_memory(
        self,
        group_key: str,
        skeleton: str,
        frame_idx: int,
        topk_rows: list[tuple[str, float]],
        crop_score: float,
        crop_type: str,
        source: str,
    ) -> dict[str, object]:
        group_bank = self._middle_slot_char_memory.setdefault(str(group_key), {})
        rec = group_bank.setdefault(str(skeleton), {
            "char_scores": {},
            "char_counts": {},
            "last_frame": int(frame_idx),
            "support": 0.0,
            "sources": set(),
        })
        decay = 0.95
        char_scores = rec.setdefault("char_scores", {})
        char_counts = rec.setdefault("char_counts", {})
        for ko in list(char_scores.keys()):
            char_scores[ko] = float(char_scores.get(ko, 0.0)) * decay
        crop_type_weight = 1.0 if str(crop_type or "") == "anchor_projection_refine" else 0.35
        update_support = 0.0
        for ko, topk_score in topk_rows:
            ch = str(ko or "")[:1]
            if not ch or ch not in VALID_KOR:
                continue
            evidence = float(topk_score) * float(crop_score) * crop_type_weight
            if evidence <= 0.0:
                continue
            char_scores[ch] = float(char_scores.get(ch, 0.0)) + evidence
            char_counts[ch] = int(char_counts.get(ch, 0)) + 1
            update_support += evidence
            for neighbor in _hog_lbp_confused_neighbors(ch):
                char_scores[neighbor] = float(char_scores.get(neighbor, 0.0)) + evidence * 0.25
        rec["support"] = float(rec.get("support", 0.0)) * decay + update_support
        rec["last_frame"] = int(frame_idx)
        sources = rec.setdefault("sources", set())
        if not isinstance(sources, set):
            sources = set(str(sources).split("|")) if sources else set()
            rec["sources"] = sources
        sources.add(str(source or "middle_slot_hog_lbp_gpu"))
        return rec

    @staticmethod
    def _middle_slot_char_memory_summary(rec: dict[str, object] | None) -> dict[str, object]:
        if not rec:
            return {
                "best": "",
                "best_score": 0.0,
                "best_count": 0,
                "second": "",
                "second_score": 0.0,
                "margin": 0.0,
                "support": 0.0,
                "support_frames": 0,
                "second_score_value": 0.0,
                "sources": "",
                "json": "",
            }
        scores = dict(rec.get("char_scores", {}) or {})
        counts = dict(rec.get("char_counts", {}) or {})
        ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
        best, best_score = (ranked[0][0], float(ranked[0][1])) if ranked else ("", 0.0)
        second, second_score = (ranked[1][0], float(ranked[1][1])) if len(ranked) > 1 else ("", 0.0)
        sources = rec.get("sources", set())
        if isinstance(sources, set):
            sources_text = "|".join(sorted(str(v) for v in sources if str(v)))
        else:
            sources_text = str(sources or "")
        payload = {
            "char_scores": {str(k): float(v) for k, v in scores.items()},
            "char_counts": {str(k): int(v) for k, v in counts.items()},
            "support": float(rec.get("support", 0.0) or 0.0),
            "last_frame": int(rec.get("last_frame", -1) or -1),
            "sources": sources_text,
        }
        return {
            "best": best,
            "best_score": best_score,
            "best_count": int(counts.get(best, 0) or 0) if best else 0,
            "second": second,
            "second_score": second_score,
            "margin": best_score - second_score,
            "support": float(rec.get("support", 0.0) or 0.0),
            "support_frames": int(sum(int(v) for v in counts.values())) if counts else 0,
            "second_score_value": second_score,
            "sources": sources_text,
            "json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        }

    def _handle_middle_slot_result(self, item: MiddleSlotResult) -> None:
        meta = dict(item.meta or {})
        topk_text = topk_to_jsonable(item.middle_slot_topk)
        diag_backend = str(meta.get("middle_slot_backend", getattr(self, "middle_slot_backend", "upl")))
        diag_is_hog_lbp = diag_backend == "hog_lbp_gpu"
        topk_actual = meta.get("middle_slot_topk_actual", len(item.middle_slot_topk or []))
        diag = {
            "frame_idx": item.frame_idx,
            "track_id": item.track_id,
            "candidate_idx": item.candidate_idx,
            "middle_slot_attempted": 1,
            "middle_slot_source": item.source,
            "event_fusion_group_key": meta.get("event_fusion_group_key", ""),
            "variant_group_key": meta.get("variant_group_key", ""),
            "variant_group_anchor_text": meta.get("variant_group_anchor_text", ""),
            "variant_group_merge_reason": meta.get("variant_group_merge_reason", ""),
            "variant_group_selected_skeleton": meta.get("variant_group_selected_skeleton", ""),
            "skeleton_anchor_used": meta.get("skeleton_anchor_used", ""),
            "skeleton_anchor_pattern": meta.get("skeleton_anchor_pattern", ""),
            "skeleton_anchor_text": meta.get("skeleton_anchor_text", ""),
            "skeleton_anchor_support": meta.get("skeleton_anchor_support", ""),
            "group_id": meta.get("group_id", ""),
            "segment_id": meta.get("segment_id", ""),
            "event_id": meta.get("event_id", ""),
            "motion_group_id": meta.get("motion_group_id", ""),
            "track_group_id": meta.get("track_group_id", ""),
            "row_type": "middle_slot_diagnostic",
            "middle_slot_evidence": 0,
            "middle_slot_status": "accepted" if item.evidence_accepted else "skipped",
            "middle_slot_crop_path": meta.get("middle_slot_crop_path", ""),
            "middle_slot_crop_save_error": meta.get("middle_slot_crop_save_error", ""),
            "middle_slot_backend": diag_backend,
            "middle_slot_model": meta.get("middle_slot_model", getattr(self, "middle_slot_hog_lbp_model", "")),
            "middle_slot_cropper_mode": meta.get("middle_slot_cropper_mode", getattr(self, "middle_slot_cropper_mode", "")),
            "middle_slot_crop_type": meta.get("middle_slot_crop_type", ""),
            "middle_slot_crop_score": meta.get("middle_slot_crop_score", ""),
            "middle_slot_crop_x1": meta.get("middle_slot_crop_x1", ""),
            "middle_slot_crop_y1": meta.get("middle_slot_crop_y1", ""),
            "middle_slot_crop_x2": meta.get("middle_slot_crop_x2", ""),
            "middle_slot_crop_y2": meta.get("middle_slot_crop_y2", ""),
            "middle_slot_crop_reason": meta.get("middle_slot_crop_reason", ""),
            "middle_slot_refine_shift_ratio": meta.get("middle_slot_refine_shift_ratio", ""),
            "middle_slot_aspect_prior_score": meta.get("middle_slot_aspect_prior_score", ""),
            "middle_slot_stroke_density": meta.get("middle_slot_stroke_density", ""),
            "middle_slot_projection_peakiness": meta.get("middle_slot_projection_peakiness", ""),
            "middle_slot_center_balance": meta.get("middle_slot_center_balance", ""),
            "middle_slot_component_plausibility": meta.get("middle_slot_component_plausibility", ""),
            "middle_slot_side_intrusion": meta.get("middle_slot_side_intrusion", ""),
            "middle_slot_digit_intrusion_penalty": meta.get("middle_slot_digit_intrusion_penalty", ""),
            "middle_slot_right_cut_recovery_score": meta.get("middle_slot_right_cut_recovery_score", ""),
            "middle_slot_left_pad_ratio": meta.get("middle_slot_left_pad_ratio", ""),
            "middle_slot_right_pad_ratio": meta.get("middle_slot_right_pad_ratio", ""),
            "middle_slot_original_skip_reason": meta.get("middle_slot_original_skip_reason", item.skip_reason),
            "middle_slot_cropper_skip_reason": meta.get("middle_slot_cropper_skip_reason", item.skip_reason),
            "middle_slot_prefix_digits": item.prefix_digits,
            "middle_slot_suffix_digits": item.suffix_digits,
            "middle_slot_top1": item.middle_slot_top1,
            "middle_slot_top1_ko": meta.get("middle_slot_top1_ko", ""),
            "middle_slot_top1_conf": item.middle_slot_top1_conf,
            "middle_slot_top2": item.middle_slot_top2,
            "middle_slot_top2_ko": meta.get("middle_slot_top2_ko", ""),
            "middle_slot_top2_conf": getattr(item, "middle_slot_top2_conf", 0.0),
            "middle_slot_top3": meta.get("middle_slot_top3", ""),
            "middle_slot_top3_ko": meta.get("middle_slot_top3_ko", ""),
            "middle_slot_top3_conf": meta.get("middle_slot_top3_conf", ""),
            "middle_slot_margin": item.middle_slot_margin,
            "middle_slot_topk": meta.get("middle_slot_topk_json", topk_text),
            "middle_slot_topk_json": meta.get("middle_slot_topk_json", topk_text),
            "middle_slot_topk_requested": meta.get("middle_slot_topk_requested", meta.get("middle_slot_topk", getattr(self, "middle_slot_hog_lbp_topk", ""))),
            "middle_slot_topk_actual": topk_actual,
            "middle_slot_hog_lbp_min_conf": getattr(self, "middle_slot_hog_lbp_min_conf", ""),
            "middle_slot_hog_lbp_min_margin": getattr(self, "middle_slot_hog_lbp_min_margin", ""),
            "middle_slot_hog_lbp_min_crop_score": getattr(self, "middle_slot_hog_lbp_min_crop_score", ""),
            "middle_slot_generated_candidates": meta.get("middle_slot_generated_candidates", ""),
            "middle_slot_prototype_id": item.prototype_id,
            "middle_slot_update_applied": meta.get("middle_slot_update_applied", 0),
            "middle_slot_update_alpha": item.update_alpha,
            "middle_slot_digit_anchor": item.digit_anchor_text,
            "middle_slot_generated_candidate": item.generated_korean_plate_candidate,
            "middle_slot_group_key": "",
            "middle_slot_group_key_source": "",
            "middle_slot_group_key_role": "variant_local_safety_key",
            "middle_slot_fastplate_skeleton_candidates": "",
            "middle_slot_generated_digit_skeleton": "",
            "middle_slot_skeleton_match": 0,
            "middle_slot_skeleton_support": 0.0,
            "middle_slot_feedback_bucket": "",
            "middle_slot_digit_skeleton": "",
            "middle_slot_char_memory_best": "",
            "middle_slot_char_memory_second": "",
            "middle_slot_char_memory_best_count": 0,
            "middle_slot_char_memory_second_score": 0.0,
            "middle_slot_char_memory_margin": 0.0,
            "middle_slot_char_memory_support": 0.0,
            "middle_slot_char_memory_support_frames": 0,
            "middle_slot_char_memory_json": "",
            "middle_slot_source_weight": meta.get("middle_slot_source_weight", item.source_weight),
            "middle_slot_diagnostic_only": meta.get("middle_slot_diagnostic_only", int(getattr(self, "middle_slot_diagnostic_only", False))),
            "middle_slot_skip_reason": item.skip_reason,
            "middle_slot_batch_size": meta.get("middle_slot_batch_size", item.meta.get("middle_slot_batch_size", "")),
            "middle_slot_batch_delay_ms": item.delay_ms,
            "middle_slot_gpu_batch": meta.get("middle_slot_gpu_batch", 1),
            "middle_slot_encoder_mode": meta.get("middle_slot_encoder_mode", getattr(self, "middle_slot_encoder_mode", "")),
            "middle_slot_prototype_encoder_mode": meta.get("middle_slot_prototype_encoder_mode", ""),
            "middle_slot_similarity_mode": meta.get("middle_slot_similarity_mode", "prototype_matmul"),
            "middle_slot_queue_priority": meta.get("middle_slot_queue_priority", ""),
            "middle_slot_evidence_skip_reason": getattr(item, "middle_slot_evidence_skip_reason", "") or ("" if item.generated_korean_plate_candidate else (item.skip_reason or "no_generated_candidate")),
            "middle_slot_gt_anchor_confused_fair": getattr(item, "middle_slot_gt_anchor_confused_fair", 0),
            "middle_slot_gt_anchor_status": getattr(item, "middle_slot_gt_anchor_status", ""),
            "middle_slot_gt_plate": getattr(item, "middle_slot_gt_plate", ""),
            "middle_slot_gt_middle": getattr(item, "middle_slot_gt_middle", ""),
            "middle_slot_gt_anchor_boost": getattr(item, "middle_slot_gt_anchor_boost", 0.0),
            "middle_slot_confused_scores": getattr(item, "middle_slot_confused_scores", ""),
            "middle_slot_confused_candidates": getattr(item, "middle_slot_confused_candidates", ""),
            "middle_slot_confused_selected_candidate": getattr(item, "middle_slot_confused_selected_candidate", ""),
            "middle_slot_confused_support_count": getattr(item, "middle_slot_confused_support_count", 0),
            "middle_slot_confused_skip_reason": getattr(item, "middle_slot_confused_skip_reason", ""),
            "middle_slot_evidence_flag": getattr(item, "middle_slot_evidence_flag", 0),
            "middle_slot_evidence_text": getattr(item, "middle_slot_evidence_text", ""),
            "source": meta.get("middle_slot_evidence_source", item.source),
            "ocr_source": "middle_slot_hog_lbp_gpu" if diag_is_hog_lbp else meta.get("middle_slot_evidence_source", item.source),
            "final_plate_source": "middle_slot_hog_lbp_gpu_soft" if diag_is_hog_lbp else "",
            "final_plate": "",
            "final_output_plate": "",
        }
        topk_scores = [float(score) for _label, score in (item.middle_slot_topk or [])]
        zero_similarity = bool(topk_scores) and all(score <= 0.0 for score in topk_scores)
        is_easyocr_middle_slot = str(meta.get("middle_slot_backend", getattr(self, "middle_slot_backend", "upl"))) == "easyocr"
        is_hog_lbp_middle_slot = str(meta.get("middle_slot_backend", getattr(self, "middle_slot_backend", "upl"))) == "hog_lbp_gpu"
        if is_hog_lbp_middle_slot:
            active_conf_thr = float(getattr(self, "middle_slot_hog_lbp_min_conf", 0.07))
            active_margin_thr = float(getattr(self, "middle_slot_hog_lbp_min_margin", 0.005))
            active_crop_score_thr = float(getattr(self, "middle_slot_hog_lbp_min_crop_score", 0.40))
        elif is_easyocr_middle_slot:
            active_conf_thr = float(getattr(self, "middle_slot_easyocr_min_conf", 0.30))
            active_margin_thr = 0.0
            active_crop_score_thr = 0.0
        else:
            active_conf_thr = float(getattr(self, "middle_slot_conf_thr", 0.70))
            active_margin_thr = float(getattr(self, "middle_slot_margin_thr", 0.12))
            active_crop_score_thr = 0.0
        below_threshold = float(item.middle_slot_top1_conf) < active_conf_thr
        if is_hog_lbp_middle_slot:
            below_threshold = below_threshold
        elif not is_easyocr_middle_slot:
            below_threshold = below_threshold or float(item.middle_slot_margin) < active_margin_thr
        evidence_source = str(meta.get("middle_slot_evidence_source", "fastplate_digits_plus_middle_slot_upl"))
        is_gt_confused_fair = evidence_source == "fastplate_digits_plus_gt_anchor_confused_fair"
        if is_easyocr_middle_slot and not is_gt_confused_fair:
            valid_easyocr_candidate = (
                item.middle_slot_top1 in VALID_KOR
                and float(item.middle_slot_top1_conf) >= float(getattr(self, "middle_slot_easyocr_min_conf", 0.30))
                and bool(item.prefix_digits)
                and bool(item.suffix_digits)
                and bool(item.generated_korean_plate_candidate)
            )
            if not valid_easyocr_candidate:
                below_threshold = True
        middle_slot_evidence_block_reason = ""
        hog_lbp_memory_candidate = ""
        hog_lbp_dynamic_weight = 0.0
        if is_hog_lbp_middle_slot:
            crop_score_raw = meta.get("middle_slot_crop_score", "")
            try:
                crop_score_value = float(crop_score_raw)
            except (TypeError, ValueError):
                crop_score_value = 1.0
            crop_type = str(meta.get("middle_slot_crop_type", "") or "")
            prefix_digits = str(item.prefix_digits or "")
            suffix_digits = str(item.suffix_digits or "")
            generated_candidate = str(item.generated_korean_plate_candidate or "")
            generated_skeleton = _digit_skeleton_of_plate(generated_candidate)

            group_key, group_key_source = self._resolve_middle_slot_group_key(meta, int(item.track_id))
            skeleton_source_texts: list[tuple[str, str]] = []
            for key in (
                "raw_text",
                "ocr_text",
                "corrected_text",
                "grammar_best_text",
                "final_output_plate",
                "final_plate",
                "string_febam_text",
                "segment_consensus_text",
                "segment_text_medoid",
                "event_fusion_ocr_text",
                "event_fusion_text",
                "event_fusion_top_candidates",
            ):
                value = str(meta.get(key, "") or "").strip()
                if value and value != generated_candidate:
                    skeleton_source_texts.append((key, value))
            skeleton_bank = self._update_middle_slot_group_skeleton_bank(group_key, int(item.frame_idx), skeleton_source_texts)
            skeleton_rec = skeleton_bank.get(generated_skeleton, {}) if generated_skeleton else {}
            skeleton_support = float(skeleton_rec.get("support", 0.0) or 0.0)
            skeleton_support_thr = 0.75

            diag["middle_slot_group_key"] = group_key
            diag["middle_slot_group_key_source"] = group_key_source
            diag["middle_slot_group_key_role"] = "variant_local_safety_key"
            diag["middle_slot_fastplate_skeleton_candidates"] = self._serialize_middle_slot_skeleton_bank(skeleton_bank)
            diag["middle_slot_generated_digit_skeleton"] = generated_skeleton
            diag["middle_slot_digit_skeleton"] = generated_skeleton
            diag["middle_slot_skeleton_support"] = skeleton_support

            topk_rows: list[tuple[str, float]] = []
            for row in item.middle_slot_topk or []:
                try:
                    label, score = row
                except (TypeError, ValueError):
                    continue
                ko = str(label or "")[:1]
                if ko and ko in VALID_KOR:
                    topk_rows.append((ko, float(score)))
            for ko_key, conf_key in (("middle_slot_top1_ko", "middle_slot_top1_conf"), ("middle_slot_top2_ko", "middle_slot_top2_conf"), ("middle_slot_top3_ko", "middle_slot_top3_conf")):
                ko = str(meta.get(ko_key, "") or "")[:1]
                if not ko or ko not in VALID_KOR:
                    continue
                try:
                    score = float(meta.get(conf_key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                if score > 0.0 and all(existing_ko != ko for existing_ko, _score in topk_rows):
                    topk_rows.append((ko, score))
            memory_summary = self._middle_slot_char_memory_summary(None)
            if generated_skeleton and not _is_placeholder_digit_skeleton(generated_skeleton) and topk_rows:
                memory_rec = self._update_middle_slot_char_memory(
                    group_key,
                    generated_skeleton,
                    int(item.frame_idx),
                    topk_rows,
                    crop_score_value,
                    crop_type,
                    "middle_slot_hog_lbp_gpu",
                )
                memory_summary = self._middle_slot_char_memory_summary(memory_rec)
            diag["middle_slot_char_memory_best"] = memory_summary["best"]
            diag["middle_slot_char_memory_second"] = memory_summary["second"]
            diag["middle_slot_char_memory_best_count"] = memory_summary["best_count"]
            diag["middle_slot_char_memory_second_score"] = memory_summary["second_score"]
            diag["middle_slot_char_memory_margin"] = memory_summary["margin"]
            diag["middle_slot_char_memory_support"] = memory_summary["support"]
            diag["middle_slot_char_memory_support_frames"] = memory_summary["support_frames"]
            diag["middle_slot_char_memory_json"] = memory_summary["json"]

            if crop_type != "anchor_projection_refine":
                middle_slot_evidence_block_reason = "hog_lbp_non_anchor_crop_type"
            elif not generated_candidate or not self._valid_final_plate_candidate(generated_candidate):
                middle_slot_evidence_block_reason = "hog_lbp_invalid_generated_candidate"
            elif not generated_skeleton or _is_placeholder_digit_skeleton(generated_skeleton):
                middle_slot_evidence_block_reason = "hog_lbp_placeholder_anchor"
            elif prefix_digits in {"00", "000"} or suffix_digits in {"1111", "0000"} or (len(suffix_digits) == 4 and len(set(suffix_digits)) == 1):
                middle_slot_evidence_block_reason = "hog_lbp_placeholder_anchor"
            elif not skeleton_bank:
                middle_slot_evidence_block_reason = "hog_lbp_no_group_skeleton"
            elif generated_skeleton not in skeleton_bank:
                middle_slot_evidence_block_reason = "hog_lbp_skeleton_mismatch"
            elif skeleton_support < skeleton_support_thr:
                middle_slot_evidence_block_reason = "hog_lbp_low_skeleton_support"
            elif float(item.middle_slot_top1_conf) < active_conf_thr:
                middle_slot_evidence_block_reason = "hog_lbp_low_conf"
            elif crop_score_value < active_crop_score_thr:
                middle_slot_evidence_block_reason = "hog_lbp_low_crop_score"
            elif not item.evidence_accepted:
                middle_slot_evidence_block_reason = item.skip_reason or "not_evidence_accepted"
            else:
                diag["middle_slot_skeleton_match"] = 1
                hog_lbp_memory_candidate = generated_candidate
                hog_lbp_dynamic_weight = min(float(getattr(self, "middle_slot_hog_lbp_source_weight", 0.10)), 0.05)

            if middle_slot_evidence_block_reason == "hog_lbp_skeleton_mismatch":
                diag["middle_slot_feedback_bucket"] = "hard_negative_candidate"
            elif middle_slot_evidence_block_reason in {"hog_lbp_low_conf", "hog_lbp_low_crop_score", "hog_lbp_low_skeleton_support", "hog_lbp_no_group_skeleton"}:
                diag["middle_slot_feedback_bucket"] = "review_candidate"
            elif middle_slot_evidence_block_reason:
                diag["middle_slot_feedback_bucket"] = "diagnostic_only"
            else:
                diag["middle_slot_feedback_bucket"] = "accepted_soft"
        if not middle_slot_evidence_block_reason and item.generated_korean_plate_candidate and (zero_similarity or below_threshold or not item.evidence_accepted):
            middle_slot_evidence_block_reason = item.skip_reason or "zero_similarity_or_below_threshold"
            if middle_slot_evidence_block_reason == "below_evidence_gate":
                middle_slot_evidence_block_reason = "zero_similarity_or_below_threshold"
            diag["middle_slot_status"] = "skipped"
            diag["middle_slot_evidence_skip_reason"] = middle_slot_evidence_block_reason
        if is_hog_lbp_middle_slot and middle_slot_evidence_block_reason:
            diag["middle_slot_status"] = "skipped"
            diag["middle_slot_evidence_skip_reason"] = middle_slot_evidence_block_reason
            if not diag.get("middle_slot_feedback_bucket"):
                diag["middle_slot_feedback_bucket"] = "review_candidate" if middle_slot_evidence_block_reason in {"zero_similarity_or_below_threshold", "hog_lbp_low_conf", "hog_lbp_low_crop_score", "hog_lbp_low_skeleton_support", "hog_lbp_no_group_skeleton"} else "diagnostic_only"
        if getattr(self, "middle_slot_diagnostic_only", False):
            diag["middle_slot_status"] = "skipped"
            diag["middle_slot_evidence_skip_reason"] = "diagnostic_only"
        if diag_is_hog_lbp:
            skip_reason = str(diag.get("middle_slot_evidence_skip_reason", "") or "")
            if skip_reason:
                diag["middle_slot_evidence"] = ""
                diag["middle_slot_evidence_text"] = ""
                diag["middle_slot_evidence_flag"] = 0
                diag["middle_slot_source_weight"] = 0.0
                diag["source_weight"] = 0.0
                diag["final_output_plate"] = ""
            else:
                dynamic_diag_weight = hog_lbp_dynamic_weight
                diag["middle_slot_evidence"] = str(hog_lbp_memory_candidate or "")
                diag["middle_slot_evidence_text"] = str(hog_lbp_memory_candidate or "")
                diag["middle_slot_evidence_flag"] = 1
                diag["middle_slot_source_weight"] = dynamic_diag_weight
                diag["source_weight"] = dynamic_diag_weight
                diag["final_output_plate"] = ""
                diag["final_plate_source"] = "middle_slot_hog_lbp_gpu_soft"
        self.middle_slot_csv_diagnostics.append(diag)
        diagnostic_row = {
            "track_id": int(item.track_id),
            "frame_idx": int(item.frame_idx),
            "candidate_idx": int(item.candidate_idx),
            "ocr_text": str(meta.get("ocr_text", "")),
            "ocr_conf": meta.get("ocr_conf", ""),
            "raw_text": str(meta.get("ocr_text", "")),
            "raw_conf": meta.get("ocr_conf", ""),
            "source": item.source,
            **diag,
        }
        self.ocr_csv_rows.append(diagnostic_row)
        diagnostic_debug_msg = (
            f"[MIDDLE_SLOT_CSV_APPEND_DEBUG] len={len(self.ocr_csv_rows)} "
            f"row_type={diagnostic_row.get('row_type', '')} "
            f"track_id={diagnostic_row.get('track_id', '')} "
            f"candidate={diagnostic_row.get('middle_slot_generated_candidate', '')}"
        )
        print(diagnostic_debug_msg, flush=True)
        self._log(diagnostic_debug_msg)
        if getattr(self, "middle_slot_diagnostic_only", False):
            self.middle_slot_evidence_diagnostic_only_skipped += 1
            return
        if middle_slot_evidence_block_reason:
            self.middle_slot_evidence_skipped += 1
            if is_gt_confused_fair:
                self.middle_slot_gt_confused_evidence_skipped += 1
            return
        if not item.generated_korean_plate_candidate:
            self.middle_slot_evidence_skipped += 1
            if is_gt_confused_fair:
                self.middle_slot_gt_confused_evidence_skipped += 1
            return
        track_id = int(item.track_id)
        frame_idx = int(item.frame_idx)
        bbox = meta.get("bbox_roi", meta.get("bbox", (0, 0, 0, 0)))
        mrs = float(meta.get("mrs", meta.get("febam_score", 0.5)) or 0.5)
        digit_quality = float(meta.get("digit_anchor_quality", 1.0) or 1.0)
        if is_hog_lbp_middle_slot:
            dynamic_weight = hog_lbp_dynamic_weight
        else:
            dynamic_weight = max(0.35, min(0.85, self.middle_slot_source_weight * float(item.middle_slot_top1_conf) * max(0.35, digit_quality)))
        self._append_ocr_history(
            track_id,
            hog_lbp_memory_candidate if is_hog_lbp_middle_slot else item.generated_korean_plate_candidate,
            float(item.middle_slot_top1_conf),
            frame_idx,
            bbox,
            mrs,
            raw_text=str(meta.get("ocr_text", "")),
            corrected_text=hog_lbp_memory_candidate if is_hog_lbp_middle_slot else item.generated_korean_plate_candidate,
            source=("middle_slot_hog_lbp_gpu" if is_hog_lbp_middle_slot else str(meta.get("middle_slot_evidence_source", "fastplate_digits_plus_middle_slot_upl"))),
            source_weight=dynamic_weight,
        )
        final_plate_source = "middle_slot_gt_anchor_confused_fair" if is_gt_confused_fair else ("middle_slot_hog_lbp_gpu_soft" if is_hog_lbp_middle_slot else ("middle_slot_easyocr" if evidence_source == "fastplate_digits_plus_easyocr_korean" else "middle_slot_upl"))
        evidence_row_extra = {
            **diag,
            "row_type": "middle_slot_evidence",
            "middle_slot_evidence": hog_lbp_memory_candidate if is_hog_lbp_middle_slot else 1,
            "middle_slot_evidence_text": hog_lbp_memory_candidate if is_hog_lbp_middle_slot else getattr(item, "middle_slot_evidence_text", ""),
            "middle_slot_evidence_flag": 1,
            "middle_slot_status": "accepted",
            "middle_slot_skip_reason": "",
            "middle_slot_evidence_skip_reason": "",
            "ocr_source": "middle_slot_hog_lbp_gpu" if is_hog_lbp_middle_slot else evidence_source,
            "source": "middle_slot_hog_lbp_gpu" if is_hog_lbp_middle_slot else evidence_source,
            "source_weight": dynamic_weight,
            "middle_slot_source_weight": dynamic_weight,
            "final_candidate_source": evidence_source,
            "final_plate_source": final_plate_source,
            "corrected_text": hog_lbp_memory_candidate if is_hog_lbp_middle_slot else item.generated_korean_plate_candidate,
            "final_plate": "" if is_hog_lbp_middle_slot else item.generated_korean_plate_candidate,
            "final_output_plate": "" if is_hog_lbp_middle_slot else item.generated_korean_plate_candidate,
            "normalized_plate_candidate": hog_lbp_memory_candidate if is_hog_lbp_middle_slot else item.generated_korean_plate_candidate,
            "raw_text": str(meta.get("ocr_text", "")),
            "raw_conf": meta.get("ocr_conf", ""),
            "corrected_conf": float(item.middle_slot_top1_conf),
        }
        evidence_row = {
            "track_id": track_id,
            "frame_idx": frame_idx,
            "candidate_idx": int(item.candidate_idx),
            "ocr_text": hog_lbp_memory_candidate if is_hog_lbp_middle_slot else item.generated_korean_plate_candidate,
            "ocr_conf": float(item.middle_slot_top1_conf),
            **evidence_row_extra,
        }
        self.ocr_csv_rows.append(evidence_row)
        evidence_debug_msg = (
            f"[MIDDLE_SLOT_CSV_EVIDENCE_DEBUG] len={len(self.ocr_csv_rows)} "
            f"row_type={evidence_row.get('row_type', '')} "
            f"ocr_source={evidence_row.get('ocr_source', '')} "
            f"candidate={evidence_row.get('middle_slot_generated_candidate', '')}"
        )
        print(evidence_debug_msg, flush=True)
        self._log(evidence_debug_msg)
        self.middle_slot_evidence_added += 1
        if is_gt_confused_fair:
            self.middle_slot_gt_confused_evidence_added += 1

    def configure_middle_slot_upl_from_args(self, args) -> None:
        self.middle_slot_upl = bool(getattr(args, "middle_slot_upl", getattr(self, "middle_slot_upl", False)))
        self.middle_slot_device = str(getattr(args, "middle_slot_device", getattr(self, "middle_slot_device", "cuda")))
        self.middle_slot_batch_size = int(max(1, getattr(args, "middle_slot_batch_size", getattr(self, "middle_slot_batch_size", 64))))
        self.middle_slot_min_batch_size = int(max(1, getattr(args, "middle_slot_min_batch_size", getattr(self, "middle_slot_min_batch_size", 4))))
        self.middle_slot_flush_timeout_ms = float(max(1.0, getattr(args, "middle_slot_flush_timeout_ms", getattr(self, "middle_slot_flush_timeout_ms", 100.0))))
        self.middle_slot_queue_max = int(max(1, getattr(args, "middle_slot_queue_max", getattr(self, "middle_slot_queue_max", 512))))
        self.middle_slot_input_size = int(max(16, getattr(args, "middle_slot_input_size", getattr(self, "middle_slot_input_size", 48))))
        self.middle_slot_max_crops_per_group = int(max(1, getattr(args, "middle_slot_max_crops_per_group", getattr(self, "middle_slot_max_crops_per_group", 3))))
        self.middle_slot_preset = str(getattr(args, "middle_slot_preset", getattr(self, "middle_slot_preset", "none")) or "none")
        self.middle_slot_backend = str(getattr(args, "middle_slot_backend", getattr(self, "middle_slot_backend", "upl"))).lower().strip()
        if self.middle_slot_backend not in {"upl", "easyocr", "hog_lbp_gpu", "none"}:
            self.middle_slot_backend = "upl"
        if self.middle_slot_backend == "hog_lbp_gpu":
            self.middle_slot_upl = True
        self.middle_slot_easyocr_single_crop = bool(getattr(args, "middle_slot_easyocr_single_crop", getattr(self, "middle_slot_easyocr_single_crop", True)))
        self.middle_slot_easyocr_gpu = bool(getattr(args, "middle_slot_easyocr_gpu", getattr(self, "middle_slot_easyocr_gpu", True)))
        self.middle_slot_easyocr_batch_size = int(max(1, getattr(args, "middle_slot_easyocr_batch_size", getattr(self, "middle_slot_easyocr_batch_size", 16))))
        self.middle_slot_easyocr_min_conf = float(max(0.0, getattr(args, "middle_slot_easyocr_min_conf", getattr(self, "middle_slot_easyocr_min_conf", 0.30))))
        self.middle_slot_easyocr_allowlist = str(getattr(args, "middle_slot_easyocr_allowlist", getattr(self, "middle_slot_easyocr_allowlist", VALID_KOR)) or VALID_KOR)
        self.middle_slot_easyocr_resize_scale = int(max(1, getattr(args, "middle_slot_easyocr_resize_scale", getattr(self, "middle_slot_easyocr_resize_scale", 6))))
        self.middle_slot_easyocr_border = int(max(0, getattr(args, "middle_slot_easyocr_border", getattr(self, "middle_slot_easyocr_border", 50))))
        self.middle_slot_wide_core_x_pad_ratio = float(max(0.0, getattr(args, "middle_slot_wide_core_x_pad_ratio", getattr(self, "middle_slot_wide_core_x_pad_ratio", 0.16))))
        self.middle_slot_wide_core_y_pad_ratio = float(max(0.0, getattr(args, "middle_slot_wide_core_y_pad_ratio", getattr(self, "middle_slot_wide_core_y_pad_ratio", 0.15))))
        self.middle_slot_wide_core_min_width_ratio = float(max(0.01, getattr(args, "middle_slot_wide_core_min_width_ratio", getattr(self, "middle_slot_wide_core_min_width_ratio", 0.14))))
        self.middle_slot_wide_core_max_width_ratio = float(max(self.middle_slot_wide_core_min_width_ratio, getattr(args, "middle_slot_wide_core_max_width_ratio", getattr(self, "middle_slot_wide_core_max_width_ratio", 0.34))))
        self.middle_slot_cropper_mode = str(getattr(args, "middle_slot_cropper_mode", getattr(self, "middle_slot_cropper_mode", "gpu_aspect_prior")))
        if self.middle_slot_backend == "easyocr" and self.middle_slot_easyocr_single_crop and self.middle_slot_cropper_mode == "gpu_aspect_prior":
            self.middle_slot_cropper_mode = "gpu_aspect_prior_wide_core"
        self.middle_slot_encoder_mode = str(getattr(args, "middle_slot_encoder_mode", getattr(self, "middle_slot_encoder_mode", "hog_torch")))
        self.middle_slot_model_path = getattr(args, "middle_slot_model_path", getattr(self, "middle_slot_model_path", None))
        self.middle_slot_prototype_path = getattr(args, "middle_slot_prototype_path", getattr(self, "middle_slot_prototype_path", None))
        self.middle_slot_source_weight = float(max(0.0, min(1.0, getattr(args, "middle_slot_source_weight", getattr(self, "middle_slot_source_weight", 0.70)))))
        self.middle_slot_hog_lbp_model = getattr(args, "middle_slot_hog_lbp_model", getattr(self, "middle_slot_hog_lbp_model", None))
        self.middle_slot_hog_lbp_topk = int(max(1, getattr(args, "middle_slot_hog_lbp_topk", getattr(self, "middle_slot_hog_lbp_topk", 3))))
        self.middle_slot_hog_lbp_source_weight = float(max(0.0, min(1.0, getattr(args, "middle_slot_hog_lbp_source_weight", getattr(self, "middle_slot_hog_lbp_source_weight", 0.10)))))
        self.middle_slot_hog_lbp_min_conf = float(max(0.0, getattr(args, "middle_slot_hog_lbp_min_conf", getattr(self, "middle_slot_hog_lbp_min_conf", 0.07))))
        self.middle_slot_hog_lbp_min_margin = float(max(0.0, getattr(args, "middle_slot_hog_lbp_min_margin", getattr(self, "middle_slot_hog_lbp_min_margin", 0.005))))
        self.middle_slot_hog_lbp_min_crop_score = float(max(0.0, getattr(args, "middle_slot_hog_lbp_min_crop_score", getattr(self, "middle_slot_hog_lbp_min_crop_score", 0.40))))
        self.middle_slot_conf_thr = float(getattr(args, "middle_slot_conf_thr", getattr(self, "middle_slot_conf_thr", 0.70)))
        self.middle_slot_margin_thr = float(getattr(args, "middle_slot_margin_thr", getattr(self, "middle_slot_margin_thr", 0.12)))
        self.middle_slot_sim_center = float(getattr(args, "middle_slot_sim_center", getattr(self, "middle_slot_sim_center", 0.65)))
        self.middle_slot_sim_k = float(getattr(args, "middle_slot_sim_k", getattr(self, "middle_slot_sim_k", 12.0)))
        self.middle_slot_alpha_min = float(getattr(args, "middle_slot_alpha_min", getattr(self, "middle_slot_alpha_min", 0.85)))
        self.middle_slot_alpha_max = float(getattr(args, "middle_slot_alpha_max", getattr(self, "middle_slot_alpha_max", 0.995)))
        self.middle_slot_debug_dir = str(getattr(args, "middle_slot_debug_dir", getattr(self, "middle_slot_debug_dir", "../data/processed/debug/middle_slot")))
        self.middle_slot_diagnostic_only = bool(getattr(args, "middle_slot_diagnostic_only", getattr(self, "middle_slot_diagnostic_only", False)))
        self.middle_slot_save_debug_crops = bool(getattr(args, "middle_slot_save_debug_crops", getattr(self, "middle_slot_save_debug_crops", False)) or self.middle_slot_diagnostic_only)
        self.middle_slot_disable_prototype_update = bool(getattr(args, "middle_slot_disable_prototype_update", getattr(self, "middle_slot_disable_prototype_update", False)))
        self.middle_slot_gt_anchor_confused_fair = bool(getattr(args, "middle_slot_gt_anchor_confused_fair", getattr(self, "middle_slot_gt_anchor_confused_fair", True)))
        self.middle_slot_gt_plates_path = getattr(args, "middle_slot_gt_plates_path", getattr(self, "middle_slot_gt_plates_path", None))
        self.middle_slot_gt_anchor_boost = float(getattr(args, "middle_slot_gt_anchor_boost", getattr(self, "middle_slot_gt_anchor_boost", 0.40)))
        self.middle_slot_gt_confused_min_score = float(getattr(args, "middle_slot_gt_confused_min_score", getattr(self, "middle_slot_gt_confused_min_score", 0.05)))
        self.middle_slot_gt_confused_min_support = int(max(1, getattr(args, "middle_slot_gt_confused_min_support", getattr(self, "middle_slot_gt_confused_min_support", 1))))
        middle_slot_v32 = bool(getattr(args, "middle_slot_v32", False))
        middle_slot_v32_model = str(getattr(args, "middle_slot_v32_model", "") or "")
        middle_slot_v32_device = str(getattr(args, "middle_slot_v32_device", "cuda") or "cuda")
        middle_slot_v32_batch_size = int(getattr(args, "middle_slot_v32_batch_size", 512) or 512)
        middle_slot_v32_min_batch_size = int(getattr(args, "middle_slot_v32_min_batch_size", 8) or 8)
        middle_slot_v32_flush_every_frames = int(getattr(args, "middle_slot_v32_flush_every_frames", 5) or 5)
        middle_slot_v32_topk = int(getattr(args, "middle_slot_v32_topk", 3) or 3)
        middle_slot_v32_source_weight = float(getattr(args, "middle_slot_v32_source_weight", 0.04) or 0.04)
        middle_slot_v32_min_conf = float(getattr(args, "middle_slot_v32_min_conf", 0.20) or 0.20)
        middle_slot_v32_min_margin = float(getattr(args, "middle_slot_v32_min_margin", 0.03) or 0.03)
        middle_slot_v32_max_crops_per_event = int(getattr(args, "middle_slot_v32_max_crops_per_event", 8) or 8)
        middle_slot_v32_evidence_mode = str(getattr(args, "middle_slot_v32_evidence_mode", "topk_soft") or "topk_soft")
        middle_slot_v32_profile = bool(getattr(args, "middle_slot_v32_profile", False))
        if middle_slot_v32 and not middle_slot_v32_model:
            middle_slot_v32_model = str(getattr(self, "middle_slot_v32_model", "") or r".\runs\middle_slot\gabor_scatter_lite_structured_v32_runtime_fuzzyres_cache_v5.pt")
        self.middle_slot_v32_enabled = middle_slot_v32
        self.middle_slot_v32_processed = 0
        self.middle_slot_v32_batch_call_count = 0
        self.middle_slot_v32_total_infer_ms = 0.0
        self.middle_slot_v32_pending_final_flush_count = 0
        self.middle_slot_v32_added_evidence = 0
        self.middle_slot_v32_skipped_low_conf = 0
        self.middle_slot_v32_skipped_low_margin = 0
        self.middle_slot_v32_skipped_other = 0
        self.middle_slot_v32_skipped_no_event_id = 0
        self.middle_slot_v32_skipped_no_track_id = 0
        self.middle_slot_v32_skipped_no_digit_skeleton = 0
        self.middle_slot_v32_skipped_invalid_digit_skeleton = 0
        self.middle_slot_v32_skipped_max_crops_per_event = 0
        self.middle_slot_v32_skipped_empty_crop = 0
        self.middle_slot_v32_skipped_duplicate = 0
        self.middle_slot_v32_skipped_pending_not_flushed = 0
        self.middle_slot_v32_skipped_model_error = 0
        self.middle_slot_v32_skipped_before_infer_unknown = 0
        self.middle_slot_v32_debug_rows = []
        self.middle_slot_v32_pending = []
        self.middle_slot_v32_pending_keys = set()
        self.middle_slot_v32_crops_by_event = {}
        self._middle_slot_v32_profile_logged = False
        self.middle_slot_v32_model_path = middle_slot_v32_model
        self.middle_slot_v32_model = middle_slot_v32_model or str(getattr(self, "middle_slot_v32_model", r".\runs\middle_slot\gabor_scatter_lite_structured_v32_runtime_fuzzyres_cache_v5.pt"))
        self.middle_slot_v32_device = middle_slot_v32_device
        self.middle_slot_v32_batch_size = max(1, middle_slot_v32_batch_size)
        self.middle_slot_v32_min_batch_size = max(1, middle_slot_v32_min_batch_size)
        self.middle_slot_v32_flush_every_frames = max(1, middle_slot_v32_flush_every_frames)
        self.middle_slot_v32_topk = max(1, min(3, middle_slot_v32_topk))
        self.middle_slot_v32_source_weight = middle_slot_v32_source_weight
        self.middle_slot_v32_min_conf = middle_slot_v32_min_conf
        self.middle_slot_v32_min_margin = middle_slot_v32_min_margin
        self.middle_slot_v32_max_crops_per_event = max(1, middle_slot_v32_max_crops_per_event)
        self.middle_slot_v32_evidence_mode = middle_slot_v32_evidence_mode
        self.middle_slot_v32_profile = middle_slot_v32_profile
        self.middle_slot_v32_runtime: MiddleSlotV32Runtime | None = None
        self.middle_slot_gt_items = []
        if self.middle_slot_gt_anchor_confused_fair and self.middle_slot_gt_plates_path:
            from ocr.middle_slot_gt_anchor_confused_fair import load_gt_plates

            self.middle_slot_gt_items = load_gt_plates(self.middle_slot_gt_plates_path)

    def _fastplate_async_stats(self) -> dict[str, float]:
        worker_stats = self.fastplate_async_worker.stats() if self.fastplate_async_worker is not None else {}
        processed = float(worker_stats.get("fastplate_async_processed", self.fastplate_async_processed))
        delay_avg = self.fastplate_result_delay_total_frames / float(self.fastplate_result_delay_count) if self.fastplate_result_delay_count else 0.0
        stats = {
            "ocr_session_create_count": float(worker_stats.get("ocr_session_create_count", 0.0)),
            "ocr_worker_create_count": float(worker_stats.get("ocr_worker_create_count", 0.0)),
            "ocr_shared_queue_enabled": 1.0,
            "fusion_ocr_shared_worker_enabled": 1.0 if self.event_roi_fusion_ocr and self.event_roi_fusion_ocr_worker_mode == "shared_async" else 0.0,
            "fusion_ocr_separate_session_create_count": 1.0 if self.event_fusion_fastplate_recognizer is not None else 0.0,
            "fusion_ocr_enqueued": float(self.fusion_ocr_enqueued),
            "fusion_ocr_processed": float(self.fusion_ocr_processed),
            "fusion_ocr_dropped": float(self.fusion_ocr_dropped),
            "fusion_ocr_duplicate_blocked": float(self.fusion_ocr_duplicate_blocked),
            "fusion_ocr_commit_blocked": float(self.fusion_ocr_commit_blocked),
            "fusion_ocr_result_applied": float(self.fusion_ocr_result_applied),
            "fusion_ocr_result_stale": float(self.fusion_ocr_result_stale),
            "fusion_ocr_csv_row_written": float(self.fusion_ocr_csv_row_written),
            "fusion_ocr_avg_queue_delay_ms": float(self.fusion_ocr_queue_delay_total_ms) / max(1.0, float(self.fusion_ocr_processed)),
            "fusion_ocr_avg_infer_ms": float(self.fusion_ocr_infer_total_ms) / max(1.0, float(self.fusion_ocr_processed)),
            "fusion_ocr_max_per_event_seen": float(self.fusion_ocr_max_per_event_seen),
            "fusion_ocr_cpu_input_count": float(self.fusion_ocr_cpu_input_count),
            "final_vehicle_sparse_route_count": float(self.final_vehicle_sparse_route_count),
            "final_vehicle_posterior_route_count": float(self.final_vehicle_posterior_route_count),
            "final_vehicle_identity_hold_count": float(self.final_vehicle_identity_hold_count),
            "final_vehicle_last_n_valid": float(self.final_vehicle_last_route.get("n_valid", 0) or 0),
            "final_vehicle_last_posterior_active": 1.0 if self.final_vehicle_last_route.get("posterior_active", False) else 0.0,
            "probability_clip_count": float(self.probability_clip_count),
            "invalid_probability_count": float(self.invalid_probability_count),
            "invalid_probability_field_examples": json.dumps(self.invalid_probability_field_examples, ensure_ascii=False),
            "fastplate_async_enqueued": float(worker_stats.get("fastplate_async_enqueued", self.fastplate_async_enqueued)),
            "fastplate_async_processed": processed,
            "fastplate_batch_call_count": float(worker_stats.get("fastplate_batch_call_count", 0.0)),
            "fastplate_avg_batch_size": float(worker_stats.get("fastplate_avg_batch_size", 0.0)),
            "fastplate_batch_size": float(worker_stats.get("fastplate_batch_size", self.fastplate_batch_size)),
            "fastplate_min_batch_size": float(worker_stats.get("fastplate_min_batch_size", self.fastplate_min_batch_size)),
            "fastplate_large_batch_mode": float(worker_stats.get("fastplate_large_batch_mode", 1.0 if getattr(self, "fastplate_large_batch_mode", False) else 0.0)),
            "fastplate_target_batch_size": float(worker_stats.get("fastplate_target_batch_size", getattr(self, "fastplate_target_batch_size", 64))),
            "fastplate_flush_timeout_ms": float(worker_stats.get("fastplate_flush_timeout_ms", self.fastplate_flush_timeout_ms)),
            "fastplate_max_flush_timeout_ms": float(worker_stats.get("fastplate_max_flush_timeout_ms", getattr(self, "fastplate_max_flush_timeout_ms", 1500.0))),
            "fastplate_batch_efficiency_ratio": float(worker_stats.get("fastplate_batch_efficiency_ratio", 0.0)),
            "fastplate_total_ocr_ms": float(worker_stats.get("fastplate_total_ocr_ms", 0.0)),
            "fastplate_avg_ocr_ms_per_crop": float(worker_stats.get("fastplate_avg_ocr_ms_per_crop", 0.0)),
            "fastplate_queue_dropped": float(worker_stats.get("fastplate_queue_dropped", self.fastplate_queue_dropped)),
            "fastplate_queue_drop_weak": float(worker_stats.get("fastplate_queue_drop_weak", 0.0)),
            "fastplate_queue_drop_duplicate": float(worker_stats.get("fastplate_queue_drop_duplicate", 0.0)),
            "fastplate_queue_drop_rate_limited": float(worker_stats.get("fastplate_queue_drop_rate_limited", 0.0)),
            "fastplate_queue_drop_backpressure": float(worker_stats.get("fastplate_queue_drop_backpressure", 0.0)),
            "fastplate_queue_max_seen": float(worker_stats.get("fastplate_queue_max_seen", 0.0)),
            "fastplate_queue_depth": float(worker_stats.get("fastplate_queue_depth", 0.0)),
            "fastplate_oldest_age_ms": float(worker_stats.get("fastplate_oldest_age_ms", 0.0)),
            "fastplate_result_delay_avg_frames": float(delay_avg),
            "fastplate_result_delay_avg_ms": float(getattr(self, "fastplate_result_delay_total_ms", 0.0)) / float(self.fastplate_result_delay_count) if self.fastplate_result_delay_count else 0.0,
        }
        for key in (
            "fastplate_direct_ort_enabled", "fastplate_direct_ort_iobinding",
            "fastplate_direct_ort_compare_wrapper", "fastplate_custom_onnx_enabled",
            "fastplate_gpu_crop_batch_enabled", "fastplate_tensor_runner_enabled", "fastplate_tensor_runner_mode_id",
            "fastplate_tensor_runner_active_used", "fastplate_tensor_runner_shadow_count", "fastplate_gpu_crop_count",
            "fastplate_cpu_crop_count", "fastplate_gpu_crop_ms", "fastplate_cpu_crop_ms", "fastplate_crop_cpu_copy_count",
            "fastplate_crop_gpu_tensor_count", "fastplate_tensor_batch_call_count", "fastplate_tensor_avg_batch_size",
            "fastplate_tensor_total_ms", "fastplate_tensor_avg_ms_per_crop", "fastplate_tensor_preprocess_ms",
            "fastplate_tensor_infer_ms", "fastplate_tensor_decode_ms", "fastplate_tensor_iobinding_used",
            "fastplate_tensor_cpu_fallback_count", "fastplate_tensor_error_count", "fastplate_parity_sample_count",
            "fastplate_parity_success_count", "fastplate_parity_error_count", "fastplate_parity_skipped_count",
            "fastplate_parity_text_match_count", "fastplate_parity_text_match_rate", "fastplate_parity_normalized_match_rate",
            "fastplate_parity_korean_slot_match_rate", "fastplate_parity_conf_abs_diff_mean", "fastplate_parity_effective_match_rate",
            "fastplate_tensor_error_sample_batch_len", "fastplate_tensor_prepare_valid_count",
            "fastplate_tensor_prepare_failed_count", "fastplate_tensor_preprocess_resize_count",
            "fastplate_tensor_direct_backend_available", "fastplate_tensor_gpu_only_claim_valid",
            "fastplate_tensor_active_requested", "fastplate_tensor_active_success_count",
            "fastplate_tensor_active_blocked_count", "fastplate_tensor_active_fallback_used",
            "fastplate_custom_cuda_used", "fastplate_custom_alphabet_len",
            "fastplate_custom_batch_call_count", "fastplate_custom_item_count",
            "fastplate_custom_avg_batch_size", "fastplate_custom_total_ms",
            "fastplate_custom_avg_ms_per_crop", "fastplate_custom_preprocess_ms",
            "fastplate_custom_infer_ms", "fastplate_custom_decode_ms",
            "fastplate_custom_error_count",
            "dual_branch_enabled", "dual_branch_session_count",
            "dual_branch_global_infer_ms", "dual_branch_korean_infer_ms",
            "dual_branch_iobinding_used", "dual_branch_h7_selected",
            "dual_branch_h8_selected", "dual_branch_cpu_fallback_count",
            "dual_branch_alignment_rejected", "dual_branch_low_digit_mass_rejected",
            "dual_branch_forced_digit_count", "dual_branch_avg_digit_mass",
            "dual_branch_avg_digit_margin", "dual_branch_raw_enqueued",
            "dual_branch_gray_enqueued", "dual_branch_raw_processed",
            "dual_branch_gray_processed", "dual_branch_global_raw_primary_count",
            "dual_branch_global_gray_primary_count",
            "dual_branch_warm_item_count", "dual_branch_warm_avg_ms_per_plate",
            "dual_branch_warm_call_p95_ms",
            "dual_branch_warm_p95_ms_per_plate",
            "dual_branch_length_margin_thr_configured",
            "dual_branch_length_margin_thr_effective",
            "dual_branch_length_hold_enabled",
        ):
            stats[key] = float(worker_stats.get(key, 0.0))
        stats["fastplate_tensor_runner_mode"] = str(worker_stats.get("fastplate_tensor_runner_mode", getattr(self, "fastplate_tensor_runner_mode", "disabled")) or "disabled")
        for key in (
            "fastplate_tensor_last_error_type", "fastplate_tensor_last_error_message", "fastplate_tensor_last_error_stage",
            "fastplate_tensor_error_sample_input_shape", "fastplate_tensor_error_sample_input_dtype",
            "fastplate_tensor_error_sample_output_shapes", "fastplate_tensor_provider", "fastplate_tensor_input_name",
            "fastplate_tensor_output_names", "fastplate_tensor_preprocess_mode", "fastplate_tensor_decoder_mode",
            "fastplate_tensor_last_input_shape", "fastplate_tensor_last_input_dtype",
            "fastplate_tensor_prepare_failed_examples", "fastplate_tensor_backend_kind",
            "fastplate_tensor_active_fallback_reason", "fastplate_custom_onnx_path",
            "fastplate_custom_provider", "fastplate_custom_input_name",
            "fastplate_custom_output_names", "fastplate_custom_input_shape",
            "fastplate_custom_output_shapes", "fastplate_custom_input_layout",
            "fastplate_batch_size_hist", "fastplate_tensor_batch_size_hist",
        ):
            stats[key] = str(worker_stats.get(key, "") or "")
        return stats

    def _drain_fastplate_results(self, max_items: int | None = None) -> int:
        worker = self.fastplate_async_worker
        if worker is None:
            return 0
        drained = 0
        while max_items is None or drained < int(max_items):
            try:
                item = worker.output_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_fastplate_async_result(item)
            drained += 1
        self._drain_middle_slot_results(max_items=max_items)
        return drained

    def _handle_fastplate_async_result(self, item: OCRResultItem | dict[str, object]) -> None:
        if isinstance(item, OCRResultItem):
            meta = dict(item.meta or {})
            result = {
                "text": item.text,
                "conf": item.conf,
                "source": item.source,
                "variant": item.variant,
                "raw_result": meta.get("raw_result", ""),
                "error": meta.get("fastplate_error", ""),
                "error_message": meta.get("fastplate_error_message", ""),
            }
            frame_idx = int(item.frame_idx)
            async_delay_ms = float(item.delay_ms)
        else:
            result = dict(item.get("result", {}) or {})
            meta = dict(item.get("meta", {}) or {})
            frame_idx = int(meta.get("frame_idx", -1))
            async_delay_ms = (time.perf_counter() - float(item.get("enqueue_time", time.perf_counter()))) * 1000.0
        current_frame_idx = getattr(self, "_current_frame_idx", None)
        async_delay_frames = ""
        if current_frame_idx is not None and frame_idx >= 0:
            async_delay_frames = int(current_frame_idx) - int(frame_idx)
            self.fastplate_result_delay_total_frames += float(async_delay_frames)
            self.fastplate_result_delay_total_ms += float(async_delay_ms)
            self.fastplate_result_delay_count += 1
        self.fastplate_async_processed += 1
        ocr_batch_size = int(meta.get("ocr_batch_size", self.fastplate_batch_size) or self.fastplate_batch_size)
        raw_result_short = str(result.get("raw_result", meta.get("raw_result", "")) or "")[:500]
        fastplate_error = str(result.get("error", meta.get("fastplate_error", meta.get("worker_error", ""))) or "")
        fastplate_result_source = str(result.get("source", "") or "")
        fastplate_exception = bool(fastplate_error) or fastplate_result_source == "fast_plate_ocr_exception"

        track_id = int(meta.get("track_id", -1))
        candidate_idx = int(meta.get("candidate_idx", -1))
        bbox_roi = meta.get("bbox_roi")
        roi_w = meta.get("roi_w", "")
        roi_h = meta.get("roi_h", "")
        candidate_count = int(meta.get("candidate_count", 0) or 0)
        track_id_count = int(meta.get("track_id_count", 0) or 0)
        febam_confirmed_count = int(meta.get("febam_confirmed_count", 0) or 0)
        attempted_count = int(meta.get("attempted_count", 0) or 0)
        gray_stretched_ocr_used = bool(meta.get("gray_stretched_ocr_used", False))
        variant_name = str(meta.get("variant_name", result.get("variant", "unknown")) or "unknown")
        plate_layout = str(meta.get("plate_layout", "unknown") or "unknown")
        mrs_value = float(meta.get("mrs", meta.get("febam_score", 0.0)) or 0.0)
        source_weight = float(meta.get("source_weight", 1.0) or 1.0)

        raw_text = str(result.get("text", "") or "")
        conf = float(result.get("conf", 0.0) or 0.0)
        restoration_shadow = getattr(
            self, "non_generative_restoration_shadow", None
        )
        if bool(meta.get("restoration_shadow_only", False)):
            if restoration_shadow is not None:
                restoration_shadow.record_result(
                    meta,
                    {
                        **result,
                        **meta,
                        "text": raw_text,
                        "conf": conf,
                    },
                    gpu_ms=float(meta.get("ocr_batch_ms", 0.0) or 0.0)
                    / max(1, int(meta.get("ocr_batch_size", 1) or 1)),
                )
            # Shadow results are terminal diagnostics. They must never enter
            # event-fusion evidence, legacy voting, String-FEBAM, or final
            # output registration.
            return
        if restoration_shadow is not None and isinstance(item, OCRResultItem):
            restoration_shadow.observe(
                item.crop_bgr,
                meta,
                {
                    **result,
                    **meta,
                    "text": raw_text,
                    "conf": conf,
                },
            )
        if str(meta.get("source_type", "") or "") == "event_fused_group_key":
            self.fusion_ocr_processed += 1
            self.fusion_ocr_queue_delay_total_ms += float(async_delay_ms)
            self.fusion_ocr_infer_total_ms += float(meta.get("ocr_batch_ms", 0.0) or 0.0) / max(
                1, int(meta.get("ocr_batch_size", 1) or 1)
            )
            already_committed = bool(track_id >= 0 and self._committed_plate_for_track(track_id))
            if already_committed:
                self.fusion_ocr_commit_blocked += 1
            text = self._postprocess_event_fusion_ocr_text(raw_text, conf=conf, source="fast_plate_ocr")
            if not text:
                self.fusion_ocr_dropped += 1
                return
            raw_result_text = str(meta.get("raw_result", "") or "")[:500]
            raw_plate = self._extract_event_fusion_raw_plate(raw_result_text, fallback_text=raw_text or text)
            obs = dict(meta)
            obs.update({
                "text": text,
                "raw_text": raw_text or text,
                "corrected_text": text,
                "conf": conf,
                "source": "event_fusion_trial020_ocr",
                "ocr_source": "event_fusion_trial020_ocr",
                "ocr_engine": "event_fusion_shared_final_wise",
                "source_weight": getattr(self, "event_roi_fusion_ocr_source_weight", 0.65),
                "event_fusion_ocr_backend": "fastplate",
                "event_fusion_ocr_flush_reason": "shared_async_result",
                "event_fusion_ocr_raw_result": raw_result_text,
                "event_fusion_ocr_raw_plate": raw_plate,
                "event_fusion_ocr_raw_matches_text": 1 if raw_plate and raw_plate == text else 0,
                "row_type": "fusion_ocr_verification" if already_committed else "fusion_ocr_observation",
                "fusion_ocr_applied": 0 if already_committed else 1,
                "fusion_ocr_block_reason": "already_committed" if already_committed else "",
            })
            self._submit_trial020_fused_observation(
                obs,
                bbox=None,
                candidate_idx=candidate_idx,
                roi_w="",
                roi_h="",
                candidate_count=0,
                track_id_count=0,
                febam_confirmed_count=0,
                febam_score=conf,
                febam_energy="",
                febam_memory="",
            )
            if not already_committed:
                self.fusion_ocr_result_applied += 1
            return
        normalized_text = self._normalize_ocr_text(raw_text)
        row_source = fastplate_result_source or "fast_plate_ocr"
        is_direct_dual_evidence = row_source.startswith("dual_")
        dual_decision_state = str(
            meta.get("dual_branch_decision_state", "") or ""
        )
        recovery_candidate = self._normalize_ocr_text(
            str(meta.get("recovery_candidate_text", "") or "")
        )
        if (
            is_direct_dual_evidence
            and dual_decision_state == "FEBAM_REQUIRED"
            and bool(meta.get("recovery_requires_string_febam", False))
            and self._valid_final_plate_candidate(recovery_candidate)
        ):
            recovery_group = self._vehicle_febam_string_group_id(
                meta,
                frame_idx=frame_idx,
                bbox=bbox_roi,
                candidate=recovery_candidate,
                track_id=track_id,
            )
            if recovery_group is not None:
                self._append_string_observation(
                    recovery_group,
                    frame_idx=frame_idx,
                    raw_text=recovery_candidate,
                    corrected_text=recovery_candidate,
                    text=recovery_candidate,
                    ocr_conf=float(meta.get("evidence_conf", conf) or conf),
                    febam_reliability=mrs_value,
                    source="dual_branch_structure_recovery",
                    source_weight=float(
                        meta.get("recovery_source_weight", 0.35) or 0.35
                    ),
                )
                meta["recovery_submitted_to_vehicle_febam"] = True
            else:
                meta["recovery_submitted_to_vehicle_febam"] = False
        dual_length_hold = bool(
            is_direct_dual_evidence and dual_decision_state == "HOLD"
        )
        if dual_length_hold:
            temporal_group = self._event_fusion_string_group_id(meta)
            if temporal_group is not None:
                hypotheses = list(
                    meta.get("dual_branch_evidence_hypotheses", []) or []
                )[:2]
                seen_hold_candidates: set[str] = set()
                submitted_count = 0
                for rank, hypothesis in enumerate(hypotheses):
                    candidate_text = str(hypothesis.get("text", "") or "")
                    normalized_candidate = self._normalize_ocr_text(candidate_text)
                    if normalized_candidate in seen_hold_candidates:
                        meta["hold_candidate_duplicate_suppressed"] = True
                        continue
                    candidate_conf = float(
                        hypothesis.get("confidence", 0.0) or 0.0
                    )
                    if not self._valid_final_plate_candidate(normalized_candidate):
                        continue
                    seen_hold_candidates.add(normalized_candidate)
                    self._append_string_observation(
                        temporal_group,
                        frame_idx=frame_idx,
                        raw_text=normalized_candidate,
                        corrected_text=normalized_candidate,
                        text=normalized_candidate,
                        ocr_conf=candidate_conf,
                        febam_reliability=mrs_value,
                        source=f"dual_branch_length_hold_top{rank + 1}",
                        source_weight=0.35 if rank == 0 else 0.20,
                    )
                    submitted_count += 1
                meta["hold_candidate_submitted_count"] = submitted_count
            else:
                meta["dual_branch_hold_skip_reason"] = (
                    "pending_event_fusion_group_key"
                )
        if is_direct_dual_evidence:
            selected_raw = raw_text
            selected_norm = normalized_text
            plate_text_final = normalized_text if self._valid_final_plate_candidate(normalized_text) else ""
            final_conf = float(np.clip(conf, 0.0, 1.0)) if plate_text_final else 0.0
            pattern_type = "direct_slot_evidence" if plate_text_final else "direct_slot_evidence_hold"
            post_applied = 0
            post_reason = "direct_slot_evidence_only" if plate_text_final else "unresolved_slot_hold"
            middle_raw = self._extract_custom_korean_slot(normalized_text)
            middle_fixed = middle_raw
        else:
            selected_raw, selected_norm, plate_text_final, final_conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed = self._select_ocr_result([(None, normalized_text, conf)])
        raw_for_csv = selected_raw or raw_text
        norm_for_csv = selected_norm or normalized_text
        async_candidate_source = (
            "fastplate_custom_onnx_async_decoded_candidate"
            if row_source == "fastplate_custom_onnx"
            else "fast_plate_ocr_async_decoded_candidate"
        )
        ocr_engine_name = "fastplate_custom_onnx" if row_source == "fastplate_custom_onnx" else "fast_plate_ocr"
        if is_direct_dual_evidence:
            context_anchor = None
        elif row_source == "fastplate_custom_onnx":
            context_anchor = self._best_plate_skeleton_memory_anchor(track_id, frame_idx)
        else:
            self._update_plate_skeleton_memory(track_id, raw_text, frame_idx, source=row_source)
            context_anchor = self._find_plate_skeleton_context_anchor(track_id, raw_text, frame_idx)
        skeleton_anchor_used = int(bool(context_anchor))
        skeleton_anchor_pattern = str((context_anchor or {}).get("pattern", "") or "")
        skeleton_anchor_text = str((context_anchor or {}).get("text", "") or "")
        skeleton_anchor_support = (context_anchor or {}).get("support", "")
        variant_group_key = f"{int(track_id)}:{int(frame_idx) if frame_idx is not None else -1}:{int(candidate_idx)}"
        preferred_skeleton = self._preferred_skeleton_from_anchor(
            context_anchor,
            reason="track_anchor_missing_middle_slot",
            score_bonus=0.08,
        ) if context_anchor and not is_direct_dual_evidence else None
        variant_group_anchor_text = skeleton_anchor_text
        variant_group_merge_reason = "track_anchor_context" if context_anchor else ""
        variant_group_selected_skeleton = str((preferred_skeleton or {}).get("skeleton", "") or "")
        grammar_candidates = [] if is_direct_dual_evidence else self._grammar_decode_text(
            plate_text_final or norm_for_csv,
            conf=final_conf or conf,
            layout=plate_layout,
            previous_consensus=self._committed_plate_for_track(track_id) or self.final_plates.get(track_id, ""),
            source=row_source,
            context_anchor=context_anchor,
            preferred_skeleton=preferred_skeleton,
        )
        if grammar_candidates and not self._valid_final_plate_candidate(plate_text_final):
            grammar_best = grammar_candidates[0]
            plate_text_final = grammar_best.text
            final_conf = min(1.0, max(0.0, float(grammar_best.score or 0.0)))
            pattern_type = grammar_best.pattern
            post_applied = 1
            post_reason = grammar_best.reason
            norm_for_csv = plate_text_final
        grammar_candidate_score = (
            float(final_conf)
            if is_direct_dual_evidence and plate_text_final
            else grammar_candidates[0].score if grammar_candidates
            else (float(final_conf or conf) + self._ocr_variant_bonus(variant_name) if self._valid_final_plate_candidate(plate_text_final) else 0.0)
        )
        grammar_candidate_reason = "direct_visual_probability" if is_direct_dual_evidence else (grammar_candidates[0].reason if grammar_candidates else post_reason)
        grammar_candidate_top3 = "|".join(f"{c.text}:{c.score:.3f}:{c.reason}" for c in grammar_candidates[:3]) if grammar_candidates else (plate_text_final if plate_text_final else "")
        grammar_best_text = grammar_candidates[0].text if grammar_candidates else plate_text_final
        grammar_best_pattern = grammar_candidates[0].pattern if grammar_candidates else ("DDDKDDDD" if len(plate_text_final) == 8 else ("DDKDDDD" if len(plate_text_final) == 7 else ""))

        is_custom_onnx_result = row_source == "fastplate_custom_onnx"
        custom_onnx_raw_text = raw_text if is_custom_onnx_result else ""
        custom_onnx_korean_slot = self._extract_custom_korean_slot(raw_text) if is_custom_onnx_result else ""
        custom_onnx_digit_skeleton_used = ""
        custom_onnx_slot_candidate = ""
        custom_onnx_slot_evidence_added = 0
        custom_onnx_slot_skip_reason = ""
        custom_fullplate_diag_only = bool(getattr(self, "fastplate_custom_fullplate_diag_only", False))
        if is_custom_onnx_result and custom_fullplate_diag_only:
            custom_onnx_digit_skeleton_used, custom_onnx_slot_skip_reason = self._resolve_custom_onnx_digit_skeleton(
                track_id=track_id,
                frame_idx=frame_idx,
                context_anchor=context_anchor,
                variant_group_selected_skeleton=variant_group_selected_skeleton,
                candidate_texts=[
                    str(meta.get("normalized_plate_candidate", "") or ""),
                    str(meta.get("grammar_best_text", "") or ""),
                    str(meta.get("final_output_plate", "") or ""),
                    str(norm_for_csv or ""),
                    str(grammar_best_text or ""),
                    str(plate_text_final or ""),
                ],
            )
            if not custom_onnx_korean_slot:
                custom_onnx_slot_skip_reason = "no_korean_slot"
            elif custom_onnx_digit_skeleton_used:
                custom_onnx_slot_candidate = self._plate_from_digit_skeleton(custom_onnx_digit_skeleton_used, custom_onnx_korean_slot)
                if custom_onnx_slot_candidate and self._valid_final_plate_candidate(custom_onnx_slot_candidate):
                    self._append_string_observation(
                        track_id,
                        frame_idx=frame_idx,
                        raw_text=raw_text,
                        corrected_text=custom_onnx_slot_candidate,
                        text=custom_onnx_slot_candidate,
                        ocr_conf=conf,
                        febam_reliability=mrs_value,
                        source="fastplate_custom_onnx_korean_slot",
                        source_weight=0.25,
                    )
                    custom_onnx_slot_evidence_added = 1
                    custom_onnx_slot_skip_reason = ""
                else:
                    custom_onnx_slot_skip_reason = "invalid_slot_candidate"
            # Explicit diagnostic mode cannot influence final OCR voting.
            plate_text_final = ""
            final_conf = 0.0
            post_applied = 0
            post_reason = custom_onnx_slot_skip_reason or "custom_onnx_korean_slot_only"
            grammar_candidate_score = 0.0
            grammar_candidate_reason = post_reason
            grammar_candidate_top3 = ""
            grammar_best_text = ""
            grammar_best_pattern = ""
            source_weight = 0.0
        elif plate_text_final:
            source_weight = min(
                float(source_weight),
                self._fastplate_candidate_source_weight(grammar_candidate_reason, skeleton_anchor_used),
            )

        middle_slot_crop = item.crop_bgr if isinstance(item, OCRResultItem) else meta.get("crop_bgr")
        bio_metadata: dict[str, object] = {}
        bio_early_exit = False
        if getattr(self, "bio_adaptive_processor", None) is not None:
            try:
                bio_result = self.bio_adaptive_processor.process(
                    middle_slot_crop,
                    fastplate_text=plate_text_final or norm_for_csv or raw_for_csv,
                    fastplate_confidence=final_conf or conf,
                    meta={
                        **meta,
                        "febam_score": mrs_value,
                    },
                )
                bio_metadata = dict(bio_result.metadata or {})
                bio_metadata.setdefault("bio_route", bio_result.route)
                bio_metadata.setdefault("bio_latency_ms", bio_result.latency_ms)
                bio_early_exit = bool(bio_metadata.get("bio_early_exit", 0))
                if bio_early_exit:
                    plate_text_final = bio_result.text
                    norm_for_csv = bio_result.text
                    final_conf = bio_result.confidence
                    middle_fixed = bio_result.hangul
                    row_source = bio_result.source
                    grammar_best_text = bio_result.text
                    grammar_best_pattern = "DDDKDDDD" if len(bio_result.text) == 8 else "DDKDDDD"
                    grammar_candidate_score = bio_result.confidence
                    grammar_candidate_reason = "bio_route_a_clear_confirmed"
                    grammar_candidate_top3 = bio_result.text
                    post_applied = 1
                    post_reason = "bio_route_a_hangul_only"
            except Exception as exc:
                bio_metadata = {
                    "bio_route": "",
                    "bio_route_reason": "bio_processing_failed",
                    "bio_fallback_reason": f"{type(exc).__name__}:{str(exc)[:160]}",
                    "bio_early_exit": 0,
                }
                self._log(
                    f"[BIO_ADAPTIVE_OCR] fallback frame={frame_idx} track={track_id} "
                    f"error={type(exc).__name__}:{str(exc)[:160]}"
                )

        row_kwargs = {
            "ocr_allowlist_mode": "disabled",
            "ocr_normalized_text": norm_for_csv,
            "ocr_pattern_type": pattern_type,
            "ocr_postprocess_applied": post_applied,
            "ocr_postprocess_reason": post_reason,
            "ocr_plate_text_final": plate_text_final,
            "ocr_korean_middle_raw": middle_raw,
            "ocr_korean_middle_fixed": middle_fixed,
            "ocr_early_stop_reason": "async_batch_result",
            "raw_text": raw_for_csv,
            "raw_conf": final_conf or conf,
            "corrected_text": plate_text_final,
            "corrected_conf": final_conf or conf,
            "plate_layout": plate_layout,
            "grammar_pattern": grammar_best_pattern,
            "grammar_candidates_top3": grammar_candidate_top3,
            "grammar_best_text": grammar_best_text,
            "grammar_best_score": grammar_candidate_score,
            "grammar_best_reason": grammar_candidate_reason,
            "grammar_valid_final": int(self._valid_final_plate_candidate(plate_text_final)),
            "grammar_decoder_used": int(bool(grammar_candidates)),
            "final_candidate_source": async_candidate_source,
            "final_candidate_score": grammar_candidate_score,
            "final_candidate_reason": grammar_candidate_reason,
            "final_output_plate": plate_text_final,
            "ocr_engine": ocr_engine_name,
            "easyocr_mode": "async_batch",
            "easyocr_batch_size": "",
            "easyocr_workers": "",
            "easyocr_recognize_only": "",
            "easyocr_readtext_fallback_used": "",
            "fastplate_model": self.fastplate_model,
            "fastplate_device": self.fastplate_device,
            "fastplate_batch_size": ocr_batch_size,
            "async_delay_frames": async_delay_frames,
            "async_delay_ms": async_delay_ms,
            "ocr_saved_raw": str(meta.get("ocr_saved_raw", "") or ""),
            "ocr_saved_gray_stretched": str(meta.get("ocr_saved_gray_stretched", "") or ""),
            "ocr_saved_rotated": str(meta.get("ocr_saved_rotated", "") or ""),
            "ocr_variant_name": variant_name,
            "ocr_raw_result_short": raw_result_short,
            "normalized_plate_candidate": norm_for_csv or plate_text_final,
            "korean_plate_valid": int(self._valid_final_plate_candidate(plate_text_final)),
            "ocr_variant_tier": self._ocr_variant_tier(variant_name),
            "ocr_variant_executed": 1,
            "ocr_variants_planned": str(meta.get("ocr_variants_planned", variant_name)),
            "ocr_variants_executed_count": meta.get("ocr_variants_executed_count", ""),
            "ocr_readtext_fallback_count": "",
            "ocr_variant_skip_reason": "",
            "source": "fastplate_custom_fullplate_onnx_diag" if is_custom_onnx_result and custom_fullplate_diag_only else row_source,
            "mrs": mrs_value,
            "source_weight": source_weight,
            "dual_branch_decision_state": dual_decision_state,
            "dual_branch_length_margin": meta.get(
                "dual_branch_length_margin", ""
            ),
            "dual_branch_evidence_text": meta.get("evidence_text", ""),
            "length_top1_text": meta.get("length_top1_text", ""),
            "length_top1_log_score": meta.get("length_top1_log_score", ""),
            "length_top1_length": meta.get("length_top1_length", ""),
            "length_top2_text": meta.get("length_top2_text", ""),
            "length_top2_log_score": meta.get("length_top2_log_score", ""),
            "length_top2_length": meta.get("length_top2_length", ""),
            "length_score_margin": meta.get("length_score_margin", ""),
            "length_decision_state": meta.get("length_decision_state", ""),
            "length_score_contract": meta.get("length_score_contract", ""),
            "hold_candidate_duplicate_suppressed": int(
                bool(meta.get("hold_candidate_duplicate_suppressed", False))
            ),
            "hold_candidate_submitted_count": meta.get(
                "hold_candidate_submitted_count", 0
            ),
            "custom_onnx_raw_text": custom_onnx_raw_text,
            "custom_onnx_korean_slot": custom_onnx_korean_slot,
            "custom_onnx_digit_skeleton_used": custom_onnx_digit_skeleton_used,
            "custom_onnx_slot_candidate": custom_onnx_slot_candidate,
            "custom_onnx_slot_evidence_added": custom_onnx_slot_evidence_added,
            "custom_onnx_slot_skip_reason": custom_onnx_slot_skip_reason,
            "skeleton_anchor_used": skeleton_anchor_used,
            "skeleton_anchor_pattern": skeleton_anchor_pattern,
            "skeleton_anchor_text": skeleton_anchor_text,
            "skeleton_anchor_support": skeleton_anchor_support,
            "variant_group_key": variant_group_key,
            "variant_group_anchor_text": variant_group_anchor_text,
            "variant_group_merge_reason": variant_group_merge_reason,
            "variant_group_selected_skeleton": variant_group_selected_skeleton,
            **bio_metadata,
        }
        row_kwargs = self._sanitize_ocr_csv_row_kwargs(row_kwargs)

        if (
            middle_slot_crop is not None
            and not bio_early_exit
            and not dual_length_hold
            and not (is_custom_onnx_result and custom_fullplate_diag_only)
        ):
            self._enqueue_middle_slot_upl(
                middle_slot_crop,
                ocr_text=norm_for_csv or raw_for_csv,
                ocr_conf=final_conf or conf,
                source=async_candidate_source,
                meta={
                    **meta,
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "candidate_idx": candidate_idx,
                    "bbox_roi": bbox_roi,
                    "bbox": bbox_roi,
                    "mrs": mrs_value,
                    "variant_group_key": variant_group_key,
                    "source_weight": self.middle_slot_source_weight,
                    "variant_name": variant_name,
                    "raw_text": raw_for_csv,
                    "corrected_text": plate_text_final,
                    "normalized_plate_candidate": norm_for_csv or plate_text_final,
                    "grammar_best_text": grammar_best_text,
                    "final_output_plate": plate_text_final,
                    "ocr_plate_text_final": plate_text_final,
                    "variant_group_selected_skeleton": variant_group_selected_skeleton,
                    "variant_group_anchor_text": variant_group_anchor_text,
                    "skeleton_anchor_text": skeleton_anchor_text,
                    "skeleton_anchor_pattern": skeleton_anchor_pattern,
                    "skeleton_anchor_support": skeleton_anchor_support,
                    "plate_layout": plate_layout,
                },
            )

        direct_observation_recorded = False
        if is_direct_dual_evidence and bool(getattr(self, "gpu_final_vehicle_grouping", False)):
            string_group_id, string_group_key = self._gpu_final_vehicle_group_id(
                meta,
                frame_idx=int(frame_idx),
                track_id=int(track_id),
                bbox=bbox_roi,
                candidate=plate_text_final or norm_for_csv or raw_for_csv,
            )
        elif is_direct_dual_evidence:
            string_group_id = self._event_fusion_string_group_id(meta)
            string_group_key = (
                str(meta.get("event_fusion_group_key", "") or "")
                if string_group_id is not None
                else ""
            )
        elif bool(getattr(self, "gpu_final_vehicle_grouping", False)):
            string_group_id, string_group_key = self._gpu_final_vehicle_group_id(
                meta,
                frame_idx=int(frame_idx),
                track_id=int(track_id),
                bbox=bbox_roi,
                candidate=norm_for_csv or raw_for_csv,
            )
        else:
            string_group_id, string_group_key = self._resolve_string_febam_group_id(meta, track_id)

        # Temporal support is counted only after final vehicle identity is
        # available. Unique frame_idx keys prevent repeated variants or split
        # fragments from manufacturing the N>=4 transition.
        routed_text = plate_text_final
        routed_conf = float(final_conf or conf)
        routed_source = row_source
        routed_source_weight = float(source_weight)
        if bool(getattr(self, "gpu_final_vehicle_grouping", False)):
            if string_group_id is None:
                self.final_vehicle_identity_hold_count += 1
                route = {
                    "state": "HOLD",
                    "reason": "FINAL_VEHICLE_ID_PENDING",
                    "n_valid": 0,
                    "posterior_active": False,
                }
            else:
                route = self.final_vehicle_posterior_gate.append(
                    video_id=str(meta.get("video_id", "") or ""),
                    event_fusion_group_key=str(meta.get("event_fusion_group_key", "") or ""),
                    vehicle_instance_id=str(meta.get("vehicle_instance_id", "") or ""),
                    frame_idx=int(frame_idx),
                    text=str(plate_text_final or ""),
                    confidence=float(final_conf or conf),
                    posterior_valid=bool(
                        plate_text_final
                        and self._valid_final_plate_candidate(plate_text_final)
                        and not fastplate_exception
                    ),
                )
                if route.get("posterior_active", False):
                    self.final_vehicle_posterior_route_count += 1
                    routed_text = str(route.get("text", "") or "")
                    routed_conf = float(route.get("confidence", 0.0) or 0.0)
                    routed_source = "position_posterior"
                    routed_source_weight = min(float(source_weight), 0.35)
                else:
                    self.final_vehicle_sparse_route_count += 1
            self.final_vehicle_last_route = dict(route)
            meta["final_vehicle_temporal_state"] = route.get("state", "HOLD")
            meta["final_vehicle_temporal_reason"] = route.get("reason", "")
            meta["final_vehicle_n_valid"] = int(route.get("n_valid", 0) or 0)
            meta["final_vehicle_posterior_active"] = int(bool(route.get("posterior_active", False)))
            row_kwargs.update({
                "final_vehicle_temporal_state": meta["final_vehicle_temporal_state"],
                "final_vehicle_temporal_reason": meta["final_vehicle_temporal_reason"],
                "final_vehicle_n_valid": meta["final_vehicle_n_valid"],
                "final_vehicle_posterior_active": meta["final_vehicle_posterior_active"],
            })

        # A frame-local dual-branch winner is evidence, never a final result.
        # Only a canonical String-FEBAM COMMIT may promote it.
        if (
            is_direct_dual_evidence
            and dual_decision_state == "COMMIT_CANDIDATE"
            and plate_text_final
        ):
            observation_text = routed_text
            plate_text_final = ""
            if string_group_id is not None:
                committed = self._append_ocr_history(
                    track_id,
                    observation_text,
                    routed_conf,
                    frame_idx,
                    bbox_roi,
                    mrs_value,
                    raw_text=raw_for_csv,
                    corrected_text=observation_text,
                    source=routed_source,
                    source_weight=routed_source_weight,
                    string_group_id=string_group_id,
                    string_group_key=string_group_key,
                    allow_legacy_vote=False,
                )
                direct_observation_recorded = True
                if self._valid_final_plate_candidate(committed):
                    plate_text_final = committed
            else:
                meta.setdefault(
                    "dual_branch_commit_skip_reason",
                    meta.get("dual_branch_hold_skip_reason", "pending_event_fusion_group_key"),
                )
            for field in ("ocr_plate_text_final", "corrected_text", "final_output_plate"):
                row_kwargs[field] = plate_text_final
            row_kwargs["grammar_valid_final"] = int(
                self._valid_final_plate_candidate(plate_text_final)
            )
            row_kwargs["korean_plate_valid"] = row_kwargs["grammar_valid_final"]
        if plate_text_final:
            self._append_ocr_csv_row(
                track_id,
                frame_idx,
                raw_for_csv,
                final_conf or conf,
                bbox_roi,
                candidate_idx=candidate_idx,
                roi_w=roi_w,
                roi_h=roi_h,
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
                gray_stretched_ocr_used=gray_stretched_ocr_used,
                attempted_count=attempted_count,
                **row_kwargs,
            )
            if self.ocr_csv_rows:
                self.ocr_csv_rows[-1]["group_id"] = string_group_key
            self._last_ocr_text = plate_text_final
            self._last_ocr_conf = float(final_conf or conf)
            if direct_observation_recorded:
                self._last_final_plate = plate_text_final
            else:
                self._last_final_plate = self._append_ocr_history(
                    track_id,
                    routed_text,
                    routed_conf,
                    frame_idx,
                    bbox_roi,
                    mrs_value,
                    raw_text=raw_for_csv,
                    corrected_text=plate_text_final,
                    source=routed_source,
                    source_weight=routed_source_weight,
                    string_group_id=string_group_id,
                    string_group_key=string_group_key,
                )
        else:
            debug_kwargs = self._sanitize_ocr_debug_csv_row_kwargs(
                row_kwargs,
                extra_duplicate_keys={
                    "custom_onnx_raw_text",
                    "custom_onnx_korean_slot",
                    "custom_onnx_digit_skeleton_used",
                    "custom_onnx_slot_candidate",
                    "custom_onnx_slot_evidence_added",
                    "custom_onnx_slot_skip_reason",
                },
            )
            self._append_ocr_debug_csv_row(
                frame_idx=frame_idx,
                track_id=track_id,
                candidate_idx=candidate_idx,
                bbox=bbox_roi,
                roi_w=roi_w,
                roi_h=roi_h,
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
                gray_stretched_ocr_used=gray_stretched_ocr_used,
                attempted_count=attempted_count,
                ocr_attempt=1,
                ocr_skip_reason=(
                    str(meta.get("ocr_skip_reason", "") or f"fastplate_async_exception_{fastplate_error}")
                    if fastplate_exception
                    else f"fastplate_async_empty_{variant_name}"
                ),
                text=raw_text,
                conf="" if conf == 0.0 else conf,
                ocr_allowlist_mode="disabled",
                ocr_normalized_text=norm_for_csv,
                ocr_pattern_type=pattern_type,
                ocr_postprocess_applied=post_applied,
                ocr_postprocess_reason=post_reason,
                ocr_plate_text_final=plate_text_final,
                ocr_engine=ocr_engine_name,
                ocr_source=row_source,
                custom_onnx_raw_text=custom_onnx_raw_text,
                custom_onnx_korean_slot=custom_onnx_korean_slot,
                custom_onnx_digit_skeleton_used=custom_onnx_digit_skeleton_used,
                custom_onnx_slot_candidate=custom_onnx_slot_candidate,
                custom_onnx_slot_evidence_added=custom_onnx_slot_evidence_added,
                custom_onnx_slot_skip_reason=custom_onnx_slot_skip_reason,
                easyocr_mode="async_batch",
                fastplate_model=self.fastplate_model,
                fastplate_device=self.fastplate_device,
                fastplate_batch_size=ocr_batch_size,
                async_delay_frames=async_delay_frames,
                async_delay_ms=async_delay_ms,
                ocr_variant_name=variant_name,
                ocr_raw_result_short=raw_result_short,
                normalized_plate_candidate=norm_for_csv,
                **debug_kwargs,
            )

    def _run_easyocr_raw_text(self, image_rgb: np.ndarray, allowlist: str | None = None, variant_name: str = "split") -> tuple[str, float]:
        if self.easyocr_recognize_only:
            recognizer = self._get_easyocr_batch_recognizer(allowlist)
            result = recognizer.recognize_one_box(image_rgb, variant_name=variant_name)
            return clean_ocr_text(str(result.get("text", "") or "")), float(result.get("conf", 0.0) or 0.0)

        reader = self._get_easyocr_reader()
        kwargs = {"detail": 1, "paragraph": False, "batch_size": self.easyocr_batch_size}
        if self.easyocr_workers >= 0:
            kwargs["workers"] = self.easyocr_workers
        if allowlist is not None:
            kwargs["allowlist"] = allowlist
        try:
            results = reader.readtext(image_rgb, **kwargs)
        except TypeError:
            kwargs.pop("workers", None)
            results = reader.readtext(image_rgb, **kwargs)
        parts: list[str] = []
        confs: list[float] = []
        for item in results or []:
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    parts.append(str(item[1]))
                if len(item) >= 3:
                    try:
                        confs.append(float(item[2]))
                    except Exception:
                        pass
            else:
                parts.append(str(item))
        return clean_ocr_text("".join(parts)), (max(confs) if confs else 0.0)

    def _run_split_ocr_fallback(self, image_rgb: np.ndarray, layout: str, previous_consensus: str = "") -> tuple[str, float, dict[str, object]]:
        if image_rgb is None or image_rgb.size == 0:
            return "", 0.0, {}
        h, w = image_rgb.shape[:2]
        if h < 4 or w < 8:
            return "", 0.0, {}
        debug: dict[str, object] = {
            "split_ocr_used": 0,
            "split_fallback_used": 0,
            "split_mode": "",
            "split_left_text": "",
            "split_mid_text": "",
            "split_right_text": "",
            "split_top_text": "",
            "split_bottom_text": "",
        }
        candidates = []
        left_img = image_rgb[:, :max(1, int(w * 0.35))]
        mid_img = image_rgb[:, max(0, int(w * 0.20)):max(1, int(w * 0.60))]
        right_img = image_rgb[:, max(0, int(w * 0.42)):]
        left, left_conf = self._run_easyocr_raw_text(left_img, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", variant_name="split_left")
        mid, mid_conf = self._run_easyocr_raw_text(mid_img, self.ocr_allowlist_expanded, variant_name="split_mid")
        right, right_conf = self._run_easyocr_raw_text(right_img, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", variant_name="split_right")
        debug.update({"split_left_text": left, "split_mid_text": mid, "split_right_text": right, "split_mode": "one_line"})
        candidates.extend(self._grammar_decode_split_texts(left=left[-3:], mid=mid[:1], right=right[-4:], conf=max(left_conf, mid_conf, right_conf), layout=layout, previous_consensus=previous_consensus))
        if layout == "two_line_possible":
            top_img = image_rgb[:max(1, int(h * 0.58)), :]
            bottom_img = image_rgb[max(0, int(h * 0.35)):, :]
            top, top_conf = self._run_easyocr_raw_text(top_img, self.ocr_allowlist_expanded, variant_name="split_top")
            bottom, bottom_conf = self._run_easyocr_raw_text(bottom_img, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", variant_name="split_bottom")
            debug.update({"split_top_text": top, "split_bottom_text": bottom, "split_mode": "two_line"})
            candidates.extend(self._grammar_decode_split_texts(top=top[-4:], bottom=bottom[-4:], conf=max(top_conf, bottom_conf), layout=layout, previous_consensus=previous_consensus))
        if not candidates:
            return "", 0.0, debug
        best = sorted(candidates, key=lambda item: item.score, reverse=True)[0]
        debug.update({
            "split_ocr_used": 1,
            "split_fallback_used": 1,
            "grammar_pattern": best.pattern,
            "grammar_best_text": best.text,
            "grammar_best_score": best.score,
            "grammar_best_reason": best.reason,
            "grammar_valid_final": int(is_valid_final_plate(best.text)),
            "grammar_candidates_top3": "|".join(f"{c.text}:{c.score:.3f}" for c in sorted(candidates, key=lambda item: item.score, reverse=True)[:3]),
        })
        return best.text, best.score, debug

    def _append_ocr_debug_csv_row(
        self,
        *,
        frame_idx: int | None,
        track_id: int | None = None,
        candidate_idx: int | None = None,
        ocr_attempt: int = 0,
        ocr_skip_reason: str = "",
        text: str = "",
        conf: float | str = "",
        bbox=None,
        roi_w: int | str = "",
        roi_h: int | str = "",
        candidate_count: int = 0,
        track_id_count: int = 0,
        febam_confirmed_count: int = 0,
        gray_stretched_ocr_used: bool = False,
        is_final: int = 0,
        final_plate: str = "",
        attempted_count: int = 0,
        febam_score: object = "",
        febam_energy: object = "",
        febam_threshold: object = "",
        febam_memory: object = "",
        febam_confirmed: int = 0,
        ocr_input_type: str = "",
        ocr_candidate_rank: str = "",
        ocr_crop_policy: str = "",
        ocr_crop_expand_top: int | str = "",
        ocr_y1_original: int | str = "",
        ocr_y1_expanded: int | str = "",
        roi_aspect: float | str = "",
        ocr_expand_top: int | str = "",
        ocr_angle: float | str = "",
        ocr_rotation_applied: int | str = "",
        ocr_angle_score: float | str = "",
        ocr_gpu_rectified: int | str = "",
        ocr_trigger_reason: str = "",
        near_confirmed_large_roi: int | str = "",
        febam_full_confirmed: int | str = "",
        near_confirmed_ocr_sample: int | str = "",
        ocr_sampling_reason: str = "",
        ocr_sampling_level: str = "",
        source_weight: float | str = "",
        pre_febam_candidate_count: int | str = "",
        post_febam_confirmed_count: int | str = "",
        febam_score_thr: float | str = "",
        febam_energy_thr: float | str = "",
        febam_memory_min: int | str = "",
        ocr_saved_raw: str = "",
        ocr_saved_gray_stretched: str = "",
        ocr_saved_rotated: str = "",
        ocr_input_variant: str = "",
        ocr_fallback_rank: int | str = "",
        ocr_fallback_used: int | str = "",
        ocr_rotation_threshold: float | str = "",
        ocr_reader_langs: str = "",
        ocr_allowlist_mode: str = "",
        ocr_normalized_text: str = "",
        ocr_pattern_type: str = "",
        ocr_postprocess_applied: int | str = "",
        ocr_postprocess_reason: str = "",
        ocr_plate_text_final: str = "",
        ocr_korean_middle_raw: str = "",
        ocr_korean_middle_fixed: str = "",
        ocr_small_roi: int | str = "",
        ocr_upscale_applied: int | str = "",
        ocr_upscale_factor: float | str = "",
        ocr_sharpen_applied: int | str = "",
        ocr_sharpen_amount: float | str = "",
        mrs: float | str = "",
        best_text: str = "",
        best_conf: float | str = "",
        final_voted_text: str = "",
        final_vote_count: int | str = "",
        mrs_max: float | str = "",
        mrs_mean: float | str = "",
        ocr_count: int | str = "",
        topk_ocr: str = "",
        raw_text: str = "",
        raw_conf: float | str = "",
        corrected_text: str = "",
        corrected_conf: float | str = "",
        voted_text: str = "",
        voted_conf: float | str = "",
        final_observed_plate: str = "",
        final_consensus_plate: str = "",
        string_febam_state: str = "",
        string_febam_activation: float | str = "",
        string_febam_score: float | str = "",
        string_febam_energy: float | str = "",
        string_febam_memory: float | str = "",
        segment_id: int | str = "",
        segment_len: int | str = "",
        segment_effective_count: float | str = "",
        segment_weight: float | str = "",
        segment_text_medoid: str = "",
        segment_consensus_text: str = "",
        cluster_count: int | str = "",
        contamination_flag: int | str = "",
        commit_frame: int | str = "",
        first_stable_frame: int | str = "",
        ocr_call_count: int | str = "",
        ocr_skipped_after_commit: int | str = "",
        ocr_source: str = "",
        final_plate_source: str = "observed",
        ocr_variant_order: str = "",
        ocr_variant_count: int | str = "",
        ocr_norm_w: int | str = "",
        ocr_norm_h: int | str = "",
        ocr_norm_scale: float | str = "",
        ocr_norm_pad_x: int | str = "",
        ocr_norm_pad_y: int | str = "",
        ocr_norm_aspect_preserved: int | str = "",
        ocr_allowlist_sweep_disabled: int | str = "",
        ocr_early_stop_reason: str = "",
        ocr_selected_final_candidate_only: int | str = "",
        plate_layout: str = "",
        grammar_pattern: str = "",
        grammar_candidates_top3: str = "",
        grammar_best_text: str = "",
        grammar_best_score: float | str = "",
        grammar_best_reason: str = "",
        grammar_valid_final: int | str = "",
        split_ocr_used: int | str = "",
        split_left_text: str = "",
        split_mid_text: str = "",
        split_right_text: str = "",
        split_top_text: str = "",
        split_bottom_text: str = "",
        committed_plate_locked: int | str = "",
        ocr_crop_expand_top_ratio: float | str = "",
        grammar_decoder_used: int | str = "",
        split_fallback_used: int | str = "",
        split_mode: str = "",
        final_candidate_source: str = "",
        final_candidate_score: float | str = "",
        final_candidate_reason: str = "",
        final_output_plate: str = "",
        ocr_engine: str = "",
        easyocr_mode: str = "",
        easyocr_batch_size: int | str = "",
        easyocr_workers: int | str = "",
        easyocr_recognize_only: int | str = "",
        easyocr_readtext_fallback_used: int | str = "",
        fastplate_model: str = "",
        fastplate_device: str = "",
        fastplate_batch_size: int | str = "",
        async_delay_frames: int | float | str = "",
        async_delay_ms: int | float | str = "",
        ocr_crop_expand_left_ratio: float | str = "",
        ocr_crop_expand_right_ratio: float | str = "",
        ocr_crop_expand_bottom_ratio: float | str = "",
        ocr_variant_name: str = "",
        ocr_raw_result_short: str = "",
        normalized_plate_candidate: str = "",
        korean_plate_valid: int | str = "",
        ocr_variant_tier: int | str = "",
        ocr_variant_executed: int | str = "",
        ocr_variants_planned: str = "",
        ocr_variants_executed_count: int | str = "",
        ocr_readtext_fallback_count: int | str = "",
        ocr_variant_skip_reason: str = "",
        skeleton_anchor_used: int | str = "",
        skeleton_anchor_pattern: str = "",
        skeleton_anchor_text: str = "",
        skeleton_anchor_support: int | str = "",
        variant_group_key: str = "",
        variant_group_anchor_text: str = "",
        variant_group_merge_reason: str = "",
        variant_group_selected_skeleton: str = "",
        custom_onnx_raw_text: str = "",
        custom_onnx_korean_slot: str = "",
        custom_onnx_digit_skeleton_used: str = "",
        custom_onnx_slot_candidate: str = "",
        custom_onnx_slot_evidence_added: int | str = "",
        custom_onnx_slot_skip_reason: str = "",
        **extra_values,
    ) -> None:
        if pre_febam_candidate_count == "":
            pre_febam_candidate_count = candidate_count
        if post_febam_confirmed_count == "":
            post_febam_confirmed_count = febam_confirmed_count
        if febam_score_thr == "":
            febam_score_thr = getattr(self, "febam_score_thr", 0.35)
        if febam_energy_thr == "":
            febam_energy_thr = getattr(self, "febam_energy_thr", 0.40)
        if febam_memory_min == "":
            febam_memory_min = getattr(self, "febam_memory_min", 2)
        if not ocr_trigger_reason:
            ocr_trigger_reason = ocr_skip_reason
        if ocr_sampling_reason == "":
            ocr_sampling_reason = ocr_trigger_reason
        if ocr_sampling_level == "":
            ocr_sampling_level = self._sampling_level_from_reason(str(ocr_sampling_reason or ocr_trigger_reason))
        if source_weight == "":
            source_weight = self._ocr_sampling_source_weight(str(ocr_sampling_level)) if ocr_sampling_level else ""
        if febam_full_confirmed == "":
            febam_full_confirmed = febam_confirmed
        if near_confirmed_ocr_sample == "":
            near_confirmed_ocr_sample = 1 if ocr_sampling_reason == "near_confirmed_ocr_sample" else 0
        if ocr_rotation_threshold == "":
            ocr_rotation_threshold = 8.0
        if not ocr_reader_langs:
            ocr_reader_langs = "+".join(getattr(self, "ocr_reader_langs", ["ko", "en"]))
        if ocr_crop_expand_top_ratio == "" and roi_h not in {"", 0, "0"}:
            try:
                ocr_crop_expand_top_ratio = float(ocr_crop_expand_top or ocr_expand_top or 0) / max(float(roi_h), 1.0)
            except Exception:
                ocr_crop_expand_top_ratio = ""
        if grammar_decoder_used == "":
            grammar_decoder_used = int(bool(grammar_best_text or grammar_candidates_top3 or grammar_best_reason))
        if split_fallback_used == "":
            split_fallback_used = split_ocr_used
        if not final_candidate_source:
            final_candidate_source = final_plate_source
        if final_candidate_score == "":
            final_candidate_score = grammar_best_score
        if not final_candidate_reason:
            final_candidate_reason = grammar_best_reason
        if not final_output_plate:
            final_output_plate = final_plate
        if not ocr_engine:
            ocr_engine = "easyocr"
        if not easyocr_mode:
            easyocr_mode = "recognize_only" if self.easyocr_recognize_only else "readtext"
        if easyocr_batch_size == "":
            easyocr_batch_size = self.easyocr_batch_size
        if easyocr_workers == "":
            easyocr_workers = self.easyocr_workers
        if easyocr_recognize_only == "":
            easyocr_recognize_only = int(self.easyocr_recognize_only)
        if easyocr_readtext_fallback_used == "":
            easyocr_readtext_fallback_used = 0
        if ocr_crop_expand_left_ratio == "":
            ocr_crop_expand_left_ratio = getattr(self, "ocr_expand_left_ratio", "")
        if ocr_crop_expand_right_ratio == "":
            ocr_crop_expand_right_ratio = getattr(self, "ocr_expand_right_ratio", "")
        if ocr_crop_expand_bottom_ratio == "":
            ocr_crop_expand_bottom_ratio = getattr(self, "ocr_expand_bottom_ratio", "")
        if not ocr_variant_name:
            ocr_variant_name = ocr_input_variant
        if not normalized_plate_candidate:
            normalized_plate_candidate = ocr_normalized_text or corrected_text or final_plate
        if korean_plate_valid == "":
            korean_plate_valid = int(self._valid_final_plate_candidate(str(normalized_plate_candidate or "")))
        if mrs == "" and febam_score != "":
            mrs = febam_score
        if track_id is not None and self.use_string_febam:
            string_values = self._string_febam_csv_values(int(track_id))
            final_observed_plate = final_observed_plate or str(string_values.get("final_observed_plate", ""))
            final_consensus_plate = final_consensus_plate or str(string_values.get("final_consensus_plate", ""))
            string_febam_state = string_febam_state or str(string_values.get("string_febam_state", ""))
            if string_febam_state == "COMMIT" and self._valid_final_plate_candidate(final_observed_plate):
                final_plate = final_observed_plate
                final_plate_source = "string_febam_commit"
                committed_plate_locked = 1
                final_candidate_source = "committed_plate"
                final_candidate_reason = "committed_plate_locked"
            string_febam_activation = string_febam_activation if string_febam_activation != "" else string_values.get("string_febam_activation", "")
            string_febam_score = string_febam_score if string_febam_score != "" else string_values.get("string_febam_score", "")
            string_febam_energy = string_febam_energy if string_febam_energy != "" else string_values.get("string_febam_energy", "")
            string_febam_memory = string_febam_memory if string_febam_memory != "" else string_values.get("string_febam_memory", "")
            segment_id = segment_id if segment_id != "" else string_values.get("segment_id", "")
            segment_len = segment_len if segment_len != "" else string_values.get("segment_len", "")
            segment_effective_count = segment_effective_count if segment_effective_count != "" else string_values.get("segment_effective_count", "")
            segment_weight = segment_weight if segment_weight != "" else string_values.get("segment_weight", "")
            segment_text_medoid = segment_text_medoid or str(string_values.get("segment_text_medoid", ""))
            segment_consensus_text = segment_consensus_text or str(string_values.get("segment_consensus_text", ""))
            cluster_count = cluster_count if cluster_count != "" else string_values.get("cluster_count", "")
            contamination_flag = contamination_flag if contamination_flag != "" else string_values.get("contamination_flag", "")
            commit_frame = commit_frame if commit_frame != "" else string_values.get("commit_frame", "")
            first_stable_frame = first_stable_frame if first_stable_frame != "" else string_values.get("first_stable_frame", "")
            ocr_call_count = ocr_call_count if ocr_call_count != "" else string_values.get("ocr_call_count", "")
            ocr_skipped_after_commit = ocr_skipped_after_commit if ocr_skipped_after_commit != "" else string_values.get("ocr_skipped_after_commit", "")
        x1, y1, x2, y2 = self._bbox_to_csv_values(bbox)
        def probability_value(value: object, field: str) -> object:
            if value == "" or value is None:
                return ""
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                self.invalid_probability_count += 1
                if len(self.invalid_probability_field_examples) < 10:
                    self.invalid_probability_field_examples.append(f"{field}={value!r}")
                return 0.0
            if not math.isfinite(numeric):
                self.invalid_probability_count += 1
                numeric = 0.0
            clipped = max(0.0, min(1.0, numeric))
            if clipped != numeric:
                self.probability_clip_count += 1
                if len(self.invalid_probability_field_examples) < 10:
                    self.invalid_probability_field_examples.append(f"{field}={numeric}")
            return clipped
        row = {
            "track_id": int(track_id) if track_id is not None else -1,
            "frame_idx": int(frame_idx) if frame_idx is not None else -1,
            "candidate_idx": int(candidate_idx) if candidate_idx is not None else -1,
            "ocr_attempt": int(ocr_attempt),
            "ocr_skip_reason": str(ocr_skip_reason or ""),
            "ocr_text": str(text or ""),
            "ocr_conf": probability_value(conf, "ocr_conf"),
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "roi_w": roi_w,
            "roi_h": roi_h,
            "candidate_count": int(candidate_count),
            "track_id_count": int(track_id_count),
            "febam_confirmed_count": int(febam_confirmed_count),
            "gray_stretched_ocr_used": int(bool(gray_stretched_ocr_used)),
            "is_final": int(is_final),
            "final_plate": str(final_plate or ""),
            "attempted_count": int(attempted_count),
            "febam_score": febam_score,
            "febam_energy": febam_energy,
            "febam_threshold": febam_threshold,
            "febam_memory": febam_memory,
            "febam_confirmed": int(febam_confirmed),
            "ocr_input_type": str(ocr_input_type or ""),
            "ocr_candidate_rank": str(ocr_candidate_rank or ""),
            "ocr_crop_policy": str(ocr_crop_policy or ""),
            "ocr_crop_expand_top": ocr_crop_expand_top,
            "ocr_y1_original": ocr_y1_original,
            "ocr_y1_expanded": ocr_y1_expanded,
            "roi_aspect": roi_aspect,
            "ocr_expand_top": ocr_expand_top,
            "ocr_angle": ocr_angle,
            "ocr_rotation_applied": ocr_rotation_applied,
            "ocr_angle_score": ocr_angle_score,
            "ocr_gpu_rectified": ocr_gpu_rectified,
            "ocr_trigger_reason": str(ocr_trigger_reason or ""),
            "near_confirmed_large_roi": near_confirmed_large_roi,
            "febam_full_confirmed": int(febam_full_confirmed) if febam_full_confirmed != "" else 0,
            "near_confirmed_ocr_sample": int(near_confirmed_ocr_sample) if near_confirmed_ocr_sample != "" else 0,
            "ocr_sampling_level": str(ocr_sampling_level or ""),
            "ocr_sampling_reason": str(ocr_sampling_reason or ""),
            "source_weight": source_weight,
            "pre_febam_candidate_count": pre_febam_candidate_count,
            "post_febam_confirmed_count": post_febam_confirmed_count,
            "febam_score_thr": febam_score_thr,
            "febam_energy_thr": febam_energy_thr,
            "febam_memory_min": febam_memory_min,
            "ocr_saved_raw": str(ocr_saved_raw or ""),
            "ocr_saved_gray_stretched": str(ocr_saved_gray_stretched or ""),
            "ocr_saved_rotated": str(ocr_saved_rotated or ""),
            "ocr_input_variant": str(ocr_input_variant or ""),
            "ocr_fallback_rank": ocr_fallback_rank,
            "ocr_fallback_used": ocr_fallback_used,
            "ocr_rotation_threshold": ocr_rotation_threshold,
            "ocr_reader_langs": str(ocr_reader_langs or ""),
            "ocr_allowlist_mode": str(ocr_allowlist_mode or ""),
            "ocr_normalized_text": str(ocr_normalized_text or ""),
            "ocr_pattern_type": str(ocr_pattern_type or ""),
            "ocr_postprocess_applied": ocr_postprocess_applied,
            "ocr_postprocess_reason": str(ocr_postprocess_reason or ""),
            "ocr_plate_text_final": str(ocr_plate_text_final or ""),
            "ocr_korean_middle_raw": str(ocr_korean_middle_raw or ""),
            "ocr_korean_middle_fixed": str(ocr_korean_middle_fixed or ""),
            "ocr_small_roi": ocr_small_roi,
            "ocr_upscale_applied": ocr_upscale_applied,
            "ocr_upscale_factor": ocr_upscale_factor,
            "ocr_sharpen_applied": ocr_sharpen_applied,
            "ocr_sharpen_amount": ocr_sharpen_amount,
            "mrs": mrs,
            "best_text": str(best_text or ""),
            "best_conf": best_conf,
            "final_voted_text": str(final_voted_text or ""),
            "final_vote_count": final_vote_count,
            "mrs_max": mrs_max,
            "mrs_mean": mrs_mean,
            "ocr_count": ocr_count,
            "topk_ocr": str(topk_ocr or ""),
            "raw_text": str(raw_text or text or ""),
            "raw_conf": probability_value(raw_conf if raw_conf != "" else conf, "raw_conf"),
            "corrected_text": str(corrected_text or final_plate or ""),
            "corrected_conf": probability_value(corrected_conf if corrected_conf != "" else conf, "corrected_conf"),
            "ocr_source": str(ocr_source or ""),
            "voted_text": str(voted_text or ("" if final_plate_source == "string_febam_commit" else final_plate) or ""),
            "voted_conf": voted_conf,
            "final_observed_plate": str(final_observed_plate or ""),
            "final_consensus_plate": str(final_consensus_plate or ""),
            "final_plate_source": str(final_plate_source or "observed"),
            "ocr_variant_order": str(ocr_variant_order or ""),
            "ocr_variant_count": ocr_variant_count,
            "ocr_norm_w": ocr_norm_w,
            "ocr_norm_h": ocr_norm_h,
            "ocr_norm_scale": ocr_norm_scale,
            "ocr_norm_pad_x": ocr_norm_pad_x,
            "ocr_norm_pad_y": ocr_norm_pad_y,
            "ocr_norm_aspect_preserved": ocr_norm_aspect_preserved,
            "ocr_allowlist_sweep_disabled": ocr_allowlist_sweep_disabled,
            "ocr_early_stop_reason": str(ocr_early_stop_reason or ""),
            "ocr_selected_final_candidate_only": ocr_selected_final_candidate_only,
            "plate_layout": str(plate_layout or ""),
            "grammar_pattern": str(grammar_pattern or ""),
            "grammar_candidates_top3": str(grammar_candidates_top3 or ""),
            "grammar_best_text": str(grammar_best_text or ""),
            "grammar_best_score": grammar_best_score,
            "grammar_best_reason": str(grammar_best_reason or ""),
            "grammar_valid_final": grammar_valid_final,
            "split_ocr_used": split_ocr_used,
            "split_left_text": str(split_left_text or ""),
            "split_mid_text": str(split_mid_text or ""),
            "split_right_text": str(split_right_text or ""),
            "split_top_text": str(split_top_text or ""),
            "split_bottom_text": str(split_bottom_text or ""),
            "committed_plate_locked": committed_plate_locked,
            "ocr_crop_expand_top_ratio": ocr_crop_expand_top_ratio,
            "grammar_decoder_used": grammar_decoder_used,
            "split_fallback_used": split_fallback_used,
            "split_mode": str(split_mode or ""),
            "final_candidate_source": str(final_candidate_source or ""),
            "final_candidate_score": final_candidate_score,
            "final_candidate_reason": str(final_candidate_reason or ""),
            "final_output_plate": str(final_output_plate or ""),
            "ocr_engine": str(ocr_engine or ""),
            "easyocr_mode": str(easyocr_mode or ""),
            "easyocr_batch_size": easyocr_batch_size,
            "easyocr_workers": easyocr_workers,
            "easyocr_recognize_only": easyocr_recognize_only,
            "easyocr_readtext_fallback_used": easyocr_readtext_fallback_used,
            "fastplate_model": str(fastplate_model or ""),
            "fastplate_device": str(fastplate_device or ""),
            "fastplate_batch_size": fastplate_batch_size,
            "async_delay_frames": async_delay_frames,
            "async_delay_ms": async_delay_ms,
            "ocr_crop_expand_left_ratio": ocr_crop_expand_left_ratio,
            "ocr_crop_expand_right_ratio": ocr_crop_expand_right_ratio,
            "ocr_crop_expand_bottom_ratio": ocr_crop_expand_bottom_ratio,
            "ocr_variant_name": str(ocr_variant_name or ""),
            "ocr_raw_result_short": str(ocr_raw_result_short or ""),
            "normalized_plate_candidate": str(normalized_plate_candidate or ""),
            "korean_plate_valid": korean_plate_valid,
            "ocr_variant_tier": ocr_variant_tier,
            "ocr_variant_executed": ocr_variant_executed,
            "ocr_variants_planned": str(ocr_variants_planned or ""),
            "ocr_variants_executed_count": ocr_variants_executed_count,
            "ocr_readtext_fallback_count": ocr_readtext_fallback_count,
            "ocr_variant_skip_reason": str(ocr_variant_skip_reason or ""),
            "skeleton_anchor_used": skeleton_anchor_used,
            "skeleton_anchor_pattern": str(skeleton_anchor_pattern or ""),
            "skeleton_anchor_text": str(skeleton_anchor_text or ""),
            "skeleton_anchor_support": skeleton_anchor_support,
            "variant_group_key": str(variant_group_key or ""),
            "variant_group_anchor_text": str(variant_group_anchor_text or ""),
            "variant_group_merge_reason": str(variant_group_merge_reason or ""),
            "variant_group_selected_skeleton": str(variant_group_selected_skeleton or ""),
            "custom_onnx_raw_text": str(custom_onnx_raw_text or ""),
            "custom_onnx_korean_slot": str(custom_onnx_korean_slot or ""),
            "custom_onnx_digit_skeleton_used": str(custom_onnx_digit_skeleton_used or ""),
            "custom_onnx_slot_candidate": str(custom_onnx_slot_candidate or ""),
            "custom_onnx_slot_evidence_added": int(custom_onnx_slot_evidence_added or 0),
            "custom_onnx_slot_skip_reason": str(custom_onnx_slot_skip_reason or ""),
            "string_febam_state": str(string_febam_state or ""),
            "string_febam_activation": string_febam_activation,
            "string_febam_score": string_febam_score,
            "string_febam_energy": string_febam_energy,
            "string_febam_memory": string_febam_memory,
            "segment_id": segment_id,
            "segment_len": segment_len,
            "segment_effective_count": segment_effective_count,
            "segment_weight": segment_weight,
            "segment_text_medoid": str(segment_text_medoid or ""),
            "segment_consensus_text": str(segment_consensus_text or ""),
            "cluster_count": cluster_count,
            "contamination_flag": contamination_flag,
            "commit_frame": commit_frame,
            "first_stable_frame": first_stable_frame,
            "ocr_call_count": ocr_call_count,
            "ocr_skipped_after_commit": ocr_skipped_after_commit,
        }
        if getattr(self, "bio_adaptive_processor", None) is not None:
            row.update({key: extra_values.get(key, "") for key in BIO_CSV_FIELDS})
        self.ocr_csv_rows.append(row)

    def _sanitize_ocr_debug_csv_row_kwargs(self, row_kwargs: dict | None, extra_duplicate_keys=None) -> dict:
        debug_kwargs = dict(row_kwargs or {})
        duplicate_keys = {
            "frame_idx",
            "track_id",
            "candidate_idx",
            "ocr_attempt",
            "ocr_skip_reason",
            "text",
            "conf",
            "bbox",
            "roi_w",
            "roi_h",
            "candidate_count",
            "track_id_count",
            "febam_confirmed_count",
            "gray_stretched_ocr_used",
            "attempted_count",
            "ocr_variant_tier",
            "ocr_variant_executed",
            "ocr_variants_planned",
            "ocr_variants_executed_count",
            "ocr_readtext_fallback_count",
            "ocr_variant_skip_reason",
            "ocr_allowlist_mode",
            "ocr_normalized_text",
            "ocr_pattern_type",
            "ocr_postprocess_applied",
            "ocr_postprocess_reason",
            "ocr_plate_text_final",
            "ocr_korean_middle_raw",
            "ocr_korean_middle_fixed",
            "ocr_early_stop_reason",
            "ocr_engine",
            "easyocr_mode",
            "easyocr_batch_size",
            "easyocr_workers",
            "easyocr_recognize_only",
            "easyocr_readtext_fallback_used",
            "fastplate_model",
            "fastplate_device",
            "fastplate_batch_size",
            "async_delay_frames",
            "async_delay_ms",
            "ocr_variant_name",
            "ocr_raw_result_short",
            "normalized_plate_candidate",
            "source",
        }
        if extra_duplicate_keys:
            duplicate_keys.update(extra_duplicate_keys)
        for duplicate_key in duplicate_keys:
            debug_kwargs.pop(duplicate_key, None)
        return debug_kwargs

    def _sanitize_ocr_csv_row_kwargs(self, row_kwargs: dict | None) -> dict:
        csv_kwargs = dict(row_kwargs or {})
        for duplicate_key in {
            "track_id",
            "frame_idx",
            "candidate_idx",
            "ocr_attempt",
            "ocr_skip_reason",
            "ocr_text",
            "ocr_conf",
            "bbox",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "roi_w",
            "roi_h",
            "candidate_count",
            "track_id_count",
            "febam_confirmed_count",
            "gray_stretched_ocr_used",
            "is_final",
            "final_plate",
            "attempted_count",
        }:
            csv_kwargs.pop(duplicate_key, None)
        return csv_kwargs

    def _append_ocr_csv_row(
        self,
        track_id: int,
        frame_idx: int | None,
        text: str,
        conf: float,
        bbox,
        *,
        candidate_idx: int | None = None,
        roi_w: int | str = "",
        roi_h: int | str = "",
        candidate_count: int = 0,
        track_id_count: int = 0,
        febam_confirmed_count: int = 0,
        gray_stretched_ocr_used: bool = False,
        attempted_count: int = 0,
        febam_score: object = "",
        febam_energy: object = "",
        febam_threshold: object = "",
        febam_memory: object = "",
        febam_confirmed: int = 0,
        ocr_input_type: str = "",
        ocr_candidate_rank: str = "",
        ocr_crop_policy: str = "",
        ocr_crop_expand_top: int | str = "",
        ocr_y1_original: int | str = "",
        ocr_y1_expanded: int | str = "",
        roi_aspect: float | str = "",
        ocr_expand_top: int | str = "",
        ocr_angle: float | str = "",
        ocr_rotation_applied: int | str = "",
        ocr_angle_score: float | str = "",
        ocr_gpu_rectified: int | str = "",
        ocr_trigger_reason: str = "",
        near_confirmed_large_roi: int | str = "",
        febam_full_confirmed: int | str = "",
        near_confirmed_ocr_sample: int | str = "",
        ocr_sampling_reason: str = "",
        ocr_sampling_level: str = "",
        source_weight: float | str = "",
        pre_febam_candidate_count: int | str = "",
        post_febam_confirmed_count: int | str = "",
        febam_score_thr: float | str = "",
        febam_energy_thr: float | str = "",
        febam_memory_min: int | str = "",
        ocr_saved_raw: str = "",
        ocr_saved_gray_stretched: str = "",
        ocr_saved_rotated: str = "",
        ocr_input_variant: str = "",
        ocr_fallback_rank: int | str = "",
        ocr_fallback_used: int | str = "",
        ocr_rotation_threshold: float | str = "",
        ocr_reader_langs: str = "",
        ocr_allowlist_mode: str = "",
        ocr_normalized_text: str = "",
        ocr_pattern_type: str = "",
        ocr_postprocess_applied: int | str = "",
        ocr_postprocess_reason: str = "",
        ocr_plate_text_final: str = "",
        ocr_korean_middle_raw: str = "",
        ocr_korean_middle_fixed: str = "",
        ocr_small_roi: int | str = "",
        ocr_upscale_applied: int | str = "",
        ocr_upscale_factor: float | str = "",
        ocr_sharpen_applied: int | str = "",
        ocr_sharpen_amount: float | str = "",
        mrs: float | str = "",
        raw_text: str = "",
        raw_conf: float | str = "",
        corrected_text: str = "",
        corrected_conf: float | str = "",
        source: str = "",
        ocr_variant_order: str = "",
        ocr_variant_count: int | str = "",
        ocr_norm_w: int | str = "",
        ocr_norm_h: int | str = "",
        ocr_norm_scale: float | str = "",
        ocr_norm_pad_x: int | str = "",
        ocr_norm_pad_y: int | str = "",
        ocr_norm_aspect_preserved: int | str = "",
        ocr_allowlist_sweep_disabled: int | str = "",
        ocr_early_stop_reason: str = "",
        ocr_selected_final_candidate_only: int | str = "",
        plate_layout: str = "",
        grammar_pattern: str = "",
        grammar_candidates_top3: str = "",
        grammar_best_text: str = "",
        grammar_best_score: float | str = "",
        grammar_best_reason: str = "",
        grammar_valid_final: int | str = "",
        split_ocr_used: int | str = "",
        split_left_text: str = "",
        split_mid_text: str = "",
        split_right_text: str = "",
        split_top_text: str = "",
        split_bottom_text: str = "",
        committed_plate_locked: int | str = "",
        ocr_crop_expand_top_ratio: float | str = "",
        grammar_decoder_used: int | str = "",
        split_fallback_used: int | str = "",
        split_mode: str = "",
        final_candidate_source: str = "",
        final_candidate_score: float | str = "",
        final_candidate_reason: str = "",
        final_output_plate: str = "",
        ocr_engine: str = "",
        easyocr_mode: str = "",
        easyocr_batch_size: int | str = "",
        easyocr_workers: int | str = "",
        easyocr_recognize_only: int | str = "",
        easyocr_readtext_fallback_used: int | str = "",
        fastplate_model: str = "",
        fastplate_device: str = "",
        fastplate_batch_size: int | str = "",
        async_delay_frames: int | float | str = "",
        async_delay_ms: int | float | str = "",
        ocr_crop_expand_left_ratio: float | str = "",
        ocr_crop_expand_right_ratio: float | str = "",
        ocr_crop_expand_bottom_ratio: float | str = "",
        ocr_variant_name: str = "",
        ocr_raw_result_short: str = "",
        normalized_plate_candidate: str = "",
        korean_plate_valid: int | str = "",
        ocr_variant_tier: int | str = "",
        ocr_variant_executed: int | str = "",
        ocr_variants_planned: str = "",
        ocr_variants_executed_count: int | str = "",
        ocr_readtext_fallback_count: int | str = "",
        ocr_variant_skip_reason: str = "",
        skeleton_anchor_used: int | str = "",
        skeleton_anchor_pattern: str = "",
        skeleton_anchor_text: str = "",
        skeleton_anchor_support: int | str = "",
        variant_group_key: str = "",
        variant_group_anchor_text: str = "",
        variant_group_merge_reason: str = "",
        variant_group_selected_skeleton: str = "",
        custom_onnx_raw_text: str = "",
        custom_onnx_korean_slot: str = "",
        custom_onnx_digit_skeleton_used: str = "",
        custom_onnx_slot_candidate: str = "",
        custom_onnx_slot_evidence_added: int | str = "",
        custom_onnx_slot_skip_reason: str = "",
        **extra_values,
    ) -> None:
        self._append_ocr_debug_csv_row(
            frame_idx=frame_idx,
            track_id=track_id,
            candidate_idx=candidate_idx,
            ocr_attempt=1,
            ocr_skip_reason="ok",
            text=text,
            conf=conf,
            bbox=bbox,
            roi_w=roi_w,
            roi_h=roi_h,
            candidate_count=candidate_count,
            track_id_count=track_id_count,
            febam_confirmed_count=febam_confirmed_count,
            gray_stretched_ocr_used=gray_stretched_ocr_used,
            is_final=0,
            final_plate="",
            attempted_count=attempted_count,
            febam_score=febam_score,
            febam_energy=febam_energy,
            febam_threshold=febam_threshold,
            febam_memory=febam_memory,
            febam_confirmed=febam_confirmed,
            ocr_input_type=ocr_input_type,
            ocr_candidate_rank=ocr_candidate_rank,
            ocr_crop_policy=ocr_crop_policy,
            ocr_crop_expand_top=ocr_crop_expand_top,
            ocr_y1_original=ocr_y1_original,
            ocr_y1_expanded=ocr_y1_expanded,
            roi_aspect=roi_aspect,
            ocr_expand_top=ocr_expand_top,
            ocr_angle=ocr_angle,
            ocr_rotation_applied=ocr_rotation_applied,
            ocr_angle_score=ocr_angle_score,
            ocr_gpu_rectified=ocr_gpu_rectified,
            ocr_trigger_reason=ocr_trigger_reason,
            near_confirmed_large_roi=near_confirmed_large_roi,
            febam_full_confirmed=febam_full_confirmed,
            near_confirmed_ocr_sample=near_confirmed_ocr_sample,
            ocr_sampling_reason=ocr_sampling_reason,
            ocr_sampling_level=ocr_sampling_level,
            source_weight=source_weight,
            pre_febam_candidate_count=pre_febam_candidate_count,
            post_febam_confirmed_count=post_febam_confirmed_count,
            febam_score_thr=febam_score_thr,
            febam_energy_thr=febam_energy_thr,
            febam_memory_min=febam_memory_min,
            ocr_saved_raw=ocr_saved_raw,
            ocr_saved_gray_stretched=ocr_saved_gray_stretched,
            ocr_saved_rotated=ocr_saved_rotated,
            ocr_input_variant=ocr_input_variant,
            ocr_fallback_rank=ocr_fallback_rank,
            ocr_fallback_used=ocr_fallback_used,
            ocr_rotation_threshold=ocr_rotation_threshold,
            ocr_reader_langs=ocr_reader_langs,
            ocr_allowlist_mode=ocr_allowlist_mode,
            ocr_normalized_text=ocr_normalized_text,
            ocr_pattern_type=ocr_pattern_type,
            ocr_postprocess_applied=ocr_postprocess_applied,
            ocr_postprocess_reason=ocr_postprocess_reason,
            ocr_plate_text_final=ocr_plate_text_final,
            ocr_korean_middle_raw=ocr_korean_middle_raw,
            ocr_korean_middle_fixed=ocr_korean_middle_fixed,
            ocr_small_roi=ocr_small_roi,
            ocr_upscale_applied=ocr_upscale_applied,
            ocr_upscale_factor=ocr_upscale_factor,
            ocr_sharpen_applied=ocr_sharpen_applied,
            ocr_sharpen_amount=ocr_sharpen_amount,
            mrs=mrs,
            raw_text=raw_text or text,
            raw_conf=raw_conf if raw_conf != "" else conf,
            corrected_text=corrected_text or ocr_plate_text_final,
            corrected_conf=corrected_conf if corrected_conf != "" else conf,
            ocr_source=source,
            ocr_variant_order=ocr_variant_order,
            ocr_variant_count=ocr_variant_count,
            ocr_norm_w=ocr_norm_w,
            ocr_norm_h=ocr_norm_h,
            ocr_norm_scale=ocr_norm_scale,
            ocr_norm_pad_x=ocr_norm_pad_x,
            ocr_norm_pad_y=ocr_norm_pad_y,
            ocr_norm_aspect_preserved=ocr_norm_aspect_preserved,
            ocr_allowlist_sweep_disabled=ocr_allowlist_sweep_disabled,
            ocr_early_stop_reason=ocr_early_stop_reason,
            ocr_selected_final_candidate_only=ocr_selected_final_candidate_only,
            plate_layout=plate_layout,
            grammar_pattern=grammar_pattern,
            grammar_candidates_top3=grammar_candidates_top3,
            grammar_best_text=grammar_best_text,
            grammar_best_score=grammar_best_score,
            grammar_best_reason=grammar_best_reason,
            grammar_valid_final=grammar_valid_final,
            split_ocr_used=split_ocr_used,
            split_left_text=split_left_text,
            split_mid_text=split_mid_text,
            split_right_text=split_right_text,
            split_top_text=split_top_text,
            split_bottom_text=split_bottom_text,
            committed_plate_locked=committed_plate_locked,
            ocr_crop_expand_top_ratio=ocr_crop_expand_top_ratio,
            grammar_decoder_used=grammar_decoder_used,
            split_fallback_used=split_fallback_used,
            split_mode=split_mode,
            final_candidate_source=final_candidate_source,
            final_candidate_score=final_candidate_score,
            final_candidate_reason=final_candidate_reason,
            final_output_plate=final_output_plate,
            ocr_engine=ocr_engine,
            easyocr_mode=easyocr_mode,
            easyocr_batch_size=easyocr_batch_size,
            easyocr_workers=easyocr_workers,
            easyocr_recognize_only=easyocr_recognize_only,
            easyocr_readtext_fallback_used=easyocr_readtext_fallback_used,
            fastplate_model=fastplate_model,
            fastplate_device=fastplate_device,
            fastplate_batch_size=fastplate_batch_size,
            async_delay_frames=async_delay_frames,
            async_delay_ms=async_delay_ms,
            ocr_crop_expand_left_ratio=ocr_crop_expand_left_ratio,
            ocr_crop_expand_right_ratio=ocr_crop_expand_right_ratio,
            ocr_crop_expand_bottom_ratio=ocr_crop_expand_bottom_ratio,
            ocr_variant_name=ocr_variant_name,
            ocr_raw_result_short=ocr_raw_result_short,
            normalized_plate_candidate=normalized_plate_candidate,
            korean_plate_valid=korean_plate_valid,
            ocr_variant_tier=ocr_variant_tier,
            ocr_variant_executed=ocr_variant_executed,
            ocr_variants_planned=ocr_variants_planned,
            ocr_variants_executed_count=ocr_variants_executed_count,
            ocr_readtext_fallback_count=ocr_readtext_fallback_count,
            ocr_variant_skip_reason=ocr_variant_skip_reason,
            skeleton_anchor_used=skeleton_anchor_used,
            skeleton_anchor_pattern=skeleton_anchor_pattern,
            skeleton_anchor_text=skeleton_anchor_text,
            skeleton_anchor_support=skeleton_anchor_support,
            variant_group_key=variant_group_key,
            variant_group_anchor_text=variant_group_anchor_text,
            variant_group_merge_reason=variant_group_merge_reason,
            variant_group_selected_skeleton=variant_group_selected_skeleton,
            custom_onnx_raw_text=custom_onnx_raw_text,
            custom_onnx_korean_slot=custom_onnx_korean_slot,
            custom_onnx_digit_skeleton_used=custom_onnx_digit_skeleton_used,
            custom_onnx_slot_candidate=custom_onnx_slot_candidate,
            custom_onnx_slot_evidence_added=custom_onnx_slot_evidence_added,
            custom_onnx_slot_skip_reason=custom_onnx_slot_skip_reason,
            **extra_values,
        )

    def _append_final_ocr_csv_row(
        self,
        track_id: int,
        frame_idx: int | None,
        reason: str = "final_voted",
        stats: dict[str, object] | None = None,
    ) -> None:
        if stats is None:
            stats = self._ocr_history_stats(self.ocr_history.get(track_id, []))
        voted_plate = str(stats.get("final_voted_text", "") or "")
        string_values = self._string_febam_csv_values(track_id) if self.use_string_febam else {}
        string_state = str(string_values.get("string_febam_state", "")) if string_values else ""
        final_observed_plate = str(string_values.get("final_observed_plate", "")) if string_values else ""
        final_consensus_plate = str(string_values.get("final_consensus_plate", "")) if string_values else ""
        committed_plate = self._committed_plate_for_track(track_id)
        grammar_consensus = final_consensus_plate if self._valid_final_plate_candidate(final_consensus_plate) else ""
        fallback_whole = str(self.final_plates.get(track_id, ""))
        final_plate, priority_source, committed_locked = choose_final_plate(
            committed_plate=committed_plate,
            grammar_decoded_consensus=grammar_consensus,
            split_ocr_decoded_plate="",
            whole_ocr_decoded_plate=fallback_whole,
            voted_text=voted_plate,
        )
        final_plate_source = "string_febam_commit" if committed_locked else priority_source
        if not final_plate:
            return
        self._append_ocr_debug_csv_row(
            frame_idx=frame_idx,
            track_id=track_id,
            candidate_idx=-1,
            ocr_attempt=0,
            ocr_skip_reason=reason,
            text="",
            conf="",
            bbox=None,
            roi_w="",
            roi_h="",
            candidate_count=0,
            track_id_count=0,
            febam_confirmed_count=0,
            gray_stretched_ocr_used=False,
            is_final=1,
            final_plate=final_plate,
            attempted_count=0,
            best_text=str(stats.get("best_text", "")),
            best_conf=stats.get("best_conf", ""),
            voted_text=voted_plate,
            voted_conf=stats.get("best_conf", ""),
            final_voted_text=voted_plate,
            final_vote_count=stats.get("final_vote_count", ""),
            final_observed_plate=final_observed_plate,
            final_consensus_plate=final_consensus_plate,
            final_plate_source=final_plate_source,
            committed_plate_locked=int(committed_locked),
            grammar_best_text=final_plate if priority_source.startswith("grammar") else "",
            grammar_valid_final=int(self._valid_final_plate_candidate(final_plate)),
            final_candidate_source=final_plate_source,
            final_candidate_score="",
            final_candidate_reason=priority_source,
            final_output_plate=final_plate,
            mrs_max=stats.get("mrs_max", ""),
            mrs_mean=stats.get("mrs_mean", ""),
            ocr_count=stats.get("ocr_count", ""),
            topk_ocr=str(stats.get("topk_ocr", "")),
        )

    def _log_ocr_debug(
        self,
        *,
        frame_idx: int | None,
        candidate_count: int,
        track_id_count: int,
        febam_confirmed_count: int,
        attempted_count: int,
        skip_reason: str,
        gate,
        force: bool = False,
    ) -> None:
        idx = int(frame_idx) if frame_idx is not None else -1
        if not force and idx >= 0 and idx % 30 != 0:
            return
        try:
            gate_text = tuple(int(v) for v in gate)
        except Exception:
            gate_text = gate
        self._log(
            "[OCR_DEBUG] "
            f"frame={idx} candidates={int(candidate_count)} track_ids={int(track_id_count)} "
            f"confirmed={int(febam_confirmed_count)} attempt={int(attempted_count)} "
            f"skip={skip_reason} gate={gate_text}"
        )

    def _safe_float(self, value: object, default: float = 0.0) -> float:
        try:
            if value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _mrs_from_febam_debug(self, febam_debug: dict[str, object]) -> float:
        # MRS는 기존 Sigmoid FEBAM의 reliability/sigmoid output을 Decision Stabilization에 재사용한다.
        # 기존 FEBAM sigmoid, threshold, memory, competition, OCR trigger gate 로직은 여기서 재작성하지 않는다.
        mrs = self._safe_float(febam_debug.get("febam_score", ""), 0.0)
        return max(0.0, min(1.0, mrs))


    def _fastplate_candidate_source_weight(self, reason: str, skeleton_anchor_used: int | bool = 0) -> float:
        reason_text = str(reason or "")
        if reason_text.startswith("slot_confusion:"):
            match = re.match(r"slot_confusion:([^->]+)->", reason_text)
            raw_middle = match.group(1) if match else ""
            if raw_middle.isdigit() and not bool(skeleton_anchor_used):
                return 0.35
            return 0.70
        if "variant_anchor_missing_middle" in reason_text or "track_anchor_missing_middle" in reason_text:
            return 0.60
        if "missing_middle" in reason_text:
            return 0.60 if bool(skeleton_anchor_used) else 0.35
        if reason_text.startswith("fastplate_direct_korean_plate"):
            return 0.70
        return 0.50

    def _ocr_sampling_source_weight(self, sampling_level: str) -> float:
        if hasattr(self, "string_febam_engine"):
            return self.string_febam_engine._source_weight_from_sampling_level(sampling_level)
        if sampling_level == "febam_confirmed":
            return 1.0
        if sampling_level == "near_confirmed":
            return 0.7
        if sampling_level == "weak_plate_sample":
            return 0.35
        return 0.5

    def _sampling_level_from_reason(self, reason: str) -> str:
        if reason in {"febam_confirmed", "febam_confirmed_ocr", "febam_last_valid_bbox"}:
            return "febam_confirmed"
        if reason in {"near_confirmed_ocr_sample", "near_confirmed_large_roi"}:
            return "near_confirmed"
        if reason in {"weak_plate_ocr_sample", "periodic_track_sample", "mandatory_frame_ocr_slot"}:
            return "weak_plate_sample"
        return ""

    def _aspect_norm_ocr_crop_cuda(self, crop_u8: torch.Tensor, target_h: int = 48) -> torch.Tensor:
        if not torch.is_tensor(crop_u8) or crop_u8.ndim != 3:
            return crop_u8
        h, w = crop_u8.shape[-2:]
        if h <= 0 or w <= 0:
            return crop_u8
        target_w = int(max(96, min(320, round(float(w) * float(target_h) / max(float(h), 1.0)))))
        crop01 = crop_u8.to(device=self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0).clamp(0, 255) / 255.0
        resized = F.interpolate(crop01, size=(int(target_h), target_w), mode="bilinear", align_corners=False)
        return (resized[0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    def _weak_plate_position_prior(self, bbox: torch.Tensor, roi_w: int, roi_h: int) -> bool:
        try:
            x1, y1, x2, y2 = [float(v.item()) for v in bbox.detach().float().reshape(-1)[:4]]
            cx = (x1 + x2) * 0.5 / max(float(roi_w), 1.0)
            cy = (y1 + y2) * 0.5 / max(float(roi_h), 1.0)
            return 0.05 <= cx <= 0.95 and 0.25 <= cy <= 0.95
        except Exception:
            return False

    def _bbox_similar_to_last_valid(self, track_id: int, bbox: torch.Tensor) -> bool:
        last = self.last_valid_track_bboxes.get(int(track_id))
        if last is None or not torch.is_tensor(bbox):
            return False
        last_bbox, _ = last
        try:
            a = bbox.detach().float().reshape(-1)[:4]
            b = last_bbox.to(device=a.device).detach().float().reshape(-1)[:4]
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = torch.maximum(ax1, bx1)
            iy1 = torch.maximum(ay1, by1)
            ix2 = torch.minimum(ax2, bx2)
            iy2 = torch.minimum(ay2, by2)
            inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
            area_a = (ax2 - ax1).clamp_min(1) * (ay2 - ay1).clamp_min(1)
            area_b = (bx2 - bx1).clamp_min(1) * (by2 - by1).clamp_min(1)
            iou = float((inter / (area_a + area_b - inter + 1e-6)).detach().item())
            acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
            bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
            diag = float(torch.sqrt(area_b).detach().item()) + 1e-6
            center_dist = float(torch.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2).detach().item())
            return iou >= 0.25 or center_dist / diag <= 0.75
        except Exception:
            return False

    def _sigmoid_scalar(self, value: float) -> float:
        value = max(-60.0, min(60.0, float(value)))
        return 1.0 / (1.0 + float(np.exp(-value)))

    def _string_confusion_cost(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        strong_pairs = {
            ("O", "0"), ("Q", "0"), ("D", "0"), ("I", "1"), ("L", "1"), ("T", "1"),
            ("Z", "2"), ("B", "3"), ("E", "3"), ("A", "4"), ("S", "5"), ("G", "6"),
            ("C", "6"), ("T", "7"), ("Y", "7"), ("B", "8"), ("G", "9"), ("Q", "9"),
            ("구", "7"), ("구", "G"), ("구", "9"), ("누", "7"), ("누", "L"), ("누", "1"),
            ("고", "7"), ("고", "G"), ("고", "9"), ("자", "7"), ("자", "Z"),
            ("모", "0"), ("모", "O"), ("모", "5"), ("무", "0"), ("무", "O"),
            ("오", "0"), ("오", "O"), ("도", "0"), ("도", "O"), ("도", "D"),
            ("너", "L"), ("너", "1"), ("너", "4"), ("너", "9"), ("바", "B"), ("바", "8"), ("사", "S"), ("사", "5"),
            ("하", "H"), ("허", "H"), ("호", "H"), ("호", "0"), ("호", "O"),
        }
        weak_pairs = {
            ("구", "누"), ("구", "고"), ("모", "무"), ("모", "오"), ("도", "오"),
            ("도", "모"), ("나", "너"), ("하", "허"), ("허", "호"),
        }
        pair = (a, b)
        reverse = (b, a)
        if pair in strong_pairs or reverse in strong_pairs:
            return 0.25
        if pair in weak_pairs or reverse in weak_pairs:
            return 0.45
        return 1.0

    def _confusion_aware_similarity(self, a: str, b: str) -> float:
        a = self._normalize_ocr_text(a)
        b = self._normalize_ocr_text(b)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        rows = len(a) + 1
        cols = len(b) + 1
        dp = [[0.0] * cols for _ in range(rows)]
        for i in range(1, rows):
            dp[i][0] = float(i)
        for j in range(1, cols):
            dp[0][j] = float(j)
        for i in range(1, rows):
            for j in range(1, cols):
                sub_cost = self._string_confusion_cost(a[i - 1], b[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + sub_cost,
                )
        max_len = max(len(a), len(b), 1)
        return max(0.0, min(1.0, 1.0 - dp[-1][-1] / float(max_len)))

    def _is_korean_plate_candidate(self, text: str) -> bool:
        normalized = self._normalize_ocr_text(text)
        if len(normalized) not in (7, 8):
            return False
        digit_count = len(re.findall(r"\d", normalized))
        korean_or_english_count = len(re.findall(r"[A-Z가-힣]", normalized))
        return digit_count >= 5 and korean_or_english_count <= 2

    def _valid_final_plate_candidate(self, text: str, source: str = "") -> bool:
        return is_valid_final_plate(text)

    def _strict_korean_commit_candidate(self, text: str, source: str = "") -> bool:
        return is_valid_final_plate(text)

    def _valid_string_commit_candidate(self, text: str, source: str = "") -> bool:
        cfg = getattr(self, "string_febam_config", None)
        if bool(getattr(cfg, "strict_korean_commit", True)):
            return self._strict_korean_commit_candidate(text, source)
        return self._valid_final_plate_candidate(text, source)

    def _string_similarity_evidence(self, sim: float) -> float:
        if hasattr(self, "string_febam_engine"):
            return self.string_febam_engine._string_similarity_evidence(sim)
        return self._sigmoid_scalar(10.0 * (float(sim) - 0.65))

    def _string_length_score(self, text: str) -> float:
        if hasattr(self, "string_febam_engine"):
            return self.string_febam_engine._string_length_score(text)
        return self._sigmoid_scalar(2.0 * (len(self._normalize_ocr_text(text)) - 6.0))

    def _string_format_score(self, text: str) -> float:
        normalized = self._normalize_ocr_text(text)
        if re.fullmatch(r"\d{2,3}[가-힣]\d{4}", normalized):
            return 1.0
        if self._is_korean_plate_candidate(normalized):
            return 0.78
        if re.fullmatch(r"[A-Z0-9가-힣]{7,8}", normalized):
            return 0.62
        if re.fullmatch(r"[A-Z0-9가-힣]{4,10}", normalized):
            return 0.45
        return 0.25

    def _resolve_string_febam_group_id(
        self,
        meta: dict[str, object] | None,
        track_id: int,
    ) -> tuple[int, str]:
        values = dict(meta or {})
        if str(getattr(self, "string_febam_group_key_mode", "track")) == "track":
            return int(track_id), f"track:{int(track_id)}"
        raw_key = (
            values.get("event_fusion_group_key")
            or values.get("pseudo_vehicle_id")
            or values.get("group_id")
            or values.get("motion_group_id")
            or values.get("track_group_id")
            or values.get("event_group_key")
        )
        if raw_key in {"", None} and values.get("event_id") not in {"", None}:
            raw_key = f"{values.get('event_id')}:{values.get('segment_id', '')}:{track_id}"
        if raw_key in {"", None}:
            return int(track_id), f"track:{int(track_id)}"
        semantic_key = str(raw_key)
        try:
            runtime_id = int(semantic_key)
        except (TypeError, ValueError):
            runtime_id = self.string_febam_group_runtime_ids.get(semantic_key, -1)
            if runtime_id < 0:
                runtime_id = int(self._next_string_febam_group_runtime_id)
                self._next_string_febam_group_runtime_id += 1
                self.string_febam_group_runtime_ids[semantic_key] = runtime_id
                self.string_febam_runtime_group_keys[runtime_id] = semantic_key
        return runtime_id, semantic_key

    def _gpu_final_vehicle_group_id(
        self, meta: dict[str, object], *, frame_idx: int, track_id: int,
        bbox: object, candidate: str,
    ) -> tuple[int | None, str]:
        """Resolve image-free CUDA vehicle identity, otherwise HOLD."""
        video_id = str(meta.get("video_id", "") or self.runtime_video_id or "").strip()
        event_id = str(meta.get("event_id", "") or "").strip()
        segment_id = str(meta.get("segment_id", "") or "").strip()
        if not video_id or not event_id or not segment_id:
            meta["gpu_final_vehicle_hold_reason"] = "missing_canonical_event_identity"
            return None, ""
        try:
            bbox_cuda = torch.as_tensor(bbox, device="cuda", dtype=torch.float32).reshape(4)
        except (TypeError, ValueError, RuntimeError) as exc:
            meta["gpu_final_vehicle_hold_reason"] = f"invalid_cuda_bbox:{type(exc).__name__}"
            return None, ""
        instance_id, state = self.gpu_final_vehicle_grouper.route(
            video_id=video_id, event_id=event_id, segment_id=segment_id,
            frame_idx=int(frame_idx), bbox=bbox_cuda,
            tracker_id=str(track_id) if track_id >= 0 else "", text=str(candidate or ""),
        )
        meta["group_association_status"] = state
        meta["canonical_group_source"] = "gpu_final_vehicle_grouping"
        if not instance_id:
            meta["gpu_final_vehicle_hold_reason"] = state
            return None, ""
        base_key = f"{video_id}:{event_id}:{segment_id}"
        semantic_key = f"{base_key}:{instance_id}"
        meta.update(video_id=video_id, event_fusion_group_key=base_key,
                    vehicle_instance_id=instance_id, vehicle_febam_group_key=semantic_key)
        runtime_id = self.string_febam_group_runtime_ids.get(semantic_key, -1)
        if runtime_id < 0:
            runtime_id = int(self._next_string_febam_group_runtime_id)
            self._next_string_febam_group_runtime_id += 1
            self.string_febam_group_runtime_ids[semantic_key] = runtime_id
            self.string_febam_runtime_group_keys[runtime_id] = semantic_key
        return runtime_id, semantic_key

    def _event_fusion_string_group_id(
        self, meta: dict[str, object] | None
    ) -> int | None:
        """Resolve temporal OCR memory only from the canonical event key.

        This path deliberately does not call the legacy track/group fallback
        resolver. Raw observations without a confirmed event association wait
        until Event-Level ROI Fusion provides the key.
        """

        from pipeline.event_group_identity import resolve_canonical_event_group
        identity = resolve_canonical_event_group(dict(meta or {}))
        if identity.key is None:
            if isinstance(meta, dict):
                meta["dual_branch_hold_skip_reason"] = (
                    "noncanonical_event_fusion_group_key"
                    if identity.reason == "noncanonical_group_source"
                    else "ambiguous_event_fusion_group_key"
                    if identity.reason == "nonunique_group_association" and identity.status in {"ambiguous", "unmatched"}
                    else "pending_event_fusion_group_key"
                )
            return None
        semantic_key = identity.key
        runtime_id = self.string_febam_group_runtime_ids.get(semantic_key, -1)
        if runtime_id < 0:
            runtime_id = int(self._next_string_febam_group_runtime_id)
            self._next_string_febam_group_runtime_id += 1
            self.string_febam_group_runtime_ids[semantic_key] = runtime_id
            self.string_febam_runtime_group_keys[runtime_id] = semantic_key
        return runtime_id

    def _vehicle_febam_string_group_id(
        self,
        meta: dict[str, object],
        *,
        frame_idx: int,
        bbox: object,
        candidate: str,
        track_id: int,
    ) -> int | None:
        """Route recovery evidence through the external vehicle hierarchy.

        Missing/ambiguous canonical identity is a HOLD.  There is deliberately
        no track-only fallback because that could mix two vehicles in one
        event.
        """
        video_id = str(meta.get("video_id", "") or "").strip()
        event_key = str(meta.get("event_fusion_group_key", "") or "").strip()
        if (
            not video_id
            or not event_key
            or str(meta.get("canonical_group_source", "") or "") != "trial020_fusion_bridge"
            or str(meta.get("group_association_status", "") or "") != "unique"
        ):
            meta["recovery_hold_reason"] = "pending_or_ambiguous_event_fusion_group"
            return None
        vehicle_group_id = self.vehicle_febam_router.resolve_vehicle_group_id(meta)
        if not vehicle_group_id:
            meta["recovery_hold_reason"] = "missing_vehicle_group_id"
            return None
        try:
            box = tuple(float(value) for value in bbox)
            vehicle_instance_id = self.vehicle_febam_router.route(
                video_id=video_id,
                event_fusion_group_key=event_key,
                vehicle_group_id=vehicle_group_id,
                frame_idx=int(frame_idx),
                bbox=box,
                candidate=str(candidate),
                tracker_id=str(track_id) if track_id >= 0 else "",
            )
        except (TypeError, ValueError):
            meta["recovery_hold_reason"] = "invalid_vehicle_routing_evidence"
            return None
        if not vehicle_instance_id:
            meta["recovery_hold_reason"] = "vehicle_route_ambiguous"
            return None
        semantic_key = f"{video_id}::{event_key}::{vehicle_instance_id}"
        runtime_id = self.string_febam_group_runtime_ids.get(semantic_key, -1)
        if runtime_id < 0:
            runtime_id = int(self._next_string_febam_group_runtime_id)
            self._next_string_febam_group_runtime_id += 1
            self.string_febam_group_runtime_ids[semantic_key] = runtime_id
            self.string_febam_runtime_group_keys[runtime_id] = semantic_key
        meta["vehicle_instance_id"] = vehicle_instance_id
        meta["vehicle_febam_group_key"] = semantic_key
        return runtime_id

    def _merge_string_febam_track_into_group(self, track_id: int, group_id: int, group_key: str) -> None:
        pair = (int(track_id), int(group_id))
        if track_id < 0 or group_id == track_id or pair in self.string_febam_group_migrations:
            return
        self.string_febam_group_migrations.add(pair)
        target = self.string_febam_observations.setdefault(int(group_id), [])
        existing = {
            (int(item.get("frame_idx", -1)), str(item.get("text", "")), str(item.get("source", "")))
            for item in target
        }
        for item in self.string_febam_observations.get(int(track_id), []):
            signature = (
                int(item.get("frame_idx", -1)),
                str(item.get("text", "")),
                str(item.get("source", "")),
            )
            if signature in existing:
                continue
            migrated = dict(item)
            migrated["origin_track_id"] = int(track_id)
            migrated["group_id"] = str(group_key)
            migrated["track_id"] = int(group_id)
            target.append(migrated)
            existing.add(signature)
        if len(target) > self.ocr_history_max:
            del target[:-self.ocr_history_max]
        if target:
            self._rebuild_string_febam(int(group_id))

    def _string_frame_weight(self, ocr_conf: float, febam_reliability: float, source_weight: float = 1.0) -> float:
        alpha = float(getattr(self, "string_febam_alpha", 0.30))
        reliability = max(0.0, min(1.0, float(febam_reliability)))
        return max(0.0, float(ocr_conf)) * max(0.0, min(1.0, float(source_weight))) * (alpha + (1.0 - alpha) * reliability)

    def _segment_effective_count(self, segment_len: int) -> float:
        cfg = getattr(self, "string_febam_config", None)
        slope = float(getattr(cfg, "segment_slope", 0.25))
        cap = float(getattr(cfg, "segment_cap", 1.75))
        return min(1.0 + slope * max(0, int(segment_len) - 1), cap)

    def _segment_medoid(self, observations: list[dict[str, object]]) -> str:
        if not observations:
            return ""
        best_text = str(observations[0].get("text", ""))
        best_cost = float("inf")
        for candidate in observations:
            candidate_text = str(candidate.get("text", ""))
            cost = 0.0
            for other in observations:
                weight = self._safe_float(other.get("frame_weight", 0.0), 0.0)
                cost += (1.0 - self._confusion_aware_similarity(candidate_text, str(other.get("text", "")))) * max(weight, 1e-3)
            if cost < best_cost:
                best_cost = cost
                best_text = candidate_text
        return best_text

    def _segment_consensus_text(self, observations: list[dict[str, object]]) -> str:
        if not observations:
            return ""
        max_len = max(len(str(item.get("text", ""))) for item in observations)
        chars: list[str] = []
        for pos in range(max_len):
            weights: dict[str, float] = {}
            for item in observations:
                text = str(item.get("text", ""))
                if pos >= len(text):
                    continue
                ch = text[pos]
                weights[ch] = weights.get(ch, 0.0) + self._safe_float(item.get("frame_weight", 0.0), 0.0)
            if weights:
                chars.append(max(weights.items(), key=lambda kv: kv[1])[0])
        return "".join(chars)

    def _build_string_segments(self, track_id: int) -> list[dict[str, object]]:
        observations = sorted(
            self.string_febam_observations.get(track_id, []),
            key=lambda item: int(item.get("frame_idx", -1)),
        )
        segments: list[dict[str, object]] = []
        current: list[dict[str, object]] = []
        for obs in observations:
            if current:
                prev = current[-1]
                gap = int(obs.get("frame_idx", -1)) - int(prev.get("frame_idx", -1))
                sim = self._confusion_aware_similarity(str(obs.get("text", "")), str(prev.get("text", "")))
                if gap > int(getattr(self, "string_febam_max_frame_gap", 5)) or sim < self.string_febam_sim_thr:
                    segments.append(self._make_string_segment(track_id, len(segments), current))
                    current = []
            current.append(obs)
        if current:
            segments.append(self._make_string_segment(track_id, len(segments), current))
        return segments

    def _make_string_segment(self, track_id: int, segment_id: int, observations: list[dict[str, object]]) -> dict[str, object]:
        medoid = self._segment_medoid(observations)
        consensus = self._segment_consensus_text(observations)
        frame_weights = [self._safe_float(item.get("frame_weight", 0.0), 0.0) for item in observations]
        confs = [self._safe_float(item.get("ocr_conf", 0.0), 0.0) for item in observations]
        reliabilities = [self._safe_float(item.get("febam_reliability", 0.0), 0.0) for item in observations]
        effective_count = self._segment_effective_count(len(observations))
        format_score = self._string_format_score(medoid)
        length_score = self._string_length_score(medoid)
        mean_weight = sum(frame_weights) / max(len(frame_weights), 1)
        segment_weight = mean_weight * effective_count * format_score * length_score
        if len(observations) <= 1:
            stability = 1.0
        else:
            sims = [self._confusion_aware_similarity(medoid, str(item.get("text", ""))) for item in observations]
            stability = sum(sims) / max(len(sims), 1)
        return {
            "segment_id": int(segment_id),
            "track_id": int(track_id),
            "start_frame": int(observations[0].get("frame_idx", -1)),
            "end_frame": int(observations[-1].get("frame_idx", -1)),
            "observations": list(observations),
            "segment_text_medoid": medoid,
            "segment_consensus_text": consensus,
            "segment_len": len(observations),
            "segment_effective_count": effective_count,
            "segment_weight": segment_weight,
            "segment_conf_mean": sum(confs) / max(len(confs), 1),
            "segment_mrs_mean": sum(reliabilities) / max(len(reliabilities), 1),
            "segment_format_score": format_score,
            "segment_length_score": length_score,
            "segment_stability": stability,
        }

    def _rebuild_string_febam(self, track_id: int) -> dict[str, object]:
        segments = self._build_string_segments(track_id)
        nodes: list[StringFEBAMNode] = []
        energy_decay = 0.90
        evidence_gain = 0.10
        memory_decay = 0.96
        for segment in segments:
            segment_text = str(segment.get("segment_text_medoid", ""))
            if not segment_text:
                continue
            best_idx = -1
            best_sim = 0.0
            for idx, node in enumerate(nodes):
                sim = self._confusion_aware_similarity(segment_text, node.text)
                if sim > best_sim:
                    best_idx = idx
                    best_sim = sim
            for node in nodes:
                node.energy *= energy_decay
                if best_idx < 0 or node is not nodes[best_idx]:
                    node.memory *= memory_decay
            if best_idx >= 0 and best_sim >= self.string_febam_node_merge_thr:
                node = nodes[best_idx]
                evidence = self._safe_float(segment.get("segment_weight", 0.0), 0.0) * self._string_similarity_evidence(best_sim)
                node.energy += evidence_gain * evidence
                node.memory = min(8.0, node.memory + self._safe_float(segment.get("segment_effective_count", 1.0), 1.0))
                node.support_segments += 1
                node.total_weight += self._safe_float(segment.get("segment_weight", 0.0), 0.0)
                node.last_frame = int(segment.get("end_frame", -1))
                node.format_score = max(node.format_score, self._safe_float(segment.get("segment_format_score", 0.0), 0.0))
            else:
                node = StringFEBAMNode(
                    text=segment_text,
                    energy=evidence_gain * self._safe_float(segment.get("segment_weight", 0.0), 0.0),
                    memory=self._safe_float(segment.get("segment_effective_count", 1.0), 1.0),
                    last_frame=int(segment.get("end_frame", -1)),
                    support_segments=1,
                    total_weight=self._safe_float(segment.get("segment_weight", 0.0), 0.0),
                    format_score=self._safe_float(segment.get("segment_format_score", 0.0), 0.0),
                    first_frame=int(segment.get("start_frame", -1)),
                )
                nodes.append(node)
            for node in nodes:
                soft_support = 0.0
                for other in nodes:
                    if node is other:
                        continue
                    sim = self._confusion_aware_similarity(node.text, other.text)
                    if sim >= self.string_febam_cluster_thr:
                        soft_support += 0.05 * self._string_similarity_evidence(sim) * other.total_weight
                memory_norm = min(node.memory / 4.0, 1.0)
                total_weight_norm = min((node.total_weight + soft_support) / 2.0, 1.0)
                stable_score = node.energy + 0.5 * memory_norm + 0.5 * total_weight_norm + 0.5 * node.format_score
                node.score = stable_score
        cluster_count = self._assign_string_clusters(nodes)
        contamination = self._detect_string_contamination(nodes)
        sorted_nodes = sorted(nodes, key=lambda n: (n.score, n.activation, n.total_weight), reverse=True)
        best = sorted_nodes[0] if sorted_nodes else None
        second = sorted_nodes[1] if len(sorted_nodes) > 1 else None
        margin = (best.score - second.score) if best is not None and second is not None else 1.0
        for node in nodes:
            other_cluster_penalty = 0.0
            for other in nodes:
                if other is node or other.cluster_id == node.cluster_id:
                    continue
                sim = self._confusion_aware_similarity(node.text, other.text)
                if sim < self.string_febam_cluster_thr:
                    other_cluster_penalty += 0.35 * min(other.score, 2.0)
            instability_penalty = max(0.0, self.string_febam_margin_thr - margin) if best is node else 0.10
            activation_raw = node.score - other_cluster_penalty - instability_penalty
            node.activation = self._sigmoid_scalar(activation_raw)
        sorted_nodes = sorted(nodes, key=lambda n: (n.score, n.activation, n.total_weight), reverse=True)
        best = sorted_nodes[0] if sorted_nodes else None
        observed_plate = best.text if best is not None else ""
        consensus_plate = self._best_consensus_for_observed(segments, observed_plate)
        state = "HOLD"
        if contamination:
            state = "CONTAMINATION"
        elif best is not None:
            has_support = best.support_segments >= int(getattr(self, "string_febam_min_support_segments", 1))
            has_frames = (best.last_frame - best.first_frame + 1) >= int(getattr(self, "string_febam_min_segment_frames", 3)) or best.memory >= 1.5
            best_source = getattr(best, "source", "") if best is not None else ""
            has_valid_plate = self._valid_string_commit_candidate(observed_plate, best_source)
            if best.activation >= self.string_febam_commit_thr and has_support and has_frames and margin >= self.string_febam_margin_thr and has_valid_plate:
                state = "COMMIT"
        result = {
            "state": state,
            "final_observed_plate": observed_plate,
            "final_consensus_plate": consensus_plate,
            "activation": best.activation if best is not None else 0.0,
            "score": best.score if best is not None else 0.0,
            "energy": best.energy if best is not None else 0.0,
            "memory": best.memory if best is not None else 0.0,
            "cluster_count": cluster_count,
            "contamination_flag": int(contamination),
            "valid_final_plate_candidate": int(self._valid_final_plate_candidate(observed_plate)),
            "margin": margin,
            "nodes": sorted_nodes,
            "segments": segments,
        }
        self.string_febam_nodes[track_id] = sorted_nodes
        self.string_febam_segments[track_id] = segments
        self.string_febam_states[track_id] = result
        if state == "COMMIT" and track_id not in self.string_febam_commit_frames:
            commit_frame = int(segments[-1].get("end_frame", -1)) if segments else -1
            self.string_febam_commit_frames[track_id] = commit_frame
            self._log_string_febam_debug(track_id, result, reason="COMMIT")
        return result

    def _assign_string_clusters(self, nodes: list[StringFEBAMNode]) -> int:
        cluster_id = 0
        for node in nodes:
            node.cluster_id = -1
        for node in nodes:
            if node.cluster_id >= 0:
                continue
            node.cluster_id = cluster_id
            changed = True
            while changed:
                changed = False
                for src in nodes:
                    if src.cluster_id != cluster_id:
                        continue
                    for dst in nodes:
                        if dst.cluster_id >= 0:
                            continue
                        if self._confusion_aware_similarity(src.text, dst.text) >= self.string_febam_cluster_thr:
                            dst.cluster_id = cluster_id
                            changed = True
            cluster_id += 1
        return cluster_id

    def _detect_string_contamination(self, nodes: list[StringFEBAMNode]) -> bool:
        clusters: dict[int, list[StringFEBAMNode]] = {}
        for node in nodes:
            clusters.setdefault(node.cluster_id, []).append(node)
        strong_clusters = []
        for members in clusters.values():
            best = max(members, key=lambda n: n.score)
            duration = max((n.last_frame - n.first_frame + 1) for n in members)
            if best.score >= 0.70 and (duration >= 3 or best.support_segments >= 1):
                strong_clusters.append(best)
        if len(strong_clusters) < 2:
            return False
        for i, left in enumerate(strong_clusters):
            for right in strong_clusters[i + 1:]:
                if self._confusion_aware_similarity(left.text, right.text) < self.string_febam_cluster_thr:
                    return True
        return False

    def _best_consensus_for_observed(self, segments: list[dict[str, object]], observed: str) -> str:
        if not segments:
            return ""
        matching = [seg for seg in segments if str(seg.get("segment_text_medoid", "")) == observed]
        if not matching:
            matching = segments
        best = max(matching, key=lambda seg: self._safe_float(seg.get("segment_weight", 0.0), 0.0))
        return str(best.get("segment_consensus_text", ""))

    def _string_febam_csv_values(self, track_id: int) -> dict[str, object]:
        state = self.string_febam_states.get(track_id, {})
        segments = state.get("segments", []) if isinstance(state, dict) else []
        last_segment = segments[-1] if segments else {}
        return {
            "final_observed_plate": state.get("final_observed_plate", "") if isinstance(state, dict) else "",
            "final_consensus_plate": state.get("final_consensus_plate", "") if isinstance(state, dict) else "",
            "string_febam_state": state.get("state", "") if isinstance(state, dict) else "",
            "string_febam_activation": state.get("activation", "") if isinstance(state, dict) else "",
            "string_febam_score": state.get("score", "") if isinstance(state, dict) else "",
            "string_febam_energy": state.get("energy", "") if isinstance(state, dict) else "",
            "string_febam_memory": state.get("memory", "") if isinstance(state, dict) else "",
            "segment_id": last_segment.get("segment_id", "") if isinstance(last_segment, dict) else "",
            "segment_len": last_segment.get("segment_len", "") if isinstance(last_segment, dict) else "",
            "segment_effective_count": last_segment.get("segment_effective_count", "") if isinstance(last_segment, dict) else "",
            "segment_weight": last_segment.get("segment_weight", "") if isinstance(last_segment, dict) else "",
            "segment_text_medoid": last_segment.get("segment_text_medoid", "") if isinstance(last_segment, dict) else "",
            "segment_consensus_text": last_segment.get("segment_consensus_text", "") if isinstance(last_segment, dict) else "",
            "cluster_count": state.get("cluster_count", "") if isinstance(state, dict) else "",
            "contamination_flag": state.get("contamination_flag", "") if isinstance(state, dict) else "",
            "commit_frame": self.string_febam_commit_frames.get(track_id, ""),
            "first_stable_frame": self.string_febam_commit_frames.get(track_id, ""),
            "ocr_call_count": self.string_febam_ocr_call_count.get(track_id, 0),
            "ocr_skipped_after_commit": self.string_febam_skipped_after_commit.get(track_id, 0),
        }

    def _log_string_febam_debug(self, track_id: int, state: dict[str, object], reason: str) -> None:
        if not self.string_febam_debug and reason not in {"COMMIT", "track_ended", "pipeline_close"}:
            return
        nodes = state.get("nodes", []) if isinstance(state, dict) else []
        segments = state.get("segments", []) if isinstance(state, dict) else []
        node_text = " | ".join(
            f"rank={idx + 1} text={node.text} score={node.score:.3f} activation={node.activation:.3f} "
            f"energy={node.energy:.3f} memory={node.memory:.3f} support={node.support_segments} cluster={node.cluster_id}"
            for idx, node in enumerate(nodes[:5])
        )
        segment_text = " | ".join(
            f"id={seg.get('segment_id')} frames={seg.get('start_frame')}-{seg.get('end_frame')} "
            f"len={seg.get('segment_len')} eff={self._safe_float(seg.get('segment_effective_count', 0.0)):.2f} "
            f"medoid={seg.get('segment_text_medoid')} weight={self._safe_float(seg.get('segment_weight', 0.0)):.3f}"
            for seg in segments[-5:]
        )
        cluster_texts: dict[int, list[str]] = {}
        for node in nodes:
            cluster_texts.setdefault(node.cluster_id, []).append(node.text)
        self._log(
            "[STRING_FEBAM] "
            f"reason={reason} track_id={int(track_id)} state={state.get('state', '')} "
            f"final_observed_plate={state.get('final_observed_plate', '')!r} "
            f"final_consensus_plate={state.get('final_consensus_plate', '')!r} "
            f"cluster_count={state.get('cluster_count', 0)} contamination_flag={state.get('contamination_flag', 0)} "
            f"nodes=[{node_text}] segments=[{segment_text}] clusters={cluster_texts}"
        )

    def _ocr_history_stats(self, history: list[dict[str, object]]) -> dict[str, object]:
        valid = [item for item in history if str(item.get("text", ""))]
        if not valid:
            return {
                "best_text": "",
                "best_conf": 0.0,
                "final_voted_text": "",
                "final_vote_count": 0,
                "mrs_max": 0.0,
                "mrs_mean": 0.0,
                "ocr_count": 0,
                "topk_ocr": "",
            }

        best = max(valid, key=lambda item: (self._safe_float(item.get("conf", 0.0)), self._safe_float(item.get("mrs", 0.0))))
        mrs_values = [self._safe_float(item.get("mrs", 0.0)) for item in valid]
        topk = sorted(
            valid,
            key=lambda item: (
                self._safe_float(item.get("mrs", 0.0)),
                self._safe_float(item.get("conf", 0.0)),
                int(item.get("frame_idx", -1)),
            ),
            reverse=True,
        )[: max(1, int(getattr(self, "ocr_topk", 3)))]
        final_text = self._vote_topk_ocr(topk)
        topk_text = ";".join(
            f"{item.get('text', '')}|conf={self._safe_float(item.get('conf', 0.0)):.3f}|mrs={self._safe_float(item.get('mrs', 0.0)):.3f}|frame={int(item.get('frame_idx', -1))}"
            for item in topk
        )
        return {
            "best_text": str(best.get("text", "")),
            "best_conf": self._safe_float(best.get("conf", 0.0)),
            "final_voted_text": final_text,
            "final_vote_count": len(topk),
            "mrs_max": max(mrs_values) if mrs_values else 0.0,
            "mrs_mean": (sum(mrs_values) / len(mrs_values)) if mrs_values else 0.0,
            "ocr_count": len(valid),
            "topk_ocr": topk_text,
        }

    def _vote_topk_ocr(self, topk: list[dict[str, object]]) -> str:
        if not topk:
            return ""
        max_len = max(len(str(item.get("text", ""))) for item in topk)
        voted_chars: list[str] = []
        for pos in range(max_len):
            char_votes: dict[str, dict[str, float]] = {}
            for item in topk:
                text = str(item.get("text", ""))
                if pos >= len(text):
                    continue
                ch = text[pos]
                conf = self._safe_float(item.get("conf", 0.0))
                mrs = self._safe_float(item.get("mrs", 0.0))
                stats = char_votes.setdefault(ch, {"count": 0.0, "conf": 0.0, "mrs": 0.0})
                stats["count"] += 1.0
                stats["conf"] += conf
                stats["mrs"] += mrs
            if not char_votes:
                continue
            voted_chars.append(
                max(
                    char_votes.items(),
                    key=lambda kv: (kv[1]["count"], kv[1]["conf"], kv[1]["mrs"]),
                )[0]
            )
        return "".join(voted_chars)

    def _vote_ocr_history(self, track_id: int) -> str:
        history = self.ocr_history.get(track_id, [])
        voted = str(self._ocr_history_stats(history).get("final_voted_text", ""))
        return voted if self._valid_final_plate_candidate(voted) else ""

    def _committed_plate_for_track(self, track_id: int) -> str:
        state = self.string_febam_states.get(track_id, {}) if hasattr(self, "string_febam_states") else {}
        if isinstance(state, dict) and state.get("state") == "COMMIT":
            for key in ("committed_text", "final_observed_plate", "final_consensus_plate"):
                value = str(state.get(key, "") or "")
                if self._valid_final_plate_candidate(value):
                    return self._normalize_ocr_text(value)
        if hasattr(self, "string_febam_engine"):
            value = str(getattr(self.string_febam_engine, "committed_text", {}).get(track_id, "") or "")
            if self._valid_final_plate_candidate(value):
                return self._normalize_ocr_text(value)
        return ""

    def _reset_string_febam_epoch(self, track_id: int, reason: str) -> None:
        self.ocr_history.pop(track_id, None)
        self.final_plates.pop(track_id, None)
        self.string_febam_observations.pop(track_id, None)
        self.string_febam_quarantine.pop(track_id, None)
        self.string_febam_nodes.pop(track_id, None)
        self.string_febam_segments.pop(track_id, None)
        self.string_febam_states.pop(track_id, None)
        self.string_febam_commit_frames.pop(track_id, None)
        if self.string_febam_debug:
            self._log(f"[STRING_FEBAM_EPOCH_RESET] track_id={int(track_id)} reason={reason}")

    def _maybe_reset_string_epoch_for_observation(self, track_id: int, frame_idx: int, text: str) -> None:
        previous_frames = [
            int(item.get("frame_idx", -1))
            for item in self.string_febam_observations.get(track_id, [])
            if int(item.get("frame_idx", -1)) >= 0
        ]
        previous_frames.extend(
            int(item.get("frame_idx", -1))
            for item in self.string_febam_quarantine.get(track_id, [])
            if int(item.get("frame_idx", -1)) >= 0
        )
        if not previous_frames or frame_idx < 0:
            return
        last_frame = max(previous_frames)
        frame_gap = int(frame_idx) - int(last_frame)
        if frame_gap <= 0:
            return

        hard_gap = int(getattr(self, "string_febam_epoch_frame_gap", 40))
        if frame_gap > hard_gap:
            self._reset_string_febam_epoch(track_id, f"frame_gap_{frame_gap}_gt_{hard_gap}")
            return

        state = self.string_febam_states.get(track_id, {})
        if not isinstance(state, dict) or state.get("state") != "COMMIT":
            return
        final_observed = str(state.get("final_observed_plate", ""))
        commit_gap = int(getattr(self, "string_febam_commit_epoch_gap", 20))
        dissim_thr = float(getattr(self, "string_febam_epoch_dissim_thr", 0.35))
        if final_observed and frame_gap > commit_gap:
            similarity = self._confusion_aware_similarity(final_observed, text)
            if similarity < dissim_thr:
                self._reset_string_febam_epoch(
                    track_id,
                    f"post_commit_gap_{frame_gap}_sim_{similarity:.3f}_lt_{dissim_thr:.3f}",
                )

    def _is_short_string_quarantine_candidate(self, text: str) -> bool:
        normalized = self._normalize_ocr_text(text)
        if len(normalized) >= 4:
            return False
        if self._is_korean_plate_candidate(normalized):
            return False
        if re.fullmatch(r"\d{2,3}[가-힣]\d{4}", normalized):
            return False
        if re.fullmatch(r"[A-Z0-9가-힣]{7,8}", normalized):
            return False
        return True

    def _append_or_promote_string_observation(self, track_id: int, observation: dict[str, object]) -> dict[str, object]:
        text = str(observation.get("text", ""))
        if self._is_short_string_quarantine_candidate(text):
            observation["quarantine"] = 1
            quarantine = self.string_febam_quarantine.setdefault(track_id, [])
            quarantine.append(observation)
            if len(quarantine) > self.ocr_history_max:
                del quarantine[:-self.ocr_history_max]
            max_gap = int(getattr(self, "string_febam_max_frame_gap", 5))
            frame_value = int(observation.get("frame_idx", -1))
            repeated = [
                item for item in quarantine
                if abs(frame_value - int(item.get("frame_idx", -1))) <= max_gap
                and self._confusion_aware_similarity(text, str(item.get("text", ""))) >= self.string_febam_node_merge_thr
            ]
            connected_to_node = any(
                self._confusion_aware_similarity(text, node.text) >= self.string_febam_node_merge_thr
                for node in self.string_febam_nodes.get(track_id, [])
            )
            if len(repeated) < 2 and not connected_to_node:
                if self.string_febam_debug:
                    self._log(
                        f"[STRING_FEBAM_QUARANTINE] track_id={int(track_id)} frame={frame_value} "
                        f"text={text!r} quarantine_count={len(quarantine)}"
                    )
                return self.string_febam_states.get(track_id, {})

            promote_ids = {id(item) for item in repeated} if repeated else {id(observation)}
            promoted = [item for item in quarantine if id(item) in promote_ids]
            self.string_febam_quarantine[track_id] = [item for item in quarantine if id(item) not in promote_ids]
            buffer = self.string_febam_observations.setdefault(track_id, [])
            for item in promoted:
                item["quarantine"] = 0
                item["source"] = f"{item.get('source', 'raw')}_quarantine_promoted"
                buffer.append(item)
            if len(buffer) > self.ocr_history_max:
                del buffer[:-self.ocr_history_max]
            return self._rebuild_string_febam(track_id)

        observation["quarantine"] = 0
        buffer = self.string_febam_observations.setdefault(track_id, [])
        buffer.append(observation)
        if len(buffer) > self.ocr_history_max:
            del buffer[:-self.ocr_history_max]
        return self._rebuild_string_febam(track_id)

    def _append_string_observation(
        self,
        track_id: int,
        *,
        frame_idx: int | None,
        raw_text: str,
        corrected_text: str,
        text: str,
        ocr_conf: float,
        febam_reliability: float,
        source: str,
        source_weight: float = 1.0,
    ) -> dict[str, object]:
        if not self.use_string_febam:
            return {}
        text = self._normalize_ocr_text(text)
        raw_text = self._normalize_ocr_text(raw_text)
        corrected_text = self._normalize_ocr_text(corrected_text)
        if not text:
            return {}
        frame_value = int(frame_idx) if frame_idx is not None else -1
        self._maybe_reset_string_epoch_for_observation(track_id, frame_value, corrected_text or text or raw_text)
        self.string_febam_ocr_call_count[track_id] = self.string_febam_ocr_call_count.get(track_id, 0) + 1
        observation = {
            "frame_idx": frame_value,
            "track_id": int(track_id),
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "text": corrected_text or text or raw_text,
            "ocr_conf": float(ocr_conf),
            "febam_reliability": float(max(0.0, min(1.0, febam_reliability))),
            "source_weight": float(max(0.0, min(1.0, source_weight))),
            "frame_weight": self._string_frame_weight(float(ocr_conf), float(febam_reliability), float(source_weight)),
            "is_korean_plate_candidate": int(self._is_korean_plate_candidate(corrected_text or text or raw_text)),
            "source": str(source or "raw"),
        }
        if hasattr(self, "string_febam_engine"):
            core_obs = StringFEBAMObservation(
                frame_idx=frame_value,
                track_id=int(track_id),
                raw_text=raw_text,
                corrected_text=corrected_text,
                text=corrected_text or text or raw_text,
                ocr_conf=float(ocr_conf),
                mrs=float(febam_reliability),
                source_weight=float(source_weight),
                source=str(source or "raw"),
                sampling_level=str(source or ""),
            )
            self.string_febam_engine.append(core_obs)
        return self._append_or_promote_string_observation(track_id, observation)


    def configure_middle_slot_v32_from_args(self, args) -> None:
        self.middle_slot_v32_enabled = bool(getattr(args, "middle_slot_v32", getattr(self, "middle_slot_v32_enabled", False)))
        self.middle_slot_v32_model = str(getattr(args, "middle_slot_v32_model", getattr(self, "middle_slot_v32_model", "")) or self.middle_slot_v32_model)
        self.middle_slot_v32_device = str(getattr(args, "middle_slot_v32_device", getattr(self, "middle_slot_v32_device", "cuda")) or "cuda")
        self.middle_slot_v32_batch_size = int(max(1, getattr(args, "middle_slot_v32_batch_size", getattr(self, "middle_slot_v32_batch_size", 512))))
        self.middle_slot_v32_topk = int(max(1, getattr(args, "middle_slot_v32_topk", getattr(self, "middle_slot_v32_topk", 3))))
        self.middle_slot_v32_source_weight = float(max(0.0, min(0.05, getattr(args, "middle_slot_v32_source_weight", getattr(self, "middle_slot_v32_source_weight", 0.04)))))
        self.middle_slot_v32_min_conf = float(max(0.0, getattr(args, "middle_slot_v32_min_conf", getattr(self, "middle_slot_v32_min_conf", 0.20))))
        self.middle_slot_v32_min_margin = float(max(0.0, getattr(args, "middle_slot_v32_min_margin", getattr(self, "middle_slot_v32_min_margin", 0.03))))
        self.middle_slot_v32_max_crops_per_event = int(max(1, getattr(args, "middle_slot_v32_max_crops_per_event", getattr(self, "middle_slot_v32_max_crops_per_event", 8))))
        self.middle_slot_v32_evidence_mode = str(getattr(args, "middle_slot_v32_evidence_mode", getattr(self, "middle_slot_v32_evidence_mode", "topk_soft")) or "topk_soft")
        self.middle_slot_v32_profile = bool(getattr(args, "middle_slot_v32_profile", getattr(self, "middle_slot_v32_profile", False)))

    def configure_event_roi_fusion_mode_from_args(self, args) -> None:
        self.event_roi_fusion_mode = str(getattr(args, "event_roi_fusion_mode", getattr(self, "event_roi_fusion_mode", "ours_all_roi_febam")) or "ours_all_roi_febam")
        self.is_b5_yolo_high_quality_fusion_mode = self.event_roi_fusion_mode == "b5_yolo_high_quality_fusion"
        self.b5_fusion_top_k = int(max(1, getattr(args, "b5_fusion_top_k", getattr(self, "b5_fusion_top_k", 8))))
        self.b5_fusion_min_quality_score = float(getattr(args, "b5_fusion_min_quality_score", getattr(self, "b5_fusion_min_quality_score", 0.0)))
        self.b5_fusion_min_crops = int(max(2, getattr(args, "b5_fusion_min_crops", getattr(self, "b5_fusion_min_crops", 2))))
        self.b5_fusion_output_tag = str(getattr(args, "b5_fusion_output_tag", getattr(self, "b5_fusion_output_tag", "b5_yolo_high_quality_fusion")) or "b5_yolo_high_quality_fusion")
        if hasattr(args, "event_roi_fusion_debug_dir"):
            self.event_roi_fusion_debug_dir = str(getattr(args, "event_roi_fusion_debug_dir"))
        if hasattr(args, "event_roi_fusion_save_debug"):
            self.event_roi_fusion_save_debug = bool(getattr(args, "event_roi_fusion_save_debug"))
        if hasattr(args, "event_roi_fusion_save_manifest"):
            self.event_roi_fusion_save_manifest = bool(getattr(args, "event_roi_fusion_save_manifest"))
        if hasattr(args, "event_roi_fusion_ocr_source_weight"):
            self.event_roi_fusion_ocr_source_weight = float(max(0.0, min(1.0, getattr(args, "event_roi_fusion_ocr_source_weight"))))
        if hasattr(args, "event_roi_fusion_ocr_batch_size"):
            self.event_roi_fusion_ocr_batch_size = int(max(1, getattr(args, "event_roi_fusion_ocr_batch_size")))
        if hasattr(args, "event_roi_fusion_ocr_flush_every_frames"):
            self.event_roi_fusion_ocr_flush_every_frames = int(max(1, getattr(args, "event_roi_fusion_ocr_flush_every_frames")))
        if self.is_b5_yolo_high_quality_fusion_mode:
            self._log("[B5_BASELINE] gpu_pipeline B5 YOLO high-quality crop fusion mode enabled")
            self._log(
                f"[B5_BASELINE] top_k={self.b5_fusion_top_k} "
                f"min_quality={self.b5_fusion_min_quality_score} min_crops={self.b5_fusion_min_crops}"
            )
            if self.trial020_fusion_bridge is not None:
                self.trial020_fusion_bridge = Trial020FusionBridge(
                    enabled=True,
                    debug_dir=self.event_roi_fusion_debug_dir,
                    save_debug=True,
                    save_manifest=self.event_roi_fusion_save_manifest,
                    source_weight=0.0,
                    min_crops=self.b5_fusion_min_crops,
                    top_k=self.b5_fusion_top_k,
                    logger=self._log,
                    center_pad_ratio=0.15,
                    preset_name=self.event_roi_fusion_preset,
                    use_motion_fusion_filter=self.event_roi_fusion_preset in {"trial013_fixed_motion", "trial013_fixed_motion_color", "trial013_fixed_motion_filter"},
                    use_color_soft_weight=self.event_roi_fusion_preset == "trial013_fixed_motion_color",
                )

    def configure_group_review_from_args(self, args) -> None:
        self.export_group_review = bool(getattr(args, "export_group_review", False))
        if not self.export_group_review:
            return
        input_path = Path(str(getattr(args, "input", "") or "video"))
        self.group_review_output = str(getattr(args, "group_review_output", "./outputs/group_review"))
        self.group_review_video_id = str(getattr(args, "group_review_video_id", "") or input_path.stem)
        self.group_review_max_crops = int(max(1, getattr(args, "group_review_max_crops", 6)))
        self.group_review_min_frame_gap = int(max(0, getattr(args, "group_review_min_frame_gap", 5)))
        self.group_review_dedup_threshold = float(
            max(0.0, min(1.0, getattr(args, "group_review_dedup_threshold", 0.95)))
        )
        self.group_review_save_raw = bool(getattr(args, "group_review_save_raw", True))
        self.group_review_save_normalized = bool(getattr(args, "group_review_save_normalized", True))
        self._log(
            f"[GROUP_REVIEW] enabled video_id={self.group_review_video_id} "
            f"output={self.group_review_output}"
        )

    def configure_labeling_febam_sink_from_args(self, args) -> None:
        sink_dir = str(getattr(args, "labeling_febam_sink_dir", "") or "")
        self.labeling_febam_sink = None
        if not sink_dir:
            return
        from tools.labeling_pipeline.febam_observation_sink import FEBAMObservationSink
        input_path = Path(str(getattr(args, "input", "") or "video"))
        self.labeling_febam_sink = FEBAMObservationSink(
            sink_dir,
            video_id=input_path.stem,
            video_path=str(input_path),
            fps=float(getattr(self, "fps", 25.0) or 25.0),
        )

    def _b5_log_limited(self, message: str) -> None:
        count = int(getattr(self, "b5_fusion_log_count", 0))
        if count < int(getattr(self, "b5_fusion_log_limit", 100)):
            self._log(message)
            self.b5_fusion_log_count = count + 1

    def _b5_yolo_quality_score(self, obs: dict[str, object]) -> float:
        """
        B5 scores YOLO/crop quality with safe optional keys only.
        OCR confidence is a weak auxiliary signal and text is not required.
        """
        score = 0.0
        for key in ["yolo_conf", "det_conf", "det_score", "candidate_score", "score", "final_candidate_score"]:
            if key in obs:
                try:
                    score += float(obs.get(key) or 0.0) * 3.0
                    break
                except Exception:
                    pass

        roi_aspect = obs.get("roi_aspect", None)
        if roi_aspect is None:
            w = obs.get("roi_w", None)
            h = obs.get("roi_h", None)
            try:
                if w is not None and h is not None and float(h) > 0:
                    roi_aspect = float(w) / float(h)
            except Exception:
                roi_aspect = None
        try:
            ar = float(roi_aspect)
            if 2.0 <= ar <= 8.5:
                score += 1.5
            elif 1.5 <= ar <= 10.0:
                score += 0.5
        except Exception:
            pass

        try:
            w = float(obs.get("roi_w", 0.0) or 0.0)
            h = float(obs.get("roi_h", 0.0) or 0.0)
            area = w * h
            if w >= 40 and h >= 12:
                score += 1.0
            if area >= 800:
                score += 0.5
        except Exception:
            pass

        policy = (str(obs.get("ocr_crop_policy", "")) + " " + str(obs.get("ocr_trigger_reason", ""))).lower()
        if "confirmed" in policy or "near" in policy or "strong" in policy:
            score += 0.5

        for key in ["ocr_conf", "corrected_conf", "raw_conf", "best_conf"]:
            if key in obs:
                try:
                    score += float(obs.get(key) or 0.0) * 0.5
                    break
                except Exception:
                    pass
        return float(score)

    def _should_submit_b5_yolo_crop(self, obs: dict[str, object]) -> tuple[bool, str, float]:
        source = str(obs.get("source", obs.get("ocr_source", "")))
        if source in ("event_fusion_trial020_ocr", "event_fusion_trial020", "b5_yolo_high_quality_fusion_ocr"):
            return False, "b5_exclude_fusion_source", 0.0
        q = self._b5_yolo_quality_score(obs)
        if q < float(getattr(self, "b5_fusion_min_quality_score", 0.0)):
            return False, "b5_quality_too_low", q
        return True, "b5_ok", q


    def _b5_group_key(self, *, track_id: int, segment_id: str | int | None, event_id: str | int | None) -> str:
        segment = str(segment_id or "default").strip() or "default"
        event = str(event_id or "evt0000").strip() or "evt0000"
        if event.isdigit():
            event = f"evt{int(event):04d}"
        elif not event.lower().startswith("evt"):
            event = f"evt{event}"
        group_key = f"track{int(track_id):04d}_{segment}_{event}"
        return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", group_key)

    def _b5_buffer_yolo_crop(self, *, crop_bgr, obs: dict[str, object], quality_score: float) -> None:
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return
        frame_idx = int(obs.get("frame_idx", -1) if obs.get("frame_idx", -1) is not None else -1)
        track_id = int(obs.get("track_id", -1) if obs.get("track_id", -1) is not None else -1)
        candidate_idx = int(obs.get("candidate_idx", -1) if obs.get("candidate_idx", -1) is not None else -1)
        variant = str(obs.get("variant", obs.get("ocr_input_variant", "")) or "")
        input_key = (frame_idx, track_id, candidate_idx, variant)
        group_key = self._b5_group_key(
            track_id=track_id,
            segment_id=obs.get("segment_id", "default"),
            event_id=obs.get("event_id", "evt0000"),
        )
        item = self.b5_fusion_input_index.get(input_key)
        if item is None:
            item = {
                "crop_bgr": np.array(crop_bgr, copy=True),
                "quality_score": float(quality_score),
                "frame_idx": frame_idx,
                "track_id": track_id,
                "candidate_idx": candidate_idx,
                "variant": variant,
                "group_key": group_key,
                "segment_id": str(obs.get("segment_id", "default") or "default"),
                "event_id": str(obs.get("event_id", "evt0000") or "evt0000"),
                "meta": dict(obs),
            }
            self.b5_fusion_input_index[input_key] = item
            self.b5_fusion_groups.setdefault(group_key, []).append(item)
            return
        if float(quality_score) >= float(item.get("quality_score", 0.0) or 0.0):
            item["crop_bgr"] = np.array(crop_bgr, copy=True)
            item["quality_score"] = float(quality_score)
            item["meta"] = {**dict(item.get("meta", {})), **dict(obs)}

    def _b5_save_fused_image(self, image: np.ndarray, group_key: str, method: str, frame_end: int, used_count: int) -> str:
        fusion_dir = Path(getattr(self, "event_roi_fusion_debug_dir", "../data/processed/debug/b5_yolo_high_quality_fusion")) / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)
        method_tag = "weighted_average" if method == "weighted_average_fusion" else "median"
        path = fusion_dir / f"{group_key}_{method_tag}_top{used_count:02d}_frame{int(frame_end):05d}.png"
        ok = cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))
        return str(path) if ok else ""

    def _finalize_b5_yolo_high_quality_fusion(self, *, reason: str = "pipeline_close") -> None:
        if not getattr(self, "is_b5_yolo_high_quality_fusion_mode", False):
            return
        if getattr(self, "b5_fusion_finalized", False):
            return
        self.b5_fusion_finalized = True
        groups = getattr(self, "b5_fusion_groups", {}) or {}
        if not groups:
            self._log(f"[B5_BASELINE] finalize reason={reason} groups=0")
            return
        fusion_size = (384, 96)
        top_k = int(max(1, getattr(self, "b5_fusion_top_k", 8)))
        min_crops = int(max(2, getattr(self, "b5_fusion_min_crops", 2)))
        finalized = 0
        for group_key, items in sorted(groups.items()):
            ranked = sorted(
                items,
                key=lambda item: (-float(item.get("quality_score", 0.0) or 0.0), int(item.get("frame_idx", -1) or -1)),
            )
            selected = ranked[:top_k]
            if len(selected) < min_crops:
                self._b5_log_limited(
                    f"[B5_BASELINE] final_skip group={group_key} reason=min_crops count={len(selected)} min_crops={min_crops}"
                )
                continue
            resized: list[np.ndarray] = []
            for item in selected:
                crop = item.get("crop_bgr")
                if crop is None or getattr(crop, "size", 0) == 0:
                    continue
                resized.append(cv2.resize(crop, fusion_size, interpolation=cv2.INTER_AREA).astype(np.float32))
            if len(resized) < min_crops:
                self._b5_log_limited(
                    f"[B5_BASELINE] final_skip group={group_key} reason=resize_failed count={len(resized)} min_crops={min_crops}"
                )
                continue
            frame_values = [int(item.get("frame_idx", -1) or -1) for item in selected]
            frame_end = max(frame_values) if frame_values else -1
            first = selected[0]
            quality_scores = [float(item.get("quality_score", 0.0) or 0.0) for item in selected]
            for method in ("median_fusion", "weighted_average_fusion"):
                if method == "median_fusion":
                    fused = np.median(np.stack(resized, axis=0), axis=0)
                else:
                    weights = np.asarray([max(score, 1e-6) for score in quality_scores[:len(resized)]], dtype=np.float32)
                    weights = weights / max(float(np.sum(weights)), 1e-6)
                    fused = np.zeros_like(resized[0], dtype=np.float32)
                    for image, weight in zip(resized, weights):
                        fused += image * float(weight)
                fused_path = self._b5_save_fused_image(fused, group_key, method, frame_end, len(resized))
                if not fused_path:
                    self._b5_log_limited(f"[B5_BASELINE] final_skip group={group_key} method={method} reason=save_failed")
                    continue
                obs = {
                    "frame_idx": frame_end,
                    "track_id": first.get("track_id", -1),
                    "candidate_idx": -1,
                    "segment_id": first.get("segment_id", "default"),
                    "event_id": first.get("event_id", "evt0000"),
                    "conf": max(quality_scores) if quality_scores else 0.0,
                    "source": "b5_yolo_high_quality_fusion_ocr",
                    "ocr_source": "b5_yolo_high_quality_fusion_ocr",
                    "ocr_engine": "b5_yolo_high_quality_fusion_fastplate",
                    "source_weight": 0.0,
                    "fused_image_path": fused_path,
                    "event_fusion_method": method,
                    "event_fusion_preset": "b5_yolo_high_quality",
                    "event_fusion_mode": "b5_yolo_high_quality_fusion",
                    "event_fusion_group_key": group_key,
                    "event_group_key": group_key,
                    "group_id": group_key,
                    "event_fusion_num_used": len(resized),
                    "event_fusion_selected_top_k": top_k,
                    "event_fusion_center_pad_ratio": 0.15,
                    "event_fusion_color_thr": "",
                    "event_fusion_max_shift_ratio": "",
                    "event_fusion_top_candidates": "",
                    "b5_yolo_quality_score": "|".join(f"{score:.6f}" for score in quality_scores[:len(resized)]),
                    "b5_fusion_output_tag": getattr(self, "b5_fusion_output_tag", "b5_yolo_high_quality_fusion"),
                }
                self._enqueue_event_fusion_ocr(obs)
                finalized += 1
        self._log(f"[B5_BASELINE] finalized reason={reason} groups={len(groups)} fused_images={finalized}")

    def _get_event_fusion_fastplate_recognizer(self):
        if not getattr(self, "event_roi_fusion_ocr", False):
            return None
        if getattr(self, "event_roi_fusion_ocr_worker_mode", "shared_async") == "shared_async":
            return None
        if str(getattr(self, "event_roi_fusion_ocr_backend", "fastplate") or "fastplate") != "fastplate":
            return None
        if self.event_fusion_fastplate_recognizer is None:
            try:
                if self.fastplate_custom_onnx:
                    self.event_fusion_fastplate_recognizer = CustomFastPlateONNXRecognizer(
                        onnx_path=self.fastplate_custom_onnx,
                        plate_config=self.fastplate_custom_plate_config,
                        input_width=self.fastplate_custom_input_width,
                        input_height=self.fastplate_custom_input_height,
                        device=self.fastplate_device,
                        logger=self._log_always,
                    )
                    self._log_always(
                        f"[EVENT_FUSION_OCR] Custom FastPlateONNX initialized "
                        f"onnx={self.fastplate_custom_onnx} device={self.fastplate_device} "
                        f"batch_size={getattr(self, 'event_roi_fusion_ocr_batch_size', self.fastplate_batch_size)}"
                    )
                else:
                    from ocr.ocr_fastplate_batch import FastPlateOCRBatchRecognizer

                    self.event_fusion_fastplate_recognizer = FastPlateOCRBatchRecognizer(
                        model=self.fastplate_model,
                        device=self.fastplate_device,
                        batch_size=getattr(self, "event_roi_fusion_ocr_batch_size", self.fastplate_batch_size),
                        preload_torch_cuda_dlls=self.fastplate_preload_torch_cuda_dlls,
                        min_text_len=self.fastplate_min_text_len,
                    )
                    self._log(
                        f"[EVENT_FUSION_OCR] FastPlateOCRBatchRecognizer initialized "
                        f"model={self.fastplate_model} device={self.fastplate_device} "
                        f"batch_size={getattr(self, 'event_roi_fusion_ocr_batch_size', self.fastplate_batch_size)}"
                    )
            except Exception as exc:
                self._log(f"[EVENT_FUSION_OCR] fastplate init failed: {type(exc).__name__}: {exc}")
                return None
        return self.event_fusion_fastplate_recognizer

    def _enqueue_event_fusion_ocr_from_manifest_rows(self, manifest_start: int, *, skip_paths: set[str] | None = None) -> None:
        if not getattr(self, "event_roi_fusion_ocr", False):
            return
        bridge = getattr(self, "trial020_fusion_bridge", None)
        rows = list(getattr(bridge, "manifest_rows", []) or [])
        skip_paths = set(skip_paths or set())
        for manifest_row in rows[max(0, int(manifest_start)):]:
            fused_path = str(manifest_row.get("fused_image_path", "") or "")
            if not fused_path or fused_path in skip_paths:
                continue
            status = str(manifest_row.get("status", "") or "")
            if status not in {"fused", "fused_update", "fused_image_only", "fused_final", "fused_anchor_only"}:
                continue
            b5_mode = getattr(self, "is_b5_yolo_high_quality_fusion_mode", False)
            obs = dict(manifest_row)
            obs.update({
                "frame_idx": manifest_row.get("frame_end", -1),
                "candidate_idx": -1,
                "conf": manifest_row.get("score", 0.0),
                "fused_image_path": fused_path,
                "event_fusion_method": manifest_row.get("method", ""),
                "event_fusion_preset": "b5_yolo_high_quality" if b5_mode else str(manifest_row.get("event_fusion_preset", manifest_row.get("fusion_preset", "")) or getattr(getattr(self, "trial020_fusion_bridge", None), "preset", object()).name if getattr(getattr(self, "trial020_fusion_bridge", None), "preset", None) is not None else ""),
                "event_fusion_mode": "b5_yolo_high_quality_fusion" if b5_mode else str(manifest_row.get("event_fusion_mode", "ours_all_roi_febam") or "ours_all_roi_febam"),
                "event_fusion_group_key": manifest_row.get("event_group_key", ""),
                "event_group_key": manifest_row.get("event_group_key", ""),
                "group_id": manifest_row.get("event_group_key", ""),
                "event_fusion_num_used": manifest_row.get("num_used", ""),
                "event_fusion_top_candidates": manifest_row.get("top_candidates", ""),
            })
            self._enqueue_event_fusion_ocr(obs)

    def _enqueue_event_fusion_ocr(self, fused_obs: dict[str, object]) -> None:
        if not getattr(self, "event_roi_fusion_ocr", False):
            return
        fused_path = str(fused_obs.get("fused_image_path", "") or "")
        if not fused_path:
            return
        if fused_path in self.event_fusion_ocr_seen_paths:
            if getattr(self, "event_roi_fusion_ocr_worker_mode", "shared_async") == "shared_async":
                self.fusion_ocr_duplicate_blocked += 1
                return
            for item in getattr(self, "event_fusion_ocr_buffer", []):
                meta = item.get("meta", {})
                if str(meta.get("fused_image_path", "") or "") == fused_path:
                    meta.update({key: value for key, value in dict(fused_obs).items() if value not in {"", None}})
            return
        fused_bgr = fused_obs.get("fused_bgr")
        if fused_bgr is None or getattr(fused_bgr, "size", 0) == 0:
            try:
                fused_bgr = cv2.imread(fused_path, cv2.IMREAD_COLOR)
            except Exception:
                fused_bgr = None
        if fused_bgr is None or getattr(fused_bgr, "size", 0) == 0:
            self._log(f"[EVENT_FUSION_OCR] fused_ocr_empty_load path={fused_path}")
            return
        try:
            fused_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            fused_rgb = fused_bgr
        meta = dict(fused_obs)
        b5_mode = getattr(self, "is_b5_yolo_high_quality_fusion_mode", False)
        ocr_source = "b5_yolo_high_quality_fusion_ocr" if b5_mode else "event_fusion_trial020_ocr"
        ocr_engine = "b5_yolo_high_quality_fusion_fastplate" if b5_mode else ocr_source
        meta.update({
            "source": ocr_source,
            "ocr_source": ocr_source,
            "ocr_engine": ocr_engine,
            "source_weight": 0.0 if b5_mode else getattr(self, "event_roi_fusion_ocr_source_weight", 0.65),
            "event_fusion_ocr_backend": "fastplate",
            "event_fusion_mode": "b5_yolo_high_quality_fusion" if b5_mode else str(meta.get("event_fusion_mode", "ours_all_roi_febam") or "ours_all_roi_febam"),
            "event_fusion_preset": "b5_yolo_high_quality" if b5_mode else str(meta.get("event_fusion_preset", "") or ""),
            "fused_image_path": fused_path,
        })
        if getattr(self, "event_roi_fusion_ocr_worker_mode", "shared_async") == "shared_async":
            event_key = str(
                meta.get("event_fusion_group_key", "")
                or meta.get("event_group_key", "")
                or meta.get("group_id", "")
                or meta.get("event_id", "")
                or fused_path
            )
            if event_key in self.fusion_ocr_enqueued_events:
                self.fusion_ocr_duplicate_blocked += 1
                return
            try:
                fused_cuda = torch.from_numpy(np.ascontiguousarray(fused_rgb)).to(
                    device=self.device, dtype=torch.uint8, non_blocking=True
                ).permute(2, 0, 1).contiguous()
                self.fusion_ocr_cpu_input_count += 1
            except Exception as exc:
                self.fusion_ocr_dropped += 1
                self._log(f"[EVENT_FUSION_OCR_SHARED] cuda_upload_failed path={fused_path} error={type(exc).__name__}: {exc}")
                return
            meta.update({
                "source_type": "event_fused_group_key",
                "variant": "event_fusion_shared",
                "variant_name": "event_fusion_shared",
                "event_id": event_key,
                "track_id": int(meta.get("track_id", -1) if meta.get("track_id", -1) not in {"", None} else -1),
                "candidate_idx": int(meta.get("candidate_idx", -1) if meta.get("candidate_idx", -1) not in {"", None} else -1),
                "frame_idx": int(meta.get("frame_idx", -1) if meta.get("frame_idx", -1) not in {"", None} else -1),
                "source_level": "febam_confirmed",
                "fusion_shared_enqueue_time": time.perf_counter(),
            })
            if self._enqueue_fastplate_async(fused_cuda, meta):
                self.event_fusion_ocr_seen_paths.add(fused_path)
                self.fusion_ocr_enqueued_events.add(event_key)
                self.fusion_ocr_enqueued += 1
                self.fusion_ocr_max_per_event_seen = max(self.fusion_ocr_max_per_event_seen, 1)
                self._log(f"[EVENT_FUSION_OCR_SHARED] enqueued event={event_key} path={fused_path}")
            else:
                self.fusion_ocr_dropped += 1
            return
        self.event_fusion_ocr_seen_paths.add(fused_path)
        self.event_fusion_ocr_buffer.append({
            "crop": fused_rgb,
            "meta": meta,
            "variant": "event_fusion_trial020_ocr",
        })
        if len(self.event_fusion_ocr_buffer) >= int(getattr(self, "event_roi_fusion_ocr_batch_size", 32)):
            self._flush_event_fusion_ocr(reason="batch_full")

    def _maybe_flush_event_fusion_ocr_by_frame(self, frame_idx: int | None) -> None:
        if not getattr(self, "event_roi_fusion_ocr", False):
            return
        if frame_idx is None:
            return
        every = int(max(1, getattr(self, "event_roi_fusion_ocr_flush_every_frames", 5)))
        frame_value = int(frame_idx)
        last_frame = int(getattr(self, "_last_event_fusion_ocr_flush_frame", -1))
        if self.event_fusion_ocr_buffer and (last_frame < 0 or frame_value - last_frame >= every):
            self._flush_event_fusion_ocr(reason=f"frame_interval_{every}", frame_idx=frame_value)

    def _flush_event_fusion_ocr(self, *, reason: str = "manual", frame_idx: int | None = None) -> None:
        if not getattr(self, "event_roi_fusion_ocr", False):
            return
        if getattr(self, "event_roi_fusion_ocr_worker_mode", "shared_async") == "shared_async":
            self._drain_fastplate_results(max_items=None)
            return
        if not self.event_fusion_ocr_buffer:
            return
        recognizer = self._get_event_fusion_fastplate_recognizer()
        if recognizer is None:
            return
        batch = self.event_fusion_ocr_buffer
        self.event_fusion_ocr_buffer = []
        crops = [item["crop"] for item in batch]
        metas = [dict(item.get("meta", {})) for item in batch]
        variant_names = [str(item.get("variant", "event_fusion_trial020_ocr") or "event_fusion_trial020_ocr") for item in batch]
        try:
            results = recognizer.recognize_batch(crops, metas=metas, variant_names=variant_names)
        except Exception as exc:
            self._log(f"[EVENT_FUSION_OCR] fused_ocr_batch_exception reason={reason} error={type(exc).__name__}: {exc}")
            return
        for batch_item, meta, result in zip(batch, metas, results):
            raw_text = str(result.get("text", "") or "")
            conf = float(result.get("conf", 0.0) or 0.0)
            text = self._postprocess_event_fusion_ocr_text(raw_text, conf=conf, source="fast_plate_ocr")
            fused_path = str(meta.get("fused_image_path", "") or "")
            if not text:
                self._log(f"[EVENT_FUSION_OCR] fused_ocr_empty path={fused_path} reason={reason}")
                continue
            obs = dict(meta)
            raw_result_text = str(result.get("raw_result", "") or "")[:500]
            raw_plate = self._extract_event_fusion_raw_plate(raw_result_text, fallback_text=raw_text or text)
            b5_mode = getattr(self, "is_b5_yolo_high_quality_fusion_mode", False)
            ocr_source = "b5_yolo_high_quality_fusion_ocr" if b5_mode else "event_fusion_trial020_ocr"
            ocr_engine = "b5_yolo_high_quality_fusion_fastplate" if b5_mode else ocr_source
            obs.update({
                "text": text,
                "raw_text": raw_text or text,
                "corrected_text": text,
                "conf": conf,
                "source": ocr_source,
                "ocr_source": ocr_source,
                "ocr_engine": ocr_engine,
                "source_weight": 0.0 if b5_mode else getattr(self, "event_roi_fusion_ocr_source_weight", 0.65),
                "event_fusion_ocr_backend": "fastplate",
                "event_fusion_ocr_flush_reason": reason,
                "event_fusion_ocr_raw_result": raw_result_text,
                "event_fusion_ocr_raw_plate": raw_plate,
                "event_fusion_ocr_raw_matches_text": 1 if raw_plate and raw_plate == text else 0,
                "event_fusion_mode": "b5_yolo_high_quality_fusion" if b5_mode else str(obs.get("event_fusion_mode", "ours_all_roi_febam") or "ours_all_roi_febam"),
                "event_fusion_preset": "b5_yolo_high_quality" if b5_mode else str(obs.get("event_fusion_preset", "") or ""),
            })
            self._enqueue_middle_slot_upl(
                batch_item.get("crop"),
                ocr_text=text,
                ocr_conf=conf,
                source=ocr_source,
                meta={
                    **obs,
                    "frame_idx": int(obs.get("frame_idx", frame_idx if frame_idx is not None else -1) or -1),
                    "track_id": int(obs.get("track_id", -1) if obs.get("track_id", -1) not in {"", None} else -1),
                    "candidate_idx": int(obs.get("candidate_idx", -1) if obs.get("candidate_idx", -1) not in {"", None} else -1),
                    "fused_image_path": fused_path,
                    "variant_group_key": str(obs.get("event_fusion_group_key", fused_path) or fused_path),
                    "source_weight": self.middle_slot_source_weight,
                },
            )
            self._submit_trial020_fused_observation(
                obs,
                bbox=None,
                candidate_idx=int(obs.get("candidate_idx", -1) if obs.get("candidate_idx", -1) not in {"", None} else -1),
                roi_w="",
                roi_h="",
                candidate_count=0,
                track_id_count=0,
                febam_confirmed_count=0,
                febam_score=conf,
                febam_energy="",
                febam_memory="",
            )
        self._last_event_fusion_ocr_flush_frame = int(frame_idx) if frame_idx is not None else int(getattr(self, "_current_frame_idx", -1) or -1)


    def _send_trial020_bridge_observations(
        self,
        fused_observations: list[dict],
        *,
        bbox,
        candidate_idx: int | None = None,
        roi_w: int | str = "",
        roi_h: int | str = "",
        candidate_count: int = 0,
        track_id_count: int = 0,
        febam_confirmed_count: int = 0,
        febam_score: object = "",
        febam_energy: object = "",
        febam_memory: object = "",
    ) -> None:
        for fused_obs in fused_observations or []:
            self._enqueue_event_fusion_ocr(fused_obs)
            if getattr(self, "is_b5_yolo_high_quality_fusion_mode", False):
                continue
            self._submit_trial020_fused_observation(
                fused_obs,
                bbox=bbox,
                candidate_idx=candidate_idx,
                roi_w=roi_w,
                roi_h=roi_h,
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
                febam_score=febam_score,
                febam_energy=febam_energy,
                febam_memory=febam_memory,
            )

    def _submit_trial020_roi_crop_to_bridge(
        self,
        *,
        frame_idx: int | None,
        track_id: int,
        candidate_idx: int,
        crop_bgr,
        text: str = "",
        conf: float = 0.0,
        source: str = "ocr_roi_image_only",
        source_weight: float = 0.25,
        segment_id: str | int | None = None,
        event_id: str | int | None = None,
        febam_score: float = 0.0,
        febam_energy: float = 0.0,
        febam_memory: float = 0.0,
        variant: str = "",
        meta: dict | None = None,
        bbox=None,
        roi_w: int | str = "",
        roi_h: int | str = "",
        candidate_count: int = 0,
        track_id_count: int = 0,
        febam_confirmed_count: int = 0,
    ) -> None:
        if self.trial020_fusion_bridge is None:
            return
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return
        try:
            meta_dict = dict(meta or {})
            bridge_obs = {
                **meta_dict,
                "frame_idx": int(frame_idx) if frame_idx is not None else -1,
                "track_id": int(track_id),
                "candidate_idx": int(candidate_idx),
                "text": text,
                "ocr_conf": float(conf or 0.0),
                "conf": float(conf or 0.0),
                "source": source,
                "source_weight": float(source_weight or 0.0),
                "segment_id": segment_id if segment_id else "default",
                "event_id": event_id,
                "febam_score": float(febam_score or 0.0),
                "febam_energy": float(febam_energy or 0.0),
                "febam_memory": float(febam_memory or 0.0),
                "variant": str(variant or ""),
                "roi_w": roi_w,
                "roi_h": roi_h,
            }
            if getattr(self, "is_b5_yolo_high_quality_fusion_mode", False):
                ok, reason, q = self._should_submit_b5_yolo_crop(bridge_obs)
                if not ok:
                    self._b5_log_limited(f"[B5_BASELINE] skip reason={reason} q={q:.4f}")
                    return
                self._b5_log_limited(f"[B5_BASELINE] buffer q={q:.4f}")
                bridge_obs.update({
                    "b5_yolo_quality_score": f"{q:.6f}",
                    "event_fusion_mode": "b5_yolo_high_quality_fusion",
                    "b5_fusion_output_tag": getattr(self, "b5_fusion_output_tag", "b5_yolo_high_quality_fusion"),
                })
                self._b5_buffer_yolo_crop(crop_bgr=crop_bgr, obs=bridge_obs, quality_score=q)
                return
            manifest_start = len(getattr(self.trial020_fusion_bridge, "manifest_rows", []) or [])
            fused_observations = self.trial020_fusion_bridge.add_observation(
                frame_idx=int(frame_idx) if frame_idx is not None else -1,
                track_id=int(track_id),
                candidate_idx=int(candidate_idx),
                crop_bgr=crop_bgr,
                text=text,
                conf=float(conf or 0.0),
                source=source,
                source_weight=float(source_weight or 0.0),
                segment_id=segment_id if segment_id else "default",
                event_id=event_id,
                febam_score=float(febam_score or 0.0),
                febam_energy=float(febam_energy or 0.0),
                febam_memory=float(febam_memory or 0.0),
                variant=str(variant or ""),
                meta=meta_dict,
            )
            self._send_trial020_bridge_observations(
                fused_observations,
                bbox=bbox,
                candidate_idx=candidate_idx,
                roi_w=roi_w,
                roi_h=roi_h,
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
                febam_score=febam_score,
                febam_energy=febam_energy,
                febam_memory=febam_memory,
            )
            direct_paths = {str(obs.get("fused_image_path", "") or "") for obs in fused_observations or [] if obs.get("fused_image_path")}
            self._enqueue_event_fusion_ocr_from_manifest_rows(manifest_start, skip_paths=direct_paths)
        except Exception as exc:
            self._record_trial020_bridge_exception_manifest(
                exc,
                frame_idx=frame_idx,
                track_id=track_id,
                segment_id=segment_id if segment_id else "default",
                event_id=event_id,
            )
            self._log_exception("trial020_event_roi_fusion_exception")

    def _record_trial020_bridge_exception_manifest(
        self,
        exc: Exception,
        *,
        frame_idx: int | None,
        track_id: int,
        segment_id: str | int | None,
        event_id: str | int | None,
    ) -> None:
        bridge = getattr(self, "trial020_fusion_bridge", None)
        if bridge is None or not hasattr(bridge, "_record_manifest"):
            return
        segment_value = str(segment_id or "default")
        event_text = str(event_id or "")
        if event_text and not event_text.lower().startswith("evt"):
            event_text = f"evt{event_text}" if not str(event_text).isdigit() else f"evt{int(event_text):04d}"
        try:
            bridge._record_manifest(
                event_group_key=f"{int(track_id)}:{segment_value}:{event_text or 'bridge_exception'}",
                track_id=int(track_id),
                segment_id=segment_value,
                event_id=event_text,
                method="debug",
                status="bridge_exception",
                skip_reason=type(exc).__name__,
                frame_start=int(frame_idx) if frame_idx is not None else "",
                frame_end=int(frame_idx) if frame_idx is not None else "",
                num_rows=0,
                num_used=0,
                final_text="",
                score=0.0,
                top_candidates="",
                fused_image_path="",
                submitted_to_string_febam=0,
                notes=str(exc),
            )
        except Exception:
            return

    def _submit_trial020_fused_observation(
        self,
        fused_obs: dict[str, object],
        *,
        bbox,
        candidate_idx: int | None = None,
        roi_w: int | str = "",
        roi_h: int | str = "",
        candidate_count: int = 0,
        track_id_count: int = 0,
        febam_confirmed_count: int = 0,
        febam_score: object = "",
        febam_energy: object = "",
        febam_memory: object = "",
    ) -> None:
        text = str(fused_obs.get("text", "") or "")
        if not text:
            return
        track_id = int(fused_obs.get("track_id", -1))
        frame_idx = int(fused_obs.get("frame_idx", -1))
        candidate_idx = int(fused_obs.get("candidate_idx", -1))
        conf = float(fused_obs.get("conf", 0.0) or 0.0)
        source = str(fused_obs.get("source", "event_fusion_trial020") or "event_fusion_trial020")
        b5_mode = getattr(self, "is_b5_yolo_high_quality_fusion_mode", False) or source == "b5_yolo_high_quality_fusion_ocr"
        if b5_mode:
            source = "b5_yolo_high_quality_fusion_ocr"
        source_weight_default = 0.0 if b5_mode else (0.65 if source == "event_fusion_trial020_ocr" else getattr(self, "event_roi_fusion_source_weight", 0.55))
        source_weight = float(fused_obs.get("source_weight", source_weight_default) or source_weight_default)
        engine = "b5_yolo_high_quality_fusion_fastplate" if b5_mode else str(fused_obs.get("ocr_engine", source) or source)
        mrs_value = float(febam_score) if febam_score not in {"", None} else conf
        string_group_id, string_group_key = self._resolve_string_febam_group_id(fused_obs, track_id)
        fused_obs["group_id"] = string_group_key
        self._merge_string_febam_track_into_group(track_id, string_group_id, string_group_key)
        if not b5_mode:
            self._last_final_plate = self._append_ocr_history(
                track_id,
                text,
                conf,
                frame_idx,
                bbox,
                mrs_value,
                raw_text=str(fused_obs.get("raw_text", text) or text),
                corrected_text=str(fused_obs.get("corrected_text", text) or text),
                source=source,
                source_weight=source_weight,
                string_group_id=string_group_id,
                string_group_key=string_group_key,
            )
        self._append_ocr_csv_row(
            track_id,
            frame_idx,
            text,
            conf,
            bbox,
            candidate_idx=candidate_idx,
            roi_w=roi_w,
            roi_h=roi_h,
            candidate_count=candidate_count,
            track_id_count=track_id_count,
            febam_confirmed_count=febam_confirmed_count,
            attempted_count=1,
            febam_score=febam_score,
            febam_energy=febam_energy,
            febam_memory=febam_memory,
            ocr_trigger_reason=source,
            ocr_sampling_reason=f"{source}_update",
            ocr_sampling_level=source,
            source_weight=source_weight,
            ocr_saved_raw=str(fused_obs.get("roi_path", "") or ""),
            ocr_saved_rotated=str(fused_obs.get("fused_image_path", "") or ""),
            ocr_input_variant=source,
            raw_text=str(fused_obs.get("raw_text", text) or text),
            raw_conf=conf,
            corrected_text=str(fused_obs.get("corrected_text", text) or text),
            corrected_conf=conf,
            source=source,
            ocr_engine=engine,
            normalized_plate_candidate=text,
            final_candidate_source=source,
            final_candidate_score=conf,
            final_candidate_reason="b5_yolo_high_quality_fusion" if b5_mode else source,
            final_output_plate=text,
            grammar_best_text=text,
            grammar_best_score=conf,
        )
        if self.ocr_csv_rows:
            self.ocr_csv_rows[-1].update({
                "row_type": str(fused_obs.get("row_type", "fusion_ocr_observation") or "fusion_ocr_observation"),
                "ocr_source": source,
                "segment_id": str(fused_obs.get("segment_id", "") or ""),
                "event_id": str(fused_obs.get("event_id", "") or ""),
                "fused_image_path": str(fused_obs.get("fused_image_path", "") or ""),
                "event_fusion_method": str(fused_obs.get("event_fusion_method", "") or ""),
                "event_fusion_preset": "b5_yolo_high_quality" if b5_mode else str(fused_obs.get("event_fusion_preset", "") or ""),
                "event_fusion_mode": "b5_yolo_high_quality_fusion" if b5_mode else str(fused_obs.get("event_fusion_mode", "ours_all_roi_febam") or "ours_all_roi_febam"),
                "event_fusion_group_key": str(fused_obs.get("event_fusion_group_key", "") or ""),
                "group_id": str(fused_obs.get("group_id", "") or ""),
                "event_fusion_num_used": fused_obs.get("event_fusion_num_used", ""),
                "event_fusion_selected_top_k": fused_obs.get("event_fusion_selected_top_k", ""),
                "event_fusion_center_pad_ratio": fused_obs.get("event_fusion_center_pad_ratio", ""),
                "event_fusion_color_thr": fused_obs.get("event_fusion_color_thr", ""),
                "event_fusion_max_shift_ratio": fused_obs.get("event_fusion_max_shift_ratio", ""),
                "event_fusion_top_candidates": str(fused_obs.get("event_fusion_top_candidates", "") or ""),
                "event_fusion_ocr_backend": str(fused_obs.get("event_fusion_ocr_backend", "") or ""),
                "event_fusion_ocr_flush_reason": str(fused_obs.get("event_fusion_ocr_flush_reason", "") or ""),
                "event_fusion_ocr_raw_result": str(fused_obs.get("event_fusion_ocr_raw_result", "") or ""),
                "event_fusion_ocr_raw_plate": str(fused_obs.get("event_fusion_ocr_raw_plate", "") or ""),
                "event_fusion_ocr_raw_matches_text": fused_obs.get("event_fusion_ocr_raw_matches_text", ""),
                "pseudo_vehicle_id": str(fused_obs.get("pseudo_vehicle_id", "") or ""),
                "rank1_crop_id": str(fused_obs.get("rank1_crop_id", "") or ""),
                "rank1_quality_score": fused_obs.get("rank1_quality_score", ""),
                "rank1_color_proto": str(fused_obs.get("rank1_color_proto", "") or ""),
                "adaptive_alpha": fused_obs.get("adaptive_alpha", ""),
                "b5_yolo_quality_score": str(fused_obs.get("b5_yolo_quality_score", "") or ""),
                "b5_fusion_output_tag": str(fused_obs.get("b5_fusion_output_tag", getattr(self, "b5_fusion_output_tag", "")) or ""),
                "motion_cluster_id": str(fused_obs.get("motion_cluster_id", "") or ""),
                "motion_state": str(fused_obs.get("motion_state", "") or ""),
                "motion_activation": fused_obs.get("motion_activation", ""),
                "motion_score": fused_obs.get("motion_score", ""),
                "motion_margin": fused_obs.get("motion_margin", ""),
                "motion_similarity": fused_obs.get("motion_similarity", ""),
                "motion_second_similarity": fused_obs.get("motion_second_similarity", ""),
                "motion_direction_bin": fused_obs.get("motion_direction_bin", ""),
                "motion_direction_name": str(fused_obs.get("motion_direction_name", "") or ""),
                "motion_speed": fused_obs.get("motion_speed", ""),
                "motion_frame_gap": fused_obs.get("motion_frame_gap", ""),
                "motion_gap_score": fused_obs.get("motion_gap_score", ""),
                "motion_y_band": fused_obs.get("motion_y_band", ""),
                "motion_contamination": fused_obs.get("motion_contamination", ""),
                "motion_split_reason": str(fused_obs.get("motion_split_reason", "") or ""),
                "motion_primary_cluster_id": str(fused_obs.get("motion_primary_cluster_id", "") or ""),
                "motion_cluster_count": fused_obs.get("motion_cluster_count", ""),
                "motion_split_applied": fused_obs.get("motion_split_applied", ""),
                "motion_filtered_rows": fused_obs.get("motion_filtered_rows", ""),
                "motion_kept_rows": fused_obs.get("motion_kept_rows", ""),
                "motion_weight_min": fused_obs.get("motion_weight_min", ""),
                "motion_weight_mean": fused_obs.get("motion_weight_mean", ""),
                "motion_weight_max": fused_obs.get("motion_weight_max", ""),
                "motion_entry_zone": str(fused_obs.get("motion_entry_zone", "") or ""),
                "motion_exit_zone": str(fused_obs.get("motion_exit_zone", "") or ""),
                "motion_scale_trend": fused_obs.get("motion_scale_trend", ""),
                "motion_position_reset_score": fused_obs.get("motion_position_reset_score", ""),
                "motion_split_evidence": fused_obs.get("motion_split_evidence", ""),
                "motion_filter_applied": fused_obs.get("motion_filter_applied", ""),
                "motion_non_primary_ratio": fused_obs.get("motion_non_primary_ratio", ""),
                "motion_contamination_ratio": fused_obs.get("motion_contamination_ratio", ""),
                "motion_representative_reason": str(fused_obs.get("motion_representative_reason", "") or ""),
                "color_soft_applied": fused_obs.get("color_soft_applied", ""),
                "color_reference_class": str(fused_obs.get("color_reference_class", "") or ""),
                "color_soft_weight_min": fused_obs.get("color_soft_weight_min", ""),
                "color_soft_weight_mean": fused_obs.get("color_soft_weight_mean", ""),
                "color_soft_weight_max": fused_obs.get("color_soft_weight_max", ""),
                "color_class_mismatch_count": fused_obs.get("color_class_mismatch_count", ""),
                "color_unknown_count": fused_obs.get("color_unknown_count", ""),
                "color_outlier_count": fused_obs.get("color_outlier_count", ""),
                "color_excluded_count": fused_obs.get("color_excluded_count", ""),
                "color_extreme_outlier_count": fused_obs.get("color_extreme_outlier_count", ""),
                "color_filter_reason": str(fused_obs.get("color_filter_reason", "") or ""),
                "fusion_final_weight_min": fused_obs.get("fusion_final_weight_min", ""),
                "fusion_final_weight_mean": fused_obs.get("fusion_final_weight_mean", ""),
                "fusion_final_weight_max": fused_obs.get("fusion_final_weight_max", ""),
                "color_distance": fused_obs.get("color_distance", ""),
                "color_class": str(fused_obs.get("color_class", "") or ""),
                "color_excluded_by_motion_color": fused_obs.get("color_excluded_by_motion_color", ""),
                "color_weight_reason": str(fused_obs.get("color_weight_reason", "") or ""),
                "fusion_final_weight": fused_obs.get("fusion_final_weight", ""),
                "fusion_rows_original": fused_obs.get("fusion_rows_original", ""),
                "fusion_rows_after_motion_filter": fused_obs.get("fusion_rows_after_motion_filter", ""),
                "fusion_rows_after_color_filter": fused_obs.get("fusion_rows_after_color_filter", ""),
                "rank1_crop_id": str(fused_obs.get("rank1_crop_id", "") or ""),
                "rank1_quality_score": fused_obs.get("rank1_quality_score", ""),
                "rank1_color_proto": str(fused_obs.get("rank1_color_proto", "") or ""),
                "adaptive_alpha": fused_obs.get("adaptive_alpha", ""),
            })
            if source in {"event_fusion_trial020_ocr", "b5_yolo_high_quality_fusion_ocr"}:
                self.fusion_ocr_csv_row_written += 1
        self._log(
            f"[EVENT_FUSION] submitted source={source} frame={frame_idx} track={track_id} "
            f"segment={fused_obs.get('segment_id', '')} event={fused_obs.get('event_id', '')} "
            f"text={text!r} conf={conf:.3f} source_weight={source_weight:.2f}"
        )

    def _append_ocr_history(
        self,
        track_id: int,
        text: str,
        conf: float,
        frame_idx: int | None,
        bbox: torch.Tensor,
        mrs: float,
        *,
        raw_text: str = "",
        corrected_text: str = "",
        source: str = "raw",
        source_weight: float = 1.0,
        string_group_id: int | None = None,
        string_group_key: str = "",
        allow_legacy_vote: bool = True,
    ) -> str:
        text = str(text or "")
        if text:
            history = self.ocr_history.setdefault(track_id, [])
            bbox_values = self._bbox_to_csv_values(bbox)
            history.append({
                "text": text,
                "conf": float(conf),
                "mrs": float(max(0.0, min(1.0, mrs))),
                "frame_idx": int(frame_idx) if frame_idx is not None else -1,
                "bbox": list(bbox_values),
            })
            if len(history) > self.ocr_history_max:
                del history[:-self.ocr_history_max]
            temporal_id = int(string_group_id) if string_group_id is not None else int(track_id)
            string_state = self._append_string_observation(
                temporal_id,
                frame_idx=frame_idx,
                raw_text=raw_text or text,
                corrected_text=corrected_text or text,
                text=text,
                ocr_conf=float(conf),
                febam_reliability=float(mrs),
                source=source,
                source_weight=source_weight,
            )
            if string_state.get("state") == "COMMIT" and self._valid_final_plate_candidate(str(string_state.get("final_observed_plate", ""))):
                final_plate = str(string_state.get("final_observed_plate", ""))
                previous_final = self.final_plates.get(track_id, "")
                if previous_final != final_plate:
                    self.final_plates[track_id] = final_plate
                    self._append_final_ocr_csv_row(track_id, frame_idx, reason="string_febam_commit_update")
                return final_plate
        committed_plate = self._committed_plate_for_track(track_id)
        if committed_plate:
            self.final_plates[track_id] = committed_plate
            return committed_plate
        if not allow_legacy_vote:
            return ""
        final_plate = self._vote_ocr_history(track_id)
        previous_final = self.final_plates.get(track_id, "")
        if final_plate and previous_final != final_plate:
            self.final_plates[track_id] = final_plate
            self._append_final_ocr_csv_row(track_id, frame_idx, reason="final_voted_update")
        return final_plate

    def _finalize_track_ocr(self, track_id: int, frame_idx: int | None, reason: str = "track_ended") -> None:
        history = self.ocr_history.get(track_id, [])
        stats = self._ocr_history_stats(history)
        if int(stats.get("ocr_count", 0)) <= 0:
            return
        final_text = str(stats.get("final_voted_text", ""))
        string_state: dict[str, object] = {}
        if self.use_string_febam:
            string_state = self._rebuild_string_febam(track_id)
        committed_plate = self._committed_plate_for_track(track_id)
        if committed_plate:
            self.final_plates[track_id] = committed_plate
        elif string_state.get("state") == "COMMIT" and self._valid_final_plate_candidate(str(string_state.get("final_observed_plate", ""))):
            self.final_plates[track_id] = str(string_state.get("final_observed_plate", ""))
        elif final_text and self._valid_final_plate_candidate(final_text):
            self.final_plates[track_id] = final_text
        self._append_final_ocr_csv_row(track_id, frame_idx, reason=reason, stats=stats)
        self._log(
            "[MRS_FINAL] "
            f"track_id={int(track_id)} ocr_count={int(stats.get('ocr_count', 0))} "
            f"best_text={str(stats.get('best_text', ''))!r} best_conf={float(stats.get('best_conf', 0.0)):.3f} "
            f"final_voted_text={final_text!r} "
            f"mrs_max={float(stats.get('mrs_max', 0.0)):.3f} mrs_mean={float(stats.get('mrs_mean', 0.0)):.3f} "
            f"topk=[{stats.get('topk_ocr', '')}]"
        )
        if self.use_string_febam:
            self._log_string_febam_debug(track_id, string_state, reason=reason)

    def _finalize_dead_tracks(self, dead_track_ids: torch.Tensor, frame_idx: int | None) -> None:
        if not torch.is_tensor(dead_track_ids):
            return
        for track_id_t in dead_track_ids.detach().cpu().reshape(-1).tolist():
            track_id = int(track_id_t)
            self._finalize_track_ocr(track_id, frame_idx, reason="track_ended")
            self.ocr_history.pop(track_id, None)
            self.string_febam_observations.pop(track_id, None)
            self.string_febam_quarantine.pop(track_id, None)
            self.string_febam_nodes.pop(track_id, None)
            self.string_febam_segments.pop(track_id, None)
            self.string_febam_states.pop(track_id, None)

    def _finalize_all_tracks(self, frame_idx: int | None) -> None:
        for track_id in list(self.ocr_history.keys()):
            self._finalize_track_ocr(int(track_id), frame_idx, reason="pipeline_close")
            self.ocr_history.pop(int(track_id), None)
            self.string_febam_observations.pop(int(track_id), None)
            self.string_febam_nodes.pop(int(track_id), None)
            self.string_febam_segments.pop(int(track_id), None)
            self.string_febam_states.pop(int(track_id), None)

    def stage12_ocr_trigger(self, frame_u8: torch.Tensor, candidates: torch.Tensor, conf_ids: torch.Tensor, gate, skip_reason: str = "", gray_stretched: torch.Tensor | None = None, frame_idx: int | None = None):
        # OCR = text evidence generator; OCR voting is isolated per track_id only.
        crops = []
        self._last_gray_stretched_ocr_used = False
        self._last_ocr_text = ""
        self._last_ocr_conf = 0.0
        self._last_final_plate = ""

        candidate_count = int(candidates.shape[0]) if torch.is_tensor(candidates) and candidates.ndim >= 1 else 0
        candidate_track_ids = getattr(self, "_last_candidate_track_ids", torch.zeros((0,), device=self.device, dtype=torch.long))
        if torch.is_tensor(candidate_track_ids):
            track_id_count = int(((candidate_track_ids >= 0) & (candidate_track_ids < self.track_state.shape[0])).sum().item())
        else:
            track_id_count = len([track_id for track_id in candidate_track_ids if 0 <= int(track_id) < self.track_state.shape[0]])
        febam_confirmed_count = int(conf_ids.numel()) if torch.is_tensor(conf_ids) else 0

        if candidate_count == 0:
            self._append_ocr_debug_csv_row(
                frame_idx=frame_idx,
                ocr_attempt=0,
                ocr_skip_reason="no_candidates",
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
            )
            self._log_ocr_debug(
                frame_idx=frame_idx,
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
                attempted_count=0,
                skip_reason="no_candidates",
                gate=gate,
            )
            return crops

        if track_id_count == 0:
            self._append_ocr_debug_csv_row(
                frame_idx=frame_idx,
                ocr_attempt=0,
                ocr_skip_reason="no_track_ids",
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
            )
            self._log_ocr_debug(
                frame_idx=frame_idx,
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
                attempted_count=0,
                skip_reason="no_track_ids",
                gate=gate,
            )
            return crops

        if torch.is_tensor(candidate_track_ids):
            candidate_valid_track_ids = candidate_track_ids[(candidate_track_ids >= 0) & (candidate_track_ids < self.track_state.shape[0])]
        else:
            candidate_valid_track_ids = torch.tensor(
                [int(track_id) for track_id in candidate_track_ids if 0 <= int(track_id) < self.track_state.shape[0]],
                device=self.device,
                dtype=torch.long,
            )
        rank_by_track: dict[int, int] = {}
        if torch.is_tensor(candidate_track_ids):
            candidate_track_id_list = candidate_track_ids.detach().cpu().reshape(-1).tolist()
        else:
            candidate_track_id_list = list(candidate_track_ids)
        for rank_idx, track_id_raw in enumerate(candidate_track_id_list):
            track_id_rank = int(track_id_raw)
            if 0 <= track_id_rank < self.track_state.shape[0] and track_id_rank not in rank_by_track:
                rank_by_track[track_id_rank] = int(rank_idx)
        if candidate_valid_track_ids.numel() > 0:
            candidate_valid_track_ids = torch.unique_consecutive(candidate_valid_track_ids)

        confirmed_valid_track_ids = conf_ids[(conf_ids >= 0) & (conf_ids < self.track_state.shape[0])]
        confirmed_track_set = {int(track_id.item()) for track_id in confirmed_valid_track_ids}
        selected_track_ids: list[int] = []
        trigger_reason_by_track: dict[int, str] = {}
        near_by_track: dict[int, int] = {}
        sample_by_track: dict[int, int] = {}
        full_by_track: dict[int, int] = {}
        relax_scores: list[float] = []
        relax_energies: list[float] = []
        relax_memories: list[float] = []
        weak_samples_this_frame = 0

        if self.ocr_ignore_febam:
            for track_id_t in candidate_valid_track_ids:
                track_id = int(track_id_t.item())
                if track_id not in selected_track_ids:
                    selected_track_ids.append(track_id)
                    trigger_reason_by_track[track_id] = "ocr_ignore_febam"
                    near_by_track[track_id] = 0
                    sample_by_track[track_id] = 0
                    full_by_track[track_id] = 0
        else:
            for track_id in confirmed_track_set:
                selected_track_ids.append(track_id)
                trigger_reason_by_track[track_id] = "febam_confirmed"
                near_by_track[track_id] = 0
                sample_by_track[track_id] = 0
                full_by_track[track_id] = 1

            for track_id_t in candidate_valid_track_ids:
                track_id = int(track_id_t.item())
                bbox = self.track_state[track_id, :4]
                roi_w_relax = int(max(float((bbox[2] - bbox[0]).detach().item()), 0.0))
                roi_h_relax = int(max(float((bbox[3] - bbox[1]).detach().item()), 0.0))
                _, _, _, aspect_relax = self._ocr_crop_top_policy(int(float(bbox[1].detach().item())), roi_w_relax, roi_h_relax)
                febam_debug = self._febam_debug_values(track_id)
                febam_score = float(febam_debug.get("febam_score", 0.0) or 0.0)
                febam_energy = float(febam_debug.get("febam_energy", 0.0) or 0.0)
                febam_memory = float(febam_debug.get("febam_memory", 0.0) or 0.0)
                is_large = roi_w_relax >= 120 or roi_h_relax >= 35
                is_plate_like = 2.0 <= aspect_relax <= 7.0
                is_near_sample = (
                    is_large
                    and is_plate_like
                    and febam_score >= self.near_confirmed_sample_score_thr
                    and febam_energy >= self.near_confirmed_sample_energy_thr
                    and febam_memory >= self.near_confirmed_sample_memory_min
                )
                is_near = (
                    is_large
                    and is_plate_like
                    and febam_energy >= self.near_febam_energy_thr
                    and febam_score >= self.near_febam_score_thr
                )
                frame_value = int(frame_idx) if frame_idx is not None else -1
                hits = int(self.track_state[track_id, 11].detach().item())
                candidate_rank = int(rank_by_track.get(track_id, 9999))
                is_final_candidate = candidate_rank == 0
                weak_size_ok = roi_w_relax >= self.weak_plate_sample_min_w and roi_h_relax >= self.weak_plate_sample_min_h
                weak_aspect_ok = 2.0 <= aspect_relax <= 7.5
                repeated_track = hits >= 3 or febam_memory >= 1.0
                prior_ok = self._weak_plate_position_prior(bbox, int(gate[2]), int(gate[3]))
                similar_to_previous = self._bbox_similar_to_last_valid(track_id, bbox)
                is_periodic_sample = (
                    is_large
                    and is_plate_like
                    and febam_memory >= 1.0
                    and hits >= 2
                    and frame_value >= 0
                    and frame_value % int(getattr(self, "periodic_track_sample_interval", 5)) == 0
                )
                is_weak_plate_sample = (
                    febam_score >= self.weak_plate_sample_score_thr
                    and weak_size_ok
                    and weak_aspect_ok
                    and (candidate_rank <= 1 or is_final_candidate)
                    and (repeated_track or prior_ok or similar_to_previous)
                    and weak_samples_this_frame < int(getattr(self, "weak_plate_sample_max_per_frame", 2))
                )
                if is_near_sample or is_near or is_periodic_sample or is_weak_plate_sample:
                    relax_scores.append(febam_score)
                    relax_energies.append(febam_energy)
                    relax_memories.append(febam_memory)
                    if track_id not in selected_track_ids:
                        selected_track_ids.append(track_id)
                        if is_near_sample:
                            trigger_reason_by_track[track_id] = "near_confirmed_ocr_sample"
                            sample_by_track[track_id] = 1
                        elif is_near:
                            trigger_reason_by_track[track_id] = "near_confirmed_large_roi"
                            sample_by_track[track_id] = 0
                        elif is_periodic_sample:
                            trigger_reason_by_track[track_id] = "periodic_track_sample"
                            sample_by_track[track_id] = 1
                        else:
                            trigger_reason_by_track[track_id] = "weak_plate_ocr_sample"
                            sample_by_track[track_id] = 1
                            weak_samples_this_frame += 1
                        near_by_track[track_id] = 1 if is_near else 0
                        full_by_track[track_id] = 0

        near_count = sum(near_by_track.values())
        self._log(
            f"[FEBAM_RELAX] frame={int(frame_idx) if frame_idx is not None else -1} "
            f"candidates={candidate_count} confirmed={febam_confirmed_count} near={near_count} "
            f"score={max(relax_scores) if relax_scores else 0.0:.3f} "
            f"energy={max(relax_energies) if relax_energies else 0.0:.3f} "
            f"memory={max(relax_memories) if relax_memories else 0.0:.3f}"
        )

        if not selected_track_ids:
            if candidate_valid_track_ids.numel() > 0:
                # Architecture invariant: weak ghost plates may last only ~3 frames,
                # so an active frame keeps exactly one OCR observation slot instead
                # of dropping to zero when Detection FEBAM has not fully opened.
                if torch.is_tensor(candidate_track_ids):
                    policy_valid_mask = (candidate_track_ids >= 0) & (candidate_track_ids < self.track_state.shape[0])
                    policy_boxes = candidates[policy_valid_mask, :4]
                    policy_scores = candidates[policy_valid_mask, 4]
                    policy_track_ids = candidate_track_ids[policy_valid_mask]
                else:
                    policy_valid_indices = [idx for idx, tid in enumerate(candidate_track_id_list) if 0 <= int(tid) < self.track_state.shape[0]]
                    policy_boxes = candidates[policy_valid_indices, :4]
                    policy_scores = candidates[policy_valid_indices, 4]
                    policy_track_ids = torch.tensor([candidate_track_id_list[idx] for idx in policy_valid_indices], device=self.device, dtype=torch.long)
                policy_candidates = PipelineCandidateBatch(
                    boxes=policy_boxes,
                    scores=policy_scores,
                    track_ids=policy_track_ids,
                )
                policy_tracks = PipelineTrackBatch(
                    track_ids=candidate_valid_track_ids,
                    boxes=self.track_state[candidate_valid_track_ids, :4],
                )
                policy_requests = self.ocr_policy.select_requests(
                    frame_idx=int(frame_idx) if frame_idx is not None else -1,
                    candidates=policy_candidates,
                    tracks=policy_tracks,
                    text_states=self.string_febam_states,
                )
                fallback_track_id = int(policy_requests[0].track_id) if policy_requests and policy_requests[0].track_id is not None else int(candidate_valid_track_ids[0].item())
                selected_track_ids.append(fallback_track_id)
                trigger_reason_by_track[fallback_track_id] = "mandatory_frame_ocr_slot"
                near_by_track[fallback_track_id] = 0
                sample_by_track[fallback_track_id] = 1
                full_by_track[fallback_track_id] = 0
                self._log_ocr_debug(
                    frame_idx=frame_idx,
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    attempted_count=1,
                    skip_reason="mandatory_frame_ocr_slot",
                    gate=gate,
                    force=True,
                )
            else:
                self._append_ocr_debug_csv_row(
                    frame_idx=frame_idx,
                    ocr_attempt=0,
                    ocr_skip_reason="no_track_ids",
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    ocr_trigger_reason="no_track_ids",
                )
                self._log_ocr_debug(
                    frame_idx=frame_idx,
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    attempted_count=0,
                    skip_reason="no_track_ids",
                    gate=gate,
                )
                return crops

        h, w = frame_u8.shape[-2:]
        gx, gy, gate_w, gate_h = [int(v) for v in gate]
        roi_h = int(gray_stretched.shape[-2]) if gray_stretched is not None else int(gate_h)
        roi_w = int(gray_stretched.shape[-1]) if gray_stretched is not None else int(gate_w)
        priority_order = {"febam_confirmed": 0, "febam_confirmed_ocr": 0, "febam_last_valid_bbox": 0, "near_confirmed_ocr_sample": 1, "near_confirmed_large_roi": 1, "weak_plate_ocr_sample": 2, "periodic_track_sample": 2, "mandatory_frame_ocr_slot": 3}
        selected_track_ids = sorted(
            selected_track_ids,
            key=lambda tid: (
                priority_order.get(trigger_reason_by_track.get(int(tid), ""), 3),
                rank_by_track.get(int(tid), 9999),
                -float(self.track_state[int(tid), 8].detach().item()) if 0 <= int(tid) < self.track_state.shape[0] else 0.0,
            ),
        )
        valid_track_ids = torch.tensor(selected_track_ids[:1], device=self.device, dtype=torch.long)

        attempted_count = min(int(valid_track_ids.numel()), 1)
        self._log_ocr_debug(
            frame_idx=frame_idx,
            candidate_count=candidate_count,
            track_id_count=track_id_count,
            febam_confirmed_count=febam_confirmed_count,
            attempted_count=attempted_count,
            skip_reason="attempt_start",
            gate=gate,
            force=True,
        )

        for candidate_idx, track_id_t in enumerate(valid_track_ids):
            track_id = int(track_id_t.item())
            febam_debug = self._febam_debug_values(track_id)
            ocr_trigger_reason = trigger_reason_by_track.get(track_id, "febam_confirmed")
            bbox_resolved, bbox_fallback_reason, bbox_from_last_valid = self._resolve_ocr_bbox(track_id, frame_idx, roi_w, roi_h)
            if bbox_resolved is None:
                self._append_ocr_debug_csv_row(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    candidate_idx=candidate_idx,
                    ocr_attempt=0,
                    ocr_skip_reason=bbox_fallback_reason,
                    bbox=self.track_state[track_id, :4],
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    attempted_count=attempted_count,
                    ocr_candidate_rank=f"candidate {candidate_idx}",
                    ocr_trigger_reason=bbox_fallback_reason,
                    near_confirmed_large_roi=near_by_track.get(track_id, 0),
                    febam_full_confirmed=full_by_track.get(track_id, 0),
                    near_confirmed_ocr_sample=sample_by_track.get(track_id, 0),
                    ocr_sampling_reason=trigger_reason_by_track.get(track_id, ""),
                    **febam_debug,
                )
                continue
            bbox_roi = bbox_resolved
            if bbox_fallback_reason:
                ocr_trigger_reason = bbox_fallback_reason
            x1_l, y1_l, x2_l, y2_l = bbox_roi.int()
            mrs_value = self._mrs_from_febam_debug(febam_debug)
            if not int(full_by_track.get(track_id, 0)):
                mrs_value = min(mrs_value, max(0.0, 0.75 * float(febam_debug.get("febam_score", 0.0) or 0.0) + 0.25 * float(febam_debug.get("febam_energy", 0.0) or 0.0), 0.65))
            sampling_level = self._sampling_level_from_reason(ocr_trigger_reason)
            source_weight_value = self._ocr_sampling_source_weight(sampling_level) if sampling_level else 0.5
            near_confirmed_flag = int(near_by_track.get(track_id, 0))
            near_sample_flag = int(sample_by_track.get(track_id, 0))
            febam_full_flag = int(full_by_track.get(track_id, 0))
            ocr_candidate_rank = f"candidate {candidate_idx}"
            ocr_input_type = ""
            original_roi_w = int(max(int((x2_l - x1_l).item()), 0))
            original_roi_h = int(max(int((y2_l - y1_l).item()), 0))
            crop_policy, crop_expand_top, _, roi_aspect = self._ocr_crop_top_policy(
                int(y1_l.item()),
                original_roi_w,
                original_roi_h,
            )
            ocr_y1_original = int(y1_l.item())
            ocr_y1_expanded = int(y1_l.item())
            local_roi_w = original_roi_w
            local_roi_h = original_roi_h
            self._append_ocr_debug_csv_row(
                frame_idx=frame_idx,
                track_id=track_id,
                candidate_idx=candidate_idx,
                ocr_attempt=1,
                ocr_skip_reason=ocr_trigger_reason if ocr_trigger_reason in {"febam_confirmed", "near_confirmed_ocr_sample", "near_confirmed_large_roi", "periodic_track_sample", "febam_last_valid_bbox"} else "attempt_start",
                bbox=bbox_roi,
                roi_w=local_roi_w,
                roi_h=local_roi_h,
                candidate_count=candidate_count,
                track_id_count=track_id_count,
                febam_confirmed_count=febam_confirmed_count,
                attempted_count=attempted_count,
                ocr_candidate_rank=ocr_candidate_rank,
                ocr_crop_policy=crop_policy,
                ocr_crop_expand_top=crop_expand_top,
                ocr_y1_original=ocr_y1_original,
                ocr_y1_expanded=ocr_y1_expanded,
                roi_aspect=roi_aspect,
                ocr_expand_top=crop_expand_top,
                ocr_trigger_reason=ocr_trigger_reason,
                near_confirmed_large_roi=near_confirmed_flag,
                febam_full_confirmed=febam_full_flag,
                near_confirmed_ocr_sample=near_sample_flag,
                ocr_sampling_reason=trigger_reason_by_track.get(track_id, ocr_trigger_reason),
                ocr_sampling_level=sampling_level,
                source_weight=source_weight_value,
                **febam_debug,
            )

            if self.use_string_febam and self.string_febam_skip_after_commit:
                current_state = self.string_febam_states.get(track_id, {})
                is_committed = current_state.get("state") == "COMMIT"
                frame_value = int(frame_idx) if frame_idx is not None else -1
                if is_committed and frame_value >= 0 and frame_value % self.string_febam_skip_interval != 0:
                    self.string_febam_skipped_after_commit[track_id] = self.string_febam_skipped_after_commit.get(track_id, 0) + 1
                    self._append_ocr_debug_csv_row(
                        frame_idx=frame_idx,
                        track_id=track_id,
                        candidate_idx=candidate_idx,
                        ocr_attempt=0,
                        ocr_skip_reason="string_febam_skip_after_commit",
                        bbox=bbox_roi,
                        roi_w=local_roi_w,
                        roi_h=local_roi_h,
                        candidate_count=candidate_count,
                        track_id_count=track_id_count,
                        febam_confirmed_count=febam_confirmed_count,
                        attempted_count=attempted_count,
                        ocr_candidate_rank=ocr_candidate_rank,
                        ocr_trigger_reason=ocr_trigger_reason,
                        near_confirmed_large_roi=near_confirmed_flag,
                        febam_full_confirmed=febam_full_flag,
                        near_confirmed_ocr_sample=near_sample_flag,
                        ocr_sampling_reason=trigger_reason_by_track.get(track_id, ocr_trigger_reason),
                        ocr_skipped_after_commit=self.string_febam_skipped_after_commit[track_id],
                        **febam_debug,
                    )
                    continue

            top_ratio = 0.0
            left_ratio = getattr(self, "ocr_expand_left_ratio", 0.0)
            right_ratio = getattr(self, "ocr_expand_right_ratio", 0.0)
            bottom_ratio = getattr(self, "ocr_expand_bottom_ratio", 0.0)
            crop_u8 = None
            raw_debug_crop_u8 = None
            gray_stretched_debug_crop_u8 = None
            used_gray_stretched = False
            if (not self.scientific_g0) and self.use_gray_stretched_ocr and gray_stretched is not None:
                x1r = int(x1_l.clamp(0, max(roi_w - 1, 0)).item())
                x2r = int(x2_l.clamp(0, max(roi_w - 1, 0)).item())
                y1r = int(y1_l.clamp(0, max(roi_h - 1, 0)).item())
                y2r = int(y2_l.clamp(0, max(roi_h - 1, 0)).item())
                local_roi_w = original_roi_w
                local_roi_h = original_roi_h
                crop_policy, crop_expand_top, y1r_ocr, roi_aspect = self._ocr_crop_top_policy(
                    y1r,
                    local_roi_w,
                    local_roi_h,
                )
                layout_for_expand = self._estimate_plate_layout(roi_w=local_roi_w, roi_h=local_roi_h)
                top_ratio, left_ratio, right_ratio, bottom_ratio = self._ocr_crop_expand_ratios(local_roi_w, local_roi_h, layout_for_expand, sampling_level)
                crop_expand_top = int(round(float(local_roi_h) * top_ratio))
                crop_expand_left = int(round(float(local_roi_w) * left_ratio))
                crop_expand_right = int(round(float(local_roi_w) * right_ratio))
                crop_expand_bottom = int(round(float(local_roi_h) * bottom_ratio))
                y1r_ocr = max(0, y1r - crop_expand_top)
                x1r_ocr = max(0, x1r - crop_expand_left)
                x2r_ocr = min(max(roi_w - 1, 0), x2r + crop_expand_right)
                y2r_ocr = min(max(roi_h - 1, 0), y2r + crop_expand_bottom)
                ocr_y1_original = y1r
                ocr_y1_expanded = y1r_ocr
                if x2r_ocr > x1r_ocr and y2r_ocr > y1r_ocr:
                    crop = gray_stretched[0, :, y1r_ocr:y2r_ocr + 1, x1r_ocr:x2r_ocr + 1]
                    crop3 = crop.repeat(3, 1, 1)
                    crop_u8 = (crop3.clamp(0.0, 1.0) * 255.0).to(torch.uint8).detach()
                    gray_stretched_debug_crop_u8 = crop_u8
                    x1_full_dbg = x1r_ocr + gx
                    x2_full_dbg = x2r_ocr + gx
                    y1_full_dbg = y1r_ocr + gy
                    y2_full_dbg = y2r_ocr + gy
                    if (
                        x2_full_dbg > x1_full_dbg
                        and y2_full_dbg > y1_full_dbg
                        and x1_full_dbg >= 0
                        and y1_full_dbg >= 0
                        and x2_full_dbg < w
                        and y2_full_dbg < h
                    ):
                        raw_debug_crop_u8 = frame_u8[0, :, y1_full_dbg:y2_full_dbg + 1, x1_full_dbg:x2_full_dbg + 1].detach()
                    used_gray_stretched = True
                    ocr_input_type = "gray_stretched_roi"
                    self._last_gray_stretched_ocr_used = True

            if crop_u8 is None:
                x1 = x1_l + gx
                x2 = x2_l + gx
                y1 = y1_l + gy
                y2 = y2_l + gy
                local_roi_w = original_roi_w
                local_roi_h = original_roi_h
                crop_policy, crop_expand_top, y1_ocr, roi_aspect = self._ocr_crop_top_policy(
                    int(y1.item()),
                    local_roi_w,
                    local_roi_h,
                )
                layout_for_expand = self._estimate_plate_layout(roi_w=local_roi_w, roi_h=local_roi_h)
                top_ratio, left_ratio, right_ratio, bottom_ratio = self._ocr_crop_expand_ratios(local_roi_w, local_roi_h, layout_for_expand, sampling_level)
                crop_expand_top = int(round(float(local_roi_h) * top_ratio))
                crop_expand_left = int(round(float(local_roi_w) * left_ratio))
                crop_expand_right = int(round(float(local_roi_w) * right_ratio))
                crop_expand_bottom = int(round(float(local_roi_h) * bottom_ratio))
                x1_ocr = torch.clamp(x1 - crop_expand_left, 0, max(w - 1, 0))
                x2_ocr = torch.clamp(x2 + crop_expand_right, 0, max(w - 1, 0))
                y1_ocr = torch.clamp(y1 - crop_expand_top, 0, max(h - 1, 0))
                y2_ocr = torch.clamp(y2 + crop_expand_bottom, 0, max(h - 1, 0))
                ocr_y1_original = int(y1.item())
                ocr_y1_expanded = int(y1_ocr.item())
                if x2_ocr <= x1_ocr or y2_ocr <= y1_ocr or x1_ocr < 0 or y1_ocr < 0 or x2_ocr >= w or y2_ocr >= h:
                    self._append_ocr_debug_csv_row(
                        frame_idx=frame_idx,
                        track_id=track_id,
                        candidate_idx=candidate_idx,
                        ocr_attempt=1,
                        ocr_skip_reason="roi_invalid",
                        bbox=bbox_roi,
                        roi_w=local_roi_w,
                        roi_h=local_roi_h,
                        candidate_count=candidate_count,
                        track_id_count=track_id_count,
                        febam_confirmed_count=febam_confirmed_count,
                        gray_stretched_ocr_used=used_gray_stretched,
                        attempted_count=attempted_count,
                        ocr_input_type=ocr_input_type or "raw_roi",
                        ocr_candidate_rank=ocr_candidate_rank,
                        ocr_crop_policy=crop_policy,
                        ocr_crop_expand_top=crop_expand_top,
                        ocr_y1_original=ocr_y1_original,
                        ocr_y1_expanded=ocr_y1_expanded,
                        roi_aspect=roi_aspect,
                        ocr_expand_top=crop_expand_top,
                        ocr_trigger_reason=ocr_trigger_reason,
                        near_confirmed_large_roi=near_confirmed_flag,
                        febam_full_confirmed=febam_full_flag,
                        near_confirmed_ocr_sample=near_sample_flag,
                        ocr_sampling_reason=trigger_reason_by_track.get(track_id, ocr_trigger_reason),
                        **febam_debug,
                    )
                    continue
                crop_u8 = frame_u8[0, :, int(y1_ocr.item()):int(y2_ocr.item()) + 1, int(x1_ocr.item()):int(x2_ocr.item()) + 1].detach()
                raw_debug_crop_u8 = crop_u8
                ocr_input_type = "raw_roi"

            if self.scientific_g0 and self.use_gray_stretched_ocr and gray_stretched is not None:
                # GPU-only validated auxiliary: retain the transform-suppressed
                # gray view as shadow evidence without replacing canonical input.
                x1r = int(x1_l.clamp(0, max(roi_w - 1, 0)).item())
                x2r = int(x2_l.clamp(0, max(roi_w - 1, 0)).item())
                y1r = int(y1_l.clamp(0, max(roi_h - 1, 0)).item())
                y2r = int(y2_l.clamp(0, max(roi_h - 1, 0)).item())
                x1r = max(0, x1r - int(round(float(local_roi_w) * left_ratio)))
                x2r = min(max(roi_w - 1, 0), x2r + int(round(float(local_roi_w) * right_ratio)))
                y1r = max(0, y1r - int(round(float(local_roi_h) * top_ratio)))
                y2r = min(max(roi_h - 1, 0), y2r + int(round(float(local_roi_h) * bottom_ratio)))
                if x2r > x1r and y2r > y1r:
                    gray_shadow = gray_stretched[0, :, y1r:y2r + 1, x1r:x2r + 1]
                    gray_stretched_debug_crop_u8 = (
                        gray_shadow.repeat(3, 1, 1).clamp(0.0, 1.0) * 255.0
                    ).to(torch.uint8).detach()

            if local_roi_w < 2 or local_roi_h < 2:
                self._append_ocr_debug_csv_row(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    candidate_idx=candidate_idx,
                    ocr_attempt=1,
                    ocr_skip_reason="roi_too_small",
                    bbox=bbox_roi,
                    roi_w=local_roi_w,
                    roi_h=local_roi_h,
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    gray_stretched_ocr_used=used_gray_stretched,
                    attempted_count=attempted_count,
                    ocr_input_type=ocr_input_type,
                    ocr_candidate_rank=ocr_candidate_rank,
                    ocr_crop_policy=crop_policy,
                    ocr_crop_expand_top=crop_expand_top,
                    ocr_crop_expand_top_ratio=top_ratio,
                    ocr_crop_expand_left_ratio=left_ratio,
                    ocr_crop_expand_right_ratio=right_ratio,
                    ocr_crop_expand_bottom_ratio=bottom_ratio,
                    ocr_y1_original=ocr_y1_original,
                    ocr_y1_expanded=ocr_y1_expanded,
                    roi_aspect=roi_aspect,
                    ocr_expand_top=crop_expand_top,
                    ocr_trigger_reason=ocr_trigger_reason,
                    near_confirmed_large_roi=near_confirmed_flag,
                    febam_full_confirmed=febam_full_flag,
                    near_confirmed_ocr_sample=near_sample_flag,
                    ocr_sampling_reason=trigger_reason_by_track.get(track_id, ocr_trigger_reason),
                    **febam_debug,
                )
                continue

            self._log(
                f"[OCR_CROP] frame={int(frame_idx) if frame_idx is not None else -1} "
                f"track={track_id} policy={crop_policy} w={local_roi_w} h={local_roi_h} "
                f"ar={roi_aspect:.3f} expand_top={crop_expand_top} "
                f"y1={ocr_y1_original} y1_ocr={ocr_y1_expanded}"
            )
            debug_images_enabled = bool(self.ocr_save_debug_crops and self._ocr_debug_image_enabled())
            raw_color_crop_u8 = raw_debug_crop_u8 if raw_debug_crop_u8 is not None else crop_u8
            raw_roi_debug_path = ""
            if debug_images_enabled:
                raw_roi_debug_path = self._save_ocr_roi_debug(frame_idx, track_id, raw_color_crop_u8, suffix="raw")
            gray_stretched_roi_debug_path = ""
            raw_gray_roi_debug_path = ""
            mild_contrast_roi_debug_path = ""
            if debug_images_enabled and gray_stretched_debug_crop_u8 is not None:
                gray_stretched_roi_debug_path = self._save_ocr_roi_debug(frame_idx, track_id, gray_stretched_debug_crop_u8, suffix="gray_stretched")

            raw_gray_crop_u8 = self._raw_gray_ocr_crop_cuda(raw_color_crop_u8)
            if debug_images_enabled:
                raw_gray_roi_debug_path = self._save_ocr_roi_debug(frame_idx, track_id, raw_gray_crop_u8, suffix="raw_gray")

            gray_stretched_saved = 1 if gray_stretched_roi_debug_path else 0
            # Frozen LOVO geometry contract: modify only bbox coordinates.
            # Never rotate/warp the pixels before OCR in geometry-preserving mode.
            enable_angle_search = crop_policy.startswith("expand_top") and not bool(
                getattr(self, "fastplate_preserve_crop_geometry", False)
            ) and not self.scientific_g0
            success_text = ""
            success_conf = 0.0
            success_raw_text = ""
            success_corrected_text = ""
            success_source = "raw"
            success_source_weight = source_weight_value
            final_rotated_roi_debug_path = ""
            plate_layout = self._estimate_plate_layout(roi_w=local_roi_w, roi_h=local_roi_h)
            split_fallback_image_rgb = None
            best_result: tuple[str, str, str, float, str, int, str, str, str, str, dict[str, float | int], float] | None = None
            best_common_kwargs: dict[str, object] = {}
            ocr_early_stop_reason = ""
            ocr_small_roi_candidate = int((local_roi_w <= 120 or local_roi_h <= 35) and (2.0 <= float(roi_aspect) <= 7.5))
            previous_ocr_failed = self._recent_ocr_failure_count(track_id)
            committed_plate_locked = bool(self._committed_plate_for_track(track_id))
            selected_variant_names = self._select_ocr_variants_for_candidate(
                roi_w=local_roi_w,
                roi_h=local_roi_h,
                roi_aspect=roi_aspect,
                febam_score=float(febam_debug.get("febam_score", 0.0) or 0.0),
                febam_confirmed=bool(febam_full_flag),
                near_confirmed_large_roi=bool(near_confirmed_flag),
                ocr_trigger_reason=ocr_trigger_reason,
                previous_ocr_failed=previous_ocr_failed,
                committed_plate_locked=committed_plate_locked,
                ocr_small_roi=bool(ocr_small_roi_candidate),
                plate_layout=plate_layout,
            )
            if self.scientific_g0:
                # Scientific input contract: auxiliary transforms may be logged,
                # but only the expanded source-pixel crop reaches the recognizer.
                selected_variant_names = ["raw_expanded"]
            variant_lookup: dict[str, tuple[int, torch.Tensor, str]] = {
                "raw_expanded": (0, raw_color_crop_u8, raw_roi_debug_path),
                "gray_stretched_norm": (1, gray_stretched_debug_crop_u8 if gray_stretched_debug_crop_u8 is not None else raw_color_crop_u8, gray_stretched_roi_debug_path),
                "small_upscaled_sharpened": (2, raw_color_crop_u8, raw_roi_debug_path),
                "raw_upscaled": (3, raw_color_crop_u8, raw_roi_debug_path),
                "mild_clahe_upscaled": (4, raw_color_crop_u8, mild_contrast_roi_debug_path),
                "weak_unsharp_upscaled": (5, raw_color_crop_u8, raw_roi_debug_path),
                "binary_fallback": (6, raw_color_crop_u8, raw_roi_debug_path),
            }
            variant_items: list[tuple[str, int, torch.Tensor, str]] = [
                (name, *variant_lookup[name]) for name in selected_variant_names if name in variant_lookup
            ]
            variant_order_text = ">".join(item[0] for item in variant_items)
            readtext_fallback_count = 0
            variants_executed_count = 0
            fastplate_variant_group_results: list[dict[str, object]] = []
            # Final WiSE custom ONNX can consume CUDA crops directly.  A
            # shared-async Fusion bridge uses the same CUDA queue/session and
            # must not force normal frame ROI crops back through NumPy.  Only
            # the legacy separate Fusion recognizer retains that compatibility
            # restriction.
            gpu_resident_fastplate = bool(
                self.fastplate_async
                and self.ocr_backend == "fastplate"
                and (
                    self.fastplate_custom_onnx
                    or self.dual_branch_ocr
                    or (
                        self.fastplate_tensor_runner_mode == "active"
                        and self.fastplate_direct_ort
                    )
                )
                and not debug_images_enabled
                and (
                    self.trial020_fusion_bridge is None
                    or self.event_roi_fusion_ocr_worker_mode == "shared_async"
                )
            )

            if committed_plate_locked:
                self._append_ocr_debug_csv_row(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    candidate_idx=candidate_idx,
                    ocr_attempt=0,
                    ocr_skip_reason="committed_plate_locked",
                    bbox=bbox_roi,
                    roi_w=local_roi_w,
                    roi_h=local_roi_h,
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    attempted_count=attempted_count,
                    plate_layout=plate_layout,
                    committed_plate_locked=1,
                    final_output_plate=self._committed_plate_for_track(track_id),
                    ocr_variants_planned=variant_order_text,
                    ocr_variants_executed_count=0,
                    ocr_variant_skip_reason="committed_plate_locked",
                    **febam_debug,
                )
                continue

            if self.ocr_backend == "none":
                self._append_ocr_debug_csv_row(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    candidate_idx=candidate_idx,
                    ocr_attempt=0,
                    ocr_skip_reason="ocr_backend_none",
                    bbox=bbox_roi,
                    roi_w=local_roi_w,
                    roi_h=local_roi_h,
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    attempted_count=attempted_count,
                    plate_layout=plate_layout,
                    ocr_engine="none",
                    easyocr_mode="not_used",
                    ocr_variants_planned=variant_order_text,
                    ocr_variants_executed_count=0,
                    ocr_variant_skip_reason="ocr_backend_none",
                    **febam_debug,
                )
                continue

            for variant_name, fallback_rank, variant_crop_u8, variant_debug_path in variant_items:
                if variant_name in {"weak_unsharp_upscaled", "small_upscaled_sharpened", "binary_fallback"}:
                    (
                        preprocessed_crop_u8,
                        upscaled_crop_u8,
                        sharpened_crop_u8,
                        ocr_upscale_applied,
                        ocr_sharpen_applied,
                        ocr_upscale_factor,
                        ocr_sharpen_amount,
                    ) = self._small_roi_ocr_preprocess_cuda(variant_crop_u8, local_roi_w, local_roi_h, roi_aspect)
                    if debug_images_enabled and upscaled_crop_u8 is not None:
                        self._save_ocr_roi_debug(frame_idx, track_id, upscaled_crop_u8, suffix="upscaled")
                    if debug_images_enabled and sharpened_crop_u8 is not None:
                        self._save_ocr_roi_debug(frame_idx, track_id, sharpened_crop_u8, suffix="sharpened")
                else:
                    preprocessed_crop_u8 = variant_crop_u8.to(device=self.device, dtype=torch.uint8, non_blocking=True)
                    ocr_upscale_applied = 0
                    ocr_sharpen_applied = 0
                    ocr_upscale_factor = 1.0
                    ocr_sharpen_amount = 0.0

                ocr_small_roi = int((local_roi_w <= 120 or local_roi_h <= 35) and (2.0 <= float(roi_aspect) <= 7.5))
                rectified_crop_u8, ocr_angle, ocr_angle_score, ocr_rotation_applied, ocr_gpu_rectified = self._rectify_ocr_crop_cuda(
                    preprocessed_crop_u8,
                    enable_angle_search,
                )
                rotated_roi_debug_path = ""
                if debug_images_enabled:
                    rotated_roi_debug_path = self._save_ocr_roi_debug(frame_idx, track_id, rectified_crop_u8, suffix=f"{variant_name}_norm")
                final_rotated_roi_debug_path = rotated_roi_debug_path
                if gpu_resident_fastplate:
                    # The worker performs the identical resize/RGB/norm01
                    # preprocessing on CUDA and binds the result to ORT.
                    image_rgb = None
                    norm_stats = {
                        "w": int(self.fastplate_custom_input_width),
                        "h": int(self.fastplate_custom_input_height),
                        "scale": 0.0,
                        "pad_x": 0,
                        "pad_y": 0,
                        "aspect_preserved": 0,
                    }
                else:
                    crops.append(rectified_crop_u8.detach().cpu())
                    image_rgb_raw = self._ocr_crop_to_rgb_np(rectified_crop_u8)
                    image_rgb, norm_stats = self._normalize_ocr_input_np(image_rgb_raw)
                    if variant_name == "mild_clahe_upscaled":
                        image_rgb = self._apply_mild_clahe_rgb(image_rgb)
                    split_fallback_image_rgb = image_rgb
                if debug_images_enabled:
                    norm_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    norm_path = self.ocr_roi_debug_dir / f"frame{int(frame_idx) if frame_idx is not None else -1:05d}_track{int(track_id):04d}_{variant_name}_input.png"
                    self.ocr_roi_debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(norm_path), norm_bgr)
                    if not variant_debug_path:
                        variant_debug_path = str(norm_path)

                self._log(
                    f"[OCR_GPU_RECTIFY] frame={int(frame_idx) if frame_idx is not None else -1} "
                    f"track={track_id} w={local_roi_w} h={local_roi_h} policy={crop_policy} "
                    f"variant={variant_name} best_angle={ocr_angle:.3f} score={ocr_angle_score:.3f} "
                    f"rotation_applied={int(ocr_rotation_applied)} threshold=8.0"
                )

                common_kwargs = dict(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    candidate_idx=candidate_idx,
                    bbox=bbox_roi,
                    roi_w=local_roi_w,
                    roi_h=local_roi_h,
                    candidate_count=candidate_count,
                    track_id_count=track_id_count,
                    febam_confirmed_count=febam_confirmed_count,
                    gray_stretched_ocr_used=used_gray_stretched and variant_name == "gray_stretched_norm",
                    attempted_count=attempted_count,
                    ocr_input_type=f"{variant_name}_roi",
                    ocr_candidate_rank=ocr_candidate_rank,
                    ocr_crop_policy=crop_policy,
                    ocr_crop_expand_top=crop_expand_top,
                    ocr_crop_expand_top_ratio=top_ratio,
                    ocr_crop_expand_left_ratio=left_ratio,
                    ocr_crop_expand_right_ratio=right_ratio,
                    ocr_crop_expand_bottom_ratio=bottom_ratio,
                    ocr_y1_original=ocr_y1_original,
                    ocr_y1_expanded=ocr_y1_expanded,
                    roi_aspect=roi_aspect,
                    ocr_expand_top=crop_expand_top,
                    ocr_angle=ocr_angle,
                    ocr_rotation_applied=ocr_rotation_applied,
                    ocr_angle_score=ocr_angle_score,
                    ocr_gpu_rectified=ocr_gpu_rectified,
                    ocr_trigger_reason=ocr_trigger_reason,
                    near_confirmed_large_roi=near_confirmed_flag,
                    febam_full_confirmed=febam_full_flag,
                    near_confirmed_ocr_sample=near_sample_flag,
                    ocr_sampling_reason=trigger_reason_by_track.get(track_id, ocr_trigger_reason),
                    ocr_sampling_level=sampling_level,
                    source_weight=(
                        0.35
                        if self.dual_branch_ocr
                        and variant_name == "gray_stretched_norm"
                        and bool(getattr(self, "dual_branch_gray_secondary", False))
                        else source_weight_value
                    ),
                    ocr_saved_raw=raw_roi_debug_path,
                    ocr_saved_gray_stretched=gray_stretched_roi_debug_path,
                    ocr_saved_rotated=rotated_roi_debug_path,
                    ocr_input_variant=variant_name,
                    ocr_fallback_rank=fallback_rank,
                    ocr_fallback_used=int(fallback_rank > 0),
                    ocr_rotation_threshold=8.0,
                    ocr_reader_langs="+".join(self.ocr_reader_langs),
                    ocr_small_roi=ocr_small_roi,
                    ocr_upscale_applied=ocr_upscale_applied,
                    ocr_upscale_factor=ocr_upscale_factor,
                    ocr_sharpen_applied=ocr_sharpen_applied,
                    ocr_sharpen_amount=ocr_sharpen_amount,
                    ocr_variant_order=variant_order_text,
                    ocr_variant_count=len(variant_items),
                    ocr_variant_tier=self._ocr_variant_tier(variant_name),
                    ocr_variant_executed=1,
                    ocr_variants_planned=variant_order_text,
                    ocr_variants_executed_count=variants_executed_count + 1,
                    ocr_readtext_fallback_count=readtext_fallback_count,
                    ocr_variant_skip_reason="",
                    ocr_norm_w=norm_stats["w"],
                    ocr_norm_h=norm_stats["h"],
                    ocr_norm_scale=norm_stats["scale"],
                    ocr_norm_pad_x=norm_stats["pad_x"],
                    ocr_norm_pad_y=norm_stats["pad_y"],
                    ocr_norm_aspect_preserved=norm_stats["aspect_preserved"],
                    ocr_allowlist_sweep_disabled=1,
                    ocr_selected_final_candidate_only=1,
                    plate_layout=plate_layout,
                    mrs=mrs_value,
                    **febam_debug,
                )
                if self.trial020_fusion_bridge is not None:
                    try:
                        string_values = self._string_febam_csv_values(int(track_id)) if self.use_string_febam else {}
                        segment_id = string_values.get("segment_id", "") or "default"
                        bridge_crop_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                        self._submit_trial020_roi_crop_to_bridge(
                            frame_idx=frame_idx,
                            track_id=track_id,
                            candidate_idx=candidate_idx,
                            crop_bgr=bridge_crop_bgr,
                            text="",
                            conf=0.0,
                            source="ocr_roi_image_only",
                            source_weight=0.25,
                            segment_id=segment_id,
                            event_id=common_kwargs.get("event_id", ""),
                            febam_score=float(febam_debug.get("febam_score", 0.0) or 0.0),
                            febam_energy=float(febam_debug.get("febam_energy", 0.0) or 0.0),
                            febam_memory=float(febam_debug.get("febam_memory", 0.0) or 0.0),
                            variant=variant_name,
                            meta={
                                **common_kwargs,
                                "image_only_fusion": "1",
                                "bridge_input_stage": "ocr_roi_generated",
                                "ocr_input_variant": variant_name,
                                "roi_w": int(bridge_crop_bgr.shape[1]),
                                "roi_h": int(bridge_crop_bgr.shape[0]),
                                "source_level": str(common_kwargs.get("ocr_sampling_level", "") or common_kwargs.get("source_level", "")),
                                "febam_confirmed": bool(febam_full_flag),
                                "near_confirmed": bool(near_confirmed_flag or near_sample_flag),
                                "weak_plate_sample": bool(common_kwargs.get("weak_plate_sample", False)),
                                "candidate_score": float(common_kwargs.get("candidate_score", 0.0) or common_kwargs.get("mrs", 0.0) or 0.0),
                                "det_score": float(common_kwargs.get("det_score", 0.0) or common_kwargs.get("mrs", 0.0) or 0.0),
                            },
                            bbox=bbox_roi,
                            roi_w=local_roi_w,
                            roi_h=local_roi_h,
                            candidate_count=candidate_count,
                            track_id_count=track_id_count,
                            febam_confirmed_count=febam_confirmed_count,
                        )
                    except Exception:
                        self._log_exception("trial020_event_roi_image_only_exception")

                mode_policy = self._easyocr_mode_policy_for_candidate(
                    roi_w=local_roi_w,
                    roi_h=local_roi_h,
                    roi_aspect=roi_aspect,
                    plate_layout=plate_layout,
                    ocr_small_roi=bool(ocr_small_roi),
                    previous_ocr_failed=previous_ocr_failed,
                )
                fallback_priority = variant_name in {"raw_expanded", "gray_stretched_norm", "small_upscaled_sharpened"}
                allow_readtext_fallback = (
                    mode_policy == "recognize_first"
                    and fallback_priority
                    and readtext_fallback_count < self.ocr_max_readtext_fallbacks_per_candidate
                    and self.easyocr_readtext_fallback
                )
                ocr_engine_calls: list[str] = []
                if self.ocr_backend in {"easyocr", "both"}:
                    ocr_engine_calls.append("easyocr")
                if self.ocr_backend in {"fastplate", "both"} and variant_name in {"raw_expanded", "gray_stretched_norm"}:
                    ocr_engine_calls.append("fast_plate_ocr")
                if self.ocr_backend in {"fastplate", "both"} and not ocr_engine_calls and variant_name not in {"raw_expanded", "gray_stretched_norm"}:
                    debug_common_kwargs = self._sanitize_ocr_debug_csv_row_kwargs(common_kwargs)
                    self._append_ocr_debug_csv_row(
                        frame_idx=frame_idx,
                        track_id=track_id,
                        candidate_idx=candidate_idx,
                        bbox=bbox_roi,
                        roi_w=local_roi_w,
                        roi_h=local_roi_h,
                        candidate_count=candidate_count,
                        track_id_count=track_id_count,
                        febam_confirmed_count=febam_confirmed_count,
                        gray_stretched_ocr_used=used_gray_stretched and variant_name == "gray_stretched_norm",
                        attempted_count=attempted_count,
                        ocr_attempt=0,
                        ocr_skip_reason=f"fastplate_variant_skipped_{variant_name}",
                        text="",
                        conf="",
                        ocr_allowlist_mode="disabled",
                        ocr_pattern_type="empty",
                        ocr_postprocess_applied=0,
                        ocr_postprocess_reason="fastplate_raw_gray_only",
                        ocr_plate_text_final="",
                        ocr_engine="fast_plate_ocr",
                        easyocr_mode="not_used",
                        easyocr_batch_size="",
                        ocr_variant_tier=self._ocr_variant_tier(variant_name),
                        ocr_variant_executed=0,
                        ocr_variants_planned=variant_order_text,
                        ocr_variants_executed_count=variants_executed_count,
                        ocr_readtext_fallback_count=readtext_fallback_count,
                        ocr_variant_skip_reason="fastplate_raw_gray_only",
                        **debug_common_kwargs,
                    )
                    continue

                for ocr_engine_name in ocr_engine_calls:
                    variants_executed_count += 1
                    if ocr_engine_name == "fast_plate_ocr" and self.fastplate_async:
                        async_meta = {
                            "frame_idx": frame_idx,
                            "track_id": track_id,
                            "candidate_idx": candidate_idx,
                            # GPU runtime uses a video-scoped stream event as
                            # the coarse container. It is deliberately not a
                            # vehicle identity and does not fall back to the
                            # tracker; GPUFinalVehicleGrouper performs the
                            # conservative vehicle split/re-link inside it.
                            "video_id": str(self.runtime_video_id or ""),
                            "event_id": f"{self.runtime_video_id}:gpu_stream_event",
                            "segment_id": "gpu_online_segment",
                            "canonical_event_source": "gpu_runtime_stream_scope",
                            "ocr_saved_raw": str(raw_roi_debug_path or ""),
                            "ocr_saved_gray_stretched": str(gray_stretched_roi_debug_path or ""),
                            "ocr_saved_rotated": str(rotated_roi_debug_path or ""),
                            "bbox_roi": self._bbox_to_csv_values(bbox_roi),
                            "roi_w": local_roi_w,
                            "roi_h": local_roi_h,
                            "candidate_count": candidate_count,
                            "track_id_count": track_id_count,
                            "febam_confirmed_count": febam_confirmed_count,
                            "gray_stretched_ocr_used": used_gray_stretched and variant_name == "gray_stretched_norm",
                            "attempted_count": attempted_count,
                            "variant_name": variant_name,
                            "variant": variant_name,
                            "source_level": sampling_level,
                            "ocr_sampling_reason": trigger_reason_by_track.get(track_id, ocr_trigger_reason),
                            "trigger_reason": trigger_reason_by_track.get(track_id, ocr_trigger_reason),
                            "febam_confirmed": febam_debug.get("febam_confirmed", 0),
                            "near_confirmed_large_roi": near_confirmed_flag,
                            "near_confirmed_ocr_sample": near_sample_flag,
                            "weak_plate_sample": int(sampling_level == "weak_plate_sample"),
                            "plate_layout": plate_layout,
                            "mrs": mrs_value,
                            "source_weight": (
                                0.35
                                if variant_name == "gray_stretched_norm"
                                and bool(getattr(self, "dual_branch_gray_secondary", False))
                                else (1.0 if self.dual_branch_ocr and variant_name == "raw_expanded" else source_weight_value)
                            ),
                            "ocr_input_color_contract": "RGB",
                            "ocr_input_layout": "CHW",
                            "ocr_input_dtype": "torch.uint8",
                            "ocr_input_range": "0..255",
                            "ocr_input_channel_swap_count": 0,
                            "ocr_input_source": "nvdec_cuda_crop",
                            "ocr_variants_planned": variant_order_text,
                            "ocr_variants_executed_count": variants_executed_count,
                            **febam_debug,
                        }
                        ocr_input = rectified_crop_u8 if gpu_resident_fastplate else image_rgb
                        async_enqueued = self._enqueue_fastplate_async(ocr_input, async_meta)
                        if not async_enqueued:
                            debug_common_kwargs = self._sanitize_ocr_debug_csv_row_kwargs(common_kwargs)
                            self._append_ocr_debug_csv_row(
                                frame_idx=frame_idx,
                                track_id=track_id,
                                candidate_idx=candidate_idx,
                                bbox=bbox_roi,
                                roi_w=local_roi_w,
                                roi_h=local_roi_h,
                                candidate_count=candidate_count,
                                track_id_count=track_id_count,
                                febam_confirmed_count=febam_confirmed_count,
                                gray_stretched_ocr_used=used_gray_stretched and variant_name == "gray_stretched_norm",
                                attempted_count=attempted_count,
                                ocr_attempt=0,
                                ocr_skip_reason=f"fastplate_async_enqueue_failed_{variant_name}",
                                text="",
                                conf="",
                                ocr_engine="fast_plate_ocr",
                                easyocr_mode="async_batch",
                                fastplate_model=self.fastplate_model,
                                fastplate_device=self.fastplate_device,
                                fastplate_batch_size=self.fastplate_batch_size,
                                ocr_variant_name=variant_name,
                                ocr_variant_skip_reason="fastplate_async_enqueue_failed",
                                **debug_common_kwargs,
                            )
                        elif self.trial020_fusion_bridge is not None:
                            bridge_crop_bgr = self._ocr_crop_to_bgr(
                                rectified_crop_u8 if gpu_resident_fastplate else image_rgb
                            )
                            if bridge_crop_bgr is not None and getattr(bridge_crop_bgr, "size", 0) > 0:
                                string_values = self._string_febam_csv_values(int(track_id)) if self.use_string_febam else {}
                                segment_id = string_values.get("segment_id", "") or "default"
                                self._submit_trial020_roi_crop_to_bridge(
                                    frame_idx=frame_idx,
                                    track_id=track_id,
                                    candidate_idx=candidate_idx,
                                    crop_bgr=bridge_crop_bgr,
                                    text="",
                                    conf=0.0,
                                    source="ocr_roi_image_only_async",
                                    source_weight=0.25,
                                    segment_id=segment_id,
                                    event_id=async_meta.get("event_id", ""),
                                    febam_score=float(febam_debug.get("febam_score", 0.0) or 0.0),
                                    febam_energy=float(febam_debug.get("febam_energy", 0.0) or 0.0),
                                    febam_memory=float(febam_debug.get("febam_memory", 0.0) or 0.0),
                                    variant=variant_name,
                                    meta={
                                        **async_meta,
                                        "image_only_fusion": "1",
                                        "bridge_input_stage": "ocr_async_enqueue",
                                        "ocr_input_variant": variant_name,
                                    },
                                    bbox=bbox_roi,
                                    roi_w=local_roi_w,
                                    roi_h=local_roi_h,
                                    candidate_count=candidate_count,
                                    track_id_count=track_id_count,
                                    febam_confirmed_count=febam_confirmed_count,
                                )
                        continue
                    try:
                        if ocr_engine_name == "easyocr":
                            raw_text, normalized_text, plate_text_final, conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed = self._run_easyocr_once(
                                image_rgb,
                                None,
                                variant_name=variant_name,
                                mode_policy=mode_policy,
                                allow_readtext_fallback=allow_readtext_fallback,
                            )
                            result_meta = dict(getattr(self, "_last_easyocr_result_meta", {}) or {})
                        else:
                            raw_text, normalized_text, plate_text_final, conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed = self._run_fastplate_once(
                                image_rgb,
                                variant_name=variant_name,
                            )
                            result_meta = dict(getattr(self, "_last_fastplate_result_meta", {}) or {})
                    except Exception as exc:
                        debug_common_kwargs = self._sanitize_ocr_debug_csv_row_kwargs(common_kwargs)
                        self._append_ocr_debug_csv_row(
                            frame_idx=frame_idx,
                            track_id=track_id,
                            candidate_idx=candidate_idx,
                            bbox=bbox_roi,
                            roi_w=local_roi_w,
                            roi_h=local_roi_h,
                            candidate_count=candidate_count,
                            track_id_count=track_id_count,
                            febam_confirmed_count=febam_confirmed_count,
                            gray_stretched_ocr_used=used_gray_stretched and variant_name == "gray_stretched_norm",
                            attempted_count=attempted_count,
                            ocr_attempt=1,
                            ocr_skip_reason=f"{ocr_engine_name}_exception_{variant_name}:{type(exc).__name__}",
                            text="",
                            conf="",
                            ocr_allowlist_mode="disabled",
                            ocr_normalized_text="",
                            ocr_pattern_type="empty",
                            ocr_postprocess_applied=0,
                            ocr_postprocess_reason="",
                            ocr_plate_text_final="",
                            ocr_korean_middle_raw="",
                            ocr_korean_middle_fixed="",
                            ocr_engine=ocr_engine_name,
                            easyocr_mode="not_used" if ocr_engine_name != "easyocr" else mode_policy,
                            easyocr_batch_size="" if ocr_engine_name != "easyocr" else self.easyocr_batch_size,
                            easyocr_workers="" if ocr_engine_name != "easyocr" else self.easyocr_workers,
                            easyocr_recognize_only="" if ocr_engine_name != "easyocr" else int(self.easyocr_recognize_only),
                            fastplate_model=self.fastplate_model if ocr_engine_name == "fast_plate_ocr" else "",
                            fastplate_device=self.fastplate_device if ocr_engine_name == "fast_plate_ocr" else "",
                            fastplate_batch_size=self.fastplate_batch_size if ocr_engine_name == "fast_plate_ocr" else "",
                            ocr_variant_tier=self._ocr_variant_tier(variant_name),
                            ocr_variant_executed=1,
                            ocr_variants_planned=variant_order_text,
                            ocr_variants_executed_count=variants_executed_count,
                            ocr_readtext_fallback_count=readtext_fallback_count,
                            **debug_common_kwargs,
                        )
                        self._log(
                            f"[OCR] frame={int(frame_idx) if frame_idx is not None else -1} "
                            f"track={track_id} engine={ocr_engine_name} variant={variant_name} text='' conf=0.000 error={type(exc).__name__}"
                        )
                        continue

                    self._update_plate_skeleton_memory(track_id, raw_text, frame_idx, source=ocr_engine_name)
                    context_anchor = self._find_plate_skeleton_context_anchor(track_id, raw_text, frame_idx) if ocr_engine_name == "fast_plate_ocr" else None
                    variant_group_key = ""
                    variant_group_anchor_text = ""
                    variant_group_merge_reason = ""
                    variant_group_selected_skeleton = ""
                    preferred_skeleton = None
                    if ocr_engine_name == "fast_plate_ocr":
                        fastplate_variant_group_results.append({
                            "raw_text": raw_text,
                            "text": raw_text,
                            "variant_name": variant_name,
                            "conf": conf,
                        })
                        merge_info = self._merge_fastplate_variant_group(track_id, frame_idx, candidate_idx, fastplate_variant_group_results)
                        variant_group_key = str(merge_info.get("group_key", "") or "")
                        variant_group_anchor_text = str(merge_info.get("anchor_text", "") or "")
                        variant_group_merge_reason = str(merge_info.get("merge_reason", "") or "")
                        variant_group_selected_skeleton = str(merge_info.get("selected_skeleton", "") or "")
                        if merge_info.get("context_anchor"):
                            context_anchor = dict(merge_info.get("context_anchor") or {})
                        preferred_skeleton = merge_info.get("preferred_skeleton")
                        if preferred_skeleton:
                            merge_info["preferred_candidates"] = self._grammar_decode_text(
                                plate_text_final or normalized_text,
                                conf=conf,
                                layout=plate_layout,
                                previous_consensus=self._committed_plate_for_track(track_id) or self.final_plates.get(track_id, ""),
                                source="fast_plate_ocr",
                                context_anchor=context_anchor,
                                preferred_skeleton=preferred_skeleton,
                            )
                    skeleton_anchor_used = int(bool(context_anchor))
                    skeleton_anchor_pattern = str((context_anchor or {}).get("pattern", "") or "")
                    skeleton_anchor_text = str((context_anchor or {}).get("text", "") or "")
                    skeleton_anchor_support = (context_anchor or {}).get("support", "")
                    grammar_candidates = self._grammar_decode_text(
                        plate_text_final or normalized_text,
                        conf=conf,
                        layout=plate_layout,
                        previous_consensus=self._committed_plate_for_track(track_id) or self.final_plates.get(track_id, ""),
                        source=variant_name if ocr_engine_name == "easyocr" else "fast_plate_ocr",
                        context_anchor=context_anchor,
                        preferred_skeleton=preferred_skeleton,
                    )
                    if ocr_engine_name == "fast_plate_ocr" and grammar_candidates and not self._valid_final_plate_candidate(plate_text_final):
                        grammar_best = grammar_candidates[0]
                        plate_text_final = grammar_best.text
                        conf = min(1.0, max(0.0, float(grammar_best.score or 0.0)))
                        pattern_type = grammar_best.pattern
                        post_applied = 1
                        post_reason = grammar_best.reason
                        normalized_text = plate_text_final
                    grammar_candidate_score = grammar_candidates[0].score if grammar_candidates else (float(conf) + self._ocr_variant_bonus(variant_name) if self._valid_final_plate_candidate(plate_text_final) else 0.0)
                    if ocr_engine_name == "fast_plate_ocr" and grammar_candidate_score:
                        grammar_candidate_score -= 0.005
                    grammar_candidate_reason = grammar_candidates[0].reason if grammar_candidates else post_reason
                    grammar_candidate_top3 = "|".join(f"{c.text}:{c.score:.3f}:{c.reason}" for c in grammar_candidates[:3]) if grammar_candidates else (plate_text_final if plate_text_final else "")
                    grammar_best_text = grammar_candidates[0].text if grammar_candidates else plate_text_final
                    grammar_best_pattern = grammar_candidates[0].pattern if grammar_candidates else ("DDDKDDDD" if len(plate_text_final) == 8 else ("DDKDDDD" if len(plate_text_final) == 7 else ""))
                    candidate_source_weight = source_weight_value
                    if ocr_engine_name == "fast_plate_ocr" and plate_text_final:
                        candidate_source_weight = min(
                            float(source_weight_value),
                            self._fastplate_candidate_source_weight(grammar_candidate_reason, skeleton_anchor_used),
                        )
                    if ocr_engine_name == "easyocr" and bool(result_meta.get("fallback_used", False)):
                        readtext_fallback_count += 1
                    early_reason = self._ocr_variant_early_stop_reason(
                        plate_text_final or normalized_text,
                        conf,
                        grammar_best_text=grammar_candidates[0].text if grammar_candidates else plate_text_final,
                        grammar_best_score=grammar_candidate_score,
                        previous_consensus=self._committed_plate_for_track(track_id) or self.final_plates.get(track_id, ""),
                    )
                    if self.ocr_backend == "both":
                        early_reason = ""
                    if ocr_engine_name == "fast_plate_ocr":
                        row_source = str(result_meta.get("source", "fast_plate_ocr") or "fast_plate_ocr")
                    else:
                        row_source = "korean_center_reocr" if post_reason.startswith("middle_crop_") else ("fallback" if fallback_rank > 0 else "raw")
                    row_kwargs = {
                        "ocr_allowlist_mode": "disabled",
                        "ocr_normalized_text": normalized_text,
                        "ocr_pattern_type": pattern_type,
                        "ocr_postprocess_applied": post_applied,
                        "ocr_postprocess_reason": post_reason,
                        "ocr_plate_text_final": plate_text_final,
                        "ocr_korean_middle_raw": middle_raw,
                        "ocr_korean_middle_fixed": middle_fixed,
                        "ocr_early_stop_reason": early_reason,
                        "raw_text": raw_text,
                        "raw_conf": conf,
                        "corrected_text": plate_text_final,
                        "corrected_conf": conf,
                        "plate_layout": plate_layout,
                        "grammar_pattern": grammar_best_pattern,
                        "grammar_candidates_top3": grammar_candidate_top3,
                        "grammar_best_text": grammar_best_text,
                        "grammar_best_score": grammar_candidate_score,
                        "grammar_best_reason": grammar_candidate_reason,
                        "grammar_valid_final": int(self._valid_final_plate_candidate(plate_text_final)),
                        "grammar_decoder_used": int(bool(grammar_candidates)),
                        "split_ocr_used": 0,
                        "split_fallback_used": 0,
                        "final_candidate_source": "fastplate_custom_onnx_decoded_candidate" if row_source == "fastplate_custom_onnx" else ("fast_plate_ocr_decoded_candidate" if ocr_engine_name == "fast_plate_ocr" else variant_name),
                        "final_candidate_score": grammar_candidate_score,
                        "final_candidate_reason": grammar_candidate_reason,
                        "final_output_plate": plate_text_final,
                        "ocr_engine": ocr_engine_name,
                        "easyocr_mode": str(result_meta.get("source", mode_policy)) if ocr_engine_name == "easyocr" else "not_used",
                        "easyocr_batch_size": self.easyocr_batch_size if ocr_engine_name == "easyocr" else "",
                        "easyocr_workers": self.easyocr_workers if ocr_engine_name == "easyocr" else "",
                        "easyocr_recognize_only": int(self.easyocr_recognize_only) if ocr_engine_name == "easyocr" else "",
                        "easyocr_readtext_fallback_used": int(bool(result_meta.get("fallback_used", False))) if ocr_engine_name == "easyocr" else "",
                        "fastplate_model": self.fastplate_model if ocr_engine_name == "fast_plate_ocr" else "",
                        "fastplate_device": self.fastplate_device if ocr_engine_name == "fast_plate_ocr" else "",
                        "fastplate_batch_size": self.fastplate_batch_size if ocr_engine_name == "fast_plate_ocr" else "",
                        "ocr_variant_name": variant_name,
                        "ocr_raw_result_short": str(result_meta.get("raw_result", ""))[:500],
                        "normalized_plate_candidate": normalized_text or plate_text_final,
                        "korean_plate_valid": int(self._valid_final_plate_candidate(plate_text_final)),
                        "ocr_variant_tier": self._ocr_variant_tier(variant_name),
                        "ocr_variant_executed": 1,
                        "ocr_variants_planned": variant_order_text,
                        "ocr_variants_executed_count": variants_executed_count,
                        "ocr_readtext_fallback_count": readtext_fallback_count,
                        "ocr_variant_skip_reason": "",
                        "source": row_source,
                        "skeleton_anchor_used": skeleton_anchor_used,
                        "skeleton_anchor_pattern": skeleton_anchor_pattern,
                        "skeleton_anchor_text": skeleton_anchor_text,
                        "skeleton_anchor_support": skeleton_anchor_support,
                        "variant_group_key": variant_group_key,
                        "variant_group_anchor_text": variant_group_anchor_text,
                        "variant_group_merge_reason": variant_group_merge_reason,
                        "variant_group_selected_skeleton": variant_group_selected_skeleton,
                    }
                    common_extra = {k: v for k, v in common_kwargs.items() if k not in {"frame_idx", "track_id", "bbox"}}
                    for duplicate_key in {
                        "plate_layout",
                        "grammar_best_text",
                        "grammar_best_score",
                        "grammar_best_reason",
                        "final_output_plate",
                        "committed_plate_locked",
                        "ocr_variant_tier",
                        "ocr_variant_executed",
                        "ocr_variants_planned",
                        "ocr_variants_executed_count",
                        "ocr_readtext_fallback_count",
                        "ocr_variant_skip_reason",
                    }:
                        common_extra.pop(duplicate_key, None)
                    row_kwargs.update(common_extra)
                    if ocr_engine_name == "fast_plate_ocr" and plate_text_final:
                        row_kwargs["source_weight"] = candidate_source_weight
                    if plate_text_final and ocr_engine_name == "fast_plate_ocr":
                        try:
                            middle_crop_bgr = self._ocr_crop_to_bgr(raw_color_crop_u8)
                        except Exception:
                            middle_crop_bgr = None
                        self._enqueue_middle_slot_upl(
                            middle_crop_bgr,
                            ocr_text=normalized_text or raw_text,
                            ocr_conf=conf,
                            source="fast_plate_ocr",
                            meta={
                                **common_kwargs,
                                **row_kwargs,
                                "frame_idx": frame_idx,
                                "track_id": track_id,
                                "candidate_idx": candidate_idx,
                                "bbox_roi": bbox_roi,
                                "bbox": bbox_roi,
                                "mrs": mrs_value,
                                "variant_group_key": variant_group_key,
                                "source_weight": self.middle_slot_source_weight,
                                "variant_name": variant_name,
                            },
                        )
                    if plate_text_final:
                        if best_result is None or grammar_candidate_score > best_result[11]:
                            best_result = (raw_text, normalized_text, plate_text_final, conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed, row_kwargs["source"], norm_stats, grammar_candidate_score)
                            best_common_kwargs = row_kwargs
                        csv_row_kwargs = self._sanitize_ocr_csv_row_kwargs(row_kwargs)
                        self._append_ocr_csv_row(
                            track_id,
                            frame_idx,
                            raw_text,
                            conf,
                            bbox_roi,
                            candidate_idx=candidate_idx,
                            roi_w=local_roi_w,
                            roi_h=local_roi_h,
                            candidate_count=candidate_count,
                            track_id_count=track_id_count,
                            febam_confirmed_count=febam_confirmed_count,
                            gray_stretched_ocr_used=used_gray_stretched and variant_name == "gray_stretched_norm",
                            attempted_count=attempted_count,
                            **csv_row_kwargs,
                        )
                        self._log(
                            f"[OCR] frame={int(frame_idx) if frame_idx is not None else -1} "
                            f"track={track_id} rank={candidate_idx} variant={variant_name} allowlist=disabled "
                            f"pattern={pattern_type} febam_score={febam_debug.get('febam_score', '')} "
                            f"confirmed={febam_debug.get('febam_confirmed', 0)} roi={local_roi_w}x{local_roi_h} "
                            f"norm={int(norm_stats['w'])}x{int(norm_stats['h'])} result={plate_text_final!r}"
                        )
                        if early_reason:
                            ocr_early_stop_reason = early_reason
                            success_text = plate_text_final
                            success_conf = conf
                            success_raw_text = raw_text
                            success_corrected_text = plate_text_final
                            success_source = row_kwargs["source"]
                            success_source_weight = float(row_kwargs.get("source_weight", source_weight_value) or source_weight_value)
                            break
                    else:
                        debug_common_kwargs = self._sanitize_ocr_debug_csv_row_kwargs(common_kwargs)
                        self._append_ocr_debug_csv_row(
                            frame_idx=frame_idx,
                            track_id=track_id,
                            candidate_idx=candidate_idx,
                            bbox=bbox_roi,
                            roi_w=local_roi_w,
                            roi_h=local_roi_h,
                            candidate_count=candidate_count,
                            track_id_count=track_id_count,
                            febam_confirmed_count=febam_confirmed_count,
                            gray_stretched_ocr_used=used_gray_stretched and variant_name == "gray_stretched_norm",
                            attempted_count=attempted_count,
                            ocr_attempt=1,
                            ocr_skip_reason=f"empty_text_{variant_name}",
                            text=raw_text,
                            conf="" if conf == 0.0 else conf,
                            ocr_allowlist_mode="disabled",
                            ocr_normalized_text=normalized_text,
                            ocr_pattern_type=pattern_type,
                            ocr_postprocess_applied=post_applied,
                            ocr_postprocess_reason=post_reason,
                            ocr_plate_text_final=plate_text_final,
                            ocr_korean_middle_raw=middle_raw,
                            ocr_korean_middle_fixed=middle_fixed,
                            ocr_early_stop_reason="",
                            ocr_engine=ocr_engine_name,
                            easyocr_mode=str(result_meta.get("source", mode_policy)) if ocr_engine_name == "easyocr" else "not_used",
                            easyocr_batch_size=self.easyocr_batch_size if ocr_engine_name == "easyocr" else "",
                            easyocr_workers=self.easyocr_workers if ocr_engine_name == "easyocr" else "",
                            easyocr_recognize_only=int(self.easyocr_recognize_only) if ocr_engine_name == "easyocr" else "",
                            easyocr_readtext_fallback_used=int(bool(result_meta.get("fallback_used", False))) if ocr_engine_name == "easyocr" else "",
                            fastplate_model=self.fastplate_model if ocr_engine_name == "fast_plate_ocr" else "",
                            fastplate_device=self.fastplate_device if ocr_engine_name == "fast_plate_ocr" else "",
                            fastplate_batch_size=self.fastplate_batch_size if ocr_engine_name == "fast_plate_ocr" else "",
                            ocr_variant_name=variant_name,
                            ocr_raw_result_short=str(result_meta.get("raw_result", ""))[:500],
                            normalized_plate_candidate=normalized_text,
                            **debug_common_kwargs,
                        )
                if success_text:
                    break
            if not success_text and best_result is not None:
                raw_text, normalized_text, plate_text_final, conf, pattern_type, post_applied, post_reason, middle_raw, middle_fixed, source_name, _, _candidate_score = best_result
                success_text = plate_text_final
                success_conf = conf
                success_raw_text = raw_text
                success_corrected_text = plate_text_final
                success_source = source_name
                success_source_weight = float(best_common_kwargs.get("source_weight", source_weight_value) or source_weight_value)

            if not success_text and split_fallback_image_rgb is not None:
                try:
                    split_text, split_conf, split_debug = self._run_split_ocr_fallback(
                        split_fallback_image_rgb,
                        plate_layout,
                        previous_consensus=self._committed_plate_for_track(track_id) or self.final_plates.get(track_id, ""),
                    )
                except Exception as exc:
                    split_text, split_conf, split_debug = "", 0.0, {"split_ocr_used": 0, "grammar_best_reason": f"split_exception:{type(exc).__name__}"}
                if split_text and self._valid_final_plate_candidate(split_text):
                    success_text = split_text
                    success_conf = split_conf
                    success_raw_text = ""
                    success_corrected_text = split_text
                    success_source = "split_ocr"
                    success_source_weight = source_weight_value
                    split_row_kwargs = {
                        **common_kwargs,
                        **split_debug,
                        "source": "split_ocr",
                        "corrected_text": split_text,
                        "corrected_conf": split_conf,
                        "raw_text": split_text,
                        "raw_conf": split_conf,
                        "final_candidate_source": "split_ocr_decoded_candidate",
                        "final_candidate_score": split_conf,
                        "final_candidate_reason": str(split_debug.get("grammar_best_reason", "split_ocr_valid")),
                        "final_output_plate": split_text,
                    }
                    split_row_kwargs = self._sanitize_ocr_csv_row_kwargs(split_row_kwargs)
                    self._append_ocr_csv_row(
                        track_id,
                        frame_idx,
                        split_text,
                        split_conf,
                        bbox_roi,
                        candidate_idx=candidate_idx,
                        roi_w=local_roi_w,
                        roi_h=local_roi_h,
                        candidate_count=candidate_count,
                        track_id_count=track_id_count,
                        febam_confirmed_count=febam_confirmed_count,
                        gray_stretched_ocr_used=used_gray_stretched and variant_name == "gray_stretched_norm",
                        attempted_count=attempted_count,
                        **split_row_kwargs,
                    )

            if success_text:
                self._last_final_plate = self._append_ocr_history(
                    track_id,
                    success_text,
                    success_conf,
                    frame_idx,
                    bbox_roi,
                    mrs_value,
                    raw_text=success_raw_text,
                    corrected_text=success_corrected_text,
                    source=success_source,
                    source_weight=success_source_weight,
                )
                if self.trial020_fusion_bridge is not None:
                    string_values = self._string_febam_csv_values(int(track_id)) if self.use_string_febam else {}
                    segment_id = string_values.get("segment_id", "") or "default"
                    crop_bgr = self._ocr_crop_to_bgr(raw_color_crop_u8)
                    success_variant = str(
                        best_common_kwargs.get("ocr_input_variant", "")
                        or best_common_kwargs.get("ocr_variant_name", "")
                        or "raw_expanded"
                    )
                    self._submit_trial020_roi_crop_to_bridge(
                        frame_idx=frame_idx,
                        track_id=track_id,
                        candidate_idx=candidate_idx,
                        crop_bgr=crop_bgr,
                        text=success_corrected_text or success_text,
                        conf=float(success_conf),
                        source=success_source,
                        source_weight=float(success_source_weight),
                        segment_id=segment_id,
                        event_id=best_common_kwargs.get("event_id", ""),
                        febam_score=float(febam_debug.get("febam_score", 0.0) or 0.0),
                        febam_energy=float(febam_debug.get("febam_energy", 0.0) or 0.0),
                        febam_memory=float(febam_debug.get("febam_memory", 0.0) or 0.0),
                        variant=success_variant,
                        meta={
                            **best_common_kwargs,
                            "image_only_fusion": "0",
                            "bridge_input_stage": "ocr_text_update",
                            "roi_path": raw_roi_debug_path,
                            "crop_policy": crop_policy,
                            "ocr_input_variant": success_variant,
                        },
                        bbox=bbox_roi,
                        roi_w=local_roi_w,
                        roi_h=local_roi_h,
                        candidate_count=candidate_count,
                        track_id_count=track_id_count,
                        febam_confirmed_count=febam_confirmed_count,
                    )
            else:
                self._log(
                    f"[OCR] frame={int(frame_idx) if frame_idx is not None else -1} "
                    f"track={track_id} rank={candidate_idx} roi={local_roi_w}x{local_roi_h} "
                    f"result='' raw_roi={raw_roi_debug_path} rotated_roi={final_rotated_roi_debug_path}"
                )
        return crops

    def _make_debug_overlay_frame(self, frame_rgb: np.ndarray, gate, candidates: np.ndarray, confirmed_ids: np.ndarray, frame_idx: int, sec: float, ocr_triggered: bool, candidate_track_ids: np.ndarray | None = None) -> np.ndarray:
        out = frame_rgb.copy()
        h, w = out.shape[:2]
        gx, gy, gw, gh = [int(v) for v in gate]
        gate_x2 = min(w - 1, gx + gw)
        gate_y2 = min(h - 1, gy + gh)
        cv2.rectangle(out, (gx, gy), (gate_x2, gate_y2), (255, 255, 0), 2)
        for i, c in enumerate(candidates):
            x1, y1, x2, y2 = [int(v) for v in c[:4]]
            x1 += gx; x2 += gx; y1 += gy; y2 += gy
            track_id = int(candidate_track_ids[i]) if candidate_track_ids is not None and i < len(candidate_track_ids) else i
            color = (0,255,0)
            thick = 2
            febam_text = ""
            if np.any(confirmed_ids == track_id):
                color = (255,0,0)
                thick = 3
                febam_text = " | febam stable"
            cv2.rectangle(out, (max(0,x1),max(0,y1)), (min(w-1,x2),min(h-1,y2)), color, thick)
            score = float(c[4]) if len(c) > 4 else 0.0
            cv2.putText(out, f"ID {track_id} | score {score:.2f}{febam_text}", (max(0,x1), max(14,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        txt = f"idx={frame_idx} sec={sec:.2f} cand={len(candidates)} conf={len(confirmed_ids)} ocr={'Y' if ocr_triggered else 'N'}"
        cv2.putText(out, txt, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return out

    def _event_overlay_candidate_data(self, item: dict, max_overlay_candidates: int = 50) -> tuple[np.ndarray, np.ndarray, str]:
        frame = item.get("frame_rgb")
        if frame is None:
            frame_h, frame_w = 0, 0
        else:
            frame_h, frame_w = frame.shape[:2]
        gx, gy, _, _ = [int(v) for v in item.get("gate", (0, 0, frame_w, frame_h))]

        boxes_full = np.asarray(item.get("event_overlay_candidate_boxes_full", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
        coord_mode = "full"
        if boxes_full.size == 0:
            boxes_roi = np.asarray(item.get("event_overlay_candidate_boxes_roi", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
            if boxes_roi.size == 0:
                boxes_roi = np.asarray(item.get("candidates", np.zeros((0, 6), dtype=np.float32)), dtype=np.float32).reshape(-1, 6)[:, :4]
            if boxes_roi.size > 0:
                boxes_full = boxes_roi + np.array([gx, gy, gx, gy], dtype=np.float32)
                coord_mode = "roi->full"

        if boxes_full.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), "none"

        if frame_w > 0 and frame_h > 0:
            boxes_full[:, 0::2] = np.clip(boxes_full[:, 0::2], 0.0, float(frame_w - 1))
            boxes_full[:, 1::2] = np.clip(boxes_full[:, 1::2], 0.0, float(frame_h - 1))
        valid = (boxes_full[:, 2] > boxes_full[:, 0]) & (boxes_full[:, 3] > boxes_full[:, 1])
        boxes_full = boxes_full[valid]

        scores = np.asarray(item.get("event_overlay_candidate_scores", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if scores.size < valid.size:
            padded = np.zeros((valid.size,), dtype=np.float32)
            padded[:scores.size] = scores
            scores = padded
        scores = scores[valid] if valid.size > 0 else np.zeros((0,), dtype=np.float32)

        limit = min(max_overlay_candidates, boxes_full.shape[0])
        return boxes_full[:limit], scores[:limit], coord_mode

    def _draw_event_candidate_overlay(self, overlay_rgb: np.ndarray, item: dict, max_overlay_candidates: int = 50) -> np.ndarray:
        out = overlay_rgb.copy()
        frame_h, frame_w = out.shape[:2]
        boxes_full, scores, coord_mode = self._event_overlay_candidate_data(item, max_overlay_candidates=max_overlay_candidates)
        item["_event_overlay_candidates_count"] = int(boxes_full.shape[0])
        item["_event_overlay_coord_mode"] = coord_mode

        yellow = (255, 255, 0)
        green = (0, 255, 0)
        black = (0, 0, 0)
        white = (255, 255, 255)

        for i, box in enumerate(boxes_full):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(out, (max(0, x1), max(0, y1)), (min(frame_w - 1, x2), min(frame_h - 1, y2)), yellow, 1)
            if i < 10:
                score = float(scores[i]) if i < scores.size else 0.0
                label = f"#{i + 1} {score:.2f}"
                tx = max(0, min(frame_w - 1, x1))
                ty = max(14, min(frame_h - 1, y1 - 4))
                cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, black, 2)
                cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, yellow, 1)

        selected_full = np.asarray(item.get("event_overlay_selected_boxes_full", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
        if selected_full.size == 0:
            gx, gy, _, _ = [int(v) for v in item.get("gate", (0, 0, frame_w, frame_h))]
            selected_roi = np.asarray(item.get("candidates", np.zeros((0, 6), dtype=np.float32)), dtype=np.float32).reshape(-1, 6)[:, :4]
            if selected_roi.size > 0:
                selected_full = selected_roi[:1] + np.array([gx, gy, gx, gy], dtype=np.float32)
        for box in selected_full[:1]:
            if box.size < 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(out, (max(0, x1), max(0, y1)), (min(frame_w - 1, x2), min(frame_h - 1, y2)), green, 3)
            cv2.putText(out, "final", (max(0, x1), max(16, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, black, 2)
            cv2.putText(out, "final", (max(0, x1), max(16, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1)

        info = f"event cand={boxes_full.shape[0]} coord={coord_mode}"
        cv2.putText(out, info, (14, max(48, min(frame_h - 10, 48))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 2)
        cv2.putText(out, info, (14, max(48, min(frame_h - 10, 48))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        return out


    def _squeeze_debug_map(self, arr) -> np.ndarray | None:
        if arr is None:
            return None
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _map_to_u8(self, arr) -> np.ndarray | None:
        arr = self._squeeze_debug_map(arr)
        if arr is None:
            return None
        max_v = float(np.max(arr)) if arr.size else 0.0
        if max_v <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)

    def _make_direct_crop_debug_overlay(self, frame_rgb: np.ndarray, item: dict) -> np.ndarray:
        # DEBUG ONLY: visualizes fuzzy_temporal -> vehicle ROI -> plate search ROI -> top-k -> MLP final selection.
        out = frame_rgb.copy()
        frame_h, frame_w = out.shape[:2]
        rx0, ry0, rw, rh = [int(v) for v in item.get("gate", (0, 0, frame_w, frame_h))]
        rx1 = min(frame_w - 1, rx0 + rw)
        ry1 = min(frame_h - 1, ry0 + rh)

        score_map = self._squeeze_debug_map(item.get("plate_score_map", item.get("direct_crop_score_map")))
        if score_map is not None and rx1 > rx0 and ry1 > ry0:
            score_u8 = self._map_to_u8(score_map)
            if score_u8 is not None:
                roi_w = max(1, rx1 - rx0)
                roi_h = max(1, ry1 - ry0)
                score_u8 = cv2.resize(score_u8, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                heat_bgr = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
                heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
                roi = out[ry0:ry1, rx0:rx1]
                out[ry0:ry1, rx0:rx1] = cv2.addWeighted(roi, 0.70, heat_rgb, 0.30, 0.0)

        red = (255, 0, 0)
        orange = (255, 128, 0)
        yellow = (255, 255, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        purple = (180, 0, 255)
        white = (255, 255, 255)
        black = (0, 0, 0)

        def _arr(name: str, shape_tail: int) -> np.ndarray:
            return np.asarray(item.get(name, np.zeros((0, shape_tail), dtype=np.float32))).reshape(-1, shape_tail)

        def _vec(name: str) -> np.ndarray:
            return np.asarray(item.get(name, np.zeros((0,), dtype=np.float32))).reshape(-1)

        def _fmt_box(boxes: np.ndarray) -> str:
            return str(boxes[0].round(1).tolist()) if boxes.size >= 4 else "[]"

        def _draw_box(box: np.ndarray, color, label: str, thick: int = 2) -> None:
            if box.size < 4 or box[2] <= box[0] or box[3] <= box[1]:
                return
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(out, (max(0, x1), max(0, y1)), (min(frame_w - 1, x2), min(frame_h - 1, y2)), color, thick)
            if label:
                cv2.putText(out, label, (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        vehicle_temporal_full = _arr("vehicle_temporal_box_full", 4)
        vehicle_gate_full = _arr("vehicle_gate_box_full", 4)
        vehicle_full = _arr("vehicle_box_full", 4)
        vehicle_roi = _arr("vehicle_box_roi", 4)
        search_full = _arr("plate_search_box_full", 4)
        search_roi = _arr("plate_search_box_roi", 4)
        candidate_full = _arr("plate_candidate_boxes_full", 4)
        candidate_roi = _arr("plate_candidate_boxes_roi", 4)
        selected_full = _arr("plate_selected_box_full", 4)
        selected_roi = _arr("plate_selected_box_roi", 4)
        stroke_merge_pair_roi = _arr("stroke_merge_pair_boxes", 4)
        crop_search = _arr("crop_box_search", 4)
        peak_search = _arr("peak_xy_search", 2)
        peak_roi_all = _arr("peak_xy_roi", 2)
        gt_roi_arr = _arr("direct_crop_gt_box_roi", 4)
        iou_arr = _vec("direct_crop_iou_pred_gt")
        final_scores = _vec("candidate_final_score")
        peak_scores = _vec("candidate_peak_score")
        region_mean = _vec("candidate_region_mean")
        temporal_mean = _vec("candidate_temporal_mean")
        region_contrast = _vec("candidate_region_contrast")
        lower_prior = _vec("candidate_lower_prior")
        diagonal_prior = _vec("candidate_diagonal_prior")
        candidate_width = _vec("candidate_width")
        candidate_height = _vec("candidate_height")
        candidate_aspect = _vec("candidate_aspect")
        candidate_area_ratio = _vec("candidate_area_ratio")
        rectangularity_score = _vec("candidate_rectangularity_score")
        mlp_score = _vec("candidate_mlp_score")
        mlp_fallback_values = _vec("candidate_mlp_fallback")
        gray_included_values = _vec("candidate_gray_included")
        rule_score_values = _vec("candidate_rule_score")
        contrast_score_values = _vec("candidate_contrast_score")
        aspect_score_values = _vec("candidate_aspect_score")
        dog_score_values = _vec("candidate_dog_score")
        dog_rect_score_values = _vec("candidate_dog_rect_score")
        uniform_rect_score_values = _vec("candidate_uniform_rect_score")
        dog_alpha_weight_values = _vec("candidate_dog_alpha_weight")
        fuzzy_score_values = _vec("candidate_fuzzy_score")
        dark_score_values = _vec("candidate_dark_score")
        dog_mean_values = _vec("dog_mean")
        dog_max_values = _vec("dog_max")
        dog_selectivity_values = _vec("dog_selectivity")
        dog_alpha_values = _vec("dog_alpha")
        dog_gain_values = _vec("dog_gain")
        uniform_sigma_sq_values = _vec("uniform_sigma_sq")
        uniform_weight_values = _vec("uniform_weight")
        uniform_mu_mean_values = _vec("uniform_mu_mean")
        uniform_mu_max_values = _vec("uniform_mu_max")
        candidate_count_values = _vec("candidate_count")
        top_candidate_score_values = _vec("top_candidate_score")
        dark_body_100pct_mask_values = _vec("dark_body_100pct_mask_mode")
        gt_uniform_rect_score_values = _vec("gt_uniform_rect_score")
        gt_uniform_candidate_generated_values = _vec("gt_uniform_candidate_generated")
        gt_dog_rect_score_values = _vec("gt_dog_rect_score")
        gt_dog_alpha_weight_values = _vec("gt_dog_alpha_weight")
        iou_gt_values = _vec("candidate_iou_gt")
        center_error_gt_values = _vec("candidate_center_error_gt")
        nearest_gt_rank_values = _vec("candidate_nearest_gt_rank")
        best_scale_idx = np.asarray(item.get("best_scale_idx", np.zeros((0,), dtype=np.int64))).reshape(-1)
        best_aspect_idx = np.asarray(item.get("best_aspect_idx", np.zeros((0,), dtype=np.int64))).reshape(-1)
        vehicle_area = _vec("vehicle_area")
        vehicle_center = _arr("vehicle_center", 2)
        vehicle_roi_mean = _vec("vehicle_roi_mean")
        vehicle_roi_max = _vec("vehicle_roi_max")
        search_wh = _arr("plate_search_wh", 2)
        rect_scale_stats = np.asarray(item.get("rect_scale_stats", np.zeros((0, 8), dtype=np.float32))).reshape(-1, 8)
        rect_gt_center_scores = np.asarray(item.get("rect_gt_center_scores", np.zeros((0, 4), dtype=np.float32))).reshape(-1, 4)

        _draw_box(vehicle_full[0], red, "70pct gate ROI", 3) if vehicle_full.size >= 4 else None

        def _score_color(score_val: float):
            score_val = float(np.clip(score_val, 0.0, 1.0))
            if score_val < 0.50:
                t = score_val / 0.50
                return (int(255 * t), int(255 * t), 255)
            t = (score_val - 0.50) / 0.50
            return (255, int(255 * (1.0 - t)), 0)

        for i, box in enumerate(candidate_full):
            score_val = float(final_scores[i]) if i < final_scores.size else 0.0
            color = _score_color(score_val)
            thick = 2 if score_val >= 0.75 else 1
            label = f"#{i + 1} {score_val:.2f}" if i < 20 else ""
            _draw_box(box, color, label, thick)
            cx = int(round(float((box[0] + box[2]) * 0.5)))
            cy = int(round(float((box[1] + box[3]) * 0.5)))
            if 0 <= cx < frame_w and 0 <= cy < frame_h and score_val >= 0.75:
                cv2.circle(out, (cx, cy), 2, blue, 1)

        for i, box_roi in enumerate(stroke_merge_pair_roi[:50]):
            box_full = box_roi + np.array([rx0, ry0, rx0, ry0], dtype=np.float32)
            _draw_box(box_full, blue, "same-row merge" if i == 0 else "", 1)

        if selected_full.size >= 4:
            _draw_box(selected_full[0], green, "final selected", 3)

        has_gt = gt_roi_arr.size >= 4 and gt_roi_arr[0, 2] > gt_roi_arr[0, 0] and gt_roi_arr[0, 3] > gt_roi_arr[0, 1]
        gt_full = None
        if has_gt:
            gt_full = gt_roi_arr[0] + np.array([rx0, ry0, rx0, ry0], dtype=np.float32)
            _draw_box(gt_full, purple, "GT", 3)

        if search_full.size >= 4 and peak_search.size >= 2:
            sf = search_full[0]
            for i, pxy in enumerate(peak_search[:10]):
                fx = int(round(float(sf[0] + pxy[0])))
                fy = int(round(float(sf[1] + pxy[1])))
                if 0 <= fx < frame_w and 0 <= fy < frame_h:
                    cv2.line(out, (max(0, fx - 9), fy), (min(frame_w - 1, fx + 9), fy), blue, 2 if i == 0 else 1)
                    cv2.line(out, (fx, max(0, fy - 9)), (fx, min(frame_h - 1, fy + 9)), blue, 2 if i == 0 else 1)
                    label_score = peak_scores[i] if i < peak_scores.size else (final_scores[i] if i < final_scores.size else 0.0)
                    cv2.putText(out, f"peak{i + 1} {label_score:.2f}", (fx + 5, max(12, fy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, blue, 1)

        crop_roi = candidate_roi[0] if candidate_roi.size >= 4 else np.zeros((4,), dtype=np.float32)
        crop_full = candidate_full[0] if candidate_full.size >= 4 else crop_roi + np.array([rx0, ry0, rx0, ry0], dtype=np.float32)
        crop_cx = 0.5 * (float(crop_roi[0]) + float(crop_roi[2]))
        crop_cy = 0.5 * (float(crop_roi[1]) + float(crop_roi[3]))
        peak_xy_roi = peak_roi_all[0] if peak_roi_all.size >= 2 else np.zeros((2,), dtype=np.float32)
        dx = float(peak_xy_roi[0]) - crop_cx
        dy = float(peak_xy_roi[1]) - crop_cy
        coord_warning = abs(dx) > 5.0 or abs(dy) > 5.0

        peak_score = float(peak_scores[0]) if peak_scores.size > 0 else 0.0
        scale_idx = int(best_scale_idx[0]) if best_scale_idx.size > 0 else -1
        aspect_idx = int(best_aspect_idx[0]) if best_aspect_idx.size > 0 else -1
        gt_center_in_vehicle = False
        gt_center_in_search = False
        topk_has_gt = False
        final_hits_gt = False
        center_error = 0.0
        iou = float(iou_arr[0]) if iou_arr.size > 0 else 0.0
        if has_gt:
            gt = gt_roi_arr[0]
            gt_cx = 0.5 * (float(gt[0]) + float(gt[2]))
            gt_cy = 0.5 * (float(gt[1]) + float(gt[3]))
            if vehicle_roi.size >= 4:
                vb = vehicle_roi[0]
                gt_center_in_vehicle = bool(vb[0] <= gt_cx <= vb[2] and vb[1] <= gt_cy <= vb[3])
            if search_roi.size >= 4:
                sb = search_roi[0]
                gt_center_in_search = bool(sb[0] <= gt_cx <= sb[2] and sb[1] <= gt_cy <= sb[3])
            for box in candidate_roi[:10]:
                bx = 0.5 * (float(box[0]) + float(box[2]))
                by = 0.5 * (float(box[1]) + float(box[3]))
                if gt[0] <= bx <= gt[2] and gt[1] <= by <= gt[3]:
                    topk_has_gt = True
                    break
            if selected_roi.size >= 4:
                fb = selected_roi[0]
                fcx = 0.5 * (float(fb[0]) + float(fb[2]))
                fcy = 0.5 * (float(fb[1]) + float(fb[3]))
                final_hits_gt = bool(gt[0] <= fcx <= gt[2] and gt[1] <= fcy <= gt[3])
                center_error = ((fcx - gt_cx) ** 2 + (fcy - gt_cy) ** 2) ** 0.5

        if coord_warning:
            diagnosis = "FAIL: COORDINATE_ALIGNMENT"
        elif has_gt and not gt_center_in_vehicle:
            diagnosis = "FAIL: VEHICLE_GATE"
        elif has_gt and gt_center_in_vehicle and not gt_center_in_search:
            diagnosis = "FAIL: PLATE_SEARCH_ROI"
        elif has_gt and gt_center_in_vehicle and gt_center_in_search and candidate_roi.shape[0] == 0:
            diagnosis = "FAIL: PLATE_CANDIDATE_GENERATION"
        elif has_gt and topk_has_gt and not final_hits_gt:
            diagnosis = "FAIL: MLP_SELECTOR"
        elif has_gt and candidate_roi.shape[0] > 0 and topk_has_gt and final_hits_gt:
            diagnosis = "PASS"
        elif not has_gt and candidate_roi.shape[0] > 0 and selected_roi.shape[0] > 0 and not coord_warning:
            diagnosis = "PASS"
        else:
            diagnosis = "FAIL: PLATE_CANDIDATE_GENERATION"

        lines = [
            f"vehicle_box_roi={_fmt_box(vehicle_roi)}",
            f"vehicle_box_full={_fmt_box(vehicle_full)}",
            f"vehicle_area={float(vehicle_area[0]):.1f}" if vehicle_area.size else "vehicle_area=[]",
            f"vehicle_center={vehicle_center[0].round(1).tolist()}" if vehicle_center.size >= 2 else "vehicle_center=[]",
            f"vehicle_roi_mean={float(vehicle_roi_mean[0]):.3f}" if vehicle_roi_mean.size else "vehicle_roi_mean=[]",
            f"vehicle_roi_max={float(vehicle_roi_max[0]):.3f}" if vehicle_roi_max.size else "vehicle_roi_max=[]",
            f"plate_search_box_roi={_fmt_box(search_roi)}",
            f"plate_search_box_full={_fmt_box(search_full)}",
            f"search_w={float(search_wh[0, 0]):.1f} search_h={float(search_wh[0, 1]):.1f}" if search_wh.size >= 2 else "search_w=[] search_h=[]",
            f"peak_x={float(peak_xy_roi[0]):.1f} peak_y={float(peak_xy_roi[1]):.1f} peak_score={peak_score:.3f}",
            f"best_scale_idx={scale_idx} best_aspect_idx={aspect_idx}",
            f"peak_xy_search={peak_search[0].round(1).tolist()}" if peak_search.size >= 2 else "peak_xy_search=[]",
            f"crop_box_search={_fmt_box(crop_search)}",
            f"crop_box_roi={_fmt_box(candidate_roi)}",
            f"crop_box_full={_fmt_box(candidate_full)}",
            f"center_dx={dx:.1f} center_dy={dy:.1f}",
        ]
        if coord_warning:
            lines.append("WARNING: PEAK_CROP_MISMATCH")
        for i in range(min(5, peak_search.shape[0])):
            score_val = float(peak_scores[i]) if i < peak_scores.size else 0.0
            lines.append(f"peak{i + 1}: x={float(peak_search[i, 0]):.1f} y={float(peak_search[i, 1]):.1f} score={score_val:.3f}")
        for i in range(min(10, candidate_roi.shape[0])):
            box = candidate_roi[i]
            cx = 0.5 * (float(box[0]) + float(box[2]))
            cy = 0.5 * (float(box[1]) + float(box[3]))
            lines.append(
                f"rank={i + 1} center_x={cx:.1f} center_y={cy:.1f} "
                f"width={float(candidate_width[i]) if i < candidate_width.size else 0.0:.1f} "
                f"height={float(candidate_height[i]) if i < candidate_height.size else 0.0:.1f} "
                f"aspect={float(candidate_aspect[i]) if i < candidate_aspect.size else 0.0:.2f} "
                f"uniform_rect_score={float(uniform_rect_score_values[i]) if i < uniform_rect_score_values.size else 0.0:.2f} "
                f"contrast_score={float(contrast_score_values[i]) if i < contrast_score_values.size else 0.0:.2f} "
                f"aspect_score={float(aspect_score_values[i]) if i < aspect_score_values.size else 0.0:.2f} "
                f"diagonal_prior={float(diagonal_prior[i]) if i < diagonal_prior.size else 0.0:.2f} "
                f"mlp_score={float(mlp_score[i]) if i < mlp_score.size else 0.0:.2f} "
                f"rule_score={float(rule_score_values[i]) if i < rule_score_values.size else 0.0:.2f} "
                f"final_score={float(final_scores[i]) if i < final_scores.size else 0.0:.2f}"
            )
        if has_gt:
            gt = gt_roi_arr[0]
            gt_cx = 0.5 * (float(gt[0]) + float(gt[2]))
            gt_cy = 0.5 * (float(gt[1]) + float(gt[3]))
            lines.append(f"GT center=({gt_cx:.1f},{gt_cy:.1f}) in_vehicle={gt_center_in_vehicle} in_search={gt_center_in_search}")
            lines.append(f"GT IoU={iou:.3f} center_error={center_error:.1f}")
        lines.append(f"AUTO_DIAG: {diagnosis}")

        selected_idx_arr = np.asarray(item.get("selected_candidate_idx", np.zeros((0,), dtype=np.int64))).reshape(-1)
        selected_idx = int(selected_idx_arr[0]) if selected_idx_arr.size > 0 else -1
        top_scores = [float(final_scores[i]) if i < final_scores.size else 0.0 for i in range(3)]
        final_score = float(final_scores[selected_idx]) if 0 <= selected_idx < final_scores.size else (float(final_scores[0]) if final_scores.size > 0 else 0.0)
        log_lines = [
            "DETECTION_DEBUG",
            f"frame={item.get('frame_idx', -1)}",
            f"LAST_UNIFORMITY_SIGMA_SQ={float(uniform_sigma_sq_values[0]) if uniform_sigma_sq_values.size else 0.0:.8f}",
            f"LAST_UNIFORM_WEIGHT={float(uniform_weight_values[0]) if uniform_weight_values.size else 0.0:.2f}",
            f"LAST_BLACKHAT_ALPHA={float(item.get('blackhat_alpha', np.array([0.0], dtype=np.float32)).reshape(-1)[0]) if np.asarray(item.get('blackhat_alpha', np.array([], dtype=np.float32))).size else 0.0:.8f}",
            f"candidate_count={int(candidate_count_values[0]) if candidate_count_values.size else candidate_roi.shape[0]}",
            f"top_candidate_score={float(top_candidate_score_values[0]) if top_candidate_score_values.size else 0.0:.4f}",
            f"final_score={final_score:.3f}",
        ]
        if coord_warning:
            log_lines.append("WARNING: PEAK_CROP_MISMATCH")
        if has_gt:
            log_lines.extend([
                f"IoU={iou:.3f}",
                f"center_error={center_error:.1f}",
                f"gt_uniform_rect_score={float(gt_uniform_rect_score_values[0]) if gt_uniform_rect_score_values.size else 0.0:.3f}",
                f"gt_uniform_candidate_generated={bool(gt_uniform_candidate_generated_values[0]) if gt_uniform_candidate_generated_values.size else False}",
                f"pred_box={selected_roi[0].round(1).tolist() if selected_roi.size >= 4 else []}",
                f"gt_box={gt_roi_arr[0].round(1).tolist()}",
            ])
        log_lines.append(diagnosis)
        self._log(" | ".join(log_lines))

        x_text, y_text = 12, 24
        for line in lines[:24]:
            color = yellow if "WARNING" in line or "FAIL" in line else (green if "PASS" in line else white)
            cv2.putText(out, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, black, 3)
            cv2.putText(out, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y_text += 16
        return out

    def configure_webapp_preview_from_args(self, args) -> None:
        self.webapp_preview_from_debug_frame = bool(
            getattr(args, "webapp_enable", False)
            and getattr(args, "webapp_preview_from_debug_frame", False)
        )
        self.webapp_preview_path = getattr(args, "webapp_preview_jpg", None)
        self.webapp_preview_format = getattr(args, "webapp_preview_format", "jpg")
        self.webapp_preview_max_width = getattr(args, "webapp_preview_max_width", 960)

    def _normalize_webapp_preview_path(self) -> Path | None:
        preview_path = getattr(self, "webapp_preview_path", None)
        if not preview_path:
            return None

        path = Path(preview_path)
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            return path

        preview_format = str(getattr(self, "webapp_preview_format", "jpg") or "jpg").strip().lower()
        suffix = ".png" if preview_format == "png" else ".jpg"
        return path.with_suffix(suffix)

    def _resize_webapp_preview_if_needed(self, image_bgr: np.ndarray) -> np.ndarray:
        try:
            max_width = int(getattr(self, "webapp_preview_max_width", 960) or 0)
        except (TypeError, ValueError):
            max_width = 960

        if max_width <= 0:
            return image_bgr

        height, width = image_bgr.shape[:2]
        if width <= max_width:
            return image_bgr

        scale = float(max_width) / float(max(width, 1))
        target_size = (max_width, max(1, int(round(height * scale))))
        return cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)

    def _write_webapp_preview_image(self, image_bgr: np.ndarray | None) -> None:
        if not getattr(self, "webapp_preview_from_debug_frame", False) or image_bgr is None:
            return

        try:
            path = self._normalize_webapp_preview_path()
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            output = self._resize_webapp_preview_if_needed(image_bgr)
            tmp_path = path.parent / f"{path.stem}.tmp{path.suffix}"
            if cv2.imwrite(str(tmp_path), output):
                tmp_path.replace(path)
            elif tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            return

    def _save_direct_crop_debug_frame_dir(self, item: dict, overlay_rgb: np.ndarray) -> None:
        if not self.debug_save_frames_dir:
            return
        os.makedirs(self.debug_save_frames_dir, exist_ok=True)
        frame_idx = int(item.get("frame_idx", -1))

        def _write_map(key: str, suffix: str) -> None:
            arr = self._map_to_u8(item.get(key))
            if arr is None:
                return
            cv2.imwrite(os.path.join(self.debug_save_frames_dir, f"frame_{frame_idx:06d}_{suffix}.png"), arr)

        # Direct crop debug stores only the candidate source map by default.
        # _write_map("gray_map", "gray")
        # _write_map("fuzzy_stretched_map", "fuzzy_stretched")
        # _write_map("uniform_mu", "uniform_mu")
        # _write_map("blackhat_weight_map", "blackhat_weight")
        _write_map("candidate_source_map", "candidate_source")
        # _write_map("scharrx_map", "scharrx")
        # _write_map("morphology_before_merge_map", "morphology_before_merge")
        # _write_map("morphology_map", "morphology")
        # _write_map("stroke_components_map", "stroke_components")
        # _write_map("motion_prior_map", "motion_prior")
        # _write_map("same_row_merge_map", "same_row_merge")
        cv2.imwrite(
            os.path.join(self.debug_save_frames_dir, f"frame_{frame_idx:06d}_candidates_overlay.png"),
            cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
        )

    def _make_candidate_stage_overlay(self, frame_rgb: np.ndarray, gate, stage_boxes: dict[str, np.ndarray], best_idx: int = -1) -> np.ndarray:
        # candidate stage colored overlay debug
        out = frame_rgb.copy()
        gx, gy, _, _ = [int(v) for v in gate]
        stage_cfg = [
            ("raw", "R", (255, 0, 0), 30),
            ("filtered", "F", (255, 255, 0), 20),
            ("score_pre", "S", (0, 255, 255), 15),
            ("final", "T", (0, 0, 255), 8),
        ]
        for key, label, color, limit in stage_cfg:
            boxes = stage_boxes.get(key)
            if boxes is None:
                continue
            for i, b in enumerate(boxes[:limit]):
                x1, y1, x2, y2 = [int(v) for v in b]
                x1 += gx; x2 += gx; y1 += gy; y2 += gy
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return out

    def _log_exception(self, msg: str) -> None:
        self._log(msg)
        tb = traceback.format_exc().rstrip()
        if tb:
            self._log(tb)

    def _resolve_debug_video_target(self) -> tuple[Path, str]:
        out_path = Path(self.debug_save_video or "debug_output.avi")
        suffix = out_path.suffix.lower()
        if suffix == ".mp4":
            return out_path, "mp4v"
        if suffix != ".avi":
            out_path = out_path.with_suffix(".avi")
        return out_path, "MJPG"

    def _normalize_debug_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a BGR uint8 HxWx3 contiguous frame sized for debug VideoWriter."""
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        else:
            raise ValueError(f"Unsupported debug video frame shape: {frame.shape}")

        frame = np.ascontiguousarray(frame)

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                finite_max = float(np.nanmax(frame)) if frame.size else 0.0
                if finite_max <= 1.0:
                    frame = frame * 255.0
                frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
                frame = np.clip(frame, 0.0, 255.0)
            frame = frame.astype(np.uint8)
        else:
            frame = frame.astype(np.uint8, copy=False)

        target_w = self.debug_video_width
        target_h = self.debug_video_height
        if target_w is not None and target_h is not None:
            h, w = frame.shape[:2]
            if w != target_w or h != target_h:
                frame = cv2.resize(
                    frame,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
        return np.ascontiguousarray(frame)

    def _open_video_writer(self, sample_bgr: np.ndarray):
        if not self.debug_save_video or self._video_writer_failed:
            return None, None
        out_path, four = self._resolve_debug_video_target()
        os.makedirs(str(out_path.parent), exist_ok=True)
        sample_bgr = self._normalize_debug_video_frame(sample_bgr)
        h, w = sample_bgr.shape[:2]
        self.debug_video_width = int(w)
        self.debug_video_height = int(h)
        fps = max(float(self.fps), 1.0)
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*four), fps, (w, h))
        if vw.isOpened():
            return vw, str(out_path)
        self._video_writer_failed = True
        self._log(f"Stage13 video_open_fail path={out_path} fourcc={four} fps={fps:.3f} size=({w},{h})")
        return None, None

    def _video_writer_loop(self):
        while True:
            item = self.video_writer_q.get()
            try:
                if item is None:
                    break
                if item.get('kind') != 'debug_snapshot':
                    continue
                self._write_debug_snapshot(item, write_video=True, write_images=False)
            except Exception:
                self._writer_error_count += 1
                frame_idx = item.get('frame_idx', 'unknown') if isinstance(item, dict) else 'unknown'
                self._log_exception(f"Stage13 video_writer_exception frame_idx={frame_idx} count={self._writer_error_count}")
            finally:
                try:
                    self.video_writer_q.task_done()
                except Exception:
                    pass

    def _image_writer_loop(self):
        while True:
            item = self.image_writer_q.get()
            try:
                if item is None:
                    break
                if item.get('kind') != 'debug_snapshot':
                    continue
                self._write_debug_snapshot(item, write_video=False, write_images=True)
            except Exception:
                self._writer_error_count += 1
                frame_idx = item.get('frame_idx', 'unknown') if isinstance(item, dict) else 'unknown'
                self._log_exception(f"Stage13 image_writer_exception frame_idx={frame_idx} count={self._writer_error_count}")
            finally:
                try:
                    self.image_writer_q.task_done()
                except Exception:
                    pass

    def _write_debug_video(self, overlay_bgr: np.ndarray, frame_idx: int) -> None:
        if self._video_writer_failed:
            return
        try:
            overlay_bgr = self._normalize_debug_video_frame(overlay_bgr)
        except Exception:
            self._log_exception(f"Stage13 video_frame_normalize_exception frame_idx={frame_idx}")
            return

        if self._video_writer is None:
            self._video_writer, self._video_path = self._open_video_writer(overlay_bgr)
        if self._video_writer is None:
            return

        overlay_bgr = self._normalize_debug_video_frame(overlay_bgr)
        if self.debug_stage_log and not self._debug_video_shape_logged:
            h, w = overlay_bgr.shape[:2]
            channels = overlay_bgr.shape[2] if overlay_bgr.ndim == 3 else 1
            self._log(
                "Stage13 video_writer_frame_check "
                f"writer_size=({self.debug_video_width},{self.debug_video_height}) "
                f"frame_size=({w},{h}) channels={channels} dtype={overlay_bgr.dtype}"
            )
            self._debug_video_shape_logged = True

        try:
            self._video_writer.write(overlay_bgr)
        except Exception:
            h, w = overlay_bgr.shape[:2]
            self._log_exception(
                "[DEBUG]\n"
                "video_writer_frame_error:\n"
                f"writer=({self.debug_video_width},{self.debug_video_height})\n"
                f"frame=({w},{h})\n"
                f"frame_idx={frame_idx}\n"
                f"path={self._video_path}"
            )

    def _write_debug_images(self, item: dict, overlay_rgb: np.ndarray, overlay_bgr: np.ndarray, frame_idx: int, sec: float) -> None:
        if self.debug_save_frames_dir and (item.get("save_frame", False) or item.get("save_binary", False)):
            self._save_direct_crop_debug_frame_dir(item, overlay_rgb)
            self._write_webapp_preview_image(overlay_bgr)
        if item.get("save_event", False) and self.debug_event_dir:
            self._log(
                f"event_overlay_candidates_count={int(item.get('_event_overlay_candidates_count', 0))} "
                f"event_overlay_coord_mode={item.get('_event_overlay_coord_mode', 'none')}"
            )
            os.makedirs(self.debug_event_dir, exist_ok=True)
            fp = os.path.join(self.debug_event_dir, f"event_{frame_idx:06d}_sec_{sec:06.2f}".replace('.', 'p') + "_overlay.jpg")
            ok = cv2.imwrite(fp, overlay_bgr)
            if not ok:
                self._log(f"Stage13 event_write fail frame_idx={item.get('frame_idx')}")
        if DEBUG_VERBOSE_DEBUG_IMAGES and item.get("save_candidate_stage", False) and self.debug_draw_candidate_stages:
            base_dir = self.debug_event_dir or self.debug_save_frames_dir
            if base_dir:
                out_dir = os.path.join(base_dir, "candidate_stage_overlay")
                os.makedirs(out_dir, exist_ok=True)
                sec_tag = f"{item['sec']:06.2f}".replace('.', 'p')
                fp = os.path.join(out_dir, f"frame_{item['frame_idx']:06d}_sec_{sec_tag}_candidate_stages.jpg")
                stage_overlay = self._make_candidate_stage_overlay(item['frame_rgb'], item['gate'], item.get('stage_boxes', {}))
                ok = cv2.imwrite(fp, cv2.cvtColor(stage_overlay, cv2.COLOR_RGB2BGR))
                if not ok:
                    self._log(f"Stage13 candidate_stage_write fail frame_idx={item.get('frame_idx')}")

    def _write_debug_snapshot(self, item: dict, write_video: bool = True, write_images: bool = True):
        frame_idx = int(item['frame_idx'])
        sec = float(item['sec'])
        overlay = self._make_debug_overlay_frame(item['frame_rgb'], item['gate'], item['candidates'], item['confirmed_ids'], frame_idx, sec, bool(item['ocr_triggered']), item.get('candidate_track_ids'))
        if DEBUG_VERBOSE_DEBUG_IMAGES:
            overlay = self._make_direct_crop_debug_overlay(overlay, item)
        if item.get('save_event', False):
            overlay = self._draw_event_candidate_overlay(overlay, item, max_overlay_candidates=50)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        if write_video and item.get('save_video', False):
            self._write_debug_video(overlay_bgr, frame_idx)
        if write_images and (item.get('save_frame', False) or item.get('save_event', False) or item.get('save_binary', False) or item.get('save_candidate_stage', False)):
            self._write_debug_images(item, overlay, overlay_bgr, frame_idx, sec)


    def save_ocr_csv(self, path: str):
        if not path:
            return
        if not getattr(self, "_async_ocr_outputs_finalized", False):
            self.finalize_async_ocr_outputs()
        middle_stats = self._middle_slot_stats()
        self._log(
            "[MIDDLE_SLOT_UPL] pre_csv_save "
            f"ocr_csv_rows={len(self.ocr_csv_rows)} "
            f"middle_slot_results_drained={int(middle_stats.get('middle_slot_results_drained', 0.0))} "
            f"middle_slot_evidence_added={int(middle_stats.get('middle_slot_evidence_added', 0.0))} "
            f"middle_slot_evidence_skipped={int(middle_stats.get('middle_slot_evidence_skipped', 0.0))}"
        )
        self._log_middle_slot_v32_profile_once()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "track_id",
            "frame_idx",
            "candidate_idx",
            "ocr_attempt",
            "ocr_skip_reason",
            "ocr_text",
            "ocr_conf",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "roi_w",
            "roi_h",
            "candidate_count",
            "track_id_count",
            "febam_confirmed_count",
            "gray_stretched_ocr_used",
            "is_final",
            "final_plate",
            "attempted_count",
            "febam_score",
            "febam_energy",
            "febam_threshold",
            "febam_memory",
            "febam_confirmed",
            "ocr_input_type",
            "ocr_candidate_rank",
            "ocr_crop_policy",
            "ocr_crop_expand_top",
            "ocr_y1_original",
            "ocr_y1_expanded",
            "roi_aspect",
            "ocr_expand_top",
            "ocr_angle",
            "ocr_rotation_applied",
            "ocr_angle_score",
            "ocr_gpu_rectified",
            "ocr_trigger_reason",
            "near_confirmed_large_roi",
            "febam_full_confirmed",
            "near_confirmed_ocr_sample",
            "ocr_sampling_level",
            "ocr_sampling_reason",
            "source_weight",
            "source",
            "pre_febam_candidate_count",
            "post_febam_confirmed_count",
            "febam_score_thr",
            "febam_energy_thr",
            "febam_memory_min",
            "ocr_saved_raw",
            "ocr_saved_gray_stretched",
            "ocr_saved_rotated",
            "ocr_input_variant",
            "ocr_fallback_rank",
            "ocr_fallback_used",
            "ocr_rotation_threshold",
            "ocr_reader_langs",
            "ocr_allowlist_mode",
            "ocr_normalized_text",
            "ocr_pattern_type",
            "ocr_postprocess_applied",
            "ocr_postprocess_reason",
            "ocr_plate_text_final",
            "ocr_korean_middle_raw",
            "ocr_korean_middle_fixed",
            "ocr_small_roi",
            "ocr_upscale_applied",
            "ocr_upscale_factor",
            "ocr_sharpen_applied",
            "ocr_sharpen_amount",
            "mrs",
            "best_text",
            "best_conf",
            "final_voted_text",
            "final_vote_count",
            "mrs_max",
            "mrs_mean",
            "ocr_count",
            "topk_ocr",
            "raw_text",
            "raw_conf",
            "corrected_text",
            "corrected_conf",
            "row_type",
            "ocr_source",
            "voted_text",
            "voted_conf",
            "final_observed_plate",
            "final_consensus_plate",
            "final_plate_source",
            "ocr_variant_order",
            "ocr_variant_count",
            "ocr_norm_w",
            "ocr_norm_h",
            "ocr_norm_scale",
            "ocr_norm_pad_x",
            "ocr_norm_pad_y",
            "ocr_norm_aspect_preserved",
            "ocr_allowlist_sweep_disabled",
            "ocr_early_stop_reason",
            "ocr_selected_final_candidate_only",
            "plate_layout",
            "grammar_pattern",
            "grammar_candidates_top3",
            "grammar_best_text",
            "grammar_best_score",
            "grammar_best_reason",
            "grammar_valid_final",
            "split_ocr_used",
            "split_left_text",
            "split_mid_text",
            "split_right_text",
            "split_top_text",
            "split_bottom_text",
            "committed_plate_locked",
            "ocr_crop_expand_top_ratio",
            "grammar_decoder_used",
            "split_fallback_used",
            "split_mode",
            "final_candidate_source",
            "final_candidate_score",
            "final_candidate_reason",
            "final_output_plate",
            "ocr_engine",
            "easyocr_mode",
            "easyocr_batch_size",
            "easyocr_workers",
            "easyocr_recognize_only",
            "easyocr_readtext_fallback_used",
            "fastplate_model",
            "fastplate_device",
            "fastplate_batch_size",
            "async_delay_frames",
            "async_delay_ms",
            "ocr_crop_expand_left_ratio",
            "ocr_crop_expand_right_ratio",
            "ocr_crop_expand_bottom_ratio",
            "ocr_variant_name",
            "ocr_raw_result_short",
            "normalized_plate_candidate",
            "korean_plate_valid",
            "ocr_variant_tier",
            "ocr_variant_executed",
            "ocr_variants_planned",
            "ocr_variants_executed_count",
            "ocr_readtext_fallback_count",
            "ocr_variant_skip_reason",
            "skeleton_anchor_used",
            "skeleton_anchor_pattern",
            "skeleton_anchor_text",
            "skeleton_anchor_support",
            "variant_group_key",
            "variant_group_anchor_text",
            "variant_group_merge_reason",
            "variant_group_selected_skeleton",
            "custom_onnx_raw_text",
            "custom_onnx_korean_slot",
            "custom_onnx_digit_skeleton_used",
            "custom_onnx_slot_candidate",
            "custom_onnx_slot_evidence_added",
            "custom_onnx_slot_skip_reason",
            "middle_slot_attempted",
            "middle_slot_source",
            "middle_slot_status",
            "middle_slot_evidence",
            "middle_slot_crop_path",
            "middle_slot_crop_save_error",
            "middle_slot_backend",
            "middle_slot_model",
            "middle_slot_cropper_mode",
            "middle_slot_crop_type",
            "middle_slot_crop_score",
            "middle_slot_crop_x1",
            "middle_slot_crop_y1",
            "middle_slot_crop_x2",
            "middle_slot_crop_y2",
            "middle_slot_crop_reason",
            "middle_slot_prefix_digits",
            "middle_slot_suffix_digits",
            "middle_slot_top1",
            "middle_slot_top1_ko",
            "middle_slot_top1_conf",
            "middle_slot_top2",
            "middle_slot_top2_ko",
            "middle_slot_top2_conf",
            "middle_slot_top3",
            "middle_slot_top3_ko",
            "middle_slot_top3_conf",
            "middle_slot_margin",
            "middle_slot_topk",
            "middle_slot_topk_json",
            "middle_slot_topk_requested",
            "middle_slot_topk_actual",
            "middle_slot_hog_lbp_min_conf",
            "middle_slot_hog_lbp_min_margin",
            "middle_slot_hog_lbp_min_crop_score",
            "middle_slot_prototype_id",
            "middle_slot_update_applied",
            "middle_slot_update_alpha",
            "middle_slot_digit_anchor",
            "middle_slot_generated_candidate",
            "middle_slot_source_weight",
            "middle_slot_skip_reason",
            "middle_slot_batch_size",
            "middle_slot_batch_delay_ms",
            "middle_slot_gpu_batch",
            "middle_slot_encoder_mode",
            "middle_slot_prototype_encoder_mode",
            "middle_slot_similarity_mode",
            "middle_slot_queue_priority",
            "middle_slot_evidence_skip_reason",
            "middle_slot_gt_anchor_confused_fair",
            "middle_slot_gt_anchor_status",
            "middle_slot_gt_plate",
            "middle_slot_gt_middle",
            "middle_slot_gt_anchor_boost",
            "middle_slot_confused_scores",
            "middle_slot_confused_candidates",
            "middle_slot_confused_selected_candidate",
            "middle_slot_confused_support_count",
            "middle_slot_confused_skip_reason",
            "middle_slot_evidence_flag",
            "middle_slot_evidence_text",
            "event_id",
            "crop_id",
            "crop_source",
            "crop_score",
            "crop_shape",
            "middle_slot_v32_enabled",
            "middle_slot_v32_model_path",
            "middle_slot_v32_topk",
            "middle_slot_v32_top1",
            "middle_slot_v32_top1_roman",
            "middle_slot_v32_top1_ko",
            "middle_slot_v32_top1_score",
            "middle_slot_v32_top2",
            "middle_slot_v32_top2_roman",
            "middle_slot_v32_top2_ko",
            "middle_slot_v32_top2_score",
            "middle_slot_v32_top3",
            "middle_slot_v32_top3_roman",
            "middle_slot_v32_top3_ko",
            "middle_slot_v32_top3_score",
            "middle_slot_v32_margin",
            "middle_slot_v32_min_conf",
            "middle_slot_v32_min_margin",
            "middle_slot_v32_source_weight",
            "middle_slot_v32_evidence_weight_top1",
            "middle_slot_v32_added_evidence",
            "middle_slot_v32_skip_reason",
            "middle_slot_v32_event_crop_count",
            "middle_slot_v32_pending_len",
            "middle_slot_v32_flush_reason",
            "middle_slot_v32_frame_idx",
            "middle_slot_v32_event_id",
            "middle_slot_v32_track_id",
            "middle_slot_v32_batch_size",
            "middle_slot_v32_infer_ms",
            "middle_slot_group_key",
            "middle_slot_group_key_source",
            "middle_slot_group_key_role",
            "middle_slot_fastplate_skeleton_candidates",
            "middle_slot_generated_digit_skeleton",
            "middle_slot_skeleton_match",
            "middle_slot_skeleton_support",
            "middle_slot_feedback_bucket",
            "middle_slot_digit_skeleton",
            "middle_slot_char_memory_best",
            "middle_slot_char_memory_second",
            "middle_slot_char_memory_best_count",
            "middle_slot_char_memory_second_score",
            "middle_slot_char_memory_margin",
            "middle_slot_char_memory_support",
            "middle_slot_char_memory_support_frames",
            "middle_slot_char_memory_json",
            "string_febam_state",
            "string_febam_activation",
            "string_febam_score",
            "string_febam_energy",
            "string_febam_memory",
            "segment_id",
            "segment_len",
            "segment_effective_count",
            "segment_weight",
            "segment_text_medoid",
            "segment_consensus_text",
            "cluster_count",
            "contamination_flag",
            "commit_frame",
            "first_stable_frame",
            "ocr_call_count",
            "ocr_skipped_after_commit",
            "group_id",
            "motion_group_id",
            "track_group_id",
            "pseudo_vehicle_id",
            "fused_image_path",
            "event_fusion_method",
            "event_fusion_preset",
            "event_fusion_mode",
            "event_fusion_group_key",
            "event_fusion_num_used",
            "event_fusion_selected_top_k",
            "event_fusion_center_pad_ratio",
            "event_fusion_color_thr",
            "event_fusion_max_shift_ratio",
            "event_fusion_top_candidates",
            "event_fusion_ocr_backend",
            "event_fusion_ocr_flush_reason",
            "event_fusion_ocr_raw_result",
            "event_fusion_ocr_raw_plate",
            "event_fusion_ocr_raw_matches_text",
            "b5_yolo_quality_score",
            "b5_fusion_output_tag",
            "motion_cluster_id",
            "motion_state",
            "motion_activation",
            "motion_score",
            "motion_margin",
            "motion_similarity",
            "motion_second_similarity",
            "motion_direction_bin",
            "motion_direction_name",
            "motion_speed",
            "motion_frame_gap",
            "motion_gap_score",
            "motion_y_band",
            "motion_contamination",
            "motion_split_reason",
            "motion_primary_cluster_id",
            "motion_cluster_count",
            "motion_split_applied",
            "motion_filtered_rows",
            "motion_kept_rows",
            "motion_weight_min",
            "motion_weight_mean",
            "motion_weight_max",
            "motion_entry_zone",
            "motion_exit_zone",
            "motion_scale_trend",
            "motion_position_reset_score",
            "motion_split_evidence",
            "motion_filter_applied",
            "motion_non_primary_ratio",
            "motion_contamination_ratio",
            "motion_representative_reason",
            "color_soft_applied",
            "color_reference_class",
            "color_soft_weight_min",
            "color_soft_weight_mean",
            "color_soft_weight_max",
            "color_class_mismatch_count",
            "color_unknown_count",
            "color_outlier_count",
            "color_excluded_count",
            "color_extreme_outlier_count",
            "color_filter_reason",
            "fusion_final_weight_min",
            "fusion_final_weight_mean",
            "fusion_final_weight_max",
            "color_distance",
            "color_class",
            "color_excluded_by_motion_color",
            "color_weight_reason",
            "fusion_final_weight",
            "fusion_rows_original",
            "fusion_rows_after_motion_filter",
            "fusion_rows_after_color_filter",
            "rank1_crop_id",
            "rank1_quality_score",
            "rank1_color_proto",
            "adaptive_alpha",
        ]
        if getattr(self, "bio_adaptive_processor", None) is not None:
            fieldnames.extend(BIO_CSV_FIELDS)
        duplicate_fieldnames = [
            name
            for name, count in Counter(fieldnames).items()
            if name and count > 1
        ]
        if duplicate_fieldnames:
            self._log(
                "[CSV_HEADER_DEDUP] "
                f"duplicate_fieldnames={duplicate_fieldnames}"
            )
            fieldnames = list(dict.fromkeys(fieldnames))

        row_type_counter = Counter(str(row.get("row_type", "")) for row in self.ocr_csv_rows)
        evidence_counter = Counter(str(row.get("middle_slot_evidence", "")) for row in self.ocr_csv_rows)
        csv_save_debug_msg = (
            f"[CSV_SAVE_DEBUG] path={path} len={len(self.ocr_csv_rows)} "
            f"row_type={dict(row_type_counter)} middle_slot_evidence={dict(evidence_counter)}"
        )
        print(csv_save_debug_msg, flush=True)
        self._log(csv_save_debug_msg)
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.ocr_csv_rows)
            os.replace(tmp_path, path)
            try:
                with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                    saved_lines = sum(1 for _ in f)
            except Exception:
                saved_lines = -1
            csv_save_after_msg = f"[CSV_SAVE_DEBUG_AFTER] path={path} saved_lines={saved_lines}"
            print(csv_save_after_msg, flush=True)
            self._log(csv_save_after_msg)
        except Exception as exc:
            csv_save_error_msg = f"[CSV_SAVE_ERROR] path={path} tmp_path={tmp_path} error={type(exc).__name__}: {exc}"
            print(csv_save_error_msg, file=sys.stderr, flush=True)
            self._log(csv_save_error_msg)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    def close(self):
        if getattr(self, "labeling_febam_sink", None) is not None:
            try:
                self.labeling_febam_sink.close()
            except Exception:
                self._log_exception("FEBAM observation sink close_exception")
        try:
            self._finalize_b5_yolo_high_quality_fusion(reason="pipeline_close")
        except Exception:
            self._log_exception("B5 YOLO high-quality fusion finalize_exception")
        try:
            self._flush_event_fusion_ocr(reason="pipeline_close", frame_idx=getattr(self, "_current_frame_idx", None))
        except Exception:
            self._log_exception("Event fusion OCR close_flush_exception")
        restoration_shadow = getattr(
            self, "non_generative_restoration_shadow", None
        )
        if restoration_shadow is not None:
            try:
                restoration_shadow.flush_pending()
                restoration_shadow.wait_alignment(timeout=5.0)
            except Exception:
                self._log_exception("Restoration shadow close_flush_exception")
        try:
            self.finalize_async_ocr_outputs()
        except Exception:
            self._log_exception("Async OCR output close_finalize_exception")
        if restoration_shadow is not None:
            try:
                restoration_shadow.close(timeout=5.0)
            except Exception:
                self._log_exception("Restoration shadow close_exception")
        try:
            self._finalize_all_tracks(getattr(self, "_current_frame_idx", None))
        except Exception:
            self._log_exception("MRS finalize_all_tracks_exception")
        if getattr(self, "export_group_review", False):
            try:
                from tools.export_group_review_dataset import export_bridge_group_review

                summary = export_bridge_group_review(
                    self.trial020_fusion_bridge,
                    video_id=self.group_review_video_id,
                    output_dir=self.group_review_output,
                    fps=float(getattr(self, "fps", 25.0) or 25.0),
                    max_crops=self.group_review_max_crops,
                    min_frame_gap=self.group_review_min_frame_gap,
                    dedup_threshold=self.group_review_dedup_threshold,
                    save_raw=self.group_review_save_raw,
                    save_normalized=self.group_review_save_normalized,
                )
                self._log(f"[GROUP_REVIEW] exported {summary}")
                print(f"[GROUP_REVIEW_EXPORT] {json.dumps(summary, ensure_ascii=False)}", flush=True)
            except Exception as exc:
                self._log(
                    f"[GROUP_REVIEW] export failed error={type(exc).__name__}:{str(exc)[:500]}"
                )
        try:
            self.video_writer_q.put(None, block=True)
        except Exception:
            self._log_exception("Stage13 video_writer_close_sentinel_exception")
        try:
            self.image_writer_q.put(None, timeout=1.0)
        except queue.Full:
            self._log("Stage13 image_writer_queue_full close_sentinel_drop")
        except Exception:
            self._log_exception("Stage13 image_writer_close_sentinel_exception")
        try:
            self.video_writer.join(timeout=30)
            if self.video_writer.is_alive():
                self._log("Stage13 video_writer_join_timeout")
        except Exception:
            self._log_exception("Stage13 video_writer_join_exception")
        try:
            self.image_writer.join(timeout=5)
            if self.image_writer.is_alive():
                self._log("Stage13 image_writer_join_timeout")
        except Exception:
            self._log_exception("Stage13 image_writer_join_exception")
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                self._log_exception("Stage13 video_release_exception")
            finally:
                self._video_writer = None
                self._video_path = None
        lock = getattr(self, "_log_lock", None)
        if lock is not None:
            with lock:
                if self._log_fh is not None:
                    try:
                        self._log_fh.flush(); self._log_fh.close()
                    finally:
                        self._log_fh = None
        elif self._log_fh is not None:
            self._log_fh.flush(); self._log_fh.close(); self._log_fh = None

    def configure_gpu_stage_profile_from_args(self, args):
        self._gpu_stage_profiler = DeferredCudaBoundaryProfiler(getattr(args,"gpu_stage_profile",False),getattr(args,"gpu_stage_profile_warmup_frames",8),getattr(args,"gpu_stage_profile_max_frames",256))
        self._gpu_stage_profile_output = str(getattr(args,"gpu_stage_profile_output","") or r"C:\plate_runtime\final_wise_gpu_resident\gpu_stage_profile.json")

    def configure_yolo_input_trace_from_args(self, args):
        self._yolo_input_trace_enabled = bool(getattr(args, "yolo_input_trace", False))
        self._yolo_input_trace_emitted = False
        self._yolo_input_trace = {}

    def _record_yolo_input_trace_once(self, can, *, source: str, frame_idx=None):
        """Capture a one-shot scalar-only lineage record; no frame is copied to CPU."""
        if not getattr(self, "_yolo_input_trace_enabled", False) or getattr(self, "_yolo_input_trace_emitted", False):
            return
        self._yolo_input_trace = {
            "yolo_input_source": source,
            "yolo_input_preprocess_source": "fused_preprocess.LAST_CAN",
            "yolo_input_frame_idx": -1 if frame_idx is None else int(frame_idx),
            "yolo_input_shape": list(can.shape),
            "yolo_input_dtype": str(can.dtype),
            "yolo_input_device": str(can.device),
            "yolo_input_stride": list(can.stride()),
            "yolo_input_channels": int(can.shape[1]),
            "yolo_input_min": float(can.amin().item()),
            "yolo_input_max": float(can.amax().item()),
        }
        self._yolo_input_trace_emitted = True
        self._log("[YOLO_INPUT_TRACE] " + str(self._yolo_input_trace))

    def finalize_gpu_stage_profile(self):
        profiler=getattr(self,"_gpu_stage_profiler",None)
        summary = profiler.write_json(getattr(self,"_gpu_stage_profile_output",r"C:\plate_runtime\final_wise_gpu_resident\gpu_stage_profile.json")) if profiler and profiler.enabled else {}
        if summary:
            stages = summary.get("stages", {})
            parent_avg = float(stages.get("preprocess", {}).get("avg_ms", 0.0))
            substage_names = [name for name in stages if name.startswith("preprocess_")]
            top_level_names = [name for name in stages if not name.startswith("preprocess_")]
            substage_sum = sum(float(stages[name].get("avg_ms", 0.0)) for name in substage_names)
            largest = max(substage_names, key=lambda name: float(stages[name].get("avg_ms", 0.0)), default="")
            summary.update({
                "preprocess_parent_avg_ms": parent_avg,
                "preprocess_substage_sum_avg_ms": substage_sum,
                "preprocess_unattributed_avg_ms": max(0.0, parent_avg - substage_sum),
                "preprocess_largest_substage": largest,
                "preprocess_largest_substage_avg_ms": float(stages.get(largest, {}).get("avg_ms", 0.0)),
                # `preprocess` is the parent boundary; its substage boundaries
                # are diagnostic children and must not be counted twice.
                "gpu_stage_total_avg_ms": sum(float(stages[name].get("avg_ms", 0.0)) for name in top_level_names),
            })
            profile_path = summary.get("profile_output_path")
            if profile_path:
                Path(profile_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._gpu_stage_profile_summary = summary
        return summary

    def _gpu_ms(self, fn, profile: bool, stage: str = ""):
        """Compatibility wrapper: never synchronize the device per stage.

        Detailed GPU timing is deferred to the stage profiler; the legacy
        per-frame profile value intentionally remains zero rather than
        serialising CUDA work merely to obtain a host float.
        """
        profiler=getattr(self,"_gpu_stage_profiler",None)
        record=getattr(self,"_gpu_stage_active_record",None)
        if profiler is not None and profiler.enabled and record is not None and stage:
            return profiler.measure(record,stage,fn),0.0
        if not profile:
            return fn(), 0.0
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); out = fn(); e.record()
        return out, 0.0

    def _yolo_batch_stats(self, enabled: float = 1.0) -> dict[str, float]:
        calls = int(getattr(self, "yolo_batch_call_count", 0))
        size_sum = int(getattr(self, "yolo_batch_size_sum", 0))
        return {
            "yolo_batch_enabled": float(enabled),
            "yolo_batch_size": float(getattr(self, "yolo_batch_last_size", 0)),
            "yolo_batch_call_count": float(calls),
            "yolo_batch_item_count": float(getattr(self, "yolo_batch_item_count", 0)),
            "yolo_batch_avg_size": float(size_sum) / float(max(1, calls)),
            "yolo_batch_last_size": float(getattr(self, "yolo_batch_last_size", 0)),
            "yolo_batch_fallback_single_count": float(getattr(self, "yolo_batch_fallback_single_count", 0)),
        }

    def _prepare_yolo_batch_can(self, frame_u8: torch.Tensor, gate=None, profile: bool = False, frame_idx: int | None = None, gt_box_full: torch.Tensor | None = None) -> dict[str, object]:
        self._assert_cuda(frame_u8, "stage0_output")
        h_full, w_full = frame_u8.shape[-2:]
        if gate is None:
            gx = int(w_full * 0.15)
            gy = int(h_full * 0.15)
            gate_x2 = int(w_full * 0.98)
            gate_y2 = int(h_full * 0.98)
            gw = max(1, gate_x2 - gx)
            gh = max(1, gate_y2 - gy)
            gate = (gx, gy, gw, gh)
        (_, roi), _ = self._gpu_ms(lambda: self.stage1_stabilize(frame_u8, gate), profile)
        feat4, _ = self._gpu_ms(lambda: fused_preprocess(roi), profile)
        evidence_channel = feat4[:, 0:1]
        can = getattr(gpu_pre_mod, "LAST_CAN", None)
        if can is None:
            can = getattr(gpu_pre_mod, "LAST_CANDIDATE_SOURCE", None)
        if can is None:
            can = evidence_channel
        can = can.to(device=evidence_channel.device, dtype=evidence_channel.dtype).clamp(0.0, 1.0)
        self._record_yolo_input_trace_once(can, source="fused_preprocess_last_can", frame_idx=frame_idx)
        return {
            "frame_u8": frame_u8,
            "gate": gate,
            "profile": profile,
            "frame_idx": frame_idx,
            "gt_box_full": gt_box_full,
            "can": can,
        }

    def step_yolo_batch(self, frames_u8: list[torch.Tensor], frame_indices: list[int], profile: bool = False, gate=None, gt_boxes_full: list[torch.Tensor | None] | None = None):
        if len(frames_u8) != len(frame_indices):
            raise ValueError("frames_u8 and frame_indices must have the same length")
        if not frames_u8:
            return []
        if self.yolo_detector is None or not hasattr(self.yolo_detector, "detect_batch"):
            outs = []
            for frame_u8, frame_idx in zip(frames_u8, frame_indices):
                self.yolo_batch_fallback_single_count += 1
                outs.append(self.step(frame_u8, gate=gate, profile=profile, frame_idx=frame_idx))
            return outs

        contexts = []
        cans = []
        for pos, (frame_u8, frame_idx) in enumerate(zip(frames_u8, frame_indices)):
            gt_box_full = None if gt_boxes_full is None or pos >= len(gt_boxes_full) else gt_boxes_full[pos]
            ctx = self._prepare_yolo_batch_can(frame_u8, gate=gate, profile=profile, frame_idx=frame_idx, gt_box_full=gt_box_full)
            contexts.append(ctx)
            cans.append(ctx["can"])

        cans_b = torch.cat(cans, dim=0).contiguous()
        yolo_outputs, batch_ms = self._gpu_ms(lambda: self.yolo_detector.detect_batch(cans_b), profile)
        batch_size = len(contexts)
        self.yolo_batch_call_count += 1
        self.yolo_batch_item_count += batch_size
        self.yolo_batch_size_sum += batch_size
        self.yolo_batch_last_size = batch_size
        per_frame_ms = float(batch_ms) / float(max(1, batch_size))

        outs = []
        for ctx, yolo_dets in zip(contexts, yolo_outputs):
            out = self.step(
                ctx["frame_u8"],
                gate=ctx["gate"],
                profile=profile,
                frame_idx=ctx["frame_idx"],
                gt_box_full=ctx["gt_box_full"],
                _yolo_dets_override=yolo_dets,
                _yolo_batch_stage_ms=per_frame_ms,
            )
            if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], dict):
                out[1].update(self._yolo_batch_stats(enabled=1.0))
                out[1]["yolo_batch_size"] = float(batch_size)
            outs.append(out)
        return outs

    def step(
        self,
        frame_u8: torch.Tensor,
        gate=None,
        profile: bool = False,
        frame_idx: int | None = None,
        gt_box_full: torch.Tensor | None = None,
        _yolo_dets_override: torch.Tensor | None = None,
        _yolo_batch_stage_ms: float = 0.0,
    ):
        self._assert_cuda(frame_u8, "stage0_output")
        self._current_frame_idx = frame_idx
        self._drain_fastplate_results(max_items=128)
        self._maybe_flush_event_fusion_ocr_by_frame(frame_idx)
        times = {}
        candidate_track_ids = None
        candidate_track_ids_cpu = []
        num_tracks = 0
        track_debug = []
        febam_debug = []
        ocr_debug = []
        final_plate_if_available = None
        t0 = time.perf_counter()
        h_full, w_full = frame_u8.shape[-2:]
        if gate is None:
            gx = int(w_full * 0.15)
            gy = int(h_full * 0.15)
            gate_x2 = int(w_full * 0.98)
            gate_y2 = int(h_full * 0.98)
            gw = max(1, gate_x2 - gx)
            gh = max(1, gate_y2 - gy)
            gate = (gx, gy, gw, gh)

        gate_x1, gate_y1, gate_w, gate_h = [int(v) for v in gate]
        gate_x2 = gate_x1 + gate_w
        gate_y2 = gate_y1 + gate_h
        self._current_gate_size = (int(gate_w), int(gate_h))
        times["gate_roi_x1"] = float(gate_x1)
        times["gate_roi_y1"] = float(gate_y1)
        times["gate_roi_x2"] = float(gate_x2)
        times["gate_roi_y2"] = float(gate_y2)

        profiler=getattr(self,"_gpu_stage_profiler",None); self._gpu_stage_active_record=profiler.begin_frame(frame_idx) if profiler else None
        (_, roi), times["stage1_stabilize_ms"] = self._gpu_ms(lambda: self.stage1_stabilize(frame_u8, gate), profile, "stabilize")
        if hasattr(gpu_pre_mod, "set_deferred_preprocess_profile_record"):
            gpu_pre_mod.set_deferred_preprocess_profile_record(profiler, self._gpu_stage_active_record)
        feat4, times["stage2_preprocess_ms"] = self._gpu_ms(lambda: fused_preprocess(roi), profile, "preprocess")
        times.update({k: float(v) for k, v in getattr(gpu_pre_mod, "LAST_PREPROCESS_PROFILE", {}).items()})
        evidence_channel, gx, gy = feat4[:,0:1], feat4[:,2:3], feat4[:,3:4]
        can = getattr(gpu_pre_mod, "LAST_CAN", None)
        if can is None:
            can = getattr(gpu_pre_mod, "LAST_CANDIDATE_SOURCE", None)
        equ = getattr(gpu_pre_mod, "LAST_EQU", None)
        if equ is None:
            equ = getattr(gpu_pre_mod, "LAST_EQUALIZED", None)
        if equ is None:
            equ = getattr(gpu_pre_mod, "LAST_LOCAL_CONTRAST", None)
        gray_stretched = getattr(gpu_pre_mod, "LAST_GRAY_STRETCHED", None)
        if gray_stretched is None:
            gray_stretched = getattr(gpu_pre_mod, "LAST_FUZZY_STRETCHED", None)
        if can is not None:
            can = can.to(device=evidence_channel.device, dtype=evidence_channel.dtype).clamp(0.0, 1.0)
            self._record_yolo_input_trace_once(can, source="fused_preprocess_last_can", frame_idx=frame_idx)
        if equ is not None:
            equ = equ.to(device=evidence_channel.device, dtype=evidence_channel.dtype).clamp(0.0, 1.0)
        if gray_stretched is not None:
            gray_stretched = gray_stretched.to(device=evidence_channel.device, dtype=evidence_channel.dtype).clamp(0.0, 1.0)
        fuzzy_evidence_region = getattr(gpu_pre_mod, "LAST_FUZZY_EVIDENCE_REGION", None)
        if fuzzy_evidence_region is None:
            fuzzy_evidence_region = torch.zeros_like(evidence_channel)

        fuzzy_evidence = fuzzy_evidence_region.to(device=evidence_channel.device, dtype=evidence_channel.dtype).clamp(0.0, 1.0)
        # temporal vehicle ROI gate + fuzzy region plate candidates + MLP selector
        # fuzzy_region is the plate candidate source; fuzzy_temporal is only the vehicle ROI gate.
        fuzzy_region = fuzzy_evidence
        if self.fuzzy_memory is None or self.fuzzy_memory.shape != fuzzy_region.shape:
            self.fuzzy_memory = fuzzy_region.clone()
        else:
            self.fuzzy_memory = 0.90 * self.fuzzy_memory + 0.10 * fuzzy_region

        _, _, roi_h, roi_w = fuzzy_region.shape
        yy = torch.linspace(0.0, 1.0, roi_h, device=fuzzy_region.device, dtype=fuzzy_region.dtype).view(1, 1, roi_h, 1)
        xx = torch.linspace(0.0, 1.0, roi_w, device=fuzzy_region.device, dtype=fuzzy_region.dtype).view(1, 1, 1, roi_w)
        diagonal_band = torch.exp(-torch.abs((yy - 0.55) - 0.75 * (xx - 0.30)) / 0.22)
        direction_prior = (0.60 + 0.40 * diagonal_band).clamp(0.0, 1.0)
        lower_prior_map = torch.exp(-torch.abs(yy - 0.80) / 0.16).clamp(0.0, 1.0)
        # temporal vehicle ROI gate + fuzzy region plate candidates + MLP selector
        fuzzy_temporal = (0.70 * self.fuzzy_memory + 0.30 * fuzzy_region).clamp(0.0, 1.0)
        fuzzy_source = fuzzy_region
        self._stage6_gx = gx
        self._stage6_fuzzy_temporal = fuzzy_temporal
        self._stage6_direction_prior = direction_prior
        self._stage6_lower_prior = lower_prior_map
        self._stage6_fuzzy_source = fuzzy_source
        self._last_fuzzy_soft_map = fuzzy_source

        times["stage3_structure_tensor_ms"] = 0.0
        times["stage4_morphology_ms"] = 0.0
        times["stage5_ccl_ms"] = 0.0
        times.update(self._yolo_batch_stats(enabled=0.0))
        if self.yolo_detector is not None and can is not None:
            if _yolo_dets_override is not None:
                candidates = _yolo_dets_override.to(device=self.device, dtype=torch.float32)
                times["stage6_9_head_ms"] = float(_yolo_batch_stage_ms)
                times["yolo_batch_override_used"] = 1.0
            else:
                candidates, times["stage6_9_head_ms"] = self._gpu_ms(lambda: self.yolo_detector.detect(can), profile, "yolo_wrapper")
                times["yolo_batch_override_used"] = 0.0
            yolo_count = float(candidates.shape[0])
            self._stage6_debug = {
                "stage6_yolo_used": 1.0,
                "stage6_yolo_count": yolo_count,
                "stage6_raw_component_count": yolo_count,
                "stage6_filter_pass_count": yolo_count,
                "stage6_nms_post_count": yolo_count,
                "stage6_topk_count": yolo_count,
                "candidate_count": torch.tensor([yolo_count], device=self.device, dtype=evidence_channel.dtype),
                "top_candidate_score": candidates[:1, 4].detach() if candidates.numel() > 0 else torch.tensor([0.0], device=self.device, dtype=evidence_channel.dtype),
                "plate_candidate_boxes_roi": candidates[:, :4] if candidates.numel() > 0 else torch.zeros((0, 4), device=self.device, dtype=evidence_channel.dtype),
                "plate_selected_box_roi": candidates[:, :4] if candidates.numel() > 0 else torch.zeros((0, 4), device=self.device, dtype=evidence_channel.dtype),
                "candidate_final_score": candidates[:, 4] if candidates.numel() > 0 else torch.zeros((0,), device=self.device, dtype=evidence_channel.dtype),
                "candidate_source_map": can,
            }
            times["stage6_yolo_used"] = 1.0
            times["stage6_yolo_count"] = yolo_count
        else:
            candidates, times["stage6_9_head_ms"] = self._gpu_ms(
                lambda: self._propose_plate_candidates_vehicle_gate_gpu(
                    fuzzy_region,
                    fuzzy_temporal,
                    value_map=getattr(gpu_pre_mod, "LAST_REGION_VALUE", None),
                    gx_map=gx,
                    gy_map=gy,
                    gate_offset_xy=(gate[0], gate[1]),
                    gt_box_full=gt_box_full,
                    pre_topk=50,
                    final_topk=1,
                ),
                profile,
            )
            times["stage6_yolo_used"] = 0.0
            times["stage6_yolo_count"] = 0.0

        raw_yolo_candidates = candidates.detach() if getattr(self, "labeling_febam_sink", None) is not None else None
        if self.use_mlp_updater and candidates.numel() > 0 and self.mlp_updater is not None and equ is not None and can is not None and gray_stretched is not None:
            candidates = self.mlp_updater.rerank(candidates, equ, can, gray_stretched)
            times["stage7_mlp_used"] = 1.0
            times["stage7_mlp_top_score"] = float(candidates[:, 4].max().item()) if candidates.numel() > 0 else 0.0
            self._stage6_debug["candidate_final_score"] = candidates[:, 4].detach()
            self._stage6_debug["top_candidate_score"] = candidates[:1, 4].detach()
        else:
            times["stage7_mlp_used"] = 0.0
            times["stage7_mlp_top_score"] = float(candidates[:, 4].max().item()) if candidates.numel() > 0 else 0.0
        stage6_dbg = getattr(self, "_stage6_debug", {})
        for k, v in stage6_dbg.items():
            if isinstance(v, torch.Tensor):
                continue
            try:
                times[k] = float(v)
            except Exception:
                continue
        times["stage6_raw_box_count"] = float(times.get("stage6_raw_component_count", 0.0))
        times["stage6_filtered_box_count"] = float(times.get("stage6_filter_pass_count", 0.0))
        times["stage6_score_pre_count"] = float(getattr(self._stage6_debug.get("score_pre_boxes", torch.zeros((0,4))), "shape", [0])[0])
        times["stage6_nms_count"] = float(getattr(self._stage6_debug.get("nms_boxes", torch.zeros((0,4))), "shape", [0])[0])
        times["stage6_final_count"] = float(getattr(self._stage6_debug.get("final_boxes", torch.zeros((0,4))), "shape", [0])[0])
        (active_ids, track_scores), times["stage10_tracker_ms"] = self._gpu_ms(lambda: self.stage10_tracker(candidates), profile, "tracker")
        candidate_track_ids = getattr(self, "_last_candidate_track_ids", None)
        if candidate_track_ids is not None:
            if torch.is_tensor(candidate_track_ids):
                candidate_track_ids_cpu = candidate_track_ids.detach().cpu().tolist()
            else:
                candidate_track_ids_cpu = list(candidate_track_ids)
        num_tracks = int(active_ids.numel())

        def febam_part():
            if active_ids.numel() == 0 or candidates.numel() == 0:
                return torch.zeros((0,), device=self.device, dtype=torch.long)
            n = int(active_ids.numel())
            boxes = self.track_state[active_ids[:n], :4]
            conf = track_scores[:n].reshape(-1)
            bw = (boxes[:,2]-boxes[:,0]).clamp(min=1.0)
            bh = (boxes[:,3]-boxes[:,1]).clamp(min=1.0)
            area_norm = ((bw*bh) / float(fuzzy_source.shape[-2]*fuzzy_source.shape[-1] + 1e-6)).reshape(-1)
            aspect_score = torch.exp(-torch.abs((bw/(bh+1e-6)) - 4.0) / 4.0).clamp(0.0,1.0).reshape(-1)
            sharpness = conf.clamp(0.0,1.0).reshape(-1)
            conf = conf.clamp(0.0,1.0).reshape(-1)
            area_term = (1.0 - area_norm.clamp(0.0,1.0)).reshape(-1)
            m = min(conf.numel(), area_term.numel(), sharpness.numel(), aspect_score.numel(), int(active_ids.numel()))
            if m == 0:
                return torch.zeros((0,), device=self.device, dtype=torch.long)
            feats = torch.stack([conf[:m], area_term[:m], sharpness[:m], aspect_score[:m]], dim=1)
            return self.febam.update(active_ids[:m], feats)

        confirmed, times["stage11_febam_ms"] = self._gpu_ms(febam_part, profile, "febam")
        times["stage11_febam_mode_id"] = self.febam_mode_id
        if raw_yolo_candidates is not None:
            self.labeling_febam_sink.observe(
                frame_idx=int(frame_idx if frame_idx is not None else -1),
                raw_candidates=raw_yolo_candidates,
                tracked_candidates=candidates,
                candidate_track_ids=candidate_track_ids,
                confirmed_track_ids=confirmed,
                febam_energy=self.febam.energy,
                gate=gate,
                group_key_resolver=self._resolve_string_febam_group_id,
                event_group_key_resolver=self._b5_group_key,
            )
        _, times["stage12_ocr_trigger_ms"] = self._gpu_ms(lambda: self.stage12_ocr_trigger(frame_u8, candidates, confirmed, gate, "", gray_stretched=gray_stretched, frame_idx=frame_idx), profile, "ocr_trigger")
        if profiler: profiler.end_frame(self._gpu_stage_active_record)
        self._gpu_stage_active_record=None
        self._drain_fastplate_results(max_items=128)
        times["stage12_gray_stretched_ocr_used"] = 1.0 if getattr(self, "_last_gray_stretched_ocr_used", False) else 0.0
        times.update(self._fastplate_async_stats())
        times.update(self._middle_slot_stats())
        primary_track_id = int(confirmed[0].item()) if confirmed.numel() > 0 else (int(active_ids[0].item()) if active_ids.numel() > 0 else -1)
        times["num_yolo_candidates"] = float(times.get("stage6_yolo_count", 0.0))
        times["num_tracks"] = float(active_ids.numel())
        times["track_id"] = float(primary_track_id)
        times["track_hits"] = float(self.track_state[primary_track_id, 11].item()) if primary_track_id >= 0 else 0.0
        times["febam_alpha"] = float(getattr(self.febam, "alpha", 0.0))
        times["febam_delta"] = float(self.febam.last_delta[primary_track_id].item()) if primary_track_id >= 0 and hasattr(self.febam, "last_delta") else 0.0
        times["febam_confirmed"] = 1.0 if confirmed.numel() > 0 else 0.0
        times["ocr_conf"] = float(getattr(self, "_last_ocr_conf", 0.0))

        sec = (float(frame_idx) / max(self.fps, 1e-6)) if frame_idx is not None else 0.0
        target_dump = int(sec) in self.debug_dump_seconds
        periodic_dump = self.debug_save_every_sec > 0 and abs((sec / self.debug_save_every_sec) - round(sec / self.debug_save_every_sec)) < 1e-3
        confirmed_dump = confirmed.numel() > 0
        event_dump = confirmed_dump or (self.debug_stage_log and int(candidates.shape[0]) > 0)

        save_video = bool(self.debug_save_video)
        save_frame = bool(self.debug_save_frames_dir) and (target_dump or periodic_dump)
        save_event = bool(self.debug_event_dir) and (target_dump or confirmed_dump or event_dump)
        save_binary = bool(self.debug_save_frames_dir) and target_dump

        if save_video or save_frame or save_event or save_binary:
            # DEBUG ONLY CPU copy
            frame_rgb = frame_u8[0].permute(1,2,0).detach().cpu().numpy()
            # DEBUG ONLY CPU copy
            candidates_cpu = candidates.detach().cpu().numpy()
            # DEBUG ONLY CPU copy
            confirmed_cpu = confirmed.detach().cpu().numpy()
            save_direct_debug = save_frame or save_binary
            save_candidate_stage = False
            stage_debug = self._stage6_debug
            zero_boxes_debug = torch.zeros((0, 4), device=self.device)
            zero_scores_debug = torch.zeros((0,), device=self.device)
            item = {
                "kind": "debug_snapshot",
                "frame_idx": int(frame_idx if frame_idx is not None else -1),
                "sec": sec,
                "frame_rgb": frame_rgb,
                "gate": gate,
                "candidates": candidates_cpu,
                "confirmed_ids": confirmed_cpu,
                "candidate_track_ids": candidate_track_ids_cpu,
                "save_video": save_video,
                "save_frame": save_frame,
                "save_event": save_event,
                "save_binary": save_binary,
                "binary_mask": None,
                "gray_map": (stage_debug.get("gray_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "fuzzy_stretched_map": (stage_debug.get("fuzzy_stretched_map", stage_debug.get("gray_map", torch.zeros_like(fuzzy_source))).detach().float().cpu().numpy() if save_direct_debug else None),
                "uniform_mu": (stage_debug.get("uniform_mu", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "blackhat_weight_map": (stage_debug.get("blackhat_weight_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "candidate_source_map": (stage_debug.get("candidate_source_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "scharrx_map": (stage_debug.get("scharrx_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "morphology_map": (stage_debug.get("morphology_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "morphology_before_merge_map": (stage_debug.get("morphology_before_merge_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "stroke_components_map": (stage_debug.get("stroke_components_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "motion_prior_map": (stage_debug.get("motion_prior_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "same_row_merge_map": (stage_debug.get("same_row_merge_map", torch.zeros_like(fuzzy_source)).detach().float().cpu().numpy() if save_direct_debug else None),
                "stroke_merge_pair_boxes": (stage_debug.get("stroke_merge_pair_boxes", zero_boxes_debug).detach().float().cpu().numpy() if save_direct_debug else None),
                "uniform_sigma_sq": (stage_debug.get("uniform_sigma_sq", torch.zeros((1,), device=self.device)).detach().float().cpu().numpy() if save_direct_debug else None),
                "uniform_weight": (stage_debug.get("uniform_weight", torch.zeros((1,), device=self.device)).detach().float().cpu().numpy() if save_direct_debug else None),
                "blackhat_alpha": (stage_debug.get("blackhat_alpha", torch.zeros((1,), device=self.device)).detach().float().cpu().numpy() if save_direct_debug else None),
                "event_overlay_candidate_boxes_full": (stage_debug.get("plate_candidate_boxes_full", zero_boxes_debug).detach().float().cpu().numpy() if save_event else None),
                "event_overlay_candidate_boxes_roi": (stage_debug.get("plate_candidate_boxes_roi", zero_boxes_debug).detach().float().cpu().numpy() if save_event else None),
                "event_overlay_candidate_scores": (stage_debug.get("candidate_final_score", zero_scores_debug).detach().float().cpu().numpy() if save_event else None),
                "event_overlay_selected_boxes_full": (stage_debug.get("plate_selected_box_full", zero_boxes_debug).detach().float().cpu().numpy() if save_event else None),
                "ocr_triggered": bool(confirmed.numel() > 0),
                "save_candidate_stage": False,
                "stage_boxes": {},
            }
            image_item = item.copy()
            image_item["save_video"] = False
            video_item = item.copy()
            video_item["save_frame"] = False
            video_item["save_event"] = False
            video_item["save_binary"] = False
            video_item["save_candidate_stage"] = False
            if save_video:
                self.video_writer_q.put(video_item, block=True)
            if save_frame or save_event or save_binary or save_candidate_stage:
                try:
                    self.image_writer_q.put_nowait(image_item)
                except queue.Full:
                    self._log(f"Stage13 image_writer_queue_full drop frame_idx={frame_idx}")

        times["total_frame_ms"] = (time.perf_counter() - t0) * 1000.0
        times["candidates_count"] = float(candidates.shape[0])
        times["tracks_count"] = float(active_ids.numel())
        times["confirmed_count"] = float(confirmed.numel())
        times["ocr_triggered"] = 1.0 if confirmed.numel() > 0 else 0.0
        times["stage6_debug_saved_count"] = 1.0 if (save_frame or save_binary) else 0.0
        times["uniform_sigma_sq"] = float(self._stage6_debug.get("uniform_sigma_sq", torch.zeros((1,), device=self.device)).reshape(-1)[0].item()) if "uniform_sigma_sq" in self._stage6_debug else float(times.get("stage2_uniform_sigma_sq", 0.0))
        times["blackhat_alpha"] = float(self._stage6_debug.get("blackhat_alpha", torch.zeros((1,), device=self.device)).reshape(-1)[0].item()) if "blackhat_alpha" in self._stage6_debug else float(times.get("stage2_blackhat_alpha", 0.0))
        times["uniform_weight"] = float(self._stage6_debug.get("uniform_weight", torch.zeros((1,), device=self.device)).reshape(-1)[0].item()) if "uniform_weight" in self._stage6_debug else float(times.get("stage2_uniform_weight", 0.30))
        times["candidate_count"] = float(self._stage6_debug.get("candidate_count", torch.zeros((1,), device=self.device)).reshape(-1)[0].item()) if "candidate_count" in self._stage6_debug else float(candidates.shape[0])
        times["top_candidate_score"] = float(self._stage6_debug.get("top_candidate_score", torch.zeros((1,), device=self.device)).reshape(-1)[0].item()) if "top_candidate_score" in self._stage6_debug else 0.0
        times["scharr_mean"] = float(self._stage6_debug.get("scharr_mean", torch.zeros((1,), device=self.device)).reshape(-1)[0].item()) if "scharr_mean" in self._stage6_debug else float(times.get("stage2_scharrx_mean", 0.0))
        times["scharr_max"] = float(self._stage6_debug.get("scharr_max", torch.zeros((1,), device=self.device)).reshape(-1)[0].item()) if "scharr_max" in self._stage6_debug else float(times.get("stage2_scharrx_max", 0.0))
        if self.debug_stage_log or self.debug_log_txt is not None:
            self._log(
                f"CANDIDATE_DEBUG frame={frame_idx if frame_idx is not None else -1} "
                f"LAST_UNIFORMITY_SIGMA_SQ={float(times.get('uniform_sigma_sq', 0.0)):.8f} "
                f"LAST_UNIFORM_WEIGHT={float(times.get('uniform_weight', 0.0)):.2f} "
                f"LAST_BLACKHAT_ALPHA={float(times.get('blackhat_alpha', 0.0)):.8f} "
                f"gate_roi_x1={int(times.get('gate_roi_x1', 0.0))} "
                f"gate_roi_y1={int(times.get('gate_roi_y1', 0.0))} "
                f"gate_roi_x2={int(times.get('gate_roi_x2', 0.0))} "
                f"gate_roi_y2={int(times.get('gate_roi_y2', 0.0))} "
                f"gate_roi=({int(times.get('gate_roi_x1', 0.0))},"
                f"{int(times.get('gate_roi_y1', 0.0))})-"
                f"({int(times.get('gate_roi_x2', 0.0))},"
                f"{int(times.get('gate_roi_y2', 0.0))}) "
                f"candidate_count={int(times.get('candidate_count', 0.0))} "
                f"top_candidate_score={float(times.get('top_candidate_score', 0.0)):.6f} "
                f"num_yolo_candidates={int(times.get('num_yolo_candidates', 0.0))} "
                f"num_tracks={int(times.get('num_tracks', 0.0))} "
                f"track_id={int(times.get('track_id', -1.0))} "
                f"track_hits={int(times.get('track_hits', 0.0))} "
                f"febam_alpha={float(times.get('febam_alpha', 0.0)):.2f} "
                f"febam_delta={float(times.get('febam_delta', 0.0)):.4f} "
                f"febam_confirmed={int(times.get('febam_confirmed', 0.0))} "
                f"ocr_text={getattr(self, '_last_ocr_text', '')} "
                f"ocr_conf={float(times.get('ocr_conf', 0.0)):.2f} "
                f"final_plate_if_available={getattr(self, '_last_final_plate', '')}"
            )

        return candidates, times


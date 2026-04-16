"""Inference service with platform-aware backend selection."""

import base64
import io
import json
import os
import platform
import pathlib
from typing import Optional, Tuple

import cv2
import numpy as np


CLASSES = ["Humain", "Objet_Chaud"]
IMG_SIZE = 320
HUMAN_CLASS_INDEX = 0
HOT_OBJECT_CLASS_INDEX = 1
THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24

# Constants aligned with the thermal preprocessing used during training.
TEMP_MIN_GLOBALE = 5.0
TEMP_MAX_GLOBALE = 55.0
SCALE_FACTOR = 10  # 32x24 -> 320x240


def _machine_name() -> str:
    """Return normalized machine architecture string."""
    return platform.machine().lower()


def _cpuinfo_text() -> str:
    """Return /proc/cpuinfo content when available."""
    cpuinfo_path = pathlib.Path("/proc/cpuinfo")
    return cpuinfo_path.read_text(errors="ignore").lower() if cpuinfo_path.exists() else ""


def letterbox(img: np.ndarray, size: int = IMG_SIZE) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize with unchanged aspect ratio and pad to model size."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dx, dy = (size - nw) / 2, (size - nh) / 2

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dy - 0.1)), int(round(dy + 0.1))
    left, right = int(round(dx - 0.1)), int(round(dx + 0.1))
    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return img, r, (dx, dy)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45):
    """Apply non-maximum suppression and return kept indexes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]

    return keep


def postprocess(
    outputs,
    orig_hw,
    ratio,
    pad,
    conf_thr: float = 0.25,
    iou_thr: float = 0.45,
):
    """Post-process model outputs and return boxes, scores and class ids."""
    nc = len(CLASSES)
    raw = np.concatenate([np.squeeze(o) for o in outputs], axis=-1)
    if raw.shape[0] == 4 + nc:
        raw = raw.T

    cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2

    cls_scores = raw[:, 4:4 + nc]
    cls_ids = np.argmax(cls_scores, axis=1)
    conf = np.max(cls_scores, axis=1)

    mask = conf >= conf_thr
    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    conf, cls_ids = conf[mask], cls_ids[mask]

    if len(conf) == 0:
        return np.empty((0, 4)), np.array([]), np.array([], dtype=int)

    x1 = (x1 - pad[0]) / ratio
    y1 = (y1 - pad[1]) / ratio
    x2 = (x2 - pad[0]) / ratio
    y2 = (y2 - pad[1]) / ratio

    oh, ow = orig_hw
    x1, x2 = np.clip(x1, 0, ow), np.clip(x2, 0, ow)
    y1, y2 = np.clip(y1, 0, oh), np.clip(y2, 0, oh)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = []
    for c in range(nc):
        idx = np.where(cls_ids == c)[0]
        if len(idx) == 0:
            continue
        k = nms(boxes[idx], conf[idx], iou_thr)
        keep.extend(idx[k])

    keep = sorted(keep)
    return boxes[keep], conf[keep], cls_ids[keep]


def thermal_to_bgr(thermal: np.ndarray) -> np.ndarray:
    """Convert thermal frame (24x32 float) into BGR image (240x320)."""
    img_clipped = np.clip(thermal, TEMP_MIN_GLOBALE, TEMP_MAX_GLOBALE)
    img_8u = ((img_clipped - TEMP_MIN_GLOBALE) / (TEMP_MAX_GLOBALE - TEMP_MIN_GLOBALE) * 255.0).astype(np.uint8)
    large_img = cv2.resize(
        img_8u,
        (THERMAL_WIDTH * SCALE_FACTOR, THERMAL_HEIGHT * SCALE_FACTOR),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(large_img, cv2.COLOR_GRAY2BGR)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure the input image is in BGR format."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Format image non supporté: shape={image.shape}")


def decode_npy_payload(payload) -> np.ndarray:
    """Decode websocket payload to numpy array from bytes or JSON string."""
    if isinstance(payload, bytes):
        try:
            with io.BytesIO(payload) as buffer:
                arr = np.load(buffer, allow_pickle=False)
            return np.asarray(arr)
        except Exception:
            if len(payload) % 4 != 0:
                raise ValueError(
                    "Payload binaire invalide: taille non multiple de 4 pour des float32"
                )

            arr = np.frombuffer(payload, dtype="<f4")
            if arr.size == THERMAL_WIDTH * THERMAL_HEIGHT:
                return arr.reshape((THERMAL_HEIGHT, THERMAL_WIDTH))
            return arr

    if isinstance(payload, str):
        message = payload.strip()
        if message.startswith("{"):
            obj = json.loads(message)
            if "npy_base64" in obj:
                raw = base64.b64decode(obj["npy_base64"])
                with io.BytesIO(raw) as buffer:
                    arr = np.load(buffer, allow_pickle=False)
                return np.asarray(arr)
            if "float32_base64" in obj:
                raw = base64.b64decode(obj["float32_base64"])
                if len(raw) % 4 != 0:
                    raise ValueError(
                        "float32_base64 invalide: taille non multiple de 4"
                    )
                arr = np.frombuffer(raw, dtype="<f4")
                if arr.size == THERMAL_WIDTH * THERMAL_HEIGHT:
                    return arr.reshape((THERMAL_HEIGHT, THERMAL_WIDTH))
                return arr
        raise ValueError("Message texte non supporté (attendu JSON avec npy_base64)")

    raise ValueError("Type de payload websocket non supporté")


class HumanDetectorNPU:
    """RKNNLite-backed detector for RK3588 NPU execution."""

    _instance: Optional["HumanDetectorNPU"] = None

    def __new__(cls, model_path: str, conf: float, iou: float):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str, conf: float, iou: float):
        if self._initialized:
            return

        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:
            raise RuntimeError("rknnlite is not available in this environment") from exc

        self.conf = conf
        self.iou = iou
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Échec load_rknn: {model_path} (ret={ret})")

        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"Échec init_runtime (ret={ret})")

        self._initialized = True

    def infer_detections(self, image: np.ndarray) -> dict:
        """Run inference and return human and hot-object counts."""
        if (
            image.ndim == 2
            and image.shape == (THERMAL_HEIGHT, THERMAL_WIDTH)
            and image.dtype in (np.float32, np.float64)
        ):
            bgr = thermal_to_bgr(image)
        else:
            bgr = ensure_bgr(image)
        img320, ratio, pad = letterbox(bgr, IMG_SIZE)
        inp = cv2.cvtColor(img320, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0)

        outputs = self.rknn.inference(inputs=[inp])
        _, _, cls_ids = postprocess(
            outputs,
            orig_hw=bgr.shape[:2],
            ratio=ratio,
            pad=pad,
            conf_thr=self.conf,
            iou_thr=self.iou,
        )
        human_count = int(np.sum(cls_ids == HUMAN_CLASS_INDEX)) if len(cls_ids) else 0
        hot_object_count = int(np.sum(cls_ids == HOT_OBJECT_CLASS_INDEX)) if len(cls_ids) else 0
        return {"human_count": human_count, "hot_object_count": hot_object_count}

    def release(self):
        """Release RKNN runtime resources."""
        if getattr(self, "rknn", None) is not None:
            self.rknn.release()




class HumanDetectorCPU:
    """RKNN simulator-backed detector for non-RK3588 machines."""

    _instance: Optional["HumanDetectorCPU"] = None

    def __new__(cls, model_path: str, conf: float, iou: float):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str, conf: float, iou: float):
        if self._initialized:
            return

        try:
            from rknn.api import RKNN
        except Exception as exc:
            raise RuntimeError("rknn toolkit is not available in this environment") from exc

        self.conf = conf
        self.iou = iou
        self.rknn = RKNN(verbose=False)

        # Required configuration before loading ONNX in simulator mode.
        self.rknn.config(target_platform="rk3588")

        # Load ONNX model for PC simulator path.
        ret = self.rknn.load_onnx(model_path)
        if ret != 0:
            raise RuntimeError("load_onnx a échoué")

        # Build model for simulator.
        ret = self.rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError("build a échoué")

        # Init simulator runtime (CPU).
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime (simulateur) a échoué")
        self._initialized = True

    def infer_detections(self, image: np.ndarray) -> dict:
        """Run inference and return human and hot-object counts."""
        if (
            image.ndim == 2
            and image.shape == (THERMAL_HEIGHT, THERMAL_WIDTH)
            and image.dtype in (np.float32, np.float64)
        ):
            bgr = thermal_to_bgr(image)
        else:
            bgr = ensure_bgr(image)
        img320, ratio, pad = letterbox(bgr, IMG_SIZE)
        inp = cv2.cvtColor(img320, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0)

        outputs = self.rknn.inference(inputs=[inp])
        _, _, cls_ids = postprocess(
            outputs,
            orig_hw=bgr.shape[:2],
            ratio=ratio,
            pad=pad,
            conf_thr=self.conf,
            iou_thr=self.iou,
        )
        human_count = int(np.sum(cls_ids == HUMAN_CLASS_INDEX)) if len(cls_ids) else 0
        hot_object_count = int(np.sum(cls_ids == HOT_OBJECT_CLASS_INDEX)) if len(cls_ids) else 0
        return {"human_count": human_count, "hot_object_count": hot_object_count}

    def release(self):
        """Release RKNN runtime resources."""
        if getattr(self, "rknn", None) is not None:
            self.rknn.release()




class InferenceService:
    """Select and expose the inference backend according to current machine."""

    def __init__(self,rknn_model,onnx_model,conf,iou):
        """Initialize platform-specific detector using RKNN NPU or simulator CPU."""
        m = _machine_name()
        c = _cpuinfo_text()


        self.backend = "unavailable"
        self._init_error: Optional[str] = None

        try:
            if m in ("x86_64", "amd64", "i386", "i686") or "intel" in c or "amd" in c:
                self.backend = "cpu"
                self.detector = HumanDetectorCPU(str(onnx_model), conf, iou)
            elif "rk3588" in c or "rockchip" in c:
                self.backend = "npu"
                self.detector = HumanDetectorNPU(str(rknn_model), conf, iou)
            else:
                raise RuntimeError(
                    f"Unsupported platform for inference (machine={m}). "
                    "Expected x86/amd64 or RK3588/rockchip."
                )
        except Exception as exc:
            self._init_error = str(exc)
            allow_fallback = os.environ.get("ALLOW_INFERENCE_FALLBACK", "1") == "1"
            if allow_fallback:
                self.detector = _UnavailableDetector(self._init_error)
            else:
                raise

    def infer(self, image: np.ndarray) -> dict:
        """Run inference on image and return detection counts."""
        return self.detector.infer_detections(image)

    def is_available(self) -> bool:
        """Return whether a real inference backend is available."""
        return self.backend in ("cpu", "npu")

    def get_init_error(self) -> Optional[str]:
        """Return backend initialization error if any."""
        return self._init_error

    def release(self):
        """Release detector resources."""
        self.detector.release()


class _UnavailableDetector:
    """Fallback detector used when backend initialization fails."""

    def __init__(self, reason: str):
        self.reason = reason

    def infer_detections(self, _image: np.ndarray) -> dict:
        raise RuntimeError(f"Inference backend unavailable: {self.reason}")

    def release(self):
        return None

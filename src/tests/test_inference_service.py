"""Unit tests for inference service and preprocessing utilities."""

import base64
import io
import json
import numpy as np
import pytest

from src.app.services import inference_service as inf


class DummyCPU:
    """Dummy CPU detector for platform-selection tests."""

    def __init__(self, model_path, conf, iou):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou

    def infer_detections(self, image):
        return {"human_count": 1, "hot_object_count": 2}

    def release(self):
        return None


class DummyNPU:
    """Dummy NPU detector for platform-selection tests."""

    def __init__(self, model_path, conf, iou):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou

    def infer_detections(self, image):
        return {"human_count": 3, "hot_object_count": 0}

    def release(self):
        return None


def test_thermal_to_bgr_shape_and_dtype(sample_thermal_frame):
    """Thermal conversion should return 240x320 uint8 BGR image."""
    bgr = inf.thermal_to_bgr(sample_thermal_frame)
    assert bgr.shape == (240, 320, 3)
    assert bgr.dtype == np.uint8


def test_ensure_bgr_accepts_gray(sample_gray_image):
    """Gray images should be converted to BGR."""
    out = inf.ensure_bgr(sample_gray_image)
    assert out.shape[2] == 3


def test_letterbox_output_shape(sample_bgr_image):
    """Letterbox should always output IMG_SIZE square image."""
    out, ratio, pad = inf.letterbox(sample_bgr_image, size=320)
    assert out.shape == (320, 320, 3)
    assert ratio > 0
    assert len(pad) == 2


def test_nms_keeps_non_overlapping_boxes():
    """NMS should keep at least one box among overlapping proposals."""
    boxes = np.array([[10, 10, 80, 80], [12, 12, 82, 82], [150, 150, 200, 200]], dtype=np.float32)
    scores = np.array([0.8, 0.7, 0.9], dtype=np.float32)
    kept = inf.nms(boxes, scores, iou_thr=0.4)
    assert len(kept) >= 2


def test_decode_npy_payload_from_raw_float32(sample_thermal_frame):
    """Raw float32 bytes should decode to 24x32 thermal frame."""
    payload = sample_thermal_frame.tobytes()
    arr = inf.decode_npy_payload(payload)
    assert arr.shape == (24, 32)


def test_decode_npy_payload_from_npy_bytes(sample_thermal_frame):
    """NPY bytes payload should decode as numpy array."""
    buffer = io.BytesIO()
    np.save(buffer, sample_thermal_frame)
    arr = inf.decode_npy_payload(buffer.getvalue())
    assert arr.shape == (24, 32)


def test_decode_npy_payload_from_json_base64(sample_thermal_frame):
    """JSON payload with float32_base64 should decode correctly."""
    raw = sample_thermal_frame.tobytes()
    payload = json.dumps({"float32_base64": base64.b64encode(raw).decode("utf-8")})
    arr = inf.decode_npy_payload(payload)
    assert arr.shape == (24, 32)


def test_inference_service_selects_cpu_on_x86(monkeypatch):
    """Service should choose CPU backend on x86-like machines."""
    monkeypatch.setattr(inf, "HumanDetectorCPU", DummyCPU)
    monkeypatch.setattr(inf, "HumanDetectorNPU", DummyNPU)
    monkeypatch.setattr(inf, "_machine_name", lambda: "x86_64")
    monkeypatch.setattr(inf, "_cpuinfo_text", lambda: "intel")

    service = inf.InferenceService()
    assert service.backend == "cpu"
    assert isinstance(service.detector, DummyCPU)


def test_inference_service_selects_npu_on_rk3588(monkeypatch):
    """Service should choose NPU backend on RK3588 machines."""
    monkeypatch.setattr(inf, "HumanDetectorCPU", DummyCPU)
    monkeypatch.setattr(inf, "HumanDetectorNPU", DummyNPU)
    monkeypatch.setattr(inf, "_machine_name", lambda: "aarch64")
    monkeypatch.setattr(inf, "_cpuinfo_text", lambda: "rockchip rk3588")

    service = inf.InferenceService()
    assert service.backend == "npu"
    assert isinstance(service.detector, DummyNPU)


def test_inference_service_forwards_infer(monkeypatch):
    """Service should forward infer call to detector implementation."""
    monkeypatch.setattr(inf, "HumanDetectorCPU", DummyCPU)
    monkeypatch.setattr(inf, "HumanDetectorNPU", DummyNPU)
    monkeypatch.setattr(inf, "_machine_name", lambda: "x86_64")
    monkeypatch.setattr(inf, "_cpuinfo_text", lambda: "intel")

    service = inf.InferenceService()
    out = service.infer(np.zeros((24, 32), dtype=np.float32))
    assert out == {"human_count": 1, "hot_object_count": 2}

"""Inference service with support for both NPU and CPU backends."""

import io
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from ..utils.platform_detector import PlatformDetector, InferencePlatform


logger = logging.getLogger(__name__)

CLASSES = ["Humain", "Objet_Chaud"]
HUMAN_CLASS_INDEX = 0
HOT_OBJECT_CLASS_INDEX = 1
THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24
TEMP_MIN_GLOBALE = 5.0
TEMP_MAX_GLOBALE = 55.0
SCALE_FACTOR = 10


class HumanDetectorBase(ABC):
    """
    Abstract base class for human/hot object detection.
    
    Enforces consistent interface for different inference backends.
    """
    
    def __init__(self, conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        """
        Initialize detector with inference parameters.
        
        Args:
            conf_threshold: Confidence threshold for object detection.
            iou_threshold: Intersection over Union threshold for NMS.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    @abstractmethod
    def infer_detections(self, image: np.ndarray) -> Dict[str, int]:
        """
        Perform inference on input image.
        
        Args:
            image: Input image (BGR or grayscale thermal).
        
        Returns:
            Dictionary with 'human_count' and 'hot_object_count'.
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release model resources."""
        pass


class HumanDetectorNPU(HumanDetectorBase):
    """
    RKNN NPU-based detector for RK3588/RK3566 platforms.
    
    Leverages hardware NPU acceleration via RKNN Lite API.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize NPU detector with RKNN model.
        
        Args:
            model_path: Path to RKNN model file.
            conf_threshold: Confidence threshold for detection.
            iou_threshold: IoU threshold for non-maximum suppression.
        
        Raises:
            RuntimeError: If RKNN initialization fails.
        """
        super().__init__(conf_threshold, iou_threshold)
        
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise RuntimeError("rknnlite not available. Install rknn-toolkit2.")
        
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path} (ret={ret})")
        
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize RKNN runtime (ret={ret})")
        
        logger.info("NPU detector initialized successfully")
    
    def infer_detections(self, image: np.ndarray) -> Dict[str, int]:
        """
        Run inference on image using NPU.
        
        Args:
            image: Input image or thermal frame.
        
        Returns:
            Detection counts for humans and hot objects.
        """
        # Convert thermal frame to BGR if needed
        if (
            image.ndim == 2
            and image.shape == (THERMAL_HEIGHT, THERMAL_WIDTH)
            and image.dtype in (np.float32, np.float64)
        ):
            bgr = self._thermal_to_bgr(image)
        else:
            bgr = self._ensure_bgr(image)
        
        # Preprocess
        img320, ratio, pad = self._letterbox(bgr, img_size=320)
        inp = cv2.cvtColor(img320, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0)
        
        # Inference
        outputs = self.rknn.inference(inputs=[inp])
        _, _, cls_ids = self._postprocess(
            outputs,
            orig_hw=bgr.shape[:2],
            ratio=ratio,
            pad=pad,
            conf_thr=self.conf_threshold,
            iou_thr=self.iou_threshold,
        )
        
        # Count detections
        human_count = int(np.sum(cls_ids == HUMAN_CLASS_INDEX)) if len(cls_ids) else 0
        hot_object_count = (
            int(np.sum(cls_ids == HOT_OBJECT_CLASS_INDEX)) if len(cls_ids) else 0
        )
        
        return {"human_count": human_count, "hot_object_count": hot_object_count}
    
    def release(self) -> None:
        """Release RKNN resources."""
        if hasattr(self, "rknn") and self.rknn is not None:
            self.rknn.release()
            logger.info("NPU detector released")
    
    @staticmethod
    def _thermal_to_bgr(thermal: np.ndarray) -> np.ndarray:
        """
        Convert thermal frame to BGR image using training normalization.
        
        Args:
            thermal: Thermal data (24x32 float).
        
        Returns:
            BGR image (240x320).
        """
        img_clipped = np.clip(thermal, TEMP_MIN_GLOBALE, TEMP_MAX_GLOBALE)
        img_8u = (
            (img_clipped - TEMP_MIN_GLOBALE)
            / (TEMP_MAX_GLOBALE - TEMP_MIN_GLOBALE)
            * 255.0
        ).astype(np.uint8)
        large_img = cv2.resize(
            img_8u,
            (THERMAL_WIDTH * SCALE_FACTOR, THERMAL_HEIGHT * SCALE_FACTOR),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv2.cvtColor(large_img, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """
        Ensure image is in BGR format.
        
        Args:
            image: Input image (can be various formats).
        
        Returns:
            Image in BGR format.
        
        Raises:
            ValueError: If image format is unsupported.
        """
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image format: shape={image.shape}")
    
    @staticmethod
    def _letterbox(
        img: np.ndarray, img_size: int = 320
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Apply letterbox transformation for model input.
        
        Args:
            img: Input image.
            img_size: Target size for model input.
        
        Returns:
            Tuple of (transformed image, scale ratio, padding offsets).
        """
        h, w = img.shape[:2]
        r = min(img_size / h, img_size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dx, dy = (img_size - nw) / 2, (img_size - nh) / 2
        
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
    
    def _postprocess(
        self,
        outputs,
        orig_hw: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process model outputs to extract detections.
        
        Args:
            outputs: Raw model outputs.
            orig_hw: Original image height, width.
            ratio: Scale ratio from letterbox.
            pad: Padding offsets from letterbox.
            conf_thr: Confidence threshold.
            iou_thr: IoU threshold for NMS.
        
        Returns:
            Tuple of (boxes, confidences, class_ids).
        """
        nc = len(CLASSES)
        raw = np.concatenate([np.squeeze(o) for o in outputs], axis=-1)
        if raw.shape[0] == 4 + nc:
            raw = raw.T
        
        cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        
        cls_scores = raw[:, 4 : 4 + nc]
        cls_ids = np.argmax(cls_scores, axis=1)
        conf = np.max(cls_scores, axis=1)
        
        mask = conf >= conf_thr
        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
        conf, cls_ids = conf[mask], cls_ids[mask]
        
        if len(conf) == 0:
            return (
                np.empty((0, 4)),
                np.array([]),
                np.array([], dtype=int),
            )
        
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
            k = self._nms(boxes[idx], conf[idx], iou_thr)
            keep.extend(idx[k])
        
        keep = sorted(keep)
        return boxes[keep], conf[keep], cls_ids[keep]
    
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list:
        """
        Non-maximum suppression to filter overlapping detections.
        
        Args:
            boxes: Detection boxes [[x1, y1, x2, y2], ...].
            scores: Detection confidence scores.
            iou_thr: IoU threshold for suppression.
        
        Returns:
            Indices of boxes to keep.
        """
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


class HumanDetectorCPU(HumanDetectorBase):
    """
    ONNX Runtime CPU-based detector for x86/x64 platforms.
    
    Uses CPU inference with ONNX Runtime for portability.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize CPU detector with ONNX model.
        
        Args:
            model_path: Path to ONNX model file.
            conf_threshold: Confidence threshold for detection.
            iou_threshold: IoU threshold for non-maximum suppression.
        
        Raises:
            RuntimeError: If ONNX Runtime initialization fails.
        """
        super().__init__(conf_threshold, iou_threshold)
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime not available. Install onnxruntime.")
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        logger.info("CPU (ONNX) detector initialized successfully")
    
    def infer_detections(self, image: np.ndarray) -> Dict[str, int]:
        """
        Run inference on image using ONNX Runtime.
        
        Args:
            image: Input image or thermal frame.
        
        Returns:
            Detection counts for humans and hot objects.
        """
        # Convert thermal frame to BGR if needed
        if (
            image.ndim == 2
            and image.shape == (THERMAL_HEIGHT, THERMAL_WIDTH)
            and image.dtype in (np.float32, np.float64)
        ):
            bgr = self._thermal_to_bgr(image)
        else:
            bgr = self._ensure_bgr(image)
        
        # Preprocess
        img320, ratio, pad = self._letterbox(bgr, img_size=320)
        inp = cv2.cvtColor(img320, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp = np.expand_dims(inp, axis=0)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: inp})
        _, _, cls_ids = self._postprocess(
            outputs,
            orig_hw=bgr.shape[:2],
            ratio=ratio,
            pad=pad,
            conf_thr=self.conf_threshold,
            iou_thr=self.iou_threshold,
        )
        
        # Count detections
        human_count = int(np.sum(cls_ids == HUMAN_CLASS_INDEX)) if len(cls_ids) else 0
        hot_object_count = (
            int(np.sum(cls_ids == HOT_OBJECT_CLASS_INDEX)) if len(cls_ids) else 0
        )
        
        return {"human_count": human_count, "hot_object_count": hot_object_count}
    
    def release(self) -> None:
        """Release ONNX Runtime resources."""
        if hasattr(self, "session") and self.session is not None:
            del self.session
            logger.info("CPU detector released")
    
    @staticmethod
    def _thermal_to_bgr(thermal: np.ndarray) -> np.ndarray:
        """Convert thermal frame to BGR image using training normalization."""
        img_clipped = np.clip(thermal, TEMP_MIN_GLOBALE, TEMP_MAX_GLOBALE)
        img_8u = (
            (img_clipped - TEMP_MIN_GLOBALE)
            / (TEMP_MAX_GLOBALE - TEMP_MIN_GLOBALE)
            * 255.0
        ).astype(np.uint8)
        large_img = cv2.resize(
            img_8u,
            (THERMAL_WIDTH * SCALE_FACTOR, THERMAL_HEIGHT * SCALE_FACTOR),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv2.cvtColor(large_img, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Ensure image is in BGR format."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image format: shape={image.shape}")
    
    @staticmethod
    def _letterbox(
        img: np.ndarray, img_size: int = 320
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Apply letterbox transformation for model input."""
        h, w = img.shape[:2]
        r = min(img_size / h, img_size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dx, dy = (img_size - nw) / 2, (img_size - nh) / 2
        
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
    
    def _postprocess(
        self,
        outputs,
        orig_hw: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post-process model outputs to extract detections."""
        nc = len(CLASSES)
        raw = np.concatenate([np.squeeze(o) for o in outputs], axis=-1)
        if raw.shape[0] == 4 + nc:
            raw = raw.T
        
        cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        
        cls_scores = raw[:, 4 : 4 + nc]
        cls_ids = np.argmax(cls_scores, axis=1)
        conf = np.max(cls_scores, axis=1)
        
        mask = conf >= conf_thr
        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
        conf, cls_ids = conf[mask], cls_ids[mask]
        
        if len(conf) == 0:
            return (
                np.empty((0, 4)),
                np.array([]),
                np.array([], dtype=int),
            )
        
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
            k = self._nms(boxes[idx], conf[idx], iou_thr)
            keep.extend(idx[k])
        
        keep = sorted(keep)
        return boxes[keep], conf[keep], cls_ids[keep]
    
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list:
        """Non-maximum suppression to filter overlapping detections."""
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


class InferenceService:
    """
    Factory service for creating and managing detector instances.
    
    Automatically selects appropriate backend (NPU or CPU ONNX).
    """
    
    _instance: Optional["InferenceService"] = None
    
    def __new__(
        cls,
        rknn_model_path: str,
        onnx_model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        rknn_model_path: str,
        onnx_model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize inference service with model paths.
        
        Args:
            rknn_model_path: Path to RKNN model.
            onnx_model_path: Path to ONNX model.
            conf_threshold: Confidence threshold for detection.
            iou_threshold: IoU threshold for NMS.
        """
        if self._initialized:
            return
        
        self.detector: Optional[HumanDetectorBase] = None
        self.platform = PlatformDetector().platform
        
        try:
            if self.platform == InferencePlatform.NPU_RK3588:
                self.detector = HumanDetectorNPU(
                    model_path=rknn_model_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
            else:
                self.detector = HumanDetectorCPU(
                    model_path=onnx_model_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize inference service: {e}")
            raise
    
    def infer(self, image: np.ndarray) -> Dict[str, int]:
        """
        Run inference on image.
        
        Args:
            image: Input image (BGR or thermal).
        
        Returns:
            Dictionary with 'human_count' and 'hot_object_count'.
        
        Raises:
            RuntimeError: If detector is not initialized.
        """
        if self.detector is None:
            raise RuntimeError("Detector not initialized")
        return self.detector.infer_detections(image)
    
    def release(self) -> None:
        """Release all resources."""
        if self.detector is not None:
            self.detector.release()
            self.detector = None

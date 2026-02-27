import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import websockets


CLASSES = ["Humain", "Objet_Chaud"]
IMG_SIZE = 320
HUMAN_CLASS_INDEX = 0
THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24

# Constantes identiques au script convert_npy_to_png.py utilisé pour l'entraînement
TEMP_MIN_GLOBALE = 5.0
TEMP_MAX_GLOBALE = 55.0
SCALE_FACTOR = 10  # 32x24 -> 320x240


def letterbox(img: np.ndarray, size: int = IMG_SIZE) -> Tuple[np.ndarray, float, Tuple[float, float]]:
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
    """Convertit une frame thermique float (24x32) en image BGR (240x320)
    exactement comme convert_npy_to_png.py le fait pour l'entraînement."""
    img_clipped = np.clip(thermal, TEMP_MIN_GLOBALE, TEMP_MAX_GLOBALE)
    img_8u = ((img_clipped - TEMP_MIN_GLOBALE) / (TEMP_MAX_GLOBALE - TEMP_MIN_GLOBALE) * 255.0).astype(np.uint8)
    large_img = cv2.resize(
        img_8u,
        (THERMAL_WIDTH * SCALE_FACTOR, THERMAL_HEIGHT * SCALE_FACTOR),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(large_img, cv2.COLOR_GRAY2BGR)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Format image non supporté: shape={image.shape}")


def decode_npy_payload(payload) -> np.ndarray:
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


class HumanDetectorSingleton:
    _instance: Optional["HumanDetectorSingleton"] = None

    def __new__(cls, model_path: str, conf: float, iou: float):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str, conf: float, iou: float):
        if self._initialized:
            return

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

    def infer_human_count(self, image: np.ndarray) -> int:
        # Si c'est une frame thermique brute (24x32 float), la convertir
        # en image BGR exactement comme le script d'entraînement
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
        if len(cls_ids) == 0:
            return 0
        return int(np.sum(cls_ids == HUMAN_CLASS_INDEX))

    def release(self):
        if getattr(self, "rknn", None) is not None:
            self.rknn.release()


async def run_client(uri: str, detector: HumanDetectorSingleton):
    while True:
        try:
            print(f"Connexion websocket vers {uri}")
            async with websockets.connect(uri, max_size=None) as ws:
                print("Connecté. En attente des frames NPY...")
                async for message in ws:
                    try:
                        frame = decode_npy_payload(message)
                        count = detector.infer_human_count(frame)
                        has_humans = count > 0
                        response = {
                            "has_humans": has_humans,
                            "human_count": count,
                        }
                        await ws.send(json.dumps(response))
                        print(f"Résultat envoyé: {response}")
                    except Exception as infer_error:
                        error_response = {
                            "has_humans": False,
                            "human_count": 0,
                            "error": str(infer_error),
                        }
                        await ws.send(json.dumps(error_response))
                        print(f"Erreur frame: {infer_error}")
        except Exception as conn_error:
            print(f"Connexion perdue: {conn_error}. Nouvelle tentative dans 2s...")
            await asyncio.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Client websocket RKNN (ESP8266)")
    parser.add_argument("--ws-url", type=str, default="ws://10.28.26.7:81/")
    parser.add_argument("--model", type=str, default=str(Path("rknn") / "Version6.rknn"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    detector = HumanDetectorSingleton(model_path=args.model, conf=args.conf, iou=args.iou)
    try:
        asyncio.run(run_client(args.ws_url, detector))
    finally:
        detector.release()


if __name__ == "__main__":
    main()

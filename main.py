"""
Détection thermique YOLOv8 sur Rock 5B (RK3588 NPU)
Usage: python3 main.py --image test.jpg
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# Configuration
MODEL_PATH = Path(__file__).parent / "rknn" / "Version6.rknn"
CLASSES = ["Humain", "Objet_Chaud"]
COLORS = [(0, 255, 0), (0, 165, 255)]
IMG_SIZE = 320  # Le modèle attend 320x320


def letterbox(img, size=IMG_SIZE):
    """Redimensionne 320x240 → 320x320 en conservant le ratio, padding gris."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dx, dy = (size - nw) / 2, (size - nh) / 2

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dy - 0.1)), int(round(dy + 0.1))
    left, right = int(round(dx - 0.1)), int(round(dx + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dx, dy)


def nms(boxes, scores, iou_thr=0.45):
    """NMS sur un ensemble de boîtes."""
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


def postprocess(outputs, orig_hw, ratio, pad, conf_thr=0.25, iou_thr=0.45):
    """Décode les sorties RKNN → boîtes, scores, classes."""
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

    # Undo letterbox → coordonnées image originale (320x240)
    x1 = (x1 - pad[0]) / ratio
    y1 = (y1 - pad[1]) / ratio
    x2 = (x2 - pad[0]) / ratio
    y2 = (y2 - pad[1]) / ratio

    oh, ow = orig_hw
    x1, x2 = np.clip(x1, 0, ow), np.clip(x2, 0, ow)
    y1, y2 = np.clip(y1, 0, oh), np.clip(y2, 0, oh)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS par classe
    keep = []
    for c in range(nc):
        idx = np.where(cls_ids == c)[0]
        if len(idx) == 0:
            continue
        k = nms(boxes[idx], conf[idx], iou_thr)
        keep.extend(idx[k])

    keep = sorted(keep)
    return boxes[keep], conf[keep], cls_ids[keep]


def draw(image, boxes, scores, cls_ids):
    """Dessine les détections et retourne l'image + compteurs."""
    counts = {name: 0 for name in CLASSES}
    out = image.copy()

    for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cls_ids):
        cid = int(cid)
        name = CLASSES[cid]
        color = COLORS[cid]
        counts[name] += 1

        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, pt1, pt2, color, 2)

        label = f"{name} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (pt1[0], pt1[1] - th - 6), (pt1[0] + tw, pt1[1]), color, -1)
        cv2.putText(out, label, (pt1[0], pt1[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out, counts


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 thermique – Rock 5B NPU")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.75)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", type=str, default="results/result.jpg")
    args = parser.parse_args()

    # Charger image (320x240)
    img0 = cv2.imread(args.image)
    assert img0 is not None, f"Image introuvable: {args.image}"
    print(f"Image chargée: {img0.shape[1]}x{img0.shape[0]}")

    # Prétraitement : 320x240 → 320x320 (padding haut/bas de 40px)
    img, ratio, pad = letterbox(img0)
    
    # --- CORRECTIONS ICI ---
    # 1. Utiliser cv2.COLOR_BGR2GRAY pour convertir l'image couleur en gris
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Ajouter les dimensions Batch (1) et Canal (1)
    # Format NHWC : (Batch, Hauteur, Largeur, Canaux) -> (1, 320, 320, 1)
    inp = np.expand_dims(inp, axis=0)
    # -----------------------

    # Initialiser NPU (3 cœurs = pleine puissance 6 TOPS)
    rknn = RKNNLite(verbose=False)
    assert rknn.load_rknn(args.model) == 0, "Échec load_rknn"
    assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, "Échec init NPU"

    # Inférence
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[inp])
    dt = (time.perf_counter() - t0) * 1000

    # Post-traitement
    boxes, scores, cls_ids = postprocess(
        outputs, img0.shape[:2], ratio, pad, args.conf, args.iou
    )

    # Résultat
    result, counts = draw(img0, boxes, scores, cls_ids)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)
    rknn.release()

    # Affichage
    print(f"\n{'='*40}")
    print(f" Inférence NPU : {dt:.1f} ms")
    print(f"{'='*40}")
    total = 0
    for name, n in counts.items():
        print(f" {name:15s}: {n}")
        total += n
    print(f" {'Total':15s}: {total}")
    print(f"{'='*40}")
    print(f" Résultat → {out_path}\n")


if __name__ == "__main__":
    main()
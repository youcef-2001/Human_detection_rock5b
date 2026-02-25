import argparse
from pathlib import Path
import cv2
import numpy as np
from rknn.api import RKNN


CLASS_NAMES = ["Humain", "Objet_Chaud"]


def letterbox(image, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def xywh2xyxy(boxes):
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms_classwise(boxes, scores, classes, iou_thres=0.45):
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        b = boxes[idx]
        s = scores[idx]
        order = s.argsort()[::-1]
        while len(order) > 0:
            i = order[0]
            keep.append(idx[i])
            if len(order) == 1:
                break
            ious = iou(b[i], b[order[1:]])
            order = order[1:][ious <= iou_thres]
    return keep


def decode_outputs(outputs, nc=2):
    preds = []
    for o in outputs:
        a = np.squeeze(np.array(o))
        if a.ndim == 2:
            if a.shape[1] in (4 + nc, 5 + nc):
                preds.append(a)
            elif a.shape[0] in (4 + nc, 5 + nc):
                preds.append(a.T)
    if not preds:
        raise RuntimeError("Format de sortie non supporté.")
    return np.concatenate(preds, axis=0)


def postprocess(outputs, orig_shape, ratio, pad, conf_thres=0.25, iou_thres=0.45, nc=2):
    pred = decode_outputs(outputs, nc=nc)
    if pred.shape[1] == 4 + nc:
        boxes = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        conf = np.max(cls_scores, axis=1)
    else:
        boxes = pred[:, :4]
        obj = pred[:, 4]
        cls_scores = pred[:, 5:5 + nc]
        cls_ids = np.argmax(cls_scores, axis=1)
        conf = obj * np.max(cls_scores, axis=1)

    m = conf >= conf_thres
    boxes, conf, cls_ids = boxes[m], conf[m], cls_ids[m]
    if len(boxes) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])

    boxes = xywh2xyxy(boxes)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

    keep = nms_classwise(boxes, conf, cls_ids, iou_thres=iou_thres)
    return boxes[keep], conf[keep], cls_ids[keep]


def draw_and_count(image, boxes, scores, cls_ids):
    counts = {name: 0 for name in CLASS_NAMES}
    out = image.copy()
    for b, s, c in zip(boxes, scores, cls_ids):
        c = int(c) if int(c) < len(CLASS_NAMES) else 0
        name = CLASS_NAMES[c]
        counts[name] += 1
        x1, y1, x2, y2 = map(int, b)
        color = (0, 255, 0) if c == 0 else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{name} {s:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rknn/best.rknn")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", type=str, default="results/result_pc.jpg")
    args = parser.parse_args()

    img0 = cv2.imread(args.image)
    if img0 is None:
        raise FileNotFoundError(args.image)

    img, ratio, pad = letterbox(img0, (args.imgsz, args.imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError("load_rknn a échoué")

    # Simulation PC RK3588
    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        raise RuntimeError("init_runtime(target='rk3588') a échoué")

    outputs = rknn.inference(inputs=[img])
    boxes, scores, cls_ids = postprocess(
        outputs, img0.shape, ratio, pad, conf_thres=args.conf, iou_thres=args.iou, nc=len(CLASS_NAMES)
    )

    result_img, counts = draw_and_count(img0, boxes, scores, cls_ids)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result_img)

    rknn.release()

    print("=== Résultats simulation PC ===")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"Image résultat: {out_path}")


if __name__ == "__main__":
    main()
import os
import numpy as np
import cv2

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    # Fallback to full TensorFlow if available
    from tensorflow.lite import Interpreter  # type: ignore


class _Boxes:
    def __init__(self, xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy  # (N,4) int
        self.cls = cls.astype(np.float32)  # (N,) float32 indices
        self.conf = conf.astype(np.float32)  # (N,) float32 confidences


class _Result:
    def __init__(self, boxes: _Boxes):
        self.boxes = boxes


class TFLiteYOLO:
    """
    Minimal TFLite wrapper that mimics the parts of Ultralytics' YOLO API
    used in this project. It returns a list with a single result object
    that has `.boxes.xyxy`, `.boxes.cls`, and `.boxes.conf`.
    """

    def __init__(self, model_path: str, names: dict | None = None, num_threads: int = 2):
        self.model_path = model_path
        self.names = names or {}
        self.interp = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        # Input shape: [1, H, W, 3]
        _, self.in_h, self.in_w, _ = self.inp["shape"]
        self._debug = os.getenv("DEBUG_DET", "0") == "1" or os.getenv("TFL_DEBUG", "0") == "1"

    def __call__(self, image_bgr: np.ndarray, imgsz: int = 320, agnostic_nms: bool = False, verbose: bool = False, device: str = "cpu"):
        h0, w0 = image_bgr.shape[:2]
        # Resize with letterbox to model input size while keeping aspect ratio simple: just resize
        # Convert BGR -> RGB as most TFLite models expect RGB
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if (self.in_w, self.in_h) != (w0, h0):
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32)
        # Assume 0-1 normalization
        if x.max() > 1.0:
            x = x / 255.0
        x = np.expand_dims(x, 0)
        # If model expects uint8, convert
        if self.inp["dtype"] == np.uint8:
            x = (x * 255).astype(np.uint8)

        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()

        # Try to parse outputs from common Ultralytics TFLite export formats
        tensors = [self.interp.get_tensor(o["index"]) for o in self.outs]
        if self._debug:
            try:
                print("[TFLiteYOLO] outputs:", [t.shape for t in tensors])
            except Exception:
                pass

        boxes, scores, classes = self._parse_outputs(tensors, (w0, h0))
        boxes_obj = _Boxes(boxes, classes, scores)
        return [_Result(boxes_obj)]

    def _parse_outputs(self, tensors: list[np.ndarray], orig_size: tuple[int, int]):
        w0, h0 = orig_size
        # Case A: TF Object Detection style outputs
        # [boxes: (1,N,4) y1,x1,y2,x2], [scores: (1,N)], [classes: (1,N)], [num:(1,)]
        if len(tensors) >= 3:
            shapes = [t.shape for t in tensors]
            # Identify likely boxes/scores/classes by shapes
            box_idx = next((i for i, t in enumerate(tensors) if t.ndim == 3 and t.shape[-1] == 4), None)
            score_idx = next((i for i, t in enumerate(tensors) if t.ndim == 2 and t.shape[-1] > 1), None)
            cls_idx = None
            for i, t in enumerate(tensors):
                if i == box_idx or i == score_idx:
                    continue
                if t.ndim in (1, 2):
                    cls_idx = i
                    break
            if box_idx is not None and score_idx is not None and cls_idx is not None:
                b = tensors[box_idx][0]
                s = tensors[score_idx][0]
                c = tensors[cls_idx][0] if tensors[cls_idx].ndim > 1 else tensors[cls_idx]
                b = b.astype(np.float32)
                s = s.astype(np.float32)
                c = c.astype(np.float32)
                # Convert y1,x1,y2,x2 (normalized 0..1) to xyxy pixel
                y1, x1, y2, x2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
                x1 = (x1 * w0).clip(0, w0 - 1)
                y1 = (y1 * h0).clip(0, h0 - 1)
                x2 = (x2 * w0).clip(0, w0 - 1)
                y2 = (y2 * h0).clip(0, h0 - 1)
                xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
                return xyxy, s, c

        # Case B: Ultralytics NMS export: (N,6) or (1,N,6) [x,y,w,h,conf,cls] or [x1,y1,x2,y2,conf,cls]
        arr = tensors[0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[1] >= 6:
            a = arr.astype(np.float32)
            if a.shape[1] > 6:
                a = a[:, :6]

            # Determine which of the last two columns is class vs confidence
            conf_col, cls_col = 4, 5
            if self.names:
                ncls = max(1, len(self.names))
                # choose the column that looks like integer class ids
                cand5 = a[:, 5]
                cand4 = a[:, 4]
                def score_as_class(col):
                    r = np.rint(col)
                    ok = ((r >= 0) & (r < ncls) & (np.abs(r - col) < 1e-3)).mean()
                    return ok
                s5 = score_as_class(cand5)
                s4 = score_as_class(cand4)
                if s4 > s5 + 0.2:
                    conf_col, cls_col = 5, 4

            conf = a[:, conf_col]
            cls = a[:, cls_col]

            # Determine box format by validating both interpretations and choosing the better one
            b = a[:, :4]
            # normalized if values mostly <= 1.2
            normalized = (b <= 1.2).mean() > 0.9
            def to_xyxy_xyxyfmt(bb):
                x1, y1, x2, y2 = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
                return x1, y1, x2, y2
            def to_xyxy_xywhfmt(bb):
                cx, cy, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
                x1, y1 = cx - w / 2.0, cy - h / 2.0
                x2, y2 = cx + w / 2.0, cy + h / 2.0
                return x1, y1, x2, y2
            # Try both
            x1a, y1a, x2a, y2a = to_xyxy_xyxyfmt(b)
            x1b, y1b, x2b, y2b = to_xyxy_xywhfmt(b)

            def valid_frac(x1, y1, x2, y2, norm):
                if norm:
                    wv = (x2 - x1) > 0
                    hv = (y2 - y1) > 0
                    within = (x1 >= 0) & (y1 >= 0) & (x2 <= 1.0) & (y2 <= 1.0)
                else:
                    wv = (x2 - x1) > 0
                    hv = (y2 - y1) > 0
                    within = (x1 >= 0) & (y1 >= 0) & (x2 <= self.in_w) & (y2 <= self.in_h)
                return (wv & hv & within).mean()

            va = valid_frac(x1a, y1a, x2a, y2a, normalized)
            vb = valid_frac(x1b, y1b, x2b, y2b, normalized)
            if vb > va:
                x1, y1, x2, y2 = x1b, y1b, x2b, y2b
            else:
                x1, y1, x2, y2 = x1a, y1a, x2a, y2a

            # Scale to original image size
            if normalized:
                x1 = (x1 * w0).clip(0, w0 - 1)
                x2 = (x2 * w0).clip(0, w0 - 1)
                y1 = (y1 * h0).clip(0, h0 - 1)
                y2 = (y2 * h0).clip(0, h0 - 1)
            else:
                # pixels relative to model input
                sx, sy = float(w0) / float(self.in_w), float(h0) / float(self.in_h)
                x1 = (x1 * sx).clip(0, w0 - 1)
                x2 = (x2 * sx).clip(0, w0 - 1)
                y1 = (y1 * sy).clip(0, h0 - 1)
                y2 = (y2 * sy).clip(0, h0 - 1)

            xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
            return xyxy, conf, cls

        # Case C: Some NMS models return two arrays each (1,N,6); try merging
        if len(tensors) >= 2:
            a = tensors[0]
            b = tensors[1]
            for arr in (a, b):
                if arr.ndim == 3 and arr.shape[-1] >= 6:
                    tmp = arr[0].astype(np.float32)
                    if tmp.shape[1] > 6:
                        tmp = tmp[:, :6]
                    x1, y1, x2, y2, conf, cls = tmp[:, 0], tmp[:, 1], tmp[:, 2], tmp[:, 3], tmp[:, 4], tmp[:, 5]
                    if (x2 <= 1).all() and (y2 <= 1).all():
                        # normalized
                        x1 = (x1 * w0).clip(0, w0 - 1)
                        y1 = (y1 * h0).clip(0, h0 - 1)
                        x2 = (x2 * w0).clip(0, w0 - 1)
                        y2 = (y2 * h0).clip(0, h0 - 1)
                    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
                    return xyxy, conf, cls

        # Fallback: no detections
        return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

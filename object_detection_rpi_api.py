


try:
    from ultralytics import YOLO  # type: ignore
    _USING_ULTRALYTICS = True
except Exception:
    from tflite_yolo import TFLiteYOLO as YOLO  # type: ignore
    _USING_ULTRALYTICS = False
import cv2
import numpy as np
import pyautogui
from config_utils_fruit import ROI_PATH as roi_path , MODEL_PATH as model_path, product_data, classNames
import os
# Use direct DB writer by default; API writer is optional
from save_to_db import save_detected_product, clear_database  # type: ignore
import math
from collections import Counter
import json
from enum import Enum, auto
from PIL import Image, ImageDraw, ImageFont
import time
import os


HEADLESS = os.environ.get("HEADLESS", "0") == "1"
try:
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

# Load the detection model (Ultralytics or TFLite wrapper)
if _USING_ULTRALYTICS:
    model = YOLO(model_path)
else:
    # Provide class name mapping to the TFLite wrapper and allow tuning threads/delegate via env
    names_map = {i: n for i, n in enumerate(classNames)}
    try:
        tfl_threads = int(os.getenv("TFL_THREADS", "4"))
    except Exception:
        tfl_threads = 2
    tfl_delegate = os.getenv("TFL_DELEGATE", "")
    model = YOLO(model_path, names=names_map, num_threads=tfl_threads, delegate=tfl_delegate)



try:
    _ = model(np.zeros((320,320,3), dtype = np.uint8), imgsz=320, verbose=False, device ='cpu')
except Exception:
    pass
names = getattr(model, 'names', {}) if hasattr(model, 'names') else {}

# Screen size with headless-safe fallback
try:
    screen_width, screen_height = pyautogui.size()
except Exception:
    screen_width, screen_height = (800, 480)

video_width = 500
full_frame = np.ones((screen_height, video_width, 3), dtype=np.uint8) * 255


def draw_text_with_pillow(image, text, position, font_path="arial.ttf", font_size=32, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x, y = position
    text_bg_rect = [x, y, x + text_width + 10, y + text_height + 10]
    draw.rectangle(text_bg_rect, fill=bg_color)
    draw.text((x + 5, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_text_centered_with_pillow(image, text, center_position, font_path="arial.ttf", font_size=24, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    cx, cy = center_position
    x = int(cx - (text_width / 2) - 5)
    y = int(cy - (text_height / 2) - 5)
    x = max(0, min(image.shape[1] - (text_width + 10), x))
    y = max(0, min(image.shape[0] - (text_height + 10), y))
    text_bg_rect = [x, y, x + text_width + 10, y + text_height + 10]
    draw.rectangle(text_bg_rect, fill=bg_color)
    draw.text((x + 5, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def camera_error_overlay(width: int = 500, height: int = 500):
    overlay = np.zeros((height, width, 3), dtype="uint8")
    overlay[:] = (0, 0, 0)
    text = "Camera not working"
    cv2.putText(overlay, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Error", overlay)


def select_or_load_roi(cap, path):
    # If previously saved, use it
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                x, y, w, h = map(int, f.read().strip().split(','))
                return (x, y, w, h)
        except Exception:
            pass

    # Try to read a frame for ROI
    ok, img = cap.read()
    if not ok or img is None:
        return None

    # Try GUI ROI selector; if it fails (headless), use full image
    try:
        
        cv2.namedWindow("select the area")
        cv2.moveWindow("select the area", 0, 0)
        x, y, w, h = cv2.selectROI("select the area", img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("select the area")
        if int(w) <= 0 or int(h) <= 0:
            raise RuntimeError("Empty ROI")
    except Exception:
        y, x, h, w = 0, 0, img.shape[0], img.shape[1]

    with open(path, 'w') as f:
        f.write(','.join(map(str, (int(x), int(y), int(w), int(h)))))
    return (int(x), int(y), int(w), int(h))


def roi_within_bounds(roi, frame_width, frame_height):
    x, y, w, h = roi
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(1, min(w, frame_width - x))
    h = max(1, min(h, frame_height - y))
    return (x, y, w, h)


def draw_overlay(img, text, position = "center",box_w_ratio = 0.75, box_h = 60,color=(255, 255, 255), bg=(0, 0, 0), alpha=0.35):
    h, w = img.shape[:2]
    box_w = int(w * box_w_ratio)
    x1 = int((w - box_w)/2)
    x2 = x1 + box_w

    if position == "center":
        y1 = int(h//2 - box_h/2)
        y2 = y1 + box_h
    else:
        y1 = 0
        y2 = int(box_h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, int(y1)), (x2, int(y2)), bg, -1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    (text_size, baseline) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_w, text_h = text_size
    text_x = int(x1 + (box_w - text_w) / 2)
    text_y = int(y1 + (box_h + text_h) / 2)
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_labels_centered(img, boxes_labels, src_size):
    if not boxes_labels:
        return
    h, w = img.shape[:2]
    if not src_size or src_size[0] == 0 or src_size[1] == 0:
        scale_x = 1.0
        scale_y = 1.0
    else:
        scale_x = float(w) / float(src_size[0])
        scale_y = float(h) / float(src_size[1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)

    for (x1, y1, x2, y2, label) in boxes_labels:
        cx = ((x1 + x2) / 2.0) * scale_x
        cy = ((y1 + y2) / 2.0) * scale_y
        (text_size, baseline) = cv2.getTextSize(str(label), font, font_scale, thickness)
        text_w, text_h = text_size
        x = int(cx - text_w / 2)
        y = int(cy + text_h / 2)
        x = max(0, min(x, w - text_w))
        y = max(text_h + 2, min(y, h - 2))
        cv2.putText(img, str(label), (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_boxes(img, boxes_labels, src_size, color=(0, 255, 0), thickness=2):
    if not boxes_labels:
        return
    h, w = img.shape[:2]
    if not src_size or src_size[0] == 0 or src_size[1] == 0:
        scale_x = 1.0
        scale_y = 1.0
    else:
        scale_x = float(w) / float(src_size[0])
        scale_y = float(h) / float(src_size[1])
    for (x1, y1, x2, y2, _label) in boxes_labels:
        p1 = (int(x1 * scale_x), int(y1 * scale_y))
        p2 = (int(x2 * scale_x), int(y2 * scale_y))
        cv2.rectangle(img, p1, p2, color, thickness)


# Motion detection
prev_frame = None
motion_area_threshold = 2000
motion_frames_required = 2
stable_frames_required = 1
motion_count = 0
stable_count = 0


class MotionState(Enum):
    IDLE = auto()
    PLACING = auto()
    STABLE = auto()


state = MotionState.IDLE
has_cleared_db = False
stable_payload_sent = False
frozen_boxes_labels = []
frozen_src_size = None
placing_just_entered = False


def _open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    # Reduce buffering and latency and enforce small frame
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def _dedupe_detections(dets, iou_same: float = 0.5, iou_diff: float = 0.7):
    """
    Suppress overlapping boxes so a single object doesn't get two labels.
    - Same-class overlaps (IoU >= iou_same): keep highest-confidence only.
    - Different-class overlaps (IoU >= iou_diff): keep highest-confidence only.
    dets: list[(x1, y1, x2, y2, label, conf)]
    """
    if not dets:
        return dets
    dets_sorted = sorted(dets, key=lambda d: float(d[5]), reverse=True)
    kept = []
    for d in dets_sorted:
        suppress = False
        for k in kept:
            iou = _iou((d[0], d[1], d[2], d[3]), (k[0], k[1], k[2], k[3]))
            if d[4] == k[4]:
                if iou >= iou_same:
                    suppress = True
                    break
            else:
                if iou >= iou_diff:
                    suppress = True
                    break
        if not suppress:
            kept.append(d)
    return kept

def _build_payload_from_counts(counts_dict):
    payload = []
    for product, count in counts_dict.items():
        meta = product_data.get(product)
        barcode = meta.get('barcode') if meta else 'N/A'
        payload.append({
            'Name': product,
            'Count': int(count),
            'barcode': barcode
        })
    return payload


def _canonicalize_product_name(raw_name: str) -> str:
    """
    Map detector-provided labels to the canonical keys used in product_data.
    Falls back to the original label (or 'Unknown') if no match is found.
    """
    if not raw_name:
        return "Unknown"
    candidate = raw_name.strip()
    candidate_lower = candidate.lower()
    for known_name in product_data.keys():
        if known_name.lower() == candidate_lower:
            return known_name
    return candidate or "Unknown"


def one_shot_detect(crop):
    try:
        # Fixed small input for speed on Pi
        preds = model(crop, imgsz=320, agnostic_nms=False, verbose=False, device='cpu')[0]
    except Exception:
        return [], Counter(), []

    boxes_for_iou = []
    counts = Counter()
    boxes_labels = []
    model_names = model.names if hasattr(model, "names") else {}

    try:
        dets = []
        if hasattr(preds, 'boxes') and preds.boxes is not None:
            # Handle both Ultralytics tensors and our numpy wrapper
            xyxy = preds.boxes.xyxy
            cls = preds.boxes.cls
            conf_arr = preds.boxes.conf
            # Convert to numpy arrays
            xyxy = xyxy if isinstance(xyxy, np.ndarray) else xyxy.cpu().numpy()
            cls = cls if isinstance(cls, np.ndarray) else cls.cpu().numpy()
            conf_arr = conf_arr if isinstance(conf_arr, np.ndarray) else conf_arr.cpu().numpy()
            xyxy = xyxy.astype(int)
            cls = cls.astype(int)
            unknown_conf = 0.85
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf_arr):
                if cf < 0.25:
                    continue
                class_id = int(c)
                raw_name = model_names.get(class_id, f"Class {class_id}")
                canonical_name = _canonicalize_product_name(raw_name)
                if cf < unknown_conf:
                    canonical_name = "Unknown"
                dets.append((int(x1), int(y1), int(x2), int(y2), canonical_name, float(cf)))

        # Deduplicate overlapping detections
        dets = _dedupe_detections(dets, iou_same=0.5, iou_diff=0.7)
        for x1, y1, x2, y2, canonical_name, cf in dets:
            boxes_for_iou.append((x1, y1, x2, y2))
            display_label = f'{canonical_name}, {cf:.2f}'
            boxes_labels.append((x1, y1, x2, y2, display_label))
            counts[canonical_name] += 1
    except Exception:
        pass

    return boxes_for_iou, counts, boxes_labels

    


def main():
    cap = _open_camera()
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        camera_error_overlay()
        cv2.waitKey(2000)

    roi_cordinates = select_or_load_roi(cap, roi_path)
    if roi_cordinates is None:
        camera_error_overlay()
        cv2.waitKey(2000)
        return

    rx, ry, rw, rh = roi_cordinates
    roi_aspect_ratio = max(1e-6, rw / float(rh))
    video_height = int(video_width / roi_aspect_ratio)
    video_height = min(video_height, screen_height)
    render_size = (video_width, video_height)

    if not HEADLESS:
        cv2.namedWindow("VBR Realtime scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VBR Realtime scanner", video_width, screen_height)

    global prev_frame, motion_count, stable_count
    state = MotionState.IDLE
    has_cleared_db = False
    stable_payload_sent = False
    pending_payload = None
    placing_counts_max = {}
    frozen_boxes_labels = []
    frozen_src_size = None
    placing_just_entered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            camera_error_overlay(*render_size)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        H, W = frame.shape[:2]
        rx, ry, rw, rh = roi_within_bounds(roi_cordinates, W, H)
        cropped_frame = frame[ry:ry+rh, rx:rx+rw]

        # Motion check on ROI
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        motion_detected = False
        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(frame_delta, 18, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_motion_area = sum(cv2.contourArea(c) for c in contours)
            motion_detected = total_motion_area > motion_area_threshold
        prev_frame = gray

        # Update motion state
        if motion_detected:
            motion_count += 1
            stable_count = 0
        else:
            stable_count += 1
            motion_count = 0

        # State transitions
        if state == MotionState.IDLE:
            if motion_count >= motion_frames_required:
                state = MotionState.PLACING
                # Clear DB immediately when placing starts
                try:
                    if not has_cleared_db:
                        clear_database()
                        has_cleared_db = True
                except Exception as e:
                    print(f"Error clearing DB on entering PLACING: {e}")
                stable_payload_sent = False
                pending_payload = None
                placing_counts_max = {}
                frozen_boxes_labels = []
                placing_just_entered = True
            else:
                boxes_for_iou, counts, boxes_labels = one_shot_detect(cropped_frame)
                has_items = bool(boxes_labels)
                if has_items:
                    frozen_boxes_labels = boxes_labels
                    frozen_src_size = (cropped_frame.shape[1], cropped_frame.shape[0])
                    state = MotionState.STABLE
                    placing_counts_max = {k: int(v) for k, v in counts.items()}
                    final_payload = _build_payload_from_counts(placing_counts_max)
                    try:
                        if not stable_payload_sent:
                            save_detected_product(json.dumps(final_payload))
                            stable_payload_sent = True
                            has_cleared_db = False
                    except Exception as e:
                        print(f"Error saving idle-detected payload: {e}")
                else:
                    if not has_cleared_db:
                        clear_database()
                        has_cleared_db = True

        elif state == MotionState.PLACING:
            boxes_for_iou, counts, boxes_labels = one_shot_detect(cropped_frame)
            has_items = bool(boxes_labels)
            if has_items:
                frozen_boxes_labels = boxes_labels
                frozen_src_size = (cropped_frame.shape[1], cropped_frame.shape[0])
                for product, cnt in counts.items():
                    prev = placing_counts_max.get(product, 0)
                    if int(cnt) > int(prev):
                        placing_counts_max[product] = int(cnt)

            if stable_count >= stable_frames_required:
                if has_items:
                    state = MotionState.STABLE
                    try:
                        if not stable_payload_sent:
                            final_payload = _build_payload_from_counts({k: int(v) for k, v in counts.items()})
                            save_detected_product(json.dumps(final_payload))
                            stable_payload_sent = True
                            has_cleared_db = False
                    except Exception as e:
                        print(f"Error saving payload on STABLE transition: {e}")
                else:
                    state = MotionState.IDLE
                    frozen_boxes_labels = []
                    stable_payload_sent = False
                    has_cleared_db = False
                    pending_payload = None
                    placing_counts_max = {}
                    try:
                        clear_database()
                        has_cleared_db = True
                    except Exception as e:
                        print(f"Error clearing database in PLACING->IDLE: {e}")
        elif state == MotionState.STABLE:
            if motion_count >= motion_frames_required:
                state = MotionState.PLACING
                stable_payload_sent = False
                frozen_boxes_labels = []
                pending_payload = None
                placing_counts_max = {}
                # Clear DB as soon as new motion starts
                try:
                    if not has_cleared_db:
                        clear_database()
                        has_cleared_db = True
                except Exception as e:
                    print(f"Error clearing DB on STABLE->PLACING: {e}")

        # Rendering
        
        if not HEADLESS:
            view = cropped_frame.copy()
            resized = cv2.resize(view, render_size)
            if state == MotionState.IDLE:
                draw_overlay(resized, "Waiting for item", position="center")
            elif state == MotionState.PLACING:
                if placing_just_entered:
                    draw_overlay(resized, "Welcome to VBR realtime scanner", position="center", color=(255, 0, 0))
                    placing_just_entered = False
                else:
                    draw_overlay(resized, "Placing items", position="center")
            elif state == MotionState.STABLE:
                # Draw boxes and labels for clarity
                draw_boxes(resized, frozen_boxes_labels, frozen_src_size)
                draw_labels_centered(resized, frozen_boxes_labels, frozen_src_size)
                draw_overlay(resized, "Proceed to payment now...", position="top", color=(0, 255, 0))

            full_frame[:render_size[1], :] = resized
            
            cv2.imshow("VBR Realtime scanner", full_frame)

            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255
        if key == ord('q'):
            break
        elif key == ord('r'):
            if os.path.exists(roi_path):
                try:
                    os.remove(roi_path)
                except Exception as e:
                    print(f"Could not remove ROI file: {e}")
            new_roi = select_or_load_roi(cap, roi_path)
            if new_roi:
                rx, ry, rw, rh = new_roi
                roi_aspect = max(1e-6, rw / float(rh))
                video_height = int(video_width / roi_aspect)
                video_height = min(video_height, screen_height)
                render_size = (video_width, video_height)
                cv2.resizeWindow("VBR Realtime scanner", video_width, screen_height)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# core/inference.py
"""Hybrid AI & Vision Inference Engine:
1. Deep Learning: YOLOv8 Object Detection (Detects trucks, machinery, haul roads)
2. Classical CV: Plume Opacity & Density Estimator (HSV Dust Masking + Laplacian Contrast Loss)
3. Visual HUD Renderer: Bounding boxes, dust plume overlay, and real-time status banners.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Any, Tuple

# Initialize YOLO model safely
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(_MODEL_DIR, exist_ok=True)
_MODEL_PATH = os.path.join(_MODEL_DIR, "yolov8n.pt")

try:
    if os.path.exists(_MODEL_PATH):
        _yolo_model = YOLO(_MODEL_PATH)
    else:
        _yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    _yolo_model = None

# Classical CV Opacity Configuration
_DUST_HSV_LO = np.array([10, 25, 120])   # Airborne dust / tan / beige lower bound
_DUST_HSV_HI = np.array([35, 160, 255])  # Airborne dust upper bound


def detect_objects(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Runs YOLOv8 nano detector to find trucks, construction vehicles and equipment."""
    if _yolo_model is None:
        return []

    try:
        # Run inference on frame (fast CPU inference)
        results = _yolo_model(frame, verbose=False, conf=0.25)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = _yolo_model.names.get(cls_id, f"obj_{cls_id}")
            # Target relevant site assets (truck, car, bus, person, etc.)
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "label": label,
                "conf": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })
        return detections
    except Exception:
        return []


def opacity_score(frame: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes airborne dust opacity (0-100%) using HSV Masking and Contrast Loss.
    Returns: (opacity_percent, dust_mask)
    """
    h, w = frame.shape[:2]
    # 1. Color Isolation in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dust_mask = cv2.inRange(hsv, _DUST_HSV_LO, _DUST_HSV_HI)
    pixel_ratio = (np.count_nonzero(dust_mask) / (h * w)) * 100.0

    # 2. Contrast Variance Loss (Laplacian blur metric)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    baseline_variance = 1200.0  # Clear air reference benchmark
    contrast_drop = max(0.0, 1.0 - (lap_var / baseline_variance)) * 100.0

    # 3. Hybrid fusion score
    raw_opacity = (0.55 * pixel_ratio) + (0.45 * contrast_drop)
    final_opacity = float(np.clip(raw_opacity * 1.5, 0.0, 100.0))

    return round(final_opacity, 1), dust_mask


def draw_hud(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    opacity: float,
    misting_status: str,
    threshold: float = 35.0
) -> np.ndarray:
    """Draws professional computer vision annotations, bounding boxes, and municipal HUD badges."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw YOLO Bounding Boxes
    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['label'].upper()} {int(obj['conf']*100)}%"
        box_color = (0, 220, 255) if obj["label"] in ["truck", "bus", "car"] else (255, 180, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        # Label badge
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + 120, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 2. Dust Heatmap Tint if high opacity
    if opacity >= threshold:
        tint_layer = np.zeros_like(annotated)
        tint_layer[:, :] = (30, 80, 180)  # Orange-amber alert tint
        cv2.addWeighted(annotated, 0.82, tint_layer, 0.18, 0, annotated)

    # 3. Top HUD Banner (Dark translucent header)
    hud_bg = annotated.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 54), (18, 22, 28), -1)
    cv2.addWeighted(hud_bg, 0.85, annotated, 0.15, 0, annotated)

    # Telemetry text
    cv2.putText(annotated, "DUSTGUARD-VISION | AI MUNICIPAL INSPECTOR", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 225, 230), 1, cv2.LINE_AA)
    
    # Opacity text with color cue
    opacity_color = (0, 70, 240) if opacity >= threshold else (80, 220, 80)
    cv2.putText(annotated, f"OPACITY: {opacity:.1f}% (LIMIT: {threshold}%)", (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, opacity_color, 2, cv2.LINE_AA)

    # Misting Cannon Status Badge
    is_active = misting_status in ("ACTIVE_MISTING", "COOL_DOWN")
    badge_color = (0, 60, 230) if is_active else (50, 160, 50)
    badge_text = f"CANNON: {misting_status.replace('_', ' ')}"
    
    tw = len(badge_text) * 11
    cv2.rectangle(annotated, (w - tw - 24, 10), (w - 12, 44), badge_color, -1)
    cv2.putText(annotated, badge_text, (w - tw - 16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def process_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Full vision pipeline on a single frame."""
    detections = detect_objects(frame)
    opacity, mask = opacity_score(frame)
    return {
        "objects": detections,
        "opacity": opacity,
        "dust_mask": mask
    }

"""
Utilities for detecting parking slots from camera images and mapping them to a grid.

The pipeline incorporates concepts from:
- Intelligent Parking Space Detection System Based on Image Processing
  (International Journal of Innovation, Management and Technology, Vol. 3, No. 3, June 2012)
- Car Parking Space Detection using Image Processing
  (International Journal of Advance Research, Ideas and Innovations in Technology, Vol. 7, Issue 4)

The core steps are:
1. System initialisation style preprocessing (resize, colour normalisation).
2. Image acquisition loading and conversion to grayscale.
3. Image segmentation via adaptive thresholding and HSV filtering.
4. Image enhancement leveraging morphological opening/closing to reduce noise and clarify slot boundaries.
5. Image detection analysing contour statistics (area, aspect ratio, solidity, circularity, eccentricity proxy)
   to decide whether a region represents a candidate parking slot and whether it is free/occupied.
6. Grid synthesis that maps detected slots to A* compatible occupancy grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SlotDetectionConfig:
    """
    Tunable parameters for the parking slot detection pipeline.
    """

    resize_width: int = 960
    gaussian_kernel: Tuple[int, int] = (5, 5)
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    hsv_lower_line: Tuple[int, int, int] = (5, 20, 70)  # Brown / yellowish lines lower bound
    hsv_upper_line: Tuple[int, int, int] = (35, 255, 255)  # Brown / yellowish lines upper bound
    min_area: int = 400
    max_area: int = 60000
    min_solidity: float = 0.35
    max_aspect_ratio: float = 4.0
    circularity_threshold: float = 0.3
    occupancy_intensity_threshold: float = 130.0
    occupancy_mask_ratio: float = 0.25
    morphology_kernel: Tuple[int, int] = (5, 5)
    morphology_iterations: int = 2
    grid_shape: Tuple[int, int] = (20, 20)


@dataclass
class YOLOParkingConfig:
    """
    Parameters for YOLOv8-based parking occupancy detection.
    """

    weights_path: str = "yolov8n.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    img_size: int = 640
    tracked_class_ids: Tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck (COCO ids)
    padding: int = 1
    grid_shape: Tuple[int, int] = (20, 20)
    device: Optional[str] = None  # e.g. "cpu", "cuda"


_YOLO_MODEL_CACHE: Dict[str, Any] = {}


def resize_keeping_aspect(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / float(w)
    dim = (width, int(h * scale))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def compute_slot_features(contour: np.ndarray) -> Dict[str, float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = 4.0 * np.pi * area / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    rect_area = float(w * h)
    extent = area / rect_area if rect_area > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    moments = cv2.moments(contour)
    eccentricity = 1.0
    if moments["mu20"] + moments["mu02"] != 0:
        numerator = ((moments["mu20"] - moments["mu02"]) ** 2) + (4 * (moments["mu11"] ** 2))
        numerator = np.sqrt(max(numerator, 0.0))
        denominator = moments["mu20"] + moments["mu02"]
        if denominator != 0:
            eccentricity = (moments["mu20"] + moments["mu02"] + numerator) / (moments["mu20"] + moments["mu02"] - numerator + 1e-6)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "bounding_rect": (x, y, w, h),
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "eccentricity": eccentricity,
    }


def detect_parking_slots(
    image_bgr: np.ndarray, config: SlotDetectionConfig
) -> Dict[str, Any]:
    """
    Detect potential parking slots inside the provided image.
    """
    debug: Dict[str, Any] = {}

    resized = resize_keeping_aspect(image_bgr, config.resize_width)
    debug["resized"] = resized

    blurred = cv2.GaussianBlur(resized, config.gaussian_kernel, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_lines = cv2.inRange(hsv, config.hsv_lower_line, config.hsv_upper_line)
    debug["mask_lines"] = mask_lines

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_tile_grid
    )
    gray_equalized = clahe.apply(gray)
    debug["gray"] = gray_equalized

    adaptive = cv2.adaptiveThreshold(
        gray_equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        5,
    )
    combined_mask = cv2.bitwise_or(adaptive, mask_lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.morphology_kernel)
    opened = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel, iterations=config.morphology_iterations
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=config.morphology_iterations
    )
    debug["binary"] = closed

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    slots: List[Dict[str, Any]] = []
    overlay = resized.copy()

    for contour in contours:
        features = compute_slot_features(contour)
        area = features["area"]
        if area < config.min_area or area > config.max_area:
            continue

        if features["solidity"] < config.min_solidity:
            continue

        if features["aspect_ratio"] > config.max_aspect_ratio:
            continue

        if features["circularity"] < config.circularity_threshold:
            continue

        x, y, w, h = features["bounding_rect"]
        centroid = (int(x + w / 2), int(y + h / 2))
        roi_gray = gray_equalized[y : y + h, x : x + w]
        roi_binary = closed[y : y + h, x : x + w]

        mean_intensity = float(np.mean(roi_gray)) if roi_gray.size else 0.0
        white_ratio = float(np.mean(roi_binary == 255)) if roi_binary.size else 0.0

        is_occupied = (
            mean_intensity < config.occupancy_intensity_threshold
            and white_ratio < config.occupancy_mask_ratio
        )

        status = "occupied" if is_occupied else "empty"

        color = (0, 0, 255) if is_occupied else (0, 200, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.circle(overlay, centroid, 4, (255, 255, 0), -1)
        cv2.putText(
            overlay,
            status.upper(),
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        slots.append(
            {
                "centroid": centroid,
                "bounding_rect": (x, y, w, h),
                "status": status,
                "features": features,
                "mean_intensity": mean_intensity,
                "white_ratio": white_ratio,
            }
        )

    debug["overlay"] = overlay
    debug["slots"] = sorted(slots, key=lambda s: (s["centroid"][1], s["centroid"][0]))

    return debug


def load_yolo_model(weights_path: str, device: Optional[str]) -> Any:
    """
    Load (or reuse) a YOLO model instance.
    """
    if weights_path in _YOLO_MODEL_CACHE:
        return _YOLO_MODEL_CACHE[weights_path]

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - hardware dependent
        raise ImportError(
            "YOLOv8 detection requires the 'ultralytics' package. "
            "Install it with 'pip install ultralytics'."
        ) from exc

    model = YOLO(weights_path)
    if device is not None:
        model.to(device)
    _YOLO_MODEL_CACHE[weights_path] = model
    return model


def detect_parking_slots_yolo(
    image_bgr: np.ndarray, config: YOLOParkingConfig
) -> Dict[str, Any]:
    """
    Detect vehicles using YOLOv8 and produce bounding boxes that map to occupied slots.
    """
    model = load_yolo_model(config.weights_path, config.device)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=image_rgb,
        imgsz=config.img_size,
        conf=config.conf_threshold,
        iou=config.iou_threshold,
        verbose=False,
    )

    if not results:
        return {
            "resized": image_bgr.copy(),
            "overlay": image_bgr.copy(),
            "slots": [],
        }

    result = results[0]
    boxes = result.boxes
    names = result.names if hasattr(result, "names") else {}

    overlay = image_bgr.copy()
    slots: List[Dict[str, Any]] = []

    if boxes is None:
        return {
            "resized": image_bgr.copy(),
            "overlay": overlay,
            "slots": [],
        }

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
    confs = (
        boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
    )
    classes = (
        boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()
    )

    for box, conf, cls_id in zip(xyxy, confs, classes):
        cls_int = int(cls_id)
        if cls_int not in config.tracked_class_ids:
            continue

        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        centroid = (int(x1 + w / 2), int(y1 + h / 2))
        label = names.get(cls_int, f"class_{cls_int}")
        label_text = f"{label} {conf:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(overlay, centroid, 4, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            label_text,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        slots.append(
            {
                "centroid": centroid,
                "bounding_rect": (x1, y1, w, h),
                "status": "occupied",
                "confidence": float(conf),
                "class_id": cls_int,
                "label": label,
                "mean_intensity": float("nan"),
                "white_ratio": float("nan"),
            }
        )

    slots_sorted = sorted(slots, key=lambda s: (s["centroid"][1], s["centroid"][0]))
    return {
        "resized": image_bgr.copy(),
        "overlay": overlay,
        "slots": slots_sorted,
    }


def slots_to_grid(
    image_shape: Tuple[int, int],
    slots: List[Dict[str, Any]],
    grid_shape: Tuple[int, int],
    padding: int = 1,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, Any]]]:
    """
    Map detected slots to a coarse occupancy grid usable by the path planning module.

    Occupied slots become obstacles. Empty slots remain traversable and tracked to allow
    targeting an empty bay as a goal.
    """
    grid = np.zeros(grid_shape, dtype=np.uint8)
    slot_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}

    height, width = image_shape[:2]
    rows, cols = grid_shape

    for idx, slot in enumerate(slots):
        x, y, w, h = slot["bounding_rect"]
        cx, cy = slot["centroid"]

        grid_row = min(rows - 1, max(0, int(cy / height * rows)))
        grid_col = min(cols - 1, max(0, int(cx / width * cols)))

        if slot["status"] == "occupied":
            r_start = max(0, grid_row - padding)
            r_end = min(rows, grid_row + padding + 1)
            c_start = max(0, grid_col - padding)
            c_end = min(cols, grid_col + padding + 1)
            grid[r_start:r_end, c_start:c_end] = 1

        slot_lookup[(grid_row, grid_col)] = {
            "slot_index": idx,
            **slot,
        }

    return grid, slot_lookup


def process_parking_image(
    image_path: str,
    config: Optional[SlotDetectionConfig] = None,
    method: str = "contour",
    yolo_config: Optional[YOLOParkingConfig] = None,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """
    Helper for reading an image, detecting parking slots and producing
    grid data along with optional plots.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at path: {image_path}")

    method = method.lower()
    if method not in {"contour", "yolo"}:
        raise ValueError("method must be either 'contour' or 'yolo'")

    if method == "yolo":
        if yolo_config is None:
            yolo_config = YOLOParkingConfig()
        detection_data = detect_parking_slots_yolo(image, yolo_config)
        grid_shape = yolo_config.grid_shape
    else:
        if config is None:
            config = SlotDetectionConfig()
        detection_data = detect_parking_slots(image, config)
        grid_shape = config.grid_shape

    slots: List[Dict[str, Any]] = detection_data.get("slots", [])

    grid, lookup = slots_to_grid(
        detection_data["resized"].shape, slots, grid_shape
    )

    suggested_goal: Optional[Tuple[int, int]] = None
    if method == "contour":
        for cell, info in sorted(lookup.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if info["status"] == "empty":
                suggested_goal = cell
                break
    else:
        zero_cells = np.argwhere(grid == 0)
        if zero_cells.size > 0:
            suggested_goal = tuple(zero_cells[0])

    result = {
        "original": image,
        "resized": detection_data["resized"],
        "gray": detection_data.get("gray"),
        "binary": detection_data.get("binary"),
        "overlay": detection_data["overlay"],
        "slots": slots,
        "grid": grid,
        "slot_lookup": lookup,
        "config": yolo_config if method == "yolo" else config,
        "suggested_goal": suggested_goal,
        "method": method,
    }

    if show_plots and method == "yolo":
        plot_detection_results_matplotlib(result)

    return result


def plot_detection_results_matplotlib(data: Dict[str, Any]) -> None:
    """
    Display YOLO detection overlay and occupancy grid using Matplotlib.
    """
    overlay = data.get("overlay")
    grid = data.get("grid")
    if overlay is None or grid is None:
        return

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(overlay_rgb)
    axes[0].set_title("YOLOv8 Detections")
    axes[0].axis("off")

    axes[1].imshow(grid, cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title("Occupancy Grid")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    fig.suptitle("Parking Occupancy Detection")
    plt.tight_layout()
    plt.show()


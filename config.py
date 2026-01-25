"""Configuration constants for the Parking Assistant application."""

import numpy as np
from dataclasses import dataclass

@dataclass
class UIConfig:
    """UI-related configuration."""
    WINDOW_TITLE = "Parking Lot Path Planner"
    WINDOW_GEOMETRY = "1400x900"
    DEFAULT_GRID_ROWS = 80
    DEFAULT_GRID_COLS = 120
    DEFAULT_CELL_SIZE = 10
    DEFAULT_SIM_SPEED = 30
    GRID_CELL_DISPLAY_SIZE = 10
    
    # Grid display mode
    # True = Fit grid to canvas (resized, no scrollbars)
    # False = Show actual grid size (with scrollbars if needed)
    FIT_GRID_TO_CANVAS = True

@dataclass
class DetectionConfig:
    """Object detection configuration."""
    YOLO_MODEL_PATH = 'yolov8s_fold_0.pt'
    YOLO_FALLBACK_MODEL = 'yolov8n.pt'
    DEFAULT_CONFIDENCE = 0.30
    
    # Parking spot detection parameters
    CONTOUR_MIN_AREA = 2500
    CONTOUR_MAX_AREA = 25000
    MIN_ASPECT_RATIO = 0.1
    MAX_ASPECT_RATIO = 0.75
    MIN_RECTANGULARITY = 0.75
    DUPLICATE_DISTANCE_THRESHOLD = 50
    
    # COCO classes considered as obstacles
    OBSTACLE_CLASSES = {
        0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 
        7: 'truck', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 13: 'bench', 15: 'cat', 16: 'dog', 17: 'fence',
        24: 'backpack', 26: 'handbag', 28: 'suitcase',
        39: 'bottle', 41: 'cup', 56: 'chair', 57: 'couch',
        58: 'potted plant', 59: 'bed', 60: 'dining table',
        62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 73: 'book'
    }

@dataclass
class GridConfig:
    """Grid generation configuration."""
    GRID_NAVIGABLE = 0
    GRID_OBSTACLE = 1
    GRID_PARKING_SPOT = 2
    
    CELL_OVERLAP_THRESHOLD = 0.2
    DEFAULT_CLEARANCE_RADIUS = 0

@dataclass
class PathPlanningConfig:
    """Path planning configuration."""
    SMOOTHING_FACTOR = 0.01
    NUM_SMOOTHING_POINTS = 100
    
    # A* movement directions (8-directional)
    MOVEMENT_DIRECTIONS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

@dataclass
class SimulationConfig:
    """Car simulation configuration."""
    DEFAULT_VELOCITY = 1.0
    CAR_WHEELBASE = 2.5
    LOOKAHEAD_DISTANCE = 3.0
    MAX_STEERING_ANGLE = np.radians(35)
    MPC_HORIZON = 5
    SIMULATION_DT = 0.15
    
    # Car dimensions relative to parking spot
    CAR_WIDTH_RATIO = 0.75
    CAR_LENGTH_RATIO = 0.65
    FALLBACK_CAR_LENGTH = 30
    FALLBACK_CAR_WIDTH = 18
    
    # Rendering
    WHEEL_LENGTH_RATIO = 5
    WHEEL_WIDTH_RATIO = 6

@dataclass
class ColorConfig:
    """Color scheme for visualization."""
    # RGB format
    OBSTACLE_COLOR = (255, 0, 0)  # Red
    EMPTY_SPOT_COLOR = (0, 255, 0)  # Green
    PATH_COLOR = (255, 200, 0)  # Yellow-orange
    SMOOTHED_PATH_COLOR = (0, 255, 0)  # Green
    START_POINT_COLOR = (0, 255, 255)  # Cyan
    END_POINT_COLOR = (255, 0, 255)  # Magenta
    CAR_BODY_COLOR = (0, 0, 255)  # Red
    CAR_BORDER_COLOR = (0, 0, 139)  # Dark red
    WHEEL_COLOR = (20, 20, 20)  # Dark gray
    HEADLIGHT_COLOR = (255, 255, 100)  # Yellow
    
    # Grid visualization colors (hex)
    GRID_OBSTACLE_HEX = "#ff3333"     # Red
    GRID_PARKING_HEX = "#33ff33"      # Green
    GRID_NAVIGABLE_HEX = "#ffffff"    # White
    GRID_PATH_HEX = "#ffff00"         # Yellow
    GRID_START_HEX = "#00ffff"        # Cyan
    GRID_END_HEX = "#ff00ff"          # Magenta

@dataclass
class OutputConfig:
    """Output file configuration."""
    PROCESS_IMAGES_FOLDER = "cv_process_images"
    
    YOLO_DETECTION_FILENAME = "16_yolo_detection.png"
    FINAL_COMBINED_FILENAME = "17_final_empty_spots_and_obstacles.png"
    GRID_MATRIX_FILENAME = "18_grid_matrix.png"
    INFLATED_GRID_FILENAME = "19_inflated_grid.png"
    ASTAR_PATH_FILENAME = "20_astar_path.png"
    SMOOTHED_PATH_FILENAME = "21_smoothed_path.png"
   
    
    IMAGE_SCALE_FACTOR = 5


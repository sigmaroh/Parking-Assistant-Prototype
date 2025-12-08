"""
PARKING SPACE DETECTION SYSTEM
===============================
A comprehensive solution for detecting parking spaces from images/videos.

Features:
- Multiple detection methods (YOLO, Mask R-CNN, Computer Vision)
- Space occupancy classification
- Real-time video processing
- Bird's eye view transformation
- Space counting and analytics
- Export results to various formats

Author: Computer Vision Parking System
Date: 2024
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import pandas as pd
import json
import warnings
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import time
from PIL import Image
from tqdm import tqdm
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES AND CONFIGURATION
# ============================================================================

@dataclass
class ParkingSpace:
    """Represents a detected parking space"""
    id: int
    vertices: np.ndarray  # 4 points in clockwise order
    center: Tuple[float, float]
    area: float
    is_occupied: bool = False
    confidence: float = 0.0
    vehicle_type: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

@dataclass
class DetectionConfig:
    """Configuration for parking space detection - Optimized for Layout 1.png"""
    # Image processing - preserve detail for parking line detection
    resize_width: int = 1200
    resize_height: int = 800
    gaussian_blur: Tuple[int, int] = (5, 5)  # Smaller blur to preserve line edges
    
    # Edge detection - tuned for white lines on gray pavement
    canny_threshold1: int = 30
    canny_threshold2: int = 100
    
    # Line detection - adjusted for parking space line segments
    hough_threshold: int = 50
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 15
    
    # Contour detection - larger areas for bird's eye view parking spaces
    contour_min_area: int = 3000
    contour_max_area: int = 50000
    
    # Space classification
    occupancy_threshold: float = 0.08  # Lower threshold for occupied detection
    iou_threshold: float = 0.4  # Slightly lower to merge overlapping detections
    
    # Deep learning
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    
    # Visualization
    draw_spaces: bool = True
    draw_occupancy: bool = True
    show_grid: bool = False

# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

class ImageProcessor:
    """Image processing utilities for parking space detection"""
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            width: Target width
            height: Target height
            
        Returns:
            Resized image
        """
        if width is None and height is None:
            return image
        
        h, w = image.shape[:2]
        
        if width is None:
            ratio = height / h
            new_width = int(w * ratio)
            new_height = height
        elif height is None:
            ratio = width / w
            new_width = width
            new_height = int(h * ratio)
        else:
            new_width = width
            new_height = height
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def preprocess_image(image: np.ndarray, config: DetectionConfig) -> np.ndarray:
        """
        Preprocess image for parking space detection
        
        Args:
            image: Input image
            config: Detection configuration
            
        Returns:
            Preprocessed image
        """
        # Resize
        if config.resize_width and config.resize_height:
            image = ImageProcessor.resize_image(image, config.resize_width, config.resize_height)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        if config.gaussian_blur[0] > 0 and config.gaussian_blur[1] > 0:
            gray = cv2.GaussianBlur(gray, config.gaussian_blur, 0)
        
        return gray
    
    @staticmethod
    def get_bird_eye_view(image: np.ndarray, src_points: np.ndarray, 
                          dst_width: int = 400, dst_height: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bird's eye view transformation
        
        Args:
            image: Input image
            src_points: Source points (4 points)
            dst_width: Output width
            dst_height: Output height
            
        Returns:
            Tuple of (transformed_image, transformation_matrix)
        """
        # Destination points (rectangle)
        dst_points = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, matrix, (dst_width, dst_height))
        
        return warped, matrix
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            image: Input image
            
        Returns:
            Contrast enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE to grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    @staticmethod
    def detect_edges(image: np.ndarray, config: DetectionConfig) -> np.ndarray:
        """
        Detect edges using Canny edge detector - optimized for parking lines
        
        Args:
            image: Input image
            config: Detection configuration
            
        Returns:
            Edge image
        """
        # Apply adaptive thresholding to enhance white parking lines
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            image, 
            config.canny_threshold1, 
            config.canny_threshold2
        )
        
        # Combine edges with binary threshold for better line detection
        combined = cv2.bitwise_or(edges, binary)
        
        # Apply morphological closing to connect parking line gaps
        kernel_close = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate to connect nearby line segments
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(closed, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    @staticmethod
    def find_contours(edge_image: np.ndarray, config: DetectionConfig) -> List[np.ndarray]:
        """
        Find contours in edge image - optimized for parking spaces
        
        Args:
            edge_image: Edge detection result
            config: Detection configuration
            
        Returns:
            List of contours
        """
        # Find all contours including nested ones
        contours, _ = cv2.findContours(
            edge_image, 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area and aspect ratio
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (config.contour_min_area <= area <= config.contour_max_area):
                continue
            
            # Check aspect ratio (parking spaces are typically taller than wide)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            # Accept contours with parking-space-like aspect ratios
            if 0.8 <= aspect_ratio <= 3.5:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    @staticmethod
    def detect_lines(edge_image: np.ndarray, config: DetectionConfig) -> List[np.ndarray]:
        """
        Detect lines using Hough Transform
        
        Args:
            edge_image: Edge detection result
            config: Detection configuration
            
        Returns:
            List of lines [x1, y1, x2, y2]
        """
        lines = cv2.HoughLinesP(
            edge_image,
            rho=1,
            theta=np.pi/180,
            threshold=config.hough_threshold,
            minLineLength=config.hough_min_line_length,
            maxLineGap=config.hough_max_line_gap
        )
        
        if lines is None:
            return []
        
        return lines.squeeze()
    
    @staticmethod
    def cluster_lines(lines: np.ndarray, angle_threshold: float = 10.0) -> List[np.ndarray]:
        """
        Cluster lines by angle and position
        
        Args:
            lines: Detected lines
            angle_threshold: Angle threshold for clustering (degrees)
            
        Returns:
            Clustered lines
        """
        if len(lines) == 0:
            return []
        
        # Calculate line angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angles.append(angle)
        
        angles = np.array(angles)
        
        # Cluster by angle using DBSCAN
        angle_rad = np.radians(angles).reshape(-1, 1)
        clustering = DBSCAN(eps=np.radians(angle_threshold), min_samples=2).fit(angle_rad)
        
        # Group lines by cluster
        clustered_lines = []
        unique_labels = np.unique(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            cluster_lines = lines[clustering.labels_ == label]
            clustered_lines.append(cluster_lines)
        
        return clustered_lines

# ============================================================================
# PARKING SPACE DETECTOR (COMPUTER VISION APPROACH)
# ============================================================================

class ParkingSpaceDetectorCV:
    """
    Parking space detector using traditional computer vision techniques
    """
    
    def __init__(self, config: DetectionConfig = None):
        """
        Initialize detector
        
        Args:
            config: Detection configuration
        """
        self.config = config or DetectionConfig()
        self.parking_spaces = []
        self.transformation_matrix = None
        self.inverse_matrix = None
        
    def detect_from_image(self, image: np.ndarray, 
                          src_points: Optional[np.ndarray] = None) -> List[ParkingSpace]:
        """
        Detect parking spaces from image
        
        Args:
            image: Input image
            src_points: Source points for perspective transform (optional)
            
        Returns:
            List of detected parking spaces
        """
        logger.info("Starting parking space detection...")
        
        # Preprocess image
        processed = ImageProcessor.preprocess_image(image, self.config)
        
        # Get bird's eye view if points are provided
        if src_points is not None and len(src_points) == 4:
            warped, self.transformation_matrix = ImageProcessor.get_bird_eye_view(
                processed, src_points
            )
            self.inverse_matrix = cv2.getPerspectiveTransform(
                np.array([[0, 0], [400, 0], [400, 600], [0, 600]], dtype=np.float32),
                src_points.astype(np.float32)
            )
            processed = warped
        
        # Detect edges
        edges = ImageProcessor.detect_edges(processed, self.config)
        
        # Method 1: Detect lines and find intersections
        spaces_from_lines = self._detect_from_lines(processed, edges)
        
        # Method 2: Detect contours
        spaces_from_contours = self._detect_from_contours(processed, edges)
        
        # Combine results
        all_spaces = spaces_from_lines + spaces_from_contours
        self.parking_spaces = self._merge_duplicate_spaces(all_spaces)
        
        # Transform back to original coordinates if perspective was applied
        if self.inverse_matrix is not None:
            self._transform_to_original_coordinates(image)
        
        logger.info(f"Detected {len(self.parking_spaces)} parking spaces")
        return self.parking_spaces
    
    def _detect_from_lines(self, image: np.ndarray, edges: np.ndarray) -> List[ParkingSpace]:
        """
        Detect parking spaces from lines
        
        Args:
            image: Preprocessed image
            edges: Edge detection result
            
        Returns:
            List of parking spaces
        """
        spaces = []
        
        # Detect lines
        lines = ImageProcessor.detect_lines(edges, self.config)
        if len(lines) == 0:
            return spaces
        
        # Cluster lines by angle
        clustered_lines = ImageProcessor.cluster_lines(lines)
        
        # Find intersections between perpendicular line clusters
        for i, cluster1 in enumerate(clustered_lines):
            for j, cluster2 in enumerate(clustered_lines[i+1:], i+1):
                # Check if clusters are approximately perpendicular
                angle1 = self._get_average_angle(cluster1)
                angle2 = self._get_average_angle(cluster2)
                
                if self._are_perpendicular(angle1, angle2, threshold=20):
                    # Find intersections and form rectangles
                    rectangles = self._form_rectangles_from_lines(cluster1, cluster2)
                    
                    for rect_id, rect in enumerate(rectangles):
                        # Create parking space object
                        space = ParkingSpace(
                            id=len(spaces) + rect_id,
                            vertices=rect,
                            center=self._get_center(rect),
                            area=cv2.contourArea(rect.reshape(-1, 1, 2)),
                            is_occupied=False
                        )
                        spaces.append(space)
        
        return spaces
    
    def _detect_from_contours(self, image: np.ndarray, edges: np.ndarray) -> List[ParkingSpace]:
        """
        Detect parking spaces from contours - optimized for parking layouts
        
        Args:
            image: Preprocessed image
            edges: Edge detection result
            
        Returns:
            List of parking spaces
        """
        spaces = []
        detected_centers = []  # Track centers to avoid duplicates
        
        # Find contours
        contours = ImageProcessor.find_contours(edges, self.config)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Try multiple epsilon values for better polygon approximation
            for eps_factor in [0.01, 0.02, 0.03, 0.04]:
                epsilon = eps_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Accept 4-6 vertex polygons (can be approximated to rectangles)
                if 4 <= len(approx) <= 6:
                    # Use minimum area rectangle for consistent shape
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    ordered = self._order_points(box)
                    
                    # Check fill ratio to ensure it's a valid rectangle
                    rect_area = cv2.contourArea(ordered)
                    if rect_area > 0 and area / rect_area > 0.6:
                        center = self._get_center(ordered)
                        
                        # Check for duplicate (nearby center)
                        is_duplicate = False
                        for existing_center in detected_centers:
                            dist = np.sqrt((center[0] - existing_center[0])**2 + 
                                         (center[1] - existing_center[1])**2)
                            if dist < 50:  # 50 pixel threshold
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            detected_centers.append(center)
                            
                            # Calculate bounding box
                            x_min = int(ordered[:, 0].min())
                            y_min = int(ordered[:, 1].min())
                            x_max = int(ordered[:, 0].max())
                            y_max = int(ordered[:, 1].max())
                            
                            space = ParkingSpace(
                                id=len(spaces),
                                vertices=ordered,
                                center=center,
                                area=rect_area,
                                is_occupied=False,
                                bbox=(x_min, y_min, x_max - x_min, y_max - y_min)
                            )
                            spaces.append(space)
                        break
        
        return spaces
    
    def _get_average_angle(self, lines: np.ndarray) -> float:
        """Calculate average angle of lines"""
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angles.append(angle)
        return np.mean(angles) if angles else 0
    
    def _are_perpendicular(self, angle1: float, angle2: float, threshold: float = 15) -> bool:
        """Check if two angles are perpendicular within threshold"""
        diff = abs(abs(angle1 - angle2) - 90)
        return diff <= threshold
    
    def _form_rectangles_from_lines(self, lines1: np.ndarray, lines2: np.ndarray) -> List[np.ndarray]:
        """Form rectangles from two sets of perpendicular lines"""
        rectangles = []
        
        # Get intersection points
        intersections = []
        for line1 in lines1:
            for line2 in lines2:
                inter = self._line_intersection(line1, line2)
                if inter is not None:
                    intersections.append(inter)
        
        if len(intersections) < 4:
            return rectangles
        
        # Cluster intersections to find rectangle corners
        intersections = np.array(intersections)
        
        # Use DBSCAN to find clusters (potential corners)
        clustering = DBSCAN(eps=20, min_samples=2).fit(intersections)
        
        # Get cluster centers
        unique_labels = np.unique(clustering.labels_)
        corners = []
        
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = intersections[clustering.labels_ == label]
            center = np.mean(cluster_points, axis=0)
            corners.append(center)
        
        # Need at least 4 corners for a rectangle
        if len(corners) >= 4:
            # Sort corners to form rectangle
            corners = np.array(corners)
            ordered = self._order_points(corners[:4])
            rectangles.append(ordered)
        
        return rectangles
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
        """Find intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        
        if abs(denom) < 1e-5:
            return None  # Lines are parallel
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return np.array([x, y])
        
        return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left"""
        # Sort by x-coordinate
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        
        # Get left-most and right-most points
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        
        # Sort left-most by y-coordinate (top-left, bottom-left)
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        tl, bl = left_most[0], left_most[1]
        
        # Sort right-most by y-coordinate (top-right, bottom-right)
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        tr, br = right_most[0], right_most[1]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    def _get_center(self, vertices: np.ndarray) -> Tuple[float, float]:
        """Calculate center of polygon"""
        center = np.mean(vertices, axis=0)
        return (float(center[0]), float(center[1]))
    
    def _merge_duplicate_spaces(self, spaces: List[ParkingSpace]) -> List[ParkingSpace]:
        """Merge duplicate or overlapping parking spaces"""
        if not spaces:
            return []
        
        # Calculate IoU between all spaces
        merged = []
        used = [False] * len(spaces)
        
        for i, space1 in enumerate(spaces):
            if used[i]:
                continue
            
            # Find overlapping spaces
            overlapping = [space1]
            for j, space2 in enumerate(spaces[i+1:], i+1):
                if used[j]:
                    continue
                
                iou = self._calculate_iou(space1.vertices, space2.vertices)
                if iou > self.config.iou_threshold:
                    overlapping.append(space2)
                    used[j] = True
            
            # Merge overlapping spaces
            if len(overlapping) > 1:
                # Use the space with largest area
                merged_space = max(overlapping, key=lambda x: x.area)
                merged.append(merged_space)
            else:
                merged.append(space1)
            
            used[i] = True
        
        # Reassign IDs
        for i, space in enumerate(merged):
            space.id = i
        
        return merged
    
    def _calculate_iou(self, vertices1: np.ndarray, vertices2: np.ndarray) -> float:
        """Calculate Intersection over Union for two polygons"""
        # Convert to contours
        cnt1 = vertices1.reshape(-1, 1, 2).astype(np.int32)
        cnt2 = vertices2.reshape(-1, 1, 2).astype(np.int32)
        
        # Calculate intersection area
        intersection = np.zeros((max(vertices1[:, 1].max(), vertices2[:, 1].max()) + 10,
                               max(vertices1[:, 0].max(), vertices2[:, 0].max()) + 10), dtype=np.uint8)
        
        cv2.drawContours(intersection, [cnt1], 0, 1, thickness=cv2.FILLED)
        cv2.drawContours(intersection, [cnt2], 0, 1, thickness=cv2.FILLED)
        
        intersection_area = np.sum(intersection == 2)
        
        # Calculate union area
        union = np.zeros_like(intersection)
        cv2.drawContours(union, [cnt1], 0, 1, thickness=cv2.FILLED)
        cv2.drawContours(union, [cnt2], 0, 1, thickness=cv2.FILLED)
        
        union_area = np.sum(union > 0)
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _transform_to_original_coordinates(self, original_image: np.ndarray):
        """Transform parking spaces back to original image coordinates"""
        if self.inverse_matrix is None:
            return
        
        for space in self.parking_spaces:
            # Transform vertices
            vertices_homogeneous = np.hstack([space.vertices, np.ones((4, 1))])
            transformed = (self.inverse_matrix @ vertices_homogeneous.T).T
            space.vertices = (transformed[:, :2] / transformed[:, 2:3]).astype(np.float32)
            
            # Update center
            space.center = self._get_center(space.vertices)
            
            # Calculate bounding box
            x_min = int(space.vertices[:, 0].min())
            y_min = int(space.vertices[:, 1].min())
            x_max = int(space.vertices[:, 0].max())
            y_max = int(space.vertices[:, 1].max())
            space.bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def detect_occupancy(self, image: np.ndarray) -> List[ParkingSpace]:
        """
        Detect which parking spaces are occupied
        
        Args:
            image: Input image
            
        Returns:
            Updated list of parking spaces with occupancy status
        """
        if not self.parking_spaces:
            logger.warning("No parking spaces detected. Run detect_from_image first.")
            return []
        
        processed = ImageProcessor.preprocess_image(image, self.config)
        
        for space in self.parking_spaces:
            # Extract region of interest
            mask = np.zeros(processed.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [space.vertices.astype(np.int32)], 255)
            
            roi = cv2.bitwise_and(processed, processed, mask=mask)
            
            # Calculate occupancy features
            occupancy_score = self._calculate_occupancy_score(roi)
            space.is_occupied = occupancy_score > self.config.occupancy_threshold
            space.confidence = occupancy_score
        
        return self.parking_spaces
    
    def _calculate_occupancy_score(self, roi: np.ndarray) -> float:
        """Calculate occupancy score for a region"""
        if roi.size == 0:
            return 0.0
        
        # Method 1: Edge density
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1] + 1e-5)
        
        # Method 2: Texture analysis using variance
        variance = np.var(roi) / 255.0
        
        # Method 3: Histogram analysis
        hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
        hist_norm = hist / (roi.size + 1e-5)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-5))
        entropy_norm = entropy / 8.0  # Normalize to [0, 1]
        
        # Combine scores
        occupancy_score = 0.4 * edge_density + 0.3 * variance + 0.3 * entropy_norm
        
        return min(max(occupancy_score, 0.0), 1.0)
    
    def visualize(self, image: np.ndarray, show_occupancy: bool = True) -> np.ndarray:
        """
        Visualize detected parking spaces
        
        Args:
            image: Original image
            show_occupancy: Whether to show occupancy status
            
        Returns:
            Image with visualization
        """
        result = image.copy()
        
        for space in self.parking_spaces:
            # Draw parking space polygon
            color = (0, 255, 0) if not space.is_occupied else (0, 0, 255)
            thickness = 2
            
            pts = space.vertices.astype(np.int32)
            cv2.polylines(result, [pts], True, color, thickness)
            
            # Draw center point
            center = tuple(map(int, space.center))
            cv2.circle(result, center, 5, color, -1)
            
            # Draw space ID
            cv2.putText(result, f'{space.id}', (center[0] - 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(result, f'{space.id}', (center[0] - 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Show occupancy status if requested
            if show_occupancy:
                status = "Occupied" if space.is_occupied else "Empty"
                cv2.putText(result, status, (pts[0][0], pts[0][1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add statistics
        total_spaces = len(self.parking_spaces)
        occupied_spaces = sum(1 for space in self.parking_spaces if space.is_occupied)
        available_spaces = total_spaces - occupied_spaces
        
        cv2.putText(result, f'Total: {total_spaces}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(result, f'Total: {total_spaces}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        cv2.putText(result, f'Occupied: {occupied_spaces}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(result, f'Occupied: {occupied_spaces}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        cv2.putText(result, f'Available: {available_spaces}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(result, f'Available: {available_spaces}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        return result

# ============================================================================
# DEEP LEARNING BASED DETECTOR (YOLO)
# ============================================================================

class ParkingSpaceDetectorYOLO:
    """
    Parking space detector using YOLO (You Only Look Once) deep learning model
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights (.weights or .pt)
            config_path: Path to YOLO config file (.cfg)
        """
        self.model = None
        self.classes = ['car', 'motorcycle', 'bus', 'truck', 'parking-space']
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Try to load YOLO
        self._load_yolo(model_path, config_path)
        
    def _load_yolo(self, model_path: str, config_path: str):
        """Load YOLO model"""
        try:
            # Try PyTorch YOLO first
            if model_path and model_path.endswith('.pt'):
                self.model = torch.hub.load('ultralytics/yolov8', 'custom', 
                                           path=model_path, force_reload=False)
                logger.info("Loaded YOLOv5 PyTorch model")
                return
            
            # Try OpenCV DNN YOLO
            if model_path and config_path:
                if os.path.exists(model_path) and os.path.exists(config_path):
                    self.model = cv2.dnn.readNet(model_path, config_path)
                    
                    # Try to set preferable backend
                    try:
                        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        logger.info("Using CUDA backend for YOLO")
                    except:
                        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                        logger.info("Using CPU backend for YOLO")
                    
                    # Get output layer names
                    layer_names = self.model.getLayerNames()
                    self.output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
                    logger.info("Loaded YOLO model via OpenCV DNN")
                    return
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}")
        
        # Fallback: Use pre-trained YOLOv5 from torch hub
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            logger.info("Loaded pre-trained YOLOv5s model")
        except Exception as e:
            logger.error(f"Failed to load any YOLO model: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.14) -> List[Dict]:
        """
        Detect objects in image using YOLO
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of detected objects with bounding boxes
        """
        if self.model is None:
            logger.error("YOLO model not loaded")
            return []
        
        height, width = image.shape[:2]
        detections = []
        
        # PyTorch YOLOv5
        if hasattr(self.model, 'predict'):
            results = self.model(image)
            
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)
                    class_name = results.names[class_id]
                    
                    detections.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': class_name,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
        
        # OpenCV DNN YOLO
        elif isinstance(self.model, cv2.dnn.Net):
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), 
                                        swapRB=True, crop=False)
            self.model.setInput(blob)
            outputs = self.model.forward(self.output_layers)
            
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidences[i],
                        'class_id': class_ids[i],
                        'class_name': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else 'unknown',
                        'center': (x + w // 2, y + h // 2)
                    })
        
        return detections
    
    def detect_parking_spaces(self, image: np.ndarray, 
                             detect_vehicles: bool = True) -> List[ParkingSpace]:
        """
        Detect parking spaces and vehicles
        
        Args:
            image: Input image
            detect_vehicles: Whether to detect vehicles
            
        Returns:
            List of parking spaces
        """
        detections = self.detect(image)
        
        # Filter for parking spaces and vehicles
        parking_detections = [d for d in detections if d['class_name'] == 'parking-space']
        vehicle_detections = [d for d in detections if d['class_name'] in ['car', 'motorcycle', 'bus', 'truck']]
        
        # Create parking space objects
        parking_spaces = []
        
        for i, det in enumerate(parking_detections):
            x, y, w, h = det['bbox']
            
            # Create vertices from bounding box
            vertices = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
            
            # Check if occupied by any vehicle
            is_occupied = False
            for vehicle in vehicle_detections:
                vx, vy, vw, vh = vehicle['bbox']
                
                # Check if vehicle center is inside parking space
                v_center_x, v_center_y = vehicle['center']
                if (x <= v_center_x <= x + w and y <= v_center_y <= y + h):
                    is_occupied = True
                    break
            
            space = ParkingSpace(
                id=i,
                vertices=vertices,
                center=det['center'],
                area=w * h,
                is_occupied=is_occupied,
                confidence=det['confidence'],
                bbox=det['bbox']
            )
            
            parking_spaces.append(space)
        
        return parking_spaces

# ============================================================================
# HYBRID DETECTOR (COMBINES CV AND DL)
# ============================================================================

class HybridParkingDetector:
    """
    Hybrid detector combining computer vision and deep learning
    """
    
    def __init__(self, use_yolo: bool = True, yolo_model_path: str = None):
        """
        Initialize hybrid detector
        
        Args:
            use_yolo: Whether to use YOLO for vehicle detection
            yolo_model_path: Path to YOLO model
        """
        self.cv_detector = ParkingSpaceDetectorCV()
        self.yolo_detector = None
        
        if use_yolo:
            self.yolo_detector = ParkingSpaceDetectorYOLO(yolo_model_path)
        
        self.parking_spaces = []
        
    def detect(self, image: np.ndarray, method: str = 'hybrid') -> List[ParkingSpace]:
        """
        Detect parking spaces using specified method
        
        Args:
            image: Input image
            method: 'cv', 'yolo', or 'hybrid'
            
        Returns:
            List of parking spaces
        """
        if method == 'cv':
            # Use only computer vision
            self.parking_spaces = self.cv_detector.detect_from_image(image)
            
        elif method == 'yolo' and self.yolo_detector:
            # Use only YOLO
            self.parking_spaces = self.yolo_detector.detect_parking_spaces(image)
            
        elif method == 'hybrid':
            # Use CV to detect spaces, YOLO to check occupancy
            self.parking_spaces = self.cv_detector.detect_from_image(image)
            
            if self.yolo_detector and self.parking_spaces:
                # Use YOLO to detect vehicles for occupancy
                vehicle_detections = self.yolo_detector.detect(image)
                
                # Update occupancy based on vehicle detections
                for space in self.parking_spaces:
                    if space.bbox:
                        x, y, w, h = space.bbox
                        
                        # Check if any vehicle is inside this space
                        for vehicle in vehicle_detections:
                            if vehicle['class_name'] in ['car', 'motorcycle', 'bus', 'truck']:
                                vx, vy, vw, vh = vehicle['bbox']
                                v_center_x, v_center_y = vehicle['center']
                                
                                # Check if vehicle center is inside parking space
                                if (x <= v_center_x <= x + w and y <= v_center_y <= y + h):
                                    space.is_occupied = True
                                    space.vehicle_type = vehicle['class_name']
                                    break
        
        return self.parking_spaces
    
    def process_video(self, video_path: str, output_path: str = None, 
                      method: str = 'hybrid', show_video: bool = True):
        """
        Process video file for parking space detection
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            method: Detection method
            show_video: Whether to display video during processing
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics
        frame_count = 0
        processing_times = []
        
        # Process frames
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Detect parking spaces
            spaces = self.detect(frame, method)
            
            # Detect occupancy
            if spaces:
                if method == 'cv' or method == 'hybrid':
                    self.cv_detector.detect_occupancy(frame)
            
            # Visualize results
            if method == 'cv' or method == 'hybrid':
                result_frame = self.cv_detector.visualize(frame)
            else:
                result_frame = frame.copy()
                for space in spaces:
                    color = (0, 255, 0) if not space.is_occupied else (0, 0, 255)
                    pts = space.vertices.astype(np.int32)
                    cv2.polylines(result_frame, [pts], True, color, 2)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Add FPS counter
            fps_text = f"FPS: {1/processing_time:.1f}" if processing_time > 0 else "FPS: N/A"
            cv2.putText(result_frame, fps_text, (width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output
            if out:
                out.write(result_frame)
            
            # Show frame
            if show_video:
                cv2.imshow('Parking Space Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        avg_time = np.mean(processing_times) if processing_times else 0
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        
        logger.info(f"Processed {frame_count} frames")
        logger.info(f"Average processing time: {avg_time:.3f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        
        if output_path:
            logger.info(f"Output saved to: {output_path}")
    
    def export_results(self, output_path: str, format: str = 'json'):
        """
        Export detection results
        
        Args:
            output_path: Output file path
            format: Export format ('json', 'csv', 'xml')
        """
        if not self.parking_spaces:
            logger.warning("No results to export")
            return
        
        results = []
        for space in self.parking_spaces:
            space_data = {
                'id': space.id,
                'vertices': space.vertices.tolist(),
                'center': space.center,
                'area': space.area,
                'is_occupied': space.is_occupied,
                'confidence': space.confidence,
                'vehicle_type': space.vehicle_type
            }
            
            if space.bbox:
                space_data['bbox'] = {
                    'x': space.bbox[0],
                    'y': space.bbox[1],
                    'width': space.bbox[2],
                    'height': space.bbox[3]
                }
            
            results.append(space_data)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results exported to JSON: {output_path}")
        
        elif format.lower() == 'csv':
            df_data = []
            for space in results:
                row = {
                    'id': space['id'],
                    'center_x': space['center'][0],
                    'center_y': space['center'][1],
                    'area': space['area'],
                    'is_occupied': space['is_occupied'],
                    'confidence': space['confidence'],
                    'vehicle_type': space['vehicle_type'] or ''
                }
                
                if 'bbox' in space:
                    row.update({
                        'bbox_x': space['bbox']['x'],
                        'bbox_y': space['bbox']['y'],
                        'bbox_width': space['bbox']['width'],
                        'bbox_height': space['bbox']['height']
                    })
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Results exported to CSV: {output_path}")
        
        else:
            logger.error(f"Unsupported format: {format}")

# ============================================================================
# MAIN APPLICATION AND EXAMPLES
# ============================================================================

class ParkingDetectionApp:
    """
    Main application class for parking space detection
    """
    
    @staticmethod
    def example_computer_vision(image_path: str):
        """
        Example using only computer vision
        
        Args:
            image_path: Path to input image
        """
        print("="*60)
        print("EXAMPLE 1: COMPUTER VISION APPROACH")
        print("="*60)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot load image: {image_path}")
            return
        
        # Create detector
        detector = ParkingSpaceDetectorCV()
        
        # Detect parking spaces
        spaces = detector.detect_from_image(image)
        
        # Detect occupancy
        if spaces:
            detector.detect_occupancy(image)
        
        # Visualize results
        result = detector.visualize(image)
        
        # Display results
        cv2.imshow('Computer Vision Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print statistics
        total = len(spaces)
        occupied = sum(1 for s in spaces if s.is_occupied)
        available = total - occupied
        
        print(f"\nDetection Results:")
        print(f"  Total spaces: {total}")
        print(f"  Occupied: {occupied}")
        print(f"  Available: {available}")
        
        return detector
    
    @staticmethod
    def example_hybrid(image_path: str, yolo_model_path: str = None):
        """
        Example using hybrid approach (CV + YOLO)
        
        Args:
            image_path: Path to input image
            yolo_model_path: Path to YOLO model (optional)
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: HYBRID APPROACH (CV + YOLO)")
        print("="*60)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot load image: {image_path}")
            return
        
        # Create hybrid detector
        detector = HybridParkingDetector(use_yolo=True, yolo_model_path=yolo_model_path)
        
        # Detect using hybrid method
        spaces = detector.detect(image, method='hybrid')
        
        # Display results
        result_image = image.copy()
        
        for space in spaces:
            color = (0, 255, 0) if not space.is_occupied else (0, 0, 255)
            pts = space.vertices.astype(np.int32)
            cv2.polylines(result_image, [pts], True, color, 2)
            
            # Add ID and status
            center = tuple(map(int, space.center))
            status = "O" if space.is_occupied else "E"
            cv2.putText(result_image, f'{space.id}:{status}', center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add statistics
        total = len(spaces)
        occupied = sum(1 for s in spaces if s.is_occupied)
        
        cv2.putText(result_image, f'Spaces: {total}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(result_image, f'Occupied: {occupied}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        cv2.imshow('Hybrid Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"\nHybrid Detection Results:")
        print(f"  Total spaces: {total}")
        print(f"  Occupied: {occupied}")
        print(f"  Available: {total - occupied}")
        
        return detector
    
    @staticmethod
    def example_video_processing(video_path: str, output_path: str = None):
        """
        Example of processing a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: VIDEO PROCESSING")
        print("="*60)
        
        # Create hybrid detector
        detector = HybridParkingDetector(use_yolo=False)  # Use CV only for speed
        
        # Process video
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            method='cv',
            show_video=True
        )
    
    @staticmethod
    def example_bird_eye_view(image_path: str):
        """
        Example with bird's eye view transformation
        
        Args:
            image_path: Path to input image
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: BIRD'S EYE VIEW TRANSFORMATION")
        print("="*60)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot load image: {image_path}")
            return
        
        # Let user select points for perspective transform
        points = []
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                
                if len(points) > 1:
                    cv2.line(image, points[-2], points[-1], (0, 255, 0), 2)
                
                if len(points) == 4:
                    cv2.line(image, points[-1], points[0], (0, 255, 0), 2)
                    cv2.imshow('Select Parking Area', image)
                
                cv2.imshow('Select Parking Area', image)
        
        cv2.imshow('Select Parking Area', image)
        cv2.setMouseCallback('Select Parking Area', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(points) != 4:
            logger.error("Need exactly 4 points for perspective transform")
            return
        
        # Convert points to numpy array
        src_points = np.array(points, dtype=np.float32)
        
        # Create detector
        detector = ParkingSpaceDetectorCV()
        
        # Detect with perspective transform
        spaces = detector.detect_from_image(image, src_points=src_points)
        
        if spaces:
            detector.detect_occupancy(image)
        
        # Visualize
        result = detector.visualize(image)
        cv2.imshow('Bird\'s Eye View Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detector
    
    @staticmethod
    def batch_process_images(image_folder: str, output_folder: str = None):
        """
        Process all images in a folder
        
        Args:
            image_folder: Path to folder containing images
            output_folder: Path to save results (optional)
        """
        print("\n" + "="*60)
        print("BATCH PROCESSING OF IMAGES")
        print("="*60)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.error(f"No images found in {image_folder}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Create output folder
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        # Create detector
        detector = HybridParkingDetector(use_yolo=False)
        
        # Process each image
        results = []
        
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Detect parking spaces
                spaces = detector.detect(image, method='cv')
                
                if spaces:
                    detector.cv_detector.detect_occupancy(image)
                
                # Save results
                if output_folder:
                    # Save annotated image
                    result = detector.cv_detector.visualize(image)
                    output_path = Path(output_folder) / f'result_{img_path.name}'
                    cv2.imwrite(str(output_path), result)
                
                # Collect statistics
                total = len(spaces)
                occupied = sum(1 for s in spaces if s.is_occupied)
                
                results.append({
                    'image': img_path.name,
                    'total_spaces': total,
                    'occupied_spaces': occupied,
                    'available_spaces': total - occupied
                })
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        # Save summary
        if results:
            df = pd.DataFrame(results)
            summary_path = Path(output_folder or '.') / 'parking_summary.csv'
            df.to_csv(summary_path, index=False)
            
            print(f"\nSummary saved to: {summary_path}")
            print(f"\nTotal images processed: {len(results)}")
            print(f"Average spaces per image: {df['total_spaces'].mean():.1f}")
            print(f"Average occupancy rate: {(df['occupied_spaces'] / df['total_spaces']).mean()*100:.1f}%")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function with example usage"""
    
    # Create sample images if none exist
    sample_images = create_sample_images()
    
    # Example 1: Computer Vision
    # app.example_computer_vision('sample_parking.jpg')
    
    # Example 2: Hybrid Approach
    # app.example_hybrid('sample_parking.jpg')
    
    # Example 3: Video Processing
    # app.example_video_processing('parking_video.mp4', 'output_video.mp4')
    
    # Example 4: Bird's Eye View
    # app.example_bird_eye_view('sample_parking.jpg')
    
    # Example 5: Batch Processing
    # app.batch_process_images('input_images/', 'output_results/')
    
    print("\nParking Space Detection System Ready!")
    print("Choose an example to run by uncommenting it in the main() function.")

def create_sample_images():
    """Create sample parking lot images for testing"""
    import numpy as np
    
    # Create a sample parking lot image
    width, height = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw parking spaces
    space_width, space_height = 80, 150
    margin = 20
    
    spaces = []
    
    # Draw parking lot markings
    for row in range(2):
        for col in range(4):
            x = margin + col * (space_width + margin)
            y = margin + row * (space_height + margin * 2)
            
            # Parking space outline
            cv2.rectangle(image, (x, y), (x + space_width, y + space_height), (0, 0, 0), 2)
            
            # Space number
            space_id = row * 4 + col + 1
            cv2.putText(image, str(space_id), (x + 30, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            spaces.append((x, y, space_width, space_height, space_id))
    
    # Randomly add some cars
    import random
    for i in range(3):
        space_idx = random.randint(0, len(spaces) - 1)
        x, y, w, h, space_id = spaces[space_idx]
        
        # Car dimensions (smaller than space)
        car_w, car_h = int(w * 0.8), int(h * 0.6)
        car_x = x + (w - car_w) // 2
        car_y = y + (h - car_h) // 2
        
        # Draw car
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(image, (car_x, car_y), (car_x + car_w, car_y + car_h), color, -1)
        
        # Car details
        cv2.rectangle(image, (car_x, car_y), (car_x + car_w, car_y + car_h), (0, 0, 0), 2)
        
        # Windows
        window_w, window_h = car_w // 3, car_h // 4
        cv2.rectangle(image, (car_x + window_w, car_y + 5),
                     (car_x + 2 * window_w, car_y + window_h), (200, 200, 255), -1)
    
    # Save sample image
    cv2.imwrite('sample_parking.jpg', image)
    print("Created sample_parking.jpg")
    
    return ['sample_parking.jpg']

if __name__ == "__main__":
    # Create application instance
    app = ParkingDetectionApp()
    
    # Run main function
    main()
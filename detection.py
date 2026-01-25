"""Object detection module for parking spots and obstacles."""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from config import DetectionConfig, OutputConfig

from ultralytics import YOLO


class ParkingSpotDetector:
    """Detects parking spots using traditional computer vision techniques."""
    
    def __init__(self):
        self.parking_spots = []
        
    def detect(self, image: np.ndarray, save_steps: bool = True) -> List[Dict]:
        """
        Detect parking spots using Hough line-based approach.
        
        Args:
            image: Input image in RGB format
            save_steps: Whether to save intermediate processing steps
            
        Returns:
            List of detected parking spot dictionaries
        """
        if save_steps:
            output_folder = OutputConfig.PROCESS_IMAGES_FOLDER
            os.makedirs(output_folder, exist_ok=True)
        
        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        if save_steps:
            cv2.imwrite(os.path.join(output_folder, "2_grayscale.png"), gray)
            cv2.imwrite(os.path.join(output_folder, "3_blurred.png"), blurred)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        if save_steps:
            cv2.imwrite(os.path.join(output_folder, "4_edges_canny.png"), edges)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=40, maxLineGap=15)
        
        # Save all Hough lines
        if save_steps and lines is not None:
            hough_vis = image.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, "5_hough_lines.png"),
                       cv2.cvtColor(hough_vis, cv2.COLOR_RGB2BGR))
        
        # Classify lines as horizontal or vertical
        h_lines, v_lines = self.classify_lines(lines)
        
        # Save classified lines
        if save_steps:
            colored_lines = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            for x1, y1, x2, y2 in h_lines:
                cv2.line(colored_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for H
            for x1, y1, x2, y2 in v_lines:
                cv2.line(colored_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for V
            cv2.imwrite(os.path.join(output_folder, "6_classified_lines_HV.png"),
                       cv2.cvtColor(colored_lines, cv2.COLOR_RGB2BGR))
        
        # Create binary image from lines
        lines_image = self.create_lines_image(gray.shape, h_lines, v_lines)
        
        if save_steps:
            cv2.imwrite(os.path.join(output_folder, "7_all_detected_lines.png"), lines_image)
        
        # Morphological operations to close gaps
        closed, thickened = self.close_line_gaps_with_intermediate(lines_image)
        
        if save_steps:
            cv2.imwrite(os.path.join(output_folder, "8_lines_closed.png"), closed)
            cv2.imwrite(os.path.join(output_folder, "9_lines_thickened.png"), thickened)
        
        # Find parking regions
        parking_spots, inverted, parking_regions, contours = self.find_parking_regions_with_intermediate(
            thickened, image)
        
        if save_steps:
            cv2.imwrite(os.path.join(output_folder, "10_inverted_regions.png"), inverted)
            cv2.imwrite(os.path.join(output_folder, "11_valid_parking_regions.png"), parking_regions)
            
            # Save contours
            contours_vis = image.copy()
            for contour_data in contours:
                cv2.drawContours(contours_vis, [contour_data['contour']], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, "12_detected_contours.png"),
                       cv2.cvtColor(contours_vis, cv2.COLOR_RGB2BGR))
            
            # Save rectangles filtered by shape
            self.save_rectangle_filtering(image, contours, parking_spots, output_folder)
        
        self.parking_spots = parking_spots
        
        if save_steps:
            self.save_detection_results(image, parking_spots, output_folder)
        
        return parking_spots
    
    def classify_lines(self, lines: Optional[np.ndarray]) -> Tuple[List, List]:
        """Classify lines as horizontal or vertical."""
        h_lines = []
        v_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:
                    h_lines.append((x1, y1, x2, y2))
                elif 70 < angle < 110:
                    v_lines.append((x1, y1, x2, y2))
        
        return h_lines, v_lines
    
    def create_lines_image(self, shape: Tuple[int, int], 
                           h_lines: List, v_lines: List) -> np.ndarray:
        """Create binary image from detected lines."""
        lines_image = np.zeros(shape, dtype=np.uint8)
        
        for x1, y1, x2, y2 in h_lines:
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
        for x1, y1, x2, y2 in v_lines:
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
        
        return lines_image
    
    def close_line_gaps(self, lines_image: np.ndarray) -> np.ndarray:
        """Apply morphological closing to connect line gaps."""
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        closed = cv2.morphologyEx(lines_image, cv2.MORPH_CLOSE, kernel_h)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
        
        kernel_dilate = np.ones((3, 3), np.uint8)
        closed = cv2.dilate(closed, kernel_dilate, iterations=1)
        
        return closed
    
    def close_line_gaps_with_intermediate(self, lines_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply morphological closing and return intermediate results."""
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        closed = cv2.morphologyEx(lines_image, cv2.MORPH_CLOSE, kernel_h)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
        
        kernel_dilate = np.ones((3, 3), np.uint8)
        thickened = cv2.dilate(closed, kernel_dilate, iterations=1)
        
        return closed, thickened
    
    def find_parking_regions(self, closed: np.ndarray, 
                              image: np.ndarray) -> List[Dict]:
        """Find valid parking spot regions from closed line image."""
        inverted = cv2.bitwise_not(closed)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=8)
        
        parking_spots = []
        spot_number = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w_region = stats[i, cv2.CC_STAT_WIDTH]
            h_region = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = h_region / w_region if w_region > 0 else 0
            
            # Filter by size and aspect ratio
            if (DetectionConfig.CONTOUR_MIN_AREA <= area <= DetectionConfig.CONTOUR_MAX_AREA and
                DetectionConfig.MIN_ASPECT_RATIO <= aspect <= DetectionConfig.MAX_ASPECT_RATIO):
                
                # Extract contour
                mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = contours[0]
                    rect_info = self.create_spot_info(contour, spot_number + 1)
                    
                    # Check rectangularity
                    if self.is_rectangular(contour, rect_info['area']):
                        parking_spots.append(rect_info)
                        spot_number += 1
        
        # Remove duplicates
        parking_spots = self.remove_duplicates(parking_spots)
        
        return parking_spots
    
    def find_parking_regions_with_intermediate(self, closed: np.ndarray, 
                                               image: np.ndarray) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """Find valid parking spot regions and return intermediate results."""
        inverted = cv2.bitwise_not(closed)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=8)
        
        # Create parking regions image
        parking_regions = np.zeros(inverted.shape, dtype=np.uint8)
        
        parking_spots = []
        all_contours = []
        spot_number = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w_region = stats[i, cv2.CC_STAT_WIDTH]
            h_region = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = h_region / w_region if w_region > 0 else 0
            
            # Filter by size and aspect ratio
            if (DetectionConfig.CONTOUR_MIN_AREA <= area <= DetectionConfig.CONTOUR_MAX_AREA and
                DetectionConfig.MIN_ASPECT_RATIO <= aspect <= DetectionConfig.MAX_ASPECT_RATIO):
                
                # Mark valid regions
                parking_regions[labels == i] = 255
                
                # Extract contour
                mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = contours[0]
                    rect_info = self.create_spot_info(contour, spot_number + 1)
                    
                    # Store contour info
                    contour_data = {
                        'contour': contour,
                        'rect_info': rect_info,
                        'rectangularity': 0.0
                    }
                    
                    # Check rectangularity
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    rect_area = cv2.contourArea(box)
                    
                    if rect_area > 0:
                        rectangularity = rect_info['area'] / rect_area
                        contour_data['rectangularity'] = rectangularity
                        
                        if rectangularity >= DetectionConfig.MIN_RECTANGULARITY:
                            parking_spots.append(rect_info)
                            spot_number += 1
                    
                    all_contours.append(contour_data)
        
        # Remove duplicates
        parking_spots = self.remove_duplicates(parking_spots)
        
        return parking_spots, inverted, parking_regions, all_contours
    
    def is_rectangular(self, contour: np.ndarray, area: float) -> bool:
        """Check if contour is sufficiently rectangular."""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        rect_area = cv2.contourArea(box)
        
        if rect_area > 0:
            rectangularity = area / rect_area
            return rectangularity >= DetectionConfig.MIN_RECTANGULARITY
        return False
    
    def create_spot_info(self, contour: np.ndarray, spot_id: int) -> Dict:
        """Create parking spot information dictionary."""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(max(w, h)) / min(w, h) if min(w, h) > 0 else 0
        
        min_rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(min_rect)
        box_points = np.int64(box_points)
        
        return {
            'spot_id': spot_id,
            'bounding_rect': {'x': x, 'y': y, 'width': w, 'height': h},
            'min_area_rect': {
                'center': min_rect[0],
                'size': min_rect[1],
                'angle': min_rect[2],
                'corner_points': box_points.tolist()
            },
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': 0.0,
            'is_rectangular': True,
            'contour': contour,
            'is_occupied': False,
            'occupancy_confidence': 0.0
        }
    
    def remove_duplicates(self, parking_spots: List[Dict]) -> List[Dict]:
        """Remove duplicate parking spots based on center distance."""
        if not parking_spots:
            return parking_spots
        
        filtered = []
        threshold = DetectionConfig.DUPLICATE_DISTANCE_THRESHOLD
        
        for spot in parking_spots:
            center = np.array(spot['min_area_rect']['center'])
            is_duplicate = False
            
            for existing in filtered:
                existing_center = np.array(existing['min_area_rect']['center'])
                dist = np.linalg.norm(center - existing_center)
                if dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(spot)
        
        return filtered
    
    def save_rectangle_filtering(self, image: np.ndarray, 
                                  all_contours: List[Dict],
                                  accepted_spots: List[Dict],
                                  output_folder: str):
        """Save visualization of rectangle filtering (accepted vs rejected)."""
        rect_vis = image.copy()
        
        accepted_ids = {spot['spot_id'] for spot in accepted_spots}
        
        for contour_data in all_contours:
            rect_info = contour_data['rect_info']
            rectangularity = contour_data['rectangularity']
            
            # Get rectangle corners
            box = np.array(rect_info['min_area_rect']['corner_points'], dtype=np.int32)
            
            # Check if this spot was accepted
            if rect_info['spot_id'] in accepted_ids:
                # Accepted - draw in cyan/yellow
                cv2.polylines(rect_vis, [box], True, (0, 255, 255), 2)
            else:
                # Rejected - draw in red with thinner line
                cv2.polylines(rect_vis, [box], True, (255, 0, 0), 1)
        
        # Save rectangles filtered by shape
        filepath = os.path.join(output_folder, "13_rectangles_filtered_by_shape.png")
        cv2.imwrite(filepath, cv2.cvtColor(rect_vis, cv2.COLOR_RGB2BGR))
        
        # Save parking spaces final (numbered accepted spots)
        final_vis = image.copy()
        for i, spot in enumerate(accepted_spots):
            box = np.array(spot['min_area_rect']['corner_points'], dtype=np.int32)
            cv2.polylines(final_vis, [box], True, (0, 255, 0), 2)
            
            center = np.mean(box, axis=0).astype(int)
            cv2.putText(final_vis, str(i), tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        filepath = os.path.join(output_folder, "14_parking_spaces_final.png")
        cv2.imwrite(filepath, cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))
    
    def save_detection_results(self, image: np.ndarray, 
                                parking_spots: List[Dict], 
                                output_folder: str):
        """Save detection results with annotations."""
        result_image = image.copy()
        
        for spot in parking_spots:
            x = spot['bounding_rect']['x']
            y = spot['bounding_rect']['y']
            w = spot['bounding_rect']['width']
            h = spot['bounding_rect']['height']
            
            color = (0, 255, 0)  # Green for empty spots
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            box = np.array(spot['min_area_rect']['corner_points'], dtype=np.int32)
            cv2.polylines(result_image, [box], True, color, 2)
            
            cx, cy = map(int, spot['min_area_rect']['center'])
            cv2.putText(result_image, f"P{spot['spot_id']}", (cx-15, cy-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        filepath = os.path.join(output_folder, "15_detected_parking_spots.png")
        cv2.imwrite(filepath, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


class ObstacleDetector:
    """Detects obstacles using YOLO object detection."""
    
    def __init__(self):
        self.model = None
        self.detected_objects = []
    
    def load_model(self, model_path: str = None):
        """Load YOLO model."""
        try:
            
            if model_path is None:
                model_path = DetectionConfig.YOLO_MODEL_PATH
            
            if not os.path.exists(model_path):
                model_path = DetectionConfig.YOLO_FALLBACK_MODEL
            
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    def detect(self, image: np.ndarray, confidence: float = None, 
               save_result: bool = True) -> List[Dict]:
        """
        Detect obstacles in image using YOLO.
        
        Args:
            image: Input image in BGR format
            confidence: Confidence threshold for detections
            save_result: Whether to save detection visualization
            
        Returns:
            List of detected object dictionaries
        """
        if self.model is None:
            return []
        
        if confidence is None:
            confidence = DetectionConfig.DEFAULT_CONFIDENCE
        
        results = self.model(image, conf=confidence, verbose=False)
        
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                detected_objects.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': class_id,
                    'class_name': class_name,
                    'is_obstacle': class_id in DetectionConfig.OBSTACLE_CLASSES
                })
        
        self.detected_objects = detected_objects
        
        if save_result:
            self.save_yolo_detection(image, detected_objects)
        
        return detected_objects
    
    def save_yolo_detection(self, image: np.ndarray, detected_objects: List[Dict]):
        """Save YOLO detection visualization."""
        output_folder = OutputConfig.PROCESS_IMAGES_FOLDER
        os.makedirs(output_folder, exist_ok=True)
        
        # Convert BGR to RGB for visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yolo_vis = image_rgb.copy()
        
        obstacle_count = 0
        for obj in detected_objects:
            if obj['is_obstacle']:
                obstacle_count += 1
                x1, y1, x2, y2 = obj['bbox']
                
                # Red fill with transparency
                overlay = yolo_vis.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, yolo_vis, 0.7, 0, yolo_vis)
                
                # Red border
                cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # Label
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(yolo_vis,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            (255, 0, 0), -1)
                cv2.putText(yolo_vis, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save YOLO detection image
        filepath = os.path.join(output_folder, OutputConfig.YOLO_DETECTION_FILENAME)
        cv2.imwrite(filepath, cv2.cvtColor(yolo_vis, cv2.COLOR_RGB2BGR))
    
    def save_combined_detection(self, image: np.ndarray, 
                               parking_spots: List[Dict],
                               detected_objects: List[Dict]):
        """Save combined parking spots and obstacles visualization."""
        output_folder = OutputConfig.PROCESS_IMAGES_FOLDER
        os.makedirs(output_folder, exist_ok=True)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_image = image_rgb.copy()
        
        # Draw parking spots (green)
        for spot in parking_spots:
            x = spot['bounding_rect']['x']
            y = spot['bounding_rect']['y']
            w = spot['bounding_rect']['width']
            h = spot['bounding_rect']['height']
            
            color = (0, 255, 0)  # Green for empty
            
            # Fill with transparency
            overlay = result_image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)
            
            # Border
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Label
            cx, cy = map(int, spot['min_area_rect']['center'])
            cv2.putText(result_image, f"P{spot['spot_id']}", (cx-10, cy-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Draw obstacles (red)
        for obj in detected_objects:
            if obj['is_obstacle']:
                x1, y1, x2, y2 = obj['bbox']
                color = (255, 0, 0)
                
                # Fill with transparency
                overlay = result_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)
                
                # Border
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                
                # Label
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result_image,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                cv2.putText(result_image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add legend
        cv2.rectangle(result_image, (5, 5), (220, 100), (255, 255, 255), -1)
        cv2.rectangle(result_image, (5, 5), (220, 100), (0, 0, 0), 1)
        cv2.rectangle(result_image, (10, 15), (25, 30), (0, 255, 0), -1)
        cv2.putText(result_image, "Empty Parking Spot", (30, 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.rectangle(result_image, (10, 40), (25, 55), (255, 0, 0), -1)
        cv2.putText(result_image, "Occupied Spot", (30, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.rectangle(result_image, (10, 65), (25, 80), (255, 0, 0), -1)
        cv2.putText(result_image, "Obstacle (YOLO)", (30, 77),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Save combined image
        filepath = os.path.join(output_folder, OutputConfig.FINAL_COMBINED_FILENAME)
        cv2.imwrite(filepath, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


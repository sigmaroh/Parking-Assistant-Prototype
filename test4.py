import cv2
import numpy as np
import sys
import os
from dataclasses import dataclass

# Create output folder for step images
OUTPUT_FOLDER = 'detection_steps'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: '{OUTPUT_FOLDER}'")

def save_step(image, step_num, name, is_grayscale=False):
    """Save an image step to the output folder"""
    filename = f"{step_num:02d}_{name}.png"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    if is_grayscale and len(image.shape) == 2:
        cv2.imwrite(filepath, image)
    else:
        cv2.imwrite(filepath, image)
    print(f"  Saved: {filepath}")

# Check if image exists
image_path = 'Layout 1.png'
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
    print("Please make sure the image exists in the current directory.")
    sys.exit(1)

# Load image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image '{image_path}'")
    print("Please check if it's a valid image file.")
    sys.exit(1)

print(f"Image loaded successfully. Size: {image.shape[1]}x{image.shape[0]}")

# ===================================================================
# OPTION 1: Using ONLY Computer Vision Approach
# ===================================================================
def option_computer_vision_only():
    print("\n" + "="*60)
    print("OPTION 1: COMPUTER VISION ONLY")
    print("="*60)
    print(f"\nSaving processing steps to '{OUTPUT_FOLDER}/' folder...")
    
    # Import here to avoid issues if not available
    try:
        # Create a simplified version of the detector
        detector = create_simple_cv_detector()
        
        # Detect parking spaces
        print("\nDetecting parking spaces...")
        spaces = detector.detect_from_image(image)
        
        if spaces:
            print(f"\nDetected {len(spaces)} parking spaces")
            
            # Detect occupancy
            print("\nDetecting occupancy...")
            detector.detect_occupancy(image)
            
            # Detect obstacles (cars, trees, walls)
            print("\nDetecting obstacles (cars, trees, walls)...")
            detector.detect_obstacles(image)
            
            # Generate grid visualization
            print("\nGenerating grid visualization...")
            grid = detector.generate_grid()
            
            # Generate A* grid matrix
            print("\nGenerating A* grid matrix...")
            astar_grid, grid_rows = detector.generate_astar_grid()
            
            # Overlay grid on original image
            print("\nOverlaying grid on original image...")
            grid_overlay = detector.overlay_grid_on_image(image, astar_grid, grid_rows)
            
            # Visualize results
            print("\nVisualizing results...")
            result = detector.visualize(image)
            
            # Show results
            cv2.imshow('Computer Vision - Parking Detection', result)
            print("\nPress any key in the window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save result
            cv2.imwrite('cv_detection_result.jpg', result)
            print("Result saved as 'cv_detection_result.jpg'")
            print(f"\nAll step images saved in '{OUTPUT_FOLDER}/' folder")
        else:
            print("No parking spaces detected.")
            
    except Exception as e:
        print(f"Error in computer vision approach: {e}")
        import traceback
        traceback.print_exc()

# ===================================================================
# OPTION 2: Using Hybrid Approach (with fallback to CV if YOLO fails)
# ===================================================================
def option_hybrid_approach():
    print("\n" + "="*60)
    print("OPTION 2: HYBRID APPROACH (CV with Vehicle Detection)")
    print("="*60)
    
    try:
        # Create hybrid detector
        print("Creating hybrid detector...")
        detector = create_hybrid_detector()
        
        # Detect parking spaces using hybrid method
        print("Detecting parking spaces (hybrid method)...")
        spaces = detector.detect(image, method='hybrid')
        
        if spaces:
            print(f"Detected {len(spaces)} parking spaces")
            
            # For hybrid detector, we need to visualize using cv_detector
            print("Visualizing results...")
            result = image.copy()
            
            # Draw parking spaces
            for i, space in enumerate(spaces):
                # Determine color based on occupancy
                if space.is_occupied:
                    color = (0, 0, 255)  # Red for occupied
                    status = "Occupied"
                else:
                    color = (0, 255, 0)  # Green for empty
                    status = "Empty"
                
                # Draw the parking space polygon
                pts = space.vertices.astype(np.int32)
                cv2.polylines(result, [pts], True, color, 2)
                
                # Add ID and status at the center
                center = tuple(map(int, space.center))
                cv2.putText(result, f'{i}:{status}', center,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw a small circle at the center
                cv2.circle(result, center, 3, color, -1)
            
            # Add statistics
            total_spaces = len(spaces)
            occupied_spaces = sum(1 for space in spaces if space.is_occupied)
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
            
            # Show results
            cv2.imshow('Hybrid - Parking Detection', result)
            print("\nPress any key in the window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save result
            cv2.imwrite('hybrid_detection_result.jpg', result)
            print("Result saved as 'hybrid_detection_result.jpg'")
        else:
            print("No parking spaces detected.")
            
    except Exception as e:
        print(f"Error in hybrid approach: {e}")
        print("Falling back to computer vision only...")
        option_computer_vision_only()

# ===================================================================
# SIMPLIFIED DETECTOR CLASSES (to avoid import errors)
# ===================================================================

@dataclass
class ParkingSpace:
    """Represents a detected parking space"""
    id: int
    vertices: np.ndarray
    center: tuple
    area: float
    is_occupied: bool = False
    confidence: float = 0.0

@dataclass 
class DetectionConfig:
    """Configuration for parking space detection - Optimized for Layout 1.png"""
    # Keep original resolution for this clean image
    resize_width: int = 0  # 0 = no resize
    resize_height: int = 0
    # Threshold for parking lines (cream/beige lines on gray pavement)
    line_threshold: int = 165  # Adjusted for cream-colored lines
    # Canny edge detection thresholds  
    canny_low: int = 50
    canny_high: int = 150
    # Area range for parking spaces (very lenient to catch all spaces)
    contour_min_area: int = 5000   # Even lower to catch smaller spaces
    contour_max_area: int = 35000  # Higher for larger spaces
    # Aspect ratio (height/width) - spaces are WIDER than tall in bird's eye view
    min_aspect_ratio: float = 0.1   # Very lenient for wider spaces
    max_aspect_ratio: float = 0.8   # Very lenient for taller spaces
    # Rectangularity threshold (contour_area / bounding_rect_area)
    min_rectangularity: float = 0.75  # More lenient for near-rectangles
    # Occupancy threshold (combined score)
    # Empty spaces typically score 0.002-0.009, so threshold set well above
    occupancy_threshold: float = 0.10  # Conservative threshold to avoid false positives
    # Only output empty spaces
    only_empty_spaces: bool = True

def create_simple_cv_detector():
    """Create a simplified computer vision detector"""
    
    class SimpleCVDetector:
        def __init__(self):
            self.config = DetectionConfig()
            self.parking_spaces = []
            self.obstacles = []  # Store detected obstacles: {'type': 'car/tree/wall', 'bbox': (x,y,w,h), 'mask': array}
            
        def preprocess_image(self, image, save_steps=True):
            """Preprocess image"""
            # Save original only
            if save_steps:
                save_step(image, 1, "original")
            
            # Store original for later use
            self._processed_image = image.copy()
            
            # Resize if needed
            if self.config.resize_width > 0 and self.config.resize_height > 0:
                h, w = image.shape[:2]
                if w > self.config.resize_width or h > self.config.resize_height:
                    self._processed_image = cv2.resize(image, (self.config.resize_width, self.config.resize_height))
                    if save_steps:
                        save_step(self._processed_image, 2, "resized")
            
            # Convert to grayscale
            if len(self._processed_image.shape) == 3:
                gray = cv2.cvtColor(self._processed_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self._processed_image.copy()
            
            if save_steps:
                save_step(gray, 3, "grayscale", is_grayscale=True)
            
            # Apply slight blur to reduce noise while preserving edges
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            if save_steps:
                save_step(blurred, 4, "blurred", is_grayscale=True)
            
            return blurred
        
        def detect_edges(self, gray, save_steps=True):
            """Detect parking spaces using Hough line-based grid detection"""
            
            # Use Canny edge detection
            edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
            if save_steps:
                save_step(edges, 5, "canny_edges", is_grayscale=True)
            
            # Step 2: Use Hough Line Transform to detect line segments
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                    minLineLength=40, maxLineGap=15)
            
            # Step 3: Separate horizontal and vertical lines
            h_lines = []  # Horizontal lines
            v_lines = []  # Vertical lines
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    
                    # Classify line orientation
                    if angle < 20 or angle > 160:  # Horizontal (±20 degrees)
                        h_lines.append((x1, y1, x2, y2))
                    elif 70 < angle < 110:  # Vertical (90±20 degrees)
                        v_lines.append((x1, y1, x2, y2))
            
            if save_steps:
                print(f"    Detected {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
            
            # Step 4: Draw classified lines
            lines_image = np.zeros_like(gray)
            for x1, y1, x2, y2 in h_lines:
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
            for x1, y1, x2, y2 in v_lines:
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
            
            if save_steps:
                # Show H and V lines in different colors
                colored_lines = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
                for x1, y1, x2, y2 in h_lines:
                    cv2.line(colored_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for H
                for x1, y1, x2, y2 in v_lines:
                    cv2.line(colored_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for V
                save_step(colored_lines, 6, "classified_lines_HV")
                save_step(lines_image, 7, "all_detected_lines", is_grayscale=True)
            
            # Step 5: Close gaps ONLY in respective directions (less aggressive)
            # Close horizontal gaps in horizontal lines
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            # Close vertical gaps in vertical lines  
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            
            closed = cv2.morphologyEx(lines_image, cv2.MORPH_CLOSE, kernel_h)
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
            if save_steps:
                save_step(closed, 8, "lines_closed", is_grayscale=True)
            
            # Dilate slightly to ensure connectivity at intersections
            kernel_dilate = np.ones((3, 3), np.uint8)
            thickened = cv2.dilate(closed, kernel_dilate, iterations=1)
            if save_steps:
                save_step(thickened, 9, "lines_thickened", is_grayscale=True)
            
            # Invert to get parking space regions
            inverted = cv2.bitwise_not(thickened)
            if save_steps:
                save_step(inverted, 10, "inverted_regions", is_grayscale=True)
            
            # Step 8: Use connected components to find separate regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
            
            parking_regions = np.zeros_like(gray)
            valid_regions = []
            region_stats = []
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w_region = stats[i, cv2.CC_STAT_WIDTH]
                h_region = stats[i, cv2.CC_STAT_HEIGHT]
                aspect = h_region / w_region if w_region > 0 else 0
                
                if area > 1000:
                    region_stats.append((i, area, w_region, h_region, aspect))
                
                # Check if this could be a parking space
                if self.config.contour_min_area <= area <= self.config.contour_max_area:
                    if self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio:
                        valid_regions.append(i)
                        parking_regions[labels == i] = 255
            
            # if save_steps:
            #     # Show all regions with different colors
            #     colored_labels = np.zeros((inverted.shape[0], inverted.shape[1], 3), dtype=np.uint8)
            #     for i in range(1, num_labels):
            #         mask = labels == i
            #         color = ((i * 50) % 255, (i * 80) % 255, (i * 110) % 255)
            #         colored_labels[mask] = color
            #     save_step(colored_labels, 11, "all_regions_colored")
            save_step(parking_regions, 11, "valid_parking_regions", is_grayscale=True)
            print(f"    Found {num_labels - 1} total regions, {len(valid_regions)} valid parking spaces")
            
            return parking_regions
        
        def find_rectangles(self, parking_regions, original_image=None, save_steps=True):
            """Find parking space rectangles from the pre-filtered regions"""
            # Find contours in the already-filtered parking regions
            contours, _ = cv2.findContours(parking_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = []
            rejected_rectangularity = []
            
            # Save contours visualization
            if save_steps and original_image is not None:
                contours_vis = original_image.copy()
                cv2.drawContours(contours_vis, contours, -1, (0, 255, 0), 2)
                save_step(contours_vis, 13, "detected_contours")
                print(f"    Found {len(contours)} contours from valid regions")
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1500:  # Skip very small contours
                    continue
                
                # Get minimum area rectangle for better fit
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                ordered = self.order_points(box)
                
                # Calculate rectangularity (how well contour fits a rectangle)
                rect_area = cv2.contourArea(ordered)
                if rect_area > 0:
                    rectangularity = area / rect_area
                else:
                    rectangularity = 0
                
                # Only keep nearly perfect rectangles
                if rectangularity >= self.config.min_rectangularity:
                    rectangles.append(ordered)
                else:
                    rejected_rectangularity.append((ordered, rectangularity))
            
            if save_steps:
                print(f"    Rectangularity filter: kept {len(rectangles)}, rejected {len(rejected_rectangularity)}")
                if rejected_rectangularity:
                    print(f"    Rejected rectangularities: {[f'{r:.2f}' for _, r in rejected_rectangularity[:5]]}")
            
            # Remove duplicate rectangles based on center proximity
            filtered = self._remove_duplicate_rectangles(rectangles, distance_threshold=50)
            
            # Save detected rectangles
            if save_steps and original_image is not None:
                rect_vis = original_image.copy()
                for rect in rectangles:
                    pts = rect.astype(np.int32)
                    cv2.polylines(rect_vis, [pts], True, (0, 255, 255), 2)
                # Also show rejected in red
                for rect, _ in rejected_rectangularity:
                    pts = rect.astype(np.int32)
                    cv2.polylines(rect_vis, [pts], True, (0, 0, 255), 1)
                save_step(rect_vis, 14, "rectangles_filtered_by_shape")
                print(f"    Rectangles after shape filter: {len(rectangles)}")
                
                # Save after duplicate removal
                rect_final_vis = original_image.copy()
                for i, rect in enumerate(filtered):
                    pts = rect.astype(np.int32)
                    cv2.polylines(rect_final_vis, [pts], True, (0, 255, 0), 2)
                    center = np.mean(rect, axis=0).astype(int)
                    cv2.putText(rect_final_vis, str(i), tuple(center), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                save_step(rect_final_vis, 15, "parking_spaces_final")
                print(f"    Final parking spaces: {len(filtered)}")
            
            return filtered
        
        def _remove_duplicate_rectangles(self, rectangles, distance_threshold=50):
            """Remove duplicate rectangles based on center distance"""
            if len(rectangles) == 0:
                return rectangles
            
            filtered = []
            for rect in rectangles:
                center = np.mean(rect, axis=0)
                is_duplicate = False
                
                for existing in filtered:
                    existing_center = np.mean(existing, axis=0)
                    dist = np.linalg.norm(center - existing_center)
                    if dist < distance_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(rect)
            
            return filtered
        
        def order_points(self, pts):
            """Order points in clockwise order"""
            # Sort by x-coordinate
            x_sorted = pts[np.argsort(pts[:, 0]), :]
            
            # Get left-most and right-most points
            left_most = x_sorted[:2, :]
            right_most = x_sorted[2:, :]
            
            # Sort left-most by y (top-left, bottom-left)
            left_most = left_most[np.argsort(left_most[:, 1]), :]
            tl, bl = left_most[0], left_most[1]
            
            # Sort right-most by y (top-right, bottom-right)
            right_most = right_most[np.argsort(right_most[:, 1]), :]
            tr, br = right_most[0], right_most[1]
            
            return np.array([tl, tr, br, bl], dtype=np.float32)
        
        def detect_from_image(self, image, save_steps=True):
            """Main detection function"""
            self.parking_spaces = []
            
            print("\n  Processing steps:")
            
            # Preprocess (this also stores _processed_image)
            gray = self.preprocess_image(image, save_steps=save_steps)
            
            # Detect edges
            edges = self.detect_edges(gray, save_steps=save_steps)
            
            # Find rectangles (use the processed image for correct sizing)
            rectangles = self.find_rectangles(edges, original_image=self._processed_image, save_steps=save_steps)
            
            # Create ParkingSpace objects
            for i, rect in enumerate(rectangles):
                # Calculate center
                center = np.mean(rect, axis=0)
                
                # Create space
                space = ParkingSpace(
                    id=i,
                    vertices=rect,
                    center=(float(center[0]), float(center[1])),
                    area=cv2.contourArea(rect),
                    is_occupied=False
                )
                self.parking_spaces.append(space)
            
            return self.parking_spaces
        
        def detect_occupancy(self, image, save_steps=True):
            """Improved occupancy detection using multiple features"""
            gray = self.preprocess_image(image, save_steps=False)
            
            for space in self.parking_spaces:
                # Create mask for this parking space
                mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                pts = space.vertices.astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
                
                # Get the ROI
                x, y, w, h = cv2.boundingRect(pts)
                roi_gray = gray[y:y+h, x:x+w]
                roi_mask = mask[y:y+h, x:x+w]
                
                if roi_gray.size == 0:
                    space.is_occupied = False
                    space.confidence = 0
                    continue
                
                mask_pixels = np.sum(roi_mask > 0)
                if mask_pixels == 0:
                    space.is_occupied = False
                    space.confidence = 0
                    continue
                
                # Method 1: Edge density (cars have more edges, but ignore weak edges from lines)
                # Use higher thresholds to ignore parking line edges
                edges = cv2.Canny(roi_gray, 80, 200)
                edges_masked = cv2.bitwise_and(edges, edges, mask=roi_mask)
                edge_density = np.sum(edges_masked > 0) / (mask_pixels + 1e-5)
                
                # Method 2: Variance (cars have more texture variation)
                roi_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)
                masked_pixels = roi_masked[roi_mask > 0]
                if len(masked_pixels) > 0:
                    variance = np.var(masked_pixels)
                    variance_normalized = variance / 2000  # Normalize (reduced sensitivity)
                else:
                    variance_normalized = 0
                
                # Method 3: Check for bright pixels (white car roofs) - more conservative
                bright_threshold = 220  # Higher threshold for bright pixels
                bright_pixels = np.sum((roi_gray > bright_threshold) & (roi_mask > 0))
                bright_ratio = bright_pixels / (mask_pixels + 1e-5)
                
                # Method 4: Check for dark pixels (car shadows/body)
                dark_threshold = 50
                dark_pixels = np.sum((roi_gray < dark_threshold) & (roi_mask > 0))
                dark_ratio = dark_pixels / (mask_pixels + 1e-5)
                
                # Combined score - adjusted weights, less sensitive to edges
                combined_score = (edge_density * 0.3 + variance_normalized * 0.25 + 
                                 bright_ratio * 0.2 + dark_ratio * 0.25)
                
                # Determine occupancy
                space.is_occupied = combined_score > self.config.occupancy_threshold
                space.confidence = combined_score
            
            # Save occupancy analysis
            if save_steps:
                occupancy_vis = image.copy()
                print("    Occupancy scores:")
                for space in self.parking_spaces:
                    status = "OCCUPIED" if space.is_occupied else "empty"
                    print(f"      Space {space.id}: score={space.confidence:.4f} -> {status}")
                    
                    color = (0, 0, 255) if space.is_occupied else (0, 255, 0)
                    pts = space.vertices.astype(np.int32)
                    cv2.polylines(occupancy_vis, [pts], True, color, 2)
                    center = tuple(map(int, space.center))
                    cv2.putText(occupancy_vis, f'{space.confidence:.3f}', 
                               (center[0] - 25, center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                save_step(occupancy_vis, 16, "occupancy_analysis")
        
        def detect_obstacles(self, image, save_steps=True):
            """Detect obstacles: cars, trees, walls in the image"""
            self.obstacles = []
            
            # Convert to different color spaces for detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Detect TREES (green vegetation)
            # Green color range in HSV
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Remove small noise
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find tree contours
            tree_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in tree_contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum tree size
                    x, y, w, h = cv2.boundingRect(contour)
                    self.obstacles.append({
                        'type': 'tree',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'mask': green_mask[y:y+h, x:x+w]
                    })
            
            # 2. Detect CARS (both in and outside parking spaces)
            # Method 1: Detect cars in occupied parking spaces
            for space in self.parking_spaces:
                if space.is_occupied:
                    # Get parking space bounds
                    pts = space.vertices.astype(np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    
                    # Extract ROI for car detection
                    roi = image[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue
                    
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                    
                    # Create mask for parking space
                    space_mask = np.zeros(roi_gray.shape[:2], dtype=np.uint8)
                    adjusted_pts = pts - np.array([x, y])
                    cv2.fillPoly(space_mask, [adjusted_pts], 255)
                    
                    # Detect car using multiple methods
                    # Method A: Edge-based detection
                    edges_roi = cv2.Canny(roi_gray, 50, 150)
                    edges_masked = cv2.bitwise_and(edges_roi, edges_roi, mask=space_mask)
                    
                    # Method B: Dark object detection (cars are darker than pavement)
                    # Use adaptive threshold for better car detection
                    _, dark_mask = cv2.threshold(roi_gray, 130, 255, cv2.THRESH_BINARY_INV)
                    dark_mask = cv2.bitwise_and(dark_mask, space_mask)
                    
                    # Method C: Color-based detection (cars have different colors than pavement)
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    # Detect non-gray objects (cars are colored, pavement is gray)
                    saturation = roi_hsv[:, :, 1]
                    _, color_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
                    color_mask = cv2.bitwise_and(color_mask, space_mask)
                    
                    # Combine all methods
                    combined_mask = cv2.bitwise_or(edges_masked, dark_mask)
                    combined_mask = cv2.bitwise_or(combined_mask, color_mask)
                    
                    # Clean up the mask
                    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    
                    # Find car contours
                    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Sort contours by area (largest first)
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)
                        
                        # Take the largest contour (likely the car)
                        largest_contour = contours[0]
                        area = cv2.contourArea(largest_contour)
                        
                        if area > 500:  # Minimum car size
                            cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                            # Adjust coordinates to image space
                            car_x = x + cx
                            car_y = y + cy
                            
                            # Verify it's car-like (rectangular, reasonable size)
                            aspect_ratio = cw / ch if ch > 0 else 0
                            if 0.3 < aspect_ratio < 4.0:  # More lenient aspect ratio
                                self.obstacles.append({
                                    'type': 'car',
                                    'bbox': (car_x, car_y, cw, ch),
                                    'area': area,
                                    'mask': None
                                })
                            else:
                                # If aspect ratio doesn't match, use parking space bounds
                                self.obstacles.append({
                                    'type': 'car',
                                    'bbox': (x, y, w, h),
                                    'area': w * h,
                                    'mask': None
                                })
                        else:
                            # If area too small, use parking space bounds
                            self.obstacles.append({
                                'type': 'car',
                                'bbox': (x, y, w, h),
                                'area': w * h,
                                'mask': None
                            })
                    else:
                        # If no contour found, use the entire parking space as car location
                        # (since we know it's occupied)
                        self.obstacles.append({
                            'type': 'car',
                            'bbox': (x, y, w, h),
                            'area': w * h,
                            'mask': None
                        })
            
            # Method 2: Detect cars outside parking spaces using improved detection
            # Use color-based detection for cars (cars are typically darker than pavement)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for non-parking areas
            parking_mask = np.ones(gray.shape[:2], dtype=np.uint8) * 255
            for space in self.parking_spaces:
                pts = space.vertices.astype(np.int32)
                cv2.fillPoly(parking_mask, [pts], 0)  # Exclude parking spaces
            
            # Detect dark objects (cars) in non-parking areas
            # Cars are typically darker than the pavement
            _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            dark_mask = cv2.bitwise_and(dark_mask, parking_mask)
            
            # Remove small noise
            kernel = np.ones((5, 5), np.uint8)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find car contours
            car_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in car_contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Car-like characteristics: rectangular, medium to large size
                if 2000 < area < 100000 and 0.5 < aspect_ratio < 5.0:  # More lenient
                    # Additional check: cars should have some texture/edges
                    roi_gray = gray[y:y+h, x:x+w]
                    edges_roi = cv2.Canny(roi_gray, 50, 150)
                    edge_density = np.sum(edges_roi > 0) / (roi_gray.size + 1e-5)
                    
                    if edge_density > 0.02:  # Lower threshold for texture
                        # Check if this car is not already detected in a parking space
                        already_detected = False
                        for existing_car in self.obstacles:
                            if existing_car['type'] == 'car':
                                ex, ey, ew, eh = existing_car['bbox']
                                # Check overlap
                                if not (x + w < ex or x > ex + ew or y + h < ey or y > ey + eh):
                                    already_detected = True
                                    break
                        
                        if not already_detected:
                            self.obstacles.append({
                                'type': 'car',
                                'bbox': (x, y, w, h),
                                'area': area,
                                'mask': None
                            })
            
            # 3. Detect WALLS (vertical/horizontal structures)
            # Detect long horizontal and vertical lines
            edges_full = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges_full, rho=1, theta=np.pi/180, threshold=100,
                                    minLineLength=200, maxLineGap=20)
            
            wall_segments = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    
                    # Detect walls: long lines that are nearly horizontal or vertical
                    if length > 300:  # Long lines
                        if angle < 10 or angle > 170:  # Horizontal wall
                            wall_segments.append(('horizontal', (x1, y1, x2, y2)))
                        elif 80 < angle < 100:  # Vertical wall
                            wall_segments.append(('vertical', (x1, y1, x2, y2)))
            
            # Group wall segments into wall regions
            if wall_segments:
                # Create wall mask
                wall_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                for wall_type, (x1, y1, x2, y2) in wall_segments:
                    cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 10)  # Thick lines
                
                # Dilate to create wall regions
                wall_mask = cv2.dilate(wall_mask, np.ones((20, 20), np.uint8), iterations=2)
                
                # Find wall contours
                wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in wall_contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum wall size
                        x, y, w, h = cv2.boundingRect(contour)
                        self.obstacles.append({
                            'type': 'wall',
                            'bbox': (x, y, w, h),
                            'area': area,
                            'mask': wall_mask[y:y+h, x:x+w]
                        })
            
            if save_steps:
                # Visualize obstacles
                obstacle_vis = image.copy()
                for obstacle in self.obstacles:
                    x, y, w, h = obstacle['bbox']
                    obs_type = obstacle['type']
                    
                    # Color by type
                    if obs_type == 'tree':
                        color = (0, 255, 0)  # Green
                    elif obs_type == 'car':
                        color = (255, 0, 0)  # Blue
                    elif obs_type == 'wall':
                        color = (0, 0, 255)  # Red
                    else:
                        color = (128, 128, 128)  # Gray
                    
                    cv2.rectangle(obstacle_vis, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(obstacle_vis, obs_type, (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                save_step(obstacle_vis, 20, "detected_obstacles")
                print(f"    Detected obstacles: {len([o for o in self.obstacles if o['type']=='tree'])} trees, "
                      f"{len([o for o in self.obstacles if o['type']=='car'])} cars, "
                      f"{len([o for o in self.obstacles if o['type']=='wall'])} walls")
        
        def visualize(self, image, save_steps=True):
            """Visualize detected spaces - only empty spaces if configured"""
            result = image.copy()
            
            # Filter to only empty spaces if configured
            spaces_to_show = [s for s in self.parking_spaces 
                             if not (self.config.only_empty_spaces and s.is_occupied)]
            
            # Use single color (white) for all parking spaces
            line_color = (255, 255, 255)  # White
            text_color = (255, 255, 255)  # White
            
            for space in spaces_to_show:
                # Draw polygon
                pts = space.vertices.astype(np.int32)
                cv2.polylines(result, [pts], True, line_color, 2)
                
                # Draw center
                center = tuple(map(int, space.center))
                cv2.circle(result, center, 3, line_color, -1)
                
                # Add text - just ID
                cv2.putText(result, f'{space.id}', 
                           (center[0] - 10, center[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Add statistics
            total = len(self.parking_spaces)
            occupied = sum(1 for s in self.parking_spaces if s.is_occupied)
            empty = total - occupied
            
            # White text with black outline for visibility
            cv2.putText(result, f'Empty Spaces: {empty}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(result, f'Empty Spaces: {empty}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            
            if not self.config.only_empty_spaces:
                cv2.putText(result, f'Occupied: {occupied}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                cv2.putText(result, f'Occupied: {occupied}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            
            # Save final result
            if save_steps:
                save_step(result, 17, "final_result_empty_spaces")
            
            return result
        
        def generate_grid(self, save_steps=True):
            """Generate a grid visualization: empty spaces (green), occupied (black), numbered"""
            if not self.parking_spaces:
                return None
            
            # Sort spaces by position (top to bottom, left to right)
            sorted_spaces = sorted(self.parking_spaces, key=lambda s: (s.center[1], s.center[0]))
            
            # Calculate grid dimensions
            # Group spaces into rows based on y-coordinate
            rows = []
            current_row = [sorted_spaces[0]]
            y_threshold = 50  # Pixels - spaces within this distance are in same row
            
            for space in sorted_spaces[1:]:
                if abs(space.center[1] - current_row[0].center[1]) < y_threshold:
                    current_row.append(space)
                else:
                    # Sort current row by x-coordinate
                    current_row.sort(key=lambda s: s.center[0])
                    rows.append(current_row)
                    current_row = [space]
            
            # Don't forget the last row
            if current_row:
                current_row.sort(key=lambda s: s.center[0])
                rows.append(current_row)
            
            # Find max columns
            max_cols = max(len(row) for row in rows) if rows else 0
            num_rows = len(rows)
            
            # Create grid image
            cell_size = 100
            grid_width = max_cols * cell_size
            grid_height = num_rows * cell_size
            grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background
            
            # Fill grid cells
            space_number = 1
            for row_idx, row in enumerate(rows):
                for col_idx, space in enumerate(row):
                    y1 = row_idx * cell_size
                    y2 = (row_idx + 1) * cell_size
                    x1 = col_idx * cell_size
                    x2 = (col_idx + 1) * cell_size
                    
                    # Color: green for empty, black for occupied
                    if space.is_occupied:
                        color = (0, 0, 0)  # Black for occupied
                    else:
                        color = (0, 255, 0)  # Green for empty
                    
                    # Fill cell
                    grid[y1:y2, x1:x2] = color
                    
                    # Add border
                    cv2.rectangle(grid, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    
                    # Add number (white text for visibility)
                    text = str(space_number)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = x1 + (cell_size - text_size[0]) // 2
                    text_y = y1 + (cell_size + text_size[1]) // 2
                    
                    # Add text with outline for visibility
                    cv2.putText(grid, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(grid, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    space_number += 1
            
            if save_steps:
                save_step(grid, 18, "parking_grid")
                print(f"    Generated grid: {num_rows} rows x {max_cols} columns")
            
            return grid
        
        def generate_astar_grid(self, save_steps=True):
            """Generate a grid matrix for A* algorithm: 0=free (empty), 1=obstacle (occupied)"""
            if not self.parking_spaces:
                return None
            
            # Sort spaces by position (top to bottom, left to right)
            sorted_spaces = sorted(self.parking_spaces, key=lambda s: (s.center[1], s.center[0]))
            
            # Group spaces into rows based on y-coordinate
            rows = []
            current_row = [sorted_spaces[0]]
            y_threshold = 50  # Pixels - spaces within this distance are in same row
            
            for space in sorted_spaces[1:]:
                if abs(space.center[1] - current_row[0].center[1]) < y_threshold:
                    current_row.append(space)
                else:
                    # Sort current row by x-coordinate
                    current_row.sort(key=lambda s: s.center[0])
                    rows.append(current_row)
                    current_row = [space]
            
            # Don't forget the last row
            if current_row:
                current_row.sort(key=lambda s: s.center[0])
                rows.append(current_row)
            
            # Find max columns
            max_cols = max(len(row) for row in rows) if rows else 0
            num_rows = len(rows)
            
            # Create grid matrix: 0 = free space, 1 = obstacle
            grid_matrix = np.zeros((num_rows, max_cols), dtype=np.int32)
            
            # Fill grid: 0 for empty spaces, 1 for occupied spaces
            for row_idx, row in enumerate(rows):
                for col_idx, space in enumerate(row):
                    if space.is_occupied:
                        grid_matrix[row_idx, col_idx] = 1  # Occupied parking space = obstacle
            
            # Map obstacles (cars, trees, walls) to grid cells
            # Calculate grid cell boundaries from parking space positions
            if rows and len(rows[0]) > 0:
                # Get approximate cell size from first row
                first_space = rows[0][0]
                cell_width = first_space.vertices[:, 0].max() - first_space.vertices[:, 0].min()
                cell_height = first_space.vertices[:, 1].max() - first_space.vertices[:, 1].min()
                
                # Get grid origin (top-left of first parking space)
                grid_origin_x = min(space.vertices[:, 0].min() for row in rows for space in row)
                grid_origin_y = min(space.vertices[:, 1].min() for row in rows for space in row)
                
                # Map obstacles to grid cells
                for obstacle in self.obstacles:
                    x, y, w, h = obstacle['bbox']
                    obs_center_x = x + w // 2
                    obs_center_y = y + h // 2
                    
                    # Calculate which grid cell this obstacle overlaps with
                    # Use center point to determine grid position
                    col = int((obs_center_x - grid_origin_x) / cell_width)
                    row = int((obs_center_y - grid_origin_y) / cell_height)
                    
                    # Mark obstacle cells (within grid bounds)
                    if 0 <= row < num_rows and 0 <= col < max_cols:
                        grid_matrix[row, col] = 1  # Mark as obstacle
                    
                    # Also mark adjacent cells if obstacle is large
                    if w > cell_width * 0.5 or h > cell_height * 0.5:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = row + dr, col + dc
                                if 0 <= nr < num_rows and 0 <= nc < max_cols:
                                    grid_matrix[nr, nc] = 1
            
            # Create visualization of the grid matrix
            if save_steps:
                # Create visual representation
                cell_size = 50
                grid_width = max_cols * cell_size
                grid_height = num_rows * cell_size
                grid_vis = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background
                
                for row_idx in range(num_rows):
                    for col_idx in range(max_cols):
                        y1 = row_idx * cell_size
                        y2 = (row_idx + 1) * cell_size
                        x1 = col_idx * cell_size
                        x2 = (col_idx + 1) * cell_size
                        
                        # Color based on grid value
                        if row_idx < len(rows) and col_idx < len(rows[row_idx]):
                            space = rows[row_idx][col_idx]
                            if space.is_occupied:
                                color = (0, 0, 0)  # Black for obstacle
                                text = "1"
                            else:
                                color = (0, 255, 0)  # Green for free
                                text = "0"
                            
                            # Fill cell
                            grid_vis[y1:y2, x1:x2] = color
                            
                            # Add border
                            cv2.rectangle(grid_vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
                            
                            # Add text
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            text_x = x1 + (cell_size - text_size[0]) // 2
                            text_y = y1 + (cell_size + text_size[1]) // 2
                            
                            text_color = (255, 255, 255) if space.is_occupied else (0, 0, 0)
                            cv2.putText(grid_vis, text, (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                        else:
                            # Empty cell (no parking space)
                            cv2.rectangle(grid_vis, (x1, y1), (x2, y2), (200, 200, 200), 1)
                
                save_step(grid_vis, 19, "astar_grid_matrix")
                
                # Save grid matrix as numpy array
                np.save(os.path.join(OUTPUT_FOLDER, "astar_grid_matrix.npy"), grid_matrix)
                
                # Save grid matrix as text file for easy inspection
                np.savetxt(os.path.join(OUTPUT_FOLDER, "astar_grid_matrix.txt"), grid_matrix, fmt='%d')
                
                print(f"    Generated A* grid matrix: {num_rows} rows x {max_cols} columns")
                print(f"    Grid format: 0=free (green), 1=obstacle (black)")
                print(f"    Grid matrix shape: {grid_matrix.shape}")
                print(f"    Free cells: {np.sum(grid_matrix == 0)}, Obstacle cells: {np.sum(grid_matrix == 1)}")
                print(f"    Saved: astar_grid_matrix.npy and astar_grid_matrix.txt")
            
            return grid_matrix, rows
        
        def overlay_grid_on_image(self, image, grid_matrix, rows, save_steps=True):
            """Overlay the A* grid on the original image using actual parking space positions"""
            if grid_matrix is None or rows is None:
                return None
            
            # Create overlay image
            overlay = image.copy()
            h, w = image.shape[:2]
            num_rows, max_cols = grid_matrix.shape
            
            # Draw grid cells using actual parking space positions
            for row_idx, row in enumerate(rows):
                for col_idx, space in enumerate(row):
                    # Get parking space boundaries
                    pts = space.vertices.astype(np.int32)
                    x_min = int(pts[:, 0].min())
                    y_min = int(pts[:, 1].min())
                    x_max = int(pts[:, 0].max())
                    y_max = int(pts[:, 1].max())
                    
                    # Clamp to image bounds
                    x_min = max(0, min(x_min, w))
                    y_min = max(0, min(y_min, h))
                    x_max = max(0, min(x_max, w))
                    y_max = max(0, min(y_max, h))
                    
                    # Get grid value
                    cell_value = grid_matrix[row_idx, col_idx]
                    
                    # Create semi-transparent overlay
                    if cell_value == 1:
                        # Obstacle - red overlay
                        overlay_color = (0, 0, 255)  # Red
                        alpha = 0.3
                    else:
                        # Free space - green overlay
                        overlay_color = (0, 255, 0)  # Green
                        alpha = 0.2
                    
                    # Draw semi-transparent rectangle
                    overlay_rect = overlay[y_min:y_max, x_min:x_max].copy()
                    if overlay_rect.size > 0 and y_max > y_min and x_max > x_min:
                        color_mask = np.full((y_max - y_min, x_max - x_min, 3), overlay_color, dtype=np.uint8)
                        overlay_rect = cv2.addWeighted(overlay_rect, 1 - alpha, color_mask, alpha, 0)
                        overlay[y_min:y_max, x_min:x_max] = overlay_rect
                    
                    # Draw grid cell border (using parking space polygon)
                    cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
                    
                    # Add cell coordinates at center
                    center = tuple(map(int, space.center))
                    cell_text = f"({row_idx},{col_idx})"
                    text_size = cv2.getTextSize(cell_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_x = center[0] - text_size[0] // 2
                    text_y = center[1] + text_size[1] // 2
                    
                    # White text with black outline
                    cv2.putText(overlay, cell_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(overlay, cell_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw obstacles on the overlay
            for obstacle in self.obstacles:
                x, y, w, h = obstacle['bbox']
                obs_type = obstacle['type']
                
                # Color by type
                if obs_type == 'tree':
                    color = (0, 255, 0)  # Green
                    label = 'TREE'
                elif obs_type == 'car':
                    color = (255, 0, 0)  # Blue
                    label = 'CAR'
                elif obs_type == 'wall':
                    color = (0, 0, 255)  # Red
                    label = 'WALL'
                else:
                    color = (128, 128, 128)  # Gray
                    label = 'OBSTACLE'
                
                # Draw bounding box
                cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 3)
                cv2.putText(overlay, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if save_steps:
                save_step(overlay, 21, "grid_overlay_on_image")
                print(f"    Grid overlaid on original image: {num_rows} rows x {max_cols} columns")
            
            return overlay
    
    return SimpleCVDetector()

def create_hybrid_detector():
    """Create a simplified hybrid detector"""
    
    class SimpleHybridDetector:
        def __init__(self, use_yolo=False):
            self.use_yolo = use_yolo
            self.cv_detector = create_simple_cv_detector()
            self.parking_spaces = []
            
            # Simple vehicle detection (mock for now)
            self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        
        def detect_vehicles_simple(self, image):
            """Simple vehicle detection using contour analysis"""
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours that could be vehicles
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vehicles = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Simple vehicle heuristics
                aspect_ratio = w / h if h > 0 else 0
                
                # Typical vehicle characteristics
                if (1000 < area < 20000 and 
                    0.8 < aspect_ratio < 3.0 and
                    w > 30 and h > 30):
                    
                    vehicles.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
            
            return vehicles
        
        def detect(self, image, method='cv'):
            """Detect parking spaces with optional vehicle detection"""
            # First detect parking spaces using CV
            spaces = self.cv_detector.detect_from_image(image)
            self.parking_spaces = spaces
            
            if method == 'hybrid' and spaces:
                # Detect vehicles
                vehicles = self.detect_vehicles_simple(image)
                
                # Check which spaces are occupied by vehicles
                for space in spaces:
                    space_occupied = False
                    
                    # Get space bounding box
                    if hasattr(space, 'bbox'):
                        sx, sy, sw, sh = space.bbox
                    else:
                        # Calculate from vertices
                        xs = space.vertices[:, 0]
                        ys = space.vertices[:, 1]
                        sx, sy = int(xs.min()), int(ys.min())
                        sw, sh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
                    
                    for vehicle in vehicles:
                        vx, vy, vw, vh = vehicle['bbox']
                        v_center_x, v_center_y = vehicle['center']
                        
                        # Check if vehicle center is inside parking space
                        if (sx <= v_center_x <= sx + sw and 
                            sy <= v_center_y <= sy + sh):
                            space_occupied = True
                            break
                    
                    space.is_occupied = space_occupied
            
            return self.parking_spaces
        
        def visualize(self, image):
            """Visualize using the CV detector"""
            return self.cv_detector.visualize(image)
    
    return SimpleHybridDetector(use_yolo=True)

# ===================================================================
# MAIN EXECUTION
# ===================================================================
def main():
    print("PARKING SPACE DETECTION SYSTEM")
    print("="*50)
    
    # Show original image
    cv2.imshow('Original Image', image)
    print("Showing original image. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ask user which method to use
    print("\nAvailable detection methods:")
    print("1. Computer Vision Only (Fast, works with any parking lot image)")
    print("2. Hybrid Approach (CV + Vehicle Detection - More accurate)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        option_computer_vision_only()
    elif choice == '2':
        option_hybrid_approach()
    else:
        print("Invalid choice. Using Computer Vision Only...")
        option_computer_vision_only()
    
    print("\n" + "="*60)
    print("DETECTION COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print(f"  - Final result: 'cv_detection_result.jpg' or 'hybrid_detection_result.jpg'")
    print(f"  - Step-by-step images: '{OUTPUT_FOLDER}/' folder")
    print(f"\nStep images saved in '{OUTPUT_FOLDER}/':")
    print("  01_original.png              - Original input image")
    print("  02_resized.png               - Resized image (if applicable)")
    print("  03_grayscale.png             - Grayscale conversion")
    print("  04_blurred.png               - Gaussian blur applied")
    print("  05_canny_edges.png           - Canny edge detection")
    print("  06_classified_lines_HV.png   - Horizontal (blue) and vertical (green) lines")
    print("  07_all_detected_lines.png    - All detected lines")
    print("  08_lines_closed.png          - Lines after morphological closing")
    print("  09_lines_thickened.png       - Lines after dilation")
    print("  10_inverted_regions.png      - Inverted regions for parking spaces")
    print("  11_all_regions_colored.png   - All detected regions (colored)")
    print("  12_valid_parking_regions.png - Filtered valid parking regions")
    print("  13_detected_contours.png     - Detected contours")
    print("  14_rectangles_filtered_by_shape.png - Rectangles after shape filter")
    print("  15_parking_spaces_final.png  - Final parking spaces")
    print("  16_occupancy_analysis.png    - Occupancy detection (red=occupied, green=empty)")
    print("  17_final_result_empty_spaces.png - Final result with empty spaces only")
    print("  18_parking_grid.png          - Grid visualization (green=empty, black=occupied, numbered)")
    print("  19_astar_grid_matrix.png     - A* grid matrix visualization")
    print("  20_detected_obstacles.png    - Detected obstacles (green=trees, blue=cars, red=walls)")
    print("  21_grid_overlay_on_image.png - A* grid overlaid on original image")
    print("\nA* Grid Matrix files:")
    print("  astar_grid_matrix.npy        - Grid matrix as numpy array (0=free, 1=obstacle)")
    print("  astar_grid_matrix.txt        - Grid matrix as text file")
    print("\nIf results are not satisfactory, try adjusting parameters:")
    print("  - white_threshold: Brightness threshold for white lines (default: 180)")
    print("  - contour_min_area/max_area: Size range for parking spaces")
    print("  - min/max_aspect_ratio: Shape constraints for parking spaces")
    print("  - occupancy_threshold: Sensitivity for occupancy detection")
    print("\nFor better results with real parking lot images:")
    print("  1. Ensure good lighting in the image")
    print("  2. Make sure parking space markings are visible")
    print("  3. Try different angles (bird's eye view works best)")

if __name__ == "__main__":
    # Make sure dataclass is available
    try:
        from dataclasses import dataclass
    except ImportError:
        print("Error: dataclasses module not available.")
        print("If using Python < 3.7, install it with: pip install dataclasses")
        sys.exit(1)
    
    main()
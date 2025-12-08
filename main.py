
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from heapq import heappush, heappop
import os
import threading
from math import sqrt
from scipy.interpolate import UnivariateSpline

import scipy.ndimage as ndimage


class ParkingGridConverter:
    """Convert parking lot images to navigable grid for A* algorithm"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Lot Path Planner")
        self.root.geometry("1400x900")
        
        # State variables
        self.original_image = None
        self.processed_image = None
        self.grid_matrix = None
        self.grid_rows = 80
        self.grid_cols = 120
        
        # Object detection
        self.yolo_model = None
        self.detected_objects = []
        self.detection_confidence = 0.3
        self.detection_method = "yolo"  # "yolo" or "traditional"
        
        # Traditional CV detection results
        self.parking_spots = []
        self.detected_cars = []
        
        # Navigation
        self.start_point = None
        self.end_point = None
        self.path = None
        self.smoothed_path = None
        self.inflated_grid = None
        self.clearance_radius = 1
        
        # Simulation
        self.simulation_running = False
        self.simulation_index = 0
        self.simulation_path = None  # The path being simulated
        self.car_heading = 0  # Car heading angle in degrees
        self.simulation_window = None  # Fullscreen simulation window
        self.simulation_canvas = None  # Canvas in fullscreen window
        
        # Create UI first (needed for status updates)
        self.create_widgets()
        
        # Load YOLO model in background after UI is ready
        self.load_yolo_model()
    
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        
        # Top Control Panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Title
        title_label = ttk.Label(
            control_frame,
            text="Parking Lot Path Planner",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.TOP, pady=10)
        
        # Button Frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.TOP, pady=5)
        
        ttk.Button(
            button_frame,
            text="1. Load Image",
            command=self.load_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="2. Detect Parking Spots",
            command=self.detect_parking_spots
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="3. Detect Obstacles (YOLO)",
            command=self.detect_obstacles_yolo
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="4. Generate Grid",
            command=self.generate_grid
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(
            button_frame,
            text="Set Start Point",
            command=lambda: self.set_mode("start")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Set End Point",
            command=lambda: self.set_mode("end")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Run A* Pathfinding",
            command=self.run_astar
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear Path",
            command=self.clear_path
        ).pack(side=tk.LEFT, padx=5)
        
        
        # ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Simulation controls
        self.sim_start_btn = ttk.Button(
            button_frame,
            text="▶ Simulate",
            command=self.start_simulation
        )
        self.sim_start_btn.pack(side=tk.LEFT, padx=5)
        
        # Grid Configuration Frame
        config_frame = ttk.LabelFrame(control_frame, text="Configuration", padding="10")
        config_frame.pack(side=tk.TOP, pady=10, fill=tk.X)
        
        ttk.Label(config_frame, text="Grid Rows:").pack(side=tk.LEFT, padx=5)
        self.rows_spinbox = ttk.Spinbox(
            config_frame,
            from_=5,
            to=100,
            width=8
        )
        self.rows_spinbox.set(self.grid_rows)
        self.rows_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(config_frame, text="Grid Columns:").pack(side=tk.LEFT, padx=5)
        self.cols_spinbox = ttk.Spinbox(
            config_frame,
            from_=5,
            to=100,
            width=8
        )
        self.cols_spinbox.set(self.grid_cols)
        self.cols_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(config_frame, text="YOLO Confidence:").pack(side=tk.LEFT, padx=5)
        self.confidence_spinbox = ttk.Spinbox(
            config_frame,
            from_=0.1,
            to=1.0,
            increment=0.05,
            width=8
        )
        self.confidence_spinbox.set(0.30)
        self.confidence_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(config_frame, text="Clearance:").pack(side=tk.LEFT, padx=5)
        self.clearance_spinbox = ttk.Spinbox(
            config_frame,
            from_=0,
            to=5,
            width=8
        )
        self.clearance_spinbox.set(1)
        self.clearance_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(config_frame, text="Cell Size (px):").pack(side=tk.LEFT, padx=5)
        self.cell_size_spinbox = ttk.Spinbox(
            config_frame,
            from_=5,
            to=50,
            width=8
        )
        self.cell_size_spinbox.set(10)  # Default cell size in pixels
        self.cell_size_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(config_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Simulation config
        ttk.Label(config_frame, text="Sim Speed:").pack(side=tk.LEFT, padx=5)
        self.sim_speed_spinbox = ttk.Spinbox(
            config_frame,
            from_=1,
            to=100,
            width=6
        )
        self.sim_speed_spinbox.set(30)  # Default speed (frames per movement)
        self.sim_speed_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Legend frame
        legend_frame = ttk.Frame(control_frame)
        legend_frame.pack(side=tk.TOP, pady=5, fill=tk.X)
        
        ttk.Label(legend_frame, text="Grid Legend:").pack(side=tk.LEFT, padx=10)
        ttk.Label(legend_frame, text="⬜ 0 = Navigable (A* path)", foreground="gray").pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="🔴 1 = Obstacle (YOLO)", foreground="red").pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="🟢 2 = Empty Parking Spot", foreground="green").pack(side=tk.LEFT, padx=5)
        
        # Main Content Area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Original/Processed Image
        left_panel = ttk.LabelFrame(content_frame, text="Parking Lot View", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_panel, bg="gray20")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Right Panel - Grid Visualization
        right_panel = ttk.LabelFrame(content_frame, text="Grid for A* Algorithm", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Grid canvas with scrollbars
        grid_scroll_frame = ttk.Frame(right_panel)
        grid_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.grid_canvas = tk.Canvas(grid_scroll_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(grid_scroll_frame, orient=tk.VERTICAL, command=self.grid_canvas.yview)
        h_scrollbar = ttk.Scrollbar(grid_scroll_frame, orient=tk.HORIZONTAL, command=self.grid_canvas.xview)
        
        self.grid_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bottom Status Panel
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.status_label = ttk.Label(
            status_frame,
            text="Ready. Load a parking lot image to begin.",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)
        
        # Info Panel
        info_frame = ttk.LabelFrame(status_frame, text="Information", padding="5")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=5, wrap=tk.WORD, font=("Courier", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        self.click_mode = None
    
    def load_yolo_model(self):
        """Load YOLO model in background thread"""
        def load_model():
            try:
                self.root.after(0, lambda: self.update_status("Loading YOLO model (downloading if first time)..."))
                from ultralytics import YOLO
                model_path = 'yolov8x-seg.pt'
                if not os.path.exists(model_path):
                    # Download the model if not present (will be automatic by yolo, but explicit for clarity)
                    self.yolo_model = YOLO('yolov8x-seg.pt')  # this will download if not already present
                else:
                    self.yolo_model = YOLO(model_path)
                self.root.after(0, lambda: self.update_status("YOLO model loaded! Ready to detect obstacles."))
                self.root.after(0, lambda: self.update_info(
                    "Workflow:\n"
                    "1. Load Image\n"
                    "2. Detect Parking Spots (Traditional CV) → Green\n"
                    "3. Detect Obstacles (YOLO) → Red\n"
                    "4. Generate Grid\n"
                    "5. Set Start/End → Run A*\n\n"
                    "6. Simulate \n"
                    "Click '1. Load Image' to begin."
                ))
            except ImportError as e:
                error_msg = "⚠ ultralytics not installed. YOLO obstacle detection disabled."
                self.root.after(0, lambda: self.update_status(error_msg))
                self.root.after(0, lambda: self.update_info(
                    "YOLO not available (ultralytics not installed).\n"
                    "Parking spot detection will still work.\n\n"
                    "To enable obstacle detection: pip install ultralytics"
                ))
                print(f"YOLO import error: {e}")
            except Exception as e:
                error_msg = f"YOLO load failed. Use Traditional CV instead."
                self.root.after(0, lambda: self.update_status(error_msg))
                print(f"YOLO load error: {e}")
        
        # Load in background to not block UI
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
        self.update_status("Initializing YOLO model...")
    
    def load_image(self):
        """Load parking lot image"""
        file_path = filedialog.askopenfilename(
            title="Select Parking Lot Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                messagebox.showerror("Error", "Failed to load image!")
                return
            
            # Reset detection results
            self.detected_objects = []
            self.parking_spots = []
            self.detected_cars = []
            self.grid_matrix = None
            self.path = None
            self.smoothed_path = None
            
            # Reset simulation state
            self.stop_simulation()
            
            # Display original image
            display_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(display_img, self.image_canvas)
            self.processed_image = display_img.copy()
            
            # Calculate expected grid dimensions
            img_height, img_width = self.original_image.shape[:2]
            cell_size = int(self.cell_size_spinbox.get())
            expected_cols = img_width // cell_size
            expected_rows = img_height // cell_size
            
            self.update_status(f"Loaded image: {os.path.basename(file_path)}")
            self.update_info(f"Image Size: {img_width}x{img_height} pixels\n"
                           f"Cell Size: {cell_size}px → Grid: {expected_cols}x{expected_rows}\n\n"
                           f"Next Step: 2. Detect Parking Spots\n"
                           f"Then: 3. Detect Obstacles (YOLO)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def detect_occupancy(self, image, gray, parking_spots, occupancy_threshold=0.10, output_folder="cv_process_images", save_steps=True):
        """Detect occupancy of parking spots using multi-feature approach from test4.py
        
        Uses multiple features:
        - Edge density (cars have more edges)
        - Variance (cars have more texture variation)
        - Bright pixels (white car roofs)
        - Dark pixels (car shadows/body)
        """
        if save_steps:
            os.makedirs(output_folder, exist_ok=True)
            
            # Create visualization images for each step
            edge_vis = np.zeros_like(gray)
            variance_vis = np.zeros_like(gray)
            bright_vis = np.zeros_like(gray)
            dark_vis = np.zeros_like(gray)
            combined_vis = np.zeros_like(gray)
        
        for spot in parking_spots:
            # Get parking space bounds
            x = spot['bounding_rect']['x']
            y = spot['bounding_rect']['y']
            w = spot['bounding_rect']['width']
            h = spot['bounding_rect']['height']
            
            # Create mask for this parking space using contour
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            if 'contour' in spot:
                cv2.fillPoly(mask, [spot['contour']], 255)
            else:
                # Fallback to bounding rectangle
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            # Get the ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_mask = mask[y:y+h, x:x+w]
            
            if roi_gray.size == 0:
                spot['is_occupied'] = False
                spot['occupancy_confidence'] = 0.0
                continue
            
            mask_pixels = np.sum(roi_mask > 0)
            if mask_pixels == 0:
                spot['is_occupied'] = False
                spot['occupancy_confidence'] = 0.0
                continue
            
            # Method 1: Edge density (cars have more edges, but ignore weak edges from lines)
            # Use higher thresholds to ignore parking line edges
            edges = cv2.Canny(roi_gray, 80, 200)
            edges_masked = cv2.bitwise_and(edges, edges, mask=roi_mask)
            edge_density = np.sum(edges_masked > 0) / (mask_pixels + 1e-5)
            
            if save_steps:
                # Copy edges to visualization (scale for visibility)
                edge_roi = edge_vis[y:y+h, x:x+w]
                edge_roi[roi_mask > 0] = edges_masked[roi_mask > 0]
            
            # Method 2: Variance (cars have more texture variation)
            roi_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)
            masked_pixels = roi_masked[roi_mask > 0]
            if len(masked_pixels) > 0:
                variance = np.var(masked_pixels)
                variance_normalized = variance / 2000  # Normalize (reduced sensitivity)
                # Create variance visualization using Laplacian (measures local variation)
                laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
                laplacian_abs = np.abs(laplacian)
                variance_img = np.clip(laplacian_abs * 2, 0, 255).astype(np.uint8)
            else:
                variance_normalized = 0
                variance_img = np.zeros_like(roi_gray)
            
            if save_steps:
                # Copy variance visualization
                var_roi = variance_vis[y:y+h, x:x+w]
                var_roi[roi_mask > 0] = variance_img[roi_mask > 0]
            
            # Method 3: Check for bright pixels (white car roofs) - more conservative
            bright_threshold = 220  # Higher threshold for bright pixels
            bright_mask = (roi_gray > bright_threshold) & (roi_mask > 0)
            bright_pixels = np.sum(bright_mask)
            bright_ratio = bright_pixels / (mask_pixels + 1e-5)
            
            if save_steps:
                # Create bright pixels visualization
                bright_roi = bright_vis[y:y+h, x:x+w]
                bright_roi[bright_mask] = 255
            
            # Method 4: Check for dark pixels (car shadows/body)
            dark_threshold = 50
            dark_mask = (roi_gray < dark_threshold) & (roi_mask > 0)
            dark_pixels = np.sum(dark_mask)
            dark_ratio = dark_pixels / (mask_pixels + 1e-5)
            
            if save_steps:
                # Create dark pixels visualization
                dark_roi = dark_vis[y:y+h, x:x+w]
                dark_roi[dark_mask] = 255
            
            # Combined score - adjusted weights, less sensitive to edges
            combined_score = (edge_density * 0.3 + variance_normalized * 0.25 + 
                             bright_ratio * 0.2 + dark_ratio * 0.25)
            
            if save_steps:
                # Create combined visualization (normalize score to 0-255)
                combined_roi = combined_vis[y:y+h, x:x+w]
                combined_value = int(np.clip(combined_score * 255 / occupancy_threshold, 0, 255))
                combined_roi[roi_mask > 0] = combined_value
            
            # Determine occupancy
            spot['is_occupied'] = combined_score > occupancy_threshold
            spot['occupancy_confidence'] = combined_score
            spot['occupancy_features'] = {
                'edge_density': edge_density,
                'variance_normalized': variance_normalized,
                'bright_ratio': bright_ratio,
                'dark_ratio': dark_ratio
            }
        
        # Save step images
        if save_steps:
            # Save edge detection visualization
            cv2.imwrite(os.path.join(output_folder, "19_occupancy_edges.png"), edge_vis)
            
            # Save variance visualization
            cv2.imwrite(os.path.join(output_folder, "20_occupancy_variance.png"), variance_vis)
            
            # Save bright pixels visualization
            cv2.imwrite(os.path.join(output_folder, "21_occupancy_bright_pixels.png"), bright_vis)
            
            # Save dark pixels visualization
            cv2.imwrite(os.path.join(output_folder, "22_occupancy_dark_pixels.png"), dark_vis)
            
            # Save combined score visualization
            cv2.imwrite(os.path.join(output_folder, "23_occupancy_combined_score.png"), combined_vis)
            
            # Create color-coded visualization showing all features
            # Use original RGB image as base
            color_vis = image.copy()
            for spot in parking_spots:
                x = spot['bounding_rect']['x']
                y = spot['bounding_rect']['y']
                w = spot['bounding_rect']['width']
                h = spot['bounding_rect']['height']
                is_occupied = spot.get('is_occupied', False)
                confidence = spot.get('occupancy_confidence', 0.0)
                
                # Draw bounding box (RGB colors: red for occupied, green for empty)
                color = (255, 0, 0) if is_occupied else (0, 255, 0)
                cv2.rectangle(color_vis, (x, y), (x + w, y + h), color, 2)
                
                # Add text with confidence
                cx, cy = map(int, spot['min_area_rect']['center'])
                status = "OCC" if is_occupied else "EMP"
                cv2.putText(color_vis, f"P{spot['spot_id']}: {status}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(color_vis, f"Score: {confidence:.3f}", (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Convert RGB to BGR for saving
            cv2.imwrite(os.path.join(output_folder, "24_occupancy_color_coded.png"), 
                       cv2.cvtColor(color_vis, cv2.COLOR_RGB2BGR))
    
    def _order_points(self, pts):
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
    
    def detect_parking_spots(self):
        """Step 2: Detect empty parking spots using Traditional CV with occupancy detection
        Uses comprehensive Hough line-based approach from test4.py"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            self.update_status("Detecting parking spots (Traditional CV - Hough Line Method)...")
            
            # Create output folder for process images
            output_folder = "cv_process_images"
            os.makedirs(output_folder, exist_ok=True)
            
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Step 1: Save original image
            cv2.imwrite(os.path.join(output_folder, "1_original.png"), self.original_image)
            
            # Preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Smaller blur to preserve edges
            
            # Step 2: Save grayscale and blurred
            cv2.imwrite(os.path.join(output_folder, "2_grayscale.png"), gray)
            cv2.imwrite(os.path.join(output_folder, "3_blurred.png"), blurred)
            
            # Step 4: Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            cv2.imwrite(os.path.join(output_folder, "4_edges_canny.png"), edges)
            
            # Step 5: Hough Line Transform to detect line segments
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                    minLineLength=40, maxLineGap=15)
            
            # Draw all Hough lines for visualization
            hough_vis = image.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(hough_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, "5_hough_lines.png"), 
                       cv2.cvtColor(hough_vis, cv2.COLOR_RGB2BGR))
            
            # Step 6: Separate horizontal and vertical lines
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
            
            # Step 6: Draw classified lines (H and V in different colors)
            colored_lines = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            for x1, y1, x2, y2 in h_lines:
                cv2.line(colored_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for H
            for x1, y1, x2, y2 in v_lines:
                cv2.line(colored_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for V
            cv2.imwrite(os.path.join(output_folder, "6_classified_lines_HV.png"), 
                       cv2.cvtColor(colored_lines, cv2.COLOR_RGB2BGR))
            
            # Step 7: Draw all detected lines
            lines_image = np.zeros_like(gray)
            for x1, y1, x2, y2 in h_lines:
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
            for x1, y1, x2, y2 in v_lines:
                cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
            cv2.imwrite(os.path.join(output_folder, "7_all_detected_lines.png"), lines_image)
            
            # Step 8: Close gaps in lines (morphological closing)
            # Close horizontal gaps in horizontal lines
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            # Close vertical gaps in vertical lines  
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            
            closed = cv2.morphologyEx(lines_image, cv2.MORPH_CLOSE, kernel_h)
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
            cv2.imwrite(os.path.join(output_folder, "8_lines_closed.png"), closed)
            
            # Step 9: Dilate slightly to ensure connectivity at intersections
            kernel_dilate = np.ones((3, 3), np.uint8)
            thickened = cv2.dilate(closed, kernel_dilate, iterations=1)
            cv2.imwrite(os.path.join(output_folder, "9_lines_thickened.png"), thickened)
            
            # Step 10: Invert to get parking space regions
            inverted = cv2.bitwise_not(thickened)
            cv2.imwrite(os.path.join(output_folder, "10_inverted_regions.png"), inverted)
            
            # Step 11: Use connected components to find separate regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
            
            parking_regions = np.zeros_like(gray)
            valid_regions = []
            
            # Parking spot detection parameters 
            contour_min_area = 1500
            contour_max_area = 25000
            min_aspect_ratio = 0.1
            max_aspect_ratio = 0.8
            
            # # Step 12: Show all regions with different colors (for debugging)
            # colored_labels = np.zeros((inverted.shape[0], inverted.shape[1], 3), dtype=np.uint8)
            # for i in range(1, num_labels):
            #     mask = labels == i
            #     color = ((i * 50) % 255, (i * 80) % 255, (i * 110) % 255)
            #     colored_labels[mask] = color
            # cv2.imwrite(os.path.join(output_folder, "12_all_regions_colored.png"), 
            #            cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR))
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w_region = stats[i, cv2.CC_STAT_WIDTH]
                h_region = stats[i, cv2.CC_STAT_HEIGHT]
                aspect = h_region / w_region if w_region > 0 else 0
                
                # Check if this could be a parking space
                if contour_min_area <= area <= contour_max_area:
                    if min_aspect_ratio <= aspect <= max_aspect_ratio:
                        valid_regions.append(i)
                        parking_regions[labels == i] = 255
            
            # Step 13: Save valid parking regions
            cv2.imwrite(os.path.join(output_folder, "13_valid_parking_regions.png"), parking_regions)
            
            # Step 14: Find contours in the filtered parking regions
            contours, _ = cv2.findContours(parking_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_vis = image.copy()
            cv2.drawContours(contours_vis, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, "14_detected_contours.png"), 
                       cv2.cvtColor(contours_vis, cv2.COLOR_RGB2BGR))
            
            # Step 15: Filter rectangles by shape (rectangularity)
            rectangles = []
            rejected_rectangularity = []
            min_rectangularity = 0.80  # From test4.py
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1500:  # Skip very small contours
                    continue
                
                # Get minimum area rectangle for better fit
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                ordered = self._order_points(box)
                
                # Calculate rectangularity (how well contour fits a rectangle)
                rect_area = cv2.contourArea(ordered)
                if rect_area > 0:
                    rectangularity = area / rect_area
                else:
                    rectangularity = 0
                
                # Only keep nearly perfect rectangles
                if rectangularity >= min_rectangularity:
                    rectangles.append(ordered)
                else:
                    rejected_rectangularity.append((ordered, rectangularity))
            
            # Save rectangles filtered by shape
            rect_vis = image.copy()
            for rect in rectangles:
                pts = rect.astype(np.int32)
                cv2.polylines(rect_vis, [pts], True, (0, 255, 255), 2)
            # Also show rejected in red
            for rect, _ in rejected_rectangularity:
                pts = rect.astype(np.int32)
                cv2.polylines(rect_vis, [pts], True, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(output_folder, "15_rectangles_filtered_by_shape.png"), 
                       cv2.cvtColor(rect_vis, cv2.COLOR_RGB2BGR))
            
            # Step 16: Remove duplicate rectangles based on center proximity
            filtered = self._remove_duplicate_rectangles(rectangles, distance_threshold=50)
            
            # Save final parking spaces
            rect_final_vis = image.copy()
            for i, rect in enumerate(filtered):
                pts = rect.astype(np.int32)
                cv2.polylines(rect_final_vis, [pts], True, (0, 255, 0), 2)
                center = np.mean(rect, axis=0).astype(int)
                cv2.putText(rect_final_vis, str(i), tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, "16_parking_spaces_final.png"), 
                       cv2.cvtColor(rect_final_vis, cv2.COLOR_RGB2BGR))
            
            # Convert filtered rectangles to parking spot format
            parking_spot_rects = []
            spot_number = 0
            
            for rect in filtered:
                spot_number += 1
                area = cv2.contourArea(rect)
                x, y, w, h = cv2.boundingRect(rect.astype(np.int32))
                aspect_ratio = float(max(w, h)) / min(w, h) if min(w, h) > 0 else 0
                
                min_rect = cv2.minAreaRect(rect.astype(np.int32))
                box_points = cv2.boxPoints(min_rect)
                box_points = np.int64(box_points)
                
                rect_info = {
                    'spot_id': spot_number,
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
                    'contour': rect.astype(np.int32),
                    'is_occupied': False,
                    'occupancy_confidence': 0.0
                }
                parking_spot_rects.append(rect_info)
            
            # Store parking spots
            self.parking_spots = parking_spot_rects
            
            # Detect occupancy using multi-feature approach from test4.py
            if parking_spot_rects:
                self.update_status("Detecting occupancy of parking spots...")
                self.detect_occupancy(image, gray, parking_spot_rects, occupancy_threshold=0.15, 
                                    output_folder=output_folder, save_steps=True)
            
            # Separate empty and occupied spots
            empty_spots = [s for s in parking_spot_rects if not s.get('is_occupied', False)]
            occupied_spots = [s for s in parking_spot_rects if s.get('is_occupied', False)]
            
            # Draw parking spots on image (GREEN for empty, RED for occupied)
            result_image = image.copy()
            occupancy_vis = image.copy()
            
            for rect in parking_spot_rects:
                x = rect['bounding_rect']['x']
                y = rect['bounding_rect']['y']
                w = rect['bounding_rect']['width']
                h = rect['bounding_rect']['height']
                is_occupied = rect.get('is_occupied', False)
                confidence = rect.get('occupancy_confidence', 0.0)
                
                # Choose color based on occupancy
                if is_occupied:
                    color = (255, 0, 0)  # Red for occupied
                    status_text = "OCC"
                else:
                    color = (0, 255, 0)  # Green for empty
                    status_text = "EMP"
                
                # Draw bounding rectangle
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw min area rectangle
                box = np.array(rect['min_area_rect']['corner_points'], dtype=np.int32)
                cv2.polylines(result_image, [box], True, color, 2)
                
                # Label with spot number and status
                cx, cy = map(int, rect['min_area_rect']['center'])
                cv2.putText(result_image, f"P{rect['spot_id']}", (cx-15, cy-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                cv2.putText(result_image, status_text, (cx-15, cy+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                
                # For occupancy visualization, also show confidence
                cv2.polylines(occupancy_vis, [box], True, color, 2)
                cv2.putText(occupancy_vis, f'{confidence:.3f}', 
                           (cx - 25, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            self.processed_image = result_image
            self.display_image(self.processed_image, self.image_canvas)
            
            # Step 17: Save final result with detected parking spots
            cv2.imwrite(os.path.join(output_folder, "17_detected_parking_spots.png"), 
                       cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            # Step 18: Save occupancy analysis
            cv2.imwrite(os.path.join(output_folder, "18_occupancy_analysis.png"), 
                       cv2.cvtColor(occupancy_vis, cv2.COLOR_RGB2BGR))
            
            self.update_status(f"Detected {len(empty_spots)} empty, {len(occupied_spots)} occupied parking spots (images saved to {output_folder}/)")
            
            info_text = f"Parking Spot Detection (Traditional CV):\n"
            info_text += f"Total parking spots found: {len(parking_spot_rects)}\n"
            info_text += f"  🟢 Empty spots: {len(empty_spots)}\n"
            info_text += f"  🔴 Occupied spots: {len(occupied_spots)}\n\n"
            info_text += "Process images saved to: cv_process_images/\n"
            info_text += "  1. Original → 2. Grayscale → 3. Blurred\n"
            info_text += "  4. Canny Edges → 5. Hough Lines\n"
            info_text += "  6. Classified Lines (HV) → 7. All Detected Lines\n"
            info_text += "  8. Lines Closed → 9. Lines Thickened\n"
            info_text += "  10. Inverted Regions → 12. All Regions Colored\n"
            info_text += "  13. Valid Parking Regions → 14. Detected Contours\n"
            info_text += "  15. Rectangles Filtered → 16. Parking Spaces Final\n"
            info_text += "  17. Detected Parking Spots → 18. Occupancy Analysis\n"
            info_text += "  19. Occupancy Edges → 20. Occupancy Variance\n"
            info_text += "  21. Occupancy Bright Pixels → 22. Occupancy Dark Pixels\n"
            info_text += "  23. Occupancy Combined Score → 24. Occupancy Color Coded\n"
            info_text += "  (Note: Step 11 processes connected components, no image saved)\n\n"
            
            if empty_spots:
                info_text += "Empty spots (first 10):\n"
                for rect in empty_spots[:10]:
                    info_text += f"  P{rect['spot_id']}: x={rect['bounding_rect']['x']}, y={rect['bounding_rect']['y']}, {rect['bounding_rect']['width']}x{rect['bounding_rect']['height']}px\n"
                if len(empty_spots) > 10:
                    info_text += f"  ... and {len(empty_spots)-10} more empty spots\n"
            
            info_text += "\nNext: Detect Obstacles (YOLO)"
            self.update_info(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting parking spots: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def detect_obstacles_yolo(self):
        """Step 3: Detect obstacles (cars, people, etc.) using YOLO"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if self.yolo_model is None:
            messagebox.showwarning("Warning", "YOLO model not loaded yet. Please wait or try again.")
            return
        
        try:
            self.update_status("Detecting obstacles (YOLO)...")
            
            # Update confidence from spinbox
            self.detection_confidence = float(self.confidence_spinbox.get())
            
            # Run YOLO detection
            results = self.yolo_model(self.original_image, conf=self.detection_confidence, verbose=False)
            
            # Get detections
            self.detected_objects = []
            
            # COCO classes we consider as obstacles
            obstacle_classes = {
                0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 
                7: 'truck', 9: 'traffic light', 10: 'fire hydrant',
                11: 'stop sign', 13: 'bench', 15: 'cat', 16: 'dog', 17: 'fence',
                24: 'backpack', 26: 'handbag', 28: 'suitcase',
                39: 'bottle', 41: 'cup', 56: 'chair', 57: 'couch',
                58: 'potted plant', 59: 'bed', 60: 'dining table',
                62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                66: 'keyboard', 67: 'cell phone', 73: 'book'
            }
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    self.detected_objects.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name,
                        'is_obstacle': class_id in obstacle_classes
                    })
            
            # Draw both parking spots (green) and obstacles (red) on the image
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            result_image = image.copy()
            
            # First draw parking spots (GREEN for empty, RED for occupied) if they exist
            if self.parking_spots:
                for rect in self.parking_spots:
                    x = rect['bounding_rect']['x']
                    y = rect['bounding_rect']['y']
                    w = rect['bounding_rect']['width']
                    h = rect['bounding_rect']['height']
                    is_occupied = rect.get('is_occupied', False)
                    
                    # Choose color based on occupancy
                    if is_occupied:
                        color = (255, 0, 0)  # Red for occupied
                        status_text = "OCC"
                    else:
                        color = (0, 255, 0)  # Green for empty
                        status_text = "EMP"
                    
                    # Fill with transparency effect
                    overlay = result_image.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                    cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)
                    
                    # Border
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Label with spot number and status
                    cx, cy = map(int, rect['min_area_rect']['center'])
                    cv2.putText(result_image, f"P{rect['spot_id']}", (cx-10, cy-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    cv2.putText(result_image, status_text, (cx-10, cy+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            
            # Then draw YOLO obstacles (RED) on top
            obstacle_count = 0
            for obj in self.detected_objects:
                if obj['is_obstacle']:
                    obstacle_count += 1
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # Red fill with transparency effect
                    overlay = result_image.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)
                    
                    # Red border
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    
                    # Label
                    label = f"{obj['class_name']} {obj['confidence']:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(result_image, 
                                (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1),
                                (255, 0, 0), -1)
                    cv2.putText(result_image, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add legend to image
            legend_y = 30
            cv2.rectangle(result_image, (5, 5), (220, 100), (255, 255, 255), -1)
            cv2.rectangle(result_image, (5, 5), (220, 100), (0, 0, 0), 1)
            cv2.rectangle(result_image, (10, 15), (25, 30), (0, 255, 0), -1)
            cv2.putText(result_image, "Empty Parking Spot", (30, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.rectangle(result_image, (10, 40), (25, 55), (255, 0, 0), -1)
            cv2.putText(result_image, "Occupied Spot", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.rectangle(result_image, (10, 65), (25, 80), (255, 0, 0), -1)
            cv2.putText(result_image, "Obstacle (YOLO)", (30, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            self.processed_image = result_image
            self.display_image(self.processed_image, self.image_canvas)
            
            empty_count = len([s for s in self.parking_spots if not s.get('is_occupied', False)]) if self.parking_spots else 0
            occupied_count = len([s for s in self.parking_spots if s.get('is_occupied', False)]) if self.parking_spots else 0
            total_parking = len(self.parking_spots) if self.parking_spots else 0
            
            self.update_status(f"YOLO: {obstacle_count} obstacles | Parking: {empty_count} empty, {occupied_count} occupied")
            
            info_text = f"Combined Detection Results:\n"
            info_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            info_text += f"🟢 Empty Parking Spots: {empty_count}\n"
            info_text += f"🔴 Occupied Parking Spots: {occupied_count}\n"
            info_text += f"🔴 Obstacles (YOLO): {obstacle_count}\n\n"
            
            obstacle_summary = {}
            for obj in self.detected_objects:
                if obj['is_obstacle']:
                    name = obj['class_name']
                    obstacle_summary[name] = obstacle_summary.get(name, 0) + 1
            
            if obstacle_summary:
                info_text += "Obstacle breakdown:\n"
                for name, count in obstacle_summary.items():
                    info_text += f"  • {name}: {count}\n"
            
            info_text += "\nNext: Generate Grid"
            self.update_info(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting obstacles: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_grid(self):
        """Generate navigable grid matrix using detected objects
        
        Grid values:
        - 0: Navigable (free space for A* pathfinding)
        - 1: Obstacle (YOLO detected objects - cars, people, etc.)
        - 2: Empty parking spot (Traditional CV detected)
        """
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            # Get image dimensions
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Get cell size from spinbox
            cell_size = int(self.cell_size_spinbox.get())
            
            # Calculate grid dimensions from image size and cell size
            self.grid_cols = width // cell_size
            self.grid_rows = height // cell_size
            
            # Update spinboxes to show calculated values
            self.rows_spinbox.delete(0, tk.END)
            self.rows_spinbox.insert(0, str(self.grid_rows))
            self.cols_spinbox.delete(0, tk.END)
            self.cols_spinbox.insert(0, str(self.grid_cols))
            
            # Calculate actual cell dimensions
            cell_height = height // self.grid_rows
            cell_width = width // self.grid_cols
            
            # Initialize grid: 0 = navigable (default)
            self.grid_matrix = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
            
            self.update_status("Generating grid from detections...")
            
            # Step 1: Mark empty parking spots (value 2) from Traditional CV
            # Only mark spots that are NOT occupied
            if self.parking_spots:
                for rect in self.parking_spots:
                    # Skip occupied spots - they should be treated as obstacles
                    if rect.get('is_occupied', False):
                        continue
                    
                    x = rect['bounding_rect']['x']
                    y = rect['bounding_rect']['y']
                    w = rect['bounding_rect']['width']
                    h = rect['bounding_rect']['height']
                    
                    # Find grid cells covered by this parking spot
                    start_col = max(0, x // cell_width)
                    end_col = min(self.grid_cols - 1, (x + w) // cell_width)
                    start_row = max(0, y // cell_height)
                    end_row = min(self.grid_rows - 1, (y + h) // cell_height)
                    
                    for row in range(start_row, end_row + 1):
                        for col in range(start_col, end_col + 1):
                            # Calculate overlap
                            cell_x1 = col * cell_width
                            cell_y1 = row * cell_height
                            cell_x2 = cell_x1 + cell_width
                            cell_y2 = cell_y1 + cell_height
                            
                            overlap_x1 = max(x, cell_x1)
                            overlap_y1 = max(y, cell_y1)
                            overlap_x2 = min(x + w, cell_x2)
                            overlap_y2 = min(y + h, cell_y2)
                            
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                cell_area = cell_width * cell_height
                                if overlap_area > cell_area * 0.2:
                                    self.grid_matrix[row, col] = 2  # Empty parking spot
            
            # Step 1b: Mark occupied parking spots as obstacles (value 1)
            if self.parking_spots:
                for rect in self.parking_spots:
                    # Only mark occupied spots as obstacles
                    if not rect.get('is_occupied', False):
                        continue
                    
                    x = rect['bounding_rect']['x']
                    y = rect['bounding_rect']['y']
                    w = rect['bounding_rect']['width']
                    h = rect['bounding_rect']['height']
                    
                    # Find grid cells covered by this occupied parking spot
                    start_col = max(0, x // cell_width)
                    end_col = min(self.grid_cols - 1, (x + w) // cell_width)
                    start_row = max(0, y // cell_height)
                    end_row = min(self.grid_rows - 1, (y + h) // cell_height)
                    
                    for row in range(start_row, end_row + 1):
                        for col in range(start_col, end_col + 1):
                            # Calculate overlap
                            cell_x1 = col * cell_width
                            cell_y1 = row * cell_height
                            cell_x2 = cell_x1 + cell_width
                            cell_y2 = cell_y1 + cell_height
                            
                            overlap_x1 = max(x, cell_x1)
                            overlap_y1 = max(y, cell_y1)
                            overlap_x2 = min(x + w, cell_x2)
                            overlap_y2 = min(y + h, cell_y2)
                            
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                cell_area = cell_width * cell_height
                                if overlap_area > cell_area * 0.2:
                                    self.grid_matrix[row, col] = 1  # Occupied parking spot = obstacle
            
            # Step 2: Mark obstacles (value 1) from YOLO - overwrites parking spots if overlapping
            if self.detected_objects:
                for obj in self.detected_objects:
                    if not obj['is_obstacle']:
                        continue
                    
                    x1, y1, x2, y2 = obj['bbox']
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    start_col = x1 // cell_width
                    end_col = x2 // cell_width
                    start_row = y1 // cell_height
                    end_row = y2 // cell_height
                    
                    for row in range(start_row, min(end_row + 1, self.grid_rows)):
                        for col in range(start_col, min(end_col + 1, self.grid_cols)):
                            cell_x1 = col * cell_width
                            cell_y1 = row * cell_height
                            cell_x2 = cell_x1 + cell_width
                            cell_y2 = cell_y1 + cell_height
                            
                            overlap_x1 = max(x1, cell_x1)
                            overlap_y1 = max(y1, cell_y1)
                            overlap_x2 = min(x2, cell_x2)
                            overlap_y2 = min(y2, cell_y2)
                            
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                cell_area = cell_width * cell_height
                                if overlap_area > cell_area * 0.2:
                                    self.grid_matrix[row, col] = 1  # Obstacle
            
            # Visualize grid on image
            grid_image = image.copy()
            
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    y1 = row * cell_height
                    x1 = col * cell_width
                    y2 = y1 + cell_height
                    x2 = x1 + cell_width
                    
                    cell_val = self.grid_matrix[row, col]
                    
                    if cell_val == 1:
                        # Obstacle (YOLO) - RED overlay
                        overlay = grid_image[y1:y2, x1:x2].copy()
                        overlay[:, :] = [255, 50, 50]
                        grid_image[y1:y2, x1:x2] = cv2.addWeighted(
                            grid_image[y1:y2, x1:x2], 0.5, overlay, 0.5, 0
                        )
                    elif cell_val == 2:
                        # Empty parking spot (Traditional CV) - GREEN overlay
                        overlay = grid_image[y1:y2, x1:x2].copy()
                        overlay[:, :] = [50, 255, 50]
                        grid_image[y1:y2, x1:x2] = cv2.addWeighted(
                            grid_image[y1:y2, x1:x2], 0.5, overlay, 0.5, 0
                        )
                    # value 0 = navigable, no overlay (original image shows through)
                    
                    # Draw grid lines
                    cv2.rectangle(grid_image, (x1, y1), (x2, y2), (200, 200, 200), 1)
            
            # Add legend to grid image
            legend_y = 25
            cv2.rectangle(grid_image, (5, 5), (280, 85), (255, 255, 255), -1)
            cv2.rectangle(grid_image, (5, 5), (280, 85), (0, 0, 0), 2)
            cv2.rectangle(grid_image, (15, 15), (30, 30), (255, 50, 50), -1)
            cv2.putText(grid_image, "1 = Obstacle (YOLO)", (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.rectangle(grid_image, (15, 40), (30, 55), (50, 255, 50), -1)
            cv2.putText(grid_image, "2 = Empty Parking Spot", (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.rectangle(grid_image, (15, 65), (30, 80), (200, 200, 200), 1)
            cv2.putText(grid_image, "0 = Navigable (A* path)", (40, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            self.display_image(grid_image, self.image_canvas)
            self.visualize_grid_matrix()
            
            # Save grid image
            self.save_grid_image(self.grid_matrix, "25_grid_matrix.png")
            
            # Statistics
            total_cells = self.grid_rows * self.grid_cols
            obstacle_cells = np.sum(self.grid_matrix == 1)
            parking_cells = np.sum(self.grid_matrix == 2)
            navigable_cells = np.sum(self.grid_matrix == 0)
            
            self.update_status(f"Grid generated: {self.grid_cols}x{self.grid_rows} from {width}x{height} image")
            
            info_text = f"Grid Matrix Generated:\n"
            info_text += f"Image: {width}x{height} pixels\n"
            info_text += f"Cell Size: {cell_size}px\n"
            info_text += f"Grid: {self.grid_cols} cols × {self.grid_rows} rows = {total_cells} cells\n\n"
            info_text += f"Navigable (0): {navigable_cells} ({navigable_cells/total_cells*100:.1f}%)\n"
            info_text += f"Obstacles (1): {obstacle_cells} ({obstacle_cells/total_cells*100:.1f}%)\n"
            info_text += f"Empty spots (2): {parking_cells} ({parking_cells/total_cells*100:.1f}%)\n\n"
            info_text += "Ready for A* pathfinding!"
            
            self.update_info(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating grid: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def visualize_grid_matrix(self):
        """Visualize the grid matrix as a 2D array"""
        if self.grid_matrix is None:
            return
        
        self.grid_canvas.delete("all")
        cell_size = 5
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                cell_val = self.grid_matrix[row, col]
                
                if cell_val == 1:
                    color = "#ff3333"  # Red for obstacles (YOLO)
                elif cell_val == 2:
                    color = "#33ff33"  # Green for empty parking spots
                else:
                    color = "#ffffff"  # White for navigable
                
                self.grid_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="#cccccc"
                )
        
        self.grid_canvas.configure(
            scrollregion=(0, 0, self.grid_cols * cell_size, self.grid_rows * cell_size)
        )
    
    def save_grid_image(self, grid, filename, is_binary=False):
        """Save grid matrix as an image file"""
        try:
            output_folder = "cv_process_images"
            os.makedirs(output_folder, exist_ok=True)
            
            rows, cols = grid.shape
            
            if is_binary:
                # Binary grid (0 = white/navigable, 1 = black/obstacle)
                grid_img = np.zeros((rows, cols), dtype=np.uint8)
                grid_img[grid == 0] = 255  # Navigable = white
                grid_img[grid == 1] = 0    # Obstacle = black
            else:
                # Color grid (0 = white, 1 = red, 2 = green)
                grid_img = np.zeros((rows, cols, 3), dtype=np.uint8)
                grid_img[grid == 0] = [255, 255, 255]  # White for navigable
                grid_img[grid == 1] = [0, 0, 255]      # Red for obstacles (BGR)
                grid_img[grid == 2] = [0, 255, 0]      # Green for parking spots (BGR)
            
            # Scale up for better visibility
            scale = 5
            grid_img_scaled = cv2.resize(grid_img, (cols * scale, rows * scale), interpolation=cv2.INTER_NEAREST)
            
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, grid_img_scaled)
            print(f"Grid image saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving grid image: {e}")
    
    def save_inflated_grid_image(self, inflated_grid, original_grid, filename):
        """Save inflated grid image with parking spots shown"""
        try:
            output_folder = "cv_process_images"
            os.makedirs(output_folder, exist_ok=True)
            
            rows, cols = inflated_grid.shape
            
            # Create color image
            grid_img = np.zeros((rows, cols, 3), dtype=np.uint8)
            
            # First set navigable areas (white)
            grid_img[inflated_grid == 0] = [255, 255, 255]
            
            # Second: Draw parking spots from original grid (green) 
            grid_img[original_grid == 2] = [0, 255, 0]  # Green for parking spots (BGR)
            
            # Third: Draw inflated obstacles on top (red) - overlaps green parking spots
            grid_img[inflated_grid == 1] = [0, 0, 255]  # Red for inflated obstacles (BGR)
            
            # Scale up for better visibility
            scale = 5
            grid_img_scaled = cv2.resize(grid_img, (cols * scale, rows * scale), interpolation=cv2.INTER_NEAREST)
            
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, grid_img_scaled)
            print(f"Inflated grid image saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving inflated grid image: {e}")
    
    def save_path_image(self, path, filename, is_smoothed=False):
        """Save path on grid as an image file"""
        try:
            output_folder = "cv_process_images"
            os.makedirs(output_folder, exist_ok=True)
            
            rows, cols = self.grid_rows, self.grid_cols
            scale = 5
            
            # Create base image from grid
            grid_img = np.zeros((rows * scale, cols * scale, 3), dtype=np.uint8)
            
            # Draw grid background
            for row in range(rows):
                for col in range(cols):
                    x1, y1 = col * scale, row * scale
                    x2, y2 = x1 + scale, y1 + scale
                    
                    if self.grid_matrix[row, col] == 1:
                        color = (50, 50, 50)  # Dark gray for obstacles
                    elif self.grid_matrix[row, col] == 2:
                        color = (0, 255, 0)  # Green for parking spots
                    else:
                        color = (255, 255, 255)  # White for navigable
                    
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, -1)
            
            # Draw path
            if is_smoothed:
                # Smoothed path - draw as continuous line
                path_color = (0, 255, 0)  # Green for smoothed path (BGR)
                for i in range(len(path) - 1):
                    row1, col1 = path[i]
                    row2, col2 = path[i + 1]
                    
                    pt1 = (int(col1 * scale + scale // 2), int(row1 * scale + scale // 2))
                    pt2 = (int(col2 * scale + scale // 2), int(row2 * scale + scale // 2))
                    
                    cv2.line(grid_img, pt1, pt2, path_color, 2)
            else:
                # A* path - draw cells and connecting lines
                path_color = (0, 200, 255)  # Yellow-orange for A* path (BGR)
                for i, (row, col) in enumerate(path):
                    x1, y1 = col * scale, row * scale
                    x2, y2 = x1 + scale, y1 + scale
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), path_color, -1)
                
                # Draw lines connecting path points
                for i in range(len(path) - 1):
                    row1, col1 = path[i]
                    row2, col2 = path[i + 1]
                    
                    pt1 = (col1 * scale + scale // 2, row1 * scale + scale // 2)
                    pt2 = (col2 * scale + scale // 2, row2 * scale + scale // 2)
                    
                    cv2.line(grid_img, pt1, pt2, (0, 140, 255), 2)
            
            # Draw start point (cyan)
            if self.start_point:
                sr, sc = self.start_point
                cv2.circle(grid_img, (sc * scale + scale // 2, sr * scale + scale // 2), 
                          scale * 2, (255, 255, 0), -1)  # Cyan (BGR)
            
            # Draw end point (magenta)
            if self.end_point:
                er, ec = self.end_point
                cv2.circle(grid_img, (ec * scale + scale // 2, er * scale + scale // 2), 
                          scale * 2, (255, 0, 255), -1)  # Magenta (BGR)
            
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, grid_img)
            print(f"Path image saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving path image: {e}")

    def set_mode(self, mode):
        """Set interaction mode for setting start/end points"""
        self.click_mode = mode
        if mode == "start":
            self.update_status("Click on the image to set START point")
        elif mode == "end":
            self.update_status("Click on the image to set END point")
    
    def on_canvas_click(self, event):
        """Handle canvas click for setting start/end points"""
        if self.click_mode is None or self.grid_matrix is None:
            return
        
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if self.processed_image is not None:
            img_height, img_width = self.processed_image.shape[:2]
            
            scale = min(canvas_width / img_width, canvas_height / img_height)
            scaled_width = int(img_width * scale)
            scaled_height = int(img_height * scale)
            offset_x = (canvas_width - scaled_width) // 2
            offset_y = (canvas_height - scaled_height) // 2
            
            img_x = int((event.x - offset_x) / scale)
            img_y = int((event.y - offset_y) / scale)
            
            cell_width = img_width // self.grid_cols
            cell_height = img_height // self.grid_rows
            
            grid_col = img_x // cell_width
            grid_row = img_y // cell_height
            
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                if self.click_mode == "start":
                    self.start_point = (grid_row, grid_col)
                    self.update_status(f"Start point set: ({grid_row}, {grid_col})")
                elif self.click_mode == "end":
                    self.end_point = (grid_row, grid_col)
                    self.update_status(f"End point set: ({grid_row}, {grid_col})")
                
                self.click_mode = None
                self.redraw_with_points()
    
    def redraw_with_points(self):
        """Redraw image with start/end points and paths marked"""
        if self.processed_image is None:
            return
        
        img = self.processed_image.copy()
        height, width = img.shape[:2]
        cell_width = width // self.grid_cols
        cell_height = height // self.grid_rows
        
        # Draw original path (yellow-orange)
        if self.path:
            for i in range(len(self.path) - 1):
                row1, col1 = self.path[i]
                row2, col2 = self.path[i + 1]
                
                center1 = (col1 * cell_width + cell_width // 2,
                          row1 * cell_height + cell_height // 2)
                center2 = (col2 * cell_width + cell_width // 2,
                          row2 * cell_height + cell_height // 2)
                
                cv2.line(img, center1, center2, (255, 200, 0), 2)
        
        # Draw smoothed path (green)
        if self.smoothed_path is not None and len(self.smoothed_path) > 1:
            for i in range(len(self.smoothed_path) - 1):
                row1, col1 = self.smoothed_path[i]
                row2, col2 = self.smoothed_path[i + 1]
                
                x1 = int(col1 * cell_width + cell_width // 2)
                y1 = int(row1 * cell_height + cell_height // 2)
                x2 = int(col2 * cell_width + cell_width // 2)
                y2 = int(row2 * cell_height + cell_height // 2)
                
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw start point
        if self.start_point:
            row, col = self.start_point
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            cv2.circle(img, (center_x, center_y), 20, (0, 255, 255), -1)
            cv2.putText(img, "START", (center_x - 30, center_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw end point
        if self.end_point:
            row, col = self.end_point
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            cv2.circle(img, (center_x, center_y), 20, (255, 0, 255), -1)
            cv2.putText(img, "END", (center_x - 20, center_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Legend
        legend_y = 30
        cv2.putText(img, "A* Path", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        cv2.putText(img, "Smoothed Path", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.display_image(img, self.image_canvas)
    
    def run_astar(self):
        """Run A* pathfinding algorithm with obstacle inflation and curve smoothing"""
        if self.grid_matrix is None:
            messagebox.showwarning("Warning", "Please generate grid first!")
            return
        
        if self.start_point is None or self.end_point is None:
            messagebox.showwarning("Warning", "Please set both start and end points!")
            return
        
        try:
            self.clearance_radius = int(self.clearance_spinbox.get())
            
            # Create working grid (treat parking spots as navigable for pathfinding)
            working_grid = np.where(self.grid_matrix == 2, 0, self.grid_matrix)
            
            if self.clearance_radius > 0:
                self.inflated_grid = self.inflate_obstacles(working_grid, self.clearance_radius)
                self.update_status(f"Inflating obstacles with clearance: {self.clearance_radius}")
                # Save inflated grid image with parking spots
                self.save_inflated_grid_image(self.inflated_grid, self.grid_matrix, "26_inflated_grid.png")
            else:
                self.inflated_grid = working_grid
            
            path = self.astar(self.start_point, self.end_point)
            
            if path:
                self.path = path
                
                path_length = 0
                for i in range(len(path) - 1):
                    path_length += sqrt(
                        (path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2
                    )
                
                try:
                    self.smoothed_path = self.smooth_path_bspline(path, smoothing_factor=0.01, num_points=100)
                except Exception as e:
                    print(f"Smoothing failed: {e}")
                    self.smoothed_path = None
                
                # Save path images
                self.save_path_image(path, "27_astar_path.png")
                if self.smoothed_path is not None:
                    self.save_path_image(self.smoothed_path, "28_smoothed_path.png", is_smoothed=True)
                
                self.redraw_with_points()
                self.visualize_grid_with_path()
                
                info_text = f"A* Pathfinding Complete!\n"
                info_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                info_text += f"Start: {self.start_point}\n"
                info_text += f"End: {self.end_point}\n"
                info_text += f"Path nodes: {len(path)}\n"
                info_text += f"Path distance: {path_length:.2f} units\n"
                if self.clearance_radius > 0:
                    info_text += f"Clearance: {self.clearance_radius} cells\n"
                if self.smoothed_path is not None:
                    info_text += f"Smoothing: B-spline (100 points)\n"
                info_text += f"Movement: 8-directional (diagonal)\n\n"
                info_text += f"Click 'Simulate' to animate car!"
                
                self.update_status(f"Path found! {len(path)} nodes, {path_length:.2f} units")
                self.update_info(info_text)
            else:
                messagebox.showwarning("No Path", 
                    "No valid path found!\n\n"
                    "Try:\n"
                    "- Reducing clearance radius\n"
                    "- Choosing different start/end points\n"
                    "- Checking for blocked areas")
                self.update_status("✗ No path found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error running A*: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def inflate_obstacles(self, grid, clearance):
        """Inflate obstacles in the grid by a given clearance"""
        if clearance == 0:
            return grid.copy()
        structure = ndimage.generate_binary_structure(2, 2)
        inflated_grid = ndimage.binary_dilation(grid, structure=structure, iterations=clearance).astype(np.uint8)
        return inflated_grid
    
    def astar(self, start, goal):
        """A* pathfinding algorithm with diagonal movement support"""
        working_grid = self.inflated_grid if self.inflated_grid is not None else self.grid_matrix
        
        def heuristic(a, b):
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos):
            row, col = pos
            neighbors = []
            moves = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            
            for dr, dc in moves:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.grid_rows and 
                    0 <= new_col < self.grid_cols and
                    working_grid[new_row, new_col] == 0):
                    neighbors.append((new_row, new_col))
            return neighbors
        
        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            _, _, current = heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                move_cost = heuristic(current, neighbor)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    counter += 1
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        return None
    
    def smooth_path_bspline(self, path, smoothing_factor=0.05, num_points=150):
        """Smooth the path using B-spline interpolation"""
        if len(path) < 3:
            return np.array(path)
        
        try:
            path_array = np.array(path)
            x_coords = path_array[:, 0]
            y_coords = path_array[:, 1]
            
            distances = np.zeros(len(path))
            for i in range(1, len(path)):
                distances[i] = distances[i-1] + sqrt(
                    (x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2
                )
            
            if distances[-1] > 0:
                distances = distances / distances[-1]
            
            t_smooth = np.linspace(0, 1, num_points)
            
            spl_x = UnivariateSpline(distances, x_coords, s=smoothing_factor)
            spl_y = UnivariateSpline(distances, y_coords, s=smoothing_factor)
            
            x_smooth = spl_x(t_smooth)
            y_smooth = spl_y(t_smooth)
            
            smoothed_path = np.column_stack((x_smooth, y_smooth))
            return smoothed_path
        except Exception as e:
            print(f"Smoothing error: {e}")
            return np.array(path)
    
    def visualize_grid_with_path(self):
        """Visualize grid matrix with path highlighted"""
        if self.grid_matrix is None:
            return
        
        self.grid_canvas.delete("all")
        cell_size = 5
        
        path_cells = set(self.path) if self.path else set()
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                if (row, col) == self.start_point:
                    color = "#00ffff"  # Cyan for start
                elif (row, col) == self.end_point:
                    color = "#ff00ff"  # Magenta for end
                elif (row, col) in path_cells:
                    color = "#ffff00"  # Yellow for path
                elif self.grid_matrix[row, col] == 1:
                    color = "#ff3333"  # Red for obstacles (YOLO)
                elif self.grid_matrix[row, col] == 2:
                    color = "#33ff33"  # Green for empty parking spots
                else:
                    color = "#ffffff"  # White for navigable
                
                self.grid_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="#cccccc"
                )
        
        self.grid_canvas.configure(
            scrollregion=(0, 0, self.grid_cols * cell_size, self.grid_rows * cell_size)
        )
    
    def clear_path(self):
        """Clear start, end points and path"""
        self.stop_simulation()  # Stop any running simulation
        self.start_point = None
        self.end_point = None
        self.path = None
        self.smoothed_path = None
        self.inflated_grid = None
        
        if self.grid_matrix is not None:
            self.visualize_grid_matrix()
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.image_canvas)
        
        self.update_status("Path cleared")
    
    def start_simulation(self):
        """Start car simulation along the smoothed path in fullscreen window"""
        if self.path is None:
            messagebox.showwarning("Warning", "Please run A* pathfinding first!")
            return
        
        # Always use smoothed path
        if self.smoothed_path is not None and len(self.smoothed_path) > 1:
            self.simulation_path = self.smoothed_path
        else:
            self.simulation_path = [(float(p[0]), float(p[1])) for p in self.path]
        
        if len(self.simulation_path) < 2:
            messagebox.showwarning("Warning", "Path too short for simulation!")
            return
        
        self.simulation_running = True
        self.simulation_index = 0
        
        # Calculate initial heading
        if len(self.simulation_path) > 1:
            dx = self.simulation_path[1][1] - self.simulation_path[0][1]
            dy = self.simulation_path[1][0] - self.simulation_path[0][0]
            self.car_heading = np.degrees(np.arctan2(dx, -dy))
        
        # Create fullscreen simulation window
        self.create_simulation_window()
        
        # Update button states
        self.sim_start_btn.config(state=tk.DISABLED)
        
        self.update_status("Simulation started (fullscreen)...")
        self.animate_car()
    
    def create_simulation_window(self):
        """Create a fullscreen window for simulation"""
        # Close existing window if any
        if self.simulation_window is not None:
            try:
                self.simulation_window.destroy()
            except:
                pass
        
        # Create new toplevel window
        self.simulation_window = tk.Toplevel(self.root)
        self.simulation_window.title("Parking Simulation")
        
        # Make it fullscreen
        self.simulation_window.attributes('-fullscreen', True)
        self.simulation_window.configure(bg='black')
        
        # Bind Escape key to close
        self.simulation_window.bind('<Escape>', lambda e: self.close_simulation_window())
        
        # Handle window close button
        self.simulation_window.protocol("WM_DELETE_WINDOW", self.close_simulation_window)
        
        # Create canvas for simulation
        self.simulation_canvas = tk.Canvas(
            self.simulation_window,
            bg='black',
            highlightthickness=0
        )
        self.simulation_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add instruction label
        instruction_label = ttk.Label(
            self.simulation_window,
            text="Press ESC to exit fullscreen",
            font=("Arial", 12),
            foreground="white",
            background="black"
        )
        instruction_label.place(relx=0.5, y=30, anchor=tk.CENTER)
        
        # Add close button (X) in top-right corner
        close_btn = tk.Button(
            self.simulation_window,
            text="✕",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#cc0000",
            activeforeground="white",
            activebackground="#ff0000",
            bd=0,
            padx=15,
            pady=5,
            cursor="hand2",
            command=self.close_simulation_window
        )
        close_btn.place(relx=1.0, x=-20, y=20, anchor=tk.NE)
        
        # Wait for window to be ready
        self.simulation_window.update()
    
    def close_simulation_window(self):
        """Close the simulation window and stop simulation"""
        self.simulation_running = False
        self.simulation_index = 0
        self.simulation_path = None
        
        if self.simulation_window is not None:
            try:
                self.simulation_window.destroy()
            except:
                pass
            self.simulation_window = None
            self.simulation_canvas = None
        
        # Update button states
        self.sim_start_btn.config(state=tk.NORMAL)
        
        # Redraw main canvas without car
        if self.path:
            self.redraw_with_points()
        
        self.update_status("Simulation window closed")
    
    def stop_simulation(self):
        """Stop the simulation and close fullscreen window"""
        self.simulation_running = False
        self.simulation_index = 0
        self.simulation_path = None
        
        # Close fullscreen window if open
        if self.simulation_window is not None:
            try:
                self.simulation_window.destroy()
            except:
                pass
            self.simulation_window = None
            self.simulation_canvas = None
        
        # Update button states
        self.sim_start_btn.config(state=tk.NORMAL)
        
        # Redraw without car
        if self.path:
            self.redraw_with_points()
    
    def animate_car(self):
        """Animate the car along the path"""
        if not self.simulation_running:
            return
        
        if self.simulation_path is None or self.simulation_index >= len(self.simulation_path):
            # Simulation complete
            self.simulation_running = False
            self.sim_start_btn.config(state=tk.NORMAL)
            
            self.update_status("Simulation complete! Press X to close window.")
            self.update_info(
                f"Simulation Complete!\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Path type: smoothed\n"
                f"Total waypoints: {len(self.simulation_path)}\n"
                f"Start: {self.start_point}\n"
                f"End: {self.end_point}\n\n"
                f"The car successfully navigated\n"
                f"from start to destination!"
            )
            return
        
        # Get current position
        current_pos = self.simulation_path[self.simulation_index]
        
        # Calculate heading based on next position
        if self.simulation_index < len(self.simulation_path) - 1:
            next_pos = self.simulation_path[self.simulation_index + 1]
            dx = next_pos[1] - current_pos[1]
            dy = next_pos[0] - current_pos[0]
            if abs(dx) > 0.001 or abs(dy) > 0.001:  # Avoid division issues
                target_heading = np.degrees(np.arctan2(dx, -dy))
                # Smooth heading transition
                heading_diff = target_heading - self.car_heading
                # Normalize to -180 to 180
                while heading_diff > 180:
                    heading_diff -= 360
                while heading_diff < -180:
                    heading_diff += 360
                self.car_heading += heading_diff * 0.3  # Smooth turn
        
        # Draw the scene with car
        self.draw_scene_with_car(current_pos)
        
        # Move to next position
        self.simulation_index += 1
        
        # Calculate delay based on speed setting
        speed = int(self.sim_speed_spinbox.get())
        delay = max(10, 200 - speed * 2)  # 10ms to 200ms delay
        
        # Update progress
        progress = (self.simulation_index / len(self.simulation_path)) * 100
        self.update_status(f"Simulating... {progress:.1f}% complete")
        
        # Schedule next frame
        self.root.after(delay, self.animate_car)
    
    def draw_scene_with_car(self, car_pos):
        """Draw the image with path and car at current position"""
        if self.processed_image is None:
            return
        
        img = self.processed_image.copy()
        height, width = img.shape[:2]
        cell_width = width // self.grid_cols
        cell_height = height // self.grid_rows
        
        # Draw original path (yellow-orange, dimmed)
        if self.path:
            for i in range(len(self.path) - 1):
                row1, col1 = self.path[i]
                row2, col2 = self.path[i + 1]
                
                center1 = (col1 * cell_width + cell_width // 2,
                          row1 * cell_height + cell_height // 2)
                center2 = (col2 * cell_width + cell_width // 2,
                          row2 * cell_height + cell_height // 2)
                
                cv2.line(img, center1, center2, (180, 140, 0), 2)
        
        # Draw smoothed path (green)
        if self.smoothed_path is not None and len(self.smoothed_path) > 1:
            for i in range(len(self.smoothed_path) - 1):
                row1, col1 = self.smoothed_path[i]
                row2, col2 = self.smoothed_path[i + 1]
                
                x1 = int(col1 * cell_width + cell_width // 2)
                y1 = int(row1 * cell_height + cell_height // 2)
                x2 = int(col2 * cell_width + cell_width // 2)
                y2 = int(row2 * cell_height + cell_height // 2)
                
                cv2.line(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        
        # Draw start point
        if self.start_point:
            row, col = self.start_point
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            cv2.circle(img, (center_x, center_y), 15, (0, 255, 255), -1)
            cv2.putText(img, "S", (center_x - 6, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw end point
        if self.end_point:
            row, col = self.end_point
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), -1)
            cv2.putText(img, "E", (center_x - 6, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw car at current position
        car_row, car_col = car_pos
        car_x = int(car_col * cell_width + cell_width // 2)
        car_y = int(car_row * cell_height + cell_height // 2)
        
        # Car dimensions based on parking spot sizes
        if self.parking_spots and len(self.parking_spots) > 0:
            # Calculate average parking spot dimensions
            avg_spot_width = sum(s['bounding_rect']['width'] for s in self.parking_spots) / len(self.parking_spots)
            avg_spot_height = sum(s['bounding_rect']['height'] for s in self.parking_spots) / len(self.parking_spots)
            
            # Car should fit inside parking spot (smaller for better visibility)
            spot_min = min(avg_spot_width, avg_spot_height)
            spot_max = max(avg_spot_width, avg_spot_height)
            
            car_width = int(spot_min * 0.80)
            car_length = int(spot_max * 0.70)
        else:
            # Fallback: scale based on cell size
            car_length = max(30, cell_height * 2)
            car_width = max(18, cell_width)
        
        # Create car shape (rectangle with direction indicator)
        angle_rad = np.radians(self.car_heading)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Car body corners (relative to center)
        half_len = car_length // 2
        half_wid = car_width // 2
        
        corners = [
            (-half_wid, -half_len),  # rear left
            (half_wid, -half_len),   # rear right
            (half_wid, half_len),    # front right
            (-half_wid, half_len),   # front left
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for cx, cy in corners:
            rx = int(car_x + cx * cos_a - cy * sin_a)
            ry = int(car_y + cx * sin_a + cy * cos_a)
            rotated_corners.append((rx, ry))
        
        # Draw car body (simple rectangle)
        pts = np.array(rotated_corners, dtype=np.int32)
        cv2.fillPoly(img, [pts], (30, 144, 255))  # Dodger blue
        cv2.polylines(img, [pts], True, (0, 0, 139), 2)  # Dark blue border
        
        # Simulation info overlay (commented out)
        # progress = (self.simulation_index / len(self.simulation_path)) * 100
        # cv2.rectangle(img, (5, 60), (250, 130), (0, 0, 0), -1)
        # cv2.rectangle(img, (5, 60), (250, 130), (255, 255, 255), 2)
        # cv2.putText(img, f"Progress: {progress:.1f}%", (15, 85),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(img, f"Heading: {self.car_heading:.1f} deg", (15, 115),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display on fullscreen simulation canvas if available, otherwise main canvas
        if self.simulation_canvas is not None:
            self.display_image(img, self.simulation_canvas)
        else:
            self.display_image(img, self.image_canvas)
    
    def display_image(self, img, canvas):
        """Display image on canvas with proper scaling"""
        try:
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 600
            
            img_height, img_width = img.shape[:2]
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            resized = cv2.resize(img, (new_width, new_height))
            
            pil_img = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(pil_img)
            
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=photo,
                anchor=tk.CENTER
            )
            
            canvas.image = photo
            
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
    
    def update_info(self, message):
        """Update info text area"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, message)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ParkingGridConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()


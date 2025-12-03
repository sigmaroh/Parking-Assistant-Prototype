"""
Parking Lot to Grid Converter for A* Pathfinding
Converts parking lot images to numerical grid for navigation algorithms
Uses YOLOv8 for accurate object detection (cars, trees, walls, etc.)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
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
        self.root.title("New Window")
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
        
        # Navigation
        self.start_point = None
        self.end_point = None
        self.path = None
        self.smoothed_path = None
        self.inflated_grid = None
        self.clearance_radius = 1
        
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
            text="Parking Project",
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.TOP, pady=10)
        
        # Button Frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.TOP, pady=5)
        
        ttk.Button(
            button_frame,
            text="Load Image",
            command=self.load_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="2. Detect Objects (YOLO)",
            command=self.detect_objects_yolo
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Generate Grid",
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
        
        # Grid Configuration Frame
        config_frame = ttk.LabelFrame(control_frame, text="Grid Configuration", padding="10")
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
        
        ttk.Label(config_frame, text="Legend:").pack(side=tk.LEFT, padx=20)
        ttk.Label(config_frame, text="⬜ 0 = Navigable", foreground="green").pack(side=tk.LEFT, padx=5)
        ttk.Label(config_frame, text="⬛ 1 = Obstacle (Car/Tree/Wall)", foreground="red").pack(side=tk.LEFT, padx=5)
        
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
        info_frame = ttk.LabelFrame(status_frame, text="Grid Information", padding="5")
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
                # Load YOLOv8 model (will download on first use)
                self.yolo_model = YOLO('models/yolo11n.pt')
                self.root.after(0, lambda: self.update_status("YOLO model loaded successfully! Ready to detect objects."))
                self.root.after(0, lambda: self.update_info(
                    "YOLO Model Ready!\n"
                    "Click '1. Load Image' to begin.\n"
                    "Then use '2. Detect Objects (YOLO)' to identify obstacles."
                ))
            except ImportError as e:
                error_msg = (
                    "⚠ ultralytics package not installed!\n"
                    "Run: pip install ultralytics"
                )
                self.root.after(0, lambda: self.update_status(error_msg))
                self.root.after(0, lambda: messagebox.showerror(
                    "YOLO Not Installed",
                    "Please install ultralytics:\n\npip install ultralytics\n\nThen restart the application."
                ))
                print(f"YOLO import error: {e}")
            except Exception as e:
                error_msg = f"⚠ YOLO model load failed: {str(e)}"
                self.root.after(0, lambda: self.update_status(error_msg))
                self.root.after(0, lambda: messagebox.showerror(
                    "YOLO Load Error",
                    f"Error loading YOLO model:\n\n{str(e)}\n\nCheck your internet connection and try again."
                ))
                print(f"YOLO load error: {e}")
                import traceback
                traceback.print_exc()
        
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
            
            # Display original image
            display_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(display_img, self.image_canvas)
            self.processed_image = display_img.copy()
            
            self.update_status(f"✓ Loaded image: {file_path.split('/')[-1]}")
            self.update_info(f"Image Size: {self.original_image.shape[1]}x{self.original_image.shape[0]} pixels\n"
                           f"Next: Run YOLO object detection to identify obstacles")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def detect_objects_yolo(self):
        """Detect objects (cars, trees, walls, etc.) using YOLO"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if self.yolo_model is None:
            messagebox.showwarning("Warning", "YOLO model not loaded yet. Please wait...")
            return
        
        try:
            self.update_status("Running YOLO object detection...")
            
            # Update confidence from spinbox
            self.detection_confidence = float(self.confidence_spinbox.get())
            
            # Run YOLO detection
            results = self.yolo_model(self.original_image, conf=self.detection_confidence, verbose=False)
            
            # Get detections
            self.detected_objects = []
            
            # COCO classes we consider as obstacles
            # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 
            # 9: traffic light, 10: fire hydrant, 11: stop sign, 
            # 13: bench, 14: bird, 15: cat, 16: dog, etc.
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
            
            # Draw detections on image
            detected_image = self.original_image.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[class_id]
                    
                    # Store detection
                    self.detected_objects.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name,
                        'is_obstacle': class_id in obstacle_classes
                    })
                    
                    # Draw bounding box
                    color = (0, 0, 255) if class_id in obstacle_classes else (0, 255, 0)
                    cv2.rectangle(detected_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(detected_image, 
                                (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)),
                                color, -1)
                    cv2.putText(detected_image, label, 
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display result
            self.processed_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.processed_image, self.image_canvas)
            
            # Count obstacles
            obstacle_count = sum(1 for obj in self.detected_objects if obj['is_obstacle'])
            other_count = len(self.detected_objects) - obstacle_count
            
            self.update_status(f"✓ Detected {len(self.detected_objects)} objects: {obstacle_count} obstacles")
            
            # Create detailed info
            info_text = f"YOLO Detection Results:\n"
            info_text += f"Total objects: {len(self.detected_objects)}\n"
            info_text += f"Obstacles (red): {obstacle_count}\n"
            info_text += f"Other objects (green): {other_count}\n\n"
            
            # List detected obstacles
            info_text += "Detected Obstacles:\n"
            obstacle_summary = {}
            for obj in self.detected_objects:
                if obj['is_obstacle']:
                    name = obj['class_name']
                    obstacle_summary[name] = obstacle_summary.get(name, 0) + 1
            
            for name, count in obstacle_summary.items():
                info_text += f"  • {name}: {count}\n"
            
            info_text += "\nNext: Apply Bird's Eye View (optional) → Generate Grid"
            
            self.update_info(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting objects: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def apply_birds_eye_view(self):
        """Apply perspective transformation for top-down view"""
        if self.processed_image is None:
            if self.original_image is not None:
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            else:
                messagebox.showwarning("Warning", "Please load an image first!")
                return
        
        try:
            # Use processed image if available (with detections)
            source_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR) if self.processed_image is not None else self.original_image
            height, width = source_image.shape[:2]
            
            # Define source points (adjust these for your specific image)
            # These create a trapezoid that will be transformed to rectangle
            src_points = np.float32([
                [width * 0.15, height * 0.35],   # Top-left
                [width * 0.85, height * 0.35],   # Top-right
                [width * 0.98, height * 0.98],   # Bottom-right
                [width * 0.02, height * 0.98]    # Bottom-left
            ])
            
            # Destination points (rectangle for bird's eye view)
            dst_points = np.float32([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ])
            
            # Get transformation matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transformation
            warped = cv2.warpPerspective(source_image, matrix, (width, height))
            
            # Convert to RGB and store
            self.processed_image = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            
            # Transform detected object bounding boxes if they exist
            if self.detected_objects:
                transformed_objects = []
                for obj in self.detected_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # Transform corners of bounding box
                    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, matrix)
                    
                    # Get new bounding box
                    transformed_corners = transformed_corners.reshape(-1, 2)
                    new_x1 = int(np.min(transformed_corners[:, 0]))
                    new_y1 = int(np.min(transformed_corners[:, 1]))
                    new_x2 = int(np.max(transformed_corners[:, 0]))
                    new_y2 = int(np.max(transformed_corners[:, 1]))
                    
                    # Create new object with transformed bbox
                    new_obj = obj.copy()
                    new_obj['bbox'] = [new_x1, new_y1, new_x2, new_y2]
                    transformed_objects.append(new_obj)
                
                self.detected_objects = transformed_objects
            
            # Show transformation points on original
            original_with_points = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()
            for i, point in enumerate(src_points):
                cv2.circle(original_with_points, tuple(point.astype(int)), 10, (255, 0, 0), -1)
                cv2.putText(original_with_points, str(i+1), tuple(point.astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            self.display_image(original_with_points, self.image_canvas)
            
            self.update_status("✓ Bird's eye view transformation applied")
            info_msg = "Perspective transformed to top-down view.\n"
            info_msg += "Red points show transformation anchors.\n"
            if self.detected_objects:
                info_msg += f"Transformed {len(self.detected_objects)} object bounding boxes.\n"
            info_msg += "Next: Generate Grid"
            self.update_info(info_msg)
            
            # Show warped image after 2 seconds
            self.root.after(2000, lambda: self.display_image(self.processed_image, self.image_canvas))
            
        except Exception as e:
            messagebox.showerror("Error", f"Error applying bird's eye view: {str(e)}")
    
    def generate_grid(self):
        """Generate navigable grid matrix using detected objects"""
        if self.processed_image is None:
            if self.original_image is not None:
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            else:
                messagebox.showwarning("Warning", "Please load an image first!")
                return
        
        try:
            # Update grid dimensions from spinboxes
            self.grid_rows = int(self.rows_spinbox.get())
            self.grid_cols = int(self.cols_spinbox.get())
            
            height, width = self.processed_image.shape[:2]
            
            # Calculate cell dimensions
            cell_height = height // self.grid_rows
            cell_width = width // self.grid_cols
            
            # Initialize grid matrix (0 = navigable, 1 = obstacle)
            self.grid_matrix = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
            
            # If we have detected objects, use them to mark obstacles
            if self.detected_objects:
                self.update_status("Generating grid using detected objects...")
                
                # Mark cells containing detected obstacles
                for obj in self.detected_objects:
                    if not obj['is_obstacle']:
                        continue  # Skip non-obstacle objects
                    
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # Clamp coordinates to image bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    # Find which grid cells this object occupies
                    start_col = x1 // cell_width
                    end_col = x2 // cell_width
                    start_row = y1 // cell_height
                    end_row = y2 // cell_height
                    
                    # Mark all cells covered by this object as obstacles
                    for row in range(start_row, min(end_row + 1, self.grid_rows)):
                        for col in range(start_col, min(end_col + 1, self.grid_cols)):
                            # Check if significant portion of cell is covered
                            cell_x1 = col * cell_width
                            cell_y1 = row * cell_height
                            cell_x2 = cell_x1 + cell_width
                            cell_y2 = cell_y1 + cell_height
                            
                            # Calculate overlap
                            overlap_x1 = max(x1, cell_x1)
                            overlap_y1 = max(y1, cell_y1)
                            overlap_x2 = min(x2, cell_x2)
                            overlap_y2 = min(y2, cell_y2)
                            
                            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                cell_area = cell_width * cell_height
                                
                                # If overlap is more than 20% of cell, mark as obstacle
                                if overlap_area > cell_area * 0.2:
                                    self.grid_matrix[row, col] = 1
            
            else:
                # Fallback: Use brightness-based classification if no objects detected
                messagebox.showinfo("Info", 
                    "No objects detected. Using brightness-based grid generation.\n"
                    "For better results, run YOLO detection first!")
                
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                threshold = 120
                
                for row in range(self.grid_rows):
                    for col in range(self.grid_cols):
                        y1 = row * cell_height
                        y2 = y1 + cell_height
                        x1 = col * cell_width
                        x2 = x1 + cell_width
                        
                        cell = gray[y1:y2, x1:x2]
                        mean_brightness = np.mean(cell)
                        
                        if mean_brightness < threshold:
                            self.grid_matrix[row, col] = 1
            
            # Visualize grid on image
            grid_image = self.processed_image.copy()
            
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    y1 = row * cell_height
                    x1 = col * cell_width
                    y2 = y1 + cell_height
                    x2 = x1 + cell_width
                    
                    # Color code cells
                    if self.grid_matrix[row, col] == 1:
                        # Obstacle - red overlay
                        overlay = grid_image[y1:y2, x1:x2].copy()
                        overlay[:, :] = [255, 0, 0]
                        grid_image[y1:y2, x1:x2] = cv2.addWeighted(
                            grid_image[y1:y2, x1:x2], 0.6, overlay, 0.4, 0
                        )
                    else:
                        # Navigable - green overlay
                        overlay = grid_image[y1:y2, x1:x2].copy()
                        overlay[:, :] = [0, 255, 0]
                        grid_image[y1:y2, x1:x2] = cv2.addWeighted(
                            grid_image[y1:y2, x1:x2], 0.8, overlay, 0.2, 0
                        )
                    
                    # Draw grid lines
                    cv2.rectangle(grid_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    
                    # Add cell coordinates
                    cv2.putText(grid_image, f"{row},{col}", 
                              (x1 + 2, y1 + 12), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.3, (255, 255, 255), 1)
            
            self.display_image(grid_image, self.image_canvas)
            self.visualize_grid_matrix()
            
            # Calculate statistics
            total_cells = self.grid_rows * self.grid_cols
            obstacle_cells = np.sum(self.grid_matrix)
            navigable_cells = total_cells - obstacle_cells
            
            self.update_status(f"✓ Grid generated: {self.grid_rows}x{self.grid_cols} = {total_cells} cells")
            
            info_text = f"Grid Matrix Generated:\n"
            info_text += f"Size: {self.grid_rows} rows × {self.grid_cols} columns\n"
            info_text += f"Navigable cells (0): {navigable_cells} ({navigable_cells/total_cells*100:.1f}%)\n"
            info_text += f"Obstacle cells (1): {obstacle_cells} ({obstacle_cells/total_cells*100:.1f}%)\n"
            
            if self.detected_objects:
                obstacle_objs = [obj for obj in self.detected_objects if obj['is_obstacle']]
                info_text += f"Based on {len(obstacle_objs)} detected obstacles\n"
            
            info_text += "\nReady for A* pathfinding!"
            
            self.update_info(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating grid: {str(e)}")
    
    def visualize_grid_matrix(self):
        """Visualize the grid matrix as a 2D array"""
        if self.grid_matrix is None:
            return
        
        # Clear canvas
        self.grid_canvas.delete("all")
        
        # Cell size for visualization
        cell_size = 5
        
        # Draw grid
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Color based on cell value
                if self.grid_matrix[row, col] == 1:
                    color = "#000000"  # Black for obstacles
                    text_color = "white"
                else:
                    color = "#44ff44"  # Green for navigable
                    text_color = "black"
                
                # Draw cell
                self.grid_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="gray"
                )
                
                # Draw value
                self.grid_canvas.create_text(
                    x1 + cell_size // 2,
                    y1 + cell_size // 2,
                    #text=str(self.grid_matrix[row, col]),
                    text='',
                    fill=text_color,
                    font=("Courier", 8, "bold")
                )
        
        # Update scroll region
        self.grid_canvas.configure(
            scrollregion=(0, 0, self.grid_cols * cell_size, self.grid_rows * cell_size)
        )
    
    def export_grid(self):
        """Export grid matrix to file"""
        if self.grid_matrix is None:
            messagebox.showwarning("Warning", "Please generate grid first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Grid Matrix",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("NumPy files", "*.npy"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                # Export as JSON
                grid_data = {
                    'grid': self.grid_matrix.tolist(),
                    'rows': self.grid_rows,
                    'cols': self.grid_cols,
                    'legend': {
                        '0': 'navigable',
                        '1': 'obstacle'
                    }
                }
                with open(file_path, 'w') as f:
                    json.dump(grid_data, f, indent=2)
            
            elif file_path.endswith('.npy'):
                # Export as NumPy binary
                np.save(file_path, self.grid_matrix)
            
            else:
                # Export as text file
                with open(file_path, 'w') as f:
                    f.write(f"# Grid Matrix: {self.grid_rows}x{self.grid_cols}\n")
                    f.write(f"# 0 = navigable, 1 = obstacle\n\n")
                    for row in self.grid_matrix:
                        f.write(' '.join(map(str, row)) + '\n')
            
            messagebox.showinfo("Success", f"Grid exported successfully!\n{file_path}")
            self.update_status(f"✓ Grid exported to {file_path.split('/')[-1]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting grid: {str(e)}")
    
    def export_grid_image(self):
        """Generate and export grid visualization as an image"""
        if self.grid_matrix is None:
            messagebox.showwarning("Warning", "Please generate grid first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Grid Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.update_status("Generating grid image...")
            
            # Create a high-resolution grid image
            cell_size = 20  # Larger cells for better visibility
            img_width = self.grid_cols * cell_size
            img_height = self.grid_rows * cell_size
            
            # Create image with white background
            grid_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
            
            # Draw grid cells
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    x1 = col * cell_size
                    y1 = row * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    
                    # Color based on cell value
                    if self.grid_matrix[row, col] == 1:
                        # Obstacle - black
                        color = (0, 0, 0)
                    else:
                        # Navigable - bright green
                        color = (68, 255, 68)
                    
                    # Fill cell
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, -1)
                    
                    # Draw grid lines
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
            
            # Add path if exists
            if self.path and self.start_point and self.end_point:
                # Draw path
                for i in range(len(self.path) - 1):
                    row1, col1 = self.path[i]
                    row2, col2 = self.path[i + 1]
                    
                    center1 = (col1 * cell_size + cell_size // 2,
                              row1 * cell_size + cell_size // 2)
                    center2 = (col2 * cell_size + cell_size // 2,
                              row2 * cell_size + cell_size // 2)
                    
                    cv2.line(grid_img, center1, center2, (255, 255, 0), 2)
                
                # Draw start point
                row, col = self.start_point
                center = (col * cell_size + cell_size // 2,
                         row * cell_size + cell_size // 2)
                cv2.circle(grid_img, center, cell_size // 3, (0, 255, 255), -1)
                cv2.putText(grid_img, "S", (center[0] - 5, center[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw end point
                row, col = self.end_point
                center = (col * cell_size + cell_size // 2,
                         row * cell_size + cell_size // 2)
                cv2.circle(grid_img, center, cell_size // 3, (255, 0, 255), -1)
                cv2.putText(grid_img, "E", (center[0] - 5, center[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Convert BGR to RGB for saving
            grid_img_rgb = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
            
            # Save image
            cv2.imwrite(file_path, grid_img)
            
            messagebox.showinfo("Success", f"Grid image exported successfully!\n{file_path}")
            self.update_status(f"✓ Grid image exported: {file_path.split('/')[-1]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting grid image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def set_mode(self, mode):
        """Set interaction mode for setting start/end points"""
        self.click_mode = mode
        if mode == "start":
            self.update_status("Click on grid to set START point")
        elif mode == "end":
            self.update_status("Click on grid to set END point")
    
    def on_canvas_click(self, event):
        """Handle canvas click for setting start/end points"""
        if self.click_mode is None or self.grid_matrix is None:
            return
        
        # Get canvas dimensions and image dimensions
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if self.processed_image is not None:
            img_height, img_width = self.processed_image.shape[:2]
            
            # Calculate scale
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            # Calculate image position on canvas
            scaled_width = int(img_width * scale)
            scaled_height = int(img_height * scale)
            offset_x = (canvas_width - scaled_width) // 2
            offset_y = (canvas_height - scaled_height) // 2
            
            # Convert click coordinates to image coordinates
            img_x = int((event.x - offset_x) / scale)
            img_y = int((event.y - offset_y) / scale)
            
            # Convert to grid coordinates
            cell_width = img_width // self.grid_cols
            cell_height = img_height // self.grid_rows
            
            grid_col = img_x // cell_width
            grid_row = img_y // cell_height
            
            # Check bounds
            if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                if self.click_mode == "start":
                    self.start_point = (grid_row, grid_col)
                    self.update_status(f"✓ Start point set: ({grid_row}, {grid_col})")
                elif self.click_mode == "end":
                    self.end_point = (grid_row, grid_col)
                    self.update_status(f"✓ End point set: ({grid_row}, {grid_col})")
                
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
        
        # Draw original path if exists (in blue)
        if self.path:
            for i in range(len(self.path) - 1):
                row1, col1 = self.path[i]
                row2, col2 = self.path[i + 1]
                
                center1 = (col1 * cell_width + cell_width // 2,
                          row1 * cell_height + cell_height // 2)
                center2 = (col2 * cell_width + cell_width // 2,
                          row2 * cell_height + cell_height // 2)
                
                cv2.line(img, center1, center2, (255, 200, 0), 2)  # Yellow-orange for A* path
        
        # Draw smoothed path if exists (in green)
        if self.smoothed_path is not None and len(self.smoothed_path) > 1:
            for i in range(len(self.smoothed_path) - 1):
                row1, col1 = self.smoothed_path[i]
                row2, col2 = self.smoothed_path[i + 1]
                
                # Convert to pixel coordinates
                x1 = int(col1 * cell_width + cell_width // 2)
                y1 = int(row1 * cell_height + cell_height // 2)
                x2 = int(col2 * cell_width + cell_width // 2)
                y2 = int(row2 * cell_height + cell_height // 2)
                
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for smoothed path
        
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
        
        # Add legend
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
            # Get clearance radius
            self.clearance_radius = int(self.clearance_spinbox.get())
            
            # Inflate obstacles if clearance > 0
            if self.clearance_radius > 0:
                self.inflated_grid = self.inflate_obstacles(self.grid_matrix, self.clearance_radius)
                self.update_status(f"Inflating obstacles with clearance radius: {self.clearance_radius}")
            else:
                self.inflated_grid = None
            
            # Run A* pathfinding
            path = self.astar(self.start_point, self.end_point, use_inflated=(self.clearance_radius > 0))
            
            if path:
                self.path = path
                
                # Calculate path length (actual distance)
                path_length = 0
                for i in range(len(path) - 1):
                    path_length += sqrt(
                        (path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2
                    )
                
                # Apply curve smoothing
                try:
                    self.smoothed_path = self.smooth_path_bspline(path, smoothing_factor=0.01, num_points=100)
                except Exception as e:
                    print(f"Smoothing failed: {e}")
                    self.smoothed_path = None
                
                self.redraw_with_points()
                self.visualize_grid_with_path()
                
                info_text = f"A* Pathfinding Complete!\n"
                info_text += f"Start: {self.start_point}\n"
                info_text += f"End: {self.end_point}\n"
                info_text += f"Path nodes: {len(path)}\n"
                info_text += f"Path distance: {path_length:.2f} units\n"
                if self.clearance_radius > 0:
                    info_text += f"Clearance: {self.clearance_radius} cells\n"
                if self.smoothed_path is not None:
                    info_text += f"Smoothing: B-spline (100 points)\n"
                info_text += f"Movement: 8-directional (diagonal)"
                
                self.update_status(f"✓ Path found! {len(path)} nodes, {path_length:.2f} units")
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
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        inflated_grid = ndimage.binary_dilation(grid, structure=structure, iterations=clearance).astype(np.uint8)
        return inflated_grid
    
    def astar(self, start, goal, use_inflated=True):
        """
        A* pathfinding algorithm with diagonal movement support
        
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            use_inflated: Whether to use inflated grid for safety clearance
        
        Returns:
            List of positions or None if no path found
        """
        
        # Use inflated grid if enabled
        working_grid = self.inflated_grid if (use_inflated and self.inflated_grid is not None) else self.grid_matrix
        
        def heuristic(a, b):
            # Euclidean distance for better diagonal movement
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos):
            row, col = pos
            neighbors = []
            # 8-directional movement (including diagonals)
            moves = [
                (-1, 0), (1, 0), (0, -1), (0, 1),     # Cardinal directions
                (-1, -1), (-1, 1), (1, -1), (1, 1)     # Diagonal directions
            ]
            
            for dr, dc in moves:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.grid_rows and 
                    0 <= new_col < self.grid_cols and
                    working_grid[new_row, new_col] == 0):  # Navigable
                    neighbors.append((new_row, new_col))
            return neighbors
        
        # Initialize data structures
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
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate distance (1 for cardinal, sqrt(2) for diagonal)
                move_cost = heuristic(current, neighbor)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    counter += 1
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        return None  # No path found
    
    def smooth_path_bspline(self, path, smoothing_factor=0.1, num_points=100):
        """
        Smooth the path using B-spline interpolation
        
        Args:
            path: List of (row, col) coordinates
            smoothing_factor: Smoothing factor (0 = no smoothing, higher = more smoothing)
            num_points: Number of points in the smoothed path
        
        Returns:
            Numpy array of smoothed path coordinates
        """
        if len(path) < 3:
            return np.array(path)
        
        try:
            # Convert path to numpy array
            path_array = np.array(path)
            x_coords = path_array[:, 0]
            y_coords = path_array[:, 1]
            
            # Create parameter array (cumulative distance along path)
            distances = np.zeros(len(path))
            for i in range(1, len(path)):
                distances[i] = distances[i-1] + sqrt(
                    (x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2
                )
            
            # Normalize distances to [0, 1]
            if distances[-1] > 0:
                distances = distances / distances[-1]
            
            # Create new parameter array for smooth curve
            t_smooth = np.linspace(0, 1, num_points)
            
            # Apply B-spline smoothing
            spl_x = UnivariateSpline(distances, x_coords, s=smoothing_factor)
            spl_y = UnivariateSpline(distances, y_coords, s=smoothing_factor)
            
            # Evaluate smoothed coordinates
            x_smooth = spl_x(t_smooth)
            y_smooth = spl_y(t_smooth)
            
            # Combine into path array
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
        
        # Create set of path cells for quick lookup
        path_cells = set(self.path) if self.path else set()
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Determine color
                if (row, col) == self.start_point:
                    color = "#00ffff"  # Cyan for start
                    text_color = "black"
                    text = "S"
                elif (row, col) == self.end_point:
                    color = "#ff00ff"  # Magenta for end
                    text_color = "black"
                    text = "E"
                elif (row, col) in path_cells:
                    color = "#ffff00"  # Yellow for path
                    text_color = "black"
                    text = "-"
                elif self.grid_matrix[row, col] == 1:
                    color = "#000000"  # Black for obstacles
                    text_color = "white"
                    text = ""#1
                else:
                    color = "#44ff44"  # Green for navigable
                    text_color = "black"
                    text = ""#0
                
                # Draw cell
                self.grid_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="gray"
                )
                
                # Draw text
                self.grid_canvas.create_text(
                    x1 + cell_size // 2,
                    y1 + cell_size // 2,
                    text=text,
                    fill=text_color,
                    font=("Courier", 10, "bold")
                )
        
        self.grid_canvas.configure(
            scrollregion=(0, 0, self.grid_cols * cell_size, self.grid_rows * cell_size)
        )
    
    def clear_path(self):
        """Clear start, end points and path"""
        self.start_point = None
        self.end_point = None
        self.path = None
        self.smoothed_path = None
        self.inflated_grid = None
        
        if self.grid_matrix is not None:
            self.visualize_grid_matrix()
            
            # Regenerate grid visualization
            if self.processed_image is not None:
                self.generate_grid()
        
        self.update_status("Path cleared")
    
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

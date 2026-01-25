"""
Parking Lot Path Planner - Main Application

A comprehensive parking assistant that detects parking spots, plans optimal paths,
and simulates vehicle navigation using MPC control.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
from typing import Optional, List, Dict, Tuple

from config import (UIConfig, DetectionConfig, GridConfig, ColorConfig, 
                   OutputConfig, SimulationConfig)
from detection import ParkingSpotDetector, ObstacleDetector
from grid_generator import GridGenerator
from path_planner import PathPlanner
from simulator import CarSimulator, CarRenderer


class ParkingPlannerUI:
    """Main UI application for parking lot path planning."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(UIConfig.WINDOW_TITLE)
        self.root.geometry(UIConfig.WINDOW_GEOMETRY)
        
        # Initialize components
        self.parking_detector = ParkingSpotDetector()
        self.obstacle_detector = ObstacleDetector()
        self.grid_generator = GridGenerator()
        self.path_planner = PathPlanner()
        self.car_simulator = CarSimulator()
        
        # State variables
        self.original_image = None
        self.processed_image = None
        self.parking_spots = []
        self.detected_objects = []
        
        # Navigation state
        self.start_point = None
        self.end_point = None
        self.click_mode = None
        self.inflated_grid = None
        self.clearance_radius = GridConfig.DEFAULT_CLEARANCE_RADIUS
        self.current_cell_size = UIConfig.GRID_CELL_DISPLAY_SIZE  # Store actual cell size used
        
        # Simulation state
        self.simulation_running = False
        self.simulation_path = None
        self.simulation_window = None
        self.simulation_canvas = None
        
        # Create UI
        self.create_widgets()
        
        # Load YOLO model in background
        self.load_yolo_model_async()
    
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Control panel
        self.create_control_panel()
        
        # Configuration panel
        self.create_config_panel()
        
        # Legend
        self.create_legend()
        
        # Main content area
        self.create_content_area()
        
        # Status panel
        self.create_status_panel()
    
    def create_control_panel(self):
        """Create top control panel with action buttons."""
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Title
        ttk.Label(
            control_frame,
            text=UIConfig.WINDOW_TITLE,
            font=("Arial", 16, "bold")
        ).pack(side=tk.TOP, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.TOP, pady=5)
        
        # Pipeline buttons
        buttons = [
            ("1. Load Image", self.load_image),
            ("2. Detect Parking Spots", self.detect_parking_spots),
            ("3. Detect Obstacles (YOLO)", self.detect_obstacles),
            ("4. Generate Grid", self.generate_grid),
        ]
        
        for text, command in buttons:
            ttk.Button(button_frame, text=text, command=command).pack(
                side=tk.LEFT, padx=5)
        
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Path planning buttons
        path_buttons = [
            ("Set Start Point", lambda: self.set_click_mode("start")),
            ("Set End Point", lambda: self.set_click_mode("end")),
            ("Run A* Pathfinding", self.run_pathfinding),
            ("Clear Path", self.clear_path),
        ]
        
        for text, command in path_buttons:
            ttk.Button(button_frame, text=text, command=command).pack(
                side=tk.LEFT, padx=5)
        
        # Simulation button
        self.sim_btn = ttk.Button(
            button_frame, text="▶ Simulate", command=self.start_simulation
        )
        self.sim_btn.pack(side=tk.LEFT, padx=5)
    
    def create_config_panel(self):
        """Create configuration panel with settings."""
        control_frame = self.root.winfo_children()[0]
        config_frame = ttk.LabelFrame(control_frame, text="Configuration", padding="10")
        config_frame.pack(side=tk.TOP, pady=10, fill=tk.X)
        
        # Grid configuration
        self.add_spinbox(config_frame, "Grid Rows:", 5, 100, 
                         UIConfig.DEFAULT_GRID_ROWS, 'rows_spinbox', state='readonly')
        self.add_spinbox(config_frame, "Grid Columns:", 5, 100, 
                         UIConfig.DEFAULT_GRID_COLS, 'cols_spinbox', state='readonly')
        self.add_spinbox(config_frame, "YOLO Confidence:", 0.1, 1.0, 
                         DetectionConfig.DEFAULT_CONFIDENCE, 'confidence_spinbox', 
                         increment=0.05)
        self.add_spinbox(config_frame, "Clearance:", 0, 10, 
                         self.clearance_radius, 'clearance_spinbox')
        self.add_spinbox(config_frame, "Cell Size (px):", 0, 50, 
                         UIConfig.GRID_CELL_DISPLAY_SIZE, 'cell_size_spinbox')
        
        ttk.Separator(config_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.add_spinbox(config_frame, "Sim Speed:", 1, 100, 
                         UIConfig.DEFAULT_SIM_SPEED, 'sim_speed_spinbox')
    
    def add_spinbox(self, parent, label: str, from_: float, to: float, 
                     default: float, attr_name: str, increment: float = 1, state: str = 'normal'):
        """Helper to add a labeled spinbox."""
        ttk.Label(parent, text=label).pack(side=tk.LEFT, padx=5)
        spinbox = ttk.Spinbox(parent, from_=from_, to=to, width=8, increment=increment, state=state)
        spinbox.set(default)
        spinbox.pack(side=tk.LEFT, padx=5)
        setattr(self, attr_name, spinbox)
    
    def create_legend(self):
        """Create legend frame."""
        control_frame = self.root.winfo_children()[0]
        legend_frame = ttk.Frame(control_frame)
        legend_frame.pack(side=tk.TOP, pady=5, fill=tk.X)
        
        ttk.Label(legend_frame, text="Grid Legend:").pack(side=tk.LEFT, padx=10)
        
        legends = [
            ("⬜ 0 = Navigable (A* path)", "gray"),
            ("🔴 1 = Obstacle (YOLO)", "red"),
            ("🟢 2 = Empty Parking Spot", "green"),
        ]
        
        for text, color in legends:
            ttk.Label(legend_frame, text=text, foreground=color).pack(
                side=tk.LEFT, padx=5)
    
    def create_content_area(self):
        """Create main content area with image and grid canvases."""
        content_frame = ttk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image view
        left_panel = ttk.LabelFrame(content_frame, text="Parking Lot View", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_panel, bg="gray20")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Right panel - Grid view
        right_panel = ttk.LabelFrame(content_frame, text="Grid for A* Algorithm", 
                                     padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        grid_scroll_frame = ttk.Frame(right_panel)
        grid_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.grid_canvas = tk.Canvas(grid_scroll_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(grid_scroll_frame, orient=tk.VERTICAL, 
                                    command=self.grid_canvas.yview)
        h_scrollbar = ttk.Scrollbar(grid_scroll_frame, orient=tk.HORIZONTAL, 
                                    command=self.grid_canvas.xview)
        
        self.grid_canvas.configure(yscrollcommand=v_scrollbar.set, 
                                   xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def create_status_panel(self):
        """Create bottom status panel."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.status_label = ttk.Label(
            status_frame,
            text="Ready. Load a parking lot image to begin.",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)
        
        info_frame = ttk.LabelFrame(status_frame, text="Information", padding="5")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=5, wrap=tk.WORD, 
                                font=("Courier", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
    
    # ========== Model Loading ==========
    
    def load_yolo_model_async(self):
        """Load YOLO model in background thread."""
        def load_model():
            self.update_status("Loading YOLO model (downloading if first time)...")
            
            success = self.obstacle_detector.load_model()
            
            if success:
                self.root.after(0, lambda: self.update_status(
                    "YOLO model loaded! Ready to detect obstacles."))
                self.root.after(0, lambda: self.update_info(
                    "Workflow:\n"
                    "1. Load Image\n"
                    "2. Detect Parking Spots (Traditional CV) → Green\n"
                    "3. Detect Obstacles (YOLO) → Red\n"
                    "4. Generate Grid\n"
                    "5. Set Start/End → Run A*\n"
                    "6. Simulate\n\n"
                    "Click '1. Load Image' to begin."
                ))
            else:
                self.root.after(0, lambda: self.update_status(
                    "⚠ YOLO not available. Parking detection still works."))
        
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    # ========== Image Loading ==========
    
    def load_image(self):
        """Load parking lot image."""
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
            
            # Reset state
            self.parking_spots = []
            self.detected_objects = []
            self.grid_generator.grid_matrix = None
            self.path_planner.path = None
            self.path_planner.smoothed_path = None
            self.stop_simulation()
            
            # Display image
            display_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(display_img, self.image_canvas)
            self.processed_image = display_img.copy()
            
            # Calculate grid dimensions
            height, width = self.original_image.shape[:2]
            cell_size = int(self.cell_size_spinbox.get())
            expected_cols = width // cell_size
            expected_rows = height // cell_size
            
            # Update spinboxes with calculated grid dimensions
            self.rows_spinbox.delete(0, tk.END)
            self.rows_spinbox.insert(0, str(expected_rows))
            self.cols_spinbox.delete(0, tk.END)
            self.cols_spinbox.insert(0, str(expected_cols))
            
            self.update_status(f"Loaded image: {os.path.basename(file_path)}")
            self.update_info(
                f"Image Size: {width}x{height} pixels\n"
                f"Cell Size: {cell_size}px → Grid: {expected_cols}x{expected_rows}\n\n"
                f"Next Step: 2. Detect Parking Spots\n"
                f"Then: 3. Detect Obstacles (YOLO)"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    # ========== Detection ==========
    
    def detect_parking_spots(self):
        """Detect parking spots using traditional CV."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            self.update_status("Detecting parking spots (Traditional CV)...")
            
            # Save original image
            os.makedirs(OutputConfig.PROCESS_IMAGES_FOLDER, exist_ok=True)
            cv2.imwrite(
                os.path.join(OutputConfig.PROCESS_IMAGES_FOLDER, "1_original.png"),
                self.original_image
            )
            
            # Run detection
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.parking_spots = self.parking_detector.detect(image, save_steps=True)
            
            # Draw results
            result_image = self.draw_parking_spots(image, self.parking_spots)
            self.processed_image = result_image
            self.display_image(result_image, self.image_canvas)
            
            # Update status
            empty_count = len([s for s in self.parking_spots 
                             if not s.get('is_occupied', False)])
            occupied_count = len(self.parking_spots) - empty_count
            
            self.update_status(
                f"Detected {empty_count} empty, {occupied_count} occupied parking spots"
            )
            
            info_text = (
                f"Parking Spot Detection (Traditional CV):\n"
                f"Total parking spots found: {len(self.parking_spots)}\n"
                f"  🟢 Empty spots: {empty_count}\n"
                f"  🔴 Occupied spots: {occupied_count}\n\n"
                f"Next: Detect Obstacles (YOLO)"
            )
            self.update_info(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting parking spots: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def detect_obstacles(self):
        """Detect obstacles using YOLO."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if self.obstacle_detector.model is None:
            messagebox.showwarning("Warning", 
                                 "YOLO model not loaded yet. Please wait or try again.")
            return
        
        try:
            self.update_status("Detecting obstacles (YOLO)...")
            
            # Run detection (saves YOLO detection image)
            confidence = float(self.confidence_spinbox.get())
            self.detected_objects = self.obstacle_detector.detect(
                self.original_image, confidence, save_result=True
            )
            
            # Save combined detection image (parking spots + obstacles)
            self.obstacle_detector.save_combined_detection(
                self.original_image, self.parking_spots, self.detected_objects
            )
            
            # Draw results
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            result_image = self.draw_combined_detections(image)
            
            self.processed_image = result_image
            self.display_image(result_image, self.image_canvas)
            
            # Update status
            obstacle_count = sum(1 for obj in self.detected_objects 
                               if obj['is_obstacle'])
            empty_count = len([s for s in self.parking_spots 
                             if not s.get('is_occupied', False)])
            occupied_count = len(self.parking_spots) - empty_count
            
            self.update_status(
                f"YOLO: {obstacle_count} obstacles | "
                f"Parking: {empty_count} empty, {occupied_count} occupied"
            )
            
            obstacle_summary = {}
            for obj in self.detected_objects:
                if obj['is_obstacle']:
                    name = obj['class_name']
                    obstacle_summary[name] = obstacle_summary.get(name, 0) + 1
            
            info_text = (
                f"Combined Detection Results:\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🟢 Empty Parking Spots: {empty_count}\n"
                f"🔴 Occupied Parking Spots: {occupied_count}\n"
                f"🔴 Obstacles (YOLO): {obstacle_count}\n\n"
            )
            
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
    
    # ========== Grid Generation ==========
    
    def generate_grid(self):
        self.clear_path()
        """Generate navigable grid matrix."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            height, width = self.original_image.shape[:2]
            cell_size = int(self.cell_size_spinbox.get())
            
            # Store the cell size for consistent display
            self.current_cell_size = cell_size
            
            self.update_status("Generating grid from detections...")
            
            # Generate grid
            grid_matrix = self.grid_generator.generate(
                (height, width),
                cell_size,
                self.parking_spots,
                self.detected_objects
            )
            
            # Update spinboxes
            self.rows_spinbox.delete(0, tk.END)
            self.rows_spinbox.insert(0, str(self.grid_generator.grid_rows))
            self.cols_spinbox.delete(0, tk.END)
            self.cols_spinbox.insert(0, str(self.grid_generator.grid_cols))
            
            # Visualize
            grid_image = self.create_grid_overlay(
                cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB),
                grid_matrix, cell_size
            )
            self.display_image(grid_image, self.image_canvas)
            self.visualize_grid_matrix(grid_matrix)
            
            # Save grid
            output_path = os.path.join(OutputConfig.PROCESS_IMAGES_FOLDER,
                                      OutputConfig.GRID_MATRIX_FILENAME)
            self.grid_generator.save_grid_image(output_path, grid_matrix)
            
            # Statistics
            total_cells = self.grid_generator.grid_rows * self.grid_generator.grid_cols
            obstacle_cells = np.sum(grid_matrix == GridConfig.GRID_OBSTACLE)
            parking_cells = np.sum(grid_matrix == GridConfig.GRID_PARKING_SPOT)
            navigable_cells = np.sum(grid_matrix == GridConfig.GRID_NAVIGABLE)
            
            self.update_status(
                f"Grid generated: {self.grid_generator.grid_cols}x"
                f"{self.grid_generator.grid_rows} from {width}x{height} image"
            )
            
            info_text = (
                f"Grid Matrix Generated:\n"
                f"Image: {width}x{height} pixels\n"
                f"Cell Size: {cell_size}px\n"
                f"Grid: {self.grid_generator.grid_cols} cols × "
                f"{self.grid_generator.grid_rows} rows = {total_cells} cells\n\n"
                f"Navigable (0): {navigable_cells} "
                f"({navigable_cells/total_cells*100:.1f}%)\n"
                f"Obstacles (1): {obstacle_cells} "
                f"({obstacle_cells/total_cells*100:.1f}%)\n"
                f"Empty spots (2): {parking_cells} "
                f"({parking_cells/total_cells*100:.1f}%)\n\n"
                f"Ready for A* pathfinding!"
            )
            self.update_info(info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating grid: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # ========== Path Planning ==========
    
    def set_click_mode(self, mode: str):
        """Set interaction mode for setting start/end points."""
        self.click_mode = mode
        if mode == "start":
            self.update_status("Click on the image to set START point")
        elif mode == "end":
            self.update_status("Click on the image to set END point")
    
    def on_canvas_click(self, event):
        """Handle canvas click for setting start/end points."""
        if self.click_mode is None or self.grid_generator.grid_matrix is None:
            return
        
        grid_row, grid_col = self.get_grid_coords_from_click(event)
        
        if grid_row is not None and grid_col is not None:
            if self.click_mode == "start":
                self.start_point = (grid_row, grid_col)
                self.update_status(f"Start point set: ({grid_row}, {grid_col})")
            elif self.click_mode == "end":
                self.end_point = (grid_row, grid_col)
                self.update_status(f"End point set: ({grid_row}, {grid_col})")
            
            self.click_mode = None
            self.redraw_with_points()
    
    def get_grid_coords_from_click(self, event) -> Tuple[Optional[int], Optional[int]]:
        """Convert canvas click coordinates to grid coordinates."""
        if self.processed_image is None:
            return None, None
        
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        img_height, img_width = self.processed_image.shape[:2]
        
        scale = min(canvas_width / img_width, canvas_height / img_height)
        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)
        offset_x = (canvas_width - scaled_width) // 2
        offset_y = (canvas_height - scaled_height) // 2
        
        img_x = int((event.x - offset_x) / scale)
        img_y = int((event.y - offset_y) / scale)
        
        cell_width = img_width // self.grid_generator.grid_cols
        cell_height = img_height // self.grid_generator.grid_rows
        
        grid_col = img_x // cell_width
        grid_row = img_y // cell_height
        
        if (0 <= grid_row < self.grid_generator.grid_rows and 
            0 <= grid_col < self.grid_generator.grid_cols):
            return grid_row, grid_col
        
        return None, None
    
    def run_pathfinding(self):
        """Run A* pathfinding algorithm."""
        if self.grid_generator.grid_matrix is None:
            messagebox.showwarning("Warning", "Please generate grid first!")
            return
        
        if self.start_point is None or self.end_point is None:
            messagebox.showwarning("Warning", "Please set both start and end points!")
            return
        
        try:
            # Calculate clearance
            clearance = int(self.clearance_spinbox.get())
            # If clearance > 0, use it as-is without updating
            if clearance >0:
                self.clearance_radius = clearance
            else:
                # Only calculate auto clearance if value is negative
                clearance = self.calculate_auto_clearance()
                self.clearance_radius = clearance
                self.clearance_spinbox.set(clearance)
            
            # Create working grid
            working_grid = np.where(
                self.grid_generator.grid_matrix == GridConfig.GRID_PARKING_SPOT,
                GridConfig.GRID_NAVIGABLE,
                self.grid_generator.grid_matrix
            )
            
            # Inflate obstacles
            if self.clearance_radius > 0:
                self.inflated_grid = self.grid_generator.inflate_obstacles(
                    self.clearance_radius
                )
                output_path = os.path.join(OutputConfig.PROCESS_IMAGES_FOLDER,
                                          OutputConfig.INFLATED_GRID_FILENAME)
                # Use the special method that preserves parking spots in green
                self.grid_generator.save_inflated_grid_image(
                    output_path, 
                    self.inflated_grid,
                    self.grid_generator.grid_matrix
                )
            else:
                self.inflated_grid = working_grid
            
            # Find path
            path = self.path_planner.find_path(
                self.inflated_grid, self.start_point, self.end_point
            )
            
            if path:
                
                # Smooth path using B-spline interpolation
                # meters_per_pixel = 0.05 * UIConfig.GRID_CELL_DISPLAY_SIZE
                # path_len_m = self.path_planner.path_length_pixels(path, meters_per_pixel)
                # ctrl_spacing_m = 0.5

                # num_points = int(path_len_m / ctrl_spacing_m)

                # # Safety limits (very important)
                # num_points = max(6, min(num_points, 12))
                smoothed_path = self.path_planner.smooth_path(path, num_points=100)

                path_length = self.path_planner.calculate_path_length(path)
                
                # Save path images
                self.save_path_images(path, smoothed_path)
                
                # Redraw and update UI
                self.redraw_with_points()
                self.visualize_grid_with_path(self.grid_generator.grid_matrix, path)
                
                info_text = (
                    f"A* Pathfinding Complete!\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Start: {self.start_point}\n"
                    f"End: {self.end_point}\n"
                    f"Path nodes: {len(path)}\n"
                    f"Path distance: {path_length:.2f} units\n"
                    f"Clearance: {self.clearance_radius} cells\n"
                    f"Smoothing: B-spline (100 points)\n"
                    f"Movement: 8-directional\n\n"
                    f"Click 'Simulate' to animate car!"
                )
                
                self.update_status(f"Path found! {len(path)} nodes, {path_length:.2f} units")
                self.update_info(info_text)
            else:
                messagebox.showwarning(
                    "No Path",
                    "No valid path found!\n\n"
                    "Try:\n"
                    "- Reducing clearance radius\n"
                    "- Choosing different start/end points\n"
                    "- Checking for blocked areas"
                )
                self.update_status("✗ No path found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error running A*: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def clear_path(self):
        """Clear start, end points and path."""
        self.stop_simulation()
        self.start_point = None
        self.end_point = None
        self.path_planner.path = None
        self.path_planner.smoothed_path = None
        self.inflated_grid = None
        
        if self.grid_generator.grid_matrix is not None:
            self.visualize_grid_matrix(self.grid_generator.grid_matrix)
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.image_canvas)
        
        self.update_status("Path cleared")
    
    # ========== Simulation ==========
    
    def start_simulation(self):
        """Start car simulation using MPC controller."""
        if self.path_planner.path is None:
            messagebox.showwarning("Warning", "Please run A* pathfinding first!")
            return
        
        # Use smoothed path if available
        if (self.path_planner.smoothed_path is not None and 
            len(self.path_planner.smoothed_path) > 1):
            self.simulation_path = self.path_planner.smoothed_path
        else:
            self.simulation_path = [(float(p[0]), float(p[1])) 
                                   for p in self.path_planner.path]
        
        if len(self.simulation_path) < 2:
            messagebox.showwarning("Warning", "Path too short for simulation!")
            return
        
        # Initialize simulator
        obstacles = self.get_grid_obstacles()
        self.car_simulator.initialize(
            self.simulation_path[0],
            self.simulation_path,
            obstacles
        )
        
        # Create simulation window
        self.create_simulation_window()
        
        self.simulation_running = True
        self.sim_btn.config(state=tk.DISABLED)
        
        self.update_status("Linear MPC Controller initialized. Starting simulation...")
        self.animate_simulation()
    
    def animate_simulation(self):
        """Animation loop for car simulation."""
        if not self.simulation_running:
            return
        
        # Check if reached goal
        dist_to_goal = self.car_simulator.distance_to_goal(self.simulation_path[-1])
        
        if dist_to_goal < 1.5:
            self.simulation_running = False
            self.sim_btn.config(state=tk.NORMAL)
            
            state = self.car_simulator.get_state()
            self.update_status("MPC Simulation complete! Press X to close window.")
            self.update_info(
                f"MPC Simulation Complete!\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Controller: Linear MPC Controller\n"
                f"Path type: smoothed\n"
                f"Total waypoints: {len(self.simulation_path)}\n"
                f"Final velocity: {state['velocity']:.2f} units/s\n"
                f"Final heading: {state['heading_deg']:.1f}°\n\n"
                f"The car successfully navigated\n"
                f"using MPC with bicycle kinematic model!"
            )
            return
        
        # Find closest point on path
        closest_idx = self.car_simulator.find_closest_path_index(self.simulation_path)
        
        # Get reference points for MPC
        reference_points = self.car_simulator.get_reference_points(
            self.simulation_path, closest_idx
        )
        
        # Perform MPC step
        speed = int(self.sim_speed_spinbox.get())
        success = self.car_simulator.step(reference_points)
        
        if not success:
            # Fallback to pure pursuit
            lookahead_idx = min(closest_idx + 3, len(self.simulation_path) - 1)
            self.car_simulator.step_pure_pursuit(
                self.simulation_path[lookahead_idx],
                SimulationConfig.SIMULATION_DT
            )
        
        # Draw scene
        state = self.car_simulator.get_state()
        current_pos = (state['y'], state['x'])  # (row, col)
        self.draw_simulation_scene(current_pos, state)
        
        # Update progress
        progress = (closest_idx / len(self.simulation_path)) * 100
        self.update_status(
            f"MPC Simulating... {progress:.1f}% | "
            f"Velocity: {state['velocity']:.2f} | "
            f"Steering: {np.degrees(state['delta']):.1f}°"
        )
        
        # Schedule next frame
        delay = max(10, 150 - speed * 1.5)
        self.root.after(int(delay), self.animate_simulation)
    
    def stop_simulation(self):
        """Stop the simulation."""
        self.simulation_running = False
        self.simulation_path = None
        
        if self.simulation_window is not None:
            try:
                self.simulation_window.destroy()
            except:
                pass
            self.simulation_window = None
            self.simulation_canvas = None
        
        self.sim_btn.config(state=tk.NORMAL)
        
        if self.path_planner.path:
            self.redraw_with_points()
    
    # ========== Drawing Methods ==========
    
    def draw_parking_spots(self, image: np.ndarray, 
                           parking_spots: List[Dict]) -> np.ndarray:
        """Draw parking spots on image."""
        result = image.copy()
        
        for spot in parking_spots:
            bbox = spot['bounding_rect']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            color = ColorConfig.EMPTY_SPOT_COLOR
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            box = np.array(spot['min_area_rect']['corner_points'], dtype=np.int32)
            cv2.polylines(result, [box], True, color, 2)
            
            cx, cy = map(int, spot['min_area_rect']['center'])
            cv2.putText(result, f"P{spot['spot_id']}", (cx-15, cy-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        return result
    
    def draw_combined_detections(self, image: np.ndarray) -> np.ndarray:
        """Draw parking spots and obstacles on image."""
        result = image.copy()
        
        # Draw parking spots
        for spot in self.parking_spots:
            bbox = spot['bounding_rect']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            color = ColorConfig.EMPTY_SPOT_COLOR
            
            overlay = result.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            cx, cy = map(int, spot['min_area_rect']['center'])
            cv2.putText(result, f"P{spot['spot_id']}", (cx-10, cy-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Draw obstacles
        for obj in self.detected_objects:
            if obj['is_obstacle']:
                x1, y1, x2, y2 = obj['bbox']
                color = ColorConfig.OBSTACLE_COLOR
                
                overlay = result.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
                
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(result, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def create_grid_overlay(self, image: np.ndarray, 
                            grid: np.ndarray, 
                            cell_size: int) -> np.ndarray:
        """Create grid visualization overlay on image."""
        result = image.copy()
        height, width = image.shape[:2]
        cell_height = height // self.grid_generator.grid_rows
        cell_width = width // self.grid_generator.grid_cols
        
        for row in range(self.grid_generator.grid_rows):
            for col in range(self.grid_generator.grid_cols):
                y1 = row * cell_height
                x1 = col * cell_width
                y2 = y1 + cell_height
                x2 = x1 + cell_width
                
                cell_val = grid[row, col]
                
                if cell_val == GridConfig.GRID_OBSTACLE:
                    color = (255, 50, 50)
                elif cell_val == GridConfig.GRID_PARKING_SPOT:
                    color = (50, 255, 50)
                else:
                    color = None
                
                if color:
                    overlay = result[y1:y2, x1:x2].copy()
                    overlay[:, :] = color
                    result[y1:y2, x1:x2] = cv2.addWeighted(
                        result[y1:y2, x1:x2], 0.5, overlay, 0.5, 0
                    )
                
                cv2.rectangle(result, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        return result
    
    def redraw_with_points(self):
        """Redraw image with start/end points and paths."""
        if self.processed_image is None:
            return
        
        img = self.processed_image.copy()
        height, width = img.shape[:2]
        cell_width = width // self.grid_generator.grid_cols
        cell_height = height // self.grid_generator.grid_rows
        
        # Draw A* path
        if self.path_planner.path:
            for i in range(len(self.path_planner.path) - 1):
                row1, col1 = self.path_planner.path[i]
                row2, col2 = self.path_planner.path[i + 1]
                
                center1 = (col1 * cell_width + cell_width // 2,
                          row1 * cell_height + cell_height // 2)
                center2 = (col2 * cell_width + cell_width // 2,
                          row2 * cell_height + cell_height // 2)
                
                cv2.line(img, center1, center2, ColorConfig.PATH_COLOR, 2)
        
        # Draw smoothed path
        if (self.path_planner.smoothed_path is not None and 
            len(self.path_planner.smoothed_path) > 1):
            for i in range(len(self.path_planner.smoothed_path) - 1):
                row1, col1 = self.path_planner.smoothed_path[i]
                row2, col2 = self.path_planner.smoothed_path[i + 1]
                
                x1 = int(col1 * cell_width + cell_width // 2)
                y1 = int(row1 * cell_height + cell_height // 2)
                x2 = int(col2 * cell_width + cell_width // 2)
                y2 = int(row2 * cell_height + cell_height // 2)
                
                cv2.line(img, (x1, y1), (x2, y2), ColorConfig.SMOOTHED_PATH_COLOR, 3)
        
        # Draw start point
        if self.start_point:
            row, col = self.start_point
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            cv2.circle(img, (center_x, center_y), 20, ColorConfig.START_POINT_COLOR, -1)
            cv2.putText(img, "START", (center_x - 30, center_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ColorConfig.START_POINT_COLOR, 2)
        
        # Draw end point
        if self.end_point:
            row, col = self.end_point
            center_x = col * cell_width + cell_width // 2
            center_y = row * cell_height + cell_height // 2
            cv2.circle(img, (center_x, center_y), 20, ColorConfig.END_POINT_COLOR, -1)
            cv2.putText(img, "END", (center_x - 20, center_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ColorConfig.END_POINT_COLOR, 2)
        
        self.display_image(img, self.image_canvas)
    
    def draw_simulation_scene(self, car_pos: Tuple[float, float], state: dict):
        """Draw the simulation scene with car."""
        if self.processed_image is None:
            return
        
        img = self.processed_image.copy()
        height, width = img.shape[:2]
        cell_width = width // self.grid_generator.grid_cols
        cell_height = height // self.grid_generator.grid_rows
        
        # Draw paths (dimmed)
        if self.path_planner.path:
            for i in range(len(self.path_planner.path) - 1):
                row1, col1 = self.path_planner.path[i]
                row2, col2 = self.path_planner.path[i + 1]
                
                center1 = (col1 * cell_width + cell_width // 2,
                          row1 * cell_height + cell_height // 2)
                center2 = (col2 * cell_width + cell_width // 2,
                          row2 * cell_height + cell_height // 2)
                
                cv2.line(img, center1, center2, (180, 140, 0), 2)
        
        # Draw smoothed path
        if (self.path_planner.smoothed_path is not None and 
            len(self.path_planner.smoothed_path) > 1):
            for i in range(len(self.path_planner.smoothed_path) - 1):
                row1, col1 = self.path_planner.smoothed_path[i]
                row2, col2 = self.path_planner.smoothed_path[i + 1]
                
                x1 = int(col1 * cell_width + cell_width // 2)
                y1 = int(row1 * cell_height + cell_height // 2)
                x2 = int(col2 * cell_width + cell_width // 2)
                y2 = int(row2 * cell_height + cell_height // 2)
                
                cv2.line(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        
        # Draw car
        self.draw_car(img, car_pos, state, cell_width, cell_height)
        
        # Display
        if self.simulation_canvas is not None:
            self.display_image(img, self.simulation_canvas)
        else:
            self.display_image(img, self.image_canvas)
    
    def draw_car(self, img: np.ndarray, pos: Tuple[float, float], 
                 state: dict, cell_width: int, cell_height: int):
        """Draw car on image."""
        car_row, car_col = pos
        car_x_px = int(car_col * cell_width + cell_width // 2)
        car_y_px = int(car_row * cell_height + cell_height // 2)
        
        # Get car dimensions
        car_width, car_length = CarRenderer.get_car_dimensions(
            self.parking_spots, cell_width, cell_height
        )
        
        psi = state['psi']
        delta = state['delta']
        visual_psi = psi - np.pi / 2  # Align front with movement direction
        
        # Car body
        half_len = car_length // 2
        half_wid = car_width // 2
        
        body_corners = [
            (car_x_px - half_wid, car_y_px - half_len),
            (car_x_px + half_wid, car_y_px - half_len),
            (car_x_px + half_wid, car_y_px + half_len),
            (car_x_px - half_wid, car_y_px + half_len),
        ]
        
        rotated_body = CarRenderer.rotate_points(body_corners, visual_psi, 
                                                (car_x_px, car_y_px))
        pts = np.array(rotated_body, dtype=np.int32)
        cv2.fillPoly(img, [pts], ColorConfig.CAR_BODY_COLOR)
        cv2.polylines(img, [pts], True, ColorConfig.CAR_BORDER_COLOR, 2)
        
        # Draw wheels (simplified)
        wheel_length = max(8, car_length // SimulationConfig.WHEEL_LENGTH_RATIO)
        wheel_width = max(4, car_width // SimulationConfig.WHEEL_WIDTH_RATIO)
        
        wheel_offsets = [
            (half_wid - wheel_width//2, half_len - wheel_length),
            (-half_wid + wheel_width//2, half_len - wheel_length),
            (half_wid - wheel_width//2, -half_len + wheel_length),
            (-half_wid + wheel_width//2, -half_len + wheel_length),
        ]
        
        for i, (wx, wy) in enumerate(wheel_offsets):
            cos_v = np.cos(visual_psi)
            sin_v = np.sin(visual_psi)
            wheel_x = car_x_px + wx * cos_v - wy * sin_v
            wheel_y = car_y_px + wx * sin_v + wy * cos_v
            
            w_half_len = wheel_length // 2
            w_half_wid = wheel_width // 2
            wheel_corners = [
                (wheel_x - w_half_wid, wheel_y - w_half_len),
                (wheel_x + w_half_wid, wheel_y - w_half_len),
                (wheel_x + w_half_wid, wheel_y + w_half_len),
                (wheel_x - w_half_wid, wheel_y + w_half_len),
            ]
            
            wheel_angle = visual_psi + (delta if i < 2 else 0)
            rotated_wheel = CarRenderer.rotate_points(wheel_corners, wheel_angle, 
                                                     (wheel_x, wheel_y))
            wheel_pts = np.array(rotated_wheel, dtype=np.int32)
            cv2.fillPoly(img, [wheel_pts], ColorConfig.WHEEL_COLOR)
        
        # Headlight
        front_offset = half_len - 3
        cos_v = np.cos(visual_psi)
        sin_v = np.sin(visual_psi)
        front_x = int(car_x_px - front_offset * sin_v)
        front_y = int(car_y_px + front_offset * cos_v)
        cv2.circle(img, (front_x, front_y), 4, ColorConfig.HEADLIGHT_COLOR, -1)
        
        # Info overlay
        cv2.rectangle(img, (5, 5), (280, 90), (0, 0, 0), -1)
        cv2.rectangle(img, (5, 5), (280, 90), (255, 255, 255), 1)
        cv2.putText(img, "MPC Controller Active", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"Heading: {state['heading_deg']:.1f} deg", (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Steering: {np.degrees(delta):.1f} deg", (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Velocity: {state['velocity']:.2f}", (15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # ========== Visualization Methods ==========
    
    def visualize_grid_matrix(self, grid: np.ndarray):
        """Visualize grid matrix in the grid canvas."""
        self.grid_canvas.delete("all")
        
        if UIConfig.FIT_GRID_TO_CANVAS:
            # MODE 1: Fit grid to canvas (no scrollbars)
            canvas_width = self.grid_canvas.winfo_width()
            canvas_height = self.grid_canvas.winfo_height()
            
            # Use default size if canvas not yet rendered
            if canvas_width <= 1:
                canvas_width = 400
            if canvas_height <= 1:
                canvas_height = 600
            
            # Calculate cell size to fit the grid in the canvas with some padding
            padding = 10
            available_width = canvas_width - padding * 2
            available_height = canvas_height - padding * 2
            
            cell_size_by_width = available_width / self.grid_generator.grid_cols
            cell_size_by_height = available_height / self.grid_generator.grid_rows
            cell_size = max(1, int(min(cell_size_by_width, cell_size_by_height)))
            
            # Calculate total grid size
            grid_width = self.grid_generator.grid_cols * cell_size
            grid_height = self.grid_generator.grid_rows * cell_size
            
            # Calculate offsets to center the grid
            offset_x = (canvas_width - grid_width) // 2
            offset_y = (canvas_height - grid_height) // 2
        else:
            # MODE 2: Show actual grid size (with scrollbars)
            cell_size = self.current_cell_size
            offset_x = 0
            offset_y = 0
        
        for row in range(self.grid_generator.grid_rows):
            for col in range(self.grid_generator.grid_cols):
                x1 = offset_x + col * cell_size
                y1 = offset_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                cell_val = grid[row, col]
                
                if cell_val == GridConfig.GRID_OBSTACLE:
                    color = ColorConfig.GRID_OBSTACLE_HEX
                elif cell_val == GridConfig.GRID_PARKING_SPOT:
                    color = ColorConfig.GRID_PARKING_HEX
                else:
                    color = ColorConfig.GRID_NAVIGABLE_HEX
                
                self.grid_canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="#cccccc"
                )
        
        # Configure scroll region if not fitting to canvas
        if not UIConfig.FIT_GRID_TO_CANVAS:
            self.grid_canvas.configure(
                scrollregion=(0, 0, 
                             self.grid_generator.grid_cols * cell_size,
                             self.grid_generator.grid_rows * cell_size)
            )
    
    def visualize_grid_with_path(self, grid: np.ndarray, path: List[Tuple[int, int]]):
        """Visualize grid with path highlighted."""
        self.grid_canvas.delete("all")
        
        if UIConfig.FIT_GRID_TO_CANVAS:
            # MODE 1: Fit grid to canvas (no scrollbars)
            canvas_width = self.grid_canvas.winfo_width()
            canvas_height = self.grid_canvas.winfo_height()
            
            # Use default size if canvas not yet rendered
            if canvas_width <= 1:
                canvas_width = 400
            if canvas_height <= 1:
                canvas_height = 600
            
            # Calculate cell size to fit the grid in the canvas with some padding
            padding = 10
            available_width = canvas_width - padding * 2
            available_height = canvas_height - padding * 2
            
            cell_size_by_width = available_width / self.grid_generator.grid_cols
            cell_size_by_height = available_height / self.grid_generator.grid_rows
            cell_size = max(1, int(min(cell_size_by_width, cell_size_by_height)))
            
            # Calculate total grid size
            grid_width = self.grid_generator.grid_cols * cell_size
            grid_height = self.grid_generator.grid_rows * cell_size
            
            # Calculate offsets to center the grid
            offset_x = (canvas_width - grid_width) // 2
            offset_y = (canvas_height - grid_height) // 2
        else:
            # MODE 2: Show actual grid size (with scrollbars)
            cell_size = self.current_cell_size
            offset_x = 0
            offset_y = 0
        
        path_cells = set(path)
        
        for row in range(self.grid_generator.grid_rows):
            for col in range(self.grid_generator.grid_cols):
                x1 = offset_x + col * cell_size
                y1 = offset_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                if (row, col) == self.start_point:
                    color = ColorConfig.GRID_START_HEX
                elif (row, col) == self.end_point:
                    color = ColorConfig.GRID_END_HEX
                elif (row, col) in path_cells:
                    color = ColorConfig.GRID_PATH_HEX
                elif grid[row, col] == GridConfig.GRID_OBSTACLE:
                    color = ColorConfig.GRID_OBSTACLE_HEX
                elif grid[row, col] == GridConfig.GRID_PARKING_SPOT:
                    color = ColorConfig.GRID_PARKING_HEX
                else:
                    color = ColorConfig.GRID_NAVIGABLE_HEX
                
                self.grid_canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="#cccccc"
                )
        
        # Configure scroll region if not fitting to canvas
        if not UIConfig.FIT_GRID_TO_CANVAS:
            self.grid_canvas.configure(
                scrollregion=(0, 0,
                             self.grid_generator.grid_cols * cell_size,
                             self.grid_generator.grid_rows * cell_size)
            )
    
    def display_image(self, img: np.ndarray, canvas: tk.Canvas):
        """Display image on canvas with proper scaling."""
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
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            
            canvas.image = photo
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    # ========== Utility Methods ==========
    
    def save_path_images(self, path: List[Tuple[int, int]], 
                        smoothed_path: Optional[np.ndarray]):
        """Save path visualization images."""
        try:
            output_folder = OutputConfig.PROCESS_IMAGES_FOLDER
            os.makedirs(output_folder, exist_ok=True)
            
            rows, cols = self.grid_generator.grid_rows, self.grid_generator.grid_cols
            scale = 5
            
            # Create base image from grid
            grid_img = np.zeros((rows * scale, cols * scale, 3), dtype=np.uint8)
            
            # Draw grid background
            for row in range(rows):
                for col in range(cols):
                    x1, y1 = col * scale, row * scale
                    x2, y2 = x1 + scale, y1 + scale
                    
                    if self.grid_generator.grid_matrix[row, col] == 1:
                        color = (0, 0, 255)  # Red for obstacles
                    elif self.grid_generator.grid_matrix[row, col] == 2:
                        color = (0, 255, 0)  # Green for parking spots
                    else:
                        color = (255, 255, 255)  # White for navigable
                    
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, -1)
            
            # Save A* path
            astar_img = grid_img.copy()
            path_color = (0, 200, 255)  # Yellow-orange for A* path (BGR)
            for i, (row, col) in enumerate(path):
                x1, y1 = col * scale, row * scale
                x2, y2 = x1 + scale, y1 + scale
                cv2.rectangle(astar_img, (x1, y1), (x2, y2), path_color, -1)
            
            # Draw lines connecting path points
            for i in range(len(path) - 1):
                row1, col1 = path[i]
                row2, col2 = path[i + 1]
                
                pt1 = (col1 * scale + scale // 2, row1 * scale + scale // 2)
                pt2 = (col2 * scale + scale // 2, row2 * scale + scale // 2)
                
                cv2.line(astar_img, pt1, pt2, (0, 140, 255), 2)
            
            # Draw start point (cyan)
            if self.start_point:
                sr, sc = self.start_point
                cv2.circle(astar_img, (sc * scale + scale // 2, sr * scale + scale // 2),
                          scale * 2, (255, 255, 0), -1)  # Cyan (BGR)
            
            # Draw end point (magenta)
            if self.end_point:
                er, ec = self.end_point
                cv2.circle(astar_img, (ec * scale + scale // 2, er * scale + scale // 2),
                          scale * 2, (255, 0, 255), -1)  # Magenta (BGR)
            
            filepath = os.path.join(output_folder, OutputConfig.ASTAR_PATH_FILENAME)
            cv2.imwrite(filepath, astar_img)
            print(f"Saved A* path image: {filepath}")
            
            # Save smoothed path
            if smoothed_path is not None and len(smoothed_path) > 1:
                smoothed_img = grid_img.copy()
                path_color = (0, 255, 255)  # Yellow for smoothed path (BGR)
                
                for i in range(len(smoothed_path) - 1):
                    row1, col1 = smoothed_path[i]
                    row2, col2 = smoothed_path[i + 1]
                    
                    pt1 = (int(col1 * scale + scale // 2), int(row1 * scale + scale // 2))
                    pt2 = (int(col2 * scale + scale // 2), int(row2 * scale + scale // 2))
                    
                    cv2.line(smoothed_img, pt1, pt2, path_color, 2)
                
                # Draw start point (cyan)
                if self.start_point:
                    sr, sc = self.start_point
                    cv2.circle(smoothed_img, (sc * scale + scale // 2, sr * scale + scale // 2),
                              scale * 2, (255, 255, 0), -1)
                
                # Draw end point (magenta)
                if self.end_point:
                    er, ec = self.end_point
                    cv2.circle(smoothed_img, (ec * scale + scale // 2, er * scale + scale // 2),
                              scale * 2, (255, 0, 255), -1)
                
                filepath = os.path.join(output_folder, OutputConfig.SMOOTHED_PATH_FILENAME)
                cv2.imwrite(filepath, smoothed_img)
                print(f"Saved smoothed path image: {filepath}")
            
        except Exception as e:
            print(f"Error saving path images: {e}")
    
    def calculate_auto_clearance(self) -> int:
        """Calculate automatic clearance based on car width."""
        if not self.parking_spots:
            return 2
        
        height, width = self.processed_image.shape[:2]
        cell_width = width // self.grid_generator.grid_cols
        
        avg_spot_width = sum(
            s['bounding_rect']['width'] for s in self.parking_spots
        ) / len(self.parking_spots)
        avg_spot_height = sum(
            s['bounding_rect']['height'] for s in self.parking_spots
        ) / len(self.parking_spots)
        
        spot_min = min(avg_spot_width, avg_spot_height)
        car_width_pixels = int(spot_min * 0.75)
        car_width_cells = car_width_pixels / cell_width
        
        half_car_width = car_width_cells / 2.0
        one_pixel_in_cells = 1.0 / cell_width
        auto_clearance = int(np.ceil(half_car_width + one_pixel_in_cells))
        
        return max(1, auto_clearance)
    
    def get_grid_obstacles(self) -> np.ndarray:
        """Extract obstacle positions for MPC environment."""
        obstacles = []
        
        if self.inflated_grid is not None:
            grid = self.inflated_grid
        elif self.grid_generator.grid_matrix is not None:
            grid = self.grid_generator.grid_matrix
        else:
            return np.array([[0, 0]])
        
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if grid[row, col] == GridConfig.GRID_OBSTACLE:
                    obstacles.append([col, row])  # (x, y)
        
        return np.array(obstacles) if obstacles else np.array([[0, 0]])
    
    def create_simulation_window(self):
        """Create fullscreen simulation window."""
        if self.simulation_window is not None:
            try:
                self.simulation_window.destroy()
            except:
                pass
        
        self.simulation_window = tk.Toplevel(self.root)
        self.simulation_window.title("Parking Simulation")
        self.simulation_window.attributes('-fullscreen', True)
        self.simulation_window.configure(bg='black')
        
        self.simulation_window.bind('<Escape>', 
                                   lambda e: self.stop_simulation())
        self.simulation_window.protocol("WM_DELETE_WINDOW", self.stop_simulation)
        
        self.simulation_canvas = tk.Canvas(
            self.simulation_window, bg='black', highlightthickness=0
        )
        self.simulation_canvas.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            self.simulation_window,
            text="Press ESC to exit fullscreen",
            font=("Arial", 12),
            foreground="white",
            background="black"
        ).place(relx=0.5, y=30, anchor=tk.CENTER)
        
        close_btn = tk.Button(
            self.simulation_window, text="✕", font=("Arial", 16, "bold"),
            fg="white", bg="#cc0000", activeforeground="white",
            activebackground="#ff0000", bd=0, padx=15, pady=5,
            cursor="hand2", command=self.stop_simulation
        )
        close_btn.place(relx=1.0, x=-20, y=20, anchor=tk.NE)
        
        self.simulation_window.update()
    
    def update_status(self, message: str):
        """Update status bar."""
        self.status_label.config(text=message)
    
    def update_info(self, message: str):
        """Update info text area."""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, message)


def main():
    """Main entry point."""
    root = tk.Tk()
    ParkingPlannerUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


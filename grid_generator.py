"""Grid generation module for path planning."""

import cv2
import numpy as np
import scipy.ndimage as ndimage
from typing import List, Dict, Tuple
from config import GridConfig


class GridGenerator:
    """Converts detection results into a navigable grid matrix."""
    
    def __init__(self):
        self.grid_matrix = None
        self.grid_rows = 0
        self.grid_cols = 0
    
    def generate(self, image_shape: Tuple[int, int], 
                 cell_size: int,
                 parking_spots: List[Dict] = None,
                 obstacles: List[Dict] = None) -> np.ndarray:
        """
        Generate grid matrix from image and detections.
        
        Args:
            image_shape: (height, width) of the original image
            cell_size: Size of each grid cell in pixels
            parking_spots: List of detected parking spots
            obstacles: List of detected obstacles
            
        Returns:
            Grid matrix where:
                0 = navigable space
                1 = obstacle
                2 = empty parking spot
        """
        height, width = image_shape
        
        self.grid_cols = width // cell_size
        self.grid_rows = height // cell_size
        
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Initialize with navigable space
        self.grid_matrix = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        # Mark empty parking spots
        if parking_spots:
            self.mark_parking_spots(parking_spots, cell_width, cell_height)
        
        # Mark obstacles (overwrites parking spots if overlapping)
        if obstacles:
            self.mark_obstacles(obstacles, cell_width, cell_height, width, height)
        
        return self.grid_matrix
    
    def mark_parking_spots(self, parking_spots: List[Dict], 
                           cell_width: int, cell_height: int):
        """Mark empty parking spots in the grid."""
        for spot in parking_spots:
            if spot.get('is_occupied', False):
                # Mark occupied spots as obstacles
                self.mark_rect_in_grid(
                    spot['bounding_rect'], 
                    cell_width, cell_height,
                    GridConfig.GRID_OBSTACLE
                )
            else:
                # Mark empty spots
                self.mark_rect_in_grid(
                    spot['bounding_rect'], 
                    cell_width, cell_height,
                    GridConfig.GRID_PARKING_SPOT
                )
    
    def mark_obstacles(self, obstacles: List[Dict], 
                       cell_width: int, cell_height: int,
                       img_width: int, img_height: int):
        """Mark detected obstacles in the grid."""
        for obj in obstacles:
            if not obj.get('is_obstacle', False):
                continue
            
            x1, y1, x2, y2 = obj['bbox']
            # Clamp to image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            bbox = {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
            self.mark_rect_in_grid(bbox, cell_width, cell_height, 
                                   GridConfig.GRID_OBSTACLE)
    
    def mark_rect_in_grid(self, bbox: Dict, cell_width: int, 
                          cell_height: int, value: int):
        """Mark a rectangular region in the grid."""
        x = bbox['x']
        y = bbox['y']
        w = bbox['width']
        h = bbox['height']
        
        start_col = max(0, x // cell_width)
        end_col = min(self.grid_cols - 1, (x + w) // cell_width)
        start_row = max(0, y // cell_height)
        end_row = min(self.grid_rows - 1, (y + h) // cell_height)
        
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                if self.cell_overlaps_rect(row, col, bbox, 
                                           cell_width, cell_height):
                    self.grid_matrix[row, col] = value
    
    def cell_overlaps_rect(self, row: int, col: int, bbox: Dict,
                           cell_width: int, cell_height: int) -> bool:
        """Check if a grid cell significantly overlaps with a rectangle."""
        cell_x1 = col * cell_width
        cell_y1 = row * cell_height
        cell_x2 = cell_x1 + cell_width
        cell_y2 = cell_y1 + cell_height
        
        x = bbox['x']
        y = bbox['y']
        w = bbox['width']
        h = bbox['height']
        
        overlap_x1 = max(x, cell_x1)
        overlap_y1 = max(y, cell_y1)
        overlap_x2 = min(x + w, cell_x2)
        overlap_y2 = min(y + h, cell_y2)
        
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            cell_area = cell_width * cell_height
            return overlap_area > cell_area * GridConfig.CELL_OVERLAP_THRESHOLD
        
        return False
    
    def inflate_obstacles(self, clearance: int) -> np.ndarray:
        """
        Inflate obstacles in the grid by a clearance radius.
        
        Args:
            clearance: Number of cells to inflate by
            
        Returns:
            Inflated grid matrix
        """
        if clearance <= 0 or self.grid_matrix is None:
            return self.grid_matrix.copy() if self.grid_matrix is not None else None
        
        # Create binary grid (treat parking spots as navigable)
        binary_grid = np.where(self.grid_matrix == GridConfig.GRID_PARKING_SPOT, 
                              GridConfig.GRID_NAVIGABLE, 
                              self.grid_matrix)
        
        # Apply binary dilation
        structure = ndimage.generate_binary_structure(2, 2)
        inflated = ndimage.binary_dilation(
            binary_grid, 
            structure=structure, 
            iterations=clearance
        ).astype(np.uint8)
        
        return inflated
    
    def save_grid_image(self, filepath: str, grid: np.ndarray = None, 
                       scale: int = 5):
        """Save grid matrix as an image."""
        if grid is None:
            grid = self.grid_matrix
        
        if grid is None:
            return
        
        rows, cols = grid.shape
        
        # Create color image
        grid_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        grid_img[grid == GridConfig.GRID_NAVIGABLE] = [255, 255, 255]  # White
        grid_img[grid == GridConfig.GRID_OBSTACLE] = [0, 0, 255]  # Red (BGR)
        grid_img[grid == GridConfig.GRID_PARKING_SPOT] = [0, 255, 0]  # Green (BGR)
        
        # Scale up for visibility
        grid_img_scaled = cv2.resize(grid_img, (cols * scale, rows * scale), 
                                    interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(filepath, grid_img_scaled)
    
    def save_inflated_grid_image(self, filepath: str, inflated_grid: np.ndarray, 
                                 original_grid: np.ndarray = None, scale: int = 5):
        """
        Save inflated grid image with parking spots preserved from original grid.
        
        Args:
            filepath: Path to save the image
            inflated_grid: Binary grid with inflated obstacles (0=navigable, 1=obstacle)
            original_grid: Original grid with parking spots (uses self.grid_matrix if None)
            scale: Scaling factor for better visibility
        """
        if original_grid is None:
            original_grid = self.grid_matrix
        
        if inflated_grid is None or original_grid is None:
            return
        
        rows, cols = inflated_grid.shape
        
        # Create color image with layered rendering
        grid_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        
        # Layer 1: Set all navigable areas from inflated grid (white)
        grid_img[inflated_grid == 0] = [255, 255, 255]  # White navigable
        
        # Layer 2: Draw inflated obstacles (red)
        grid_img[inflated_grid == 1] = [0, 0, 255]  # Red obstacles (BGR)
        
        # Layer 3: Draw parking spots from original grid ON TOP (green)
        # This ensures parking spots are ALWAYS visible, even if in inflated areas
        grid_img[original_grid == 2] = [0, 255, 0]  # Green parking spots (BGR)
        
        # Scale up for visibility
        grid_img_scaled = cv2.resize(grid_img, (cols * scale, rows * scale), 
                                    interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(filepath, grid_img_scaled)
        print(f"Saved inflated grid image with parking spots: {filepath}")


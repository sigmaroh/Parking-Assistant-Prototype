"""Path planning module using A* algorithm."""

import numpy as np
from heapq import heappush, heappop
from math import sqrt, hypot
from scipy.interpolate import UnivariateSpline
from typing import List, Tuple, Optional
from config import PathPlanningConfig


class PathPlanner:
    """A* path planning with path smoothing."""
    
    def __init__(self):
        self.path = None
        self.smoothed_path = None
    
    @staticmethod
    def calculate_distance(a: Tuple[int, int], b: Tuple[int, int], method: str = "euclidean") -> float:
        """
        Calculate distance between two points.
        
        Args:
            a: First point (row, col)
            b: Second point (row, col)
            method: Which heuristic to use ("euclidean", "manhattan", "chebyshev", "diagonal")
        Returns:
            Distance between a and b.
        """
        if method == "euclidean":
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        elif method == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        elif method == "chebyshev":
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        elif method == "diagonal":
            # Diagonal distance (with cost sqrt(2) per diagonal step)
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            D = 1
            D2 = sqrt(2)
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        else:
            # Default to euclidean if unknown method
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
    def path_length_pixels(self, path, meters_per_pixel):
        length = 0.0
        for i in range(1, len(path)):
            dx = (path[i][0] - path[i-1][0]) * meters_per_pixel
            dy = (path[i][1] - path[i-1][1]) * meters_per_pixel
            length += hypot(dx, dy)
        return length
    
    @staticmethod
    def get_neighbors(grid: np.ndarray, rows: int, cols: int, 
                      pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-directional)."""
        row, col = pos
        neighbors = []
        
        for dr, dc in PathPlanningConfig.MOVEMENT_DIRECTIONS:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 
                0 <= new_col < cols and
                grid[new_row, new_col] == 0):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def find_path(self, grid: np.ndarray, 
                  start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from start to goal using A* algorithm.
        
        Args:
            grid: Binary grid where 0 = navigable, 1 = obstacle
            start: Starting position (row, col)
            goal: Goal position (row, col)
            
        Returns:
            List of waypoints from start to goal, or None if no path exists
        """
        self.path = self.astar(grid, start, goal)
        return self.path
    
    def smooth_path(self, path: List[Tuple[int, int]] = None,
                   smoothing_factor: float = None,
                   num_points: int = None) -> Optional[np.ndarray]:
        """
        Smooth path using B-spline interpolation.
        
        Args:
            path: Path to smooth (uses self.path if None)
            smoothing_factor: Spline smoothing parameter
            num_points: Number of points in smoothed path
            
        Returns:
            Smoothed path as numpy array of shape (N, 2)
        """
        if path is None:
            path = self.path
        
        if path is None or len(path) < 3:
            return np.array(path) if path else None
        
        if smoothing_factor is None:
            smoothing_factor = PathPlanningConfig.SMOOTHING_FACTOR
        if num_points is None:
            num_points = PathPlanningConfig.NUM_SMOOTHING_POINTS
        
        try:
            path_array = np.array(path)
            x_coords = path_array[:, 0]
            y_coords = path_array[:, 1]
            
            # Calculate cumulative distances along path
            distances = np.zeros(len(path))
            for i in range(1, len(path)):
                distances[i] = distances[i-1] + self.heuristic(
                    (x_coords[i-1], y_coords[i-1]), 
                    (x_coords[i], y_coords[i])
                )
            
            # Normalize to [0, 1]
            if distances[-1] > 0:
                distances = distances / distances[-1]
            
            # Create splines
            t_smooth = np.linspace(0, 1, num_points)
            spl_x = UnivariateSpline(distances, x_coords, s=smoothing_factor)
            spl_y = UnivariateSpline(distances, y_coords, s=smoothing_factor)
            
            x_smooth = spl_x(t_smooth)
            y_smooth = spl_y(t_smooth)
            
            self.smoothed_path = np.column_stack((x_smooth, y_smooth))
            return self.smoothed_path
            
        except Exception as e:
            print(f"Path smoothing failed: {e}")
            return np.array(path)
    
    def astar(self, grid: np.ndarray, 
              start: Tuple[int, int], 
              goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm implementation."""
        rows, cols = grid.shape
        
        # A* algorithm
        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.calculate_distance(start, goal)} #Heuristic default euclidean
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
            
            for neighbor in self.get_neighbors(grid, rows, cols, current):
                if neighbor in closed_set:
                    continue
                
                move_cost = self.calculate_distance(current, neighbor)
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.calculate_distance(neighbor, goal)
                    
                    counter += 1
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        return None
    
    def calculate_path_length(self, path: List[Tuple[int, int]] = None, 
                             distance_method: str = "euclidean") -> float:
        """
        Calculate total path length.
        
        Args:
            path: Path to calculate length for (uses self.path if None)
            distance_method: Distance metric to use ("euclidean", "manhattan", "chebyshev", "diagonal")
        
        Returns:
            Total path length
        """
        if path is None:
            path = self.path
        
        # check if there are at least two points to calculate a distance.
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += self.calculate_distance(path[i], path[i+1], method=distance_method)
        
        return length



from typing import List, Tuple, Dict, Set
import numpy as np
import heapq
from math import sqrt
from scipy.interpolate import UnivariateSpline, splprep, splev
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.cm as cm  # for colormap

def create_node(position: Tuple[int, int], g: float = float('inf'), 
                h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Create a node for the A* algorithm.
    
    Args:
        position: (x, y) coordinates of the node
        g: Cost from start to this node (default: infinity)
        h: Estimated cost from this node to goal (default: 0)
        parent: Parent node (default: None)
    
    Returns:
        Dictionary containing node information
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

#Calculate the estimated distance between two points using Euclidean distance.
def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get all valid neighboring positions in the grid.
    
    Args:
        grid: 2D numpy array where 0 represents walkable cells and 1 represents obstacles
        position: Current position (x, y)
    
    Returns:
        List of valid neighboring positions
    """
    x, y = position
    rows, cols = grid.shape
    
    # All possible moves (including diagonals)
    possible_moves = [
        (x+1, y), (x-1, y),    # Right, Left
        (x, y+1), (x, y-1),    # Up, Down
        (x+1, y+1), (x-1, y-1),  # Diagonal moves
        (x+1, y-1), (x-1, y+1)
    ]
    
    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
        and grid[nx, ny] == 0                # Not an obstacle
    ]

def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from goal to start by following parent pointers.
    """
    path = []
    current = goal_node
    
    while current is not None:
        path.append(current['position'])
        current = current['parent']
        
    return path[::-1]  # Reverse to get path from start to goal

def find_path(grid: np.ndarray, start: Tuple[int, int], 
              goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the optimal path using A* algorithm.
    
    Args:
        grid: 2D numpy array (0 = free space, 1 = obstacle)
        start: Starting position (x, y)
        goal: Goal position (x, y)
    
    Returns:
        List of positions representing the optimal path
    """
    # Initialize start node
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )
    
    # Initialize open and closed sets
    open_list = [(start_node['f'], start)]  # Priority queue
    open_dict = {start: start_node}         # For quick node lookup
    closed_set = set()                      # Explored nodes
    
    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]
        
        # Check if we've reached the goal
        if current_pos == goal:
            return reconstruct_path(current_node)
            
        closed_set.add(current_pos)
        
        # Explore neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue
                
            # Calculate new path cost
            tentative_g = current_node['g'] + calculate_heuristic(current_pos, neighbor_pos)
            
            # Create or update neighbor
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
    
    return []  # No path found

def smooth_path_bspline(path: List[Tuple[int, int]], smoothing_factor: float = 0.1, num_points: int = 100) -> np.ndarray:
    """
    Smooth the path using B-spline interpolation.
    
    Args:
        path: List of (x, y) coordinates representing the path
        smoothing_factor: Smoothing factor for the B-spline (0 = no smoothing, higher = more smoothing)
        num_points: Number of points in the smoothed path
    
    Returns:
        Numpy array of smoothed path coordinates
    """
    if len(path) < 3:
        return np.array(path)
    
    # Convert path to numpy array and separate x, y coordinates
    path_array = np.array(path)
    x_coords = path_array[:, 0]
    y_coords = path_array[:, 1]
    
    # Create parameter array (cumulative distance along path)
    distances = np.zeros(len(path))
    for i in range(1, len(path)):
        distances[i] = distances[i-1] + np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
    
    # Normalize distances to [0, 1]
    if distances[-1] > 0:
        distances = distances / distances[-1]
    
    # Create new parameter array for smooth curve
    t_smooth = np.linspace(0, 1, num_points)
    
    # Apply B-spline smoothing to both x and y coordinates
    spl_x = UnivariateSpline(distances, x_coords, s=smoothing_factor)
    spl_y = UnivariateSpline(distances, y_coords, s=smoothing_factor)
    
    # Evaluate smoothed coordinates
    x_smooth = spl_x(t_smooth)
    y_smooth = spl_y(t_smooth)
    
    # Combine into path array
    smoothed_path = np.column_stack((x_smooth, y_smooth))
    
    return smoothed_path
def visualize_path(grid: np.ndarray,original_path:List[Tuple[int, int]], path: List[Tuple[int, int]], show_smoothed: bool = True):
    """
    Visualize the grid and found path with optional B-spline smoothing.
    
    Args:
        grid: 2D numpy array representing the grid
        path: List of (x, y) coordinates representing the path
        show_smoothed: Whether to show the smoothed path
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='binary')

    if original_path:
        opath_array  = np.array(original_path)
        plt.plot(opath_array[:, 1], opath_array[:, 0], 'r', linewidth=3, alpha=0.8, label='Original Path')
    if path:
        path_array = np.array(path)
        
        # Plot original path in blue
        plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, alpha=0.8, label='Optimized Path')
        
        # Plot smoothed path if requested
        if show_smoothed and len(path) >= 3:
            try:
                smoothing_factors = [0.0, 0.1 ,0.5]
                # colors = cm.viridis(np.linspace(0, 1, len(smoothing_factors)))
                colors = ['g','c','m']

                for i, factor in enumerate(smoothing_factors):
                    try:
                        basic_smoothed = smooth_path_bspline(path, smoothing_factor=factor, num_points=100)
                        print(f"Smoothing factor {factor:.2f}: {len(basic_smoothed)} points")

                        plt.plot(
                            basic_smoothed[:, 1],
                            basic_smoothed[:, 0],
                            color=colors[i],
                            linewidth=2,
                            alpha=0.7,
                            label=f'Smoothing factor = {factor:.2f}',
                            linestyle='solid'
                        )
                    except Exception as e:
                        print(f"Smoothing factor {factor}: Error - {e}")

            except Exception as e:
                print(f"Warning: Could not create smoothed paths: {e}")

        
        # Plot start and goal points
        plt.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=12, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        plt.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=12, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.title("A* Pathfinding with B-spline Smoothing", fontsize=14, fontweight='bold')
    plt.xlabel("Y Coordinate", fontsize=12)
    plt.ylabel("X Coordinate", fontsize=12)
    
    # Add text box with path information
    if path:
        path_length = len(path)
        info_text = f"Path Length: {path_length} steps\n"
        if show_smoothed and len(path) >= 3:
            info_text += "Smoothing: B-spline\n"
           # info_text += "Safe Distance: 1.5 units"
        else:
            info_text += "Smoothing: Disabled"
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()




def inflate_obstacles(grid: np.ndarray, clearance: int = 1) -> np.ndarray:
    """
    Inflate obstacles in the grid by a given clearance (in cells).
    
    Args:
        grid: Original grid (0 = free, 1 = obstacle)
        clearance: Radius in grid cells to expand each obstacle
    
    Returns:
        A new grid with inflated obstacles
    """
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    inflated_grid = ndimage.binary_dilation(grid, structure=structure, iterations=clearance).astype(np.uint8)
    return inflated_grid




def plot_multiple_smoothing_factors(grid: np.ndarray, path: List[Tuple[int, int]], factors: List[float]):
    """
    Plot the original path and multiple smoothed versions with varying smoothing factors.
    
    Args:
        grid: 2D grid
        path: Original A* path
        factors: List of smoothing factors to test
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(grid, cmap='binary')

    if not path:
        print("No path to plot.")
        return

    path_array = np.array(path)

    # Plot original path
    plt.plot(path_array[:, 1], path_array[:, 0], 'b--', linewidth=2, alpha=0.7, label='Original Path')

    # Color map for different factors
    colors = cm.viridis(np.linspace(0, 1, len(factors)))

    # Plot each smoothed path
    for i, factor in enumerate(factors):
        basic_smoothed = smooth_path_bspline(path, smoothing_factor=factor, num_points=100)
        print(f"Smoothing factor {factor:.2f}: {len(basic_smoothed)} points")

        plt.plot(
            basic_smoothed[:, 1],
            basic_smoothed[:, 0],
            color=colors[i],
            linewidth=2,
            alpha=0.8,
            label=f'Smoothing factor = {factor:.2f}'
        )

    # Start and goal markers
    plt.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=12, label='Start')
    plt.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=12, label='Goal')

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    plt.title("Path with Multiple B-spline Smoothing Factors", fontsize=14)
    plt.xlabel("Y Coordinate")
    plt.ylabel("X Coordinate")
    plt.tight_layout()
    plt.show()

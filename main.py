import numpy as np
from a_star import find_path, visualize_path, smooth_path_bspline

#creating grid with 20x20 with all free spaces
grid = np.zeros((20, 20))
# Add some obstacles
# Let's make columns 4 to 15 as "car rows", and leave (for example) column 10 in each row empty for parking.
for row in range(4, 15):
    grid[row, 4:15] = 1  # fill parking row with cars (obstacles)
    grid[row, 8:12] = 0    # leave one spot as an empty parking space
# Define start and goal positions
start_pos = (2, 2)
goal_pos = (18, 18)
# Find the path
path = find_path(grid, start_pos, goal_pos)
if path:
    print(f"Path found with {len(path)} steps!")
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print("Visualizing path with B-spline smoothing...")
    
    # Demonstrate different smoothing levels
    print("\nTesting different smoothing factors:")
    smoothing_factors = [0.0, 0.1, 0.5, 1.0]
    for factor in smoothing_factors:
        try:
            smoothed = smooth_path_bspline(path, smoothing_factor=factor, num_points=100)
            print(f"Smoothing factor {factor}: {len(smoothed)} points")
        except Exception as e:
            print(f"Smoothing factor {factor}: Error - {e}")
    
    visualize_path(grid, path, show_smoothed=True)
else:
    print("No path found!")
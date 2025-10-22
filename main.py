import numpy as np
from a_star import find_path, visualize_path, smooth_path_bspline,inflate_obstacles

#creating grid with 20x20 with all free spaces
grid = np.zeros((20, 20))
# Add some obstacles
# Let's make columns 4 to 15 as "car rows", and leave (for example) column 10 in each row empty for parking.
for row in range(4, 15):
    grid[row, 2:17] = 1  # fill parking row with cars (obstacles)
    grid[row, 5:8] = 0    # leave one spot as an empty parking space
    grid[row, 11:14] = 0    # leave one spot as an empty parking space
# Define start and goal positions
start_pos = (2, 2)
goal_pos = (18, 12)
# Find the path

clearance_radius = 1  # Number of cells around obstacle to avoid
inflated_grid = inflate_obstacles(grid, clearance=clearance_radius)
origina_path = find_path(grid, start_pos, goal_pos)
path = find_path(inflated_grid, start_pos, goal_pos)
if path:
    print(f"Path found with {len(path)} steps!")
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print("Visualizing path with B-spline smoothing...")
    visualize_path(grid,origina_path, path, show_smoothed=True)
else:
    print("No path found!")
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from a_star import find_path, visualize_path, smooth_path_bspline,inflate_obstacles

def detect_parking_lot_features(image_path, grid_width=50, grid_height=30, visualize=False):
    """
    Detect cars and parking spots in a parking lot image using traditional computer vision techniques.
    Args:
        image_path: Path to the parking lot image.
        grid_width: Number of grid cells along the width.
        grid_height: Number of grid cells along the height.
        visualize: If True, display intermediate results.
    Returns:
        grid: 2D NumPy array where:
            - 0: Free space
            - 1: Car (obstacle)
            - 2: Parking spot
    """
    # Step 1: Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if visualize:
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title("Original Parking Lot Image")
        plt.axis('off')
        plt.show()

    # Step 2: Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    if visualize:
        plt.figure(figsize=(10, 8))
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection")
        plt.axis('off')
        plt.show()

    # Step 3: Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if visualize:
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(image_with_contours)
        plt.title("Contour Detection")
        plt.axis('off')
        plt.show()

    # Step 4: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply dilation to fill small holes in objects
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if visualize:
        image_with_closed_contours = image.copy()
        cv2.drawContours(image_with_closed_contours, contours, -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(image_with_closed_contours)
        plt.title("Contours After Morphological Closing")
        plt.axis('off')
        plt.show()

    # Dedicated parking-spot mask: detect painted white lines via HSV (value high, low saturation)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # white-ish paint: high V, low-medium S (loosened thresholds)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 120, 255])
    spot_mask = cv2.inRange(hsv, lower_white, upper_white)
    # small opening to remove speckle, then close gaps in painted lines
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # close gaps in painted lines to form continuous rectangles
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    spot_closed = cv2.morphologyEx(spot_mask, cv2.MORPH_CLOSE, kernel_rect, iterations=2)
    spot_closed = cv2.dilate(spot_closed, kernel_rect, iterations=1)
    spot_contours, _ = cv2.findContours(spot_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Attempt center-based spot detection using distance transform + floodFill
    try:
        inv = cv2.bitwise_not(spot_closed)
        # distance to nearest white line
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        # pick peaks where distance sufficiently large (likely spot centers)
        peak_thresh = max(12, int(min(image.shape[:2]) / 60))
        peaks_mask = (dist > peak_thresh).astype('uint8') * 255
        # remove small peaks
        peaks_mask = cv2.morphologyEx(peaks_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=1)
        peak_cnts, _ = cv2.findContours(peaks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for pc in peak_cnts:
            M = cv2.moments(pc)
            if M.get('m00', 0) == 0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # flood fill from centroid on inverted spot mask to get slot interior
            ff_img = inv.copy()
            h, w = ff_img.shape
            mask = np.zeros((h+2, w+2), np.uint8)
            try:
                retval, ff_img, mask, rect = cv2.floodFill(ff_img, mask, (cx, cy), 128)
            except Exception:
                continue
            # region where value == 128 is the filled spot interior
            region = (ff_img == 128).astype('uint8') * 255
            region_cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for rc in region_cnts:
                ra = cv2.contourArea(rc)
                if min_parking_spot_area/2 < ra < max_parking_spot_area*1.5:
                    # dedupe: ensure not already in spot_contours
                    keep = True
                    rx, ry, rw, rh = cv2.boundingRect(rc)
                    rcx = rx + rw/2
                    rcy = ry + rh/2
                    for existing in spot_contours:
                        if cv2.pointPolygonTest(existing, (int(rcx), int(rcy)), False) >= 0:
                            keep = False
                            break
                    if keep:
                        spot_contours.append(rc)
    except Exception:
        pass

    if visualize:
        vis_spots = image.copy()
        cv2.drawContours(vis_spots, spot_contours, -1, (255, 0, 0), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_spots)
        plt.title("Detected Spot Contours from Thresholding")
        plt.axis('off')
        plt.show()

    # Step 5: Geometric feature analysis
    cars = []
    parking_spots = []

    # Thresholds for geometric features
    # Cars: solid objects with moderate size and relatively compact
    min_car_area = 900
    max_car_area = 30000
    min_car_aspect_ratio = 0.7
    max_car_aspect_ratio = 4.0

    # Parking spots: outlined rectangles (larger, detected by edge/contour)
    min_parking_spot_area = 800
    max_parking_spot_area = 80000
    min_parking_spot_aspect_ratio = 0.5
    max_parking_spot_aspect_ratio = 2.0

    # First, use the spot contours derived from the HSV thresholding pass
    for s_cont in spot_contours:
        s_area = cv2.contourArea(s_cont)
        if s_area < min_parking_spot_area:
            continue
        s_x, s_y, s_w, s_h = cv2.boundingRect(s_cont)
        s_ar = float(s_w) / s_h if s_h > 0 else 0
        s_peri = cv2.arcLength(s_cont, True)
        s_approx = cv2.approxPolyDP(s_cont, 0.02 * s_peri, True)

        # Skip if center already inside an existing detected spot (dedupe)
        cx = s_x + s_w / 2
        cy = s_y + s_h / 2
        already = False
        for existing in parking_spots:
            if cv2.pointPolygonTest(existing, (int(cx), int(cy)), False) >= 0:
                already = True
                break
        if already:
            continue

        if len(s_approx) == 4 and (min_parking_spot_aspect_ratio < s_ar < max_parking_spot_aspect_ratio):
            parking_spots.append(s_approx)
        else:
            # keep the raw contour if it roughly matches size/aspect constraints
            if min_parking_spot_area < s_area < max_parking_spot_area and (min_parking_spot_aspect_ratio < s_ar < max_parking_spot_aspect_ratio):
                parking_spots.append(s_cont)

    # If HSV-based detection found no spots, try a simple bright threshold fallback
    if len(parking_spots) == 0:
        try:
            _, fb = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
            fb = cv2.morphologyEx(fb, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (20,6)), iterations=2)
            fb_cnts, _ = cv2.findContours(fb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for fc in fb_cnts:
                fa = cv2.contourArea(fc)
                if min_parking_spot_area < fa < max_parking_spot_area:
                    f_x, f_y, f_w, f_h = cv2.boundingRect(fc)
                    f_ar = float(f_w)/f_h if f_h>0 else 0
                    if min_parking_spot_aspect_ratio < f_ar < max_parking_spot_aspect_ratio:
                        parking_spots.append(fc)
        except Exception:
            pass

    # Now iterate all contours to find cars and any spots missed above
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        solidity = area / float(w * h) if (w * h) > 0 else 0
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Filter out very small contours (noise)
        if area < 300:
            continue

        # If this contour looks like a rectangular parking spot and wasn't detected earlier, add it
        if len(approx) == 4 and (min_parking_spot_area < area < max_parking_spot_area) and (min_parking_spot_aspect_ratio < aspect_ratio < max_parking_spot_aspect_ratio) and solidity < 0.8:
            # dedupe by center test
            cx = x + w / 2
            cy = y + h / 2
            dup = False
            for existing in parking_spots:
                if cv2.pointPolygonTest(existing, (int(cx), int(cy)), False) >= 0:
                    dup = True
                    break
            if not dup:
                parking_spots.append(approx)
            continue

        # Otherwise classify as car (solid, relatively compact)
        if (min_car_area < area < max_car_area*2) and (min_car_aspect_ratio < aspect_ratio < max_car_aspect_ratio*1.5) and solidity > 0.2:
            cars.append(contour)

    # Additional car detection using adaptive thresholding to catch cars with paint similar to ground
    try:
        adp = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        # remove small noise and close car regions
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        adp = cv2.morphologyEx(adp, cv2.MORPH_OPEN, k, iterations=1)
        adp = cv2.morphologyEx(adp, cv2.MORPH_CLOSE, k, iterations=2)
        car_cnts2, _ = cv2.findContours(adp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in car_cnts2:
            a = cv2.contourArea(c)
            if a > min_car_area/2 and a < max_car_area*3:
                # avoid adding duplicates (overlap with existing cars)
                dup = False
                M = cv2.moments(c)
                if M.get("m00", 0) != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    for ec in cars:
                        if cv2.pointPolygonTest(ec, (cx, cy), False) >= 0:
                            dup = True
                            break
                if not dup:
                    cars.append(c)
    except Exception:
        pass

    # Helper to compute overlap area between two contours (in pixels)
    def contour_overlap_area(c1, c2):
        # create masks at image size
        mask1 = np.zeros(gray.shape, dtype=np.uint8)
        mask2 = np.zeros(gray.shape, dtype=np.uint8)
        try:
            cv2.drawContours(mask1, [c1], -1, 255, -1)
            cv2.drawContours(mask2, [c2], -1, 255, -1)
            inter = cv2.bitwise_and(mask1, mask2)
            return float(np.count_nonzero(inter))
        except Exception:
            return 0.0

    if visualize:
        image_with_classification = image.copy()
        cv2.drawContours(image_with_classification, cars, -1, (0, 0, 255), 2)  # Red for cars
        cv2.drawContours(image_with_classification, parking_spots, -1, (255, 0, 0), 2)  # Blue for parking spots
        plt.figure(figsize=(10, 8))
        plt.imshow(image_with_classification)
        plt.title("Classified Contours (Red: Cars, Blue: Parking Spots)")
        plt.axis('off')
        plt.show()

    # Step 6: Create grid representation
    # Grid values: 0=free space, 2=empty parking spot, 1=occupied parking spot, 3=car outside parking spot, 4=obstacle
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Prepare image masks for overlap testing
    img_h, img_w = gray.shape
    spot_masks = []  # list of (contour, mask, area_pixels)
    for contour in parking_spots:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        try:
            cv2.drawContours(mask, [contour], -1, 255, -1)
        except Exception:
            cv2.drawContours(mask, contour, -1, 255, -1)
        area_px = float(np.count_nonzero(mask))
        if area_px > 0:
            spot_masks.append((contour, mask, area_px))

    car_masks = []
    for c in cars:
        m = np.zeros((img_h, img_w), dtype=np.uint8)
        try:
            cv2.drawContours(m, [c], -1, 255, -1)
        except Exception:
            cv2.drawContours(m, c, -1, 255, -1)
        car_masks.append((c, m, float(np.count_nonzero(m))))

    # Determine occupancy by overlap ratio between car mask and spot mask
    occupied_spots = set()
    car_assigned = [False] * len(car_masks)
    spot_overlaps = []
    for si, (s_cont, s_mask, s_area) in enumerate(spot_masks):
        best_overlap = 0.0
        best_ci = None
        for ci, (c_cont, c_mask, c_area) in enumerate(car_masks):
            inter = cv2.bitwise_and(s_mask, c_mask)
            ov = float(np.count_nonzero(inter))
            # measure overlap relative to spot area
            if s_area > 0 and ov / s_area > best_overlap:
                best_overlap = ov / s_area
                best_ci = ci
        spot_overlaps.append((si, best_overlap, best_ci))
        # if overlap significant, mark spot occupied (lowered threshold to be more sensitive)
        if best_overlap >= 0.079 and best_ci is not None:
            occupied_spots.add(si)
            car_assigned[best_ci] = True
            try:
                print(f"Marked spot {si} occupied (overlap={best_overlap:.6f}, car={best_ci})")
            except Exception:
                pass

    # Print per-spot overlap information to help tuning
    try:
        print("\nPer-spot overlap ratios (spot_index, overlap_ratio, matched_car_index):")
        for si, ov, ci in spot_overlaps:
            print(f"  Spot {si}: overlap={ov:.3f}, car_index={ci}")
    except Exception:
        pass

    # Map parking spots to grid: occupied -> 1, empty -> 2
    for si, (s_cont, s_mask, s_area) in enumerate(spot_masks):
        val = 1 if si in occupied_spots else 2
        # fill grid cells that fall inside the spot contour
        for gx in range(grid_width):
            for gy in range(grid_height):
                pixel_x = int(gx * (img_w / grid_width) + (img_w / grid_width) / 2)
                pixel_y = int(gy * (img_h / grid_height) + (img_h / grid_height) / 2)
                if pixel_x < 0 or pixel_y < 0 or pixel_x >= img_w or pixel_y >= img_h:
                    continue
                if s_mask[pixel_y, pixel_x] > 0:
                    grid[gy, gx] = val

    # Map cars that were not assigned to spots as cars outside spots (3)
    for ci, (c_cont, c_mask, c_area) in enumerate(car_masks):
        if car_assigned[ci]:
            continue
        for gx in range(grid_width):
            for gy in range(grid_height):
                pixel_x = int(gx * (img_w / grid_width) + (img_w / grid_width) / 2)
                pixel_y = int(gy * (img_h / grid_height) + (img_h / grid_height) / 2)
                if pixel_x < 0 or pixel_y < 0 or pixel_x >= img_w or pixel_y >= img_h:
                    continue
                if c_mask[pixel_y, pixel_x] > 0 and grid[gy, gx] == 0:
                    grid[gy, gx] = 3

    # Detect large obstacles (islands/curbs) from contours: mark as 4
    for contour in contours:
        a = cv2.contourArea(contour)
        if a > 20000:
            # draw mask for contour
            m = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(m, [contour], -1, 255, -1)
            for gx in range(grid_width):
                for gy in range(grid_height):
                    pixel_x = int(gx * (img_w / grid_width) + (img_w / grid_width) / 2)
                    pixel_y = int(gy * (img_h / grid_height) + (img_h / grid_height) / 2)
                    if pixel_x < 0 or pixel_y < 0 or pixel_x >= img_w or pixel_y >= img_h:
                        continue
                    if m[pixel_y, pixel_x] > 0 and grid[gy, gx] == 0:
                        grid[gy, gx] = 4

    if visualize:
        plt.figure(figsize=(10, 8))
        cmap = plt.cm.get_cmap('tab10')
        plt.imshow(grid, cmap=cmap, interpolation='nearest')
        plt.title("Grid Representation (0: Free Space, 1: Occupied Spot, 2: Empty Spot, 3: Car Outside Spot)")
        cbar = plt.colorbar()
        cbar.set_label('Grid Value')
        plt.show()

    # Save a visualization image with detected parking spots and cars for quick review
    try:
        vis_img = image.copy()
        # draw parking spots (blue) and cars (red) on the RGB image
        if len(parking_spots) > 0:
            cv2.drawContours(vis_img, parking_spots, -1, (0, 0, 255), 2)
        if len(cars) > 0:
            cv2.drawContours(vis_img, cars, -1, (255, 0, 0), 2)
        # convert RGB -> BGR for OpenCV saving
        save_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(os.path.dirname(image_path), 'detection_result.png')
        cv2.imwrite(out_path, save_img)
    except Exception:
        pass

    return grid

# Create grid from image
image_path = "parking_lot.jpg"  # Replace with your image path
print(f"Attempting to load image from: {os.path.abspath(image_path)}")
try:
    grid = detect_parking_lot_features(image_path, grid_width=50, grid_height=30, visualize=True)
    print("Successfully created grid from image")
    print(f"Grid shape: {grid.shape}")
    print(f"\nGrid Statistics:")
    print(f"  Free spaces: {np.sum(grid == 0)}")
    print(f"  Occupied parking spots: {np.sum(grid == 1)}")
    print(f"  Empty parking spots: {np.sum(grid == 2)}")
    print(f"  Cars outside parking spots: {np.sum(grid == 3)}")
    total_parking = np.sum(grid == 1) + np.sum(grid == 2)
    occupied = np.sum(grid == 1)
    if total_parking > 0:
        print(f"\nParking Occupancy: {occupied}/{total_parking} ({100*occupied/total_parking:.1f}%)")
    
except Exception as e:
    print(f"Error loading image: {e}")
    print("Falling back to default grid...")
    grid = np.zeros((20, 20))

# Define start and goal positions - these can be set based on user input or image analysis
start_pos = (2, 2)  # Starting position (outside parking lot)
goal_pos = (25, 15)  # Goal position (middle of grid, in a free area)

# Find the path
clearance_radius = 1  # Number of cells around obstacle to avoid
inflated_grid = inflate_obstacles(grid, clearance=clearance_radius)
original_path = find_path(grid, start_pos, goal_pos)
path = find_path(inflated_grid, start_pos, goal_pos)

if path:
    print(f"Path found with {len(path)} steps!")
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print("Visualizing path with B-spline smoothing...")
    visualize_path(grid, original_path, path, show_smoothed=True)
else:
    print("No path found!")
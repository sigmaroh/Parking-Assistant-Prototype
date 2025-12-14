import cv2
import numpy as np
import matplotlib.pyplot as plt

# Helper function to show images with matplotlib
def show_image(title, img, cmap=None, subplot_index=1, total=1):
    plt.subplot(1, total, subplot_index)
    plt.title(title)
    if len(img.shape) == 2:  # grayscale
        plt.imshow(img, cmap=cmap if cmap else 'gray')
    else:
        # convert BGR to RGB for matplotlib display
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

img = cv2.imread("Layout 1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Blur + Canny
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 60, 150)

# 2. Morphological Closing (connect broken lines)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 3. Optional: Dilate to thicken boundaries
dilated = cv2.dilate(closed, kernel, iterations=1)

# 4. Detect contours for visualization
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Prepare output image to draw green parking slots AND red for filtered rectangles/contours together
combined_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100:
        continue  # skip tiny contours
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    # Check if it's a candidate parking slot rectangle
    if len(approx) == 4 and area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
        if 0.5 <= aspect_ratio <= 5.0:
            # Draw parking area as green rectangle and continue to next. (Green: (0, 255, 0))
            cv2.rectangle(combined_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            continue

    # Otherwise: draw as filtered with red (either as rectangle or contour)
    if 4 <= len(approx) <= 6 and area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(combined_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        cv2.drawContours(combined_img, [cnt], -1, (0, 0, 255), 2)

# Draw detected slots as green rectangles only for a clean "Detected Slots" image as reference
output = img.copy()
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 4 and area > 2000:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
        if 1.1 <= aspect_ratio <= 4.0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw all contours (red) for baseline reference
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

# Compose visualizations
stage_imgs = [
    ("Original", img, None),
    ("Gray", gray, 'gray'),
    ("Blurred", blur, 'gray'),
    ("Edges", edges, 'gray'),
    ("Closed Lines", closed, 'gray'),
    ("Dilated", dilated, 'gray'),
    ("All Contours (red)", contour_img, None),
    ("Combined: Green=Parking, Red=Filtered", combined_img, None),
    ("Detected Slots (green)", output, None),
]

plt.figure(figsize=(22, 10))
for i, (title, im, cmap) in enumerate(stage_imgs, 1):
    show_image(title, im, cmap, subplot_index=i, total=len(stage_imgs))
plt.tight_layout()
plt.show()

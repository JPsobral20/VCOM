import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

from sklearn.cluster import DBSCAN

image_num = 0

def save_image(path, image, verbose=False):
    if verbose:
        cv2.imwrite(path, image)
        print(f"Image saved to {path}")
def get_image_num():
    global image_num
    return image_num

def set_image_num(num):
    global image_num
    image_num = num

def build_image_path(image_name, images_dir):
    num = get_image_num()
    return os.path.join(images_dir, f"{num}_{image_name}")

def preprocess_image(image, output_path, verbose=False):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(build_image_path("gray.jpg", output_path), gray, verbose)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    save_image(build_image_path("hsv.jpg", output_path), hsv, verbose)
    # cv2.imshow("hsv", hsv)
    # while True:
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (31,31), 0)
    save_image(build_image_path("blurred.jpg", output_path), blurred, verbose)

    return gray, blurred, hsv

def line_intersection(p1, p2, p3, p4):
    """Find intersection of two lines (p1-p2) and (p3-p4)"""
    A1, B1, C1 = np.cross([*p1, 1], [*p2, 1])
    A2, B2, C2 = np.cross([*p3, 1], [*p4, 1])
    D = A1 * B2 - A2 * B1
    if abs(D) < 1e-6:  # Lines are parallel
        return None
    x = (B1 * C2 - B2 * C1) / D
    y = (A2 * C1 - A1 * C2) / D
    return np.array([x, y])

def find_rotation(warped, output_path, verbose=False):
    ret, new_thresh = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)
    save_image(build_image_path("new_thresh.jpg", output_path), new_thresh, verbose)
    new_thresh = new_thresh / 255

    grid_size = 50
    half_window = grid_size // 2
    epsilon = 0.3

    corners = [
        [half_window, half_window, cv2.ROTATE_90_COUNTERCLOCKWISE],
        [new_thresh.shape[1] - half_window, half_window, cv2.ROTATE_180],
        [half_window, new_thresh.shape[0] - half_window, None],
        [new_thresh.shape[1] - half_window, new_thresh.shape[0] - half_window, cv2.ROTATE_90_CLOCKWISE]
    ]
    while grid_size < 100:
        while epsilon < 0.8:
            for corner in corners:
                center_x, center_y, rotation = corner
                x_start, x_end = max(0, center_x - half_window), min(new_thresh.shape[1], center_x + half_window)
                y_start, y_end = max(0, center_y - half_window), min(new_thresh.shape[0], center_y + half_window)
                window = new_thresh[y_start:y_end, x_start:x_end]
                pixel_sum = np.sum(window)
                thresh_value = grid_size * grid_size * epsilon # If the sum of the pixels is less than epsilon% of the window size, it is considered to be the knight

                if (pixel_sum < thresh_value):        
                    if verbose:
                        print("Pixel sum:", pixel_sum)
                        print("Threshold value:", thresh_value)
                        print(x_start, x_end, y_start, y_end)
                    if rotation is not None:
                        warped = cv2.rotate(warped, rotation)
                        save_image(build_image_path("warped_rotated.jpg", output_path), warped, verbose)
                        return warped, rotation
                    else:
                        save_image(build_image_path("warped_rotated.jpg", output_path), warped, verbose)
                        return warped, None
            epsilon += 0.05
        grid_size += 10
        epsilon = 0.3
    print("No rotation found")
    save_image(build_image_path("warped_rotated.jpg", output_path), warped, verbose)
    return warped, None

def border_intersections(rho, theta, image_shape):
    height, width = image_shape
    pts = []
    
    # Left border: x = 0
    if abs(np.sin(theta)) > 1e-6:
        y = (rho - 0 * np.cos(theta)) / np.sin(theta)
        if 0 <= y < height:
            pts.append((0, int(round(y))))
    
    # Right border: x = width - 1
    if abs(np.sin(theta)) > 1e-6:
        y = (rho - (width - 1) * np.cos(theta)) / np.sin(theta)
        if 0 <= y < height:
            pts.append((width - 1, int(round(y))))
    
    # Top border: y = 0
    if abs(np.cos(theta)) > 1e-6:
        x = (rho - 0 * np.sin(theta)) / np.cos(theta)
        if 0 <= x < width:
            pts.append((int(round(x)), 0))
    
    # Bottom border: y = height - 1
    if abs(np.cos(theta)) > 1e-6:
        x = (rho - (height - 1) * np.sin(theta)) / np.cos(theta)
        if 0 <= x < width:
            pts.append((int(round(x)), height - 1))
    
    pts = list(set(pts))
    
    if len(pts) > 2:
        max_dist = 0
        best_pair = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.sqrt(pts[i][0]**2 + pts[i][1]**2) + np.sqrt(pts[j][0]**2 + pts[j][1]**2)
                if d > max_dist:
                    max_dist = d
                    best_pair = (pts[i], pts[j])
        pts = list(best_pair)
    
    pts.sort(key=lambda p: p[0])
    
    return pts

def filter_lines(lines, image_shape, horizontal=True, threshold=50):
    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        pts = border_intersections(rho, theta, image_shape)
        if len(pts) == 2:
            filtered_lines.append([line[0], pts[0], pts[1]])
        else:
            filtered_lines.append(line)
    
    filtered_lines.sort(key=lambda line: (line[1][0], line[1][1]))

    new_filtered_lines = []
    image_center = np.array([image_shape[1] // 2, image_shape[0] // 2])
    lines_to_remove = set()
    for i in range(len(filtered_lines)):
        intersections = []
        x1, y1 = filtered_lines[i][1]
        x2, y2 = filtered_lines[i][2]
        for j in range(i + 1, len(filtered_lines)):
            if j in lines_to_remove:
                continue
            x3, y3 = filtered_lines[j][1]
            x4, y4 = filtered_lines[j][2]

            p = line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            if p is not None:
                p = np.round(p).astype(int)
                if 0 <= p[0] < image_shape[1] and 0 <= p[1] < image_shape[0]:
                    midpoint_2 = np.array([(x3 + x4) // 2, (y3 + y4) // 2])
                    dist = np.linalg.norm(midpoint_2 - image_center)
                    intersections.append((j, dist))
        if len(intersections) > 0:
            midpoint_1 = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
            dist = np.linalg.norm(midpoint_1 - image_center)
            for intersection in intersections:
                if intersection[1] < dist:
                    lines_to_remove.add(intersection[0])
                else:
                    lines_to_remove.add(i)
            if i not in lines_to_remove:
                new_filtered_lines.append(filtered_lines[i])
        elif i not in lines_to_remove:
            new_filtered_lines.append(filtered_lines[i])
    return new_filtered_lines

    unique_lines = []
    distances = []
    for line in new_filtered_lines:
        if not unique_lines:
            unique_lines.append(line)
        else:
            last_line = unique_lines[-1]
            dist = np.linalg.norm(np.array(line[1]) - np.array(last_line[1]))
            dist2 = np.linalg.norm(np.array(line[2]) - np.array(last_line[2]))
            dist_min = min(dist, dist2)
            if dist_min > threshold:
                unique_lines.append(line)
                distances.append(dist_min)
            else:
                center = np.array([image_shape[1] // 2, image_shape[0] // 2])
                midpoint = np.array([(line[1][0] + line[2][0]) // 2, (line[1][1] + line[2][1]) // 2])
                dist_center_max = np.linalg.norm(midpoint - center)
                last_line_center = np.array([(last_line[1][0] + last_line[2][0]) // 2, (last_line[1][1] + last_line[2][1]) // 2])
                last_line_center_max = np.linalg.norm(last_line_center - center)

                if dist_center_max > last_line_center_max:
                    unique_lines[-1] = line
                    if len(distances) > 0:
                        distances[-1] = dist_min + distances[-1]

    return unique_lines

def board_detection(image, gray, hsv, output_path, verbose=False):
    saturation = hsv[:, :, 1]
    saturation_blurred = cv2.GaussianBlur(saturation, (31, 31), 0)
    mask = cv2.inRange(saturation_blurred, 0, 20)
    save_image(build_image_path("mask.jpg", output_path), mask, verbose)
    # Threshold the image using Otsu's method
    ret, otsu = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    save_image(build_image_path("otsu.jpg", output_path), otsu, verbose)

    # mix the two masks
    otsu = cv2.bitwise_and(otsu, otsu, mask=mask)
    save_image(build_image_path("otsu_mask.jpg", output_path), otsu, verbose)

    # Morphological operations
    kernel = np.ones((31, 31), np.uint8)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=5)
    save_image(build_image_path("otsu_morph.jpg", output_path), otsu, verbose)
    num_labels, labels = cv2.connectedComponents(otsu)

    areas = np.bincount(labels.ravel())
    cleaned = np.zeros_like(mask)
    max_area = np.max(areas[1:])  
    for label_id in range(1, num_labels):
        if areas[label_id] >= max_area * 0.5:
            cleaned[labels == label_id] = 255
    save_image(build_image_path("cleaned.jpg", output_path), cleaned, verbose)

    # Fill holes
    kernel = np.ones((31, 31), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=5)
    save_image(build_image_path("cleaned_filled.jpg", output_path), cleaned, verbose)
    
    canny = cv2.Canny(cleaned, 0, 200)
    save_image(build_image_path("canny.jpg", output_path), canny, verbose)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Hough Transform
    lines = cv2.HoughLines(canny, 1, np.pi/180, 200)
    line_image = np.zeros_like(image)
    invalid_thetas = [0, np.pi/2, np.pi, 3*np.pi/2]
    new_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            for invalid_theta in invalid_thetas:
                if abs(theta - invalid_theta) < 1e-6:
                    break
            else:
                pts = border_intersections(rho, theta, image.shape)
                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a * rho
                # y0 = b * rho
                # x1 = int(x0 + 5000 * (-b))
                # y1 = int(y0 + 5000 * (a))
                # x2 = int(x0 - 5000 * (-b))
                # y2 = int(y0 - 5000 * (a))
                # new_theta = np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
                new_theta = theta
                new_lines.append([[rho, new_theta], pts[0], pts[1]])
                cv2.line(line_image, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), 255, 2)
    save_image(build_image_path("lines.jpg", output_path), line_image, verbose)

    # Median of the theta of the lines
    if len(new_lines) > 0:
        theta_values = [line[0][1] for line in new_lines]
        theta_values_sorted = sorted(theta_values)
        median_index = len(theta_values_sorted) // 2
        median_theta = theta_values_sorted[median_index]
        if verbose:
            print("Median theta:", median_theta)
    else:
        print("No lines detected.")
        return None, None, None

    # Horizontal lines threshold
    threshold_degree = 30 # degrees
    max_theta = median_theta + threshold_degree*np.pi/180
    min_theta = median_theta - threshold_degree*np.pi/180
    if verbose:
        print("Max theta:", max_theta)
        print("Min theta:", min_theta)

    # Lines into two groups
    horizontal_lines = []
    vertical_lines = []

    for line in new_lines:
        rho, theta = line[0]
        if min_theta < theta < max_theta:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    if verbose:
        print("Horizontal lines:", len(horizontal_lines))
        print("Vertical lines:", len(vertical_lines))
    
    # check if any horizontal line intersects with any horizontal line inside the image and change the group if so
    for i in range(len(horizontal_lines)):
        for j in range(len(horizontal_lines)):
            if i != j and i < len(horizontal_lines) and j < len(horizontal_lines):
                x1, y1 = horizontal_lines[i][1]
                x2, y2 = horizontal_lines[i][2]
                x3, y3 = horizontal_lines[j][1]
                x4, y4 = horizontal_lines[j][2]
                theta1 = horizontal_lines[i][0][1]
                theta2 = horizontal_lines[j][0][1]

                p = line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                if p is not None and 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0] and abs(theta1 - theta2) > threshold_degree*np.pi/180:
                    horizontal_lines[j][0][1] = median_theta
                    vertical_lines.append(horizontal_lines[j])
                    horizontal_lines.pop(j)
                    break
    for i in range(len(vertical_lines)):
        for j in range(len(vertical_lines)):
            if i != j and i < len(vertical_lines) and j < len(vertical_lines):
                x1, y1 = vertical_lines[i][1]
                x2, y2 = vertical_lines[i][2]
                x3, y3 = vertical_lines[j][1]
                x4, y4 = vertical_lines[j][2]
                theta1 = vertical_lines[i][0][1]
                theta2 = vertical_lines[j][0][1]

                p = line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                if p is not None and 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0] and abs(theta1 - theta2) > threshold_degree*np.pi/180:
                    vertical_lines[j][0][1] = median_theta
                    horizontal_lines.append(vertical_lines[j])
                    vertical_lines.pop(j)
                    break


    if verbose:
        print("Horizontal lines:", len(horizontal_lines))
        print("Vertical lines:", len(vertical_lines))
    

    # Remove near-duplicates (within threshold)
    # lines_threshold = 20 # pixels
    # filtered_horizontal_lines = filter_lines(horizontal_lines, image.shape, True, lines_threshold)
    # filtered_vertical_lines = filter_lines(vertical_lines, image.shape, False, lines_threshold)
    # line_image = np.zeros_like(image)
    # for line in filtered_horizontal_lines:
    #     rho, theta = line[0]
    #     x1, y1 = line[1]
    #     x2, y2 = line[2]
    #     cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    # for line in filtered_vertical_lines:
    #     rho, theta = line[0]
    #     x1, y1 = line[1]
    #     x2, y2 = line[2]
    #     cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    # save_image(build_image_path("filtered_lines.jpg", output_path), line_image, verbose)
    
    # horizontal_lines = filtered_horizontal_lines
    # vertical_lines = filtered_vertical_lines

    # Remove lines with theta = to pi/2 and lines with thehta = 0
    # print(horizontal_lines)
    # print(vertical_lines)
    # new_horizontal_lines = []
    # for line in horizontal_lines:
    #     for invalid_theta in invalid_thetas:
    #         if abs(line[0][1] - invalid_theta) < 1e-6:
    #             break
    #     else:
    #         new_horizontal_lines.append(line)
    # horizontal_lines = new_horizontal_lines
    # new_vertical_lines = []
    # for line in vertical_lines:
    #     for invalid_theta in invalid_thetas:
    #         if abs(line[0][1] - invalid_theta) < 1e-6:
    #             break
    #     else:
    #         new_vertical_lines.append(line)
    # vertical_lines = new_vertical_lines 
    line_image = np.zeros_like(image)
    for line in horizontal_lines:
        rho, theta = line[0]
        x1, y1 = line[1]
        x2, y2 = line[2]
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    for line in vertical_lines:
        rho, theta = line[0]
        x1, y1 = line[1]
        x2, y2 = line[2]
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    save_image(build_image_path("filtered_lines.jpg", output_path), line_image, verbose)

    # Get line intersections
    points = []
    for i in range(len(horizontal_lines)):
        for j in range(len(vertical_lines)):
            x1, y1 = horizontal_lines[i][1]
            x2, y2 = horizontal_lines[i][2]
            x3, y3 = vertical_lines[j][1]
            x4, y4 = vertical_lines[j][2]

            p = line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            if p is not None and 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                p = np.round(p).astype(int)
                points.append(p)
    # Draw points
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(color, tuple(point), 5, (0, 0, 255), 10)
    save_image(build_image_path("points.jpg", output_path), color, verbose)

    points = np.array(points)
    # Find the four corners of the board
    if len(points) < 4:
        print("Not enough points found")
        return None, None, None
    square = cv2.convexHull(points, clockwise=True)
    square = square.reshape(-1, 2)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in square:
        cv2.circle(color, tuple(point), 20, (0, 255, 0), 10)
    save_image(build_image_path("square.jpg", output_path), color, verbose)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    iterations = 0
    distance = 100
    while len(square) > 4 and iterations < 50:
        # Remove points that are too close to each other
        new_square = []
        for point in square:
            valid = True
            for point2 in new_square:
                if np.linalg.norm(point - point2) < distance:
                    valid = False
                    break
            if valid:
                new_square.append(point)
        square = np.array(new_square)
        iterations += 1
        distance += 50
    
    for point in square:
        cv2.circle(color, tuple(point), 20, (0, 255, 0), 10)
    save_image(build_image_path("final_square.jpg", output_path), color, verbose)

    dst_points = np.array([[0, 1000], [1000, 1000], [1000, 0], [0, 0]], dtype="float32")
    # Perform the perspective transform
    M = cv2.getPerspectiveTransform(square.astype("float32"), dst_points)
    warped = cv2.warpPerspective(gray, M, (1000, 1000))
    save_image(build_image_path("warped.jpg", output_path), warped, verbose)

    warped, rotation = find_rotation(warped, output_path, verbose)    

    return warped, M, rotation
    
def square_detection_processed(rotation_image_warped, output_path, verbose=False):
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(rotation_image_warped)
    save_image(build_image_path("clahe.jpg", output_path), clahe_image, verbose)
    
    # Apply adaptive Gaussian thresholding to CLAHE image
    blurred_clahe = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    save_image(build_image_path("blurred_clahe.jpg", output_path), blurred_clahe, verbose)

    # Apply Canny edge detection
    canny = cv2.Canny(blurred_clahe, 100, 200)
    save_image(build_image_path("canny_2.jpg", output_path), canny, verbose)

    # Apply dilation to strengthen edges
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)
    save_image(build_image_path("dilated_2.jpg", output_path), dilated, verbose)

    return clahe_image, blurred_clahe, canny, dilated
    
    
def square_detection(image, dilated, output_path, verbose=False):
    def is_vertical(x1, y1, x2, y2, tolerance_deg=10):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return abs(angle - 90) < tolerance_deg or abs(angle + 90) < tolerance_deg

    def is_horizontal(x1, y1, x2, y2, tolerance_deg=10):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return abs(angle) < tolerance_deg or abs(angle - 180) < tolerance_deg

    def filter_close_lines(lines, threshold=10):
        lines = sorted(lines)
        filtered = []
        for line in lines:
            if not filtered or abs(line - filtered[-1]) > threshold:
                filtered.append(line)
        return filtered

    def get_best_spaced_lines(lines, expected=9):
        lines = sorted(lines)
        if len(lines) < expected:
            return lines

        best_group = lines[:expected]
        min_variance = float("inf")

        for i in range(len(lines) - expected + 1):
            group = lines[i:i + expected]
            spacings = [group[j+1] - group[j] for j in range(len(group)-1)]
            mean_spacing = sum(spacings) / len(spacings)
            variance = sum((s - mean_spacing) ** 2 for s in spacings)

            if variance < min_variance:
                min_variance = variance
                best_group = group

        return best_group

    new_image = image.copy()
    height, width = image.shape[:2]
    margin = int(min(height, width) * 0.05)  # slightly larger to cut off board border

    # Step 1: Hough Line Transform
    raw_lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    if verbose:
        debug_lines = image.copy()
        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_image(build_image_path("__HoughLinesP.jpg", output_path), debug_lines, verbose)

    # Step 2: Filter lines by length and margin
    lines = []
    if raw_lines is not None:

        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if (
                margin <= min(x1, x2) <= width - margin and
                margin <= min(y1, y2) <= height - margin and
                margin <= max(x1, x2) <= width - margin and
                margin <= max(y1, y2) <= height - margin
            ):
                lines.append([x1, y1, x2, y2])

    if verbose:
        filtered_debug = image.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(filtered_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_image(build_image_path("___filtered_lines.jpg", output_path), filtered_debug, verbose)

    if not lines:
        if verbose:
            print("No valid lines detected.")
        return None

    # Step 3: Separate into horizontal and vertical with min distance filtering
    horizontal_lines = []
    vertical_lines = []

    min_distance = 100

    for x1, y1, x2, y2 in lines:
        if is_horizontal(x1, y1, x2, y2):
            center_y = (y1 + y2) // 2

            # Check distance from existing horizontal lines
            if all(abs(center_y - existing_y) >= min_distance for existing_y in horizontal_lines):
                horizontal_lines.append(center_y)

        elif is_vertical(x1, y1, x2, y2):
            center_x = (x1 + x2) // 2

            # Check distance from existing vertical lines
            if all(abs(center_x - existing_x) >= min_distance for existing_x in vertical_lines):
                vertical_lines.append(center_x)


    # Step 4: Filter and select 9 lines
    horizontal_lines = get_best_spaced_lines(filter_close_lines(horizontal_lines), expected=9)
    vertical_lines = get_best_spaced_lines(filter_close_lines(vertical_lines), expected=9)

    if verbose:
        print(f"Filtered Horizontal: {horizontal_lines}")
        print(f"Filtered Vertical:   {vertical_lines}")

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        if verbose:
            print("Not enough grid lines detected.")
        return None
    
    # Debugging: Draw filtered lines
    debug_lines = image.copy()
    for y in horizontal_lines:
        cv2.line(debug_lines, (0, y), (width, y), (255, 0, 0), 2)
    for x in vertical_lines:
        cv2.line(debug_lines, (x, 0), (x, height), (0, 255, 0), 2)
    save_image(build_image_path("____filtered_lines.jpg", output_path), debug_lines, verbose)
    

    # Step 5: Compute intersections
    intersections = []
    for y in horizontal_lines:
        row = []
        for x in vertical_lines:
            row.append((x, y))
        intersections.append(row)

    if verbose:
        debug_intersections = image.copy()
        for row in intersections:
            for x, y in row:
                cv2.circle(debug_intersections, (int(x), int(y)), 5, (0, 0, 255), -1)
        save_image(build_image_path("_____intersections.jpg", output_path), debug_intersections, verbose)

    # Step 6: Build squares
    squares = []
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            top_left = intersections[i][j]
            top_right = intersections[i][j + 1]
            bottom_right = intersections[i + 1][j + 1]
            bottom_left = intersections[i + 1][j]
            squares.append([top_left, top_right, bottom_right, bottom_left])

    squares = [[(int(x), int(y)) for (x, y) in square] for square in squares]

    if verbose:
        debug_squares = image.copy()
        for square in squares:
            cv2.polylines(debug_squares, [np.array(square)], isClosed=True, color=(0, 255, 255), thickness=2)
        save_image(build_image_path("______squares.jpg", output_path), debug_squares, verbose)
        print(f"Total squares detected: {len(squares)}")
        
    # Print squares coordinates
    for i, square in enumerate(squares):
        print(f"Square {i}: {square}")
    

    return squares

### Bounding Boxes ### 
def point_on_square_border(pt, square, margin=5):
    x_coords = [p[0] for p in square]
    y_coords = [p[1] for p in square]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    x, y = pt

    near_left   = abs(x - min_x) <= margin
    near_right  = abs(x - max_x) <= margin
    near_top    = abs(y - min_y) <= margin
    near_bottom = abs(y - max_y) <= margin

    on_vertical_edge = (near_left or near_right) and (min_y - margin <= y <= max_y + margin)
    on_horizontal_edge = (near_top or near_bottom) and (min_x - margin <= x <= max_x + margin)

    return on_vertical_edge or on_horizontal_edge

def remove_points_on_square_borders(points, squares, margin=5):
    filtered = []
    for pt in points:
        is_on_border = False
        for square in squares:
            if point_on_square_border(pt, square, margin):
                is_on_border = True
                break
        if not is_on_border:
            filtered.append(pt)
    return np.array(filtered, dtype=np.float32)

def bounding_boxes_corner_func(image, gray, blurred, board, squares, output_path, verbose):
    board_copy = board.copy()

    ret, new_thresh_white = cv2.threshold(board_copy, 150, 255, cv2.THRESH_BINARY)
    corners = cv2.goodFeaturesToTrack(new_thresh_white, maxCorners=500, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        points = np.squeeze(corners, axis=1).astype(np.float32)
        points = remove_points_on_square_borders(points, squares)
    else:
        points = np.array([], dtype=np.float32).reshape(0, 2)

    # Optional visualization
    image_with_corners = cv2.cvtColor(board_copy, cv2.COLOR_GRAY2BGR)
    for pt in points:
        x, y = pt.astype(int)
        cv2.circle(image_with_corners, (x, y), 3, (0, 255, 0), -1)

    save_image(build_image_path("corners.jpg", output_path), image_with_corners, verbose)

    if len(points) == 0:
        print("No corners detected.")
        return []
    
    clustering = DBSCAN(eps=37, min_samples=5).fit(points)
    labels = clustering.labels_
    bounding_boxes = []

    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # Noise
            continue
        cluster_points = points[labels == label]
        x, y, w, h = cv2.boundingRect(cluster_points.astype(np.int32))
        area = cv2.contourArea(cluster_points.astype(np.int32))
        aspect_ratio = w / float(h)
        if (50 < area and 0.1 < aspect_ratio < 1.2 and y > 25): 
            bounding_boxes.append((x, y, x + w, y + h))
            cv2.rectangle(board_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #text = f"Area: {int(area)}"
            #cv2.putText(board_copy, text, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    ### Black Pieces ###
    corners_b = cv2.goodFeaturesToTrack(board_copy, maxCorners=500, qualityLevel=0.01, minDistance=10)
    if corners_b is not None:
        points = np.squeeze(corners_b, axis=1).astype(np.float32)
        points = remove_points_on_square_borders(points, squares)
    else:
        points = np.array([], dtype=np.float32).reshape(0, 2)

    def is_inside_any_box(pt):
        x, y = pt
        for (x1, y1, x2, y2) in bounding_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    filtered_points = [pt for pt in points if not is_inside_any_box(pt)]

    # Optional visualization
    image_with_corners_b = cv2.cvtColor(board_copy, cv2.COLOR_GRAY2BGR)
    for pt in filtered_points:
        x, y = pt.astype(int)
        cv2.circle(image_with_corners_b, (x, y), 3, (0, 255, 0), -1)

    save_image(build_image_path("corners_B.jpg", output_path), image_with_corners_b, verbose)
    
    filtered_points = np.array(filtered_points)
    clustering_b = DBSCAN(eps=37, min_samples=5).fit(filtered_points)
    labels_b = clustering_b.labels_
    unique_labels_b = set(labels_b)
    for label in unique_labels_b:
        if label == -1:  # Noise
            continue
        cluster_points_b = filtered_points[labels_b == label]
        x, y, w, h = cv2.boundingRect(cluster_points_b.astype(np.int32))
        area = cv2.contourArea(cluster_points_b.astype(np.int32))
        aspect_ratio = w / float(h)
        if (50 < area and 0.1 < aspect_ratio < 1.2): 
            bounding_boxes.append((x, y, x + w, y + h))
            cv2.rectangle(board_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #text = f"Area: {int(area)}"
            #cv2.putText(board_copy, text, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    save_image(build_image_path("bounding_boxes_corner.jpg", output_path), board_copy, verbose)
    return bounding_boxes

### PIECE DETECTION ### 

# morphological closing is a dilation operation followed by an erosion operation, which fills in any space between the edges. -> see: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def method_a(square_detection_canny, kernel): 
    closed_edges = cv2.morphologyEx(square_detection_canny, cv2.MORPH_CLOSE, kernel)

    return closed_edges

# applie morphological closing to the contrasted image before applying Canny edge detection and dilation
def method_b(board_image, kernel): 
    closed_edges = cv2.morphologyEx(board_image, cv2.MORPH_CLOSE, kernel)

    blurred = cv2.GaussianBlur(closed_edges, (5, 5), 0)
    canny = cv2.Canny(blurred, 100, 200) 
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

    return dilated

# We then determine the average intensity of each square in the occupancy grid, comparing it to an
# empirically determined threshold to determine if there is a piece present or not. A parameter sweep
# in increments of 10 was conducted to find a suitable value. Under standard lighting conditions, a
# threshold value of 70 out of 255 was suitable
def piece_positions(board_image, squares, raw_image, threshold=17):
    piece_positions = []
    for square in squares:
        contour = np.array(square, dtype=np.int32).reshape((-1, 1, 2))
        
        mask = np.zeros_like(board_image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        avg_intensity = cv2.mean(board_image, mask=mask)[0]
        
        if avg_intensity > threshold:
            centroid = np.mean(square, axis=0)
            cx, cy = centroid.astype(int)
            
            if raw_image is not None:
                pixel_value = raw_image[cy, cx]
                piece_color = "black" if pixel_value < 128 else "white"
            else:
                piece_color = "unknown"

            piece_positions.append((cx, cy, piece_color))
    
    return piece_positions

# i need to somehow get square_detection_canny
def piece_detection(board_image, square_detection_canny, squares, raw_image): 
    kernel = np.ones((5,5),np.uint8) 
    edges_method_a = method_a(square_detection_canny, kernel)
    edges_method_b = method_b(board_image, kernel)

    # merge the methods
    merged_methods = cv2.bitwise_and(edges_method_a, edges_method_b)

    piece_position = piece_positions(merged_methods, squares, raw_image)

    return piece_position, merged_methods

def visualize_piece_positions(board_image, piece_positions, output_path, verbose=False):
    # Convert grayscale to BGR so we can draw in color
    if len(board_image.shape) == 2:
        display_img = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    else:
        display_img = board_image.copy()
    
    for cx, cy, piece_color in piece_positions:
        color = (0, 0, 0) if piece_color == "black" else (255, 255, 255)
        text_color = (255, 255, 255) if piece_color == "black" else (0, 0, 0)
        
        # Draw a circle at the piece position
        cv2.circle(display_img, (int(cx), int(cy)), 10, color, -1)
        cv2.putText(display_img, piece_color, (int(cx) - 20, int(cy) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    save_image(build_image_path("detected_pieces.jpg", output_path), display_img, verbose)

### PIECE DETECTION ### 

def run_pipeline(image, output_path, verbose=False):
    set_image_num(get_image_num() + 1)
    # Preprocess image
    gray_image, blurred_image, hsv_image = preprocess_image(image, output_path, verbose)

    # Board detection
    board_image, M, rotation = board_detection(blurred_image, gray_image, hsv_image, output_path, verbose)

    # Square Detection
    clahe, blurred_clahe, canny, dilated = square_detection_processed(board_image, output_path, verbose)
    squares = square_detection(board_image, dilated, output_path, verbose)

    # Piece detection with bounding boxes
    bounding_boxes = bounding_boxes_corner_func(image, gray_image, blurred_image, board_image, squares, output_path, verbose)
    #bounding_boxes = bounding_boxes_segment_func(image, gray_image, blurred_image, board_image, squares, output_path, verbose)
    # Piece Classification (white or black)
    pieces, merged_methods = piece_detection(clahe, canny, squares, board_image)
    visualize_piece_positions(board_image, pieces, output_path, verbose=True)

    # Map into 8x8 grid

    # Output data
    output_data = {}
    return output_data

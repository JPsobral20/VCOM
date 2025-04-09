import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

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
    
    return None, None

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
    
def square_detection_processed(gray_image, output_path, verbose=False):
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    save_image(build_image_path("clahe.jpg", output_path), clahe_image, verbose)
    
    # Apply adaptive Gaussian thresholding to CLAHE image
    blurred_clahe = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    save_image(build_image_path("blurred_clahe.jpg", output_path), blurred_clahe, verbose)
    
    # Apply Cannny edge detection
    canny = cv2.Canny(blurred_clahe, 100, 200)
    save_image(build_image_path("canny_2.jpg", output_path), canny, verbose)
    
    # Apply dilation
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)
    save_image(build_image_path("dilated_2.jpg", output_path), dilated, verbose)
    
    return clahe_image, canny, dilated
    
    
def square_detection(dilated, output_path, verbose=False):
    # Edge detection
    edges = cv2.Canny(dilated, 50, 150)
    
    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("No lines detected.")
        return None

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < abs(x1 - x2):  # Horizontal line
            horizontal_lines.append((y1 + y2) // 2)
        else:  # Vertical line
            vertical_lines.append((x1 + x2) // 2)

    # Remove near-duplicates (within threshold)
    def filter_close_lines(lines, threshold=10):
        lines = sorted(lines)
        filtered = []
        for line in lines:
            if not filtered or abs(line - filtered[-1]) > threshold:
                filtered.append(line)
        return filtered

    horizontal_lines = filter_close_lines(horizontal_lines)
    vertical_lines = filter_close_lines(vertical_lines)

    if len(horizontal_lines) < 9 or len(vertical_lines) < 9:
        print("Not enough grid lines detected.")
        return None

    # Compute intersections
    intersections = []
    for y in horizontal_lines:
        row = []
        for x in vertical_lines:
            row.append((x, y))
        intersections.append(row)

    # Build squares from 4 intersection points
    squares = []
    for i in range(8):
        for j in range(8):
            top_left = intersections[i][j]
            top_right = intersections[i][j+1]
            bottom_right = intersections[i+1][j+1]
            bottom_left = intersections[i+1][j]
            squares.append([top_left, top_right, bottom_right, bottom_left])
    
    
    squares = [
    [(int(x), int(y)) for (x, y) in square]
    for square in squares
    ]

    if verbose:
        print(f"Total squares detected: {len(squares)}")
        for square in squares:
            print(square)

    return squares

### PIECE DETECTION ### 

# morphological closing is a dilation operation followed by an erosion operation, which fills in any space between the edges. -> see: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def method_a(square_detection_canny, kernel): # this should receive the canny from square detection
    closed_edges = cv2.morphologyEx(square_detection_canny, cv2.MORPH_CLOSE, kernel)

    return closed_edges

# applie morphological closing to the contrasted image before applying Canny edge detection and dilation
def method_b(gray_image, kernel): 
    closed_edges = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    canny = cv2.Canny(closed_edges, 100, 200) 
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

    return dilated

# We then determine the average intensity of each square in the occupancy grid, comparing it to an
# empirically determined threshold to determine if there is a piece present or not. A parameter sweep
# in increments of 10 was conducted to find a suitable value. Under standard lighting conditions, a
# threshold value of 70 out of 255 was suitable
def piece_positions(gray_image, squares, threshold=70):
    piece_positions = []
    for square in squares:
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [square], -1, 255, -1)
        avg_intensity = cv2.mean(gray_image, mask=mask)[0]
        if avg_intensity < threshold: # there's a piece
            # take the intensity of the centroid of the square from the original image to determine whether the piece is white or black based on another preset threshold
            cx, cy = np.mean(square, axis=0)[0]
            piece_color = "black" if gray_image[int(cy), int(cx)] < 128 else "white"
            piece_positions.append((cx, cy, piece_color))
    return piece_positions

# i need to somehow get square_detection_canny
def piece_detection(gray_image, square_detection_canny): 
    kernel = np.ones((5,5),np.uint8) 
    method_a = method_a(square_detection_canny, kernel)
    method_b = method_b(gray_image, kernel)

    # merge the methods
    merged_methods = cv2.bitwise_and(method_a, method_b)

    # square detection with merged edges
    squares = square_detection(merged_methods) # i'm not sure what to give to square dection, i need your help sobral

    piece_positions = piece_positions(gray_image, squares)

    return piece_positions

### PIECE DETECTION ### 

def run_pipeline(image, output_path, verbose=False):
    set_image_num(get_image_num() + 1)
    # Preprocess image
    gray_image, blurred_image, hsv_image = preprocess_image(image, output_path, verbose)

    # Board detection
    board_image, M, rotation = board_detection(blurred_image, gray_image, hsv_image, output_path, verbose)

    # # Square Detection (INCOMPLETE)
    # clahe, canny, dilated = square_detection_processed(gray_image, output_path, verbose)
    # squares = square_detection(dilated, output_path, verbose)

    # Piece detection with bounding boxes

    # Piece Classification (white or black)
    # pieces = piece_detection(gray, canny, squares)

    # Map into 8x8 grid

    # Output data
    output_data = {}
    return output_data

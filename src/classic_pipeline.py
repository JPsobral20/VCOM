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

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (11,11), 0)
    save_image(build_image_path("blurred.jpg", output_path), blurred, verbose)

    return gray, blurred

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

    grid_size = 70
    half_window = grid_size // 2

    corners = [
        [half_window, half_window, cv2.ROTATE_90_COUNTERCLOCKWISE],
        [new_thresh.shape[1] - half_window, half_window, cv2.ROTATE_180],
        [half_window, new_thresh.shape[0] - half_window, None],
        [new_thresh.shape[1] - half_window, new_thresh.shape[0] - half_window, cv2.ROTATE_90_CLOCKWISE]
    ]

    for corner in corners:
        center_x, center_y, rotation = corner
        x_start, x_end = max(0, center_x - half_window), min(new_thresh.shape[1], center_x + half_window)
        y_start, y_end = max(0, center_y - half_window), min(new_thresh.shape[0], center_y + half_window)
        window = new_thresh[y_start:y_end, x_start:x_end]
        pixel_sum = np.sum(window)
        thresh_value = grid_size * grid_size * 0.6

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
    
    return None, None



def board_detection(image, output_path, verbose=False):
    # Threshold the image using Otsu's method
    ret, otsu = cv2.threshold(image, 203, 255, cv2.THRESH_BINARY)
    save_image(build_image_path("otsu.jpg", output_path), otsu, verbose)
    
    canny = cv2.Canny(otsu, 0, 200)
    save_image(build_image_path("canny.jpg", output_path), canny, verbose)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Hough Transform
    lines = cv2.HoughLinesP(canny, 1, 1*np.pi/180, 100, minLineLength=100, maxLineGap=120)
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    save_image(build_image_path("lines.jpg", output_path), line_image, verbose)
    # separate the lines into two groups
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < abs(x2 - x1):
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)
    
    for i in range(len(horizontal_lines)):
        x1, y1, x2, y2 = horizontal_lines[i][0]
        for j in range(i + 1, len(horizontal_lines)):
            x3, y3, x4, y4 = horizontal_lines[j][0]

    

    # Draw lines
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(color, (x1, y1), (x2, y2), (255, 0, 0), 2)
    save_image(build_image_path("lines_separated.jpg", output_path), color, verbose)

    # Get line intersections
    points = []
    for i in range(len(horizontal_lines)):
        for j in range(len(vertical_lines)):
            x1, y1, x2, y2 = horizontal_lines[i][0]
            x3, y3, x4, y4 = vertical_lines[j][0]
            p = line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            # check if point is integer
            if p is not None:
                p = np.round(p).astype(int)
                points.append(p)
    # Draw points
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(color, tuple(point), 5, (0, 0, 255))
    save_image(build_image_path("points.jpg", output_path), color, verbose)
    
    # percentage_w = 0.05
    # percentage_h = 0.05

    # # Remove points that are too close to border of the image
    # h, w = image.shape
    # for i in range(len(points)):
    #     x, y = points[i]
    #     if x < percentage_w * w or x > (1 - percentage_w) * w or y < percentage_h * h or y > (1 - percentage_h) * h:
    #         points[i] = None
    # points = [p for p in points if p is not None]

    # points = np.array(points)
    # # draw points
    # for point in points:
    #     cv2.circle(color, tuple(point), 5, (0, 0, 255))
    # save_image(build_image_path("points.jpg", output_path), color, verbose)

    points = np.array(points)
    # Find the four corners of the board
    if len(points) < 4:
        print("Not enough points found")
        return None
    square = cv2.convexHull(points, clockwise=True)
    square = square.reshape(-1, 2)
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

    dst_points = np.array([[0, 1000], [1000, 1000], [1000, 0], [0, 0]], dtype="float32")
    # Perform the perspective transform
    M = cv2.getPerspectiveTransform(square.astype("float32"), dst_points)
    warped = cv2.warpPerspective(image, M, (1000, 1000))
    save_image(build_image_path("warped.jpg", output_path), warped, verbose)

    warped, rotation = find_rotation(warped, output_path, verbose)    
    
def square_detection(blurred_image, gray_image, output_path, verbose=False):

    # Apply adaptive thresholding
    adaptive = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    save_image(build_image_path("adaptive.jpg", output_path), adaptive, verbose)

    # Find contours
    contours, hierarchy = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply Otus thresholding to gray image
    ret, otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #save_image(build_image_path("otsu.jpg", output_path), otsu, verbose)
    
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
    
    # Dynamic Hough Line Transform
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    line_image = np.zeros_like(gray_image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    save_image(build_image_path("lines.jpg", output_path), line_image, verbose)
    



    # Filter contours based on area
    #min_area = 1000
    #max_area = 10000
    #filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Draw contours
    #square_image = np.zeros_like(image)
    #cv2.drawContours(square_image, filtered_contours, -1, 255, 2)
    # save_image(build_image_path("square.jpg", output_path), square_image, verbose)



def run_pipeline(image, output_path, verbose=False):
    set_image_num(get_image_num() + 1)
    # Preprocess image
    gray_image, blurred_image = preprocess_image(image, output_path, verbose)

    # Board detection
    board_image = board_detection(blurred_image, output_path, verbose)

    # Square Detection (INCOMPLETE)
    # square_image = square_detection(blurred_image, gray_image, output_path, verbose)

    # Piece detection with bounding boxes

    # Piece Classification (white or black)

    # Map into 8x8 grid

    # Output data
    output_data = {}
    return output_data

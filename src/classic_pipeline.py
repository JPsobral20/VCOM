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
    blurred = cv2.GaussianBlur(gray, (15,15), 0)
    save_image(build_image_path("blurred.jpg", output_path), blurred, verbose)

    return gray, blurred

def board_detection(image, output_path, verbose=False):
    # Threshold the image using Otsu's method
    ret, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_image(build_image_path("otsu.jpg", output_path), otsu, verbose)
    
    canny = cv2.Canny(otsu, 100, 200)
    save_image(build_image_path("canny.jpg", output_path), canny, verbose)

    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)
    save_image(build_image_path("dilated.jpg", output_path), dilated, verbose)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
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



        

def run_pipeline(image, output_path, verbose=False):
    set_image_num(get_image_num() + 1)
    # Preprocess image
    gray_image, blurred_image = preprocess_image(image, output_path, verbose)

    # Board detection
    board_image = board_detection(blurred_image, output_path, verbose)

    # Square Detection (INCOMPLETE)
    clahe, canny, dilated = square_detection_processed(gray_image, output_path, verbose)
    squares = square_detection(dilated, output_path, verbose)

    # Piece detection with bounding boxes

    # Piece Classification (white or black)

    # Map into 8x8 grid

    # Output data
    output_data = {}
    return output_data

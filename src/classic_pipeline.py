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
    save_image(build_image_path("adaptive_clahe.jpg", output_path), blurred_clahe, verbose)
    
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
    square_image = square_detection(blurred_image, gray_image, output_path, verbose)

    # Piece detection with bounding boxes

    # Piece Classification (white or black)

    # Map into 8x8 grid

    # Output data
    output_data = {}
    return output_data

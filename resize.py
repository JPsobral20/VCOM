import argparse
import json
import sys
from pathlib import Path
import os
import cv2
import numpy as np

def resize_image(image_path, output_path, width, height):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False

    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, resized_image)
    return True

input_dir = Path("input")
output_dir = Path("output")
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
target_size = (256, 256)
def process_images(input_dir, output_dir, target_size):
    for dir in input_dir.iterdir():
        if not dir.is_dir():
            continue
        output_path = output_dir / dir.name
        for image_path in dir.iterdir():
            output_image_path = output_path / image_path.name
            output_path.mkdir(parents=True, exist_ok=True)
            if resize_image(image_path, output_image_path, *target_size):
                print(f"Resized {image_path} to {output_image_path}")
            else:
                print(f"Failed to resize {image_path}")

if __name__ == "__main__":
    process_images(input_dir, output_dir, target_size)
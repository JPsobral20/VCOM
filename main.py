import argparse
import json
import sys
from pathlib import Path
import os
import cv2

from src import classic_pipeline as classic

def application(list_of_images, mode, verbose=False, out="."):
    output_data = []

    if mode == "classic":
        print("Running classic computer vision...\n")
        for i, image_path in enumerate(list_of_images):
            print(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            
            num_pieces, board, bounding_boxes = classic.run_pipeline(image, output_path=out, verbose=verbose)
            if verbose:
                print(f"Output data: {num_pieces}, {board}, {bounding_boxes}")
            output_data.append({
                "image": image_path,
                "num_pieces": num_pieces,
                "board": board,
                "detected_pieces": bounding_boxes
            })
    elif mode == "deep":
        print("Running deep learning...")
        print("Not implemented yet.")
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chessboard detection")

    # Add argument for json input file
    parser.add_argument(
        "--input",
        type=str,
        default="./data/input.json",
        help="Path to the input JSON file. Path is relative to the project root directory.",
    )

    # Add argument for json output file
    parser.add_argument(
        "--output",
        type=str,
        default="./output.json",
        help="Path to the output JSON file. Path is relative to the project root directory.",
    )

    # Add argument for mode of operation
    parser.add_argument(
        "--mode",
        type=str,
        default="classic",
        choices=["classic", "deep"],
        help="Mode of operation. 'classic' for classic computer vision, 'deep' for deep learning.",
    )

    # Add argument for verbose output
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output.",
    )

    # Add argument for verbose output directory
    parser.add_argument(
        "--out",
        type=str,
        default="./output",
        help="Path to the output directory. Path is relative to the project root directory.",
    )

    try:
        # Parse the command-line arguments
        args = parser.parse_args()

        input_file = args.input
        output_file = args.output
        mode = args.mode
        verbose = args.verbose
        out = args.out

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out).mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Input file: {input_file}")
            print(f"Output file: {output_file}")
            print(f"Mode: {mode}")
            print(f"Verbose: {verbose}")
            print(f"Output directory: {out}")
        
        # Read the input JSON file
        with open(input_file, "r") as f:
            input_data = json.load(f)

        if verbose:
            print(f"Input data: {input_data}")
        
        # Get the list of images
        list_of_images = input_data["image_files"]

        # Run the application
        output_data = application(list_of_images, mode, verbose, out)

        # Write the output data to the output JSON file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        if verbose:
            print(f"Output data: {output_data}")
            print(f"Output data written to {output_file}")
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        sys.exit(1)        

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON input. Unable to parse Solr response.")
        sys.exit(1)
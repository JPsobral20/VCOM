# VCOM – Chessboard and Chess Piece Detection

A computer vision project focused on detecting chessboards and identifying chess pieces in images using classical image processing techniques and deep learning.

The project implements two complementary approaches: a traditional computer vision pipeline based on image processing and geometric analysis, and a deep learning pipeline using a fine-tuned MobileNetV2 model to estimate the number of pieces on a chessboard.

## Overview

The system processes input images and produces structured JSON results containing information about the detected chessboard and its pieces.

Two processing modes are available:

- **Classic Computer Vision:** Uses image preprocessing, board detection, perspective transformation, grid and square detection, and piece identification.
- **Deep Learning:** Uses a fine-tuned MobileNetV2 regression model to estimate the number of chess pieces in an image.

## Features

- Chessboard detection and perspective correction.
- Image preprocessing using grayscale conversion, Gaussian blur, and color-space transformations.
- Chessboard grid and square detection using edge detection and line analysis.
- Piece detection using image segmentation, contours, and color-based filtering.
- Bounding box generation for detected pieces.
- Deep learning-based piece counting using MobileNetV2.
- Configurable input and output JSON files.
- Optional verbose mode for generating intermediate images and debugging visualizations.
- Image resizing utility for preparing training data.

## Technologies

- **Python**
- **OpenCV** – Image processing and computer vision.
- **NumPy** – Numerical operations and image manipulation.
- **Matplotlib** – Visualization.
- **PyTorch** – Deep learning model implementation and inference.
- **Torchvision** – MobileNetV2 architecture and image transformations.
- **Scikit-learn** – Machine learning utilities.

## Project Structure

```text
VCOM/
├── data/
│   ├── input.json
│   ├── input2.json
│   ├── input3.json
│   ├── input4.json
│   ├── input5.json
│   └── input_deep.json
│
├── images/                 # Sample input images
│
├── models/
│   └── MobileNet_V2_regression_base_multiclass_finetune_best_model.pth
│
├── src/
│   ├── classic_pipeline.py # Traditional computer vision pipeline
│   └── deep_pipeline.py    # Deep learning pipeline
│
├── main.py                 # Main application entry point
├── resize.py               # Image resizing utility
├── task2_baseline.ipynb    # Model training and experimentation
├── requirements.txt
└── output.json             # Example output
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/JPsobral20/VCOM.git
cd VCOM
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The deep learning pipeline also requires PyTorch and Torchvision. Scikit-learn is used by the classic pipeline.

```bash
pip install torch torchvision scikit-learn
```

For GPU acceleration, install the PyTorch version compatible with your CUDA environment.

## Usage

Run the application from the project root directory:

```bash
python main.py
```

The application accepts the following command-line arguments:

| Argument | Description | Default |
|---|---|---|
| `--input` | Path to the input JSON file | `./data/input_deep.json` |
| `--output` | Path to the output JSON file | `./output.json` |
| `--mode` | Processing mode: `classic` or `deep` | `deep` |
| `--verbose` | Enables detailed output and intermediate image generation | Disabled |
| `--out` | Directory for generated intermediate images | `./output` |
| `--model` | Path to the trained model | `./models/MobileNet_V2_regression_base_multiclass_finetune_best_model.pth` |

### Processing Modes

**1. Classic Computer Vision**

```bash
python main.py --input=./data/input.json --output=./output.json --mode=classic
```

This pipeline performs chessboard detection, perspective correction, grid analysis, piece identification, and bounding box generation.

**2. Deep Learning**

```bash
python main.py --input=./data/input_deep.json --output=./output.json --mode=deep
```

This pipeline loads the trained MobileNetV2 model and estimates the number of chess pieces in each input image.

### Verbose Mode

To generate intermediate images and additional processing information for debugging:

```bash
python main.py --mode=classic --verbose
```

Intermediate visualizations are saved in the directory specified by `--out`. This mode may generate a large number of files.

## Input Format

The application expects a JSON file containing a list of image paths:

```json
{
    "image_files": [
        "images/G056_IMG097.jpg"
    ]
}
```

Image paths are resolved relative to the project root directory.

## Output

The application writes the results to a JSON file.

In **classic mode**, each processed image includes:

- Image path.
- Number of detected pieces.
- An 8×8 board matrix.
- Bounding boxes for detected pieces.

In **deep learning mode**, each processed image includes:

- Image path.
- Estimated number of pieces.

## Model Training

The `task2_baseline.ipynb` notebook contains the model training experiments and the selection of the best-performing model.

The deep learning pipeline uses a fine-tuned MobileNetV2 architecture with a regression output to estimate the number of chess pieces.

The trained model is stored in the `models/` directory and loaded during inference.

## Image Preprocessing

The `resize.py` utility resizes images to 256×256 pixels using cubic interpolation.

It processes images organized in subdirectories under an `input/` directory and saves the resized images under `output/`, preserving the directory structure.

## Academic Context

Developed as part of the **VCOM** academic project, exploring classical computer vision and deep learning techniques for chessboard image analysis.

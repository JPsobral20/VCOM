import numpy as np, cv2, random, torch, os
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2 as transforms
random.seed(42)

data_aug = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_path):
    """
    Load an image from the specified path and apply data augmentation.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        torch.Tensor: The processed image tensor.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return data_aug(image)  # Apply data augmentation transformations

def load_model(model_file):
    """
    Load a pre-trained model from the specified file.
    
    Args:
        model_file (str): Path to the model file.
        
    Returns:
        nn.Module: The loaded model.
    """
    if not os.path.exists(model_file):
        print(f"Model file {model_file} does not exist.")
        return None
    
    print(f"Loading model from {model_file} on device {device}")
    model = models.efficientnet_b2(pretrained=True)
    model.classifier[1] = nn.Linear(1408, 1)

    model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=device)['model'])
    model.eval()  # Set the model to evaluation mode
    return model

def run_pipeline(image, model, verbose=False):
    """
    Run the deep learning pipeline on the provided image.
    
    Args:
        image (torch.Tensor): The input image tensor.
        model (nn.Module): The pre-trained model.
        verbose (bool): If True, print additional information.
        
    Returns:
        int: The number of detected pieces in the image.
    """

    image = image.to(device)  # Move image to the appropriate device
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    
    num_pieces = int(torch.round(output).item())
    
    if verbose:
        print(f"Detected {num_pieces} pieces in the image.")
    
    # Remove image from GPU memory if it was moved there without removing the model
    del image
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return num_pieces
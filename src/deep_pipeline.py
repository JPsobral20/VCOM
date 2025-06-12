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
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return data_aug(image) 

def load_model(model_file):
    if not os.path.exists(model_file):
        print(f"Model file {model_file} does not exist.")
        return None
    
    print(f"Loading model from {model_file} on device {device}")
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 1) 

    model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=device)['model'])
    model.eval() 
    return model

def run_pipeline(image, model, verbose=False):
    image = image.to(device) 
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
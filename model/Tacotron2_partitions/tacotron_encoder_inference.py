# inference.py
import torch
import torch.nn.functional as F
from PIL import Image
import io

# Load the model
model
model.eval()

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, predicted = outputs.max(1)
    return predicted.item()

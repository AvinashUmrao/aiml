"""Shared utilities for screen recapture detection."""

import os
import sys
import logging
import torch
from PIL import Image
from torchvision import transforms
from model import MrcNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CHECKPOINT_PATH = './log/best_model.pth'
DEFAULT_PATCH_SIZE = 96
DEFAULT_NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
DEFAULT_DATASET_ROOT = './dataset/'


def get_device():
    """Get the device to use (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transform(patch_size=DEFAULT_PATCH_SIZE, normalize=DEFAULT_NORMALIZE, num_crops=10):
    """
    Get image transform for inference.
    
    Args:
        patch_size: Size of image patches
        normalize: Normalization parameters
        num_crops: Number of crops (10 for TenCrop, 5 for FiveCrop)
    
    Returns:
        Transform composition
    """
    if num_crops == 10:
        crop_transform = transforms.TenCrop(patch_size)
    elif num_crops == 5:
        crop_transform = transforms.FiveCrop(patch_size)
    else:
        crop_transform = transforms.CenterCrop(patch_size)
    
    return transforms.Compose([
        crop_transform,
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])


def load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=None):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on (auto-detected if None)
    
    Returns:
        Loaded model in eval mode
    
    Raises:
        SystemExit: If checkpoint not found or loading fails
    """
    if device is None:
        device = get_device()
    
    try:
        model = MrcNet()
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)


def validate_image(image_path, min_size=96):
    """
    Validate image file and size.
    
    Args:
        image_path: Path to image file
        min_size: Minimum required image dimension
    
    Returns:
        PIL Image if valid, None otherwise
    
    Raises:
        SystemExit: If file not found
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)
    
    try:
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        if img.size[0] < min_size or img.size[1] < min_size:
            logger.warning(f"Image {image_path} is smaller than {min_size}x{min_size}, skipping")
            return None
        
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def predict_image(image_path, model, transform, device=None):
    """
    Predict screen probability for a single image.
    
    Args:
        image_path: Path to image file
        model: Loaded model
        transform: Image transform
        device: Device to use (auto-detected if None)
    
    Returns:
        Tuple of (screen_probability, confidence) or (None, None) if error
    """
    if device is None:
        device = get_device()
    
    img = validate_image(image_path)
    if img is None:
        return None, None
    
    try:
        test_input = transform(img)
        test_input = test_input.to(device)
        
        with torch.no_grad():
            ncrops, c, h, w = test_input.size()
            output = model(test_input.view(-1, c, h, w))
            pred = torch.nn.functional.softmax(output, dim=1)
            mean = torch.mean(pred, dim=0)
            screen_probability = mean[1].item()
            confidence = max(mean[0].item(), mean[1].item())
        
        return screen_probability, confidence
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None, None

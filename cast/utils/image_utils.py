"""
Image processing utilities
"""
import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Tuple, Optional, Union
from pathlib import Path

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from file path"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save image to file path"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """Resize image to target size"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if keep_aspect_ratio:
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return result
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def crop_image_with_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float],
                        padding: int = 0) -> np.ndarray:
    """Crop image using bounding box coordinates"""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Add padding and clamp to image bounds
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    
    return image[y1:y2, x1:x2]

def image_to_base64(image: np.ndarray, format: str = "PNG") -> str:
    """Convert numpy image to base64 string"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_data))
    return np.array(pil_image)

def create_mask_from_bbox(image_shape: Tuple[int, int], 
                         bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Create a binary mask from bounding box"""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Clamp coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    mask[y1:y2, x1:x2] = 255
    return mask

def apply_mask(image: np.ndarray, mask: np.ndarray, 
              background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Apply mask to image"""
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Create 3-channel mask
    mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0
    
    # Create background
    background = np.full_like(image, background_color, dtype=image.dtype)
    
    # Apply mask
    result = image * mask_3ch + background * (1 - mask_3ch)
    return result.astype(image.dtype)

def normalize_image(image: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Normalize image to target range"""
    min_val, max_val = target_range
    
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Normalize to [0, 1] first
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    
    # Scale to target range
    image = image * (max_val - min_val) + min_val
    
    return image
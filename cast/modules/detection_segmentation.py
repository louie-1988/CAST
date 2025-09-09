"""
Object Detection and Segmentation Module

This module handles combined object recognition, detection and segmentation using RAM-Grounded-SAM.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import cv2
from pathlib import Path

from ..core.common import DetectedObject, BoundingBox
from ..utils.api_clients import ReplicateClient
from ..utils.image_utils import crop_image_with_bbox, save_image, image_to_base64
from ..config.settings import config

class DetectionSegmentationModule:
    """Module for object detection and segmentation using RAM-Grounded-SAM"""
    
    def __init__(self):
        """Initialize the detection and segmentation module"""
        self.replicate_client = ReplicateClient()
    
    def detect_and_segment_objects(self, image: np.ndarray, 
                                  use_sam_hq: bool = True) -> List[DetectedObject]:
        """
        Detect and segment objects using RAM-Grounded-SAM in a single call
        
        Args:
            image: Input RGB image as numpy array
            use_sam_hq: Use sam_hq instead of SAM for prediction
            
        Returns:
            List of detected objects with bounding boxes, descriptions, and masks
        """
        print("Running RAM-Grounded-SAM for detection and segmentation...")
        
        # Run RAM-Grounded-SAM with visualization enabled to get masks
        ram_result = self.replicate_client.run_ram_grounded_sam(
            image, use_sam_hq=use_sam_hq, show_visualisation=True
        )
        
        if not ram_result:
            print("Warning: RAM-Grounded-SAM failed")
            return []
        
        # Store the result for potential saving
        self._last_ram_result = ram_result
        
        detected_objects = []
        
        # Parse RAM-Grounded-SAM results based on the actual API response format
        # Expected format: {'tags': str, 'json_data': {...}, 'masked_img': file, 'rounding_box_img': file}
        json_data = ram_result.get("json_data", {})
        tags = ram_result.get("tags", "")
        masked_img = ram_result.get("masked_img")
        assert masked_img is not None, "Masked image is None"
        
        print(f"Detected tags: {tags}")
        
        # Download and process the masked image to extract individual masks
        mask_image = None
        # Handle different response formats
        if hasattr(masked_img, 'read'):
            # File-like object from Replicate
            mask_data = masked_img.read()
            from PIL import Image
            import io
            mask_pil = Image.open(io.BytesIO(mask_data))
            mask_image = np.array(mask_pil.convert('RGB'))
        elif isinstance(masked_img, str) and masked_img.startswith('http'):
            # URL response
            import requests
            response = requests.get(masked_img)
            assert response.status_code == 200, "Failed to download masked image"
            from PIL import Image
            import io
            mask_pil = Image.open(io.BytesIO(response.content))
            mask_image = np.array(mask_pil.convert('RGB'))
        
        assert json_data is not None and "mask" in json_data, "Mask data not found in json_data"
        # Extract mask data from json_data
        mask_items = json_data["mask"]
        num_classes = max([mask_item.get("value", 0) for mask_item in mask_items]) + 1
        # Verify the correctness of mask_image 
        uvalues = np.unique(mask_image.reshape(-1, 3), axis=0)
        assert len(uvalues) == num_classes, "Number of unique values in mask_image does not match num_classes"
        for uindex, uvalue in enumerate(uvalues):
            mask_image[np.where(mask_image == uvalue)] = uindex

        # Skip the background entry (value=0) and process actual objects
        for i, mask_item in enumerate(mask_items):
            if mask_item.get("value", 0) == 0:  # Skip background
                continue
                
            # Extract bounding box in format [x1, y1, x2, y2]
            if "box" not in mask_item:
                continue
                
            bbox_coords = mask_item["box"]
            if len(bbox_coords) < 4:
                continue
            
            bbox = BoundingBox(
                x1=float(bbox_coords[0]),
                y1=float(bbox_coords[1]),
                x2=float(bbox_coords[2]),
                y2=float(bbox_coords[3]),
                confidence=mask_item.get("logit", 1.0)
            )
            
            # Extract label and confidence
            label = mask_item.get("label", f"object_{mask_item.get('value', 0)}")
            confidence = mask_item.get("logit", 1.0)
            mask_value = mask_item.get("value", 0)
            
            # Create detected object
            detected_obj = DetectedObject(
                id=mask_value,
                bbox=bbox,
                description=label,
                confidence=float(confidence)
            )
            
            # Crop the object from the image
            crop_coords = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            detected_obj.cropped_image = crop_image_with_bbox(image, crop_coords)
            
            # Extract actual mask from the masked image if available
            detected_obj.mask, detected_obj.occ_mask = self._extract_mask_from_colored(mask_image, mask_value, num_classes, image.shape[:2])
            assert detected_obj.mask.shape[:2] == image.shape[:2], "Mask shape does not match image shape"
            
            # Create cropped mask for the object
            detected_obj.cropped_mask = crop_image_with_bbox(detected_obj.mask, crop_coords)
            detected_obj.cropped_occ_mask = crop_image_with_bbox(detected_obj.occ_mask, crop_coords)

            # we blend the cropped image with a purely black background using the mask 
            # also we add an alpha channel 
            # detected_obj.cropped_image = detected_obj.cropped_image * detected_obj.cropped_mask[:, :, np.newaxis] + (1 - detected_obj.cropped_mask[:, :, np.newaxis]) * np.zeros_like(detected_obj.cropped_image)
            # detected_obj.cropped_image = np.concatenate([detected_obj.cropped_image, 255 * detected_obj.cropped_mask[:, :, np.newaxis].astype(np.uint8)], axis=2)

            detected_objects.append(detected_obj)

        print(f"RAM-Grounded-SAM detected and segmented {len(detected_objects)} objects")
        return detected_objects
    
    def _extract_mask_from_colored(self, mask_image: np.ndarray, mask_value: int,  num_classes: int,
                                  target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract a binary mask for a specific object from the colored mask image
        
        Args:
            mask_image: RGB colored mask image from RAM-Grounded-SAM
            mask_value: Integer value representing the object in the mask
            target_shape: Target shape (height, width) for the output mask
            
        Returns:
            Binary mask for the specified object
        """
        # Convert to grayscale if needed
        if len(mask_image.shape) == 3:
            # Use the red channel or convert to grayscale
            gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_mask = mask_image
        
        # Resize to target shape if needed
        if gray_mask.shape != target_shape:
            gray_mask = cv2.resize(gray_mask, (target_shape[1], target_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Create binary mask for the specific object
        # RAM-Grounded-SAM typically uses different colors/intensities for different objects
        # We'll use thresholding to extract the object mask
        # binary_mask = np.zeros(target_shape, dtype=np.uint8)
        
        # # Find unique values in the mask to identify object regions
        # unique_vals = np.unique(gray_mask)
        # if len(unique_vals) > 1:
        #     # Use the mask_value to select the appropriate threshold
        #     # If mask_value is within the unique values, use it directly
        #     if mask_value < len(unique_vals) and mask_value > 0:
        #         threshold_val = unique_vals[mask_value] if mask_value < len(unique_vals) else unique_vals[-1]
        #         binary_mask = (gray_mask == threshold_val).astype(np.uint8) * 255
        #     else:
        #         # Fallback: use any non-zero regions
        #         binary_mask = (gray_mask > 0).astype(np.uint8) * 255
        binary_mask = (gray_mask == mask_value).astype(np.uint8) * 255
        # also we extract the occlusion mask, where the object and the background marked as black, else marked as white 
        occ_mask = np.ones(target_shape, dtype=np.uint8) * 255
        occ_mask[gray_mask == 0] = 0
        occ_mask[gray_mask == mask_value] = 0

        return binary_mask, occ_mask
    
    def run(self, image: np.ndarray, output_dir: Optional[Path] = None,
            use_sam_hq: bool = False) -> List[DetectedObject]:
        """
        Run the complete detection and segmentation pipeline using RAM-Grounded-SAM
        
        Args:
            image: Input RGB image
            output_dir: Optional directory to save intermediate results
            use_sam_hq: Use sam_hq instead of SAM for prediction
            
        Returns:
            List of detected objects with bounding boxes, descriptions, and masks
        """
        print("Starting RAM-Grounded-SAM detection and segmentation pipeline...")
        
        # Run RAM-Grounded-SAM pipeline (single call for detection + segmentation)
        detected_objects = self.detect_and_segment_objects(
            image, use_sam_hq=use_sam_hq
        )
        
        if not detected_objects:
            print("No objects detected")
            return []
        
        # Save intermediate results if output directory is provided
        if output_dir:
            self._save_results(image, detected_objects, output_dir)
        
        print(f"Detection and segmentation complete. Found {len(detected_objects)} objects.")
        return detected_objects
    
    def _save_results(self, image: np.ndarray, detected_objects: List[DetectedObject], 
                     output_dir: Path) -> None:
        """Save detection and segmentation results"""
        detection_dir = output_dir / "detection_segmentation"
        detection_dir.mkdir(exist_ok=True, parents=True)
        
        # Save annotated image with bounding boxes
        annotated_image = self._draw_bounding_boxes(image.copy(), detected_objects)
        save_image(annotated_image, detection_dir / "annotated_image.png")
        
        # Save visualization images from RAM-Grounded-SAM if available
        if hasattr(self, '_last_ram_result') and self._last_ram_result:
            masked_img = self._last_ram_result.get("masked_img")
            bbox_img = self._last_ram_result.get("rounding_box_img")
            
            if masked_img and hasattr(masked_img, 'read'):
                try:
                    with open(detection_dir / "ram_masked.png", "wb") as f:
                        f.write(masked_img.read())
                except Exception as e:
                    print(f"Warning: Could not save RAM masked image: {e}")
            
            if bbox_img and hasattr(bbox_img, 'read'):
                try:
                    with open(detection_dir / "ram_bboxes.png", "wb") as f:
                        f.write(bbox_img.read())
                except Exception as e:
                    print(f"Warning: Could not save RAM bbox image: {e}")
        
        # Save individual object crops and masks
        for obj in detected_objects:
            obj_dir = detection_dir / f"object_{obj.id}"
            obj_dir.mkdir(exist_ok=True)
            
            # Save cropped image
            if obj.cropped_image is not None:
                save_image(obj.cropped_image, obj_dir / "cropped.png")
            # Save mask
            if obj.cropped_mask is not None:
                cv2.imwrite(str(obj_dir / "cropped_mask.png"), obj.cropped_mask)
            # Save the occlusion mask
            if obj.cropped_occ_mask is not None:
                cv2.imwrite(str(obj_dir / "cropped_occ_mask.png"), obj.cropped_occ_mask)
            
            # Save object info
            obj_info = {
                "id": obj.id,
                "description": obj.description,
                "confidence": obj.confidence,
                "bbox": {
                    "x1": obj.bbox.x1, "y1": obj.bbox.y1,
                    "x2": obj.bbox.x2, "y2": obj.bbox.y2
                }
            }
            
            with open(obj_dir / "info.json", "w") as f:
                json.dump(obj_info, f, indent=2)
    
    def _draw_bounding_boxes(self, image: np.ndarray, 
                           detected_objects: List[DetectedObject]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        for obj in detected_objects:
            x1, y1, x2, y2 = int(obj.bbox.x1), int(obj.bbox.y1), int(obj.bbox.x2), int(obj.bbox.y2)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{obj.id}: {obj.description}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2)
        
        return image
"""
Detection Filtering Module

This module provides filtering capabilities to remove spurious detections
and improve the quality of object detection results.
"""
import numpy as np
import cv2
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils.api_clients import QwenVLClient
from ..core.common import DetectedObject, OCCLUSION_LEVELS
from ..utils.image_utils import save_image
from ..config.settings import config


class DetectionFilteringModule:
    """Module for filtering object detections"""
    
    def __init__(self):
        self.qwen_client = QwenVLClient()
    
    def filter_objects_by_occlusion(self, detected_objects: List[DetectedObject], 
                                   discard_threshold: str = "severe_occlusion") -> Tuple[List[DetectedObject], List[DetectedObject]]:
        """
        Filter objects based on occlusion levels for generation and discard decisions
        
        Args:
            detected_objects: List of detected objects with occlusion levels
            discard_threshold: Minimum occlusion level to discard objects
            
        Returns:
            objects to process
        """
        # Define occlusion level hierarchy
        discard_level = OCCLUSION_LEVELS.get(discard_threshold, 2)
        
        objects_to_process = []
        
        for obj in detected_objects:
            obj_level = OCCLUSION_LEVELS.get(obj.occlusion_level, 0)
            
            # Discard heavily occluded objects
            if obj_level >= discard_level:
                print(f"Discarding object {obj.id} ({obj.description}) due to {obj.occlusion_level}")
                continue
                
            objects_to_process.append(obj)
        
        
        return objects_to_process
    
    def create_annotated_image(self, image: np.ndarray, 
                             detected_objects: List[DetectedObject]) -> np.ndarray:
        """
        Create annotated image with numbered bounding boxes for Qwen-VL analysis
        
        Args:
            image: Original RGB image
            detected_objects: List of detected objects
            
        Returns:
            Annotated image with numbered bounding boxes
        """
        annotated = image.copy()
        
        # Generate N distinct colors for bounding boxes
        def generate_distinct_colors(n):
            """Generate N visually distinct colors in BGR format"""
            colors = []
            if n == 0:
                return colors
            
            # Use HSV color space to generate distinct hues
            import colorsys
            for i in range(n):
                hue = i / n
                saturation = 0.9
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                # Convert to BGR for OpenCV (and scale to 0-255)
                bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                colors.append(bgr)
            return colors
        
        bbox_colors = generate_distinct_colors(len(detected_objects))
        
        for i, obj in enumerate(detected_objects):
            bbox_color = bbox_colors[i] if i < len(bbox_colors) else (0, 255, 0)
            x1, y1, x2, y2 = int(obj.bbox.x1), int(obj.bbox.y1), int(obj.bbox.x2), int(obj.bbox.y2)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bbox_color, 3)
            
            # Draw object ID and description
            label = f"{obj.id}: {obj.description}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), bbox_color, -1)
            
            # Text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Draw large ID number in center of bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            id_text = str(obj.id)
            id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            
            # Background circle for ID
            cv2.circle(annotated, (center_x, center_y), max(id_size) // 2 + 20, bbox_color, -1)
            cv2.circle(annotated, (center_x, center_y), max(id_size) // 2 + 20, (0, 0, 0), 3)
            
            # ID text
            cv2.putText(annotated, id_text, 
                       (center_x - id_size[0] // 2, center_y + id_size[1] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        
        return annotated
    
    def filter_by_size(self, detected_objects: List[DetectedObject], 
                      image_shape: Tuple[int, int],
                      min_area_ratio: float = 0.001,
                      max_area_ratio: float = 0.8) -> List[DetectedObject]:
        """
        Filter detections based on bounding box size
        
        Args:
            detected_objects: List of detected objects
            image_shape: Shape of the image (height, width)
            min_area_ratio: Minimum area ratio relative to image size
            max_area_ratio: Maximum area ratio relative to image size
            
        Returns:
            Filtered list of detected objects
        """
        h, w = image_shape[:2]
        image_area = h * w
        
        filtered_objects = []
        
        for obj in detected_objects:
            bbox_area = obj.bbox.width * obj.bbox.height
            area_ratio = bbox_area / image_area
            
            if min_area_ratio <= area_ratio <= max_area_ratio:
                filtered_objects.append(obj)
            else:
                print(f"Filtered out object {obj.id} ({obj.description}): "
                      f"area ratio {area_ratio:.4f} outside range [{min_area_ratio}, {max_area_ratio}]")
        
        print(f"Size filtering: {len(detected_objects)} -> {len(filtered_objects)} objects")
        return filtered_objects
    
    def filter_by_qwen_vl(self, image: np.ndarray, 
                         detected_objects: List[DetectedObject], 
                         enable_occlusion_filter: bool = True, 
                         occlusion_threshold: str = "severe_occlusion") -> List[DetectedObject]:
        """
        Filter detections using Qwen-VL to remove spurious and scene-level detections
        while also assessing occlusion levels for remaining objects
        
        Args:
            image: Original RGB image
            detected_objects: List of detected objects
            enable_occlusion_filter: Whether to use occlusion-based filtering
            occlusion_threshold: Minimum occlusion level to discard objects
        Returns:
            Filtered list of detected objects with occlusion levels
        """
        if not detected_objects:
            return detected_objects
        
        print("Filtering detections using Qwen-VL...")
        
        # Create annotated image
        annotated_image = self.create_annotated_image(image, detected_objects)
        
        # Prepare detection data for Qwen-VL
        detection_data = {
            "objects": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "confidence": obj.confidence,
                    "bbox": {
                        "x1": obj.bbox.x1, "y1": obj.bbox.y1,
                        "x2": obj.bbox.x2, "y2": obj.bbox.y2
                    }
                }
                for obj in detected_objects
            ]
        }
        
        # Call Qwen-VL for filtering
        try:
            filter_response = self.qwen_client.filter_detections(
                image, annotated_image, detection_data
            )
            
            if not filter_response:
                print("Warning: No response from Qwen-VL, keeping all detections")
                return detected_objects
            
            # Parse the response
            filter_result = json.loads(filter_response.strip('```json').strip('```').strip('\n'))
            keep_objects = filter_result.get("keep", [])
            remove_info = filter_result.get("remove", [])
            reasoning = filter_result.get("reasoning", "")
            
            print(f"Qwen-VL filtering reasoning: {reasoning}")
            
            # Create mapping from object ID to occlusion level
            occlusion_map = {}
            if isinstance(keep_objects, list):
                for keep_item in keep_objects:
                    if isinstance(keep_item, dict):
                        obj_id = keep_item.get("id")
                        occlusion_level = keep_item.get("occlusion_level", "no_occlusion")
                        # fallback to no occlusion
                        if occlusion_level not in OCCLUSION_LEVELS:
                            occlusion_level = "no_occlusion"
                        if obj_id is not None:
                            occlusion_map[obj_id] = occlusion_level
                    # elif isinstance(keep_item, int):
                        # Fallback for old format
                        # occlusion_map[keep_item] = "no_occlusion"
            
            keep_ids = set(occlusion_map.keys())
            
            # Filter objects based on Qwen-VL response
            filtered_objects = []
            for obj in detected_objects:
                if obj.id in keep_ids:
                    # Add occlusion level to the object
                    obj.occlusion_level = occlusion_map.get(obj.id, "no_occlusion")
                    if enable_occlusion_filter and OCCLUSION_LEVELS.get(obj.occlusion_level, 0) >= OCCLUSION_LEVELS.get(occlusion_threshold, 2):
                        print(f"Discarding object {obj.id} ({obj.description}) due to {obj.occlusion_level}")
                        continue
                    filtered_objects.append(obj)
                    print(f"Keeping object {obj.id} ({obj.description}) with occlusion level: {obj.occlusion_level}")
                else:
                    # Find removal reason
                    remove_reason = "Not specified"
                    for remove_item in remove_info:
                        if isinstance(remove_item, dict) and remove_item.get("id") == obj.id:
                            remove_reason = remove_item.get("reason", "Not specified")
                        elif isinstance(remove_item, list) and len(remove_item) >= 2 and remove_item[0] == obj.id:
                            remove_reason = remove_item[1]
                    
                    print(f"Qwen-VL filtered out object {obj.id} ({obj.description}): {remove_reason}")
            
            print(f"Qwen-VL filtering: {len(detected_objects)} -> {len(filtered_objects)} objects")
            return filtered_objects
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse Qwen-VL response: {e}")
            print(f"Response was: {filter_response}")
            return detected_objects
        except Exception as e:
            print(f"Warning: Error in Qwen-VL filtering: {e}")
            return detected_objects
    
    def run(self, image: np.ndarray, detected_objects: List[DetectedObject],
            use_qwen_filter: bool = True,
            use_size_filter: bool = True,
            use_occlusion_filter: bool = True,
            occlusion_threshold: str = "severe_occlusion",
            min_area_ratio: float = 0.001,
            max_area_ratio: float = 0.8,
            output_dir: Optional[Path] = None) -> List[DetectedObject]:
        """
        Run the complete detection filtering pipeline
        
        Args:
            image: Original RGB image
            detected_objects: List of detected objects to filter
            use_qwen_filter: Whether to use Qwen-VL filtering
            use_size_filter: Whether to use size-based filtering
            use_occlusion_filter: Whether to use occlusion-based filtering
            occlusion_threshold: Minimum occlusion level to discard objects
            min_area_ratio: Minimum area ratio for size filtering
            max_area_ratio: Maximum area ratio for size filtering
            output_dir: Optional directory to save filtering results
            
        Returns:
            Filtered list of detected objects
        """
        print("Starting detection filtering pipeline...")
        
        filtered_objects = detected_objects.copy()
        
        # Step 1: Size-based filtering
        if use_size_filter:
            print("\nApplying size-based filtering...")
            filtered_objects = self.filter_by_size(
                filtered_objects, image.shape, min_area_ratio, max_area_ratio
            )
        
        # Step 2: Qwen-VL based filtering
        if use_qwen_filter and len(filtered_objects) > 0:
            print("\nApplying Qwen-VL based filtering...")
            filtered_objects = self.filter_by_qwen_vl(image, filtered_objects, use_occlusion_filter, occlusion_threshold)
        
        # Save results if output directory is provided
        if output_dir:
            self._save_filtering_results(
                image, detected_objects, filtered_objects, output_dir
            )
        
        print(f"\nFiltering complete: {len(detected_objects)} -> {len(filtered_objects)} objects")
        return filtered_objects
    
    def _save_filtering_results(self, image: np.ndarray, 
                              original_objects: List[DetectedObject],
                              filtered_objects: List[DetectedObject],
                              output_dir: Path) -> None:
        """Save filtering results for analysis"""
        filter_dir = output_dir / "detection_filtering"
        filter_dir.mkdir(exist_ok=True, parents=True)
        
        # Save annotated images
        original_annotated = self.create_annotated_image(image, original_objects)
        filtered_annotated = self.create_annotated_image(image, filtered_objects)
        
        save_image(original_annotated, filter_dir / "original_detections.png")
        save_image(filtered_annotated, filter_dir / "filtered_detections.png")
        
        # Save filtering summary
        filtering_summary = {
            "original_count": len(original_objects),
            "filtered_count": len(filtered_objects),
            "removed_objects": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "confidence": obj.confidence
                }
                for obj in original_objects 
                if obj.id not in [f_obj.id for f_obj in filtered_objects]
            ],
            "kept_objects": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "confidence": obj.confidence,
                    "occlusion_level": getattr(obj, 'occlusion_level', 'no_occlusion')
                }
                for obj in filtered_objects
            ]
        }
        
        with open(filter_dir / "filtering_summary.json", "w") as f:
            json.dump(filtering_summary, f, indent=2)
        
        print(f"Filtering results saved to: {filter_dir}")

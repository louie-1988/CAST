"""
Image Generation Module

This module handles image generation/completion for detected objects
using Kontext generation model via Replicate API.
"""
import numpy as np
from typing import Optional, List
from pathlib import Path
import cv2

from ..core.common import OCCLUSION_LEVELS, DetectedObject
from ..utils.api_clients import ReplicateClient, QwenVLClient
from ..utils.image_utils import save_image, base64_to_image


class ImageGenerationModule:
    """Module for image generation using Kontext or Qwen"""
    
    def __init__(self, provider: str = "replicate"):
        """
        Initialize the image generation module
        
        Args:
            provider: Generation provider - "replicate" for Kontext or "qwen" for Qwen image edit
        """
        self.provider = provider.lower()
        if self.provider == "replicate":
            self.replicate_client = ReplicateClient()
            self.qwen_client = None
        elif self.provider == "qwen":
            self.qwen_client = QwenVLClient()
            self.replicate_client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'replicate' or 'qwen'")
    
    def generate_object_image(self, detected_object: DetectedObject, 
                            output_dir: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        Generate/complete an object image using the selected provider
        
        Args:
            detected_object: DetectedObject with cropped and matted image
            output_dir: Optional directory to save results
            
        Returns:
            Generated image or None if generation failed
        """
        if detected_object.cropped_image is None:
            print(f"Warning: No cropped image available for object {detected_object.id}")
            return None
        
        print(f"Generating image for object {detected_object.id}: {detected_object.description} using {self.provider}")
        
        # Create matted image (cropped image with alpha channel for transparency)
        if detected_object.cropped_mask is not None:
            # Apply mask to create transparency
            mask_3d = detected_object.cropped_mask[..., np.newaxis] / 255.0
            matted_image = detected_object.cropped_image * mask_3d
            
            # Convert to RGBA
            alpha_channel = detected_object.cropped_mask[..., np.newaxis]
            matted_rgba = np.concatenate([matted_image.astype(np.uint8), alpha_channel], axis=2)
        else:
            # Use original cropped image if no mask available
            matted_rgba = detected_object.cropped_image
        
        # Generate prompt based on object description
        prompt = f"Inpaint the image from visible parts. It's a {detected_object.description}"
        
        try:
            if self.provider == "replicate":
                return self._generate_with_replicate(detected_object, matted_rgba, prompt, output_dir)
            elif self.provider == "qwen":
                return self._generate_with_qwen(detected_object, matted_rgba, prompt, output_dir)
            else:
                print(f"Unsupported provider: {self.provider}")
                return None
                
        except Exception as e:
            print(f"Error during generation for object {detected_object.id}: {e}")
            return None
    
    def _generate_with_replicate(self, detected_object: DetectedObject, matted_rgba: np.ndarray, 
                               prompt: str, output_dir: Optional[Path]) -> Optional[np.ndarray]:
        """Generate image using Replicate Kontext"""
        generation_result = self.replicate_client.run_kontext_generation(
            image=matted_rgba,
            prompt=prompt
        )
        
        if generation_result:
            # Handle the response format
            generated_image = None
            
            if hasattr(generation_result, 'read'):
                # File-like object from Replicate
                try:
                    image_data = generation_result.read()
                    from PIL import Image
                    import io
                    pil_image = Image.open(io.BytesIO(image_data))
                    generated_image = np.array(pil_image.convert('RGB'))
                except Exception as e:
                    print(f"Error reading generated image data: {e}")
                    generated_image = None
            else:
                print("Unexpected generation result format")
                print(generation_result)

            # Resize to match original cropped image dimensions if successful
            if generated_image is not None:
                h, w = detected_object.cropped_image.shape[:2]
                generated_image = cv2.resize(generated_image, (w, h))
                
                # Save generated image if output directory provided
                if output_dir:
                    generation_dir = output_dir / "generation"
                    generation_dir.mkdir(exist_ok=True, parents=True)
                    save_image(generated_image, generation_dir / f"object_{detected_object.id}_generated.png")
                
                print(f"Successfully generated image for object {detected_object.id}")
                return generated_image
            else:
                print(f"Generation processing failed for object {detected_object.id}")
                return None
        else:
            print(f"Generation failed for object {detected_object.id}")
            return None
    
    def _generate_with_qwen(self, detected_object: DetectedObject, matted_rgba: np.ndarray,
                          prompt: str, output_dir: Optional[Path]) -> Optional[np.ndarray]:
        """Generate image using Qwen image edit"""
        generation_result = self.qwen_client.run_qwen_image_edit(
            image=matted_rgba,
            prompt=prompt
        )
        
        if generation_result:
            try:
                # Handle Qwen response - it should be a URL or base64 data
                generated_image = None
                
                if generation_result.startswith('http'):
                    # It's a URL, download the image
                    import requests
                    response = requests.get(generation_result)
                    if response.status_code == 200:
                        from PIL import Image
                        import io
                        pil_image = Image.open(io.BytesIO(response.content))
                        generated_image = np.array(pil_image.convert('RGB'))
                elif generation_result.startswith('data:image'):
                    # It's base64 data
                    generated_image = base64_to_image(generation_result)
                else:
                    print(f"Unexpected Qwen response format: {type(generation_result)}")
                    return None

                # Resize to match original cropped image dimensions if successful
                if generated_image is not None:
                    h, w = detected_object.cropped_image.shape[:2]
                    generated_image = cv2.resize(generated_image, (w, h))
                    
                    # Save generated image if output directory provided
                    if output_dir:
                        generation_dir = output_dir / "generation"
                        generation_dir.mkdir(exist_ok=True, parents=True)
                        save_image(generated_image, generation_dir / f"object_{detected_object.id}_generated.png")
                    
                    print(f"Successfully generated image for object {detected_object.id}")
                    return generated_image
                else:
                    print(f"Failed to process Qwen generation result for object {detected_object.id}")
                    return None
                    
            except Exception as e:
                print(f"Error processing Qwen generation result: {e}")
                return None
        else:
            print(f"Qwen generation failed for object {detected_object.id}")
            return None

    def _create_matted_image(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create a matted image from RGB and mask"""
        mask_float = mask[..., np.newaxis] / 255.0
        matted_image = rgb * mask_float
        alpha_channel = mask[..., np.newaxis]
        matted_rgba = np.concatenate([matted_image.astype(np.uint8), alpha_channel], axis=2)
        return matted_rgba

    
    def run(self, detected_objects: List[DetectedObject], generate_threshold: int, 
            output_dir: Optional[Path] = None):
        """
        Run generation for specified objects
        
        Args:
            detected_objects: List of all detected objects
            generate_threshold: Minimum occlusion level to trigger generation
            output_dir: Optional directory to save results
            
        Returns:
            Updated detected objects with generated images
        """
        print(f"Starting image generation pipeline using {self.provider}...")
        
        # Generate images for occluded objects
        for obj in detected_objects:
            if OCCLUSION_LEVELS.get(obj.occlusion_level, 2) >= generate_threshold:
                generated_image = self.generate_object_image(
                    obj, output_dir=output_dir
                )
                if generated_image is not None:
                    obj.generated_image = generated_image
                else:
                    # the generated image will ONLY be used for mesh generation
                    # if it's NOT created, we mat original cropped image 
                    obj.generated_image = self._create_matted_image(obj.cropped_image, obj.cropped_mask)
                    print(f"Failed to generate image for object {obj.id} ({obj.description}). Create the matted image instead.")
            else:
                obj.generated_image = self._create_matted_image(obj.cropped_image, obj.cropped_mask)
                print(f"Skipping generation for object {obj.id} ({obj.description}) due to occlusion level {obj.occlusion_level}. Create the matted image instead.")
        
        print("Image generation pipeline complete.")
        return detected_objects
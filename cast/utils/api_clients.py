"""
API client wrappers for external services
"""
import os
import requests
import time
import base64
import asyncio
import mimetypes
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import replicate
from httpx import Timeout
from openai import OpenAI

from ..config.settings import config
from .image_utils import image_to_base64
import numpy as np

REPLICATE_TIMEOUT = Timeout(
    10080.0,      # default timeout
    read=10080.0,  # 3 hour minutes read timeout
    write=600.0,  # write timeout
    connect=600.0,  # connect timeout
    pool=600.0     # pool timeout
)

class ReplicateClient:
    """Wrapper for Replicate API calls"""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or config.api.replicate_token
        self.client = replicate.Client(api_token=self.api_token, timeout=600)

    def run_ram_grounded_sam(self, image: Union[np.ndarray, str], 
                            use_sam_hq: bool = True,
                            show_visualisation: bool = False) -> Dict[str, Any]:
        """
        Run RAM-Grounded-SAM for combined object recognition, detection and segmentation
        
        Args:
            image: Input image as numpy array or base64 string
            use_sam_hq: Use sam_hq instead of SAM for prediction (default: False)
            show_visualisation: Output bounding box and masks on the image (default: False)
            
        Returns:
            Dictionary containing:
            - tags: String of detected tags
            - json_data: List of detected objects with bounding boxes and labels
            - masked_img: Base64 encoded masked image (if show_visualisation=True)
            - rounding_box_img: Base64 encoded image with bounding boxes (if show_visualisation=True)
        """
        async def run_core(model_name: str, input: Any):
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self.client.async_run(model_name, input=input))
                ]
                return await asyncio.gather(*tasks)

        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image
            
        try:
            # loop = asyncio.get_event_loop()
            output = asyncio.run(run_core(
                config.models.ram_grounded_sam_model,
                input={
                    "input_image": image_b64,
                    "use_sam_hq": use_sam_hq,
                    "show_visualisation": show_visualisation
                }, 
            ))[0]
            return output
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error running RAM-Grounded-SAM: {e}")
            return {}
    
    def run_inpainting(self, image: Union[np.ndarray, str], 
                      mask: Union[np.ndarray, str],
                      prompt: str = "high quality, detailed") -> str:
        """Run stable diffusion inpainting"""
        async def run_core(model_name: str, input: Any):
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self.client.async_run(model_name, input=input))
                ]
                return await asyncio.gather(*tasks)

        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image
            
        if isinstance(mask, np.ndarray):
            mask_b64 = image_to_base64(mask)
        else:
            mask_b64 = mask
            
        try:
            loop = asyncio.get_event_loop()
            output = asyncio.run(run_core(
                config.models.sd_inpainting_model,
                input={
                    "image": image_b64,
                    "mask": mask_b64,
                    "prompt": prompt,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "strength": 0.99
                }
            ))[0]
            return output
        except Exception as e:
            print(f"Error running stable diffusion inpainting: {e}")
            return None

    def run_kontext_generation(self, image: Union[np.ndarray, str], 
                              prompt: str = "complete the image") -> str:
        """
        Run Flux Kontext for image generation/completion
        
        Args:
            image: Input image as numpy array or base64 string  
            prompt: Generation prompt
            
        Returns:
            Generated image URL or base64 string
        """
        async def run_core(model_name: str, input: Any):
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self.client.async_run(model_name, input=input))
                ]
                return await asyncio.gather(*tasks)

        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image
            
        try:
            loop = asyncio.get_event_loop()
            output = loop.run_until_complete(run_core(
                "black-forest-labs/flux-kontext-dev",
                input={
                    "input_image": image_b64,
                    "prompt": prompt,
                    "num_inference_steps": 30,
                }
            ))[0]
            return output
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error running Kontext generation: {e}")
            return None

class Tripo3DClient:
    """Wrapper for Tripo3D API calls"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.api.tripo3d_key
        self.base_url = "https://api.tripo3d.ai/v2/openapi"
        
    def upload_image(self, image: Union[np.ndarray, str, Path]) -> str:
        """Upload image and get file token"""
        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
            # Convert to bytes for upload
            import io
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
        else:
            # Read from file
            with open(image, 'rb') as f:
                img_bytes = f.read()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        files = {
            "file": ("image.png", img_bytes, "image/png")
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/upload",
                headers=headers,
                files=files
            )
            response.raise_for_status()
            return response.json()["data"]["image_token"]
        except Exception as e:
            print(f"Error uploading image to Tripo3D: {e}")
            return ""
    
    def create_3d_model(self, file_token: str) -> str:
        """Create 3D model from uploaded image"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "type": "image_to_model",
            "file": {
                "type": "png",
                "file_token": file_token
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/task",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["data"]["task_id"]
        except Exception as e:
            print(f"Error creating 3D model: {e}")
            return ""
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and results"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/task/{task_id}",
                headers=headers
            )
            response.raise_for_status()
            return response.json()["data"]
        except Exception as e:
            print(f"Error getting task status: {e}")
            return {}
    
    def wait_for_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for task completion with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            
            if status.get("status") == "success":
                return status
            elif status.get("status") == "failed":
                raise Exception(f"Task failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(5)  # Wait 5 seconds before checking again
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def download_model(self, download_url: str, output_path: Path) -> bool:
        """Download 3D model file"""
        try:
            response = requests.get(download_url)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False

class QwenVLClient:
    """Wrapper for Qwen-VL API calls"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.api.dashscope_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def analyze_scene_graph(self, image: Union[np.ndarray, str], 
                          system_prompt: str, user_prompt: str) -> str:
        """Analyze image to extract scene graph relationships"""
        if isinstance(image, np.ndarray):
            image_url = image_to_base64(image)
        else:
            image_url = image
        
        try:
            completion = self.client.chat.completions.create(
                model=config.models.qwen_model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing scene graph: {e}")
            return ""
    
    def analyze_image(self, image: Union[np.ndarray, str], 
                     system_prompt: str, user_prompt: str) -> str:
        """
        General image analysis using Qwen-VL
        
        Args:
            image: Input image as numpy array or base64 string
            system_prompt: System prompt for the task
            user_prompt: User prompt for the task
            
        Returns:
            Response from Qwen-VL
        """
        if isinstance(image, np.ndarray):
            image_url = image_to_base64(image)
        else:
            image_url = image
        
        try:
            completion = self.client.chat.completions.create(
                model=config.models.qwen_model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return ""
    
    def filter_detections(self, original_image: Union[np.ndarray, str], 
                         annotated_image: Union[np.ndarray, str],
                         detection_data: Dict[str, Any]) -> str:
        """
        Filter object detections to remove spurious and scene-level detections
        while also assessing occlusion levels
        
        Args:
            original_image: Original RGB image
            annotated_image: Image with numbered bounding boxes
            detection_data: Detection data with bounding boxes and labels
            
        Returns:
            JSON string with filtered object IDs, occlusion levels, and reasoning
        """
        if isinstance(original_image, np.ndarray):
            original_url = image_to_base64(original_image)
        else:
            original_url = original_image
            
        if isinstance(annotated_image, np.ndarray):
            annotated_url = image_to_base64(annotated_image)
        else:
            annotated_url = annotated_image
        
        # Create system prompt for detection filtering with occlusion assessment
        system_prompt = """You are an expert computer vision system for filtering object detections and assessing occlusion levels. Your task is to identify meaningful constituent objects, remove spurious detections, and rate occlusion levels.

Rules for filtering:
1. Remove scene-level detections (e.g., room, sky, ground, wall, background)
2. Remove background elements that are not distinct objects
3. Remove parts of objects when the whole object is also detected (e.g., wheel of a car if car is detected)
4. Keep meaningful, distinct objects that can be isolated and reconstructed in 3D
5. Prioritize objects that have clear boundaries and can be manipulated independently

Occlusion Level Assessment:
For each object you decide to keep, assess its occlusion level:
- "no_occlusion": Object is completely visible with no parts hidden (0~15%)
- "some_occlusion": Object has moderate occlusion (~15-50% part occluded)
- "severe_occlusion": Object is heavily occluded, only small parts visible (more than 50% part occluded)

You must respond with a JSON object containing:
- "keep": list of objects to keep, each with {"id": int, "occlusion_level": str}
- "remove": list of object IDs to remove with reasons
- "reasoning": overall reasoning for the filtering decisions"""

        # Create user prompt with detection data
        user_prompt = f"""Analyze these object detections and filter out spurious/scene-level detections.
For each object you decide to keep, also assess its occlusion level.

Detection data:
{detection_data}

Please examine both the original image and the annotated image with numbered bounding boxes. Filter the detections to keep only meaningful constituent objects that can be reconstructed in 3D, and rate their occlusion levels.

Return your response as a JSON object with "keep", "remove", and "reasoning" fields."""

        try:
            completion = self.client.chat.completions.create(
                model=config.models.qwen_model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": original_url}
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": annotated_url}
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error filtering detections: {e}")
            return ""
    
    def run_qwen_image_edit(self, image: Union[np.ndarray, str], 
                           prompt: str = "Inpaint the image from visible parts") -> Optional[str]:
        """
        Run Qwen image editing/inpainting
        
        Args:
            image: Input image as numpy array or base64 string
            prompt: Inpainting prompt
            
        Returns:
            Generated image URL/data or None if generation failed
        """
        try:
            from dashscope import MultiModalConversation
            
            # Convert image to base64 format expected by Qwen
            if isinstance(image, np.ndarray):
                # Save image temporarily to get proper base64 encoding
                from PIL import Image as PILImage
                import io
                import tempfile
                
                # Convert numpy array to PIL Image
                if image.shape[2] == 4:  # RGBA
                    pil_image = PILImage.fromarray(image, 'RGBA')
                else:  # RGB
                    pil_image = PILImage.fromarray(image, 'RGB')
                
                # Save to temporary file to get proper MIME type
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    pil_image.save(tmp_file.name, format='PNG')
                    image_b64 = self._encode_file_for_qwen(tmp_file.name)
                    
                # Clean up temp file
                import os
                os.unlink(tmp_file.name)
            else:
                # Assume it's already a file path or base64
                if image.startswith('data:'):
                    image_b64 = image
                else:
                    image_b64 = self._encode_file_for_qwen(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_b64},
                        {"text": prompt}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model="qwen-image-edit",
                messages=messages,
                result_format='message',
                stream=False,
                watermark=True,
                negative_prompt=""
            )
            
            if response.status_code == 200:
                # Extract the generated image from response
                if hasattr(response, 'output') and response.output:
                    if hasattr(response.output, 'choices') and response.output.choices:
                        choice = response.output.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            if hasattr(choice.message, 'content') and choice.message.content:
                                for content in choice.message.content:
                                    if hasattr(content, 'image') and content.image:
                                        return content.image
                print("No image found in Qwen response")
                return None
            else:
                print(f"Qwen API error - HTTP {response.status_code}: {response.code} - {response.message}")
                return None
                
        except ImportError:
            print("dashscope package not available. Please install: pip install dashscope")
            return None
        except Exception as e:
            print(f"Error running Qwen image edit: {e}")
            return None
    
    def _encode_file_for_qwen(self, file_path: str) -> str:
        """
        Encode file for Qwen API in the required format
        Format: data:{MIME_type};base64,{base64_data}
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("Unsupported or unrecognizable image format")
        
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return f"data:{mime_type};base64,{encoded_string}"


class TrellisClient:
    """Wrapper for TRELLIS 3D generation via Replicate with async and sync support"""
    
    def __init__(self, api_token: Optional[str] = None, base_url: Optional[str] = None):
        self.api_token = api_token or config.api.replicate_token
        self.base_url = base_url  # No default - None means use Replicate
        
        # Determine if this is a local deployment
        self.is_local = self._is_local_deployment(self.base_url) if self.base_url else False
        
        if self.is_local:
            self.client = replicate.Client(api_token=self.api_token, base_url=self.base_url, timeout=300)
            print(f"TrellisClient initialized for local deployment at {self.base_url}")
        else:
            self.client = replicate.Client(api_token=self.api_token, timeout=300)
            print("TrellisClient initialized for remote Replicate deployment")
            
        self.model_version = "firtoz/trellis:e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c"
    
    def _is_local_deployment(self, base_url: str) -> bool:
        """
        Determine if the base_url points to a local deployment
        
        Args:
            base_url: The base URL to check
            
        Returns:
            True if it's a local deployment, False otherwise
        """
        if not base_url:
            return False
            
        # Extract hostname from URL
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        hostname = parsed.hostname
        
        if not hostname:
            return False
        
        # Check for local IP patterns
        local_patterns = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0"
        ]
        
        # Check for private IP ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
        if hostname.startswith("192.168.") or hostname.startswith("10."):
            return True
        
        # Check 172.16.0.0 to 172.31.255.255 range
        if hostname.startswith("172."):
            try:
                second_octet = int(hostname.split('.')[1])
                if 16 <= second_octet <= 31:
                    return True
            except (ValueError, IndexError):
                pass
        
        return hostname in local_patterns
    
    def create_trellis_input(self, image: Union[np.ndarray, str], 
                           texture_size: int = 1024,
                           mesh_simplify: float = 0.9,
                           ss_sampling_steps: int = 12,
                           slat_sampling_steps: int = 12,
                           generate_model: bool = True,
                           generate_color: bool = False,
                           save_gaussian_ply: bool = False) -> Dict[str, Any]:
        """
        Create input dictionary for TRELLIS model
        
        Args:
            image: Input image as numpy array or base64 string
            texture_size: Texture resolution (default: 1024)
            mesh_simplify: Mesh simplification ratio (default: 0.9)
            ss_sampling_steps: Sampling steps for sparse structure (default: 12)
            slat_sampling_steps: Sampling steps for SLAT (default: 12)
            generate_model: Whether to generate 3D model (default: True)
            generate_color: Whether to generate color texture (default: False)
            save_gaussian_ply: Whether to save Gaussian PLY (default: False)
            
        Returns:
            Input dictionary for TRELLIS model
        """
        if isinstance(image, np.ndarray):
            image_b64 = image_to_base64(image)
        else:
            image_b64 = image
            
        return {
            "images": [image_b64],
            "texture_size": texture_size,
            "mesh_simplify": mesh_simplify,
            "generate_model": generate_model,
            "save_gaussian_ply": save_gaussian_ply,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_sampling_steps": slat_sampling_steps,
            "generate_color": generate_color,
        }
    
    async def generate_3d_async(self, image: Union[np.ndarray, str], 
                              **kwargs) -> str:
        """
        Generate 3D model asynchronously using TRELLIS
        
        Args:
            image: Input image
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            Prediction ID for tracking the job
        """
        input_data = self.create_trellis_input(image, **kwargs)
        
        try:
            # Create async prediction
            prediction = await self.client.predictions.async_create(
                version=self.model_version,
                input=input_data
            )
            
            print(f"Started TRELLIS 3D generation with prediction ID: {prediction.id}")
            return prediction.id
            
        except Exception as e:
            print(f"Error starting TRELLIS generation: {e}")
            return ""
    
    def generate_3d_sync(self, image: Union[np.ndarray, str], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate 3D model synchronously using TRELLIS (for local deployments)
        
        Args:
            image: Input image
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            Result dictionary with model file URL or None if failed
        """
        if not self.is_local:
            raise RuntimeError("Synchronous generation is only supported for local deployments. Use generate_3d_async for remote deployments.")
        
        input_data = self.create_trellis_input(image, **kwargs)
        
        try:
            print("Starting synchronous TRELLIS 3D generation...")
            start_time = time.time()
            
            # Run synchronous prediction for local deployment
            output = self.client.run(self.model_version, input=input_data)
            
            end_time = time.time()
            print(f"TRELLIS generation completed in {end_time - start_time:.2f} seconds")
            
            return {"output": output}
            
        except Exception as e:
            import traceback 
            traceback.print_exc()
            print(f"Error in synchronous TRELLIS generation: {e}")
            return None
    
    async def get_prediction_status(self, prediction_id: str) -> Dict[str, Any]:
        """
        Get the status of an async prediction
        
        Args:
            prediction_id: The prediction ID to check
            
        Returns:
            Dictionary with prediction status and results
        """
        try:
            prediction = await replicate.predictions.async_get(prediction_id)
            return {
                "id": prediction.id,
                "status": prediction.status,
                "output": prediction.output,
                "error": prediction.error,
                "logs": prediction.logs
            }
        except Exception as e:
            print(f"Error getting prediction status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def wait_for_completion_async(self, prediction_id: str, 
                                      timeout: int = 600,
                                      check_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for async prediction to complete
        
        Args:
            prediction_id: The prediction ID to wait for
            timeout: Maximum time to wait in seconds
            check_interval: How often to check status in seconds
            
        Returns:
            Final prediction result
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.get_prediction_status(prediction_id)
            
            if status["status"] == "succeeded":
                return status
            elif status["status"] == "failed":
                raise Exception(f"TRELLIS generation failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "canceled":
                raise Exception("TRELLIS generation was canceled")
            
            await asyncio.sleep(check_interval)
        
        raise TimeoutError(f"TRELLIS generation did not complete within {timeout} seconds")
    
    async def batch_generate_async(self, images: List[Union[np.ndarray, str]], 
                                 **kwargs) -> List[str]:
        """
        Start multiple async 3D generations
        
        Args:
            images: List of input images
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            List of prediction IDs
        """
        prediction_ids = []
        
        for i, image in enumerate(images):
            print(f"Starting generation {i+1}/{len(images)}")
            pred_id = await self.generate_3d_async(image, **kwargs)
            if pred_id:
                prediction_ids.append(pred_id)
            
            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.5)
        
        print(f"Started {len(prediction_ids)} async TRELLIS generations")
        return prediction_ids
    
    async def collect_results_async(self, prediction_ids: List[str],
                                  output_dir: Optional[Path] = None) -> List[Optional[str]]:
        """
        Collect results from multiple async predictions
        
        Args:
            prediction_ids: List of prediction IDs to collect
            output_dir: Optional directory to save downloaded models
            
        Returns:
            List of model file paths (None for failed generations)
        """
        results = []
        
        # Wait for all predictions to complete
        tasks = [self.wait_for_completion_async(pred_id) for pred_id in prediction_ids]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and download models
        for i, (pred_id, result) in enumerate(zip(prediction_ids, completed_results)):
            if isinstance(result, Exception):
                print(f"Generation {i+1} failed: {result}")
                results.append(None)
                continue
            
            try:
                if result["output"] and "model_file" in result["output"]:
                    model_url = result["output"]["model_file"]
                    
                    if output_dir:
                        model_path = output_dir / f"trellis_model_{i+1}.glb"
                        if await self._download_model_async(model_url, model_path):
                            results.append(str(model_path))
                        else:
                            results.append(None)
                    else:
                        results.append(model_url)
                else:
                    print(f"No model file in result for prediction {pred_id}")
                    results.append(None)
                    
            except Exception as e:
                print(f"Error processing result for prediction {pred_id}: {e}")
                results.append(None)
        
        successful = sum(1 for r in results if r is not None)
        print(f"Collected {successful}/{len(prediction_ids)} successful TRELLIS results")
        
        return results
    
    def batch_generate_sync(self, images: List[Union[np.ndarray, str]], 
                           output_dir: Optional[Path] = None,
                           **kwargs) -> List[Optional[str]]:
        """
        Generate multiple 3D models synchronously (for local deployments)
        
        Args:
            images: List of input images
            output_dir: Optional directory to save downloaded models
            **kwargs: Additional parameters for TRELLIS
            
        Returns:
            List of model file paths (None for failed generations)
        """
        if not self.is_local:
            raise RuntimeError("Synchronous batch generation is only supported for local deployments.")
        
        results = []
        
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)} synchronously...")
            
            result = self.generate_3d_sync(image, **kwargs)
            
            if result and result["output"] and "model_file" in result["output"]:
                model_url = result["output"]["model_file"]
                
                if output_dir:
                    model_path = output_dir / f"trellis_model_{i+1}.glb"
                    try:
                        with open(model_path, "wb") as f:
                            f.write(model_url.read())
                        if os.path.exists(model_path):
                            results.append(str(model_path))
                        else:
                            results.append(None)
                    except:
                        print(f"Failed to download model for image {i+1}")
                        results.append(None)
                else:
                    results.append(model_url)
            else:
                print(f"Failed to generate model for image {i+1}")
                results.append(None)
            
            # Small delay between requests to be respectful
            time.sleep(1)
        
        successful = sum(1 for r in results if r is not None)
        print(f"Completed synchronous batch generation: {successful}/{len(images)} successful")
        
        return results
    
    async def _download_model_async(self, model_url: str, output_path: Path) -> bool:
        """
        Download model file asynchronously
        
        Args:
            model_url: URL of the model file
            output_path: Local path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import aiohttp
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as response:
                    if response.status == 200:
                        with open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        print(f"Downloaded TRELLIS model to: {output_path}")
                        return True
                    else:
                        print(f"Failed to download model: HTTP {response.status}")
                        return False
                        
        except ImportError:
            print("aiohttp not available, falling back to sync download")
            return self._download_model_sync(model_url, output_path)
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    
    def _download_model_sync(self, model_url: str, output_path: Path) -> bool:
        """
        Download model file synchronously (fallback)
        
        Args:
            model_url: URL of the model file
            output_path: Local path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            response = requests.get(model_url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded TRELLIS model to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
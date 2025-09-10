"""
Configuration settings for CAST pipeline
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration settings"""
    replicate_token: Optional[str] = None
    tripo3d_key: Optional[str] = None
    dashscope_key: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment variables
        self.replicate_token = os.getenv("REPLICATE_API_TOKEN")
        self.tripo3d_key = os.getenv("TRIPO3D_API_KEY")
        self.dashscope_key = os.getenv("DASHSCOPE_API_KEY")


@dataclass
class ModelConfig:
    """Model configuration settings"""
    # RAM-Grounded-SAM for object detection and segmentation
    # ram_grounded_sam_model: str = "idea-research/ram-grounded-sam:80a2aede4cf8e3c9f26e96c308d45b23c350dd36f1c381de790715007f1ac0ad"
    # Use the version that fixed the mask (rgb -> gray) so that we can get the true values
    ram_grounded_sam_model: str = "fishwowater/ram-grounded-sam-maskfixed:ea5ba20f4689f8e44f124214dd2831feaad23d3f92fac380b34669f6bb9eaf18"

    # Stable Diffusion Inpainting settings
    inpainting_model: str = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
    
    # MoGe settings
    moge_model: str = "Ruicheng/moge-2-vitl-normal"
    
    # Qwen-VL settings
    qwen_model: str = "qwen-vl-max-latest"

@dataclass
class ProcessingConfig:
    """Processing configuration settings"""
    # ICP parameters
    icp_max_iterations: int = 1000
    icp_tolerance: float = 1e-6
    icp_fitness_threshold: float = 0.3
    
    # Scene graph optimization parameters (using Open3D SDF)
    sdf_learning_rate: float = 0.01
    sdf_max_iterations: int = 500
    sdf_penetration_weight: float = 1.0
    sdf_contact_weight: float = 0.5

class Config:
    """Main configuration class"""
    def __init__(self):
        self.api = APIConfig()
        self.models = ModelConfig()
        self.processing = ProcessingConfig()
        
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            self.api.replicate_token,
            self.api.tripo3d_key,
            self.api.dashscope_key
        ]
        
        missing_keys = [key for key in required_keys if key is None]
        if missing_keys:
            print("Warning: Missing API keys. Please set environment variables.")
            return False
        return True

# Global config instance
config = Config()
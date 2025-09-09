"""
Data structures for the CAST pipeline
"""
import numpy as np
from pathlib import Path
from enum import Enum 
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

@dataclass
class BoundingBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

@dataclass
class DetectedObject:
    """Represents a detected object in the scene"""
    id: int
    bbox: BoundingBox
    description: str
    confidence: float
    mask: Optional[np.ndarray] = None
    occ_mask: Optional[np.ndarray] = None
    cropped_image: Optional[np.ndarray] = None
    cropped_mask: Optional[np.ndarray] = None
    cropped_occ_mask: Optional[np.ndarray] = None
    occlusion_level: str = "no_occlusion"  # no_occlusion, some_occlusion, severe_occlusion
    generated_image: Optional[np.ndarray] = None

@dataclass
class MeshPose:
    translation: np.ndarray  # 3D translation vector
    rotation: np.ndarray     # 3x3 rotation matrix or quaternion
    scale: np.ndarray        # 3D scale vector
    transform: Optional[np.ndarray] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.translation.shape != (3,):
            raise ValueError("Translation must be a 3D vector")
        if self.rotation.shape != (3,3):
            raise ValueError("Rotation must be a 3x3 matrix")
        if self.scale.shape != (3,):
            raise ValueError("Scale must be a 3D vector")
            
        # Compose the transformation matrix from rotation, scale and translation
        transform = np.eye(4)
        # Add rotation and scale
        transform[:3,:3] = self.rotation * self.scale.reshape(3,1)
        # Add translation
        transform[:3,3] = self.translation
        self.transform = transform


@dataclass
class Mesh3D:
    """3D mesh representation"""
    vertices: np.ndarray
    faces: np.ndarray
    textures: Optional[np.ndarray] = None
    file_path: Optional[Path] = None
    input_image: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3)")
        if self.faces.shape[1] != 3:
            raise ValueError("Faces must have shape (M, 3)")

@dataclass
class Object3D:
    """3D object with mesh and pose"""
    id: int
    mesh: Mesh3D
    pose: MeshPose
    detected_object: DetectedObject
    point_cloud: Optional[np.ndarray] = None

@dataclass
class SceneGraph:
    """Scene graph representing object relationships"""
    relationships: List[Dict[str, Any]]
    objects: List[int]  # Object IDs
    
    def add_relationship(self, obj1_id: int, obj2_id: int, 
                        relationship_type: str, reason: str):
        """Add a relationship between two objects"""
        self.relationships.append({
            "pair": [obj1_id, obj2_id],
            "relationship": relationship_type,
            "reason": reason
        })

@dataclass
class DepthEstimation:
    """Depth estimation results"""
    depth_map: np.ndarray
    point_cloud: np.ndarray
    normal_map: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None

@dataclass
class SceneReconstruction:
    """Complete scene reconstruction result"""
    input_image: np.ndarray
    detected_objects: List[DetectedObject]
    depth_estimation: DepthEstimation
    objects_3d: List[Object3D]
    output_dir: Path
    scene_graph: Optional[SceneGraph] = None
    
    def save_summary(self) -> Dict[str, Any]:
        """Generate a summary of the reconstruction"""
        return {
            "num_objects": len(self.detected_objects),
            "num_relationships": len(self.scene_graph.relationships) if self.scene_graph is not None else 0,
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
                for obj in self.detected_objects
            ],
            "relationships": self.scene_graph.relationships if self.scene_graph is not None else []
        }

OCCLUSION_LEVELS = {
    "no_occlusion": 0,
    "some_occlusion": 1,
    "severe_occlusion": 2
}

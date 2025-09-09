"""
Scene Graph Extraction and Physical Optimization Module

This module handles:
1. Scene graph extraction using Qwen-VL
2. Physical correctness optimization using SDF and PyTorch following the paper algorithm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import open3d as o3d
import roma

# Visualization imports
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not available. Scene graph visualization will be disabled.")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# Import Open3D-based SDF implementation
from ..utils.open3d_sdf import Open3DSDFManager

from ..core.common import Object3D, SceneGraph, MeshPose
from ..utils.api_clients import QwenVLClient
from ..utils.image_utils import image_to_base64
from ..config.settings import config

class SceneGraphOptimizationModule:
    """Module for scene graph extraction and physical optimization following the paper algorithm"""
    
    def __init__(self):
        self.qwen_client = QwenVLClient()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimization parameters from config
        self.learning_rate = config.processing.sdf_learning_rate
        self.max_iterations = config.processing.sdf_max_iterations
        self.penetration_weight = config.processing.sdf_penetration_weight
        self.contact_weight = config.processing.sdf_contact_weight
        
        # Paper algorithm parameters
        self.sigma = 0.02  # Threshold for regularization (2cm)
        self.surface_sample_count = 1000  # Number of surface points to sample
        
        # Initialize Open3D SDF manager
        self.sdf_manager = Open3DSDFManager(
            resolution=0.01,  # 1cm voxel resolution
            padding=0.05,     # 5cm padding around bounding boxes
            device=str(self.device)
        )
        
    def create_annotated_image(self, image: np.ndarray, 
                             objects_3d: List[Object3D]) -> np.ndarray:
        """
        Create annotated image with numbered objects for scene graph extraction
        
        Args:
            image: Original scene image
            objects_3d: List of 3D objects with detected objects
            
        Returns:
            Annotated image with object numbers
        """
        import cv2
        
        annotated_image = image.copy()
        
        for obj in objects_3d:
            detected_obj = obj.detected_object
            
            # Get bounding box center
            center_x = int((detected_obj.bbox.x1 + detected_obj.bbox.x2) / 2)
            center_y = int((detected_obj.bbox.y1 + detected_obj.bbox.y2) / 2)
            
            # Draw bright numeric ID at center
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 4
            
            # Draw white background circle
            cv2.circle(annotated_image, (center_x, center_y), 30, (255, 255, 255), -1)
            cv2.circle(annotated_image, (center_x, center_y), 30, (0, 0, 0), 3)
            
            # Draw object ID number
            text = str(obj.id)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            
            cv2.putText(annotated_image, text, (text_x, text_y), font, 
                       font_scale, (0, 0, 0), thickness)
        
        return annotated_image
    
    def extract_scene_graph(self, image: np.ndarray, 
                           objects_3d: List[Object3D]) -> SceneGraph:
        """
        Extract scene graph relationships using Qwen-VL
        
        Args:
            image: Original scene image
            objects_3d: List of 3D objects
            
        Returns:
            SceneGraph with extracted relationships
        """
        print("Extracting scene graph using Qwen-VL...")
        
        # Create annotated image with object IDs
        annotated_image = self.create_annotated_image(image, objects_3d)
        
        # Load prompts from the scene graph prompt file
        system_prompt = self._load_scene_graph_prompt()
        
        # Create user prompt with object details
        object_descriptions = []
        for obj in objects_3d:
            object_descriptions.append(f"Object {obj.id}: {obj.detected_object.description}")
        
        user_prompt = f"""
### Object Details ###
I have labeled a bright numeric ID at the center for each visual object in the image.

Objects in the scene:
{chr(10).join(object_descriptions)}

Please analyze all relationships between the numbered objects and output JSON objects following the specified format.
Ensure each relationship includes:
1. The correct relationship type
2. A clear reason for the relationship

List all relationships as a JSON array.
        """
        
        # Query Qwen-VL
        try:
            response = self.qwen_client.analyze_scene_graph(
                annotated_image, system_prompt, user_prompt
            )
            
            if not response:
                print("Warning: Empty response from Qwen-VL")
                return SceneGraph(relationships=[], objects=[obj.id for obj in objects_3d])
            
            # Parse JSON response
            relationships = self._parse_scene_graph_response(response)
            
            scene_graph = SceneGraph(
                relationships=relationships,
                objects=[obj.id for obj in objects_3d]
            )
            
            print(f"Extracted {len(relationships)} relationships")
            return scene_graph
            
        except Exception as e:
            print(f"Error extracting scene graph: {e}")
            return SceneGraph(relationships=[], objects=[obj.id for obj in objects_3d])
    
    def _load_scene_graph_prompt(self) -> str:
        """Load system prompt for scene graph extraction"""
        try:
            prompt_file = Path(__file__).parent.parent.parent / "docs" / "scene_graph_prompt.txt"
            with open(prompt_file, 'r') as f:
                content = f.read()
                
            # Extract the system prompt part
            if 'prompting_text_system = """' in content:
                start = content.find('prompting_text_system = """') + len('prompting_text_system = """')
                end = content.find('"""', start)
                return content[start:end].strip()
            else:
                return content
                
        except Exception as e:
            print(f"Error loading scene graph prompt: {e}")
            # Fallback prompt
            return """
You are an expert in object recognition and spatial reasoning.
Analyze an image with numbered objects and determine their relationships.
For each pair of related objects, output a JSON object containing the relationship details.
Only objects that are in contact with each other should have a relationship.
"""
    
    def _parse_scene_graph_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse Qwen-VL response to extract relationships"""
        relationships = []
        
        try:
            # Try to extract JSON from response
            if '[' in response and ']' in response:
                # Find JSON array
                start = response.find('[')
                end = response.rfind(']') + 1
                json_str = response[start:end]
                relationships = json.loads(json_str)
            elif '{' in response:
                # Multiple JSON objects
                import re
                json_matches = re.findall(r'\{[^}]+\}', response)
                for match in json_matches:
                    try:
                        rel = json.loads(match)
                        relationships.append(rel)
                    except:
                        continue
            
            # Validate and clean relationships
            valid_relationships = []
            for rel in relationships:
                if isinstance(rel, dict) and 'pair' in rel and 'relationship' in rel:
                    valid_relationships.append(rel)
            
            return valid_relationships
            
        except Exception as e:
            print(f"Error parsing scene graph response: {e}")
            print(f"Raw response: {response}")
            return []
    
    def sample_surface_points(self, mesh: o3d.geometry.TriangleMesh, 
                            num_points: int = 1000) -> torch.Tensor:
        """
        Sample points from mesh surface using Open3D
        
        Args:
            mesh: Open3D triangle mesh
            num_points: Number of points to sample
            
        Returns:
            Sampled surface points as torch tensor (num_points, 3)
        """
        # Use Open3D's Poisson disk sampling for uniform surface sampling
        pcd = mesh.sample_points_poisson_disk(num_points)
        points = np.asarray(pcd.points)
        
        # If we don't get enough points, fall back to uniform sampling
        if len(points) < num_points // 2:
            pcd = mesh.sample_points_uniformly(num_points)
            points = np.asarray(pcd.points)
        
        return torch.tensor(points, dtype=torch.float32, device=self.device)
    
    def compute_sdf_values(self, obj: Object3D, query_points: torch.Tensor) -> torch.Tensor:
        """
        Compute SDF values using Open3D implementation
        
        Args:
            obj: Object3D to compute SDF for
            query_points: Query points (N, 3)
            
        Returns:
            SDF values (N,)
        """
        try:
            return self.sdf_manager.compute_sdf_values(obj, query_points)
        except Exception as e:
            print(f"Warning: SDF query failed for object {obj.id}: {e}")
            return torch.zeros(query_points.shape[0], device=self.device)
    

    
    def compute_contact_cost(self, obj_i: Object3D, obj_j: Object3D,
                           surface_points_i: torch.Tensor, 
                           surface_points_j: torch.Tensor) -> torch.Tensor:
        """
        Compute contact cost following the paper algorithm
        
        C(T_i, T_j) = C(T_i, T_j; o_i -> o_j) + C(T_i, T_j; o_j -> o_i)
        
        Args:
            obj_i: First object
            obj_j: Second object
            surface_points_i: Surface points of object i
            surface_points_j: Surface points of object j
            
        Returns:
            Total contact cost
        """
        # Transform surface points to world coordinates
        transform_i = self._pose_to_transform_matrix(obj_i.pose)
        transform_j = self._pose_to_transform_matrix(obj_j.pose)
        
        # Transform surface points
        points_i_world = self._transform_points(surface_points_i, transform_i)
        points_j_world = self._transform_points(surface_points_j, transform_j)
        
        # Compute C(T_i, T_j; o_i -> o_j)
        # Query SDF of object i at surface points of object j
        sdf_i_at_j = self.compute_sdf_values(obj_i, points_j_world)
        
        # Penetration indicator: I(D_i(p) < 0)
        penetration_mask_j = sdf_i_at_j < 0
        num_penetrating_j = torch.sum(penetration_mask_j.float())
        
        if num_penetrating_j > 0:
            # Average penetration depth
            avg_penetration_j = -torch.sum(sdf_i_at_j * penetration_mask_j.float()) / num_penetrating_j
        else:
            avg_penetration_j = torch.tensor(0.0, device=self.device)
        
        # Minimum distance constraint
        min_distance_j = torch.max(torch.min(sdf_i_at_j), torch.tensor(0.0, device=self.device))
        
        cost_i_to_j = avg_penetration_j + min_distance_j
        
        # Compute C(T_i, T_j; o_j -> o_i)
        # Query SDF of object j at surface points of object i
        sdf_j_at_i = self.compute_sdf_values(obj_j, points_i_world)
        
        # Penetration indicator: I(D_j(p) < 0)
        penetration_mask_i = sdf_j_at_i < 0
        num_penetrating_i = torch.sum(penetration_mask_i.float())
        
        if num_penetrating_i > 0:
            # Average penetration depth
            avg_penetration_i = -torch.sum(sdf_j_at_i * penetration_mask_i.float()) / num_penetrating_i
        else:
            avg_penetration_i = torch.tensor(0.0, device=self.device)
        
        # Minimum distance constraint
        min_distance_i = torch.max(torch.min(sdf_j_at_i), torch.tensor(0.0, device=self.device))
        
        cost_j_to_i = avg_penetration_i + min_distance_i
        
        # Total bilateral contact cost
        total_cost = cost_i_to_j + cost_j_to_i
        
        return total_cost
    
    def compute_support_cost(self, obj_i: Object3D, obj_j: Object3D,
                           surface_points_j: torch.Tensor,
                           is_flat_surface: bool = False) -> torch.Tensor:
        """
        Compute support cost following the paper algorithm
        
        For support: C(T_i, T_j) = |min_{p in ∂o_j} D_i(p(T_j))|
        For flat surfaces: regularization term with threshold σ
        
        Args:
            obj_i: Supporting object (static)
            obj_j: Supported object (to be optimized)
            surface_points_j: Surface points of supported object j
            is_flat_surface: Whether object i is a flat supporting surface
            
        Returns:
            Support cost
        """
        # Transform surface points of object j to world coordinates
        transform_j = self._pose_to_transform_matrix(obj_j.pose)
        points_j_world = self._transform_points(surface_points_j, transform_j)
        
        # Query SDF of supporting object i at surface points of supported object j
        sdf_i_at_j = self.compute_sdf_values(obj_i, points_j_world)
        
        if is_flat_surface:
            # Regularization for flat supporting surfaces
            # C(T_i, T_j) = Σ(D_i(p) * I(0 < D_i(p) < σ)) / Σ(I(0 < D_i(p) < σ))
            close_mask = (sdf_i_at_j > 0) & (sdf_i_at_j < self.sigma)
            num_close = torch.sum(close_mask.float())
            
            if num_close > 0:
                cost = torch.sum(sdf_i_at_j * close_mask.float()) / num_close
            else:
                # Fallback to minimum distance if no points are close
                cost = torch.abs(torch.min(sdf_i_at_j))
        else:
            # Standard support constraint: |min_{p in ∂o_j} D_i(p(T_j))|
            cost = torch.abs(torch.min(sdf_i_at_j))
        
        return cost
    
    def _transform_points(self, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """
        Transform points using 4x4 transformation matrix
        
        Args:
            points: Points to transform (N, 3)
            transform: 4x4 transformation matrix
            
        Returns:
            Transformed points (N, 3)
        """
        # Convert to homogeneous coordinates
        points_homo = torch.cat([points, torch.ones(points.shape[0], 1, device=self.device)], dim=1)
        
        # Apply transformation
        transformed_homo = (transform @ points_homo.T).T
        
        # Return 3D coordinates
        return transformed_homo[:, :3]
    
    def _pose_to_transform_matrix(self, pose: MeshPose) -> torch.Tensor:
        """Convert Pose6D to 4x4 transformation matrix using current pose values"""
        transform = torch.eye(4, device=self.device, dtype=torch.float32)
        
        # Handle rotation matrix
        if hasattr(pose, '_rotation_tensor'):
            # Use the current tensor from optimization
            rotation_matrix = pose._rotation_tensor
        else:
            rotation_matrix = torch.tensor(pose.rotation, device=self.device, dtype=torch.float32)
        
        # Handle scale
        if hasattr(pose, '_scale_tensor'):
            # Use the current tensor from optimization
            scale = pose._scale_tensor
        else:
            # Check if pose has scale field, otherwise default to ones
            if hasattr(pose, 'scale') and pose.scale is not None:
                scale = torch.tensor(pose.scale, device=self.device, dtype=torch.float32)
            else:
                scale = torch.ones(3, device=self.device, dtype=torch.float32)
        
        # Handle translation
        if hasattr(pose, '_translation_tensor'):
            # Use the current tensor from optimization
            translation = pose._translation_tensor
        else:
            translation = torch.tensor(pose.translation, device=self.device, dtype=torch.float32)
        
        # Apply scale to rotation matrix
        scaled_rotation = rotation_matrix * scale.unsqueeze(0)
        
        transform[:3, :3] = scaled_rotation
        transform[:3, 3] = translation
        return transform
    
    def _rotvec_to_matrix_roma(self, rotvec: torch.Tensor) -> torch.Tensor:
        """Convert rotation vector to rotation matrix using RoMa"""
        return roma.rotvec_to_rotmat(rotvec)
    
    def _matrix_to_rotvec_roma(self, rotmat: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to rotation vector using RoMa"""
        return roma.rotmat_to_rotvec(rotmat)
    
    def prepare_optimization_data(self, objects_3d: List[Object3D]) -> Tuple[List[Dict], List[torch.Tensor]]:
        """
        Prepare data structures for optimization
        
        Args:
            objects_3d: List of 3D objects
            
        Returns:
            Tuple of (pose_params, surface_points)
        """
        pose_params = []
        surface_points = []
        
        for obj in objects_3d:
            # Create optimizable pose parameters
            translation = torch.tensor(obj.pose.translation, requires_grad=True, 
                                     dtype=torch.float32, device=self.device)
            
            # Convert rotation matrix to rotation vector using RoMa
            rotation_matrix = torch.tensor(obj.pose.rotation, dtype=torch.float32, device=self.device)
            rotation_vec = self._matrix_to_rotvec_roma(rotation_matrix)
            rotation_vec.requires_grad_(True)
            
            pose_params.append({
                'translation': translation, 
                'rotation': rotation_vec,
                'object_id': obj.id
            })
            
            # Initialize SDF calculator for this object
            self.sdf_manager.get_sdf_calculator(obj)
            
            # Create Open3D mesh and sample surface points
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(obj.mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(obj.mesh.faces)
            o3d_mesh.compute_vertex_normals()
            
            # Sample surface points using Open3D
            surface_pts = self.sample_surface_points(o3d_mesh, self.surface_sample_count)
            surface_points.append(surface_pts)
        
        return pose_params, surface_points
    
    def update_object_poses(self, objects_3d: List[Object3D], pose_params: List[Dict]) -> None:
        """
        Update object poses with current optimization parameters
        
        Args:
            objects_3d: List of objects to update
            pose_params: Current pose parameters from optimization
        """
        for obj, params in zip(objects_3d, pose_params):
            # Convert rotation vector to matrix using RoMa
            rotation_matrix = self._rotvec_to_matrix_roma(params['rotation'])
            
            # Store tensors for transformation computation
            obj.pose._rotation_tensor = rotation_matrix
            obj.pose._translation_tensor = params['translation']
            
            # Update numpy arrays for external access
            obj.pose.translation = params['translation'].detach().cpu().numpy()
            obj.pose.rotation = rotation_matrix.detach().cpu().numpy()
            
            # Preserve scale if it exists, otherwise set to ones
            if not hasattr(obj.pose, 'scale') or obj.pose.scale is None:
                obj.pose.scale = np.ones(3)
            
            # Update SDF manager with new pose
            self.sdf_manager.update_object_pose(obj)
    
    def compute_total_cost(self, objects_3d: List[Object3D], scene_graph: SceneGraph,
                          pose_params: List[Dict], 
                          surface_points: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute total cost following the paper formulation:
        min_{T={T_1, T_2, ..., T_N}} Σ_{i,j} C(T_i, T_j; o_i, o_j)
        
        Args:
            objects_3d: List of objects
            scene_graph: Scene graph with relationships
            pose_params: Current pose parameters
            surface_points: Surface points for each object
            
        Returns:
            Total cost value
        """
        total_cost = torch.tensor(0.0, device=self.device)
        
        # Update poses with current parameters
        self.update_object_poses(objects_3d, pose_params)
        
        # Compute cost for each relationship with normalization
        for rel in scene_graph.relationships:
            obj1_id, obj2_id = rel['pair']
            relationship_type = rel['relationship'].lower()
            
            # Find object indices
            obj1_idx = next((i for i, obj in enumerate(objects_3d) if obj.id == obj1_id), None)
            obj2_idx = next((i for i, obj in enumerate(objects_3d) if obj.id == obj2_id), None)
            
            if obj1_idx is None or obj2_idx is None:
                continue
            
            obj1 = objects_3d[obj1_idx]
            obj2 = objects_3d[obj2_idx]
            points1 = surface_points[obj1_idx]
            points2 = surface_points[obj2_idx]
            
            # Compute normalization transform for this object pair
            scale_factor, center, extent = self.compute_normalization_transform(obj1, obj2)
            
            # Store original poses
            orig_trans1 = obj1.pose._translation_tensor.clone() if hasattr(obj1.pose, '_translation_tensor') else None
            orig_trans2 = obj2.pose._translation_tensor.clone() if hasattr(obj2.pose, '_translation_tensor') else None
            
            # Normalize object pair
            self.normalize_object_pair(obj1, obj2, scale_factor, center)
            
            try:
                # Determine relationship type and compute appropriate cost
                if relationship_type in ['support', 'supports', 'on', 'above']:
                    # Support relationship: obj1 supports obj2
                    # Check if obj1 is a flat surface (ground, table, etc.)
                    is_flat = self._is_flat_surface(obj1)
                    cost = self.compute_support_cost(obj1, obj2, points2, is_flat)
                    
                elif relationship_type in ['stack', 'stacked', 'contact', 'touch', 'touching', 'lean', 'leaning']:
                    # Contact relationship: bilateral constraint
                    cost = self.compute_contact_cost(obj1, obj2, points1, points2)
                    
                else:
                    # Default to contact constraint
                    cost = self.compute_contact_cost(obj1, obj2, points1, points2)
                
                total_cost += cost
                
            finally:
                # Denormalize object pair back to original scale
                self.denormalize_object_pair(obj1, obj2, scale_factor, center)
        
        return total_cost
    
    def _is_flat_surface(self, obj: Object3D) -> bool:
        """
        Determine if an object represents a flat supporting surface
        
        Args:
            obj: Object to check
            
        Returns:
            True if object is likely a flat surface
        """
        description = obj.detected_object.description.lower()
        flat_keywords = ['ground', 'floor', 'table', 'desk', 'surface', 'platform', 'base', 'wall']
        return any(keyword in description for keyword in flat_keywords)
    
    def compute_normalization_transform(self, obj1: Object3D, obj2: Object3D) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute normalization transform for a pair of objects to fit in [-1, 1] range
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            Tuple of (scale_factor, translation_offset, combined_bbox_center)
        """
        # Get transformed vertices for both objects
        transform1 = self._pose_to_transform_matrix(obj1.pose)
        transform2 = self._pose_to_transform_matrix(obj2.pose)
        
        # Transform vertices to world coordinates
        vertices1 = torch.tensor(obj1.mesh.vertices, dtype=torch.float32, device=self.device)
        vertices2 = torch.tensor(obj2.mesh.vertices, dtype=torch.float32, device=self.device)
        
        vertices1_world = self._transform_points(vertices1, transform1)
        vertices2_world = self._transform_points(vertices2, transform2)
        
        # Combine all vertices
        all_vertices = torch.cat([vertices1_world, vertices2_world], dim=0)
        
        # Compute bounding box of the union
        min_coords = torch.min(all_vertices, dim=0)[0]
        max_coords = torch.max(all_vertices, dim=0)[0]
        
        # Compute center and scale
        center = (min_coords + max_coords) / 2.0
        extent = max_coords - min_coords
        
        # Scale factor to fit in [-1, 1] range (with small margin)
        scale_factor = 1.8 / torch.max(extent)  # 1.8 instead of 2.0 for margin
        
        return scale_factor, center, extent
    
    def normalize_object_pair(self, obj1: Object3D, obj2: Object3D, 
                             scale_factor: torch.Tensor, center: torch.Tensor) -> None:
        """
        Normalize object pair to [-1, 1] range
        
        Args:
            obj1: First object
            obj2: Second object  
            scale_factor: Scale factor for normalization
            center: Center point for normalization
        """
        # Update translations
        if hasattr(obj1.pose, '_translation_tensor'):
            obj1.pose._translation_tensor = (obj1.pose._translation_tensor - center) * scale_factor
        if hasattr(obj2.pose, '_translation_tensor'):
            obj2.pose._translation_tensor = (obj2.pose._translation_tensor - center) * scale_factor
            
        # Update SDF managers with normalized poses
        self.sdf_manager.update_object_pose(obj1)
        self.sdf_manager.update_object_pose(obj2)
    
    def denormalize_object_pair(self, obj1: Object3D, obj2: Object3D,
                               scale_factor: torch.Tensor, center: torch.Tensor) -> None:
        """
        Denormalize object pair back to original scale
        
        Args:
            obj1: First object
            obj2: Second object
            scale_factor: Scale factor used for normalization
            center: Center point used for normalization
        """
        # Restore original translations
        if hasattr(obj1.pose, '_translation_tensor'):
            obj1.pose._translation_tensor = obj1.pose._translation_tensor / scale_factor + center
        if hasattr(obj2.pose, '_translation_tensor'):
            obj2.pose._translation_tensor = obj2.pose._translation_tensor / scale_factor + center
            
        # Update numpy arrays for external access
        obj1.pose.translation = obj1.pose._translation_tensor.detach().cpu().numpy()
        obj2.pose.translation = obj2.pose._translation_tensor.detach().cpu().numpy()
        
        # Update SDF managers with denormalized poses
        self.sdf_manager.update_object_pose(obj1)
        self.sdf_manager.update_object_pose(obj2)
    
    def optimize_poses(self, objects_3d: List[Object3D], 
                      scene_graph: SceneGraph) -> List[Object3D]:
        """
        Optimize object poses for physical correctness using the paper algorithm
        
        Args:
            objects_3d: List of 3D objects with initial poses
            scene_graph: Scene graph with relationships
            
        Returns:
            List of objects with optimized poses
        """
        print("Optimizing poses using paper algorithm with RoMa and Open3D SDF...")
        
        if not scene_graph.relationships:
            print("No relationships found, skipping optimization")
            return objects_3d
        
        # Prepare optimization data
        pose_params, surface_points = self.prepare_optimization_data(objects_3d)
        
        # Setup optimizer
        all_params = []
        for params in pose_params:
            all_params.extend([params['translation'], params['rotation']])
        
        optimizer = optim.Adam(all_params, lr=self.learning_rate)
        
        print(f"Starting optimization with {len(all_params)} parameters...")
        
        # Optimization loop
        best_loss = float('inf')
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Compute total cost following paper formulation
            total_cost = self.compute_total_cost(objects_3d, scene_graph, pose_params, 
                                               surface_points)
            
            # Backward pass
            if total_cost.requires_grad and not torch.isnan(total_cost):
                total_cost.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
            
            current_loss = total_cost.item()
            
            # Track best loss
            if current_loss < best_loss:
                best_loss = current_loss
            
            # Print progress
            if iteration % 50 == 0:
                print(f"  Iteration {iteration}, Loss: {current_loss:.6f}, Best: {best_loss:.6f}")
            
            # Early stopping if loss is very small or not improving
            if current_loss < 1e-6:
                print(f"  Converged at iteration {iteration}")
                break
        
        # Final pose update
        self.update_object_poses(objects_3d, pose_params)
        
        # Clear SDF cache to free memory
        self.sdf_manager.clear_cache()
        
        print(f"Pose optimization complete. Final loss: {best_loss:.6f}")
        return objects_3d
    
    def fix_penetrations(self, objects_3d: List[Object3D], 
                        scene_graph: SceneGraph) -> List[Object3D]:
        """
        Fix object penetrations using Open3D SDF
        
        Args:
            objects_3d: List of objects
            scene_graph: Scene graph with relationships
            
        Returns:
            List of objects with fixed penetrations
        """
        print("Fixing penetrations using Open3D SDF...")
        
        # Initialize SDF calculators for all objects
        for obj in objects_3d:
            self.sdf_manager.get_sdf_calculator(obj)
        
        # Check and fix penetrations for each relationship
        for rel in scene_graph.relationships:
            obj1_id, obj2_id = rel['pair']
            
            # Find objects
            obj1_idx = next((i for i, obj in enumerate(objects_3d) if obj.id == obj1_id), None)
            obj2_idx = next((i for i, obj in enumerate(objects_3d) if obj.id == obj2_id), None)
            
            if obj1_idx is None or obj2_idx is None:
                continue
            
            obj1 = objects_3d[obj1_idx]
            obj2 = objects_3d[obj2_idx]
            
            # Sample points from both objects
            o3d_mesh1 = o3d.geometry.TriangleMesh()
            o3d_mesh1.vertices = o3d.utility.Vector3dVector(obj1.mesh.vertices)
            o3d_mesh1.triangles = o3d.utility.Vector3iVector(obj1.mesh.faces)
            points1 = self.sample_surface_points(o3d_mesh1, 500)
            
            o3d_mesh2 = o3d.geometry.TriangleMesh()
            o3d_mesh2.vertices = o3d.utility.Vector3dVector(obj2.mesh.vertices)
            o3d_mesh2.triangles = o3d.utility.Vector3iVector(obj2.mesh.faces)
            points2 = self.sample_surface_points(o3d_mesh2, 500)
            
            # Transform points to world coordinates
            transform1 = self._pose_to_transform_matrix(obj1.pose)
            transform2 = self._pose_to_transform_matrix(obj2.pose)
            
            points1_world = self._transform_points(points1, transform1)
            points2_world = self._transform_points(points2, transform2)
            
            # Check for penetrations
            sdf2_at_1 = self.compute_sdf_values(obj2, points1_world)
            penetration_depth1 = torch.min(sdf2_at_1)
            
            if penetration_depth1 < 0:
                # Move obj1 away from obj2
                separation_vector = self._compute_separation_vector(obj1, obj2)
                obj1.pose.translation += separation_vector * abs(penetration_depth1.item()) * 1.1
                # Update SDF after pose change
                self.sdf_manager.update_object_pose(obj1)
            
            sdf1_at_2 = self.compute_sdf_values(obj1, points2_world)
            penetration_depth2 = torch.min(sdf1_at_2)
            
            if penetration_depth2 < 0:
                # Move obj2 away from obj1
                separation_vector = self._compute_separation_vector(obj2, obj1)
                obj2.pose.translation += separation_vector * abs(penetration_depth2.item()) * 1.1
                # Update SDF after pose change
                self.sdf_manager.update_object_pose(obj2)
        
        print("Penetration fixing complete")
        return objects_3d
    
    def _compute_separation_vector(self, obj1: Object3D, obj2: Object3D) -> np.ndarray:
        """
        Compute separation vector between two objects
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            Unit vector pointing from obj2 to obj1
        """
        center1 = np.mean(obj1.mesh.vertices, axis=0) + obj1.pose.translation
        center2 = np.mean(obj2.mesh.vertices, axis=0) + obj2.pose.translation
        
        direction = center1 - center2
        norm = np.linalg.norm(direction)
        
        if norm < 1e-8:
            # Default separation along z-axis
            return np.array([0, 0, 1])
        
        return direction / norm
    
    def visualize_scene_graph(self, scene_graph: SceneGraph, objects_3d: List[Object3D], 
                             output_dir: Path, method: str = 'both') -> None:
        """
        Visualize scene graph using different methods
        
        Args:
            scene_graph: Scene graph to visualize
            objects_3d: List of 3D objects for context
            output_dir: Output directory for visualization files
            method: Visualization method ('graphviz', 'matplotlib', 'both')
        """
        print(f"Creating scene graph visualization using {method}...")
        
        if method in ['graphviz', 'both'] and GRAPHVIZ_AVAILABLE:
            self._visualize_with_graphviz(scene_graph, objects_3d, output_dir)
        
        if method in ['matplotlib', 'both']:
            self._visualize_with_matplotlib(scene_graph, objects_3d, output_dir)
    
    def _visualize_with_graphviz(self, scene_graph: SceneGraph, objects_3d: List[Object3D], 
                                output_dir: Path) -> None:
        """Create scene graph visualization using Graphviz"""
        try:
            # Create a directed graph
            dot = graphviz.Digraph(comment='Scene Graph')
            dot.attr(rankdir='TB', size='12,8', dpi='300')
            dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
            dot.attr('edge', fontname='Arial', fontsize='10')
            
            # Create object lookup
            obj_lookup = {obj.id: obj for obj in objects_3d}
            
            # Add nodes for each object
            for obj_id in scene_graph.objects:
                if obj_id in obj_lookup:
                    obj = obj_lookup[obj_id]
                    description = obj.detected_object.description
                    confidence = obj.pose.confidence
                    
                    # Color nodes based on confidence
                    if confidence > 0.7:
                        color = '#90EE90'  # Light green for high confidence
                    elif confidence > 0.4:
                        color = '#FFE4B5'  # Light orange for medium confidence
                    else:
                        color = '#FFB6C1'  # Light red for low confidence
                    
                    # Create node label with object info
                    label = f"Object {obj_id}\\n{description}\\nConf: {confidence:.2f}"
                    dot.node(str(obj_id), label, fillcolor=color)
                else:
                    dot.node(str(obj_id), f"Object {obj_id}", fillcolor='lightgray')
            
            # Add edges for relationships
            relationship_colors = {
                'support': 'blue',
                'supports': 'blue', 
                'on': 'blue',
                'above': 'blue',
                'contact': 'red',
                'touch': 'red',
                'touching': 'red',
                'stack': 'purple',
                'stacked': 'purple',
                'lean': 'orange',
                'leaning': 'orange'
            }
            
            for rel in scene_graph.relationships:
                obj1_id, obj2_id = rel['pair']
                relationship = rel['relationship'].lower()
                reason = rel.get('reason', '')
                
                # Get edge color
                color = relationship_colors.get(relationship, 'black')
                
                # Create edge label
                edge_label = f"{relationship}"
                if reason and len(reason) < 50:
                    edge_label += f"\\n({reason[:50]})"
                
                dot.edge(str(obj1_id), str(obj2_id), label=edge_label, color=color)
            
            # Save the graph
            output_file = output_dir / "scene_graph_visualization"
            dot.render(output_file, format='png', cleanup=True)
            dot.render(output_file, format='pdf', cleanup=True)
            
            print(f"Graphviz visualization saved to {output_file}.png and {output_file}.pdf")
            
        except Exception as e:
            print(f"Error creating Graphviz visualization: {e}")
    
    def _visualize_with_matplotlib(self, scene_graph: SceneGraph, objects_3d: List[Object3D], 
                                  output_dir: Path) -> None:
        """Create scene graph visualization using matplotlib and networkx"""
        try:
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Create object lookup
            obj_lookup = {obj.id: obj for obj in objects_3d}
            
            # Add nodes
            for obj_id in scene_graph.objects:
                if obj_id in obj_lookup:
                    obj = obj_lookup[obj_id]
                    G.add_node(obj_id, 
                              description=obj.detected_object.description,
                              confidence=obj.pose.confidence)
                else:
                    G.add_node(obj_id, description=f"Object {obj_id}", confidence=0.0)
            
            # Add edges
            edge_labels = {}
            edge_colors = []
            relationship_color_map = {
                'support': 'blue', 'supports': 'blue', 'on': 'blue', 'above': 'blue',
                'contact': 'red', 'touch': 'red', 'touching': 'red',
                'stack': 'purple', 'stacked': 'purple',
                'lean': 'orange', 'leaning': 'orange'
            }
            
            for rel in scene_graph.relationships:
                obj1_id, obj2_id = rel['pair']
                relationship = rel['relationship'].lower()
                
                G.add_edge(obj1_id, obj2_id)
                edge_labels[(obj1_id, obj2_id)] = relationship
                edge_colors.append(relationship_color_map.get(relationship, 'black'))
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            
            # Use spring layout for better node positioning
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes with colors based on confidence
            node_colors = []
            for node in G.nodes():
                confidence = G.nodes[node].get('confidence', 0.0)
                if confidence > 0.7:
                    node_colors.append('#90EE90')  # Light green
                elif confidence > 0.4:
                    node_colors.append('#FFE4B5')  # Light orange
                else:
                    node_colors.append('#FFB6C1')  # Light red
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=3000, alpha=0.9)
            
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                  arrows=True, arrowsize=20, width=2,
                                  connectionstyle="arc3,rad=0.1")
            
            # Add node labels with object descriptions
            node_labels = {}
            for node in G.nodes():
                description = G.nodes[node].get('description', f'Object {node}')
                confidence = G.nodes[node].get('confidence', 0.0)
                # Truncate long descriptions
                if len(description) > 20:
                    description = description[:17] + "..."
                node_labels[node] = f"Obj {node}\n{description}\n({confidence:.2f})"
            
            nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold')
            
            # Add edge labels
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
            
            # Add title and legend
            plt.title("Scene Graph Visualization", fontsize=16, fontweight='bold', pad=20)
            
            # Create legend
            legend_elements = [
                patches.Patch(color='#90EE90', label='High Confidence (>0.7)'),
                patches.Patch(color='#FFE4B5', label='Medium Confidence (0.4-0.7)'),
                patches.Patch(color='#FFB6C1', label='Low Confidence (<0.4)'),
                patches.Patch(color='blue', label='Support Relations'),
                patches.Patch(color='red', label='Contact Relations'),
                patches.Patch(color='purple', label='Stack Relations'),
                patches.Patch(color='orange', label='Lean Relations')
            ]
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save the visualization
            output_file = output_dir / "scene_graph_matplotlib.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / "scene_graph_matplotlib.pdf", bbox_inches='tight')
            plt.close()
            
            print(f"Matplotlib visualization saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating matplotlib visualization: {e}")
    
    def run(self, image: np.ndarray, objects_3d: List[Object3D], 
            output_dir: Optional[Path] = None) -> Tuple[SceneGraph, List[Object3D]]:
        """
        Run the complete scene graph extraction and optimization pipeline
        
        Args:
            image: Original scene image
            objects_3d: List of 3D objects with initial poses
            output_dir: Directory to save results
            
        Returns:
            Tuple of (scene_graph, optimized_objects_3d)
        """
        print("Starting scene graph extraction and optimization...")
        
        # Step 1: Extract scene graph
        scene_graph = self.extract_scene_graph(image, objects_3d)
        #  first save the scene graph for debugging
        if output_dir:
            self._save_results(scene_graph, None, image, output_dir)
            # Create scene graph visualization
            sg_dir = output_dir / "scene_graph_optimization"
            self.visualize_scene_graph(scene_graph, objects_3d, sg_dir)
        
        # Step 2: Fix obvious penetrations first
        # objects_3d = self.fix_penetrations(objects_3d, scene_graph)
        
        # Step 3: Optimize poses for physical correctness using paper algorithm
        optimized_objects = self.optimize_poses(objects_3d, scene_graph)
        # Then save the optimized object poses
        if output_dir:
            self._save_results(None, optimized_objects, image, output_dir)
        
        print("Scene graph extraction and optimization complete")
        return scene_graph, optimized_objects
    
    def _save_results(self, scene_graph: SceneGraph, objects_3d: List[Object3D],
                     image: np.ndarray, output_dir: Path) -> None:
        """Save scene graph and optimization results"""
        sg_dir = output_dir / "scene_graph_optimization"
        sg_dir.mkdir(exist_ok=True, parents=True)
        
        # Save scene graph
        if scene_graph is not None:
            with open(sg_dir / "scene_graph.json", "w") as f:
                json.dump({
                    "objects": scene_graph.objects,
                    "relationships": scene_graph.relationships
                }, f, indent=2)

            print(f"Saved scene graph with {len(scene_graph.relationships)} relationships")
        
        # Save annotated image
        if objects_3d is not None:
            annotated_image = self.create_annotated_image(image, objects_3d)
            from ..utils.image_utils import save_image
            save_image(annotated_image, sg_dir / "annotated_image.png")
            
            # Save optimized poses
            optimized_poses = []
            for obj in objects_3d:
                pose_data = {
                    "object_id": obj.id,
                    "description": obj.detected_object.description,
                    "translation": obj.pose.translation.tolist(),
                    "rotation_matrix": obj.pose.rotation.tolist(),
                    "scale": obj.pose.scale.tolist() if hasattr(obj.pose, 'scale') and obj.pose.scale is not None else [1.0, 1.0, 1.0],
                    "confidence": obj.pose.confidence
                }
                optimized_poses.append(pose_data)
        
            with open(sg_dir / "optimized_poses.json", "w") as f:
                json.dump(optimized_poses, f, indent=2)

            print(f"Saved optimized poses for {len(objects_3d)} objects")
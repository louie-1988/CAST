"""
Open3D-based SDF implementation for scene graph optimization

This module provides SDF (Signed Distance Function) computation using Open3D
as a replacement for the external sdf package dependency.
"""
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from typing import List, Tuple, Optional
from ..core.common import Object3D


class Open3DSDF:
    """
    Open3D-based SDF implementation that precomputes SDF values on a voxel grid
    for individual objects and uses PyTorch grid sampling for interpolation during optimization.
    """
    
    def __init__(self, obj: Object3D, resolution: float = 0.01, padding: float = 0.05, device: str = "cuda"):
        """
        Initialize Open3D SDF calculator for a single object
        
        Args:
            obj: Object3D instance to compute SDF for
            resolution: Voxel grid resolution in meters
            padding: Extra padding around bounding box
            device: PyTorch device for tensors
        """
        self.obj = obj
        self.resolution = resolution
        self.padding = padding
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize raycast scene
        self.scene = o3d.t.geometry.RaycastingScene()
        self.sdf_grid = None
        self.grid_bounds = None
        self.grid_shape = None
        
        # Setup the scene with this object
        self._setup_scene()
        
    def _setup_scene(self) -> None:
        """
        Setup the raycast scene for this object
        """
        # Clear previous scene
        self.scene = o3d.t.geometry.RaycastingScene()
        
        # Convert mesh to Open3D tensor format
        mesh = self._object_to_tensor_mesh(self.obj)
        
        # Add mesh to scene
        self.scene.add_triangles(mesh)
        
        # Compute bounding box with padding
        bbox = mesh.get_axis_aligned_bounding_box()
        
        min_bound = bbox.min_bound.numpy() - self.padding
        max_bound = bbox.max_bound.numpy() + self.padding
        
        self.grid_bounds = (min_bound, max_bound)
        
        # Generate voxel grid
        self._generate_sdf_grid()
        
    def update_for_pose_change(self) -> None:
        """
        Update SDF grid when object pose changes during optimization
        """
        self._setup_scene()
        
    def _object_to_tensor_mesh(self, obj: Object3D) -> o3d.t.geometry.TriangleMesh:
        """
        Convert Object3D to Open3D tensor mesh with current pose applied
        
        Args:
            obj: Object3D instance
            
        Returns:
            Open3D tensor mesh
        """
        # Create Open3D legacy mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(obj.mesh.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(obj.mesh.faces)
        mesh.compute_vertex_normals()
        
        # Apply pose transformation
        transform = np.eye(4)
        transform[:3, :3] = obj.pose.rotation
        transform[:3, 3] = obj.pose.translation
        mesh.transform(transform)
        
        # Convert to tensor mesh
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        return tensor_mesh
        
    def _generate_sdf_grid(self) -> None:
        """
        Generate SDF values on a uniform voxel grid within the bounding box union
        """
        min_bound, max_bound = self.grid_bounds
        
        # Create coordinate arrays
        x = np.arange(min_bound[0], max_bound[0] + self.resolution, self.resolution)
        y = np.arange(min_bound[1], max_bound[1] + self.resolution, self.resolution)
        z = np.arange(min_bound[2], max_bound[2] + self.resolution, self.resolution)
        
        self.grid_shape = (len(x), len(y), len(z))
        
        # Generate grid points
        grid_points = np.array(np.meshgrid(x, y, z, indexing='ij')).T.reshape(-1, 3)
        
        # Convert to Open3D tensor
        query_points = o3d.core.Tensor(grid_points, dtype=o3d.core.Dtype.Float32)
        
        # Compute signed distances
        sdf_values = self.scene.compute_signed_distance(query_points)
        
        # Reshape to grid and convert to PyTorch tensor
        sdf_grid_np = sdf_values.numpy().reshape(self.grid_shape)
        self.sdf_grid = torch.tensor(sdf_grid_np, dtype=torch.float32, device=self.device)
        
        # Store grid coordinates for interpolation
        self.grid_coords = {
            'x': torch.tensor(x, dtype=torch.float32, device=self.device),
            'y': torch.tensor(y, dtype=torch.float32, device=self.device),
            'z': torch.tensor(z, dtype=torch.float32, device=self.device)
        }
        
    def query(self, points: torch.Tensor) -> torch.Tensor:
        """
        Query SDF values at given points using PyTorch grid sampling
        
        Args:
            points: Query points (N, 3) in world coordinates
            
        Returns:
            SDF values (N,) 
        """
        if self.sdf_grid is None:
            raise RuntimeError("SDF grid not initialized. Call setup_scene first.")
            
        # Convert world coordinates to grid coordinates
        grid_coords = self._world_to_grid_coords(points)
        
        # Normalize coordinates to [-1, 1] for grid_sample
        normalized_coords = self._normalize_grid_coords(grid_coords)
        
        # For 3D grid sampling, we need to reshape coordinates correctly
        # grid_sample expects (N, D_out, H_out, W_out, 3) for 3D
        N = normalized_coords.shape[0]
        
        # Reshape to (1, N, 1, 1, 3) for individual point sampling
        grid_sample_coords = normalized_coords.view(1, N, 1, 1, 3)
        
        # Add batch and channel dimensions to SDF grid: (1, 1, D, H, W)
        sdf_grid_batched = self.sdf_grid.unsqueeze(0).unsqueeze(0)
        
        # Sample SDF values using trilinear interpolation
        sampled_values = F.grid_sample(
            sdf_grid_batched, 
            grid_sample_coords, 
            mode='bilinear',  # 3D bilinear is trilinear
            padding_mode='border',
            align_corners=True
        )
        
        # Extract values and reshape: (1, 1, N, 1, 1) -> (N,)
        sdf_values = sampled_values.squeeze().flatten()
        
        return sdf_values
        
    def _world_to_grid_coords(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to grid indices (float)
        
        Args:
            points: World coordinates (N, 3)
            
        Returns:
            Grid coordinates (N, 3)
        """
        min_bound = torch.tensor(self.grid_bounds[0], device=self.device, dtype=torch.float32)
        
        # Convert to grid coordinates
        grid_coords = (points - min_bound) / self.resolution
        
        return grid_coords
        
    def _normalize_grid_coords(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize grid coordinates to [-1, 1] range for grid_sample
        
        Args:
            grid_coords: Grid coordinates (N, 3)
            
        Returns:
            Normalized coordinates (N, 3)
        """
        # Get grid dimensions
        grid_dims = torch.tensor(self.grid_shape, device=self.device, dtype=torch.float32)
        
        # Normalize to [0, 1] then to [-1, 1]
        normalized = 2.0 * grid_coords / (grid_dims - 1) - 1.0
        
        # Clamp to valid range
        normalized = torch.clamp(normalized, -1.0, 1.0)
        
        return normalized


class Open3DSDFManager:
    """
    Manager class for handling multiple SDF computations for individual objects
    """
    
    def __init__(self, resolution: float = 0.01, padding: float = 0.05, device: str = "cuda"):
        """
        Initialize SDF manager
        
        Args:
            resolution: Voxel grid resolution
            padding: Bounding box padding
            device: PyTorch device
        """
        self.resolution = resolution
        self.padding = padding
        self.device = device
        
        # Cache for individual object SDF computations
        self.sdf_cache = {}
        
    def get_sdf_calculator(self, obj: Object3D) -> Open3DSDF:
        """
        Get or create SDF calculator for an object
        
        Args:
            obj: Object3D instance
            
        Returns:
            Open3DSDF calculator instance
        """
        if obj.id not in self.sdf_cache:
            sdf_calc = Open3DSDF(obj, self.resolution, self.padding, self.device)
            self.sdf_cache[obj.id] = sdf_calc
            
        return self.sdf_cache[obj.id]
        
    def update_object_pose(self, obj: Object3D):
        """
        Update SDF calculator when object pose changes during optimization
        
        Args:
            obj: Object3D with updated pose
        """
        if obj.id in self.sdf_cache:
            # Update the cached object reference
            self.sdf_cache[obj.id].obj = obj
            # Regenerate SDF grid with new pose
            self.sdf_cache[obj.id].update_for_pose_change()
        
    def clear_cache(self):
        """Clear SDF cache to free memory"""
        self.sdf_cache.clear()
        
    def compute_sdf_values(self, obj: Object3D, query_points: torch.Tensor) -> torch.Tensor:
        """
        Compute SDF values for query points with respect to an object
        
        Args:
            obj: Object3D to compute SDF for
            query_points: Points to query (N, 3)
            
        Returns:
            SDF values (N,)
        """
        sdf_calc = self.get_sdf_calculator(obj)
        return sdf_calc.query(query_points)
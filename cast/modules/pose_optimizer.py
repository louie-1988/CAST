"""
PyTorch-based Pose Optimization Module

This module provides differentiable pose optimization using chamfer distance
with support for RGB colors and 6D rotation representation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
try:
    from pytorch3d.loss import chamfer_distance
except Exception as e: 
    # TODO  
    print("Pytorch3d is not installed, using custom chamfer distance")
    pass 
from typing import Optional, Tuple


class PyTorchPoseOptimizer(nn.Module):
    """PyTorch-based pose optimization using chamfer distance and 6D rotation representation"""
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"PyTorch pose optimizer using device: {self.device}")
    
    def chamfer_distance_3d(self, source_points: torch.Tensor, target_points: torch.Tensor,
                           source_normals: Optional[torch.Tensor] = None, target_normals: Optional[torch.Tensor] = None, 
                           source_colors: Optional[torch.Tensor] = None, 
                           target_colors: Optional[torch.Tensor] = None,
                           forward_weight: float = 0.5, backward_weight: float = 0.5, 
                           normal_weight: float = 0.1) -> torch.Tensor:
        """
        Compute bi-directional chamfer distance between two point clouds with optional RGB colors
        
        Args:
            source_points: Source point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            source_normals: Source normals (N, 3), optional
            target_normals: Target normals (M, 3), optional
            source_colors: Source colors (N, 3), optional
            target_colors: Target colors (M, 3), optional
            forward_weight: Weight for forward direction (source to target)
            backward_weight: Weight for backward direction (target to source)
            
        Returns:
            Chamfer distance loss
        """
        if (source_normals is not None and source_colors is not None):
            # according to pytorch3d, it's not allowed to use both in optimization
            # thus we clearup the colors 
            source_colors = None 
            target_colors = None 

        if source_colors is not None and target_colors is not None:
            source = torch.cat([source_points, source_colors], dim=1)
            target = torch.cat([target_points, target_colors], dim=1)
        else:
            source = source_points 
            target = target_points

        source = source[None]
        target = target[None]
        if source_normals is not None and target_normals is not None:
            source_normals = source_normals[None]
            target_normals = target_normals[None]

        losses, loss_normals = chamfer_distance(source, target, x_normals=source_normals, y_normals=target_normals, batch_reduction=None, point_reduction=None)
        loss_s2t, loss_t2s = losses 
        weighted_loss = forward_weight * loss_s2t.mean() + backward_weight * loss_t2s.mean()
        if loss_normals is not None:
            weighted_loss += sum([normal_weight * loss_val.mean() for loss_val in loss_normals])
        return weighted_loss

    def rotation_6d_to_matrix(self, rotation_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to rotation matrix
        Based on "On the Continuity of Rotation Representations in Neural Networks"
        
        Args:
            rotation_6d: 6D rotation representation (6,)
            
        Returns:
            Rotation matrix (3, 3)
        """
        # Reshape to two 3D vectors
        a1 = rotation_6d[:3]
        a2 = rotation_6d[3:]
        
        # Normalize first vector
        b1 = a1 / torch.norm(a1, dim=0, keepdim=True)
        
        # Gram-Schmidt orthogonalization
        b2 = a2 - torch.sum(b1 * a2, dim=0, keepdim=True) * b1
        b2 = b2 / torch.norm(b2, dim=0, keepdim=True)
        
        # Cross product for third vector
        b3 = torch.cross(b1, b2, dim=0)
        
        # Stack to form rotation matrix
        rotation_matrix = torch.stack([b1, b2, b3], dim=1)
        
        return rotation_matrix
    
    def matrix_to_rotation_6d(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to 6D rotation representation
        
        Args:
            rotation_matrix: Rotation matrix (3, 3)
            
        Returns:
            6D rotation representation (6,)
        """
        # Take first two columns of rotation matrix
        rotation_6d = rotation_matrix[:, :2].flatten()
        return rotation_6d
    
    def apply_transformation(self, points: torch.Tensor, translation: torch.Tensor, 
                           rotation_matrix: torch.Tensor, scale: torch.Tensor, normals: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply transformation (rotation, scale, translation) to points
        
        Args:
            points: Input points (N, 3)
            translation: Translation vector (3,)
            rotation_matrix: Rotation matrix (3, 3)
            scale: Scale factors (3,) or scalar
            normals: Normal vectors (N, 3), optional
            
        Returns:
            Transformed points (N, 3)
            Transformed normals (N, 3), optional
        """
        # Apply scale
        if scale.dim() == 0:  # scalar
            scaled_points = points * scale
        else:  # vector
            scaled_points = points * scale.unsqueeze(0)
        
        # Apply rotation
        rotated_points = torch.matmul(scaled_points, rotation_matrix.T)
        rotated_normals =  torch.matmul(normals, rotation_matrix.T) if normals is not None else None
        
        # Apply translation
        transformed_points = rotated_points + translation.unsqueeze(0)

        return transformed_points, rotated_normals
    
    def optimize_pose(self, source_points: torch.Tensor, target_points: torch.Tensor,
                      source_normals: Optional[torch.Tensor] = None, target_normals: Optional[torch.Tensor] = None,
                     source_colors: Optional[torch.Tensor] = None, 
                     target_colors: Optional[torch.Tensor] = None,
                     initial_translation: Optional[torch.Tensor] = None,
                     initial_rotation: Optional[torch.Tensor] = None,
                     initial_scale: Optional[torch.Tensor] = None,
                     xyz_weight: float = 1.0, color_weight: float = 0.1,
                     forward_weight: float = 1.0, backward_weight: float = 1.0,
                     learning_rate: float = 0.01, num_iterations: int = 1000,
                     verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Optimize pose parameters using chamfer distance with 6D rotation representation
        
        Args:
            source_points: Source point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            source_normals: Source normals (N, 3), optional
            target_normals: Target normals (M, 3), optional
            source_colors: Source colors (N, 3), optional
            target_colors: Target colors (M, 3), optional
            initial_translation: Initial translation (3,), optional
            initial_rotation: Initial rotation matrix (3, 3), optional
            initial_scale: Initial scale (3,) or scalar, optional
            xyz_weight: Weight for XYZ distance component
            color_weight: Weight for RGB color component
            forward_weight: Weight for forward direction
            backward_weight: Weight for backward direction
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple of (optimized_translation, optimized_rotation_matrix, optimized_scale, final_loss)
        """
        # Initialize translation parameter
        if initial_translation is None:
            translation = torch.zeros(3, device=self.device, requires_grad=True)
        else:
            translation = initial_translation.clone().detach().requires_grad_(True)
        
        # Initialize 6D rotation parameter
        if initial_rotation is None:
            # Initialize as identity rotation in 6D representation
            rotation_6d = torch.tensor([1., 0., 0., 0., 1., 0.], device=self.device, requires_grad=True)
        else:
            rotation_6d = self.matrix_to_rotation_6d(initial_rotation).requires_grad_(True)
        
        # Initialize scale parameter
        if initial_scale is None:
            scale = torch.ones(3, device=self.device, requires_grad=True)
        else:
            if initial_scale.dim() == 0:
                scale = torch.tensor([initial_scale, initial_scale, initial_scale], 
                                   device=self.device, requires_grad=True)
            else:
                scale = initial_scale.clone().detach().requires_grad_(True)
        
        # Setup optimizer
        optimizer = optim.Adam([translation, rotation_6d, scale], lr=learning_rate)
        
        best_loss = float('inf')
        best_translation = translation.clone()
        best_rotation_6d = rotation_6d.clone()
        best_scale = scale.clone()
        
        if verbose:
            print(f"  Starting optimization with {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Convert 6D rotation to rotation matrix
            rotation_matrix = self.rotation_6d_to_matrix(rotation_6d)
            
            # Transform source points
            transformed_source, transformed_source_normals = self.apply_transformation(source_points, translation, rotation_matrix, scale, normals=source_normals)
            
            # Compute chamfer distance
            loss = self.chamfer_distance_3d(
                transformed_source, target_points,
                transformed_source_normals, target_normals,
                source_colors, target_colors,
                forward_weight, backward_weight
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([translation, rotation_6d, scale], max_norm=1.0)
            
            optimizer.step()
            
            # Track best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_translation = translation.clone().detach()
                best_rotation_6d = rotation_6d.clone().detach()
                best_scale = scale.clone().detach()
            
            # Print progress
            if verbose and (iteration % (num_iterations // 10) == 0 or iteration == num_iterations - 1):
                print(f"    Iteration {iteration:4d}/{num_iterations}: Loss = {loss.item():.6f}", end='\r')
        
        # Convert final 6D rotation to rotation matrix
        final_rotation = self.rotation_6d_to_matrix(best_rotation_6d)
        
        if verbose:
            print(f"  Optimization complete. Best loss: {best_loss:.6f}")
        
        return best_translation, final_rotation, best_scale, best_loss
    
    def register_point_clouds(self, mesh_points: np.ndarray, scene_points: np.ndarray,
                              mesh_normals:Optional[np.ndarray] = None,  scene_normals: Optional[np.ndarray] = None, 
                             mesh_colors: Optional[np.ndarray] = None, scene_colors: Optional[np.ndarray] = None,
                             use_colors: bool = True, xyz_weight: float = 1.0, color_weight: float = 0.1,
                             forward_weight: float = 1.0, backward_weight: float = 1.0,
                             learning_rate: float = 0.01, num_iterations: int = 1000,
                             initial_transform: Optional[np.ndarray] = None,
                             verbose: bool = True) -> Tuple[np.ndarray, float, bool]:
        """
        Register mesh points to scene points using PyTorch optimization
        
        Args:
            mesh_points: Point cloud sampled from mesh (N, 3)
            scene_points: Point cloud from scene (M, 3)
            mesh_normals: Normal vectors for mesh points (N, 3), optional
            scene_normals: Normal vectors for scene points (M, 3), optional
            mesh_colors: RGB colors for mesh points (N, 3), optional
            scene_colors: RGB colors for scene points (M, 3), optional
            use_colors: Whether to include RGB colors in distance calculation
            xyz_weight: Weight for XYZ distance component
            color_weight: Weight for RGB color component
            forward_weight: Weight for forward direction (source to target)
            backward_weight: Weight for backward direction (target to source)
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            initial_transform: Initial transformation matrix (4, 4), optional
            verbose: Whether to print progress
            
        Returns:
            Tuple of (transformation_matrix, final_loss, success)
        """
        if len(mesh_points) == 0 or len(scene_points) == 0:
            if verbose:
                print("Warning: Empty point clouds for registration")
            return np.eye(4), float('inf'), False
        
        if verbose:
            print(f"Starting PyTorch registration: {len(mesh_points)} mesh → {len(scene_points)} scene points")
        
        # Convert to PyTorch tensors
        source_points = torch.tensor(mesh_points, dtype=torch.float32, device=self.device)
        target_points = torch.tensor(scene_points, dtype=torch.float32, device=self.device)

        source_normals = None
        target_normals = None
        if mesh_normals is not None:
            source_normals = torch.tensor(mesh_normals, dtype=torch.float32, device=self.device)
        if scene_normals is not None:
            target_normals = torch.tensor(scene_normals, dtype=torch.float32, device=self.device)
        
        source_colors = None
        target_colors = None
        if use_colors and scene_colors is not None:
            source_colors = torch.tensor(mesh_colors, dtype=torch.float32, device=self.device)
            target_colors = torch.tensor(scene_colors, dtype=torch.float32, device=self.device)
        
        # Parse initial transformation if provided
        initial_translation = None
        initial_rotation = None
        initial_scale = None
        
        if initial_transform is not None:
            # Decompose transformation matrix
            translation = initial_transform[:3, 3]
            upper_left = initial_transform[:3, :3]
            
            # SVD decomposition to extract rotation and scale
            U, S, Vt = np.linalg.svd(upper_left)
            rotation_matrix = U @ Vt
            
            # Ensure proper rotation (det = 1)
            if np.linalg.det(rotation_matrix) < 0:
                U[:, -1] *= -1
                rotation_matrix = U @ Vt
            
            scale = S
            
            initial_translation = torch.tensor(translation, dtype=torch.float32, device=self.device)
            initial_rotation = torch.tensor(rotation_matrix, dtype=torch.float32, device=self.device)
            initial_scale = torch.tensor(scale, dtype=torch.float32, device=self.device)
        
        try:
            # Optimize pose using PyTorch
            final_translation, final_rotation, final_scale, final_loss = self.optimize_pose(
                source_points, target_points,
                source_normals, target_normals,
                source_colors, target_colors,
                initial_translation, initial_rotation, initial_scale,
                xyz_weight, color_weight,
                forward_weight, backward_weight,
                learning_rate, num_iterations,
                verbose
            )
            
            # Convert back to numpy and construct transformation matrix
            translation_np = final_translation.cpu().numpy()
            rotation_np = final_rotation.cpu().numpy()
            scale_np = final_scale.cpu().numpy()
            
            # Construct transformation matrix
            transformation = np.eye(4)
            transformation[:3, :3] = rotation_np * scale_np.reshape(3, 1)
            transformation[:3, 3] = translation_np
            
            # Calculate success based on loss improvement
            success = final_loss < 0.1  # Threshold can be adjusted
            
            if verbose:
                print(f"Registration complete. Final loss: {final_loss:.6f}")
            
            return transformation, min(1.0, max(0.0, 1 - final_loss)), success
            
        except Exception as e:
            if verbose:
                import traceback
                traceback.print_exc()
                print(f"Error during PyTorch registration: {e}")
            return np.eye(4), float('inf'), False


def create_pose_optimizer(device: str = "auto") -> PyTorchPoseOptimizer:
    """
    Factory function to create a PyTorch pose optimizer
    
    Args:
        device: Device to use ("auto", "cuda", "cpu")
        
    Returns:
        Initialized PyTorchPoseOptimizer
    """
    return PyTorchPoseOptimizer(device=device)
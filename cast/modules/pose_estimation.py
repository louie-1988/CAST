"""
Pose Estimation Module using ICP Registration

This module handles 6D pose estimation by registering generated 3D meshes
to the scene point cloud using robust ICP algorithms.
"""
import numpy as np
import open3d as o3d

from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
import json
from scipy.spatial.transform import Rotation
from typing import Literal
from ..core.common import (DetectedObject, Mesh3D, MeshPose, Object3D, 
                                   DepthEstimation)
from ..config.settings import config
from .render_compare import RenderCompareModule
from .pose_optimizer import create_pose_optimizer

class PoseEstimationModule:
    """Module for 6D pose estimation using ICP registration"""
    
    def __init__(self, backend: Literal["icp", "pytorch"] = "pytorch", enable_render_and_compare: bool = False, debug: bool = False):
        self.icp_threshold = config.processing.icp_fitness_threshold
        self.max_iterations = config.processing.icp_max_iterations
        self.tolerance = config.processing.icp_tolerance
        self.backend = backend
        self.enable_render_and_compare = enable_render_and_compare
        
        # Initialize render-and-compare module
        self.render_compare = RenderCompareModule()
        self.debug = debug

    def sample_point_cloud_from_mesh(self, mesh: Mesh3D, 
                                   num_points: int = 50000, 
                                   sample_colors: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Sample point cloud from mesh surface with improved sampling
        
        Args:
            mesh: 3D mesh to sample from
            num_points: Number of points to sample (increased from 10000)
            sample_colors: Whether to sample RGB colors from mesh textures
            
        Returns:
            Tuple of (sampled_points, sampled_normals, sampled_colors) where colors is None if sample_colors=False
        """
        try:
            # Create Open3D mesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            
            # Add vertex colors if sampling colors and mesh has textures
            if sample_colors:
                if mesh.textures is not None and len(mesh.textures) > 0:
                    # Assume textures are vertex colors (RGB values per vertex)
                    if mesh.textures.shape[0] == len(mesh.vertices):
                        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.textures)
                    else:
                        # If textures don't match vertices, use default colors
                        default_color = np.array([0.5, 0.5, 0.5])  # Gray
                        colors = np.tile(default_color, (len(mesh.vertices), 1))
                        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                else:
                    # Use default colors if no textures
                    default_color = np.array([0.5, 0.5, 0.5])  # Gray
                    colors = np.tile(default_color, (len(mesh.vertices), 1))
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # Ensure mesh is valid
            o3d_mesh.remove_degenerate_triangles()
            o3d_mesh.remove_duplicated_triangles()
            o3d_mesh.remove_duplicated_vertices()
            o3d_mesh.remove_non_manifold_edges()
            
            # Use Poisson disk sampling for better distribution
            try:
                pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=num_points, use_triangle_normal=True)
                if len(pcd.points) < num_points // 2:
                    # Fall back to uniform sampling if Poisson doesn't generate enough points
                    pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_points, use_triangle_normal=True)
            except:
                # Fallback to uniform sampling
                pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_points, use_triangle_normal=True)
            
            # Extract points, normals, and optionally colors
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)
            colors = np.asarray(pcd.colors) if sample_colors and pcd.has_colors() else None
            
            return points, normals, colors
            
        except Exception as e:
            print(f"Error sampling point cloud from mesh: {e}")
            return np.array([]), np.array([]), None
    

    
    def extract_object_point_cloud_from_instance(self, depth_estimation: DepthEstimation,
                                                image: np.ndarray, 
                                                detected_object: DetectedObject,
                                                extract_colors: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract object point cloud using the depth module's instance extraction method
        
        Args:
            depth_estimation: DepthEstimation object with extract methods
            image: Original RGB image
            detected_object: Object with mask information
            extract_colors: Whether to extract RGB colors from the image
            
        Returns:
            Tuple of (object_points, object_normals, object_colors) in OpenGL coordinates
            Colors is None if extract_colors=False
        """
        if detected_object.mask is None:
            print(f"Warning: No mask for object {detected_object.id}")
            return np.array([]), np.array([]), None
        
        # Use the depth module's extract_object_point_cloud method with OpenGL coordinates
        from ..modules.depth_estimation import DepthEstimationModule
        depth_module = DepthEstimationModule()
        
        result = depth_module.extract_object_point_cloud(
            depth_estimation, image, detected_object.mask, use_opengl_coords=True
        )
        
        # Handle both single return (points only) and tuple return (points, normals)
        if isinstance(result, tuple):
            object_points, object_normals = result
        else:
            object_points = result
            object_normals = None
        
        # Extract colors if requested
        object_colors = None
        if extract_colors and len(object_points) > 0:
            object_mask = detected_object.mask
            if object_mask.dtype != bool:
                object_mask = object_mask > 127
            
            # Extract corresponding RGB colors from the original image
            mask_colors = image[object_mask] / 255.0  # Normalize to [0, 1]
            
            # Remove invalid points (same indices as object_points)
            points_3d = depth_estimation.point_cloud
            masked_points = points_3d[object_mask]
            valid_indices = ~np.any(np.isnan(masked_points) | (masked_points == 0), axis=1)
            object_colors = mask_colors[valid_indices]
        
        print(f"Extracted {len(object_points)} points for object {detected_object.id} (OpenGL coords)")
        return object_points, object_normals, object_colors
    

    
    def preprocess_point_cloud(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                             voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
        """
        Improved preprocessing of point cloud for robust ICP registration
        
        Args:
            points: Point cloud as numpy array (should already be in OpenGL coordinates)
            voxel_size: Voxel size for downsampling (reduced from 0.02 to 0.015)
            
        Returns:
            Preprocessed Open3D point cloud with normals
        """
        if len(points) == 0:
            return o3d.geometry.PointCloud()
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Remove outliers with improved parameters
        # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.2)
        pcd, _ = pcd.remove_radius_outlier(nb_points=25, radius=0.05)
        
        # Estimate normals with better parameters based on Open3D recommendations
        # radius_normal = voxel_size * 3  # Increased radius for better normal estimation
        # pcd.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #         radius=radius_normal, 
        #         max_nn=50  # Increased neighbors for more robust normal estimation
        #     )
        # )
        
        # Orient normals consistently
        # pcd.orient_normals_consistent_tangent_plane(k=15)
        
        return pcd
    
    def normalize_point_cloud(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """
        Normalize point cloud to [-1, 1] range and return normalization info
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Tuple of (normalized_pcd, normalization_info)
        """
        if len(pcd.points) == 0:
            return pcd, {"center": np.zeros(3), "scale": 1.0}
        
        points = np.asarray(pcd.points)
        
        # Compute bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Compute center and scale
        # center = (min_coords + max_coords) / 2.0
        center = points.mean(axis=0)
        extent = max_coords - min_coords
        # scale = np.max(extent) / 2.0  # Scale to fit in [-1, 1]
        scale = np.max(extent[:2]) / 2.0
        
        if scale < 1e-8:  # Avoid division by zero
            scale = 1.0
        
        # Normalize points
        normalized_points = (points - center) / scale
        
        # Create normalized point cloud
        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)
        
        # Copy normals if they exist
        if pcd.has_normals():
            normalized_pcd.normals = pcd.normals
        
        # Copy colors if they exist
        if pcd.has_colors():
            normalized_pcd.colors = pcd.colors
        
        normalization_info = {
            "center": center,
            "scale": scale,
            "min_coords": min_coords,
            "max_coords": max_coords
        }
        
        return normalized_pcd, normalization_info
    
    def denormalize_transformation(self, transformation: np.ndarray, source_norm_info: Dict,
                                  target_norm_info: Dict) -> np.ndarray:
        """
        Transform the normalized transformation back to original coordinate system
        
        Args:
            transformation: Transformation matrix from normalized ICP
            target_norm_info: Normalization info for target point cloud, with keys scale 
            
        Returns:
            Transformation matrix in original coordinate system with proper scale handling
        """
        # the transformation to map original source target to the normalised one 
        source_center, source_scale = source_norm_info['center'], source_norm_info['scale']
        norm_transform = np.eye(4)
        norm_transform[:3, :3] = np.diag(np.ones(3) / (source_scale + 1e-8))
        norm_transform[:3, 3] = -source_center / (source_scale + 1e-8)

        # the transformation to map normalised target to original pose 
        target_center, target_scale = target_norm_info['center'], target_norm_info['scale']
        denorm_transform = np.eye(4)
        denorm_transform[:3, :3] = np.diag(np.ones(3) * target_scale)
        denorm_transform[:3, 3] = target_center
        
        # left multiply: first apply ICP transformation, and the de-norm
        final_transform = denorm_transform @ transformation @ norm_transform
        return final_transform
    
    def _decompose_transformation_matrix(self, transformation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose 4x4 transformation matrix into translation, rotation, and scale
        
        Args:
            transformation: 4x4 transformation matrix
            
        Returns:
            Tuple of (translation, rotation_matrix, scale)
        """
        # Extract translation
        translation = transformation[:3, 3]
        
        # Extract upper-left 3x3 matrix
        upper_left = transformation[:3, :3]
        
        # Decompose into rotation and scale using SVD
        U, S, Vt = np.linalg.svd(upper_left)
        
        # Rotation matrix (ensure proper rotation)
        rotation_matrix = U @ Vt
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(rotation_matrix) < 0:
            U[:, -1] *= -1
            rotation_matrix = U @ Vt
        
        # Scale factors are the singular values
        scale = S
        
        return translation, rotation_matrix, scale
    
    def global_registration(self, source: o3d.geometry.PointCloud, 
                          target: o3d.geometry.PointCloud,
                          voxel_size: float = 0.01, 
                          use_ransac: bool = True) -> o3d.pipelines.registration.RegistrationResult:
        """
        Perform global registration using RANSAC
        
        Args:
            source: Source point cloud (mesh)
            target: Target point cloud (scene)
            voxel_size: Voxel size for feature computation
            
        Returns:
            Registration result
        """
        # First downsample the point cloud to get some more global descriptive features 
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        # Compute FPFH features
        radius_feature = voxel_size * 5
        
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        
        # RANSAC-based global registration
        if use_ransac:
            print(f":: Apply RANSAC-based global registration (voxel size: {voxel_size}), source points: {len(source_down.points)}, target points: {len(target_down.points)}")
            distance_threshold = voxel_size * 1.5
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.85)
            )
        else:
            distance_threshold = voxel_size * 0.5
            print(":: Apply fast global registration with distance threshold %.3f" \
                    % distance_threshold)
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold))
        
        return result
    
    def local_registration(self, source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud,
                          initial_transform: np.ndarray,
                          voxel_size: float = 0.05) -> o3d.pipelines.registration.RegistrationResult:
        """
        Perform local ICP registration
        
        Args:
            source: Source point cloud (mesh)
            target: Target point cloud (scene)
            initial_transform: Initial transformation matrix
            voxel_size: Voxel size for distance threshold
            
        Returns:
            Registration result
        """
        distance_threshold = voxel_size * 0.4
        
        # Point-to-plane ICP
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.tolerance,
                relative_rmse=self.tolerance,
                max_iteration=self.max_iterations
            )
        )
        
        return result
    
    def robust_local_registration(self, source: o3d.geometry.PointCloud,
                                 target: o3d.geometry.PointCloud,
                                 initial_transform: np.ndarray,
                                 use_robust_kernel: bool = True,
                                 voxel_size: float = 0.02) -> o3d.pipelines.registration.RegistrationResult:
        """
        Perform robust local ICP registration with improved parameters and robust kernel
        
        Args:
            source: Source point cloud (mesh)
            target: Target point cloud (scene)
            initial_transform: Initial transformation matrix
            use_robust_kernel: Whether to use robust kernel
            voxel_size: Voxel size for distance threshold
            
        Returns:
            Registration result
        """
        # Use tighter distance threshold for better precision
        mu, sigma = 0., 0.1
        distance_threshold = voxel_size * 10
        
        if use_robust_kernel:
            # Create robust kernel to handle outliers (Tukey loss)
            robust_kernel = o3d.pipelines.registration.TukeyLoss(k=0.1)
            
            # Point-to-plane ICP with robust kernel and improved parameters
            result = o3d.pipelines.registration.registration_icp(
                source, target, distance_threshold, initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(robust_kernel),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,  # Tighter convergence criteria
                    relative_rmse=1e-6,
                    max_iteration=100  
                )
            )
        else:
            result = o3d.pipelines.registration.registration_icp(
                source, target, distance_threshold, initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,  # Tighter convergence criteria
                    relative_rmse=1e-6,
                    max_iteration=100  
                )
            )
        
        return result

    def _vis_pcd(self, source_pcd: Union[np.ndarray, o3d.geometry.PointCloud], target_pcd: Union[np.ndarray, o3d.geometry.PointCloud], transformation: np.ndarray) -> None:
        """
        Visualize point clouds
        """
        if isinstance(source_pcd, np.ndarray):
            source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pcd))
        if isinstance(target_pcd, np.ndarray):
            target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pcd))
        source_pcd.paint_uniform_color([1, 0, 0])
        target_pcd.paint_uniform_color([0, 1, 0])
        transformed_pcd = source_pcd.transform(transformation)
        transformed_pcd.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([target_pcd, transformed_pcd])
    


    def robust_icp_registration(self, mesh_points: np.ndarray, mesh_normals: np.ndarray,
                              scene_points: np.ndarray, scene_normals: Optional[np.ndarray] = None,
                              voxel_size: float = 0.01) -> Tuple[np.ndarray, float, bool]:
        """
        Perform robust ICP registration with point cloud normalization
        
        Args:
            mesh_points: Point cloud sampled from mesh
            mesh_normals: Normal vectors for mesh points
            scene_points: Point cloud from scene
            scene_normals: Normal vectors for scene points (optional)
            voxel_size: Voxel size for preprocessing (default 0.01 for finer detail)
            
        Returns:
            Tuple of (transformation_matrix, fitness_score, global_registration_success)
        """
        if len(mesh_points) == 0 or len(scene_points) == 0:
            print("Warning: Empty point clouds for registration")
            return np.eye(4), 0.0, False
        
        # Preprocess point clouds (mesh points don't need OpenCV->OpenGL conversion as it's done in depth-estimation module)
        source_pcd = self.preprocess_point_cloud(mesh_points, mesh_normals, voxel_size)
        target_pcd = self.preprocess_point_cloud(scene_points, scene_normals, voxel_size)
        if self.debug:
            self._vis_pcd(source_pcd, target_pcd, np.eye(4))
        print(f"Original point clouds - Source: {len(source_pcd.points)} points, Target: {len(target_pcd.points)} points")
        
        if len(source_pcd.points) < 10 or len(target_pcd.points) < 10:
            print("Warning: Insufficient points after preprocessing")
            return np.eye(4), 0.0, False
        
        try:
            # Step 1: Normalize both point clouds to [-1, 1] independently
            print("  Normalizing point clouds...")
            source_norm, source_norm_info = self.normalize_point_cloud(source_pcd)
            target_norm, target_norm_info = self.normalize_point_cloud(target_pcd)
            if self.debug:
                self._vis_pcd(source_norm, target_norm, np.eye(4))
            print(f"  Source normalization - Center: {source_norm_info['center']}, Scale: {source_norm_info['scale']:.3f}")
            print(f"  Target normalization - Center: {target_norm_info['center']}, Scale: {target_norm_info['scale']:.3f}")

            if self.debug:
                import os 
                os.makedirs("output/debug", exist_ok=True)
                import time 
                o3d.io.write_point_cloud(f"output/debug/{int(time.time())}_source_norm.ply", source_norm)
                o3d.io.write_point_cloud(f"output/debug/{int(time.time())}_target_norm.ply", target_norm)
            
            # Step 2: Global registration on normalized point clouds
            print("  Running global registration on normalized point clouds...")
            global_result = self.global_registration(source_norm, target_norm, voxel_size)

            # Use global registration result as initial transform
            global_success = global_result.fitness >= 0.1
            
            if not global_success:
                print(f"  Global registration failed with fitness {global_result.fitness:.3f}, using identity transform")
                initial_transform = np.eye(4)
            else:
                initial_transform = global_result.transformation
                print(f"  Global registration fitness: {global_result.fitness:.3f}")
            
            if self.debug:
                self._vis_pcd(source_norm, target_norm, initial_transform)
            
            # Step 3: Robust local ICP refinement on normalized point clouds
            print("  Running robust local ICP refinement on normalized point clouds...")
            local_result = self.robust_local_registration(
                source_norm, target_norm, initial_transform, voxel_size
            )
            
            print(f"  Local ICP fitness: {local_result.fitness:.3f}, RMSE: {local_result.inlier_rmse:.3f}")
            if self.debug:
                self._vis_pcd(source_norm, target_norm, local_result.transformation)
            
            # Step 4: Transform the result back to original coordinate system
            print("  Transforming result back to original coordinates...")
            final_transformation = self.denormalize_transformation(
                local_result.transformation, source_norm_info, target_norm_info
            )
            
            return final_transformation, local_result.fitness, global_success
            
        except Exception as e:
            import traceback 
            traceback.print_exc()
            print(f"Error during ICP registration: {e}")
            return np.eye(4), 0.0, False
    
    def pytorch_registration(self, mesh_points: np.ndarray, mesh_normals: np.ndarray, mesh_colors: np.ndarray,
                           scene_points: np.ndarray, scene_normals: Optional[np.ndarray] = None, 
                           scene_colors: Optional[np.ndarray] = None,
                           use_colors: bool = True, xyz_weight: float = 1.0, color_weight: float = 0.1,
                           forward_weight: float = 1.0, backward_weight: float = 1.0,
                           learning_rate: float = 0.01, num_iterations: int = 1000) -> Tuple[np.ndarray, float, bool]:
        """
        Perform PyTorch-based pose optimization using chamfer distance
        
        Args:
            mesh_points: Point cloud sampled from mesh
            mesh_normals: Normal vectors for mesh points
            mesh_colors: RGB colors for mesh points
            scene_points: Point cloud from scene
            scene_normals: Normal vectors for scene points (optional)
            scene_colors: RGB colors for scene points (optional)
            use_colors: Whether to include RGB colors in distance calculation
            xyz_weight: Weight for XYZ distance component
            color_weight: Weight for RGB color component
            forward_weight: Weight for forward direction (source to target)
            backward_weight: Weight for backward direction (target to source)
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            
        Returns:
            Tuple of (transformation_matrix, final_loss, success)
        """
        if len(mesh_points) == 0 or len(scene_points) == 0:
            print("Warning: Empty point clouds for registration")
            return np.eye(4), float('inf'), False
        
        print(f"Starting PyTorch-based registration with {len(mesh_points)} mesh points and {len(scene_points)} scene points")
        
        # Initialize PyTorch optimizer
        optimizer = create_pose_optimizer()
        
        # Perform PyTorch-based registration
        print("  Running PyTorch pose optimization...")
        transformation, final_loss, success = optimizer.register_point_clouds(
            mesh_points, scene_points,
            mesh_normals, scene_normals,
            mesh_colors, scene_colors,
            use_colors, xyz_weight, color_weight,
            forward_weight, backward_weight,
            learning_rate, num_iterations,
            initial_transform=None, verbose=True
        )
        
        print(f"  PyTorch registration complete. Final loss: {final_loss:.6f}")
        return transformation, final_loss, success
    
    def estimate_object_pose_icp(self, mesh: Mesh3D, detected_object: DetectedObject,
                           depth_estimation: DepthEstimation, image: np.ndarray = None,
                           output_dir: Optional[Path] = None) -> MeshPose:
        """
        Estimate 6D pose of an object by registering its mesh to scene point cloud
        
        Args:
            mesh: 3D mesh of the object
            detected_object: Detected object with mask
            depth_estimation: Scene depth estimation
            image: Original RGB image for instance point cloud extraction
            output_dir: Output directory for render-and-compare results
            
        Returns:
            Estimated 6D pose
        """
        print(f"Estimating pose for object {detected_object.id}: {detected_object.description}")
        
        # Step 1: Sample point cloud from mesh with more points
        mesh_points, mesh_normals, _ = self.sample_point_cloud_from_mesh(mesh, sample_colors=False)
        
        if len(mesh_points) == 0:
            print(f"Failed to sample points from mesh for object {detected_object.id}")
            return MeshPose(translation=np.zeros(3), rotation=np.eye(3), scale=np.ones(3), confidence=0.0)
        
        # Step 2: Extract object point cloud from scene using depth module's method
        scene_points, scene_normals, _ = self.extract_object_point_cloud_from_instance(
            depth_estimation, image, detected_object, extract_colors=False
        )
        
        if len(scene_points) == 0:
            print(f"Failed to extract scene points for object {detected_object.id}")
            return MeshPose(translation=np.zeros(3), rotation=np.eye(3), scale=np.ones(3), confidence=0.0)
        
        # Step 3: Perform ICP registration
        transformation, fitness, global_registration_success = self.robust_icp_registration(
            mesh_points, mesh_normals, scene_points, scene_normals
        )
        
        # Step 4: Extract pose components from transformation matrix
        translation, rotation_matrix, scale = self._decompose_transformation_matrix(transformation)
        
        # If global registration failed and we have output directory, use render-and-compare
        if not global_registration_success and self.enable_render_and_compare and output_dir is not None:
            print(f"  Global registration failed for object {detected_object.id}, using render-and-compare...")
            
            try:
                # Use render-and-compare to estimate rotation
                render_rotation = self.render_compare.estimate_rotation_render_compare(
                    mesh, detected_object, output_dir
                )
                
                # Keep the translation and scale from ICP but use rotation from render-and-compare
                rotation_matrix = render_rotation
                print(f"  Using render-and-compare rotation for object {detected_object.id}")
                
            except Exception as e:
                print(f"  Render-and-compare failed: {e}, using ICP rotation")
        elif not global_registration_success:
            rotation_matrix = np.eye(3)
        
        pose = MeshPose(
            translation=translation,
            rotation=rotation_matrix,
            scale=scale,
            confidence=fitness
        )
        
        print(f"Pose estimation complete for object {detected_object.id}. Fitness: {fitness:.3f}")
        return pose
    
    def estimate_object_pose_torch(self, mesh: Mesh3D, detected_object: DetectedObject,
                                   depth_estimation: DepthEstimation, image: np.ndarray = None,
                                   use_colors: bool = True, xyz_weight: float = 1.0, color_weight: float = 0.1,
                                   forward_weight: float = 1.0, backward_weight: float = 1.0,
                                   learning_rate: float = 0.01, num_iterations: int = 1000,
                                   output_dir: Optional[Path] = None) -> MeshPose:
        """
        Estimate 6D pose of an object using PyTorch-based optimization with chamfer distance
        
        Args:
            mesh: 3D mesh of the object
            detected_object: Detected object with mask
            depth_estimation: Scene depth estimation
            image: Original RGB image for instance point cloud extraction
            use_colors: Whether to include RGB colors in distance calculation
            xyz_weight: Weight for XYZ distance component
            color_weight: Weight for RGB color component
            forward_weight: Weight for forward direction (source to target)
            backward_weight: Weight for backward direction (target to source)
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            output_dir: Output directory for render-and-compare results (fallback)
            
        Returns:
            Estimated 6D pose
        """
        print(f"Estimating pose for object {detected_object.id} using PyTorch optimization: {detected_object.description}")
        
        # Step 1: Sample point cloud from mesh with colors
        # use fewer points in torch-based optimization
        mesh_points, mesh_normals, mesh_colors = self.sample_point_cloud_from_mesh(mesh, sample_colors=use_colors, num_points=15000)
        
        if len(mesh_points) == 0:
            print(f"Failed to sample points from mesh for object {detected_object.id}")
            return MeshPose(translation=np.zeros(3), rotation=np.eye(3), scale=np.ones(3), confidence=0.0)
        
        # Step 2: Extract object point cloud from scene with colors
        scene_points, scene_normals, scene_colors = self.extract_object_point_cloud_from_instance(
            depth_estimation, image, detected_object, extract_colors=use_colors
        )
        # downsample in  the torch-based optimization
        downsample_factor = 2 
        scene_points, scene_normals, scene_colors = scene_points[::downsample_factor], scene_normals[::downsample_factor], scene_colors[::downsample_factor]
        
        if len(scene_points) == 0:
            print(f"Failed to extract scene points for object {detected_object.id}")
            return MeshPose(translation=np.zeros(3), rotation=np.eye(3), scale=np.ones(3), confidence=0.0)
        
        # Step 3: Perform PyTorch-based registration
        transformation, final_loss, success = self.pytorch_registration(
            mesh_points, mesh_normals, mesh_colors,
            scene_points, scene_normals, scene_colors,
            use_colors, xyz_weight, color_weight,
            forward_weight, backward_weight,
            learning_rate, num_iterations
        )
        if self.debug:
            self._vis_pcd(mesh_points, scene_points, transformation)
        
        # Step 4: Extract pose components from transformation matrix
        translation, rotation_matrix, scale = self._decompose_transformation_matrix(transformation)
        
        # Convert loss to confidence (lower loss = higher confidence)
        confidence = 1.0 / (1.0 + final_loss) if final_loss < float('inf') else 0.0
        
        pose = MeshPose(
            translation=translation,
            rotation=rotation_matrix,
            scale=scale,
            confidence=confidence
        )
        
        print(f"PyTorch pose estimation complete for object {detected_object.id}. Loss: {final_loss:.6f}, Confidence: {confidence:.3f}")
        return pose
    
    def run(self, meshes: List[Optional[Mesh3D]], detected_objects: List[DetectedObject],
                   depth_estimation: DepthEstimation, 
                   output_dir: Optional[Path] = None,
                   filter_low_fitness: bool = True,
                   min_fitness_threshold: float = 0.01,
                   image: np.ndarray = None,
                   use_colors: bool = True,
                   xyz_weight: float = 1.0,
                   color_weight: float = 0.1,
                   forward_weight: float = 1.0,
                   backward_weight: float = 1.0,
                   learning_rate: float = 0.01,
                   num_iterations: int = 1000) -> List[Object3D]:
        """
        Run PyTorch-based pose estimation for all objects
        
        Args:
            meshes: List of generated meshes
            detected_objects: List of detected objects
            depth_estimation: Scene depth estimation
            output_dir: Directory to save results
            filter_low_fitness: Whether to filter out objects with low fitness scores
            min_fitness_threshold: Minimum fitness threshold for filtering
            image: Original RGB image for instance point cloud extraction
            method: Method to use for pose estimation, either "icp" or "pytorch"
            use_colors: Whether to include RGB colors in distance calculation
            xyz_weight: Weight for XYZ distance component
            color_weight: Weight for RGB color component
            forward_weight: Weight for forward direction (source to target)
            backward_weight: Weight for backward direction (target to source)
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            
        Returns:
            List of 3D objects with estimated poses
        """
        print("Starting PyTorch-based pose estimation pipeline...")
        
        objects_3d = []
        
        for i, (mesh, detected_obj) in enumerate(zip(meshes, detected_objects)):
            if mesh is None:
                print(f"Skipping object {detected_obj.id} - no mesh available")
                continue
            
            print(f"Processing object {i+1}/{len(meshes)}")
            
            if self.backend == "icp":
                # Estimate pose with ICP 
                pose = self.estimate_object_pose_icp(mesh, detected_obj, depth_estimation, image, output_dir)
            else:
                # Estimate pose using PyTorch method
                pose = self.estimate_object_pose_torch(
                    mesh, detected_obj, depth_estimation, image,
                    use_colors, xyz_weight, color_weight,
                    forward_weight, backward_weight,
                    learning_rate, num_iterations, output_dir
                )
            
            # Check fitness threshold if filtering is enabled
            if filter_low_fitness and pose.confidence < min_fitness_threshold:
                print(f"Filtered out object {detected_obj.id} ({detected_obj.description}) "
                      f"due to low confidence: {pose.confidence:.3f} < {min_fitness_threshold}")
                continue
            
            # Create Object3D
            obj_3d = Object3D(
                id=detected_obj.id,
                mesh=mesh,
                pose=pose,
                detected_object=detected_obj
            )
            
            objects_3d.append(obj_3d)
        
        # Apply additional fitness-based filtering if requested
        if filter_low_fitness and objects_3d:
            original_count = len(objects_3d)
            objects_3d = self._filter_by_fitness(objects_3d, min_fitness_threshold)
            filtered_count = len(objects_3d)
            
            if filtered_count < original_count:
                print(f"Fitness filtering: {original_count} -> {filtered_count} objects "
                      f"(removed {original_count - filtered_count} low-fitness objects)")
        
        # Save results
        if output_dir:
            self._save_results(objects_3d, output_dir)
        
        print(f"PyTorch pose estimation complete. Processed {len(objects_3d)} objects.")
        return objects_3d
    
    def _filter_by_fitness(self, objects_3d: List[Object3D], 
                          min_fitness_threshold: float) -> List[Object3D]:
        """
        Filter objects based on pose estimation fitness scores
        
        Args:
            objects_3d: List of 3D objects with poses
            min_fitness_threshold: Minimum fitness threshold
            
        Returns:
            Filtered list of objects
        """
        filtered_objects = []
        
        for obj in objects_3d:
            if obj.pose.confidence >= min_fitness_threshold:
                filtered_objects.append(obj)
            else:
                print(f"Removing object {obj.id} ({obj.detected_object.description}) "
                      f"with low fitness: {obj.pose.confidence:.3f}")
        
        return filtered_objects
    
    def _save_results(self, objects_3d: List[Object3D], output_dir: Path) -> None:
        """Save pose estimation results"""
        pose_dir = output_dir / "pose_estimation"
        pose_dir.mkdir(exist_ok=True, parents=True)
        
        # Save poses as JSON
        poses_data = []
        fitness_stats = {"min": float('inf'), "max": -float('inf'), "mean": 0.0, "count": 0}
        
        for obj in objects_3d:
            fitness = obj.pose.confidence
            fitness_stats["min"] = min(fitness_stats["min"], fitness)
            fitness_stats["max"] = max(fitness_stats["max"], fitness)
            fitness_stats["count"] += 1
            
            pose_data = {
                "object_id": obj.id,
                "description": obj.detected_object.description,
                "translation": obj.pose.translation.tolist(),
                "rotation_matrix": obj.pose.rotation.tolist(),
                "scale": obj.pose.scale.tolist(),
                "confidence": obj.pose.confidence,
                "fitness_score": fitness,  # Explicit fitness field
                "euler_angles": self._rotation_matrix_to_euler(obj.pose.rotation).tolist()
            }
            poses_data.append(pose_data)
        
        # Calculate mean fitness
        if fitness_stats["count"] > 0:
            fitness_stats["mean"] = sum(obj.pose.confidence for obj in objects_3d) / fitness_stats["count"]
            if fitness_stats["min"] == float('inf'):
                fitness_stats["min"] = 0.0
        else:
            fitness_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
        
        # Save complete results with fitness statistics
        results_data = {
            "poses": poses_data,
            "fitness_statistics": fitness_stats,
            "num_objects": len(objects_3d)
        }
        
        with open(pose_dir / "estimated_poses.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save individual pose files
        for obj in objects_3d:
            obj_dir = pose_dir / f"object_{obj.id}"
            obj_dir.mkdir(exist_ok=True)
            
            # Save transformation matrix with scale
            transform_matrix = np.eye(4)
            # Apply scale to rotation matrix
            scaled_rotation = obj.pose.rotation * obj.pose.scale
            transform_matrix[:3, :3] = scaled_rotation
            transform_matrix[:3, 3] = obj.pose.translation
            np.save(obj_dir / "transformation_matrix.npy", transform_matrix)
            
            # Save pose components
            np.save(obj_dir / "translation.npy", obj.pose.translation)
            np.save(obj_dir / "rotation_matrix.npy", obj.pose.rotation)
            np.save(obj_dir / "scale.npy", obj.pose.scale)
            
            # Save fitness score
            with open(obj_dir / "fitness_score.txt", "w") as f:
                f.write(f"{obj.pose.confidence:.6f}\n")
        
        print(f"Saved pose estimation results for {len(objects_3d)} objects")
        print(f"Fitness statistics - Min: {fitness_stats['min']:.3f}, "
              f"Max: {fitness_stats['max']:.3f}, Mean: {fitness_stats['mean']:.3f}")
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (XYZ order)"""
        try:
            r = Rotation.from_matrix(rotation_matrix)
            return r.as_euler('xyz', degrees=True)
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    
    def visualize_registration(self, mesh_points: np.ndarray, scene_points: np.ndarray,
                             transformation: np.ndarray) -> None:
        """
        Visualize ICP registration result
        
        Args:
            mesh_points: Original mesh points
            scene_points: Scene points
            transformation: Estimated transformation
        """
        try:
            # Create point clouds
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(mesh_points)
            source_pcd.paint_uniform_color([1, 0, 0])  # Red for original mesh
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(scene_points)
            target_pcd.paint_uniform_color([0, 1, 0])  # Green for scene
            
            # Transform source
            transformed_pcd = source_pcd.transform(transformation)
            transformed_pcd.paint_uniform_color([0, 0, 1])  # Blue for transformed mesh
            
            # Visualize
            print("Visualizing registration result...")
            print("Red: Original mesh, Green: Scene points, Blue: Registered mesh")
            o3d.visualization.draw_geometries([target_pcd, transformed_pcd])
            
        except Exception as e:
            print(f"Error visualizing registration: {e}")

"""
Depth Estimation Module using MoGe

This module handles metric-level depth estimation and point cloud generation
using the MoGe (Metric-Level Depth Estimation) model.
"""
import numpy as np
import torch
import cv2
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d

from ..core.common import DepthEstimation
from ..utils.image_utils import normalize_image, save_image
from ..config.settings import config

class DepthEstimationModule:
    """Module for depth estimation using MoGe"""
    
    def __init__(self, model_name: str = "Ruicheng/moge-2-vitl-normal"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_model(self) -> bool:
        """Load the MoGe model"""
        try:
            print(f"Loading MoGe model: {self.model_name}")
            
            # Import MoGe model
            try:
                from moge.model.v2 import MoGeModel
            except ImportError:
                print("Error: MoGe not installed. Please install with:")
                print("pip install git+https://github.com/microsoft/MoGe.git")
                return False
            
            # Load model from HuggingFace
            self.model = MoGeModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            print("MoGe model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading MoGe model: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for MoGe inference
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            
        Returns:
            Preprocessed image tensor (3, H, W)
        """
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and move to device
        image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)
        
        # Change from (H, W, 3) to (3, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return image_tensor
    
    def estimate_depth(self, image: np.ndarray) -> DepthEstimation:
        """
        Estimate depth and generate point cloud from RGB image
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            DepthEstimation object with depth map, point cloud, and other outputs
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load MoGe model")
        
        print("Estimating depth with MoGe...")
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model.infer(input_tensor, use_fp16=False)
        
        # Extract outputs
        depth_map = output["depth"].cpu().numpy()
        point_cloud = output["points"].cpu().numpy()  # (H, W, 3)
        
        # Optional outputs
        normal_map = output.get("normal")
        if normal_map is not None:
            normal_map = normal_map.cpu().numpy()
        
        mask = output.get("mask")
        if mask is not None:
            mask = mask.cpu().numpy().astype(np.uint8) * 255
        
        intrinsics = output.get("intrinsics")
        if intrinsics is not None:
            intrinsics = intrinsics.cpu().numpy()
        
        # Create DepthEstimation object
        depth_estimation = DepthEstimation(
            depth_map=depth_map,
            point_cloud=point_cloud,
            normal_map=normal_map,
            mask=mask,
            intrinsics=intrinsics
        )
        
        print(f"Depth estimation complete. Point cloud shape: {point_cloud.shape}")
        return depth_estimation
    
    def opencv_to_opengl_coordinates(self, points: np.ndarray) -> np.ndarray:
        """
        Convert point cloud from OpenCV coordinate frame to OpenGL coordinate frame
        OpenCV: +X right, +Y down, +Z forward (into scene)
        OpenGL: +X right, +Y up, +Z backward (out of scene)
        
        Args:
            points: Point cloud in OpenCV coordinates (N, 3)
            
        Returns:
            Point cloud in OpenGL coordinates (N, 3)
        """
        if len(points) == 0:
            return points
        
        # Convert OpenCV to OpenGL: flip Y and Z axes
        opengl_points = points.copy()
        opengl_points[:, 1] = -opengl_points[:, 1]  # Flip Y axis
        opengl_points[:, 2] = -opengl_points[:, 2]  # Flip Z axis
        
        return opengl_points

    def create_point_cloud_o3d(self, depth_estimation: DepthEstimation,
                              image: np.ndarray, use_opengl_coords: bool = True) -> o3d.geometry.PointCloud:
        """
        Create Open3D point cloud from depth estimation results
        
        Args:
            depth_estimation: Depth estimation results
            image: Original RGB image for coloring
            
        Returns:
            Open3D point cloud object
        """
        # Get point coordinates (H, W, 3) -> (N, 3)
        points_3d = depth_estimation.point_cloud
        h, w = points_3d.shape[:2]
        
        # Reshape to (N, 3)
        points = points_3d.reshape(-1, 3)
        
        # Get colors from image
        colors = image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
        
        # Filter valid points if mask is available
        if depth_estimation.mask is not None:
            valid_mask = depth_estimation.mask.reshape(-1) > 0
            points = points[valid_mask]
            colors = colors[valid_mask]
        
        # Convert to OpenGL coordinates if requested
        if use_opengl_coords:
            points = self.opencv_to_opengl_coordinates(points)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def extract_object_point_cloud(self, depth_estimation: DepthEstimation,
                                  image: np.ndarray, object_mask: np.ndarray, 
                                  use_opengl_coords: bool = True) -> np.ndarray:
        """
        Extract point cloud for a specific object using its mask
        
        Args:
            depth_estimation: Full scene depth estimation
            image: Original RGB image
            object_mask: Binary mask of the object
            
        Returns:
            Object point cloud as numpy array (N, 3)
        """
        # Get point coordinates
        points_3d = depth_estimation.point_cloud
        normals_3d = depth_estimation.normal_map
        
        # Apply object mask
        if object_mask.dtype != bool:
            object_mask = object_mask > 127
        
        # Extract points within the object mask
        object_points = points_3d[object_mask]
        object_normals = normals_3d[object_mask]
        
        # Remove invalid points (zeros or NaN)
        valid_indices = ~np.any(np.isnan(object_points) | (object_points == 0), axis=1)
        object_points = object_points[valid_indices]
        object_normals = object_normals[valid_indices]

        # Convert to OpenGL coordinates if requested
        if use_opengl_coords:
            object_points = self.opencv_to_opengl_coordinates(object_points)
            object_normals = self.opencv_to_opengl_coordinates(object_normals)
        
        return object_points, object_normals
    
    def extract_instance_point_clouds(self, depth_estimation: DepthEstimation,
                                     image: np.ndarray, 
                                     detected_objects: List) -> Dict[int, Dict[str, Any]]:
        """
        Extract individual point clouds for each detected object
        
        Args:
            depth_estimation: Full scene depth estimation
            image: Original RGB image
            detected_objects: List of detected objects with masks
            
        Returns:
            Dictionary mapping object IDs to their point cloud data
        """
        instance_point_clouds = {}
        
        for obj in detected_objects:
            if obj.mask is None:
                print(f"Warning: No mask for object {obj.id}, skipping point cloud extraction")
                continue
            
            # Extract object point cloud (with OpenGL coordinates)
            object_points, object_normals = self.extract_object_point_cloud(depth_estimation, image, obj.mask, use_opengl_coords=True)
            
            if len(object_points) == 0:
                print(f"Warning: No valid points for object {obj.id}")
                continue
            
            # Extract corresponding colors from the image
            mask = obj.mask > 127 if obj.mask.dtype != bool else obj.mask
            object_colors = image[mask] / 255.0  # Normalize to [0, 1]
            
            # Filter colors to match valid points
            if len(object_colors) > len(object_points):
                # Get valid indices again to align colors with points
                points_3d = depth_estimation.point_cloud
                all_object_points = points_3d[mask]
                valid_indices = ~np.any(np.isnan(all_object_points) | (all_object_points == 0), axis=1)
                object_colors = object_colors[valid_indices]
            
            # Ensure colors and points have same length
            min_len = min(len(object_points), len(object_colors))
            object_points = object_points[:min_len]
            object_colors = object_colors[:min_len]
            
            instance_point_clouds[obj.id] = {
                'points': object_points,
                'colors': object_colors,
                'description': obj.description,
                'num_points': len(object_points)
            }
            
            print(f"Extracted {len(object_points)} points for object {obj.id}: {obj.description}")
        
        return instance_point_clouds
    
    def create_aggregated_point_cloud(self, instance_point_clouds: Dict[int, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create aggregated point cloud with different colors for each instance
        
        Args:
            instance_point_clouds: Dictionary of instance point clouds
            
        Returns:
            Tuple of (aggregated_points, aggregated_colors)
        """
        if not instance_point_clouds:
            return np.array([]), np.array([])
        
        # Generate distinct colors for each instance
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        num_instances = len(instance_point_clouds)
        colormap = cm.get_cmap('tab20' if num_instances <= 20 else 'hsv')
        
        all_points = []
        all_colors = []
        
        for i, (obj_id, data) in enumerate(instance_point_clouds.items()):
            points = data['points']
            
            # Generate unique color for this instance
            color = colormap(i / max(num_instances - 1, 1))[:3]  # RGB only
            instance_colors = np.tile(color, (len(points), 1))
            
            all_points.append(points)
            all_colors.append(instance_colors)
            
            print(f"Instance {obj_id} ({data['description']}): {len(points)} points, "
                  f"color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        
        aggregated_points = np.vstack(all_points)
        aggregated_colors = np.vstack(all_colors)
        
        print(f"Created aggregated point cloud with {len(aggregated_points)} points from {num_instances} instances")
        return aggregated_points, aggregated_colors
    
    def run(self, image: np.ndarray, output_dir: Optional[Path] = None,
            detected_objects: Optional[List] = None) -> DepthEstimation:
        """
        Run the complete depth estimation pipeline
        
        Args:
            image: Input RGB image
            output_dir: Optional directory to save results
            detected_objects: Optional list of detected objects for instance point clouds
            
        Returns:
            DepthEstimation object with all results
        """
        print("Starting depth estimation pipeline...")
        
        # Estimate depth
        depth_estimation = self.estimate_depth(image)
        
        # Save results if output directory is provided
        if output_dir:
            self._save_results(depth_estimation, image, output_dir, detected_objects)
        
        print("Depth estimation pipeline complete.")
        return depth_estimation
    
    def _save_results(self, depth_estimation: DepthEstimation, 
                     image: np.ndarray, output_dir: Path,
                     detected_objects: Optional[List] = None) -> None:
        """Save depth estimation results"""
        depth_dir = output_dir / "depth_estimation"
        depth_dir.mkdir(exist_ok=True, parents=True)
        
        # Save depth map as image
        depth_normalized = self._normalize_depth_for_visualization(depth_estimation.depth_map)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(depth_normalized, cmap='plasma')
        plt.title("Depth Map")
        plt.colorbar()
        plt.axis('off')
        
        if depth_estimation.normal_map is not None:
            plt.subplot(2, 2, 3)
            normal_vis = (depth_estimation.normal_map + 1) / 2  # Normalize to [0, 1]
            plt.imshow(normal_vis)
            plt.title("Normal Map")
            plt.axis('off')
        
        if depth_estimation.mask is not None:
            plt.subplot(2, 2, 4)
            plt.imshow(depth_estimation.mask, cmap='gray')
            plt.title("Valid Mask")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(depth_dir / "depth_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save raw depth data
        np.save(depth_dir / "depth_map.npy", depth_estimation.depth_map)
        np.save(depth_dir / "point_cloud.npy", depth_estimation.point_cloud)
        
        if depth_estimation.normal_map is not None:
            np.save(depth_dir / "normal_map.npy", depth_estimation.normal_map)
        
        if depth_estimation.mask is not None:
            cv2.imwrite(str(depth_dir / "mask.png"), depth_estimation.mask)
        
        if depth_estimation.intrinsics is not None:
            np.save(depth_dir / "intrinsics.npy", depth_estimation.intrinsics)
        
        # Save Open3D point cloud
        try:
            pcd = self.create_point_cloud_o3d(depth_estimation, image)
            o3d.io.write_point_cloud(str(depth_dir / "point_cloud.ply"), pcd)
            print(f"Saved point cloud with {len(pcd.points)} points")
        except Exception as e:
            print(f"Error saving point cloud: {e}")

        # Save individual instance point clouds if detected_objects are provided
        if detected_objects:
            print("Extracting and saving instance point clouds...")
            
            # Extract instance point clouds
            instance_point_clouds = self.extract_instance_point_clouds(
                depth_estimation, image, detected_objects
            )
            
            if instance_point_clouds:
                # Create directory for instance point clouds
                instance_dir = depth_dir / "instance_point_clouds"
                instance_dir.mkdir(exist_ok=True, parents=True)
                
                # Save individual instance point clouds
                for obj_id, data in instance_point_clouds.items():
                    # Create Open3D point cloud for this instance
                    instance_pcd = o3d.geometry.PointCloud()
                    instance_pcd.points = o3d.utility.Vector3dVector(data['points'])
                    instance_pcd.colors = o3d.utility.Vector3dVector(data['colors'])
                    
                    # Save instance point cloud
                    instance_file = instance_dir / f"instance_{obj_id}_{data['description'].replace(' ', '_')}.ply"
                    o3d.io.write_point_cloud(str(instance_file), instance_pcd)
                    print(f"Saved instance point cloud for object {obj_id} ({data['description']}) "
                          f"with {data['num_points']} points")
                
                # Create and save aggregated colored point cloud
                try:
                    aggregated_points, aggregated_colors = self.create_aggregated_point_cloud(instance_point_clouds)
                    
                    if len(aggregated_points) > 0:
                        aggregated_pcd = o3d.geometry.PointCloud()
                        aggregated_pcd.points = o3d.utility.Vector3dVector(aggregated_points)
                        aggregated_pcd.colors = o3d.utility.Vector3dVector(aggregated_colors)
                        
                        aggregated_file = depth_dir / "aggregated_instances_point_cloud.ply"
                        o3d.io.write_point_cloud(str(aggregated_file), aggregated_pcd)
                        print(f"Saved aggregated instance point cloud with {len(aggregated_points)} points")
                        
                        # Save instance summary
                        instance_summary = {
                            "num_instances": len(instance_point_clouds),
                            "total_points": len(aggregated_points),
                            "instances": {
                                str(obj_id): {
                                    "description": data['description'],
                                    "num_points": data['num_points']
                                }
                                for obj_id, data in instance_point_clouds.items()
                            }
                        }
                        
                        import json
                        with open(instance_dir / "instance_summary.json", "w") as f:
                            json.dump(instance_summary, f, indent=2)
                            
                except Exception as e:
                    print(f"Error creating aggregated point cloud: {e}")
            else:
                print("No valid instance point clouds extracted")
    
    def _normalize_depth_for_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """Normalize depth map for visualization"""
        # Remove invalid values
        valid_depth = depth_map[depth_map > 0]
        
        if len(valid_depth) == 0:
            return depth_map
        
        # Use percentile-based normalization for better visualization
        min_depth = np.percentile(valid_depth, 2)
        max_depth = np.percentile(valid_depth, 98)
        
        normalized = np.clip((depth_map - min_depth) / (max_depth - min_depth), 0, 1)
        
        # Set invalid pixels to 0
        normalized[depth_map <= 0] = 0
        
        return normalized
    
    def visualize_point_cloud(self, depth_estimation: DepthEstimation, 
                            image: np.ndarray) -> None:
        """Visualize point cloud using Open3D"""
        try:
            pcd = self.create_point_cloud_o3d(depth_estimation, image)
            
            # Remove outliers
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Visualize
            print("Visualizing point cloud... Close the window to continue.")
            o3d.visualization.draw_geometries([pcd])
            
        except Exception as e:
            print(f"Error visualizing point cloud: {e}")
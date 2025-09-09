"""
Render-and-Compare Module for Rotation Estimation

This module implements a render-and-compare approach for estimating object rotations
when ICP fails. It uses Blender to render the mesh from multiple viewpoints and 
uses Qwen-VL to find the best matching orientation.
"""
import sys 
import contextlib
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import os
from io import StringIO

try:
    import bpy
    import bmesh
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("Warning: Blender Python API not available. Render-and-compare will be disabled.")

from ..core.common import Mesh3D, DetectedObject
from ..utils.api_clients import QwenVLClient
from ..utils.image_utils import image_to_base64

class SuppressOutput:
    """Enhanced context manager to suppress Blender rendering output"""
    
    def __enter__(self):
        # Store original file descriptors
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        # Redirect Python stdout/stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        # Also redirect OS-level stdout/stderr (for C/C++ output)
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        
        # Create null file descriptor
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect file descriptors
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore OS-level stdout/stderr
        os.dup2(self._stdout_fd, 1)
        os.dup2(self._stderr_fd, 2)
        
        # Close file descriptors
        os.close(self._stdout_fd)
        os.close(self._stderr_fd)
        os.close(self._devnull)
        
        # Restore Python stdout/stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class RenderCompareModule:
    """Module for rotation estimation using render-and-compare approach"""
    
    def __init__(self):
        self.qwen_client = QwenVLClient()
        
        # Rendering parameters
        self.num_elevations = 3  # 3 elevation angles
        self.num_azimuths = 8    # 8 azimuth angles
        self.image_size = 384    # Size of each rendered image
        
        # Elevation angles (in degrees)
        self.elevations = [-20., 0., 20.]  # Low, medium, high viewpoints
        
        # Azimuth angles (in degrees) - evenly distributed around object
        self.azimuths = [i * 45 for i in range(8)]  # 0, 45, 90, 135, 180, 225, 270, 315
        
    def setup_blender_scene(self) -> None:
        """Setup Blender scene for rendering"""
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender Python API not available")
        
        # Clear existing objects more carefully
        bpy.ops.object.select_all(action='DESELECT')
        
        # Delete all mesh objects
        for obj in bpy.context.scene.objects:
            if obj.type in ['MESH', 'CAMERA', 'LIGHT']:
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # Set up camera
        bpy.ops.object.camera_add(location=(3, 0, 1))
        camera = bpy.context.object
        camera.name = "RenderCamera"
        
        # Set camera as active camera for the scene
        bpy.context.scene.camera = camera
        
        # Set up lighting - three-point lighting setup
        # Key light
        bpy.ops.object.light_add(type='SUN', location=(2, 2, 3))
        key_light = bpy.context.object
        key_light.data.energy = 3.0
        
        # Fill light
        bpy.ops.object.light_add(type='SUN', location=(-1, 1, 2))
        fill_light = bpy.context.object
        fill_light.data.energy = 1.5
        
        # Back light
        bpy.ops.object.light_add(type='SUN', location=(0, -2, 1))
        back_light = bpy.context.object
        back_light.data.energy = 1.0
        
        # Set render settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.resolution_x = self.image_size
        bpy.context.scene.render.resolution_y = self.image_size
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.cycles.samples = 64  # Good quality/speed balance
        
        # Set background to white
        world = bpy.context.scene.world
        world.use_nodes = True
        bg_node = world.node_tree.nodes["Background"]
        bg_node.inputs[0].default_value = (1, 1, 1, 1)  # White background
        
    def import_mesh_to_blender(self, mesh: Mesh3D) -> str:
        """
        Import mesh to Blender scene using GLB file if available
        
        Args:
            mesh: 3D mesh to import
            
        Returns:
            Name of the imported object
        """
        # Try to import from GLB file first (preserves materials and textures)
        if mesh.file_path and mesh.file_path.exists() and mesh.file_path.suffix.lower() == '.glb':
            return self._import_glb_file(mesh.file_path)
        else:
            # Fallback to raw mesh data import
            return self._import_raw_mesh(mesh)
    
    def _import_glb_file(self, glb_path: Path) -> str:
        """
        Import GLB file to Blender scene
        
        Args:
            glb_path: Path to GLB file
            
        Returns:
            Name of the imported object
        """
        try:
            # Import GLB file
            bpy.ops.import_scene.gltf(filepath=str(glb_path))
            
            # Get the imported object (should be the last selected object)
            imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            
            if not imported_objects:
                print(f"Warning: No mesh objects found in GLB file {glb_path}")
                return self._create_fallback_object()
            
            # Use the first mesh object
            obj = imported_objects[0]
            obj.name = "ObjectToRender"
            
            # Center the object at origin
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
            obj.location = (0, 0, 0)
            
            print(f"Successfully imported GLB file: {glb_path}")
            return obj.name
            
        except Exception as e:
            print(f"Error importing GLB file {glb_path}: {e}")
            return self._create_fallback_object()
    
    def _import_raw_mesh(self, mesh: Mesh3D) -> str:
        """
        Import raw mesh data to Blender (fallback method)
        
        Args:
            mesh: 3D mesh with vertices and faces
            
        Returns:
            Name of the imported object
        """
        try:
            # Create mesh in Blender
            mesh_data = bpy.data.meshes.new("ObjectMesh")
            mesh_data.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
            mesh_data.update()
            
            # Create object
            obj = bpy.data.objects.new("ObjectToRender", mesh_data)
            bpy.context.collection.objects.link(obj)
            
            # Center the object at origin
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
            obj.location = (0, 0, 0)
            
            # Add material for better rendering (since raw mesh has no materials)
            material = bpy.data.materials.new(name="ObjectMaterial")
            material.use_nodes = True
            
            # Set up basic material
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Base color
            bsdf.inputs[4].default_value = 0.5  # Metallic
            bsdf.inputs[7].default_value = 0.3  # Roughness
            
            obj.data.materials.append(material)
            
            print("Imported raw mesh data (no textures/materials from GLB)")
            return obj.name
            
        except Exception as e:
            print(f"Error importing raw mesh: {e}")
            return self._create_fallback_object()
    
    def _create_fallback_object(self) -> str:
        """Create a simple fallback object when import fails"""
        try:
            bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
            obj = bpy.context.object
            obj.name = "FallbackObject"
            return obj.name
        except:
            return "Cube"  # Default cube should exist
    
    def set_camera_position(self, elevation: float, azimuth: float, distance: float = 2.4) -> None:
        """
        Set camera position based on spherical coordinates
        
        Args:
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees  
            distance: Distance from object center
        """
        camera = bpy.data.objects.get("RenderCamera")
        if camera is None:
            print("Warning: RenderCamera not found, creating new camera")
            bpy.ops.object.camera_add(location=(3, 0, 1))
            camera = bpy.context.object
            camera.name = "RenderCamera"
            bpy.context.scene.camera = camera
        
        # Convert to radians
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)
        
        # Spherical to Cartesian conversion
        x = distance * np.cos(elev_rad) * np.cos(azim_rad)
        y = distance * np.cos(elev_rad) * np.sin(azim_rad)
        z = distance * np.sin(elev_rad)
        
        camera.location = (x, y, z)
        
        # Point camera at origin using Blender's track-to constraint approach
        # Clear existing rotation
        camera.rotation_euler = (0, 0, 0)
        
        # Use Blender's built-in look-at functionality
        direction = np.array([0, 0, 0]) - np.array([x, y, z])  # Look at origin
        direction = direction / np.linalg.norm(direction)
        
        # Set rotation to look at origin
        camera.rotation_euler = self._look_at_rotation(direction)
        
        # Ensure camera is set as active camera
        bpy.context.scene.camera = camera
    
    def _look_at_rotation(self, direction: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate Euler rotation to look in given direction
        
        Args:
            direction: Normalized direction vector
            
        Returns:
            Euler angles (x, y, z) in radians
        """
        # Calculate rotation to align -Z axis (camera forward) with direction
        z_axis = -direction
        
        # Calculate Y axis (up vector)
        world_up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([0, 1, 0])
        
        x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # Create rotation matrix
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Convert to Euler angles
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rot_matrix)
        return tuple(r.as_euler('xyz'))
    
    def render_single_view(self, elevation: float, azimuth: float, 
                          output_path: str) -> bool:
        """
        Render single view of the object
        
        Args:
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
            output_path: Path to save rendered image
            
        Returns:
            True if rendering successful
        """
        try:
            # Set camera position
            self.set_camera_position(elevation, azimuth)
            
            # Ensure we have a valid camera set
            if bpy.context.scene.camera is None:
                print("Error: No active camera in scene")
                return False
            
            # Set output path
            bpy.context.scene.render.filepath = output_path
            
            # Render
            with SuppressOutput():
                bpy.ops.render.render(write_still=True)
            
            # Verify file was created
            if os.path.exists(output_path):
                return True
            else:
                print(f"Warning: Render completed but file not found at {output_path}")
                return False
            
        except Exception as e:
            print(f"Error rendering view (elev={elevation}, azim={azimuth}): {e}")
            return False
    
    def render_all_views(self, mesh: Mesh3D, output_dir: Path) -> List[str]:
        """
        Render all viewpoints of the mesh
        
        Args:
            mesh: 3D mesh to render
            output_dir: Directory to save rendered images
            
        Returns:
            List of rendered image paths
        """
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender Python API not available")
        
        print("Setting up Blender scene for rendering...")
        self.setup_blender_scene()
        
        print("Importing mesh to Blender...")
        obj_name = self.import_mesh_to_blender(mesh)
        
        rendered_paths = []
        
        print(f"Rendering {len(self.elevations)} x {len(self.azimuths)} = {len(self.elevations) * len(self.azimuths)} views...")
        
        for i, elevation in enumerate(self.elevations):
            for j, azimuth in enumerate(self.azimuths):
                # Create filename
                filename = f"render_elev_{i}_azim_{j}.png"
                output_path = output_dir / filename
                
                # Render view
                success = self.render_single_view(elevation, azimuth, str(output_path))
                
                if success:
                    rendered_paths.append(str(output_path))
                    print(f"  Rendered view {len(rendered_paths)}/{len(self.elevations) * len(self.azimuths)}", end='\r')
                else:
                    print(f"  Failed to render view elev={elevation}, azim={azimuth}")
        
        return rendered_paths
    
    def create_comparison_grid(self, rendered_paths: List[str], 
                              output_path: str) -> np.ndarray:
        """
        Create a 3x8 grid image with numbered annotations
        
        Args:
            rendered_paths: List of rendered image paths
            output_path: Path to save the grid image
            
        Returns:
            Grid image as numpy array
        """
        if len(rendered_paths) != 24:  # 3 x 8
            print(f"Warning: Expected 24 rendered images, got {len(rendered_paths)}")
        
        # Load images
        images = []
        for path in rendered_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (self.image_size, self.image_size))
                    images.append(img)
                else:
                    # Create placeholder if image failed to load
                    placeholder = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                    images.append(placeholder)
            else:
                # Create placeholder if file doesn't exist
                placeholder = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                images.append(placeholder)
        
        # Pad with placeholders if needed
        while len(images) < 24:
            placeholder = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            images.append(placeholder)
        
        # Create grid
        grid_height = 3 * self.image_size
        grid_width = 8 * self.image_size
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place images in grid with annotations
        for i in range(3):  # rows (elevations)
            for j in range(8):  # columns (azimuths)
                idx = i * 8 + j
                if idx < len(images):
                    y_start = i * self.image_size
                    y_end = (i + 1) * self.image_size
                    x_start = j * self.image_size
                    x_end = (j + 1) * self.image_size
                    
                    # Place image
                    grid_image[y_start:y_end, x_start:x_end] = images[idx]
                    
                    # Add number annotation
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    
                    # Position number in top-left corner
                    text = str(idx)
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # White background for number
                    cv2.rectangle(grid_image, 
                                (x_start + 5, y_start + 5),
                                (x_start + text_size[0] + 15, y_start + text_size[1] + 15),
                                (255, 255, 255), -1)
                    
                    # Black text
                    cv2.putText(grid_image, text, 
                              (x_start + 10, y_start + text_size[1] + 10),
                              font, font_scale, (0, 0, 0), thickness)
        
        # Save grid image
        cv2.imwrite(output_path, grid_image)
        print(f"Created comparison grid: {output_path}")
        
        return grid_image
    
    def query_best_orientation(self, grid_image: np.ndarray, 
                              cropped_object: np.ndarray) -> int:
        """
        Use Qwen-VL to find the best matching orientation
        
        Args:
            grid_image: 3x8 grid of rendered views
            cropped_object: Cropped image of the object from original scene
            
        Returns:
            Index of best matching view (0-23)
        """
        system_prompt = """
You are an expert in 3D object analysis and orientation matching.
You will be shown two images:
1. A grid of 24 rendered views of a 3D object (3 rows x 8 columns, numbered 0-23)
2. A cropped image of the same object from a real scene

Your task is to identify which numbered view in the grid has the most similar orientation/pose to the object in the cropped image.

Consider:
- Object orientation and viewing angle
- Visible surfaces and features
- Lighting and shadows (but focus more on geometry)
- Overall pose similarity

Respond with ONLY the number (0-23) of the best matching view.
"""
        
        user_prompt = """
Please analyze the grid of rendered views and the cropped object image.
Identify which numbered view (0-23) in the grid has the most similar orientation to the object in the cropped image.

Return only the number of the best matching view.
"""
        
        try:
            # Create a combined image for analysis
            # Resize cropped object to match grid cell size
            cropped_resized = cv2.resize(cropped_object, (self.image_size, self.image_size))
            
            # Create combined image: grid on top, cropped object on bottom
            combined_height = grid_image.shape[0] + self.image_size + 20  # 20px spacing
            combined_width = max(grid_image.shape[1], self.image_size)
            combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
            
            # Place grid image
            combined_image[:grid_image.shape[0], :grid_image.shape[1]] = grid_image
            
            # Add label for reference image
            cv2.putText(combined_image, "Reference Object:", 
                       (10, grid_image.shape[0] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Place cropped object
            y_start = grid_image.shape[0] + 20
            combined_image[y_start:y_start + self.image_size, :self.image_size] = cropped_resized
            import datetime 
            from imageio.v2 import imread, imsave
            imsave(f"combined_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", combined_image)
            
            # Query Qwen-VL
            response = self.qwen_client.analyze_image(
                combined_image, system_prompt, user_prompt
            )
            
            # Parse response to get the index
            try:
                # Extract number from response
                import re
                numbers = re.findall(r'\b(\d+)\b', response)
                if numbers:
                    best_idx = int(numbers[0])
                    if 0 <= best_idx <= 23:
                        return best_idx
                    else:
                        print(f"Warning: Invalid index {best_idx} from Qwen-VL, using 0")
                        return 0
                else:
                    print(f"Warning: No valid index found in response: {response}")
                    return 0
                    
            except Exception as e:
                print(f"Error parsing Qwen-VL response: {e}")
                print(f"Raw response: {response}")
                return 0
                
        except Exception as e:
            print(f"Error querying Qwen-VL for orientation: {e}")
            return 0
    
    def get_rotation_from_index(self, index: int) -> np.ndarray:
        """
        Get rotation matrix from view index
        
        Args:
            index: View index (0-23)
            
        Returns:
            3x3 rotation matrix
        """
        if not (0 <= index <= 23):
            print(f"Warning: Invalid index {index}, using 0")
            index = 0
        
        # Calculate elevation and azimuth from index
        elev_idx = index // 8
        azim_idx = index % 8
        
        elevation = self.elevations[elev_idx]
        azimuth = self.azimuths[azim_idx]
        
        print(f"Selected view {index}: elevation={elevation}°, azimuth={azimuth}°")
        
        # Convert to rotation matrix
        # This represents the rotation needed to achieve the selected viewpoint
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)
        
        # Create rotation matrix (object rotation, not camera rotation)
        # We want to rotate the object to match the selected viewpoint
        from scipy.spatial.transform import Rotation
        
        # Rotation around Z-axis (azimuth) then X-axis (elevation)
        rot_z = Rotation.from_euler('z', azimuth, degrees=True)
        rot_x = Rotation.from_euler('x', -elevation, degrees=True)  # Negative for object rotation
        
        # Combine rotations
        combined_rotation = rot_z * rot_x
        
        return combined_rotation.as_matrix()

    def _create_matted_image(self, cropped_image: np.ndarray, cropped_mask: np.ndarray) -> np.ndarray:
         # Create matted image (cropped image with alpha channel for transparency)
        if cropped_mask is not None:
            # Apply mask to create transparency
            mask_3d = cropped_mask[..., np.newaxis] / 255.0
            matted_image = cropped_image * mask_3d
            
            # Convert to RGBA
            alpha_channel = cropped_mask[..., np.newaxis]
            matted_rgba = np.concatenate([matted_image.astype(np.uint8), alpha_channel], axis=2)
        else:
            # Use original cropped image if no mask available
            matted_rgba = cropped_image
        return matted_rgba
    
    def estimate_rotation_render_compare(self, mesh: Mesh3D, 
                                       detected_object: DetectedObject,
                                       output_dir: Path) -> np.ndarray:
        """
        Estimate object rotation using render-and-compare approach
        
        Args:
            mesh: 3D mesh of the object
            detected_object: Detected object with cropped image
            output_dir: Directory to save intermediate results
            
        Returns:
            3x3 rotation matrix
        """
        if not BLENDER_AVAILABLE:
            print("Warning: Blender not available, returning identity rotation")
            return np.eye(3)
        
        if detected_object.cropped_image is None:
            print("Warning: No cropped image available for render-compare")
            return np.eye(3)
        
        print(f"Estimating rotation for object {detected_object.id} using render-and-compare...")
        
        # Create output directory for this object
        render_dir = output_dir / "render_and_compare" / f"obj_{detected_object.id}"
        render_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Render all views
            rendered_paths = self.render_all_views(mesh, render_dir)
            
            if not rendered_paths:
                print("Warning: No views were rendered successfully")
                return np.eye(3)
            
            # Create comparison grid
            grid_path = render_dir / "comparison_grid.png"
            grid_image = self.create_comparison_grid(rendered_paths, str(grid_path))
            
            # Query Qwen-VL for best match
            # best_index = self.query_best_orientation(grid_image, detected_object.cropped_image)
            # use the matted image since it's much more concise  
            best_index = self.query_best_orientation(grid_image, self._create_matted_image(detected_object.cropped_image, detected_object.cropped_mask))
            
            # Get rotation matrix from selected index
            rotation_matrix = self.get_rotation_from_index(best_index)
            
            # Save results
            results = {
                "best_view_index": best_index,
                "elevation": self.elevations[best_index // 8],
                "azimuth": self.azimuths[best_index % 8],
                "rotation_matrix": rotation_matrix.tolist(),
                "rendered_views": len(rendered_paths)
            }
            
            with open(render_dir / "render_compare_results.json", 'w') as f:
                import json
                json.dump(results, f, indent=2)
            
            print(f"Render-and-compare complete. Selected view {best_index}")
            return rotation_matrix
            
        except Exception as e:
            print(f"Error in render-and-compare: {e}")
            return np.eye(3)
    
    def cleanup_blender_scene(self) -> None:
        """Clean up Blender scene after rendering"""
        if BLENDER_AVAILABLE:
            try:
                # Clear all objects
                bpy.ops.object.select_all(action='SELECT')
                bpy.ops.object.delete(use_global=False, confirm=False)
                
                # Clear materials
                for material in bpy.data.materials:
                    bpy.data.materials.remove(material)
                
                # Clear meshes
                for mesh in bpy.data.meshes:
                    bpy.data.meshes.remove(mesh)
                    
            except Exception as e:
                print(f"Warning: Error cleaning up Blender scene: {e}")

"""
3D Mesh Generation Module using Multiple APIs

This module handles 3D mesh generation from inpainted object images
using various API services including Tripo3D and TRELLIS.
"""
import os
import numpy as np
import asyncio
from typing import List, Optional, Literal
from pathlib import Path
import time
import trimesh

from ..core.common import DetectedObject, Mesh3D
from ..utils.api_clients import Tripo3DClient, TrellisClient
from ..utils.image_utils import save_image
from ..config.settings import config

class MeshGenerationModule:
    """Module for 3D mesh generation using multiple APIs"""
    
    def __init__(self, provider: Literal["tripo3d", "trellis"] = "tripo3d", base_url: Optional[str] = None):
        self.provider = provider
        self.base_url = base_url
        # Default to local TRELLIS deployment if no base_url specified for TRELLIS
        if provider == "tripo3d":
            self.tripo_client = Tripo3DClient()
            self.trellis_client = None
        elif provider == "trellis":
            self.trellis_client = TrellisClient(base_url=self.base_url)
            self.tripo_client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        print(f"Initialized MeshGenerationModule with provider: {provider}")
        if provider == "trellis" and self.trellis_client:
            mode = "synchronous" if self.trellis_client.is_local else "asynchronous"
            print(f"TRELLIS mode: {mode} ({'local' if self.trellis_client.is_local else 'remote'} deployment)")
        
    def generate_mesh_for_object_tripo_sync(self, detected_object: DetectedObject, 
                                output_dir: Optional[Path] = None) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using its inpainted image
        
        Args:
            detected_object: Object with inpainted image
            output_dir: Directory to save the mesh file
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None
        
        print(f"Generating 3D mesh for object {detected_object.id}: {detected_object.description}")
        
        try:
            # Step 1: Upload image to Tripo3D
            print("  Uploading image...")
            file_token = self.tripo_client.upload_image(detected_object.generated_image)
            
            if not file_token:
                print(f"  Failed to upload image for object {detected_object.id}")
                return None
            
            # Step 2: Create 3D model task
            print("  Creating 3D model task...")
            task_id = self.tripo_client.create_3d_model(file_token)
            
            if not task_id:
                print(f"  Failed to create 3D model task for object {detected_object.id}")
                return None
            
            # Step 3: Wait for completion
            print("  Waiting for 3D model generation...")
            try:
                result = self.tripo_client.wait_for_completion(task_id, timeout=600)  # 10 minutes
            except TimeoutError:
                print(f"  Timeout waiting for object {detected_object.id}")
                return None
            except Exception as e:
                print(f"  Error during generation: {e}")
                return None
            
            # Step 4: Download the mesh
            if "result" in result and "pbr_model" in result["result"]:
                model_url = result["result"]["pbr_model"]
                print(f"  Downloading mesh from: {model_url}")
                
                # Create output path
                if output_dir:
                    mesh_dir = output_dir / "meshes"
                    mesh_dir.mkdir(exist_ok=True, parents=True)
                    mesh_path = mesh_dir / f"object_{detected_object.id}.glb"
                else:
                    mesh_path = Path(f"object_{detected_object.id}.glb")
                
                # Download mesh file
                if self.tripo_client.download_model(model_url, mesh_path):
                    # Load mesh using trimesh
                    mesh_3d = self._load_mesh_from_file(mesh_path)
                    if mesh_3d:
                        print(f"  Successfully generated mesh for object {detected_object.id}")
                        return mesh_3d
                    else:
                        print(f"  Failed to load downloaded mesh for object {detected_object.id}")
                else:
                    print(f"  Failed to download mesh for object {detected_object.id}")
            else:
                print(f"  Invalid result format for object {detected_object.id}")
                
        except Exception as e:
            import traceback 
            traceback.print_exc()
            print(f"Error generating mesh for object {detected_object.id}: {e}")
        
        return None
    
    def generate_mesh_for_object_trellis_sync(self, detected_object: DetectedObject,
                                            output_dir: Optional[Path] = None,
                                            **trellis_kwargs) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using TRELLIS synchronously (for local deployments)
        
        Args:
            detected_object: Object with inpainted image
            output_dir: Directory to save the mesh file
            **trellis_kwargs: Additional parameters for TRELLIS
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None and detected_object.cropped_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None
        
        if not self.trellis_client.is_local:
            raise RuntimeError("Synchronous generation requires local TRELLIS deployment")
        
        print(f"Generating 3D mesh using TRELLIS (sync) for object {detected_object.id}: {detected_object.description}")
        
        try:
            # Generate synchronously
            result = self.trellis_client.generate_3d_sync(
                detected_object.generated_image if detected_object.generated_image is not None else detected_object.cropped_image, **trellis_kwargs
            )
            
            if not result:
                print(f"Failed TRELLIS sync generation for object {detected_object.id}")
                return None
            
            if result["output"] is not None and "model_file" in result["output"]:
                model_url = result["output"]["model_file"]
                
                # Create output path
                if output_dir:
                    mesh_dir = output_dir / "meshes"
                    mesh_dir.mkdir(exist_ok=True, parents=True)
                    mesh_path = mesh_dir / f"object_{detected_object.id}_trellis.glb"
                else:
                    mesh_path = Path(f"object_{detected_object.id}_trellis.glb")
                
                with open(mesh_path, "wb") as f:
                    f.write(model_url.read())

                # Download mesh file
                if os.path.exists(mesh_path):
                    # Load mesh using trimesh
                    mesh_3d = self._load_mesh_from_file(mesh_path)
                    if mesh_3d:
                        print(f"Successfully generated TRELLIS sync mesh for object {detected_object.id}")
                        return mesh_3d
                    else:
                        print(f"Failed to load downloaded TRELLIS sync mesh for object {detected_object.id}")
                else:
                    print(f"Failed to download TRELLIS sync mesh for object {detected_object.id}")
            else:
                print(f"Invalid TRELLIS sync result format for object {detected_object.id}")
                
        except Exception as e:
            import traceback 
            traceback.print_exc()
            print(f"Error generating TRELLIS sync mesh for object {detected_object.id}: {e}")
        
        return None
    
    async def generate_mesh_for_object_trellis_async(self, detected_object: DetectedObject,
                                                   output_dir: Optional[Path] = None,
                                                   **trellis_kwargs) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using TRELLIS asynchronously
        
        Args:
            detected_object: Object with inpainted image
            output_dir: Directory to save the mesh file
            **trellis_kwargs: Additional parameters for TRELLIS
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None and detected_object.cropped_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None
        
        print(f"Generating 3D mesh using TRELLIS for object {detected_object.id}: {detected_object.description}")
        
        try:
            # Start async generation
            prediction_id = await self.trellis_client.generate_3d_async(
                detected_object.generated_image if detected_object.generated_image is not None else detected_object.cropped_image, **trellis_kwargs
            )
            
            if not prediction_id:
                print(f"Failed to start TRELLIS generation for object {detected_object.id}")
                return None
            
            # Wait for completion
            result = await self.trellis_client.wait_for_completion_async(prediction_id)
            
            if result["output"] is not None and "model_file" in result["output"]:
                model_url = result["output"]["model_file"]
                
                # Create output path
                if output_dir:
                    mesh_dir = output_dir / "meshes"
                    mesh_dir.mkdir(exist_ok=True, parents=True)
                    mesh_path = mesh_dir / f"object_{detected_object.id}_trellis.glb"
                else:
                    mesh_path = Path(f"object_{detected_object.id}_trellis.glb")
                
                # Download mesh file
                if await self.trellis_client._download_model_async(model_url, mesh_path):
                    # Load mesh using trimesh
                    mesh_3d = self._load_mesh_from_file(mesh_path)
                    if mesh_3d:
                        print(f"Successfully generated TRELLIS mesh for object {detected_object.id}")
                        return mesh_3d
                    else:
                        print(f"Failed to load downloaded TRELLIS mesh for object {detected_object.id}")
                else:
                    print(f"Failed to download TRELLIS mesh for object {detected_object.id}")
            else:
                print(f"Invalid TRELLIS result format for object {detected_object.id}")
                
        except Exception as e:
            print(f"Error generating TRELLIS mesh for object {detected_object.id}: {e}")
        
        return None
    
    def _load_mesh_from_file(self, mesh_path: Path) -> Optional[Mesh3D]:
        """
        Load mesh from file using trimesh
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Mesh3D object if successful
        """
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(mesh_path))
            
            # Handle different mesh types
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, get the first mesh
                mesh_geometries = [geom for geom in mesh.geometry.values() 
                                 if isinstance(geom, trimesh.Trimesh)]
                if mesh_geometries:
                    mesh = mesh_geometries[0]
                else:
                    print(f"No valid mesh found in scene file: {mesh_path}")
                    return None
            
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Loaded object is not a valid mesh: {mesh_path}")
                return None
            
            # Extract vertices and faces
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            # Extract texture/color information if available
            textures = None
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                textures = np.array(mesh.visual.vertex_colors)
            elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                textures = np.array(mesh.visual.face_colors)
            
            # Create Mesh3D object
            mesh_3d = Mesh3D(
                vertices=vertices,
                faces=faces,
                textures=textures,
                file_path=mesh_path
            )
            
            print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh_3d
            
        except Exception as e:
            print(f"Error loading mesh from {mesh_path}: {e}")
            return None
    
    def batch_generate_meshes(self, detected_objects: List[DetectedObject],
                            output_dir: Optional[Path] = None,
                            max_concurrent: int = 3,
                            **provider_kwargs) -> List[Optional[Mesh3D]]:
        """
        Generate meshes for multiple objects with controlled concurrency
        
        Args:
            detected_objects: List of objects to generate meshes for
            output_dir: Directory to save mesh files
            max_concurrent: Maximum number of concurrent API calls
            **provider_kwargs: Additional parameters for the mesh generation provider
            
        Returns:
            List of Mesh3D objects (None for failed generations)
        """
        print(f"Starting batch mesh generation for {len(detected_objects)} objects using {self.provider}...")
        
        if self.provider == "trellis":
            # Choose sync or async based on deployment type
            if self.trellis_client.is_local:
                return self._batch_generate_meshes_trellis_sync(detected_objects, output_dir, **provider_kwargs)
            else:
                return asyncio.run(self._batch_generate_meshes_async(detected_objects, output_dir, **provider_kwargs))
        else:
            # Use sync generation for Tripo3D
            return self._batch_generate_meshes_tripo_sync(detected_objects, output_dir, max_concurrent)
    
    def _batch_generate_meshes_tripo_sync(self, detected_objects: List[DetectedObject],
                                  output_dir: Optional[Path] = None,
                                  max_concurrent: int = 3) -> List[Optional[Mesh3D]]:
        """Synchronous batch generation for Tripo3D"""
        meshes = []
        
        # Process objects in batches to avoid overwhelming the API
        for i in range(0, len(detected_objects), max_concurrent):
            batch = detected_objects[i:i + max_concurrent]
            print(f"Processing batch {i//max_concurrent + 1}...")
            
            batch_meshes = []
            for obj in batch:
                mesh = self.generate_mesh_for_object_tripo_sync(obj, output_dir)
                batch_meshes.append(mesh)
                
                # Add delay between requests to be respectful to the API
                time.sleep(2)
            
            meshes.extend(batch_meshes)
        
        successful_meshes = sum(1 for mesh in meshes if mesh is not None)
        print(f"Batch mesh generation complete. {successful_meshes}/{len(detected_objects)} successful.")
        
        return meshes
    
    def _batch_generate_meshes_trellis_sync(self, detected_objects: List[DetectedObject],
                                          output_dir: Optional[Path] = None,
                                          **trellis_kwargs) -> List[Optional[Mesh3D]]:
        """Synchronous batch generation for TRELLIS (local deployments)"""
        print(f"Starting sync TRELLIS batch generation for {len(detected_objects)} objects...")
        
        # Filter objects with valid images
        valid_objects = [obj for obj in detected_objects 
                        if obj.generated_image is not None or obj.cropped_image is not None]
        
        if not valid_objects:
            print("No valid objects with inpainted images for TRELLIS sync generation")
            return []
        
        # Generate meshes synchronously
        meshes = []
        for i, obj in enumerate(valid_objects):
            print(f"Processing object {i+1}/{len(valid_objects)} synchronously...")
            mesh = self.generate_mesh_for_object_trellis_sync(obj, output_dir, **trellis_kwargs)
            meshes.append(mesh)
        
        successful_meshes = sum(1 for mesh in meshes if mesh is not None)
        print(f"Sync TRELLIS batch generation complete. {successful_meshes}/{len(valid_objects)} successful.")
        
        return meshes
    
    async def _batch_generate_meshes_async(self, detected_objects: List[DetectedObject],
                                         output_dir: Optional[Path] = None,
                                         **trellis_kwargs) -> List[Optional[Mesh3D]]:
        """Asynchronous batch generation for TRELLIS"""
        print(f"Starting async TRELLIS batch generation for {len(detected_objects)} objects...")
        
        # Start all generations concurrently
        images = [obj.generated_image for obj in detected_objects if obj.generated_image is not None]
        valid_objects = [obj for obj in detected_objects if obj.generated_image is not None]
        
        if not images:
            print("No valid objects with inpainted images for TRELLIS generation")
            return []
        
        # Start batch async generation
        prediction_ids = await self.trellis_client.batch_generate_async(images, **trellis_kwargs)
        
        # Collect results
        if output_dir:
            mesh_dir = output_dir / "meshes"
            mesh_dir.mkdir(exist_ok=True, parents=True)
        else:
            mesh_dir = None
        
        model_paths = await self.trellis_client.collect_results_async(prediction_ids, mesh_dir)
        
        # Load meshes from downloaded files
        meshes = []
        for i, (obj, model_path) in enumerate(zip(valid_objects, model_paths)):
            if model_path:
                mesh_3d = self._load_mesh_from_file(Path(model_path))
                meshes.append(mesh_3d)
            else:
                meshes.append(None)
        
        successful_meshes = sum(1 for mesh in meshes if mesh is not None)
        print(f"Async TRELLIS batch generation complete. {successful_meshes}/{len(valid_objects)} successful.")
        
        return meshes
    
    def run(self, detected_objects: List[DetectedObject], 
            output_dir: Optional[Path] = None,
            **provider_kwargs) -> List[Optional[Mesh3D]]:
        """
        Run the complete mesh generation pipeline
        
        Args:
            detected_objects: List of objects with inpainted images
            output_dir: Directory to save results
            
        Returns:
            List of generated meshes
        """
        print("Starting mesh generation pipeline...")
        if not detected_objects:
            print("No objects with inpainted images found")
            return []
        
        print(f"Generating meshes for {len(detected_objects)} objects...")
        
        # Generate meshes
        meshes = self.batch_generate_meshes(detected_objects, output_dir, **provider_kwargs)
        assert not any([mesh is None for mesh in meshes])
        # we also pair the mesh the input images 
        assert len(meshes) == len(detected_objects)
        for (mesh, obj) in zip(meshes, detected_objects):
            mesh.input_image = obj.generated_image
        
        # Save summary
        if output_dir:
            self._save_summary(detected_objects, meshes, output_dir)
        
        print("Mesh generation pipeline complete.")
        return meshes
    
    def _save_summary(self, detected_objects: List[DetectedObject], 
                     meshes: List[Optional[Mesh3D]], output_dir: Path) -> None:
        """Save mesh generation summary"""
        mesh_dir = output_dir / "meshes"
        mesh_dir.mkdir(exist_ok=True, parents=True)
        
        summary = {
            "total_objects": len(detected_objects),
            "successful_meshes": sum(1 for mesh in meshes if mesh is not None),
            "failed_meshes": sum(1 for mesh in meshes if mesh is None),
            "objects": []
        }
        
        for i, (obj, mesh) in enumerate(zip(detected_objects, meshes)):
            if mesh.input_image is not None:
                save_image(mesh.input_image, mesh_dir / f"object_{obj.id}_input_image.png")
            
            obj_summary = {
                "id": obj.id,
                "description": obj.description,
                "mesh_generated": mesh is not None,
                "vertices_count": len(mesh.vertices) if mesh else 0,
                "faces_count": len(mesh.faces) if mesh else 0,
                "mesh_file": str(mesh.file_path) if mesh and mesh.file_path else None
            }
            summary["objects"].append(obj_summary)
        
        # Save summary as JSON
        import json
        with open(mesh_dir / "generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved mesh generation summary: {summary['successful_meshes']}/{summary['total_objects']} successful")
    
    def visualize_mesh(self, mesh: Mesh3D) -> None:
        """
        Visualize a 3D mesh using trimesh
        
        Args:
            mesh: Mesh3D object to visualize
        """
        try:
            # Create trimesh object
            trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            
            # Add colors if available
            if mesh.textures is not None:
                if mesh.textures.shape[0] == len(mesh.vertices):
                    trimesh_obj.visual.vertex_colors = mesh.textures
                elif mesh.textures.shape[0] == len(mesh.faces):
                    trimesh_obj.visual.face_colors = mesh.textures
            
            # Show mesh
            print("Visualizing mesh... Close the window to continue.")
            trimesh_obj.show()
            
        except Exception as e:
            print(f"Error visualizing mesh: {e}")
    
    def export_mesh_formats(self, mesh: Mesh3D, output_path: Path, 
                          formats: List[str] = ["obj", "ply"]) -> None:
        """
        Export mesh to different formats
        
        Args:
            mesh: Mesh3D object to export
            output_path: Base output path (without extension)
            formats: List of formats to export to
        """
        try:
            # Create trimesh object
            trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            
            # Add colors if available
            if mesh.textures is not None:
                if mesh.textures.shape[0] == len(mesh.vertices):
                    trimesh_obj.visual.vertex_colors = mesh.textures
            
            # Export to requested formats
            for fmt in formats:
                export_path = output_path.with_suffix(f".{fmt}")
                trimesh_obj.export(str(export_path))
                print(f"Exported mesh to {export_path}")
                
        except Exception as e:
            print(f"Error exporting mesh: {e}")
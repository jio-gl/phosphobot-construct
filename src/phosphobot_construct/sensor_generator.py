"""
3D-to-Sensor Conversion for the Phosphobot Construct.

This module converts 3D scene representations to realistic sensor data
for training perception models.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import cv2

logger = logging.getLogger(__name__)

# Import conditional to make the module work even without dependencies
try:
    import torch
    import torch.nn as nn
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        PointLights,
        look_at_view_transform,
        TexturesVertex
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.io import load_obj
    HAS_PYTORCH3D = True
except ImportError:
    logger.warning("PyTorch3D not installed. Install with: pip install pytorch3d")
    HAS_PYTORCH3D = False


class SensorGenerator:
    """
    Generates realistic sensor data from 3D scene representations.
    
    This class simulates RGB-D cameras, proprioceptive sensors, and other
    sensor modalities by rendering 3D scenes.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        image_size: Tuple[int, int] = (320, 240),
        num_cameras: int = 3
    ):
        """
        Initialize the sensor generator.
        
        Args:
            device: Device to run rendering on ('cuda' or 'cpu').
            image_size: Output image resolution (width, height).
            num_cameras: Number of camera views to generate.
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.image_size = image_size
        self.width, self.height = image_size
        self.num_cameras = num_cameras
        
        # Initialize renderer if PyTorch3D is available
        if HAS_PYTORCH3D:
            try:
                logger.info(f"Initializing PyTorch3D renderer on {self.device}")
                self._setup_renderer()
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize renderer: {str(e)}")
                self.initialized = False
        else:
            logger.warning("PyTorch3D not available. Sensor generation will be limited.")
            self.initialized = False
        
        # Setup camera positions
        self.camera_positions = self._setup_camera_positions()
    
    def _setup_renderer(self):
        """Set up the PyTorch3D renderer."""
        # Create default rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=self.height,
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        
        # Configure lights
        self.lights = PointLights(
            device=self.device,
            location=[[0, 0, 3]],
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.3, 0.3, 0.3),)
        )
    
    def _setup_camera_positions(self) -> List[Dict[str, Any]]:
        """
        Set up camera positions for multi-view rendering.
        
        Returns:
            List of camera position dictionaries.
        """
        camera_positions = []
        
        # Front camera
        camera_positions.append({
            "position": (0, 0, 2.0),  # (x, y, z)
            "target": (0, 0, 0),      # looking at origin
            "up": (0, 1, 0)           # y-axis is up
        })
        
        # Side cameras
        if self.num_cameras > 1:
            camera_positions.append({
                "position": (1.5, 0.5, 1.5),
                "target": (0, 0, 0),
                "up": (0, 1, 0)
            })
        
        if self.num_cameras > 2:
            camera_positions.append({
                "position": (-1.5, 0.5, 1.5),
                "target": (0, 0, 0),
                "up": (0, 1, 0)
            })
        
        # Add more camera positions if needed
        for i in range(3, self.num_cameras):
            angle = (i - 2) * (2 * np.pi / (self.num_cameras - 2))
            x = 2.0 * np.cos(angle)
            z = 2.0 * np.sin(angle)
            camera_positions.append({
                "position": (x, 0.5, z),
                "target": (0, 0, 0),
                "up": (0, 1, 0)
            })
        
        return camera_positions
    
    def load_mesh(self, mesh_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a 3D mesh from file.
        
        Args:
            mesh_path: Path to the mesh file.
            
        Returns:
            Dictionary with mesh data or None if loading failed.
        """
        if not HAS_PYTORCH3D or not self.initialized:
            logger.error("PyTorch3D not available. Cannot load mesh.")
            return None
        
        try:
            # Check if file exists
            if not os.path.exists(mesh_path):
                logger.error(f"Mesh file not found: {mesh_path}")
                return None
            
            # Load mesh
            verts, faces, aux = load_obj(mesh_path)
            
            # Move to device
            verts = verts.to(self.device)
            faces = faces.verts_idx.to(self.device)
            
            # Create default vertex colors (white)
            verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb)
            
            # Create mesh
            mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=textures
            )
            
            return {
                "verts": verts,
                "faces": faces,
                "mesh": mesh
            }
            
        except Exception as e:
            logger.error(f"Error loading mesh: {str(e)}")
            return None
    
    def render_rgb_depth(
        self,
        meshes: List[Dict[str, Any]],
        camera_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render RGB and depth images of the scene.
        
        Args:
            meshes: List of mesh dictionaries.
            camera_idx: Index of the camera to use.
            
        Returns:
            Tuple of (RGB image, depth image) as numpy arrays.
        """
        if not HAS_PYTORCH3D or not self.initialized:
            logger.warning("PyTorch3D not available. Returning placeholder images.")
            return self._generate_placeholder_images()
        
        if camera_idx >= len(self.camera_positions):
            logger.error(f"Camera index {camera_idx} out of range (max {len(self.camera_positions)-1})")
            return self._generate_placeholder_images()
        
        try:
            # Get camera parameters
            camera_params = self.camera_positions[camera_idx]
            
            # Create camera transformation
            R, T = look_at_view_transform(
                eye=[camera_params["position"]],
                at=[camera_params["target"]],
                up=[camera_params["up"]]
            )
            
            # Create camera
            cameras = FoVPerspectiveCameras(
                device=self.device,
                R=R,
                T=T,
                fov=60  # Field of view in degrees
            )
            
            # Create renderer
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=self.raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=self.lights
                )
            )
            
            # Combine all meshes into a single scene
            if len(meshes) > 0:
                vertices = []
                faces = []
                face_offset = 0
                
                for mesh_data in meshes:
                    mesh = mesh_data["mesh"]
                    vertices.append(mesh.verts_padded()[0])
                    faces.append(mesh.faces_padded()[0] + face_offset)
                    face_offset += mesh.verts_padded().shape[1]
                
                # Create combined mesh
                verts_batch = torch.cat(vertices, dim=0).unsqueeze(0)
                faces_batch = torch.cat(faces, dim=0).unsqueeze(0)
                
                # Create textures
                verts_rgb = torch.ones_like(verts_batch)
                textures = TexturesVertex(verts_features=verts_rgb)
                
                # Create combined mesh
                combined_mesh = Meshes(
                    verts=verts_batch,
                    faces=faces_batch,
                    textures=textures
                )
            else:
                # Create an empty mesh
                logger.warning("No meshes provided. Creating empty scene.")
                empty_verts = torch.zeros((1, 3, 3), device=self.device)
                empty_faces = torch.zeros((1, 1, 3), dtype=torch.int64, device=self.device)
                combined_mesh = Meshes(
                    verts=empty_verts,
                    faces=empty_faces
                )
            
            # Render scene
            rendered_images = renderer(combined_mesh)
            
            # Extract RGB image
            rgb_image = rendered_images[0, :, :, :3].cpu().numpy()
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
            # Extract Z-buffer for depth
            zbuf = rendered_images[0, :, :, 3].cpu().numpy()
            
            # Convert Z-buffer to proper depth map
            depth_image = zbuf.copy()
            valid_mask = zbuf > 0
            if valid_mask.any():
                # Normalize to 0-1 range for valid depth values
                depth_min = zbuf[valid_mask].min()
                depth_max = zbuf[valid_mask].max()
                depth_range = depth_max - depth_min
                if depth_range > 0:
                    depth_image[valid_mask] = (zbuf[valid_mask] - depth_min) / depth_range
            
            # Scale to 0-255 for visualization
            depth_image = (depth_image * 255).astype(np.uint8)
            
            # Resize to desired dimensions
            if rgb_image.shape[:2] != (self.height, self.width):
                rgb_image = cv2.resize(rgb_image, (self.width, self.height))
                depth_image = cv2.resize(depth_image, (self.width, self.height))
            
            return rgb_image, depth_image
            
        except Exception as e:
            logger.error(f"Error rendering images: {str(e)}")
            return self._generate_placeholder_images()
    
    def _generate_placeholder_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate placeholder RGB and depth images.
        
        Returns:
            Tuple of (RGB image, depth image) as numpy arrays.
        """
        # Create a simple checkerboard pattern
        checker_size = 20
        xsize, ysize = self.width, self.height
        x_tiles = xsize // checker_size + 1
        y_tiles = ysize // checker_size + 1
        
        # Create the grid
        grid = np.zeros((y_tiles, x_tiles), dtype=np.uint8)
        grid[::2, ::2] = 255
        grid[1::2, 1::2] = 255
        
        # Resize the grid to the image size
        checker = np.kron(grid, np.ones((checker_size, checker_size), dtype=np.uint8))
        checker = checker[:ysize, :xsize]
        
        # Create RGB image
        rgb_image = np.zeros((ysize, xsize, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = checker
        rgb_image[:, :, 1] = checker
        rgb_image[:, :, 2] = checker
        
        # Create depth image with a radial gradient
        y, x = np.ogrid[-ysize/2:ysize/2, -xsize/2:xsize/2]
        distance = np.sqrt(x*x + y*y)
        distance = distance / distance.max() * 255
        depth_image = distance.astype(np.uint8)
        
        return rgb_image, depth_image
    
    def generate_sensor_data(
        self,
        scene_3d_models: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate sensor data from 3D models.
        
        Args:
            scene_3d_models: List of 3D model dictionaries.
            
        Returns:
            Dictionary with sensor data.
        """
        # Load meshes
        meshes = []
        for model_data in scene_3d_models:
            mesh_path = model_data.get("3d_model_path")
            if mesh_path and os.path.exists(mesh_path):
                mesh_data = self.load_mesh(mesh_path)
                if mesh_data:
                    meshes.append(mesh_data)
        
        # Generate sensor data
        rgb_images = []
        depth_images = []
        
        # Render from each camera viewpoint
        for i in range(min(self.num_cameras, len(self.camera_positions))):
            rgb, depth = self.render_rgb_depth(meshes, camera_idx=i)
            rgb_images.append(rgb)
            depth_images.append(depth)
        
        # Generate proprioception data (placeholder)
        proprioception = self._generate_proprioception_data(scene_3d_models)
        
        # Return all sensor data
        return {
            "rgb_images": rgb_images,
            "depth_images": depth_images,
            "proprioception": proprioception
        }
    
    def _generate_proprioception_data(self, scene_3d_models: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate proprioception data based on the scene.
        
        Args:
            scene_3d_models: List of 3D model dictionaries.
            
        Returns:
            Numpy array with proprioception data.
        """
        # In a real implementation, this would compute realistic joint angles
        # based on the scene and robot geometry. Here we use placeholders.
        
        # Generate random joint angles in a plausible range
        joint_positions = np.random.uniform(-0.5, 0.5, 6)
        joint_velocities = np.random.uniform(-0.1, 0.1, 6)
        joint_torques = np.random.uniform(-0.2, 0.2, 6)
        
        # Combine into a single proprioception vector
        proprioception = np.concatenate([joint_positions, joint_velocities, joint_torques])
        
        return proprioception
    
    def save_sensor_data(
        self,
        sensor_data: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, str]:
        """
        Save sensor data to disk.
        
        Args:
            sensor_data: Dictionary with sensor data.
            output_dir: Directory to save data to.
            
        Returns:
            Dictionary mapping data types to file paths.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save RGB images
        rgb_paths = []
        for i, rgb in enumerate(sensor_data.get("rgb_images", [])):
            rgb_path = os.path.join(output_dir, f"rgb_{i}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rgb_paths.append(rgb_path)
        
        # Save depth images
        depth_paths = []
        for i, depth in enumerate(sensor_data.get("depth_images", [])):
            depth_path = os.path.join(output_dir, f"depth_{i}.png")
            cv2.imwrite(depth_path, depth)
            depth_paths.append(depth_path)
        
        # Save proprioception data
        prop_path = os.path.join(output_dir, "proprioception.npy")
        np.save(prop_path, sensor_data.get("proprioception", np.zeros(18)))
        
        # Return file paths
        return {
            "rgb_paths": rgb_paths,
            "depth_paths": depth_paths,
            "proprioception_path": prop_path
        }


def generate_training_data(
    scene_3d_models: List[Dict[str, Any]],
    output_dir: str = "data/sensor_data",
    num_cameras: int = 3,
    image_size: Tuple[int, int] = (320, 240)
) -> Dict[str, Any]:
    """
    Generate training data from 3D models.
    
    Args:
        scene_3d_models: List of 3D model dictionaries.
        output_dir: Directory to save data to.
        num_cameras: Number of camera views to generate.
        image_size: Output image resolution (width, height).
        
    Returns:
        Dictionary with dataset information.
    """
    logger.info(f"Generating sensor data for {len(scene_3d_models)} 3D models")
    
    # Create sensor generator
    generator = SensorGenerator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_size=image_size,
        num_cameras=num_cameras
    )
    
    # Generate sensor data
    sensor_data = generator.generate_sensor_data(scene_3d_models)
    
    # Save sensor data
    file_paths = generator.save_sensor_data(sensor_data, output_dir)
    
    # Return dataset information
    return {
        "sensor_data": sensor_data,
        "file_paths": file_paths,
        "output_dir": output_dir,
        "num_samples": len(scene_3d_models)
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the sensor generator with placeholder data
    output_dir = "data/test_sensor_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample 3D model dictionaries
    sample_models = [
        {"3d_model_path": "models/cube.obj"},
        {"3d_model_path": "models/sphere.obj"}
    ]
    
    # Generate sensor data
    try:
        dataset = generate_training_data(
            scene_3d_models=sample_models,
            output_dir=output_dir,
            num_cameras=2,
            image_size=(320, 240)
        )
        
        logger.info(f"Generated sensor data saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating sensor data: {str(e)}")
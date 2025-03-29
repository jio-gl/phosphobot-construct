"""
Unit tests for the phosphobot_construct.sensor_generation module.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, PropertyMock

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock torch and pytorch3d since they're optional dependencies
sys.modules['torch'] = MagicMock()
sys.modules['pytorch3d'] = MagicMock()
sys.modules['pytorch3d.renderer'] = MagicMock()
sys.modules['pytorch3d.structures'] = MagicMock()
sys.modules['pytorch3d.io'] = MagicMock()

# Import the module under test with mocked dependencies
from src.phosphobot_construct.sensor_generation import SensorGenerator, generate_training_data


class TestSensorGenerator(unittest.TestCase):
    """Tests for the SensorGenerator class."""
    
    def setUp(self):
        """Setup for tests."""
        # Create patchers
        self.patch_has_pytorch3d = patch('src.phosphobot_construct.sensor_generation.HAS_PYTORCH3D', True)
        self.patch_torch_device = patch('torch.device')
        self.patch_os_path_exists = patch('os.path.exists')
        self.patch_load_obj = patch('src.phosphobot_construct.sensor_generation.load_obj')
        self.patch_meshes = patch('src.phosphobot_construct.sensor_generation.Meshes')
        self.patch_cv2_resize = patch('cv2.resize')
        self.patch_cv2_imwrite = patch('cv2.imwrite')
        self.patch_np_save = patch('np.save')
        
        # Start patches
        self.mock_has_pytorch3d = self.patch_has_pytorch3d.start()
        self.mock_torch_device = self.patch_torch_device.start()
        self.mock_os_path_exists = self.patch_os_path_exists.start()
        self.mock_load_obj = self.patch_load_obj.start()
        self.mock_meshes = self.patch_meshes.start()
        self.mock_cv2_resize = self.patch_cv2_resize.start()
        self.mock_cv2_imwrite = self.patch_cv2_imwrite.start()
        self.mock_np_save = self.patch_np_save.start()
        
        # Configure mocks
        self.mock_torch_device.return_value = "mock_device"
        self.mock_os_path_exists.return_value = True
        
        # Mock cv2.resize to return the input
        self.mock_cv2_resize.side_effect = lambda img, size: img
        
        # Create sample data
        self.sample_scene_3d_models = [
            {"3d_model_path": "models/cube.obj"},
            {"3d_model_path": "models/sphere.obj"}
        ]
        
        # Create temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.patch_has_pytorch3d.stop()
        self.patch_torch_device.stop()
        self.patch_os_path_exists.stop()
        self.patch_load_obj.stop()
        self.patch_meshes.stop()
        self.patch_cv2_resize.stop()
        self.patch_cv2_imwrite.stop()
        self.patch_np_save.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_pytorch3d(self):
        """Test initialization with PyTorch3D available."""
        # Mock setup_renderer
        with patch.object(SensorGenerator, '_setup_renderer') as mock_setup_renderer:
            # Create sensor generator
            generator = SensorGenerator(device="cuda", image_size=(640, 480), num_cameras=4)
            
            # Check initialization
            self.assertEqual(generator.device, "mock_device")
            self.assertEqual(generator.width, 640)
            self.assertEqual(generator.height, 480)
            self.assertEqual(generator.num_cameras, 4)
            self.assertTrue(generator.initialized)
            
            # Check that renderer was set up
            mock_setup_renderer.assert_called_once()
            
            # Check camera positions
            self.assertEqual(len(generator.camera_positions), 4)
            
            # Check each camera position has required properties
            for camera in generator.camera_positions:
                self.assertIn("position", camera)
                self.assertIn("target", camera)
                self.assertIn("up", camera)
    
    def test_init_without_pytorch3d(self):
        """Test initialization without PyTorch3D available."""
        # Patch to simulate PyTorch3D not available
        with patch('src.phosphobot_construct.sensor_generation.HAS_PYTORCH3D', False):
            # Create sensor generator
            generator = SensorGenerator(device="cuda")
            
            # Check initialization
            self.assertEqual(generator.device, "mock_device")
            self.assertFalse(generator.initialized)
    
    @patch('src.phosphobot_construct.sensor_generation.RasterizationSettings')
    @patch('src.phosphobot_construct.sensor_generation.PointLights')
    def test_setup_renderer(self, mock_lights, mock_raster_settings):
        """Test renderer setup."""
        # Configure mocks
        mock_raster_settings.return_value = "mock_raster_settings"
        mock_lights.return_value = "mock_lights"
        
        # Create sensor generator
        generator = SensorGenerator()
        
        # Check renderer setup
        self.assertEqual(generator.raster_settings, "mock_raster_settings")
        self.assertEqual(generator.lights, "mock_lights")
        
        # Check that rasterization settings were created correctly
        mock_raster_settings.assert_called_once_with(
            image_size=generator.height,
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        
        # Check that lights were created correctly
        mock_lights.assert_called_once_with(
            device="mock_device",
            location=[[0, 0, 3]],
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.3, 0.3, 0.3),)
        )
    
    def test_setup_camera_positions(self):
        """Test camera positions setup."""
        # Test with different numbers of cameras
        for num_cameras in range(1, 6):
            # Create sensor generator
            generator = SensorGenerator(num_cameras=num_cameras)
            
            # Check number of camera positions
            self.assertEqual(len(generator.camera_positions), num_cameras)
            
            # Check each camera position has required properties
            for camera in generator.camera_positions:
                self.assertIn("position", camera)
                self.assertIn("target", camera)
                self.assertIn("up", camera)
                
                # Check position is a tuple of 3 values
                self.assertEqual(len(camera["position"]), 3)
                
                # Check target is a tuple of 3 values
                self.assertEqual(len(camera["target"]), 3)
                
                # Check up vector is a tuple of 3 values
                self.assertEqual(len(camera["up"]), 3)
    
    def test_load_mesh_success(self):
        """Test successful mesh loading."""
        # Configure load_obj mock
        verts = MagicMock()
        faces = MagicMock()
        faces.verts_idx = MagicMock()
        aux = MagicMock()
        self.mock_load_obj.return_value = (verts, faces, aux)
        
        # Create mock mesh
        mock_mesh = MagicMock()
        self.mock_meshes.return_value = mock_mesh
        
        # Create sensor generator
        generator = SensorGenerator()
        
        # Load mesh
        mesh_data = generator.load_mesh("models/cube.obj")
        
        # Check mesh loading
        self.mock_load_obj.assert_called_once_with("models/cube.obj")
        self.mock_meshes.assert_called_once()
        
        # Check returned data
        self.assertIsNotNone(mesh_data)
        self.assertIn("verts", mesh_data)
        self.assertIn("faces", mesh_data)
        self.assertIn("mesh", mesh_data)
        self.assertEqual(mesh_data["mesh"], mock_mesh)
    
    def test_load_mesh_file_not_found(self):
        """Test mesh loading when file is not found."""
        # Configure os.path.exists mock to return False
        with patch('os.path.exists', return_value=False):
            # Create sensor generator
            generator = SensorGenerator()
            
            # Load mesh
            mesh_data = generator.load_mesh("models/nonexistent.obj")
            
            # Check that load_obj was not called
            self.mock_load_obj.assert_not_called()
            
            # Check that None was returned
            self.assertIsNone(mesh_data)
    
    def test_load_mesh_pytorch3d_not_available(self):
        """Test mesh loading when PyTorch3D is not available."""
        # Patch to simulate PyTorch3D not available
        with patch('src.phosphobot_construct.sensor_generation.HAS_PYTORCH3D', False):
            # Create sensor generator
            generator = SensorGenerator()
            
            # Load mesh
            mesh_data = generator.load_mesh("models/cube.obj")
            
            # Check that None was returned
            self.assertIsNone(mesh_data)
    
    def test_load_mesh_exception(self):
        """Test mesh loading when an exception occurs."""
        # Configure load_obj mock to raise an exception
        self.mock_load_obj.side_effect = Exception("Load error")
        
        # Create sensor generator
        generator = SensorGenerator()
        
        # Load mesh
        mesh_data = generator.load_mesh("models/cube.obj")
        
        # Check that load_obj was called
        self.mock_load_obj.assert_called_once()
        
        # Check that None was returned
        self.assertIsNone(mesh_data)
    
    @patch('src.phosphobot_construct.sensor_generation.MeshRenderer')
    @patch('src.phosphobot_construct.sensor_generation.MeshRasterizer')
    @patch('src.phosphobot_construct.sensor_generation.SoftPhongShader')
    @patch('src.phosphobot_construct.sensor_generation.FoVPerspectiveCameras')
    @patch('src.phosphobot_construct.sensor_generation.look_at_view_transform')
    def test_render_rgb_depth_success(
        self, mock_look_at, mock_cameras, mock_shader, mock_rasterizer, mock_renderer
    ):
        """Test successful RGB and depth rendering."""
        # Configure mocks
        mock_look_at.return_value = ("mock_R", "mock_T")
        mock_cameras.return_value = "mock_cameras"
        mock_shader.return_value = "mock_shader"
        mock_rasterizer.return_value = "mock_rasterizer"
        
        # Create mock renderer
        mock_renderer_instance = MagicMock()
        mock_renderer.return_value = mock_renderer_instance
        
        # Configure renderer return value
        rendered_image = torch.ones((1, 240, 320, 4))
        mock_renderer_instance.return_value = rendered_image
        
        # Create mock meshes
        mock_mesh1 = {
            "mesh": MagicMock(),
            "verts": torch.ones((3, 3)),
            "faces": torch.ones((1, 3), dtype=torch.int64)
        }
        mock_mesh1["mesh"].verts_padded.return_value = torch.ones((1, 3, 3))
        mock_mesh1["mesh"].faces_padded.return_value = torch.ones((1, 1, 3), dtype=torch.int64)
        
        mock_mesh2 = {
            "mesh": MagicMock(),
            "verts": torch.ones((3, 3)),
            "faces": torch.ones((1, 3), dtype=torch.int64)
        }
        mock_mesh2["mesh"].verts_padded.return_value = torch.ones((1, 3, 3))
        mock_mesh2["mesh"].faces_padded.return_value = torch.ones((1, 1, 3), dtype=torch.int64)
        
        # Create sensor generator
        generator = SensorGenerator(image_size=(320, 240))
        
        # Render RGB and depth
        rgb, depth = generator.render_rgb_depth([mock_mesh1, mock_mesh2])
        
        # Check camera creation
        mock_look_at.assert_called_once()
        mock_cameras.assert_called_once()
        
        # Check renderer creation
        mock_rasterizer.assert_called_once()
        mock_shader.assert_called_once()
        mock_renderer.assert_called_once()
        
        # Check that renderer was called
        mock_renderer_instance.assert_called_once()
        
        # Check output shapes
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(depth.shape, (240, 320))
    
    def test_render_rgb_depth_no_pytorch3d(self):
        """Test RGB and depth rendering without PyTorch3D."""
        # Patch to simulate PyTorch3D not available
        with patch('src.phosphobot_construct.sensor_generation.HAS_PYTORCH3D', False):
            # Create sensor generator
            generator = SensorGenerator(image_size=(320, 240))
            
            # Render RGB and depth
            rgb, depth = generator.render_rgb_depth([])
            
            # Check that placeholder images were generated
            self.assertEqual(rgb.shape, (240, 320, 3))
            self.assertEqual(depth.shape, (240, 320))
    
    def test_render_rgb_depth_not_initialized(self):
        """Test RGB and depth rendering when not initialized."""
        # Create sensor generator
        generator = SensorGenerator(image_size=(320, 240))
        
        # Set initialized to False
        generator.initialized = False
        
        # Render RGB and depth
        rgb, depth = generator.render_rgb_depth([])
        
        # Check that placeholder images were generated
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(depth.shape, (240, 320))
    
    def test_render_rgb_depth_invalid_camera(self):
        """Test RGB and depth rendering with invalid camera index."""
        # Create sensor generator
        generator = SensorGenerator(image_size=(320, 240), num_cameras=1)
        
        # Render RGB and depth with invalid camera index
        rgb, depth = generator.render_rgb_depth([], camera_idx=1)
        
        # Check that placeholder images were generated
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(depth.shape, (240, 320))
    
    def test_render_rgb_depth_exception(self):
        """Test RGB and depth rendering when an exception occurs."""
        # Create sensor generator
        generator = SensorGenerator(image_size=(320, 240))
        
        # Patch render method to raise an exception
        with patch.object(generator, '_convert_latents_to_mesh', side_effect=Exception("Render error")):
            # Render RGB and depth
            rgb, depth = generator.render_rgb_depth([{"mesh": MagicMock()}])
            
            # Check that placeholder images were generated
            self.assertEqual(rgb.shape, (240, 320, 3))
            self.assertEqual(depth.shape, (240, 320))
    
    def test_generate_placeholder_images(self):
        """Test placeholder image generation."""
        # Create sensor generator
        generator = SensorGenerator(image_size=(320, 240))
        
        # Generate placeholder images
        rgb, depth = generator._generate_placeholder_images()
        
        # Check output shapes
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(depth.shape, (240, 320))
        
        # Check that RGB image has a checkerboard pattern
        self.assertEqual(rgb.dtype, np.uint8)
        
        # Check that depth image is a gradient
        self.assertEqual(depth.dtype, np.uint8)
        self.assertGreaterEqual(depth.min(), 0)
        self.assertLessEqual(depth.max(), 255)
    
    def test_generate_proprioception_data(self):
        """Test proprioception data generation."""
        # Create sensor generator
        generator = SensorGenerator()
        
        # Generate proprioception data
        prop_data = generator._generate_proprioception_data(self.sample_scene_3d_models)
        
        # Check output shape (6 joint positions + 6 velocities + 6 torques)
        self.assertEqual(prop_data.shape, (18,))
        
        # Check that values are within reasonable ranges
        self.assertTrue(np.all(prop_data > -1.0))
        self.assertTrue(np.all(prop_data < 1.0))
    
    @patch.object(SensorGenerator, 'load_mesh')
    @patch.object(SensorGenerator, 'render_rgb_depth')
    def test_generate_sensor_data(self, mock_render, mock_load_mesh):
        """Test sensor data generation."""
        # Configure mocks
        mock_mesh = MagicMock()
        mock_load_mesh.return_value = mock_mesh
        
        mock_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_depth = np.zeros((240, 320), dtype=np.uint8)
        mock_render.return_value = (mock_rgb, mock_depth)
        
        # Create sensor generator
        generator = SensorGenerator(num_cameras=2)
        
        # Generate sensor data
        sensor_data = generator.generate_sensor_data(self.sample_scene_3d_models)
        
        # Check that load_mesh was called for each model
        self.assertEqual(mock_load_mesh.call_count, 2)
        
        # Check that render_rgb_depth was called for each camera
        self.assertEqual(mock_render.call_count, 2)
        
        # Check sensor data structure
        self.assertIn("rgb_images", sensor_data)
        self.assertIn("depth_images", sensor_data)
        self.assertIn("proprioception", sensor_data)
        
        # Check sensor data contents
        self.assertEqual(len(sensor_data["rgb_images"]), 2)
        self.assertEqual(len(sensor_data["depth_images"]), 2)
        self.assertIsInstance(sensor_data["proprioception"], np.ndarray)
    
    @patch.object(SensorGenerator, 'load_mesh')
    @patch.object(SensorGenerator, 'render_rgb_depth')
    def test_generate_sensor_data_missing_models(self, mock_render, mock_load_mesh):
        """Test sensor data generation with missing models."""
        # Configure load_mesh to return None (failed loading)
        mock_load_mesh.return_value = None
        
        # Configure render to return empty images
        mock_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_depth = np.zeros((240, 320), dtype=np.uint8)
        mock_render.return_value = (mock_rgb, mock_depth)
        
        # Create sensor generator
        generator = SensorGenerator(num_cameras=1)
        
        # Generate sensor data
        sensor_data = generator.generate_sensor_data(self.sample_scene_3d_models)
        
        # Check that load_mesh was called for each model
        self.assertEqual(mock_load_mesh.call_count, 2)
        
        # Check that render_rgb_depth was called once with empty meshes list
        mock_render.assert_called_once_with([], camera_idx=0)
        
        # Check sensor data structure
        self.assertIn("rgb_images", sensor_data)
        self.assertIn("depth_images", sensor_data)
        self.assertEqual(len(sensor_data["rgb_images"]), 1)
        self.assertEqual(len(sensor_data["depth_images"]), 1)
    
    def test_save_sensor_data(self):
        """Test saving sensor data to disk."""
        # Create sample sensor data
        rgb_images = [
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.zeros((240, 320, 3), dtype=np.uint8)
        ]
        
        depth_images = [
            np.zeros((240, 320), dtype=np.uint8),
            np.zeros((240, 320), dtype=np.uint8)
        ]
        
        proprioception = np.zeros(18)
        
        sensor_data = {
            "rgb_images": rgb_images,
            "depth_images": depth_images,
            "proprioception": proprioception
        }
        
        # Create sensor generator
        generator = SensorGenerator()
        
        # Save sensor data
        output_dir = os.path.join(self.temp_dir, "sensor_data")
        file_paths = generator.save_sensor_data(sensor_data, output_dir)
        
        # Check that directory was created
        self.assertTrue(os.path.exists(output_dir))
        
        # Check that cv2.imwrite was called for each image
        self.assertEqual(self.mock_cv2_imwrite.call_count, 4)  # 2 RGB + 2 depth
        
        # Check that np.save was called for proprioception
        self.mock_np_save.assert_called_once()
        
        # Check returned file paths
        self.assertIn("rgb_paths", file_paths)
        self.assertIn("depth_paths", file_paths)
        self.assertIn("proprioception_path", file_paths)
        self.assertEqual(len(file_paths["rgb_paths"]), 2)
        self.assertEqual(len(file_paths["depth_paths"]), 2)


@patch('src.phosphobot_construct.sensor_generation.SensorGenerator')
def test_generate_training_data(mock_generator_class):
    """Test the generate_training_data function."""
    # Configure mocks
    mock_generator = MagicMock()
    mock_generator_class.return_value = mock_generator
    
    # Mock sensor data and file paths
    sensor_data = {"rgb_images": [], "depth_images": [], "proprioception": np.zeros(18)}
    file_paths = {"rgb_paths": [], "depth_paths": [], "proprioception_path": "test.npy"}
    mock_generator.generate_sensor_data.return_value = sensor_data
    mock_generator.save_sensor_data.return_value = file_paths
    
    # Call function
    with tempfile.TemporaryDirectory() as temp_dir:
        result = generate_training_data(
            scene_3d_models=[{"3d_model_path": "test.obj"}],
            output_dir=temp_dir,
            num_cameras=3,
            image_size=(640, 480)
        )
        
        # Check generator initialization
        mock_generator_class.assert_called_once_with(
            device="cuda" if torch.cuda.is_available() else "cpu",
            image_size=(640, 480),
            num_cameras=3
        )
        
        # Check that generate_sensor_data was called
        mock_generator.generate_sensor_data.assert_called_once_with([{"3d_model_path": "test.obj"}])
        
        # Check that save_sensor_data was called
        mock_generator.save_sensor_data.assert_called_once_with(sensor_data, temp_dir)
        
        # Check result structure
        self.assertIn("sensor_data", result)
        self.assertIn("file_paths", result)
        self.assertIn("output_dir", result)
        self.assertIn("num_samples", result)
        self.assertEqual(result["sensor_data"], sensor_data)
        self.assertEqual(result["file_paths"], file_paths)
        self.assertEqual(result["output_dir"], temp_dir)
        self.assertEqual(result["num_samples"], 1)


if __name__ == "__main__":
    unittest.main()
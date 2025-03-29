"""
Unit tests for the phosphobot_construct.text_to_3d module.
"""

import unittest
import os
import json
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock torch and diffusers since they're optional dependencies
sys.modules['torch'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers.ShapEPipeline'] = MagicMock()
sys.modules['trimesh'] = MagicMock()

# Import the module under test with mocked dependencies
from src.phosphobot_construct.text_to_3d import TextTo3DConverter, convert_scenarios_to_3d


class TestTextTo3DConverter(unittest.TestCase):
    """Tests for the TextTo3DConverter class."""
    
    def setUp(self):
        """Setup for tests."""
        # Create patchers
        self.patch_has_shape_e = patch('src.phosphobot_construct.text_to_3d.HAS_SHAPE_E', True)
        self.patch_torch_device = patch('torch.device')
        self.patch_os_path_exists = patch('os.path.exists')
        self.patch_os_makedirs = patch('os.makedirs')
        self.patch_shape_e_pipeline = patch('src.phosphobot_construct.text_to_3d.ShapEPipeline')
        self.patch_trimesh = patch('src.phosphobot_construct.text_to_3d.trimesh')
        
        # Start patches
        self.mock_has_shape_e = self.patch_has_shape_e.start()
        self.mock_torch_device = self.patch_torch_device.start()
        self.mock_os_path_exists = self.patch_os_path_exists.start()
        self.mock_os_makedirs = self.patch_os_makedirs.start()
        self.mock_shape_e_pipeline = self.patch_shape_e_pipeline.start()
        self.mock_trimesh = self.patch_trimesh.start()
        
        # Configure mocks
        self.mock_torch_device.return_value = "mock_device"
        self.mock_os_path_exists.return_value = True
        
        # Mock ShapEPipeline
        self.mock_pipeline = MagicMock()
        self.mock_shape_e_pipeline.from_pretrained.return_value = self.mock_pipeline
        
        # Mock Trimesh
        self.mock_mesh = MagicMock()
        self.mock_trimesh.Trimesh.return_value = self.mock_mesh
        
        # Sample 3D model data
        self.sample_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        self.sample_faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
        ])
        
        # Sample scenario
        self.sample_scenario = {
            "id": "scenario_0001",
            "description": "A tabletop scene with three distinct objects for robot manipulation.",
            "objects": [
                {
                    "name": "red_cube",
                    "shape": "cube",
                    "color": "red",
                    "material": "plastic",
                    "size": "small"
                },
                {
                    "name": "blue_sphere",
                    "shape": "sphere",
                    "color": "blue",
                    "material": "rubber",
                    "size": "medium"
                }
            ]
        }
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.patch_has_shape_e.stop()
        self.patch_torch_device.stop()
        self.patch_os_path_exists.stop()
        self.patch_os_makedirs.stop()
        self.patch_shape_e_pipeline.stop()
        self.patch_trimesh.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_init_with_shape_e(self):
        """Test initialization with Shap-E available."""
        # Create converter
        converter = TextTo3DConverter(model_name="openai/shap-e", device="cuda")
        
        # Check initialization
        self.assertEqual(converter.device, "mock_device")
        self.assertEqual(converter.model_name, "openai/shap-e")
        self.assertEqual(converter.pipeline, self.mock_pipeline)
        
        # Check that pipeline was loaded correctly
        self.mock_shape_e_pipeline.from_pretrained.assert_called_once_with(
            "openai/shap-e", torch_dtype=torch.float16
        )
        self.mock_pipeline.to.assert_called_once_with("mock_device")
    
    def test_init_without_shape_e(self):
        """Test initialization without Shap-E available."""
        # Patch to simulate Shap-E not available
        with patch('src.phosphobot_construct.text_to_3d.HAS_SHAPE_E', False):
            # Create converter
            converter = TextTo3DConverter(device="cuda")
            
            # Check initialization
            self.assertEqual(converter.device, "mock_device")
            self.assertIsNone(converter.pipeline)
            
            # Check that pipeline was not loaded
            self.mock_shape_e_pipeline.from_pretrained.assert_not_called()
    
    def test_init_load_failure(self):
        """Test initialization when loading the model fails."""
        # Configure pipeline to raise an exception
        self.mock_shape_e_pipeline.from_pretrained.side_effect = Exception("Load error")
        
        # Create converter
        converter = TextTo3DConverter(device="cuda")
        
        # Check initialization
        self.assertEqual(converter.device, "mock_device")
        self.assertIsNone(converter.pipeline)
        
        # Check that pipeline was attempted to be loaded
        self.mock_shape_e_pipeline.from_pretrained.assert_called_once()
    
    def test_text_to_3d_conversion_success(self):
        """Test successful text-to-3D conversion."""
        # Configure mocks
        mock_images = [MagicMock()]
        self.mock_pipeline.return_value.images = mock_images
        
        # Create converter and patch _convert_latents_to_mesh
        converter = TextTo3DConverter(device="cuda")
        with patch.object(
            converter, '_convert_latents_to_mesh', 
            return_value=(self.sample_vertices, self.sample_faces)
        ):
            # Perform conversion
            output_path = os.path.join(self.test_dir, "test_model.glb")
            result_path = converter.text_to_3d_conversion(
                description="a red cube",
                output_path=output_path
            )
            
            # Check pipeline call
            self.mock_pipeline.assert_called_once_with(
                prompt="a red cube",
                guidance_scale=15.0,
                num_inference_steps=64,
                frame_size=128
            )
            
            # Check mesh creation and export
            self.mock_trimesh.Trimesh.assert_called_once_with(
                vertices=self.sample_vertices, faces=self.sample_faces
            )
            self.mock_mesh.export.assert_called_once_with(output_path)
            
            # Check result
            self.assertEqual(result_path, output_path)
    
    def test_text_to_3d_conversion_no_shape_e(self):
        """Test text-to-3D conversion without Shap-E."""
        # Create converter without pipeline
        converter = TextTo3DConverter(device="cuda")
        converter.pipeline = None
        
        # Perform conversion
        result_path = converter.text_to_3d_conversion(description="a red cube")
        
        # Check that no mesh was created
        self.mock_trimesh.Trimesh.assert_not_called()
        self.mock_mesh.export.assert_not_called()
        
        # Check result
        self.assertIsNone(result_path)
    
    def test_text_to_3d_conversion_exception(self):
        """Test text-to-3D conversion when an exception occurs."""
        # Configure pipeline to raise an exception
        self.mock_pipeline.side_effect = Exception("Conversion error")
        
        # Create converter
        converter = TextTo3DConverter(device="cuda")
        
        # Perform conversion
        result_path = converter.text_to_3d_conversion(description="a red cube")
        
        # Check that pipeline was called
        self.mock_pipeline.assert_called_once()
        
        # Check that no mesh was created
        self.mock_trimesh.Trimesh.assert_not_called()
        self.mock_mesh.export.assert_not_called()
        
        # Check result
        self.assertIsNone(result_path)
    
    def test_convert_latents_to_mesh(self):
        """Test conversion of latents to mesh."""
        # Create converter
        converter = TextTo3DConverter(device="cuda")
        
        # Mock pipeline's internal methods for conversion
        converter.pipeline = None
        
        # Call method (should use the placeholder implementation)
        vertices, faces = converter._convert_latents_to_mesh(MagicMock())
        
        # Check returned shapes
        self.assertEqual(vertices.shape, (8, 3))  # 8 vertices for a cube
        self.assertEqual(faces.shape, (12, 3))    # 12 faces for a cube
    
    def test_convert_latents_to_mesh_no_pipeline(self):
        """Test latent conversion without pipeline."""
        # Create converter without pipeline
        converter = TextTo3DConverter(device="cuda")
        converter.pipeline = None
        
        # Call method and expect ValueError
        with self.assertRaises(ValueError):
            converter._convert_latents_to_mesh(MagicMock())
    
    @patch.object(TextTo3DConverter, 'text_to_3d_conversion')
    def test_batch_convert_scenario(self, mock_convert):
        """Test batch conversion of a scenario's objects."""
        # Configure mock to return different paths for each object
        mock_convert.side_effect = [
            os.path.join(self.test_dir, "red_cube.glb"),
            os.path.join(self.test_dir, "blue_sphere.glb")
        ]
        
        # Create converter
        converter = TextTo3DConverter(device="cuda")
        
        # Batch convert scenario
        output_dir = os.path.join(self.test_dir, "models")
        object_models = converter.batch_convert_scenario(
            scenario=self.sample_scenario,
            output_dir=output_dir
        )
        
        # Check that makedirs was called
        self.mock_os_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        
        # Check that text_to_3d_conversion was called for each object
        self.assertEqual(mock_convert.call_count, 2)
        
        # Check conversion of first object
        first_call = mock_convert.call_args_list[0]
        self.assertIn("red cube", first_call[1]["description"].lower())
        self.assertEqual(
            first_call[1]["output_path"],
            os.path.join(output_dir, f"{self.sample_scenario['id']}_red_cube.glb")
        )
        
        # Check conversion of second object
        second_call = mock_convert.call_args_list[1]
        self.assertIn("blue sphere", second_call[1]["description"].lower())
        self.assertEqual(
            second_call[1]["output_path"],
            os.path.join(output_dir, f"{self.sample_scenario['id']}_blue_sphere.glb")
        )
        
        # Check returned model paths
        self.assertEqual(len(object_models), 2)
        self.assertEqual(
            object_models["red_cube"],
            os.path.join(self.test_dir, "red_cube.glb")
        )
        self.assertEqual(
            object_models["blue_sphere"],
            os.path.join(self.test_dir, "blue_sphere.glb")
        )
    
    @patch.object(TextTo3DConverter, 'text_to_3d_conversion')
    def test_batch_convert_scenario_conversion_failure(self, mock_convert):
        """Test batch conversion when an object conversion fails."""
        # Configure mock to succeed for first object and fail for second
        mock_convert.side_effect = [
            os.path.join(self.test_dir, "red_cube.glb"),
            None  # Failure
        ]
        
        # Create converter
        converter = TextTo3DConverter(device="cuda")
        
        # Batch convert scenario
        output_dir = os.path.join(self.test_dir, "models")
        object_models = converter.batch_convert_scenario(
            scenario=self.sample_scenario,
            output_dir=output_dir
        )
        
        # Check that text_to_3d_conversion was called for each object
        self.assertEqual(mock_convert.call_count, 2)
        
        # Check returned model paths (only successful conversion)
        self.assertEqual(len(object_models), 1)
        self.assertEqual(
            object_models["red_cube"],
            os.path.join(self.test_dir, "red_cube.glb")
        )
        self.assertNotIn("blue_sphere", object_models)


@patch('src.phosphobot_construct.text_to_3d.TextTo3DConverter')
def test_convert_scenarios_to_3d(mock_converter_class):
    """Test the convert_scenarios_to_3d function."""
    # Configure mocks
    mock_converter = MagicMock()
    mock_converter_class.return_value = mock_converter
    
    # Mock batch_convert_scenario to return object models
    mock_converter.batch_convert_scenario.side_effect = [
        {"red_cube": "models/red_cube.glb"},
        {"blue_sphere": "models/blue_sphere.glb"}
    ]
    
    # Create sample scenarios
    scenarios = [
        {"id": "scenario_0001", "objects": [{"name": "red_cube"}]},
        {"id": "scenario_0002", "objects": [{"name": "blue_sphere"}]}
    ]
    
    # Load scenarios from directory
    with patch('os.listdir') as mock_listdir, \
         patch('builtins.open', create=True) as mock_open:
        
        # Configure mock_listdir to list scenario files
        mock_listdir.return_value = ["scenario_0001.json", "scenario_0002.json"]
        
        # Configure mock_open context manager
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file
        mock_file.read.side_effect = [
            json.dumps(scenarios[0]),
            json.dumps(scenarios[1])
        ]
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as scenarios_dir, \
             tempfile.TemporaryDirectory() as output_dir:
            
            # Convert scenarios to 3D
            convert_scenarios_to_3d(
                scenarios_dir=scenarios_dir,
                output_dir=output_dir,
                device="cpu"
            )
            
            # Check converter initialization
            mock_converter_class.assert_called_once_with(device="cpu")
            
            # Check that batch_convert_scenario was called for each scenario
            self.assertEqual(mock_converter.batch_convert_scenario.call_count, 2)
            
            # Check conversion of first scenario
            first_call = mock_converter.batch_convert_scenario.call_args_list[0]
            self.assertEqual(first_call[0][0], scenarios[0])
            self.assertIn(output_dir, first_call[0][1])
            
            # Check conversion of second scenario
            second_call = mock_converter.batch_convert_scenario.call_args_list[1]
            self.assertEqual(second_call[0][0], scenarios[1])
            self.assertIn(output_dir, second_call[0][1])


if __name__ == "__main__":
    unittest.main()
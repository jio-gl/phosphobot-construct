"""
Unit tests for the phosphobot_construct.perception module.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import torch

# Add parent directory to path to make imports work in testing
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock torch and related imports before importing perception
sys.modules['torch'] = MagicMock()
# We keep these mocks for clip and segment_anything if you don't want them loaded:
sys.modules['clip'] = MagicMock()
sys.modules['segment_anything'] = MagicMock()
sys.modules['segment_anything.sam_model_registry'] = MagicMock()
sys.modules['segment_anything.SamPredictor'] = MagicMock()

from src.phosphobot_construct.perception import SceneUnderstanding, perception_pipeline


class TestSceneUnderstanding(unittest.TestCase):
    """Tests for the SceneUnderstanding class."""
    
    def setUp(self):
        """Setup for tests."""
        # Create sample RGB image (100x100 with a red square)
        self.rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.rgb_image[30:70, 30:70, 0] = 255  # Red square
        
        # Create sample depth image
        self.depth_image = np.zeros((100, 100), dtype=np.float32)
        self.depth_image[30:70, 30:70] = 0.5  # Object at depth 0.5
        
        # Create patchers for external dependencies
        self.patch_clip = patch('src.phosphobot_construct.perception.HAS_CLIP', True)
        self.patch_sam = patch('src.phosphobot_construct.perception.HAS_SAM', True)
        
        # Start patches
        self.patch_clip.start()
        self.patch_sam.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.patch_clip.stop()
        self.patch_sam.stop()
    
    @patch('src.phosphobot_construct.perception.torch.device')
    @patch('src.phosphobot_construct.perception.clip.load')
    @patch('src.phosphobot_construct.perception.sam_model_registry')
    @patch('src.phosphobot_construct.perception.SamPredictor')
    def test_init(self, mock_sam_predictor, mock_sam_registry, mock_clip_load, mock_torch_device):
        # 1. Mock torch.device
        mock_torch_device.return_value = "mock_device"

        # 2. Mock the clip.load return
        mock_clip_model = MagicMock()
        mock_clip_preprocess = MagicMock()
        mock_clip_load.return_value = (mock_clip_model, mock_clip_preprocess)

        # 3. Make sam_model_registry["vit_h"] return a callable "sam_ctor"
        mock_sam_ctor = MagicMock(name="sam_ctor")
        mock_sam_registry.__getitem__.return_value = mock_sam_ctor

        # 4. Make calling sam_ctor(...) return a "sam_model"
        mock_sam_model = MagicMock(name="sam_model")
        mock_sam_ctor.return_value = mock_sam_model

        # 5. When we do sam_model.to("mock_device"), have it return sam_model again
        mock_sam_model.to.return_value = mock_sam_model

        # 6. SamPredictor(...) should produce a final SamPredictor instance
        mock_sam_predictor_instance = MagicMock(name="sam_predictor_instance")
        mock_sam_predictor.return_value = mock_sam_predictor_instance

        # Now do the actual init
        scene = SceneUnderstanding(device="cpu")

        # Assertions
        self.assertEqual(scene.device, "mock_device")
        self.assertEqual(scene.clip_model, mock_clip_model)
        self.assertEqual(scene.clip_preprocess, mock_clip_preprocess)
        self.assertEqual(scene.sam_predictor, mock_sam_predictor_instance)

        mock_torch_device.assert_called_once_with("cpu")
        mock_clip_load.assert_called_once()

        # The code does: sam = sam_model_registry["vit_h"](checkpoint=...)
        # so check getitem:
        mock_sam_registry.__getitem__.assert_called_once_with("vit_h")
        # then sam_ctor(checkpoint=...), so check that it was indeed called
        mock_sam_ctor.assert_called_once_with(checkpoint="sam_vit_h_4b8939.pth")
        # then sam_model.to("mock_device") was called
        mock_sam_model.to.assert_called_once_with("mock_device")
        # finally SamPredictor(...) was called with the same object
        mock_sam_predictor.assert_called_once_with(mock_sam_model)

    @patch('src.phosphobot_construct.perception.torch.device')
    def test_init_no_models(self, mock_torch_device):
        """Test initialization when models are not available."""
        # Set up mock to return a specific value
        mock_torch_device.return_value = "mock_device"
        
        # Setup patches to simulate unavailable models
        with patch('src.phosphobot_construct.perception.HAS_CLIP', False):
            with patch('src.phosphobot_construct.perception.HAS_SAM', False):
                # Create scene understanding instance
                scene = SceneUnderstanding(device="cpu")
                
                # Check initialization
                self.assertEqual(scene.device, "mock_device")
                self.assertIsNone(scene.clip_model)
                self.assertIsNone(scene.clip_preprocess)
                self.assertIsNone(scene.sam_predictor)
    
    @patch('src.phosphobot_construct.perception.torch.device')
    def test_segment_objects_with_sam(self, mock_torch_device):
        """Test object segmentation with SAM."""
        # Create mock SAM predictor
        mock_predictor = MagicMock()
        mock_mask_data = {
            "segmentation": np.ones((100, 100), dtype=np.uint8)
        }
        mock_predictor.generate.return_value = [mock_mask_data]
        
        # Create scene understanding with mock SAM
        scene = SceneUnderstanding(device="cpu")
        scene.sam_predictor = mock_predictor
        
        # Segment objects
        objects = scene.segment_objects(self.rgb_image)
        
        # Check that SAM was called
        mock_predictor.set_image.assert_called_once()
        mock_predictor.generate.assert_called_once()
        
        # Check segmentation results
        self.assertEqual(len(objects), 1)
        self.assertIn("mask", objects[0])
        self.assertIn("bbox", objects[0])
        self.assertIn("area", objects[0])
        self.assertIn("patch", objects[0])
    
    @patch('src.phosphobot_construct.perception.torch.device')
    def test_segment_objects_fallback(self, mock_torch_device):
        """Test fallback object segmentation when SAM is not available."""
        # Create scene understanding without SAM
        scene = SceneUnderstanding(device="cpu")
        scene.sam_predictor = None
        
        # Segment objects using fallback
        objects = scene.segment_objects(self.rgb_image)
        
        # Since the fallback method uses color-based segmentation, the red square should be detected
        self.assertGreater(len(objects), 0)
    
    @patch('src.phosphobot_construct.perception.torch.device')
    @patch('src.phosphobot_construct.perception.torch.no_grad')
    @patch('src.phosphobot_construct.perception.clip.tokenize')
    def test_classify_objects_with_clip(self, mock_tokenize, mock_no_grad, mock_torch_device):
        """Test object classification with CLIP."""
        # Setup mocks
        mock_no_grad_context = MagicMock()
        mock_no_grad.return_value = mock_no_grad_context
        mock_no_grad_context.__enter__ = MagicMock()
        mock_no_grad_context.__exit__ = MagicMock()
        
        mock_tokens = MagicMock()
        mock_tokenize.return_value = mock_tokens
        
        # Create mock CLIP model and processor
        mock_clip_model = MagicMock()
        mock_clip_preprocess = MagicMock()
        
        # Setup mock for tensor operations
        mock_text_features = MagicMock()
        mock_text_features.norm.return_value = MagicMock()
        mock_text_features.__truediv__ = MagicMock(return_value=mock_text_features)
        
        mock_image_features = MagicMock()
        mock_image_features.norm.return_value = MagicMock()
        mock_image_features.__truediv__ = MagicMock(return_value=mock_image_features)
        
        # Matrix multiplication result
        mock_similarity = MagicMock()
        mock_values = MagicMock()
        mock_indices = MagicMock()
        mock_values[0].topk.return_value = (mock_values, mock_indices)
        mock_similarity.softmax.return_value = mock_values
        mock_image_features.__matmul__ = MagicMock(return_value=mock_similarity)
        
        # Configure CLIP mock behavior
        mock_clip_model.encode_text.return_value = mock_text_features
        mock_clip_model.encode_image.return_value = mock_image_features
        
        # Create sample objects with area and patch
        objects = [{
            "patch": np.ones((50, 50, 3), dtype=np.uint8),
            "area": 2500
        }]
        
        # Mock fromarray -> We could patch PIL.Image.fromarray if needed
        mock_processed_image = MagicMock()
        mock_processed_image.unsqueeze.return_value = mock_processed_image
        mock_processed_image.to.return_value = mock_processed_image
        
        mock_clip_preprocess.return_value = mock_processed_image
        
        # Create scene understanding with mock CLIP
        scene = SceneUnderstanding(device="cpu")
        scene.clip_model = mock_clip_model
        scene.clip_preprocess = mock_clip_preprocess
        
        # Classify objects
        classified_objects = scene.classify_objects(objects)
        
        # Check classification results
        self.assertEqual(len(classified_objects), 1)
        self.assertIn("class", classified_objects[0])
        self.assertIn("confidence", classified_objects[0])
    
    @patch('src.phosphobot_construct.perception.torch.device')
    def test_classify_objects_fallback(self, mock_torch_device):
        """Test fallback object classification when CLIP is not available."""
        # Create sample objects with area
        objects = [{
            "patch": np.ones((50, 50, 3), dtype=np.uint8) * 255,  # White patch
            "area": 2500
        }]
        
        # Create scene understanding without CLIP
        scene = SceneUnderstanding(device="cpu")
        scene.clip_model = None
        
        # Classify objects using fallback
        classified_objects = scene.classify_objects(objects)
        
        # Check classification results
        self.assertEqual(len(classified_objects), 1)
        self.assertIn("class", classified_objects[0])
        self.assertIn("confidence", classified_objects[0])
    
    @patch('src.phosphobot_construct.perception.torch.device')
    def test_estimate_3d_positions(self, mock_torch_device):
        """Test 3D position estimation."""
        # Create sample objects
        objects = [{
            "mask": np.ones((100, 100), dtype=np.uint8),
            "bbox": [30, 30, 70, 70]  # [x_min, y_min, x_max, y_max]
        }]
        
        # Create scene understanding
        scene = SceneUnderstanding(device="cpu")
        
        # Estimate 3D positions with depth
        objects_with_depth = scene.estimate_3d_positions(objects, self.depth_image)
        
        # Check position estimation results
        self.assertEqual(len(objects_with_depth), 1)
        self.assertIn("position_3d", objects_with_depth[0])
        self.assertIn("x", objects_with_depth[0]["position_3d"])
        self.assertIn("y", objects_with_depth[0]["position_3d"])
        self.assertIn("z", objects_with_depth[0]["position_3d"])
        
        # Check that depth was used
        self.assertAlmostEqual(objects_with_depth[0]["position_3d"]["z"], 0.5, places=1)
        
        # Estimate 3D positions without depth
        objects_without_depth = scene.estimate_3d_positions(objects, None)
        
        # Check position estimation results
        self.assertEqual(len(objects_without_depth), 1)
        self.assertIn("position_3d", objects_without_depth[0])
    
    @patch('src.phosphobot_construct.perception.SceneUnderstanding')
    def test_perception_pipeline_full(self, mock_scene_class):
        """Test the full perception pipeline."""
        # Setup mock scene understanding
        mock_scene = MagicMock()
        mock_scene_class.return_value = mock_scene
        
        # Configure mock behavior
        mock_scene.segment_objects.return_value = [{"id": 0}]
        mock_scene.classify_objects.return_value = [{"id": 0, "class": "cube"}]
        mock_scene.estimate_3d_positions.return_value = [
            {"id": 0, "class": "cube", "position_3d": {"x": 0.1, "y": 0.2, "z": 0.3}}
        ]
        
        # Run perception pipeline
        scene_3d = perception_pipeline(
            rgb_image=self.rgb_image,
            depth_image=self.depth_image,
            proprioception=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        )
        
        # Check pipeline results
        self.assertIn("objects", scene_3d)
        self.assertIn("workspace", scene_3d)
        self.assertIn("robot_state", scene_3d)
        
        # Check that each stage was called
        mock_scene.segment_objects.assert_called_once_with(self.rgb_image)
        mock_scene.classify_objects.assert_called_once()
        mock_scene.estimate_3d_positions.assert_called_once()
    
    @patch('src.phosphobot_construct.perception.SceneUnderstanding')
    def test_perception_pipeline_no_proprioception(self, mock_scene_class):
        """Test perception pipeline without proprioception data."""
        # Setup mock scene understanding
        mock_scene = MagicMock()
        mock_scene_class.return_value = mock_scene
        
        # Configure mock behavior
        mock_scene.segment_objects.return_value = [{"id": 0}]
        mock_scene.classify_objects.return_value = [{"id": 0, "class": "cube"}]
        mock_scene.estimate_3d_positions.return_value = [
            {"id": 0, "class": "cube", "position_3d": {"x": 0.1, "y": 0.2, "z": 0.3}}
        ]
        
        # Run perception pipeline without proprioception
        scene_3d = perception_pipeline(
            rgb_image=self.rgb_image,
            depth_image=self.depth_image,
            proprioception=None
        )
        
        # Check pipeline results
        self.assertIn("objects", scene_3d)
        self.assertIn("workspace", scene_3d)
        self.assertNotIn("robot_state", scene_3d)


class TestSegmentationMethods(unittest.TestCase):
    """Tests for specific segmentation methods."""
    
    def setUp(self):
        """Setup for tests."""
        # Create sample images with distinct objects
        self.rgb_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Red square
        self.rgb_image[30:70, 30:70, 0] = 255
        
        # Green square
        self.rgb_image[30:70, 130:170, 1] = 255
        
        # Blue square
        self.rgb_image[130:170, 30:70, 2] = 255
        
        # Create patchers for external dependencies
        self.patch_clip = patch('src.phosphobot_construct.perception.HAS_CLIP', False)
        self.patch_sam = patch('src.phosphobot_construct.perception.HAS_SAM', False)
        
        # Start patches
        self.patch_clip.start()
        self.patch_sam.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.patch_clip.stop()
        self.patch_sam.stop()
    
    @patch('src.phosphobot_construct.perception.torch.device')
    def test_fallback_segmentation(self, mock_torch_device):
        """Test fallback color-based segmentation."""
        # Create scene understanding
        scene = SceneUnderstanding(device="cpu")
        
        # Force using fallback segmentation
        objects = scene._fallback_segmentation(self.rgb_image)
        
        # Check segmentation results (should find 3 colored squares)
        self.assertEqual(len(objects), 3)
        
        # Check that each object has required fields
        for obj in objects:
            self.assertIn("id", obj)
            self.assertIn("mask", obj)
            self.assertIn("bbox", obj)
            self.assertIn("area", obj)
            self.assertIn("patch", obj)
            self.assertIn("color", obj)
            
            # Check mask dimensions
            self.assertEqual(obj["mask"].shape, (200, 200))
            
            # Check that area is reasonable (should be ~40x40=1600)
            self.assertGreater(obj["area"], 1000)
            self.assertLess(obj["area"], 2500)
    
    @patch('src.phosphobot_construct.perception.torch.device')
    def test_fallback_classification(self, mock_torch_device):
        """Test fallback classification based on color and size."""
        # Create objects with color information
        objects = [
            {"color": "red", "area": 1600, "patch": np.ones((40, 40, 3), dtype=np.uint8)},
            {"color": "green", "area": 1600, "patch": np.ones((40, 40, 3), dtype=np.uint8)},
            {"color": "blue", "area": 1600, "patch": np.ones((40, 40, 3), dtype=np.uint8)}
        ]
        
        # Create scene understanding
        scene = SceneUnderstanding(device="cpu")
        
        # Force using fallback classification
        classified_objects = scene._fallback_classification(objects)
        
        # Check classification results
        self.assertEqual(len(classified_objects), 3)
        
        # Check that each object has been classified
        for i, obj in enumerate(classified_objects):
            self.assertIn("class", obj)
            self.assertIn("confidence", obj)
            
            # Check that color information was used
            expected_colors = ["red", "green", "blue"]
            self.assertIn(expected_colors[i], obj["class"].lower())


if __name__ == "__main__":
    unittest.main()

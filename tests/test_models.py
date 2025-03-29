"""
Unit tests for the phosphobot_construct.models module.
"""

import unittest
import numpy as np
import torch
import os
import tempfile
from typing import Dict, Any

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.phosphobot_construct.models import PhosphoConstructModel

class TestPhosphoConstructModel(unittest.TestCase):
    """Tests for the PhosphoConstructModel class."""
    
    def setUp(self):
        """Setup for tests, create model instance."""
        # Use CPU for testing
        self.model = PhosphoConstructModel(device="cpu")
        
        # Create dummy inputs
        self.dummy_inputs = {
            "state": np.random.rand(6).astype(np.float32),  # 6 DoF state
            "images": np.random.randint(0, 255, size=(3, 240, 320, 3), dtype=np.uint8)  # 3 RGB images
        }
    
    def test_init(self):
        """Test model initialization."""
        # Check model attributes
        self.assertIsNotNone(self.model.vision_encoder)
        self.assertIsNotNone(self.model.state_encoder)
        self.assertIsNotNone(self.model.transformer)
        self.assertIsNotNone(self.model.action_head)
        
        # Check device is set correctly
        self.assertEqual(self.model.device, torch.device("cpu"))
    
    def test_prepare_inputs(self):
        """Test input preparation."""
        processed_inputs = self.model.prepare_inputs(self.dummy_inputs)
        
        # Check that inputs were converted to tensors
        self.assertIsInstance(processed_inputs["state"], torch.Tensor)
        self.assertIsInstance(processed_inputs["images"], torch.Tensor)
        
        # Check that shapes are correct
        self.assertEqual(processed_inputs["state"].shape, torch.Size([6]))
        self.assertEqual(processed_inputs["images"].shape, torch.Size([3, 3, 240, 320]))
        
        # Check that values were normalized
        self.assertTrue(torch.all(processed_inputs["images"] <= 1.0))
        self.assertTrue(torch.all(processed_inputs["images"] >= 0.0))
    
    def test_forward(self):
        """Test forward pass."""
        # Prepare inputs
        processed_inputs = self.model.prepare_inputs(self.dummy_inputs)
        
        # Run forward pass
        actions = self.model.forward(processed_inputs)
        
        # Check output shape (should match the DoF of the robot)
        self.assertEqual(actions.shape, torch.Size([6]))
    
    def test_sample_actions(self):
        """Test action sampling."""
        # Sample actions
        actions = self.model.sample_actions(self.dummy_inputs)
        
        # Check output type and shape
        self.assertIsInstance(actions, np.ndarray)
        self.assertEqual(actions.shape, (10, 6))  # 10 steps, 6 DoF
    
    def test_save_load_model(self):
        """Test saving and loading model weights."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            model_path = tmp_file.name
            
            # Save model
            self.model.save_model(model_path)
            
            # Check that file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Create new model and load weights
            loaded_model = PhosphoConstructModel(model_path=model_path, device="cpu")
            
            # Sample actions from both models
            original_actions = self.model.sample_actions(self.dummy_inputs)
            loaded_actions = loaded_model.sample_actions(self.dummy_inputs)
            
            # Check that actions are the same
            np.testing.assert_allclose(original_actions, loaded_actions, rtol=1e-5)
    
    def test_call_method(self):
        """Test the __call__ method."""
        # Call model directly
        actions = self.model(self.dummy_inputs)
        
        # Check output type and shape
        self.assertIsInstance(actions, np.ndarray)
        self.assertEqual(actions.shape, (10, 6))  # 10 steps, 6 DoF


if __name__ == "__main__":
    unittest.main()
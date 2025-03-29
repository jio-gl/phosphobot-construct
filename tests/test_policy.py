"""
Unit tests for the phosphobot_construct.policy module.
"""

import unittest
import os
import numpy as np
import torch
from unittest.mock import patch, MagicMock, PropertyMock

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock stable-baselines3 since it's an optional dependency
sys.modules['stable_baselines3'] = MagicMock()
sys.modules['stable_baselines3.common'] = MagicMock()
sys.modules['stable_baselines3.common.vec_env'] = MagicMock()
sys.modules['stable_baselines3.PPO'] = MagicMock()

from src.phosphobot_construct.policy import PolicyExecutor, execute_policy


class TestPolicyExecutor(unittest.TestCase):
    """Tests for the PolicyExecutor class."""
    
    def setUp(self):
        """Setup for tests."""
        # Create sample data
        self.sample_scene = {
            "objects": [
                {"class": "a red cube", "position_3d": {"x": 0.1, "y": 0.2, "z": 0.3}},
                {"class": "a blue sphere", "position_3d": {"x": -0.2, "y": 0.1, "z": 0.1}}
            ],
            "robot_state": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        self.sample_goal = {
            "objects": [
                {"class": "a red cube", "goal_position": {"x": 0.1, "y": 0.2, "z": 0.5}},
                {"class": "a blue sphere", "goal_position": {"x": -0.2, "y": 0.1, "z": 0.1}}
            ]
        }
        
        # Create patchers
        self.patch_torch_device = patch('torch.device')
        self.patch_sb3 = patch('src.phosphobot_construct.policy.HAS_SB3', True)
        self.patch_os_path_exists = patch('os.path.exists')
        self.patch_ppo_load = patch('src.phosphobot_construct.policy.PPO.load')
        self.patch_vec_normalize_load = patch('src.phosphobot_construct.policy.VecNormalize.load')
        
        # Start patches
        self.mock_torch_device = self.patch_torch_device.start()
        self.patch_sb3.start()
        self.mock_os_path_exists = self.patch_os_path_exists.start()
        self.mock_ppo_load = self.patch_ppo_load.start()
        self.mock_vec_normalize_load = self.patch_vec_normalize_load.start()
        
        # Configure mocks
        self.mock_torch_device.return_value = "mock_device"
        self.mock_os_path_exists.return_value = True
        
        # Mock PPO model
        self.mock_model = MagicMock()
        self.mock_ppo_load.return_value = self.mock_model
        
        # Mock VecNormalize
        self.mock_env = MagicMock()
        self.mock_vec_normalize_load.return_value = self.mock_env
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.patch_torch_device.stop()
        self.patch_sb3.stop()
        self.patch_os_path_exists.stop()
        self.patch_ppo_load.stop()
        self.patch_vec_normalize_load.stop()
    
    def test_init_success(self):
        """Test successful initialization of PolicyExecutor."""
        # Create policy executor
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        
        # Check initialization
        self.assertEqual(executor.device, "mock_device")
        self.assertEqual(executor.model_path, "models/test_model.zip")
        self.assertEqual(executor.model, self.mock_model)
        self.assertEqual(executor.env, self.mock_env)
        self.assertTrue(executor.loaded)
        
        # Check model loading
        self.mock_torch_device.assert_called_once_with("cuda")
        self.mock_ppo_load.assert_called_once_with("models/test_model.zip", device="mock_device")
        self.mock_vec_normalize_load.assert_called_once()
    
    def test_init_no_env_file(self):
        """Test initialization without environment file."""
        # Configure mock to indicate env file doesn't exist
        with patch('os.path.exists', side_effect=lambda path: "model" in path):
            # Create policy executor
            executor = PolicyExecutor(
                model_path="models/test_model.zip",
                env_path="models/test_env.pkl",
                device="cuda"
            )
            
            # Check initialization
            self.assertEqual(executor.model, self.mock_model)
            self.assertIsNone(executor.env)
            self.assertTrue(executor.loaded)
    
    def test_init_model_not_found(self):
        """Test initialization when model file is not found."""
        # Configure mock to indicate model file doesn't exist
        with patch('os.path.exists', return_value=False):
            # Create policy executor
            executor = PolicyExecutor(model_path="models/nonexistent_model.zip", device="cuda")
            
            # Check initialization
            self.assertIsNone(executor.model)
            self.assertIsNone(executor.env)
            self.assertFalse(executor.loaded)
    
    def test_init_sb3_not_available(self):
        """Test initialization when Stable-Baselines3 is not available."""
        # Patch to simulate SB3 not available
        with patch('src.phosphobot_construct.policy.HAS_SB3', False):
            # Create policy executor
            executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
            
            # Check initialization
            self.assertIsNone(executor.model)
            self.assertIsNone(executor.env)
            self.assertFalse(executor.loaded)
    
    def test_preprocess_observation(self):
        """Test observation preprocessing."""
        # Create policy executor
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        
        # Preprocess observation
        observation = executor.preprocess_observation(self.sample_scene, self.sample_goal)
        
        # Check preprocessing results
        self.assertIsInstance(observation, dict)
        self.assertIn("state", observation)
        self.assertIn("image", observation)
        
        # Check state extraction
        np.testing.assert_array_equal(
            observation["state"], 
            np.array(self.sample_scene["robot_state"])
        )
        
        # Check normalization
        self.mock_env.normalize_obs.assert_called_once()
    
    def test_preprocess_observation_no_robot_state(self):
        """Test observation preprocessing without robot state."""
        # Create policy executor
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        
        # Create scene without robot state
        scene_without_state = {
            "objects": self.sample_scene["objects"]
        }
        
        # Preprocess observation
        observation = executor.preprocess_observation(scene_without_state, self.sample_goal)
        
        # Check preprocessing results
        self.assertIsInstance(observation, dict)
        self.assertIn("state", observation)
        self.assertIn("image", observation)
        
        # Check default state
        self.assertEqual(observation["state"].shape, (12,))  # Default zeros
        self.assertEqual(observation["state"].dtype, np.float32)
    
    def test_preprocess_observation_no_env(self):
        """Test observation preprocessing without environment normalization."""
        # Create policy executor without environment
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        executor.env = None
        
        # Preprocess observation
        observation = executor.preprocess_observation(self.sample_scene, self.sample_goal)
        
        # Check preprocessing results
        self.assertIsInstance(observation, dict)
        self.assertIn("state", observation)
        self.assertIn("image", observation)
        
        # Check that normalization was skipped
        self.mock_env.normalize_obs.assert_not_called()
    
    def test_predict_action_success(self):
        """Test successful action prediction."""
        # Configure mock model
        predicted_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.mock_model.predict.return_value = (predicted_action, {})
        
        # Create policy executor
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        
        # Create observation
        observation = {
            "state": np.zeros(12),
            "image": np.zeros((64, 64, 4), dtype=np.uint8)
        }
        
        # Predict action
        action, info = executor.predict_action(observation, deterministic=True)
        
        # Check prediction
        self.mock_model.predict.assert_called_once_with(
            observation=observation,
            deterministic=True
        )
        
        np.testing.assert_array_equal(action, predicted_action)
    
    def test_predict_action_model_not_loaded(self):
        """Test action prediction when model is not loaded."""
        # Create policy executor without model
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        executor.loaded = False
        executor.model = None
        
        # Create observation
        observation = {
            "state": np.zeros(12),
            "image": np.zeros((64, 64, 4), dtype=np.uint8)
        }
        
        # Predict action
        action, info = executor.predict_action(observation, deterministic=True)
        
        # Check default action
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertIn("error", info)
    
    def test_predict_action_exception(self):
        """Test action prediction when model raises an exception."""
        # Configure mock model to raise an exception
        self.mock_model.predict.side_effect = Exception("Model error")
        
        # Create policy executor
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        
        # Create observation
        observation = {
            "state": np.zeros(12),
            "image": np.zeros((64, 64, 4), dtype=np.uint8)
        }
        
        # Predict action
        action, info = executor.predict_action(observation, deterministic=True)
        
        # Check default action and error info
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertIn("error", info)
        self.assertEqual(info["error"], "Model error")
    
    def test_generate_trajectory(self):
        """Test trajectory generation."""
        # Configure mock model
        predicted_actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.mock_model.predict.return_value = (predicted_actions, {})
        
        # Create policy executor
        executor = PolicyExecutor(model_path="models/test_model.zip", device="cuda")
        
        # Generate trajectory
        trajectory = executor.generate_trajectory(
            scene_3d=self.sample_scene,
            goal_3d=self.sample_goal,
            num_steps=5
        )
        
        # Check trajectory
        self.assertEqual(len(trajectory), 5)
        for action in trajectory:
            np.testing.assert_array_equal(action, predicted_actions)
            
        # Check that predict was called for each step
        self.assertEqual(self.mock_model.predict.call_count, 5)


class TestExecutePolicy(unittest.TestCase):
    """Tests for the execute_policy function."""
    
    def setUp(self):
        """Setup for tests."""
        # Create sample data
        self.sample_scene = {
            "objects": [
                {"class": "a red cube", "position_3d": {"x": 0.1, "y": 0.2, "z": 0.3}},
                {"class": "a blue sphere", "position_3d": {"x": -0.2, "y": 0.1, "z": 0.1}}
            ],
            "robot_state": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        self.sample_goal = {
            "objects": [
                {"class": "a red cube", "goal_position": {"x": 0.1, "y": 0.2, "z": 0.5}},
                {"class": "a blue sphere", "goal_position": {"x": -0.2, "y": 0.1, "z": 0.1}}
            ]
        }
        
        # Create patchers
        self.patch_policy_executor = patch('src.phosphobot_construct.policy.PolicyExecutor')
        self.patch_phospho_api = patch('src.phosphobot_construct.policy.PhosphoApi')
        self.patch_time_sleep = patch('src.phosphobot_construct.policy.time.sleep')
        
        # Start patches
        self.mock_policy_executor = self.patch_policy_executor.start()
        self.mock_phospho_api = self.patch_phospho_api.start()
        self.patch_time_sleep.start()
        
        # Configure mocks
        self.mock_executor_instance = MagicMock()
        self.mock_policy_executor.return_value = self.mock_executor_instance
        
        self.mock_client = MagicMock()
        self.mock_phospho_api.return_value = self.mock_client
        
        # Mock executor's loaded property and methods
        type(self.mock_executor_instance).loaded = PropertyMock(return_value=True)
        
        # Mock joint state
        self.mock_state = MagicMock()
        self.mock_state.angles_rad = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.mock_client.control.read_joints.return_value = self.mock_state
        
        # Mock trajectory generation
        self.mock_trajectory = [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) for _ in range(20)]
        self.mock_executor_instance.generate_trajectory.return_value = self.mock_trajectory
        
        # Mock action prediction
        self.mock_executor_instance.predict_action.return_value = (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            {"check_goal": False}
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.patch_policy_executor.stop()
        self.patch_phospho_api.stop()
        self.patch_time_sleep.stop()
    
    def test_execute_policy_success(self):
        """Test successful policy execution."""
        # Configure mock to indicate goal reached after 5 steps
        self.mock_executor_instance.predict_action.side_effect = [
            (np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), {"check_goal": False}) 
            for _ in range(5)
        ] + [
            (np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), {"check_goal": True})
        ]
        
        # Execute policy
        result = execute_policy(
            model_path="models/test_model.zip",
            scene_3d=self.sample_scene,
            goal_3d=self.sample_goal,
            client=self.mock_client,
            max_steps=100
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 6)  # 5 regular steps + 1 goal step
        self.assertEqual(len(result["trajectory"]), 6)
        
        # Check that client was used
        self.mock_client.control.read_joints.assert_called()
        self.mock_client.control.write_joints.assert_called()
        
        # Check that executor was used
        self.mock_policy_executor.assert_called_once_with(
            model_path="models/test_model.zip",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mock_executor_instance.generate_trajectory.assert_called_once()
        self.assertEqual(self.mock_executor_instance.predict_action.call_count, 6)
    
    def test_execute_policy_max_steps(self):
        """Test policy execution reaching max steps."""
        # Configure mock to never reach goal
        self.mock_executor_instance.predict_action.return_value = (
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            {"check_goal": False}
        )
        
        # Execute policy with small max_steps
        result = execute_policy(
            model_path="models/test_model.zip",
            scene_3d=self.sample_scene,
            goal_3d=self.sample_goal,
            client=self.mock_client,
            max_steps=10
        )
        
        # Check result
        self.assertFalse(result["success"])
        self.assertEqual(result["steps"], 10)
        self.assertEqual(len(result["trajectory"]), 10)
    
    def test_execute_policy_model_not_loaded(self):
        """Test policy execution when model fails to load."""
        # Configure mock to indicate model not loaded
        type(self.mock_executor_instance).loaded = PropertyMock(return_value=False)
        
        # Execute policy
        result = execute_policy(
            model_path="models/test_model.zip",
            scene_3d=self.sample_scene,
            goal_3d=self.sample_goal,
            client=self.mock_client,
            max_steps=100
        )
        
        # Check result
        self.assertFalse(result["success"])
        self.assertEqual(result["steps"], 0)
        self.assertIn("error", result)
    
    def test_execute_policy_client_error(self):
        """Test policy execution with client errors."""
        # Configure mock to raise an exception
        self.mock_client.control.read_joints.side_effect = Exception("Client error")
        
        # Execute policy
        result = execute_policy(
            model_path="models/test_model.zip",
            scene_3d=self.sample_scene,
            goal_3d=self.sample_goal,
            client=self.mock_client,
            max_steps=100
        )
        
        # Check result
        self.assertFalse(result["success"])
        self.assertEqual(result["steps"], 0)
        self.assertIn("error", result)
    
    def test_execute_policy_create_client(self):
        """Test policy execution creating a client."""
        # Execute policy without providing a client
        result = execute_policy(
            model_path="models/test_model.zip",
            scene_3d=self.sample_scene,
            goal_3d=self.sample_goal,
            client=None,
            max_steps=100,
            server_url="http://test-server",
            server_port=8080
        )
        
        # Check client creation
        self.mock_phospho_api.assert_called_once_with(
            base_url="http://test-server:8080"
        )


if __name__ == "__main__":
    unittest.main()
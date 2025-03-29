"""
Unit tests for the phosphobot_construct.reinforcement_learning module.
"""

import unittest
import numpy as np
import os
from unittest.mock import patch, MagicMock, PropertyMock

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock torch, gymnasium, and stable_baselines3 since they're optional dependencies
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.distributions'] = MagicMock()
sys.modules['gymnasium'] = MagicMock()
sys.modules['gymnasium.spaces'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()
sys.modules['stable_baselines3.PPO'] = MagicMock()
sys.modules['stable_baselines3.common'] = MagicMock()
sys.modules['stable_baselines3.common.vec_env'] = MagicMock()
sys.modules['stable_baselines3.common.vec_env.dummy_vec_env'] = MagicMock()
sys.modules['stable_baselines3.common.vec_env.vec_normalize'] = MagicMock()

# Set up mock imports
import torch
import torch.nn as nn
import torch.distributions
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Now that we've mocked the dependencies, we can import the module being tested
from src.phosphobot_construct.reinforcement_learning import (
    RobotTransformerPolicy, RobotEnv, train_robot_policy, HAS_RL_DEPS
)


class MockModule(MagicMock):
    """Mock nn.Module with properly mocked functions."""
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TestRobotTransformerPolicy(unittest.TestCase):
    """Tests for the RobotTransformerPolicy class."""
    
    def setUp(self):
        """Setup for tests."""
        # Set up torch module mocks
        nn.Module = MockModule
        nn.Conv2d = MockModule
        nn.ReLU = MockModule
        nn.Flatten = MockModule
        nn.Linear = MockModule
        nn.Sequential = MockModule
        nn.TransformerEncoderLayer = MockModule
        nn.TransformerEncoder = MockModule
        
        # Mock tensor operations
        torch.zeros = MagicMock(return_value=torch.tensor([0.0]))
        torch.ones = MagicMock(return_value=torch.tensor([1.0]))
        torch.randn = MagicMock(return_value=torch.tensor([0.5]))
        torch.exp = MagicMock(return_value=torch.tensor([1.0]))
        torch.clamp = MagicMock(return_value=torch.tensor([0.0]))
        
        # Mock shape and device attributes
        tensor_mock = MagicMock()
        tensor_mock.shape = [2, 6]
        tensor_mock.device = "mock_device"
        torch.tensor = MagicMock(return_value=tensor_mock)
        
        # Mock distributions
        normal_mock = MagicMock()
        normal_mock.sample = MagicMock(return_value=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        normal_mock.log_prob = MagicMock(return_value=torch.tensor([0.0]))
        normal_mock.entropy = MagicMock(return_value=torch.tensor([1.0]))
        torch.distributions.Normal = MagicMock(return_value=normal_mock)
        
    def test_init(self):
        """Test initialization of RobotTransformerPolicy."""
        # Create policy with default parameters
        policy = RobotTransformerPolicy()
        
        # Check attributes
        self.assertEqual(policy.state_dim, 6)
        self.assertEqual(policy.action_dim, 6)
        self.assertEqual(policy.hidden_dim, 256)
        self.assertFalse(policy.use_language)
        
        # Check component initialization
        self.assertIsNotNone(policy.vision_encoder)
        self.assertIsNotNone(policy.prop_encoder)
        self.assertIsNotNone(policy.transformer)
        self.assertIsNotNone(policy.policy_mean)
        self.assertIsNotNone(policy.policy_log_std)
        self.assertIsNotNone(policy.value_head)
        
        # Create policy with custom parameters
        policy = RobotTransformerPolicy(
            state_dim=12,
            action_dim=7,
            hidden_dim=512,
            num_layers=8,
            num_heads=16,
            use_language=True,
            language_dim=1024
        )
        
        # Check custom attributes
        self.assertEqual(policy.state_dim, 12)
        self.assertEqual(policy.action_dim, 7)
        self.assertEqual(policy.hidden_dim, 512)
        self.assertTrue(policy.use_language)
        
        # Check language encoder initialization
        self.assertIsNotNone(policy.language_encoder)
    
    @patch('torch.unsqueeze')
    def test_forward(self, mock_unsqueeze):
        """Test forward pass of RobotTransformerPolicy."""
        # Mock torch tensors
        mock_state = MagicMock()
        mock_images = MagicMock()
        mock_language = MagicMock()
        
        # Mock unsqueeze return value
        mock_unsqueeze.return_value = MagicMock()
        
        # Mock batch size
        mock_images.shape = [2, 3, 64, 64]
        
        # Create policy
        policy = RobotTransformerPolicy()
        
        # Mock encoder outputs
        policy.vision_encoder.return_value = torch.tensor([1.0])
        policy.prop_encoder.return_value = torch.tensor([1.0])
        policy.transformer.return_value = torch.tensor([1.0])
        policy.policy_mean.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        policy.policy_log_std.return_value = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        policy.value_head.return_value = torch.tensor([5.0])
        
        # Forward pass
        mean, std, value = policy.forward(mock_state, mock_images)
        
        # Check that encoders were called
        policy.vision_encoder.assert_called()
        policy.prop_encoder.assert_called()
        policy.transformer.assert_called()
        
        # Check outputs
        self.assertEqual(mean, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    
    def test_get_action(self):
        """Test action sampling from policy."""
        # Mock tensors
        mock_state = MagicMock()
        mock_images = MagicMock()
        
        # Create policy
        policy = RobotTransformerPolicy()
        
        # Mock forward method
        policy.forward = MagicMock(return_value=(
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            torch.tensor([5.0])
        ))
        
        # Sample action deterministically
        actions, log_probs = policy.get_action(
            mock_state, mock_images, deterministic=True
        )
        
        # Check deterministic action
        self.assertEqual(actions, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        
        # Sample action stochastically
        actions, log_probs = policy.get_action(
            mock_state, mock_images, deterministic=False
        )
        
        # Check stochastic action
        self.assertEqual(actions, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        
        # Check that Normal distribution was created
        torch.distributions.Normal.assert_called_once()
    
    def test_evaluate_actions(self):
        """Test action evaluation for training."""
        # Mock tensors
        mock_state = MagicMock()
        mock_images = MagicMock()
        mock_actions = MagicMock()
        
        # Create policy
        policy = RobotTransformerPolicy()
        
        # Mock forward method
        policy.forward = MagicMock(return_value=(
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            torch.tensor([5.0])
        ))
        
        # Evaluate actions
        log_probs, entropy, values = policy.evaluate_actions(
            mock_state, mock_images, mock_actions
        )
        
        # Check normal distribution creation
        torch.distributions.Normal.assert_called_once()
        
        # Check outputs
        normal = torch.distributions.Normal.return_value
        normal.log_prob.assert_called_once_with(mock_actions)
        normal.entropy.assert_called_once()


class TestRobotEnv(unittest.TestCase):
    """Tests for the RobotEnv Gymnasium environment."""
    
    def setUp(self):
        """Setup for tests."""
        # Mock gymnasium components
        gym.Env = MagicMock
        spaces.Box = MagicMock
        spaces.Dict = MagicMock
        
        # Sample scene and goal data
        self.scene_data = {
            "objects": [
                {"class": "a red cube", "position_3d": {"x": 0.1, "y": 0.2, "z": 0.3}}
            ]
        }
        
        self.goal_data = {
            "objects": [
                {"class": "a red cube", "goal_position": {"x": 0.1, "y": 0.2, "z": 0.5}}
            ]
        }
        
        # Mock simulator
        self.mock_simulator = MagicMock()
    
    @patch('src.phosphobot_construct.reinforcement_learning.HAS_RL_DEPS', True)
    def test_init(self):
        """Test initialization of RobotEnv."""
        # Mock PhosphobotSimulator import
        with patch('src.phosphobot_construct.reinforcement_learning.PhosphobotSimulator', 
                  return_value=self.mock_simulator):
            # Create environment
            env = RobotEnv(
                scene_data=self.scene_data,
                goal_data=self.goal_data,
                use_simulator=True
            )
            
            # Check attributes
            self.assertEqual(env.max_steps, 100)
            self.assertEqual(env.reward_type, "exponential")
            self.assertTrue(env.use_simulator)
            self.assertEqual(env.scene_data, self.scene_data)
            self.assertEqual(env.goal_data, self.goal_data)
            self.assertEqual(env.simulator, self.mock_simulator)
            
            # Check space creation
            spaces.Box.assert_called()
            spaces.Dict.assert_called()
    
    @patch('src.phosphobot_construct.reinforcement_learning.HAS_RL_DEPS', False)
    def test_init_no_deps(self):
        """Test initialization without RL dependencies."""
        # Create environment (should still work but simulator not initialized)
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            use_simulator=True
        )
        
        # Check that simulator is None
        self.assertIsNone(env.simulator)
    
    def test_reset(self):
        """Test environment reset."""
        # Create environment without simulator
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            use_simulator=False
        )
        
        # Reset environment
        observation, info = env.reset()
        
        # Check reset state
        self.assertEqual(env.current_step, 0)
        self.assertEqual(env.trajectory, [])
        self.assertFalse(env.success)
        
        # Check observation structure
        self.assertIn("state", observation)
        self.assertIn("image", observation)
    
    def test_reset_with_simulator(self):
        """Test environment reset with simulator."""
        # Create environment with mock simulator
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            use_simulator=False  # We'll set simulator manually
        )
        env.simulator = self.mock_simulator
        
        # Reset environment
        observation, info = env.reset()
        
        # Check simulator reset
        self.mock_simulator.reset_robot.assert_called_once()
        self.mock_simulator.step_simulation.assert_called_once()
    
    def test_step(self):
        """Test environment step function."""
        # Create environment without simulator
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            use_simulator=False
        )
        
        # Reset to initialize
        env.reset()
        
        # Take a step
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check step updates
        self.assertEqual(env.current_step, 1)
        self.assertEqual(len(env.trajectory), 1)
        np.testing.assert_array_equal(env.trajectory[0], action)
        
        # Check observation structure
        self.assertIn("state", next_obs)
        self.assertIn("image", next_obs)
        
        # Check info
        self.assertIn("success", info)
        self.assertIn("distance_to_goal", info)
    
    def test_step_with_simulator(self):
        """Test environment step with simulator."""
        # Create environment with mock simulator
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            use_simulator=False  # We'll set simulator manually
        )
        env.simulator = self.mock_simulator
        
        # Configure simulator mock
        joint_state = {
            "positions": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "velocities": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        self.mock_simulator.get_joint_states.return_value = joint_state
        
        camera_data = {
            "rgb": np.zeros((240, 320, 3), dtype=np.uint8),
            "depth": np.zeros((240, 320), dtype=np.float32)
        }
        self.mock_simulator.capture_camera_image.return_value = camera_data
        
        # Reset to initialize
        env.reset()
        
        # Take a step
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check simulator interaction
        self.mock_simulator.set_joint_positions.assert_called_once_with(action)
        self.mock_simulator.step_simulation.assert_called()
        self.mock_simulator.get_joint_states.assert_called_once()
        self.mock_simulator.capture_camera_image.assert_called_once()
    
    def test_calculate_reward_sparse(self):
        """Test sparse reward calculation."""
        # Create environment with sparse rewards
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            reward_type="sparse"
        )
        
        # Override distance calculation for testing
        env._get_distance_to_goal = MagicMock()
        
        # Test with goal not reached
        env._get_distance_to_goal.return_value = 0.1
        reward = env._calculate_reward(np.zeros(6))
        self.assertEqual(reward, 0.0)
        self.assertFalse(env.success)
        
        # Test with goal reached
        env._get_distance_to_goal.return_value = 0.01
        reward = env._calculate_reward(np.zeros(6))
        self.assertEqual(reward, 1.0)
        self.assertTrue(env.success)
    
    def test_calculate_reward_dense(self):
        """Test dense reward calculation."""
        # Create environment with dense rewards
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            reward_type="dense"
        )
        
        # Override distance calculation for testing
        env._get_distance_to_goal = MagicMock(return_value=0.5)
        
        # Test initial reward
        reward = env._calculate_reward(np.zeros(6))
        self.assertLess(reward, 0)  # Negative reward based on distance
        
        # Test improvement reward
        env.previous_distance = 0.7
        reward = env._calculate_reward(np.zeros(6))
        self.assertGreater(reward, -0.5)  # Less negative due to improvement
    
    def test_calculate_reward_exponential(self):
        """Test exponential reward calculation."""
        # Create environment with exponential rewards
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            reward_type="exponential"
        )
        
        # Override distance calculation for testing
        env._get_distance_to_goal = MagicMock()
        
        # Test with goal far away
        env._get_distance_to_goal.return_value = 1.0
        reward = env._calculate_reward(np.zeros(6))
        self.assertLess(reward, 0.1)  # Small reward
        
        # Test with goal close
        env._get_distance_to_goal.return_value = 0.1
        reward = env._calculate_reward(np.zeros(6))
        self.assertGreater(reward, 0.5)  # Larger reward
        
        # Test with goal reached
        env._get_distance_to_goal.return_value = 0.01
        reward = env._calculate_reward(np.zeros(6))
        self.assertGreater(reward, 10.0)  # Success bonus
        self.assertTrue(env.success)
    
    def test_is_done(self):
        """Test done condition."""
        # Create environment
        env = RobotEnv(max_steps=10)
        
        # Not done initially
        env.current_step = 0
        env.success = False
        self.assertFalse(env._is_done())
        
        # Done when max steps reached
        env.current_step = 10
        env.success = False
        self.assertTrue(env._is_done())
        
        # Done when success achieved
        env.current_step = 5
        env.success = True
        self.assertTrue(env._is_done())
    
    def test_get_trajectory(self):
        """Test trajectory recording."""
        # Create environment
        env = RobotEnv()
        env.trajectory = [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
        
        # Get trajectory
        trajectory = env.get_trajectory()
        
        # Check trajectory
        self.assertEqual(len(trajectory), 1)
        np.testing.assert_array_equal(trajectory[0], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    
    def test_is_success(self):
        """Test success state."""
        # Create environment
        env = RobotEnv()
        
        # Not success initially
        self.assertFalse(env.is_success())
        
        # Set success flag
        env.success = True
        self.assertTrue(env.is_success())


class TestTrainRobotPolicy(unittest.TestCase):
    """Tests for the train_robot_policy function."""
    
    def setUp(self):
        """Setup for tests."""
        # Create environment data
        self.env_data = {
            "scene": {
                "objects": [
                    {"class": "a red cube", "position_3d": {"x": 0.1, "y": 0.2, "z": 0.3}}
                ]
            },
            "goal": {
                "objects": [
                    {"class": "a red cube", "goal_position": {"x": 0.1, "y": 0.2, "z": 0.5}}
                ]
            },
            "use_simulator": True,
            "max_steps": 50,
            "reward_type": "exponential"
        }
        
        # Mock PPO and environment
        self.mock_ppo = MagicMock()
        self.mock_env = MagicMock()
        self.mock_vec_env = MagicMock()
        self.mock_normalize_env = MagicMock()
        
        # Create patches
        self.patch_has_rl_deps = patch('src.phosphobot_construct.reinforcement_learning.HAS_RL_DEPS', True)
        self.patch_robot_env = patch('src.phosphobot_construct.reinforcement_learning.RobotEnv')
        self.patch_dummy_vec_env = patch('src.phosphobot_construct.reinforcement_learning.DummyVecEnv')
        self.patch_vec_normalize = patch('src.phosphobot_construct.reinforcement_learning.VecNormalize')
        self.patch_ppo = patch('src.phosphobot_construct.reinforcement_learning.PPO')
        self.patch_os_makedirs = patch('os.makedirs')
        
        # Start patches
        self.patch_has_rl_deps.start()
        self.mock_robot_env = self.patch_robot_env.start()
        self.mock_dummy_vec_env = self.patch_dummy_vec_env.start()
        self.mock_vec_normalize = self.patch_vec_normalize.start()
        self.mock_ppo_class = self.patch_ppo.start()
        self.patch_os_makedirs.start()
        
        # Configure mocks
        self.mock_robot_env.return_value = self.mock_env
        self.mock_dummy_vec_env.return_value = self.mock_vec_env
        self.mock_vec_normalize.return_value = self.mock_normalize_env
        self.mock_ppo_class.return_value = self.mock_ppo
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.patch_has_rl_deps.stop()
        self.patch_robot_env.stop()
        self.patch_dummy_vec_env.stop()
        self.patch_vec_normalize.stop()
        self.patch_ppo_class.stop()
        self.patch_os_makedirs.stop()
    
    def test_train_success(self):
        """Test successful training."""
        # Train policy
        model_path = train_robot_policy(
            env_data=self.env_data,
            output_dir="models/test",
            total_timesteps=1000,
            use_transformer=True,
            seed=42
        )
        
        # Check environment creation
        self.mock_robot_env.assert_called_once()
        self.mock_dummy_vec_env.assert_called_once()
        self.mock_vec_normalize.assert_called_once()
        
        # Check PPO creation
        self.mock_ppo_class.assert_called_once()
        kwargs = self.mock_ppo_class.call_args[1]
        self.assertEqual(kwargs["policy"], "MultiInputPolicy")
        self.assertEqual(kwargs["env"], self.mock_normalize_env)
        self.assertEqual(kwargs["seed"], 42)
        
        # Check training
        self.mock_ppo.learn.assert_called_once_with(total_timesteps=1000)
        self.mock_ppo.save.assert_called_once()
        
        # Check return value
        self.assertEqual(model_path, "models/test/robot_policy")
    
    def test_train_custom_params(self):
        """Test training with custom parameters."""
        # Train policy with custom params
        train_robot_policy(
            env_data=self.env_data,
            output_dir="models/custom",
            total_timesteps=5000,
            use_transformer=False,  # Use default MLP policy
            seed=123,
            use_language=True
        )
        
        # Check PPO creation with custom params
        kwargs = self.mock_ppo_class.call_args[1]
        self.assertEqual(kwargs["seed"], 123)
        
        # Check that policy was trained for correct steps
        self.mock_ppo.learn.assert_called_once_with(total_timesteps=5000)
    
    @patch('src.phosphobot_construct.reinforcement_learning.HAS_RL_DEPS', False)
    def test_train_no_deps(self):
        """Test training without RL dependencies."""
        # Train policy (should return None)
        model_path = train_robot_policy(
            env_data=self.env_data,
            output_dir="models/test",
            total_timesteps=1000
        )
        
        # Check that function returned None
        self.assertIsNone(model_path)


if __name__ == "__main__":
    unittest.main()
"""
Unit tests for the phosphobot_construct.reinforcement_learning module.
"""

import unittest
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORTANT: Remove the lines that globally replace torch, gym, stable_baselines3 with MagicMocks.
# We want to test real code, so do not do sys.modules['torch'] = MagicMock() etc.

# Now import the relevant real libraries (or they can be conditionally installed):
import torch
import torch.nn as nn
import torch.distributions
import gymnasium as gym
from gymnasium import spaces

# If you do want to skip real calls to PPO, you can keep a partial patch:
# But do not do sys.modules['stable_baselines3'] = MagicMock().
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Now that we haven't globally replaced them with mocks, we can import the code under test
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
        """Set up partial mocks if needed."""
        # Instead of globally mocking torch.nn, we can do partial patches if desired.
        pass

    def tearDown(self):
        """Tear down anything if needed."""
        pass

    def test_init(self):
        """Test initialization of RobotTransformerPolicy."""
        # Create policy with default parameters (uses real nn layers now)
        policy = RobotTransformerPolicy()

        self.assertEqual(policy.state_dim, 6)
        self.assertEqual(policy.action_dim, 6)
        self.assertEqual(policy.hidden_dim, 256)
        self.assertFalse(policy.use_language)

        # Ensure submodules exist
        self.assertIsNotNone(policy.vision_encoder)
        self.assertIsNotNone(policy.prop_encoder)
        self.assertIsNotNone(policy.transformer)
        self.assertIsNotNone(policy.policy_mean)
        self.assertIsNotNone(policy.policy_log_std)
        self.assertIsNotNone(policy.value_head)

        # Create policy with custom parameters
        policy_custom = RobotTransformerPolicy(
            state_dim=12,
            action_dim=7,
            hidden_dim=512,
            num_layers=8,
            num_heads=16,
            use_language=True,
            language_dim=1024
        )
        self.assertEqual(policy_custom.state_dim, 12)
        self.assertEqual(policy_custom.action_dim, 7)
        self.assertEqual(policy_custom.hidden_dim, 512)
        self.assertTrue(policy_custom.use_language)
        self.assertIsNotNone(policy_custom.language_encoder)

    def test_forward(self):
        """Test forward pass of RobotTransformerPolicy with real partial data."""
        policy = RobotTransformerPolicy()
        batch_size = 2

        # Fake input data
        state = torch.zeros(batch_size, policy.state_dim)
        images = torch.zeros(batch_size, 4, 64, 64)  # 4 channels: RGB-D
        mean, std, value = policy.forward(state, images)

        # Check shapes
        self.assertEqual(mean.shape, (batch_size, policy.action_dim))
        self.assertEqual(std.shape, (batch_size, policy.action_dim))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_get_action(self):
        """Test action sampling from policy."""
        policy = RobotTransformerPolicy()
        batch_size = 1

        state = torch.zeros(batch_size, policy.state_dim)
        images = torch.zeros(batch_size, 4, 64, 64)

        # Deterministic
        actions, log_probs = policy.get_action(state, images, deterministic=True)
        self.assertEqual(actions.shape, (batch_size, policy.action_dim))
        self.assertEqual(log_probs.shape, (batch_size,))

        # Stochastic
        actions, log_probs = policy.get_action(state, images, deterministic=False)
        self.assertEqual(actions.shape, (batch_size, policy.action_dim))
        self.assertEqual(log_probs.shape, (batch_size,))

    def test_evaluate_actions(self):
        """Test action evaluation for training."""
        policy = RobotTransformerPolicy()
        batch_size = 2

        state = torch.zeros(batch_size, policy.state_dim)
        images = torch.zeros(batch_size, 4, 64, 64)
        actions = torch.zeros(batch_size, policy.action_dim)

        log_probs, entropy, values = policy.evaluate_actions(state, images, actions)
        self.assertEqual(log_probs.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))
        self.assertEqual(values.shape, (batch_size,))


class TestRobotEnv(unittest.TestCase):
    """Tests for the RobotEnv Gymnasium environment."""

    def setUp(self):
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

    def test_init(self):
        """Test initialization of RobotEnv."""
        env = RobotEnv(
            scene_data=self.scene_data,
            goal_data=self.goal_data,
            use_simulator=False,  # we won’t run the real simulator
            max_steps=100,
            reward_type="exponential"
        )
        self.assertEqual(env.max_steps, 100)
        self.assertEqual(env.reward_type, "exponential")
        self.assertFalse(env.use_simulator)
        self.assertEqual(env.scene_data, self.scene_data)
        self.assertEqual(env.goal_data, self.goal_data)
        self.assertIsNone(env.simulator)

        self.assertIn('state', env.observation_space.spaces)
        self.assertIn('image', env.observation_space.spaces)

    def test_reset(self):
        env = RobotEnv(self.scene_data, self.goal_data, use_simulator=False)
        obs, info = env.reset()
        self.assertEqual(env.current_step, 0)
        self.assertFalse(env.success)
        self.assertIn('state', obs)
        self.assertIn('image', obs)

    def test_step(self):
        env = RobotEnv(self.scene_data, self.goal_data, use_simulator=False)
        env.reset()
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        obs, reward, done, trunc, info = env.step(action)
        self.assertEqual(env.current_step, 1)
        np.testing.assert_array_equal(env.trajectory[0], action)
        self.assertIn('state', obs)
        self.assertIn('image', obs)
        self.assertIn('success', info)
        self.assertIn('distance_to_goal', info)

    def test_calculate_reward_sparse(self):
        env = RobotEnv(self.scene_data, self.goal_data, reward_type="sparse", use_simulator=False)
        env._get_distance_to_goal = MagicMock(return_value=0.1)
        r = env._calculate_reward(np.zeros(6))
        # Not within 0.05 -> 0.0 reward
        self.assertEqual(r, 0.0)
        self.assertFalse(env.success)

        env._get_distance_to_goal.return_value = 0.01
        r = env._calculate_reward(np.zeros(6))
        self.assertEqual(r, 1.0)
        self.assertTrue(env.success)

    def test_calculate_reward_dense(self):
        env = RobotEnv(self.scene_data, self.goal_data, reward_type="dense", use_simulator=False)
        env._get_distance_to_goal = MagicMock(return_value=0.5)
        r = env._calculate_reward(np.zeros(6))
        # distance=0.5 => reward = -0.5 + possible minus for action penalty
        self.assertLess(r, 0)  # negative

        env.previous_distance = 0.7
        r2 = env._calculate_reward(np.zeros(6))
        # now distance improved from 0.7 to 0.5 => that improvement bonus is added
        self.assertGreater(r2, r)

    def test_calculate_reward_exponential(self):
        env = RobotEnv(self.scene_data, self.goal_data, reward_type="exponential", use_simulator=False)
        env._get_distance_to_goal = MagicMock(return_value=1.0)
        r = env._calculate_reward(np.zeros(6))
        self.assertLess(r, 0.1)

        env._get_distance_to_goal.return_value = 0.1
        r2 = env._calculate_reward(np.zeros(6))
        self.assertGreater(r2, 0.5)

        env._get_distance_to_goal.return_value = 0.01
        r3 = env._calculate_reward(np.zeros(6))
        self.assertGreater(r3, 10.0)
        self.assertTrue(env.success)

    def test_is_done(self):
        env = RobotEnv(max_steps=10, use_simulator=False)
        env.current_step = 10
        self.assertTrue(env._is_done())
        env.current_step = 5
        env.success = True
        self.assertTrue(env._is_done())

    def test_get_trajectory(self):
        env = RobotEnv(use_simulator=False)
        env.trajectory = [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
        traj = env.get_trajectory()
        self.assertEqual(len(traj), 1)
        np.testing.assert_array_equal(traj[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_is_success(self):
        env = RobotEnv(use_simulator=False)
        self.assertFalse(env.is_success())
        env.success = True
        self.assertTrue(env.is_success())


class TestTrainRobotPolicy(unittest.TestCase):
    """Tests for the train_robot_policy function."""

    def setUp(self):
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

        # Force RL_DEPS to True so train_robot_policy doesn't abort
        self.patch_has_rl_deps = patch(
            'src.phosphobot_construct.reinforcement_learning.HAS_RL_DEPS', True
        )
        self.patch_has_rl_deps.start()

        # Patch RobotEnv but return a real RobotEnv object (so SB3 sees a valid env).
        self.patch_robot_env = patch(
            'src.phosphobot_construct.reinforcement_learning.RobotEnv',
            wraps=RobotEnv
        )
        self.mock_robot_env_class = self.patch_robot_env.start()

        # Patch DummyVecEnv to call its real constructor.
        def _dummy_vec_env_side_effect(env_fns):
            from stable_baselines3.common.vec_env import DummyVecEnv
            return DummyVecEnv(env_fns)

        self.patch_dummy_vec_env = patch(
            'src.phosphobot_construct.reinforcement_learning.DummyVecEnv',
            side_effect=_dummy_vec_env_side_effect
        )
        self.mock_dummy_vec_env = self.patch_dummy_vec_env.start()

        # Patch VecNormalize to return a real VecNormalize object.
        def _vec_normalize_side_effect(venv, **kwargs):
            from stable_baselines3.common.vec_env import VecNormalize
            return VecNormalize(venv, **kwargs)

        self.patch_vec_normalize = patch(
            'src.phosphobot_construct.reinforcement_learning.VecNormalize',
            side_effect=_vec_normalize_side_effect
        )
        self.mock_vec_normalize = self.patch_vec_normalize.start()

        # Patch PPO to avoid calling real training code
        self.patch_ppo = patch('src.phosphobot_construct.reinforcement_learning.PPO')
        self.mock_ppo_class = self.patch_ppo.start()

        # Mocked PPO instance
        self.mock_ppo_instance = MagicMock()
        self.mock_ppo_class.return_value = self.mock_ppo_instance

    def tearDown(self):
        self.patch_has_rl_deps.stop()
        self.patch_robot_env.stop()
        self.patch_dummy_vec_env.stop()
        self.patch_vec_normalize.stop()
        self.patch_ppo.stop()
        shutil.rmtree("models/test", ignore_errors=True)
        shutil.rmtree("models/custom", ignore_errors=True)

    def test_train_success(self):
        """Test successful training."""
        model_path = train_robot_policy(
            env_data=self.env_data,
            output_dir="models/test",
            total_timesteps=1000,
            use_transformer=True,
            seed=42
        )
        # RobotEnv should have been called exactly once:
        self.mock_robot_env_class.assert_called_once()

        # DummyVecEnv, VecNormalize each once
        self.mock_dummy_vec_env.assert_called_once()
        self.mock_vec_normalize.assert_called_once()

        # Check PPO creation
        self.mock_ppo_class.assert_called_once()
        kwargs = self.mock_ppo_class.call_args[1]
        self.assertEqual(kwargs["policy"], "MultiInputPolicy")
        # Here "env" is a real VecNormalize object now
        self.assertEqual(kwargs["seed"], 42)

        self.mock_ppo_instance.learn.assert_called_once_with(total_timesteps=1000)
        self.mock_ppo_instance.save.assert_called_once()

        self.assertEqual(model_path, "models/test/robot_policy")

    def test_train_custom_params(self):
        train_robot_policy(
            env_data=self.env_data,
            output_dir="models/custom",
            total_timesteps=5000,
            use_transformer=False,
            seed=123,
            use_language=True
        )
        kwargs = self.mock_ppo_class.call_args[1]
        self.assertEqual(kwargs["seed"], 123)
        self.mock_ppo_instance.learn.assert_called_once_with(total_timesteps=5000)

    @patch('src.phosphobot_construct.reinforcement_learning.HAS_RL_DEPS', new=False)
    def test_train_no_deps(self):
        model_path = train_robot_policy(
            env_data=self.env_data,
            output_dir="models/test",
            total_timesteps=1000
        )
        self.assertIsNone(model_path)


if __name__ == "__main__":
    unittest.main()

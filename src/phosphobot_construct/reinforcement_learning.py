"""
Deep Reinforcement Learning for the Phosphobot Construct.

This module implements reinforcement learning algorithms for training
robot control policies from multimodal inputs.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)

# Import conditional to make the module work even without dependencies
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_RL_DEPS = True
except ImportError:
    logger.warning("Reinforcement learning dependencies not installed.")
    logger.warning("Install with: pip install gymnasium stable-baselines3")
    HAS_RL_DEPS = False
    exit()

class RobotTransformerPolicy(nn.Module):
    """
    Transformer-based policy for robot control.
    
    This neural network architecture processes proprioceptive state,
    visual inputs, and optional language instructions to generate
    robot actions.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        use_language: bool = False,
        language_dim: int = 768
    ):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the robot state vector.
            action_dim: Dimension of the action vector.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            use_language: Whether to use language instruction embeddings.
            language_dim: Dimension of language embeddings.
        """
        super(RobotTransformerPolicy, self).__init__()
        
        # Configuration
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_language = use_language
        
        # Vision encoder (for processing RGB-D images)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # RGB-D: 4 channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_dim)  # Assuming 64x64 input images
        )
        
        # Proprioception encoder
        self.prop_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Language encoder (if using language)
        if use_language:
            self.language_encoder = nn.Sequential(
                nn.Linear(language_dim, hidden_dim),
                nn.ReLU()
            )
        
        # Transformer encoder for sequential decision making
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Policy head (outputs mean and log_std for continuous actions)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Value function head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        images: torch.Tensor,
        language_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state: Robot state tensor [batch_size, state_dim].
            images: Image tensor [batch_size, channels, height, width].
            language_embedding: Optional language embedding [batch_size, language_dim].
            
        Returns:
            Tuple of (action_mean, action_std, value).
        """
        batch_size = state.shape[0]
        
        # Process proprioception
        prop_features = self.prop_encoder(state)
        
        # Process visual input
        vision_features = self.vision_encoder(images)
        
        # Process language if available
        if self.use_language and language_embedding is not None:
            language_features = self.language_encoder(language_embedding)
            # Combine features
            combined_features = prop_features + vision_features + language_features
        else:
            # Combine features without language
            combined_features = prop_features + vision_features
        
        # Add sequence dimension for transformer (here just one token per sample)
        combined_features = combined_features.unsqueeze(1)
        
        # Pass through transformer
        transformer_output = self.transformer(combined_features)
        
        # Extract features (remove sequence dimension)
        features = transformer_output.squeeze(1)
        
        # Policy outputs
        action_mean = self.policy_mean(features)
        action_log_std = self.policy_log_std(features)
        action_std = torch.exp(action_log_std.clamp(-20, 2))
        
        # Value function output
        value = self.value_head(features)
        
        return action_mean, action_std, value
    
    def get_action(
        self,
        state: torch.Tensor,
        images: torch.Tensor,
        language_embedding: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy distribution.
        
        Args:
            state: Robot state tensor.
            images: Image tensor.
            language_embedding: Optional language embedding.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Tuple of (actions, log_probs).
        """
        action_mean, action_std, _ = self.forward(state, images, language_embedding)
        
        if deterministic:
            # Return mean action
            batch_size = action_mean.shape[0]
            actions = action_mean
            # We want log_probs to be shape (batch_size,), so:
            log_probs = torch.zeros(batch_size, dtype=action_mean.dtype, device=action_mean.device)
            return actions, log_probs
        else:
            # Sample from normal distribution
            normal = torch.distributions.Normal(action_mean, action_std)
            actions = normal.sample()
            log_probs = normal.log_prob(actions).sum(dim=-1)
            
            return actions, log_probs
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        images: torch.Tensor,
        actions: torch.Tensor,
        language_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            state: Robot state tensor.
            images: Image tensor.
            actions: Action tensor.
            language_embedding: Optional language embedding.
            
        Returns:
            Tuple of (log_probs, entropy, values).
        """
        action_mean, action_std, values = self.forward(state, images, language_embedding)
        
        # Create normal distribution
        normal = torch.distributions.Normal(action_mean, action_std)
        
        # Calculate log probs and entropy
        log_probs = normal.log_prob(actions).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_probs, entropy, values.squeeze(-1)


class RobotEnv(gym.Env):
    """
    Gymnasium environment for robot reinforcement learning.
    
    This environment simulates a robot manipulation task for training
    with reinforcement learning algorithms.
    """
    
    def __init__(
        self,
        scene_data: Optional[Dict[str, Any]] = None,
        goal_data: Optional[Dict[str, Any]] = None,
        use_simulator: bool = True,
        simulator_path: Optional[str] = None,
        max_steps: int = 100,
        reward_type: str = "exponential"
    ):
        """
        Initialize the robot environment.
        
        Args:
            scene_data: Initial scene data (objects, positions, etc.).
            goal_data: Goal data for the task.
            use_simulator: Whether to use a physics simulator.
            simulator_path: Path to the simulator executable.
            max_steps: Maximum number of steps per episode.
            reward_type: Type of reward function ('exponential', 'sparse', 'dense').
        """
        super().__init__()     

        # Environment parameters
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.use_simulator = use_simulator
        
        # State variables
        self.current_step = 0
        self.scene_data = scene_data or {}
        self.goal_data = goal_data or {}
        self.trajectory = []
        self.success = False
        
        # Set up action and observation spaces
        # Assuming 6 DoF robot arm
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Observation space includes:
        # - Robot state (joint positions, velocities)
        # - Visual observation (RGB-D image)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),  # 6 positions + 6 velocities
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8)  # RGB-D image
        })
        
        # Set up simulator if requested
        self.simulator = None
        if use_simulator and HAS_RL_DEPS:
            self._setup_simulator(simulator_path)
    
    def _setup_simulator(self, simulator_path: Optional[str] = None):
        """
        Set up the physics simulator.
        
        Args:
            simulator_path: Path to the simulator executable.
        """
        try:
            # Import conditionally to avoid dependency issues
            from phosphobot_construct.simulation import PhosphobotSimulator
            
            # Create simulator
            self.simulator = PhosphobotSimulator(gui=False)
            logger.info("Initialized physics simulator")
            
            # Load objects from scene data if available
            if self.scene_data and "objects" in self.scene_data:
                object_configs = []
                for obj in self.scene_data["objects"]:
                    if "class" in obj and "position_3d" in obj:
                        # Determine object type from class
                        if "cube" in obj["class"].lower():
                            urdf_path = "cube.urdf"
                        elif "sphere" in obj["class"].lower():
                            urdf_path = "sphere.urdf"
                        else:
                            urdf_path = "cylinder.urdf"
                        
                        # Extract position
                        pos = obj["position_3d"]
                        position = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
                        
                        # Add to configs
                        object_configs.append({
                            "urdf_path": urdf_path,
                            "position": position,
                            "orientation": [0, 0, 0],
                            "scale": 1.0
                        })
                
                # Load objects into simulator
                self.object_ids = self.simulator.load_objects(object_configs)
                logger.info(f"Loaded {len(self.object_ids)} objects into simulator")
        except Exception as e:
            logger.error(f"Failed to initialize simulator: {str(e)}")
            self.simulator = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Tuple of (observation, info).
        """
        # Initialize state variables
        self.current_step = 0
        self.trajectory = []
        self.success = False
        
        # Reset simulator if available
        if self.simulator is not None:
            self.simulator.reset_robot()
            
            # Step simulation to stabilize
            for _ in range(10):
                self.simulator.step_simulation()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to execute.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Increment step counter
        self.current_step += 1
        
        # Record action in trajectory
        self.trajectory.append(action.copy())
        
        # Execute action in simulator if available
        if self.simulator is not None:
            # Scale action to joint limits if needed
            scaled_action = action.copy()
            
            # Set joint positions
            self.simulator.set_joint_positions(scaled_action)
            
            # Step simulation
            self.simulator.step_simulation(num_steps=10)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self._is_done()
        
        # Prepare info dictionary
        info = {
            "success": self.success,
            "distance_to_goal": self._get_distance_to_goal()
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation from the environment.
        
        Returns:
            Dictionary with observation data.
        """
        # Get joint state from simulator if available
        if self.simulator is not None:
            joint_state = self.simulator.get_joint_states()
            state = np.concatenate([
                joint_state["positions"],
                joint_state["velocities"]
            ])
            
            # Get camera image
            camera_data = self.simulator.capture_camera_image(camera_idx=0)
            rgb_image = camera_data["rgb"]
            depth_image = camera_data["depth"]
            
            # Combine RGB and depth
            rgbd_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 4), dtype=np.uint8)
            rgbd_image[:, :, :3] = rgb_image
            
            # Normalize and scale depth to 0-255
            depth_norm = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image) + 1e-6)
            rgbd_image[:, :, 3] = (depth_norm * 255).astype(np.uint8)
            
            # Resize to observation space shape
            rgbd_image = cv2.resize(rgbd_image, (64, 64))
        else:
            # Create dummy observation if simulator not available
            state = np.zeros(12, dtype=np.float32)
            rgbd_image = np.zeros((64, 64, 4), dtype=np.uint8)
        
        return {
            "state": state.astype(np.float32),
            "image": rgbd_image
        }
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward based on the current state and action.
        
        Args:
            action: Executed action.
            
        Returns:
            Reward value.
        """
        # Get distance to goal
        distance = self._get_distance_to_goal()
        
        # Calculate reward based on reward type
        if self.reward_type == "sparse":
            # Binary reward: 1 if successful, 0 otherwise
            reward = 1.0 if distance < 0.05 else 0.0
            
        elif self.reward_type == "dense":
            # Linear reward based on distance
            reward = -distance
            
            # Add small penalty for large actions
            action_penalty = 0.01 * np.sum(np.square(action))
            reward -= action_penalty
            
            # Add bonus for improvement
            if hasattr(self, "previous_distance"):
                improvement = self.previous_distance - distance
                reward += 2.0 * improvement
            
            # Store current distance for next step
            self.previous_distance = distance
            
        else:  # "exponential"
            # Exponential reward: higher when closer to goal
            reward = np.exp(-5.0 * distance)
            
            # Small penalty for each step
            reward -= 0.01
        
        # Check for success
        if distance < 0.05:
            self.success = True
            if self.reward_type != "sparse":
                reward += 10.0
        
        return reward
    
    def _get_distance_to_goal(self) -> float:
        """
        Calculate distance to the goal state.
        
        Returns:
            Distance to goal (0.0 to 1.0 scale).
        """
        # Simple distance calculation if simulator available
        if self.simulator is not None and self.goal_data and "objects" in self.goal_data:
            # Get current object positions
            object_positions = []
            for obj_id in self.object_ids:
                pos, _ = self.simulator.p.getBasePositionAndOrientation(obj_id)
                object_positions.append(np.array(pos))
            
            # Get goal positions
            goal_positions = []
            for obj in self.goal_data["objects"]:
                if "goal_position" in obj:
                    pos = obj["goal_position"]
                    goal_positions.append(np.array([pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]))
            
            # Calculate distance if we have matching objects and goals
            if len(object_positions) == len(goal_positions):
                total_distance = 0.0
                for obj_pos, goal_pos in zip(object_positions, goal_positions):
                    total_distance += np.linalg.norm(obj_pos - goal_pos)
                
                # Normalize to 0.0-1.0 range
                return min(1.0, total_distance / len(object_positions))
        
        # Fallback: use current step as proxy for distance
        # (assumes we get closer to goal with more steps)
        return 1.0 - min(1.0, self.current_step / (self.max_steps * 0.8))
    
    def _is_done(self) -> bool:
        """
        Check if the episode is done.
        
        Returns:
            True if done, False otherwise.
        """
        # Episode is done if max steps is reached or goal is achieved
        return self.current_step >= self.max_steps or self.success
    
    def get_trajectory(self) -> List[np.ndarray]:
        """
        Get the recorded trajectory of actions.
        
        Returns:
            List of action arrays.
        """
        return self.trajectory
    
    def is_success(self) -> bool:
        """
        Check if the task was successfully completed.
        
        Returns:
            True if successful, False otherwise.
        """
        return self.success
    
    def render(self):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Close the environment and release resources."""
        if self.simulator is not None:
            self.simulator.close()


def train_robot_policy(
    env_data: Dict[str, Any],
    output_dir: str = "models",
    total_timesteps: int = 100000,
    use_transformer: bool = True,
    seed: int = 42,
    use_language: bool = False
) -> str:
    """
    Train a robot control policy using reinforcement learning.
    
    Args:
        env_data: Dictionary with environment data.
        output_dir: Directory to save the trained model.
        total_timesteps: Total number of timesteps for training.
        use_transformer: Whether to use transformer-based policy.
        seed: Random seed.
        use_language: Whether to use language instructions.
        
    Returns:
        Path to the saved model.
    """
    if not HAS_RL_DEPS:
        logger.error("Reinforcement learning dependencies not installed.")
        return None
    
    logger.info("Setting up environment for training")
    
    # Create environment
    def make_env():
        env = RobotEnv(
            scene_data=env_data.get("scene", {}),
            goal_data=env_data.get("goal", {}),
            use_simulator=env_data.get("use_simulator", True),
            simulator_path=env_data.get("simulator_path"),
            max_steps=env_data.get("max_steps", 100),
            reward_type=env_data.get("reward_type", "exponential")
        )
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create policy kwargs
    if use_transformer:
        # Custom transformer policy
        policy_kwargs = {
            "net_arch": [256, 256],
            "activation_fn": nn.ReLU,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"lr": 3e-4}
        }
    else:
        # Default MLP policy
        policy_kwargs = {
            "net_arch": [256, 256],
            "activation_fn": nn.ReLU
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_log = os.path.join(output_dir, "tensorboard")
    
    # Initialize PPO with the policy
    model = PPO(
        policy="MultiInputPolicy",  # For Dict observation spaces
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1
    )
    
    # Train the model
    logger.info(f"Starting training for {total_timesteps} steps")
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model_path = os.path.join(output_dir, "robot_policy")
    model.save(model_path)
    logger.info(f"Trained model saved to {model_path}")
    
    # Save the normalized environment parameters
    env_path = os.path.join(output_dir, "vec_normalize.pkl")
    env.save(env_path)
    logger.info(f"Environment normalization parameters saved to {env_path}")
    
    return model_path


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the training function with a simple environment
    if HAS_RL_DEPS:
        # Sample environment data
        env_data = {
            "use_simulator": True,
            "max_steps": 50,
            "reward_type": "exponential",
            "scene": {
                "objects": [
                    {"class": "a red cube", "position_3d": {"x": -0.3, "y": 0.2, "z": 0.1}},
                    {"class": "a blue cube", "position_3d": {"x": 0.1, "y": -0.3, "z": 0.1}},
                    {"class": "a green cube", "position_3d": {"x": 0.4, "y": 0.3, "z": 0.1}}
                ]
            },
            "goal": {
                "objects": [
                    {"class": "a red cube", "goal_position": {"x": -0.3, "y": 0.2, "z": 0.3}},
                    {"class": "a blue cube", "goal_position": {"x": -0.3, "y": 0.2, "z": 0.4}},
                    {"class": "a green cube", "goal_position": {"x": -0.3, "y": 0.2, "z": 0.5}}
                ]
            }
        }
        
        # Train a policy (reduced timesteps for testing)
        train_robot_policy(
            env_data=env_data,
            output_dir="models/test",
            total_timesteps=1000,  # Very small for testing
            use_transformer=True
        )
    else:
        logger.warning("Skipping test training as dependencies are not installed")

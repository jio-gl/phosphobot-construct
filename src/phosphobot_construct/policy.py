"""
Policy Execution for the Phosphobot Construct.

This module handles the execution of trained policies for robot control,
generating optimal trajectories toward the goal state.
"""

import os
import logging
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from phosphobot.api.client import PhosphoApi

logger = logging.getLogger(__name__)

# Import conditional to make the module work even without dependencies
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    HAS_SB3 = True
except ImportError:
    logger.warning("Stable-Baselines3 not installed. Install with: pip install stable-baselines3")
    HAS_SB3 = False


class PolicyExecutor:
    """
    Executes trained policies to generate optimal trajectories.
    
    This class loads and runs trained policies to control the robot
    based on the current scene and goal state.
    """
    
    def __init__(
        self,
        model_path: str,
        env_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize the policy executor.
        
        Args:
            model_path: Path to trained model.
            env_path: Path to saved environment normalization parameters.
            device: Device to run inference on ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_path = model_path
        
        # Load the model
        if HAS_SB3 and os.path.exists(model_path):
            try:
                logger.info(f"Loading policy from {model_path}")
                self.model = PPO.load(model_path, device=self.device)
                
                # Load environment normalization parameters if available
                if env_path and os.path.exists(env_path):
                    logger.info(f"Loading environment stats from {env_path}")
                    self.env = VecNormalize.load(env_path, None)
                    self.env.training = False
                    self.env.norm_reward = False
                else:
                    self.env = None
                    
                self.loaded = True
                logger.info("Policy loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load policy: {str(e)}")
                self.model = None
                self.env = None
                self.loaded = False
        else:
            logger.warning(f"Model not found at {model_path} or Stable-Baselines3 not installed")
            self.model = None
            self.env = None
            self.loaded = False
    
    def preprocess_observation(
        self,
        scene_3d: Dict[str, Any],
        goal_3d: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess scene and goal data into observation format.
        
        Args:
            scene_3d: Current 3D scene representation.
            goal_3d: Goal 3D representation (optional).
            
        Returns:
            Dictionary with observation data.
        """
        # Extract robot state if available
        if "robot_state" in scene_3d:
            state = np.array(scene_3d["robot_state"])
        else:
            # Default to zeros if not available
            state = np.zeros(12)
        
        # Create dummy image if no images provided
        # In a real implementation, this would process the actual images
        image = np.zeros((64, 64, 4), dtype=np.uint8)
        
        # Create observation dictionary
        observation = {
            "state": state.astype(np.float32),
            "image": image
        }
        
        # Normalize observation if environment is available
        if self.env is not None:
            observation = self.env.normalize_obs(observation)
        
        return observation
    
    def predict_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action using the loaded policy.
        
        Args:
            observation: Preprocessed observation.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Tuple of (action, info).
        """
        if not self.loaded or self.model is None:
            logger.error("Model not loaded. Cannot predict action.")
            return np.zeros(6), {}
        
        try:
            action, _states = self.model.predict(
                observation=observation,
                deterministic=deterministic
            )
            return action, {"states": _states}
        except Exception as e:
            logger.error(f"Error predicting action: {str(e)}")
            return np.zeros(6), {"error": str(e)}
    
    def generate_trajectory(
        self,
        scene_3d: Dict[str, Any],
        goal_3d: Dict[str, Any],
        num_steps: int = 10
    ) -> List[np.ndarray]:
        """
        Generate a trajectory to reach the goal state.
        
        Args:
            scene_3d: Current 3D scene representation.
            goal_3d: Goal 3D representation.
            num_steps: Number of steps to look ahead.
            
        Returns:
            List of action arrays forming a trajectory.
        """
        logger.info(f"Generating trajectory to reach goal")
        
        # Initialize trajectory
        trajectory = []
        
        # Initialize current scene
        current_scene = scene_3d.copy()
        
        # Generate sequence of actions
        for i in range(num_steps):
            # Preprocess observation
            observation = self.preprocess_observation(current_scene, goal_3d)
            
            # Predict action
            action, info = self.predict_action(observation, deterministic=True)
            
            # Add action to trajectory
            trajectory.append(action)
            
            # Simulate next scene (simple approximation)
            # In a real implementation, this would use more sophisticated forward dynamics
            if "robot_state" in current_scene:
                # Update robot state
                current_state = np.array(current_scene["robot_state"])
                next_state = current_state + action  # Simple update
                current_scene["robot_state"] = next_state.tolist()
            
            # Log progress
            if i % 5 == 0:
                logger.info(f"Generated {i+1}/{num_steps} trajectory steps")
        
        logger.info(f"Generated trajectory with {len(trajectory)} steps")
        return trajectory


def execute_policy(
    model_path: str,
    scene_3d: Dict[str, Any],
    goal_3d: Dict[str, Any],
    client: Optional[PhosphoApi] = None,
    max_steps: int = 100,
    server_url: str = "http://localhost",
    server_port: int = 80
) -> Dict[str, Any]:
    """
    Execute the trained policy to reach the goal state.
    
    Args:
        model_path: Path to trained model.
        scene_3d: Current 3D scene representation.
        goal_3d: Goal 3D representation.
        client: PhosphoApi client instance (optional).
        max_steps: Maximum number of steps to execute.
        server_url: URL of the Phosphobot server.
        server_port: Port of the Phosphobot server.
        
    Returns:
        Dictionary with execution results.
    """
    # Initialize result
    result = {
        "success": False,
        "steps": 0,
        "total_reward": 0.0,
        "trajectory": []
    }
    
    # Create client if not provided
    if client is None:
        client = PhosphoApi(base_url=f"{server_url}:{server_port}")
    
    # Load policy
    executor = PolicyExecutor(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if not executor.loaded:
        logger.error("Failed to load policy. Cannot execute.")
        result["error"] = "Failed to load policy"
        return result
    
    # Initialize environment
    try:
        # Reset robot
        client.move.init()
        time.sleep(2)  # Wait for robot to reach initial position
        
        # Get initial robot state
        current_state = np.array(client.control.read_joints().angles_rad)
        scene_3d["robot_state"] = current_state.tolist()
    except Exception as e:
        logger.error(f"Error initializing robot: {str(e)}")
        result["error"] = f"Error initializing robot: {str(e)}"
        return result
    
    # Generate initial trajectory
    trajectory = executor.generate_trajectory(
        scene_3d=scene_3d,
        goal_3d=goal_3d,
        num_steps=20  # Look ahead 20 steps
    )
    
    # Execute policy
    steps = 0
    done = False
    
    logger.info("Starting policy execution")
    
    while not done and steps < max_steps:
        try:
            # Update scene
            current_state = np.array(client.control.read_joints().angles_rad)
            scene_3d["robot_state"] = current_state.tolist()
            
            # Get action
            observation = executor.preprocess_observation(scene_3d, goal_3d)
            action, info = executor.predict_action(observation)
            
            # Add action to result trajectory
            result["trajectory"].append(action.tolist())
            
            # Execute action
            client.control.write_joints(angles=action.tolist())
            
            # Wait for action execution
            time.sleep(1 / 30)  # 30 Hz control frequency
            
            # Update step counter
            steps += 1
            
            # Check if goal is reached
            # This is a simplified check and would be more sophisticated in practice
            if "check_goal" in info and info["check_goal"]:
                done = True
                result["success"] = True
                logger.info("Goal reached. Policy execution successful.")
            
            # Log progress every 10 steps
            if steps % 10 == 0:
                logger.info(f"Executed {steps}/{max_steps} steps")
                
                # Regenerate trajectory every 10 steps
                trajectory = executor.generate_trajectory(
                    scene_3d=scene_3d,
                    goal_3d=goal_3d,
                    num_steps=20
                )
            
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            break
    
    # Update result
    result["steps"] = steps
    
    if steps >= max_steps:
        logger.info(f"Reached maximum number of steps ({max_steps}). Stopping execution.")
    
    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    if HAS_SB3:
        try:
            # Sample data
            sample_scene = {
                "objects": [
                    {"class": "a red cube", "position_3d": {"x": -0.3, "y": 0.2, "z": 0.1}},
                    {"class": "a blue cube", "position_3d": {"x": 0.1, "y": -0.3, "z": 0.1}},
                    {"class": "a green cube", "position_3d": {"x": 0.4, "y": 0.3, "z": 0.1}}
                ]
            }
            
            sample_goal = {
                "objects": [
                    {"class": "a red cube", "goal_position": {"x": -0.3, "y": 0.2, "z": 0.3}},
                    {"class": "a blue cube", "goal_position": {"x": -0.3, "y": 0.2, "z": 0.4}},
                    {"class": "a green cube", "goal_position": {"x": -0.3, "y": 0.2, "z": 0.5}}
                ],
                "spatial_relations": "The cubes are stacked with red at the bottom, blue in the middle, and green on top."
            }
            
            # Path to trained model (replace with actual path)
            model_path = "models/robot_policy.zip"
            
            # Check if model exists
            if os.path.exists(model_path):
                # Create policy executor
                executor = PolicyExecutor(model_path=model_path)
                
                # Generate trajectory
                if executor.loaded:
                    trajectory = executor.generate_trajectory(
                        scene_3d=sample_scene,
                        goal_3d=sample_goal,
                        num_steps=5  # Small number for testing
                    )
                    
                    logger.info(f"Generated trajectory with {len(trajectory)} steps")
                    for i, action in enumerate(trajectory):
                        logger.info(f"Step {i+1}: {action}")
                else:
                    logger.warning("Skipping trajectory generation as model could not be loaded")
            else:
                logger.warning(f"Model not found at {model_path}. Skipping test.")
                
        except Exception as e:
            logger.error(f"Error in example: {str(e)}")
    else:
        logger.warning("Stable-Baselines3 not installed. Skipping example.")
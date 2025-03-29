"""
Simulation environment for the Phosphobot Construct.

This module provides utilities for creating high-fidelity physics simulations
of the robot and its environment for pre-training.
"""

import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class PhosphobotSimulator:
    """
    High-fidelity physics simulator for the Phosphobot robot.
    
    This class provides an interface to pybullet for simulating
    the robot's interactions with its environment.
    """

    def __init__(
        self,
        gui: bool = False,
        robot_urdf_path: Optional[str] = None,
        gravity: List[float] = [0, 0, -9.81],
        timestep: float = 0.002
    ):
        """
        Initialize the simulator.
        
        Args:
            gui: Whether to use GUI for visualization.
            robot_urdf_path: Path to robot URDF file.
            gravity: Gravity vector.
            timestep: Simulation timestep.
        """
        # Connect to the physics server
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        
        # Set up physics parameters
        p.setGravity(*gravity)
        p.setTimeStep(timestep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot model
        if robot_urdf_path and os.path.exists(robot_urdf_path):
            self.robot_id = p.loadURDF(
                robot_urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            )
        else:
            # Use a default robot model if specific one not provided
            logger.warning("Robot URDF not found, using default robot")
            self.robot_id = p.loadURDF(
                "kuka_iiwa/model.urdf", 
                basePosition=[0, 0, 0],
                useFixedBase=True
            )
        
        # Store joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = range(self.num_joints)
        
        # Setup camera
        self.setup_cameras()
        
        # Initialize robot in a default pose
        self.reset_robot()
        
    def setup_cameras(self, num_cameras: int = 3) -> None:
        """
        Setup cameras for observation.
        
        Args:
            num_cameras: Number of cameras to setup.
        """
        self.cameras = []
        
        # Setup camera positions around the robot
        camera_positions = [
            [1.0, 0.0, 0.5],  # Front camera
            [0.0, 1.0, 0.5],  # Side camera
            [0.7, 0.7, 1.0]   # Top-angled camera
        ]
        
        camera_targets = [[0, 0, 0.2]] * num_cameras  # All cameras look at this point
        
        for i in range(min(num_cameras, len(camera_positions))):
            self.cameras.append({
                "position": camera_positions[i],
                "target": camera_targets[i],
                "up_vector": [0, 0, 1],
                "fov": 60,
                "aspect": 1.0,
                "near_val": 0.1,
                "far_val": 10.0
            })
    
    def reset_robot(self, joint_positions: Optional[List[float]] = None) -> None:
        """
        Reset the robot to a specified or default position.
        
        Args:
            joint_positions: Target joint positions. If None, uses default.
        """
        # Use default positions if not specified
        if joint_positions is None:
            joint_positions = [0.0] * self.num_joints
        
        # Set joint positions
        for i, pos in zip(self.joint_indices, joint_positions):
            p.resetJointState(self.robot_id, i, pos)
    
    def step_simulation(self, num_steps: int = 1) -> None:
        """
        Step the simulation forward.
        
        Args:
            num_steps: Number of simulation steps to take.
        """
        for _ in range(num_steps):
            p.stepSimulation()
    
    def set_joint_positions(self, positions: List[float]) -> None:
        """
        Set robot joint positions using position control.
        
        Args:
            positions: Target joint positions.
        """
        for i, pos in zip(self.joint_indices, positions):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos
            )
    
    def get_joint_states(self) -> Dict[str, np.ndarray]:
        """
        Get current joint states.
        
        Returns:
            Dictionary with joint positions, velocities, and reaction forces.
        """
        joint_states = [p.getJointState(self.robot_id, i) for i in self.joint_indices]
        
        positions = np.array([state[0] for state in joint_states])
        velocities = np.array([state[1] for state in joint_states])
        reaction_forces = np.array([state[2] for state in joint_states])
        
        return {
            "positions": positions,
            "velocities": velocities,
            "reaction_forces": reaction_forces
        }
    
    def capture_camera_image(self, camera_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Capture RGB and depth image from a camera.
        
        Args:
            camera_idx: Index of the camera to use.
            
        Returns:
            Dictionary with RGB and depth images.
        """
        if camera_idx >= len(self.cameras):
            raise ValueError(f"Camera index {camera_idx} out of range (max {len(self.cameras)-1})")
        
        camera = self.cameras[camera_idx]
        
        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera["position"],
            cameraTargetPosition=camera["target"],
            cameraUpVector=camera["up_vector"]
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera["fov"],
            aspect=camera["aspect"],
            nearVal=camera["near_val"],
            farVal=camera["far_val"]
        )
        
        # Capture image
        width, height = 320, 240
        img_data = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Extract RGB and depth images
        rgb_img = np.array(img_data[2]).reshape(height, width, 4)[:, :, :3]
        depth_img = np.array(img_data[3]).reshape(height, width)
        
        return {
            "rgb": rgb_img,
            "depth": depth_img
        }
    
    def load_objects(self, object_configs: List[Dict[str, Union[str, List[float], float]]]) -> List[int]:
        """
        Load objects into the simulation.
        
        Args:
            object_configs: List of object configurations with keys:
                - urdf_path: Path to URDF file
                - position: [x, y, z] position
                - orientation: [roll, pitch, yaw] orientation in radians
                - scale: Scaling factor (optional)
                
        Returns:
            List of object IDs.
        """
        object_ids = []
        
        for config in object_configs:
            # Extract parameters with defaults
            urdf_path = config["urdf_path"]
            position = config.get("position", [0, 0, 0])
            orientation_euler = config.get("orientation", [0, 0, 0])
            scale = config.get("scale", 1.0)
            
            # Convert Euler angles to quaternion
            orientation_quat = p.getQuaternionFromEuler(orientation_euler)
            
            # Load the object
            if os.path.exists(urdf_path):
                obj_id = p.loadURDF(
                    urdf_path,
                    basePosition=position,
                    baseOrientation=orientation_quat,
                    globalScaling=scale
                )
                object_ids.append(obj_id)
            else:
                logger.warning(f"Object URDF not found: {urdf_path}")
        
        return object_ids
    
    def close(self) -> None:
        """Disconnect from the physics server."""
        p.disconnect(self.physics_client)


def create_box_stacking_env() -> Tuple[PhosphobotSimulator, List[int]]:
    """
    Create a simulation environment for the box stacking task.
    
    Returns:
        Tuple of (simulator, box_ids)
    """
    # Initialize simulator
    sim = PhosphobotSimulator(gui=True)
    
    # Create boxes of different sizes
    box_configs = [
        {
            "urdf_path": "cube.urdf",
            "position": [-0.25, 0.15, 0.025],
            "scale": 0.25,  # Large box
            "color": [1, 0, 0, 1]  # Red
        },
        {
            "urdf_path": "cube.urdf",
            "position": [0.10, -0.20, 0.025],
            "scale": 0.15,  # Medium box
            "color": [0, 0, 1, 1]  # Blue
        },
        {
            "urdf_path": "cube.urdf",
            "position": [0.30, 0.25, 0.025],
            "scale": 0.10,  # Small box
            "color": [0, 1, 0, 1]  # Green
        }
    ]
    
    # Load boxes
    box_ids = sim.load_objects(box_configs)
    
    # Let the simulation settle
    for _ in range(100):
        sim.step_simulation()
    
    return sim, box_ids
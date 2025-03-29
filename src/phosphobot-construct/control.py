"""
Closed-Loop Control and Adaptation for the Phosphobot Construct.

This module provides real-time feedback control to handle disturbances
and uncertainties during task execution.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from phosphobot.api.client import PhosphoApi

logger = logging.getLogger(__name__)


class ClosedLoopController:
    """
    Real-time feedback controller for robot manipulation tasks.
    
    This class implements closed-loop control strategies to adapt
    to changes in the environment during task execution.
    """
    
    def __init__(
        self,
        client: PhosphoApi,
        feedback_rate: float = 30.0,
        error_threshold: float = 0.01,
        max_velocity: float = 0.5
    ):
        """
        Initialize the closed-loop controller.
        
        Args:
            client: PhosphoApi client for robot control.
            feedback_rate: Control loop frequency in Hz.
            error_threshold: Position error threshold for success.
            max_velocity: Maximum joint velocity in rad/s.
        """
        self.client = client
        self.dt = 1.0 / feedback_rate
        self.error_threshold = error_threshold
        self.max_velocity = max_velocity
        
        # PID controller gains (default values, should be tuned)
        self.kp = 2.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.3  # Derivative gain
        
        # State variables
        self.integral_error = np.zeros(6)  # Integral term for 6 DoF
        self.previous_error = np.zeros(6)  # Previous error for derivative term
    
    def pid_control(self, error: np.ndarray) -> np.ndarray:
        """
        PID control law for smooth trajectory following.
        
        Args:
            error: Position error vector.
            
        Returns:
            Control action (joint velocity).
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * self.dt
        i_term = self.ki * self.integral_error
        
        # Derivative term
        d_term = self.kd * (error - self.previous_error) / self.dt
        self.previous_error = error.copy()
        
        # Combine terms
        control = p_term + i_term + d_term
        
        # Limit control action
        control = np.clip(control, -self.max_velocity, self.max_velocity)
        
        return control
    
    def move_to_position(
        self,
        target_position: np.ndarray,
        feedback_func: Optional[Callable[[], np.ndarray]] = None,
        timeout: float = 10.0,
        stop_condition: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Move the robot to a target position with closed-loop control.
        
        Args:
            target_position: Target joint positions.
            feedback_func: Function to get current robot state. If None, uses client.
            timeout: Maximum execution time in seconds.
            stop_condition: Additional stopping condition. If None, uses error threshold.
            
        Returns:
            Dictionary with execution results.
        """
        logger.info(f"Moving to target position with closed-loop control")
        
        # Reset state variables
        self.integral_error = np.zeros(len(target_position))
        self.previous_error = np.zeros(len(target_position))
        
        # Get feedback function
        if feedback_func is None:
            feedback_func = lambda: np.array(self.client.control.read_joints().angles_rad)
        
        # Start control loop
        start_time = time.time()
        current_time = start_time
        current_position = feedback_func()
        
        # Initialize result
        result = {
            "success": False,
            "time_elapsed": 0.0,
            "final_position": current_position.tolist(),
            "final_error": None
        }
        
        # Main control loop
        while current_time - start_time < timeout:
            # Get current state
            try:
                current_position = feedback_func()
            except Exception as e:
                logger.error(f"Error getting feedback: {str(e)}")
                break
                
            # Compute error
            error = target_position - current_position
            error_norm = np.linalg.norm(error)
            
            # Check if goal is reached
            if error_norm < self.error_threshold:
                result["success"] = True
                break
                
            # Check additional stop condition
            if stop_condition is not None and stop_condition():
                break
                
            # Compute control action
            control = self.pid_control(error)
            
            # Apply control
            try:
                # Convert velocity to position increment
                next_position = current_position + control * self.dt
                
                # Send position command to robot
                self.client.control.write_joints(angles=next_position.tolist())
            except Exception as e:
                logger.error(f"Error applying control: {str(e)}")
                break
                
            # Log progress every second
            if int(current_time - start_time) != int(time.time() - start_time):
                logger.info(f"Error: {error_norm:.4f}, Time: {time.time() - start_time:.1f}s")
                
            # Wait for next control cycle
            time.sleep(self.dt)
            current_time = time.time()
        
        # Update result
        result["time_elapsed"] = current_time - start_time
        result["final_position"] = current_position.tolist()
        result["final_error"] = np.linalg.norm(target_position - current_position)
        
        logger.info(f"Motion completed with result: {result['success']}, error: {result['final_error']:.4f}")
        return result
    
    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        feedback_func: Optional[Callable[[], np.ndarray]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute a trajectory with closed-loop control.
        
        Args:
            trajectory: List of target joint positions.
            feedback_func: Function to get current robot state. If None, uses client.
            timeout: Maximum execution time in seconds.
            
        Returns:
            Dictionary with execution results.
        """
        logger.info(f"Executing trajectory with {len(trajectory)} waypoints")
        
        # Initialize result
        result = {
            "success": False,
            "time_elapsed": 0.0,
            "waypoints_reached": 0,
            "waypoint_results": []
        }
        
        # Start execution
        start_time = time.time()
        current_time = start_time
        
        # Execute each waypoint
        for i, waypoint in enumerate(trajectory):
            logger.info(f"Moving to waypoint {i+1}/{len(trajectory)}")
            
            # Check timeout
            if current_time - start_time >= timeout:
                logger.warning(f"Trajectory execution timed out after {i} waypoints")
                break
                
            # Move to waypoint
            waypoint_timeout = min(timeout - (current_time - start_time), 5.0)  # Limit per-waypoint timeout
            waypoint_result = self.move_to_position(
                target_position=waypoint,
                feedback_func=feedback_func,
                timeout=waypoint_timeout
            )
            
            # Update result
            result["waypoint_results"].append(waypoint_result)
            
            if waypoint_result["success"]:
                result["waypoints_reached"] += 1
            else:
                logger.warning(f"Failed to reach waypoint {i+1}, continuing to next waypoint")
            
            # Update time
            current_time = time.time()
        
        # Update final result
        result["time_elapsed"] = current_time - start_time
        result["success"] = result["waypoints_reached"] == len(trajectory)
        
        logger.info(f"Trajectory execution completed: {result['waypoints_reached']}/{len(trajectory)} waypoints reached")
        return result


def adaptive_control(
    client: PhosphoApi,
    target_positions: Union[np.ndarray, List[np.ndarray]],
    perception_func: Optional[Callable[[], Dict[str, Any]]] = None,
    feedback_rate: float = 30.0,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Implement adaptive control with real-time scene monitoring.
    
    Args:
        client: PhosphoApi client for robot control.
        target_positions: Target joint positions or trajectory.
        perception_func: Function to update scene perception. If None, uses pure feedback control.
        feedback_rate: Control loop frequency in Hz.
        timeout: Maximum execution time in seconds.
        
    Returns:
        Dictionary with execution results.
    """
    logger.info("Starting adaptive control execution")
    
    # Initialize controller
    controller = ClosedLoopController(
        client=client,
        feedback_rate=feedback_rate
    )
    
    # Check if input is a single position or a trajectory
    if isinstance(target_positions, list) and len(target_positions) > 0 and isinstance(target_positions[0], np.ndarray):
        # Execute trajectory
        trajectory = target_positions
        
        # Define feedback function with perception if available
        if perception_func is not None:
            def enhanced_feedback():
                # Get joint positions
                joint_positions = np.array(client.control.read_joints().angles_rad)
                
                # Update perception every 10 cycles
                if enhanced_feedback.counter % 10 == 0:
                    try:
                        scene = perception_func()
                        enhanced_feedback.last_scene = scene
                    except Exception as e:
                        logger.error(f"Error in perception: {str(e)}")
                
                enhanced_feedback.counter += 1
                return joint_positions
                
            # Initialize function attributes
            enhanced_feedback.counter = 0
            enhanced_feedback.last_scene = None
            
            # Execute with enhanced feedback
            result = controller.execute_trajectory(
                trajectory=trajectory,
                feedback_func=enhanced_feedback,
                timeout=timeout
            )
            
            # Add perception information
            result["final_scene"] = enhanced_feedback.last_scene
            
        else:
            # Execute with normal feedback
            result = controller.execute_trajectory(
                trajectory=trajectory,
                timeout=timeout
            )
    else:
        # Convert single position to numpy array if needed
        if not isinstance(target_positions, np.ndarray):
            target_position = np.array(target_positions)
        else:
            target_position = target_positions
            
        # Execute single movement
        result = controller.move_to_position(
            target_position=target_position,
            timeout=timeout
        )
    
    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the controller with a sample movement
    try:
        # Connect to the robot
        client = PhosphoApi(base_url="http://localhost:80")
        
        # Initialize the robot
        client.move.init()
        time.sleep(2)  # Wait for robot to reach initial position
        
        # Get current joint positions
        current_joints = np.array(client.control.read_joints().angles_rad)
        logger.info(f"Current joint positions: {current_joints}")
        
        # Create target position (small offset from current)
        target_joints = current_joints.copy()
        target_joints[0] += 0.2  # Small movement of first joint
        
        # Execute movement with closed-loop control
        result = adaptive_control(
            client=client,
            target_positions=target_joints,
            timeout=5.0
        )
        
        # Print result
        logger.info(f"Movement result: {result}")
        
    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")
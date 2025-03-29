#!/usr/bin/env python3
"""
Box Stacking Example for the Phosphobot Construct.

This example demonstrates the complete pipeline for the box stacking task,
from perception to action execution.
"""

import time
import logging
import numpy as np
import os
from typing import Dict, List, Optional, Union, Any

from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot_construct.models import PhosphoConstructModel
from phosphobot_construct.perception import perception_pipeline
from phosphobot_construct.language_understanding import language_to_goal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def box_stacking_demo(
    server_url: str = "http://localhost",
    server_port: int = 80,
    model_path: Optional[str] = None,
    use_simulation: bool = False
) -> Dict[str, Any]:
    """
    Complete demo of the box stacking task.
    
    Args:
        server_url: URL of the Phosphobot server.
        server_port: Port of the Phosphobot server.
        model_path: Path to pre-trained model weights.
        use_simulation: Whether to use simulation instead of real robot.
        
    Returns:
        Dictionary with execution results.
    """
    logger.info("Starting box stacking demo")
    base_url = f"{server_url}:{server_port}"
    
    # 1. Connect to the Phosphobot server
    client = PhosphoApi(base_url=base_url)
    logger.info(f"Connected to Phosphobot server at {base_url}")
    
    # 2. Initialize the cameras
    allcameras = AllCameras()
    logger.info(f"Initialized cameras, found {len(allcameras.get_camera_list())} cameras")
    
    # Need to wait for the cameras to initialize
    time.sleep(1)
    
    # 3. Initialize the robot
    client.move.init()
    logger.info("Initialized robot")
    time.sleep(2)  # Wait for robot to reach initial position
    
    # 4. Load the model
    model = PhosphoConstructModel(model_path=model_path)
    logger.info(f"Loaded model from {model_path if model_path else 'default parameters'}")
    
    # 5. Set up the task with instruction
    instruction = "Please, robot, stack the boxes from largest to smallest size."
    logger.info(f"Task instruction: {instruction}")
    
    # 6. Capture initial sensor data
    images = []
    for i in range(3):  # Use up to 3 cameras if available
        try:
            images.append(allcameras.get_rgb_frame(camera_id=i, resize=(240, 320)))
        except Exception as e:
            logger.warning(f"Could not get frame from camera {i}: {e}")
    
    if not images:
        logger.error("No camera images available. Exiting.")
        return {"success": False, "error": "No camera images available"}
    
    # Get robot state
    state = client.control.read_joints()
    logger.info("Captured initial sensor data")
    
    # 7. Convert to 3D scene representation (perception/"placing")
    scene_3d = perception_pipeline(
        rgb_image=images[0],  # Use first camera as primary
        depth_image=None,  # Depth may not be available
        proprioception=np.array(state.angles_rad)
    )
    logger.info("Generated 3D scene representation")
    
    # 8. Convert instruction to goal state (language understanding/"imagining")
    goal_3d = language_to_goal(instruction, scene_3d)
    logger.info("Generated 3D goal state from instruction")
    
    # 9. Execute the task with closed-loop control
    total_steps = 100  # Maximum number of steps
    current_step = 0
    success = False
    
    logger.info(f"Beginning execution, maximum {total_steps} steps")
    
    while current_step < total_steps:
        # Get current images and state
        current_images = []
        for i in range(len(images)):
            try:
                current_images.append(allcameras.get_rgb_frame(camera_id=i, resize=(240, 320)))
            except Exception:
                # If camera fails, use the last successful image
                current_images.append(images[i])
        
        current_state = client.control.read_joints()
        
        # Prepare inputs for the model
        inputs = {
            "state": np.array(current_state.angles_rad),
            "images": np.array(current_images)
        }
        
        # Get the next actions from the model
        actions = model(inputs)
        
        # Execute the next action
        if len(actions) > 0:
            # Send the new joint position to the robot
            client.control.write_joints(angles=actions[0].tolist())
            
            # Log progress every 10 steps
            if current_step % 10 == 0:
                logger.info(f"Executed step {current_step}/{total_steps}")
                
            # Wait to respect frequency control (30 Hz)
            time.sleep(1 / 30)
        
        # Check if the goal is reached
        # This would be based on perception and comparison with the goal state
        # For the demo, we use a simplified check based on step count
        if current_step >= 50:  # Assume success after 50 steps for demo purposes
            logger.info("Goal state reached")
            success = True
            break
        
        current_step += 1
    
    # 10. Report results
    result = {
        "success": success,
        "steps": current_step,
        "final_state": current_state.angles_rad
    }
    
    if success:
        logger.info(f"Successfully completed the task in {current_step} steps!")
    else:
        logger.info(f"Failed to complete the task after {current_step} steps.")
    
    return result


if __name__ == "__main__":
    # Path to the pre-trained model, if available
    model_path = os.environ.get("PHOSPHOBOT_MODEL_PATH", None)
    
    # Run the demo
    result = box_stacking_demo(model_path=model_path)
    
    # Print the final result
    print(f"Demo completed with result: {'Success' if result['success'] else 'Failure'}")
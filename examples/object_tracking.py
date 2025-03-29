#!/usr/bin/env python3
"""
Object Tracking Example for the Phosphobot Construct.

This example demonstrates object tracking and following using the
perception pipeline and closed-loop control.
"""

import time
import logging
import numpy as np
import os
from typing import Dict, List, Optional, Union, Any

from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot_construct.perception import perception_pipeline
from phosphobot_construct.control import adaptive_control
from phosphobot_construct.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_target_object(
    scene_3d: Dict[str, Any],
    target_class: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Find a target object in the scene.
    
    Args:
        scene_3d: 3D scene representation.
        target_class: Target object class to track. If None, track any object.
        
    Returns:
        Target object data or None if not found.
    """
    # Extract objects
    objects = scene_3d.get("objects", [])
    
    if not objects:
        logger.warning("No objects found in scene")
        return None
    
    # Find target object
    if target_class:
        # Find object by class
        for obj in objects:
            if target_class.lower() in obj.get("class", "").lower():
                return obj
        
        logger.warning(f"No object matching class '{target_class}' found")
        return None
    else:
        # Use the object with highest confidence
        objects_with_confidence = [(i, obj.get("confidence", 0)) for i, obj in enumerate(objects)]
        if not objects_with_confidence:
            return None
            
        best_idx, _ = max(objects_with_confidence, key=lambda x: x[1])
        return objects[best_idx]


def calculate_target_position(
    obj: Dict[str, Any],
    current_position: np.ndarray,
    z_offset: float = 0.15
) -> np.ndarray:
    """
    Calculate target position for tracking object.
    
    Args:
        obj: Target object data.
        current_position: Current robot position.
        z_offset: Height offset for tracking.
        
    Returns:
        Target position array.
    """
    # Get object position
    if "position_3d" not in obj:
        logger.warning("Object has no 3D position information")
        return current_position
    
    pos = obj["position_3d"]
    
    # Extract position components with fallbacks
    obj_x = pos.get("x", 0)
    obj_y = pos.get("y", 0)
    obj_z = pos.get("z", 0)
    
    # Calculate offset position above object
    # Keep same orientation
    target_position = current_position.copy()
    
    # Update position components (first 3 values in joint state)
    # Scale factor converts from scene coordinates to robot joint coordinates
    scale = 50.0  # This would depend on the robot kinematics and scene scale
    
    # Keep current position but modify according to object position
    target_position[0] = obj_x * scale
    target_position[1] = obj_y * scale
    target_position[2] = (obj_z + z_offset) * scale
    
    return target_position


def update_scene_perception(
    cameras: AllCameras,
    client: PhosphoApi,
    primary_camera_idx: int = 0
) -> Dict[str, Any]:
    """
    Update the scene perception using current camera data.
    
    Args:
        cameras: Camera interface.
        client: Robot client.
        primary_camera_idx: Index of primary camera.
        
    Returns:
        Updated 3D scene representation.
    """
    # Get camera frame
    try:
        rgb_image = cameras.get_rgb_frame(camera_id=primary_camera_idx, resize=(240, 320))
    except Exception as e:
        logger.error(f"Could not get frame from camera {primary_camera_idx}: {e}")
        # Create blank image as fallback
        rgb_image = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # Get robot state
    state = client.control.read_joints()
    
    # Process scene
    scene_3d = perception_pipeline(
        rgb_image=rgb_image,
        depth_image=None,
        proprioception=np.array(state.angles_rad)
    )
    
    return scene_3d


def object_tracking_demo(
    server_url: str = "http://localhost",
    server_port: int = 80,
    target_class: Optional[str] = None,
    max_duration: float = 60.0
) -> Dict[str, Any]:
    """
    Complete demo of the object tracking task.
    
    Args:
        server_url: URL of the Phosphobot server.
        server_port: Port of the Phosphobot server.
        target_class: Target object class to track. If None, track any object.
        max_duration: Maximum duration in seconds.
        
    Returns:
        Dictionary with execution results.
    """
    logger.info("Starting object tracking demo")
    base_url = f"{server_url}:{server_port}"
    
    # Load configuration
    config = get_config()
    
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
    
    # 4. Initial scene perception
    scene_3d = update_scene_perception(allcameras, client)
    
    # Find target object
    target_obj = get_target_object(scene_3d, target_class)
    
    if not target_obj:
        logger.error("No target object found. Exiting.")
        return {"success": False, "error": "No target object found"}
    
    logger.info(f"Found target object: {target_obj.get('class', 'unknown')}")
    
    # Get current position
    current_position = np.array(client.control.read_joints().angles_rad)
    
    # 5. Start tracking loop
    logger.info("Starting object tracking loop")
    start_time = time.time()
    tracking_active = True
    
    results = {
        "success": False,
        "tracked_object": target_obj.get("class", "unknown"),
        "tracking_duration": 0.0,
        "lost_track_count": 0,
        "trajectory": []
    }
    
    # Create perception function for closed-loop control
    def perception_func():
        nonlocal target_obj
        
        # Update scene perception
        scene = update_scene_perception(allcameras, client)
        
        # Find target object
        new_target = get_target_object(scene, target_class)
        
        if new_target:
            target_obj = new_target
            return scene
        else:
            # Keep using previous target if not found
            logger.warning("Lost track of target object")
            results["lost_track_count"] += 1
            return scene
    
    while tracking_active and (time.time() - start_time) < max_duration:
        try:
            # Calculate target position
            target_position = calculate_target_position(
                target_obj, 
                current_position,
                z_offset=0.15
            )
            
            # Move robot to track object with closed-loop control
            move_result = adaptive_control(
                client=client,
                target_positions=target_position,
                perception_func=perception_func,
                feedback_rate=config.get("robot.frequency", 30),
                timeout=2.0  # Short timeout for responsiveness
            )
            
            # Update current position
            current_position = np.array(client.control.read_joints().angles_rad)
            
            # Record trajectory
            results["trajectory"].append(current_position.tolist())
            
            # Check if tracking is lost
            if results["lost_track_count"] > 5:
                logger.warning("Lost track of object for too long. Stopping.")
                tracking_active = False
            
            # Log progress every 5 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0:
                logger.info(f"Tracking for {elapsed:.1f}s, lost track {results['lost_track_count']} times")
                
        except KeyboardInterrupt:
            logger.info("Tracking interrupted by user")
            tracking_active = False
            
        except Exception as e:
            logger.error(f"Error during tracking: {str(e)}")
            tracking_active = False
    
    # 6. Record results
    tracking_duration = time.time() - start_time
    results["tracking_duration"] = tracking_duration
    results["success"] = tracking_duration >= 5.0  # Minimum tracking time for success
    
    if tracking_duration >= max_duration:
        logger.info(f"Tracking completed successfully for maximum duration of {max_duration:.1f}s")
    else:
        logger.info(f"Tracking completed for {tracking_duration:.1f}s")
    
    return results


def main():
    """Main function for running the example."""
    try:
        # Get configuration
        config = get_config()
        
        # Set up parameters
        server_url = config.get("robot.server_url", "http://localhost")
        server_port = config.get("robot.server_port", 80)
        
        # Define target class (use "cube" as default for tracking a cube)
        target_class = "cube"
        
        # Run the demo
        result = object_tracking_demo(
            server_url=server_url,
            server_port=server_port,
            target_class=target_class,
            max_duration=60.0
        )
        
        # Print result summary
        print("\nTracking Result Summary:")
        print(f"  Success: {result['success']}")
        print(f"  Tracked object: {result['tracked_object']}")
        print(f"  Duration: {result['tracking_duration']:.1f}s")
        print(f"  Lost track count: {result['lost_track_count']}")
        print(f"  Trajectory points: {len(result['trajectory'])}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in object tracking demo: {str(e)}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    main()
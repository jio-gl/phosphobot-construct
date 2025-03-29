"""
Command-Line Interface for the Phosphobot Construct.

This module provides a command-line interface to run various components
of the Phosphobot Construct system.
"""

import os
import sys
import argparse
import logging
import time
from typing import Dict, List, Optional, Union, Any

from phosphobot_construct.config import get_config, create_parser
from phosphobot_construct.scenario_generator import generate_tabletop_scenarios
from phosphobot_construct.goal_generator import generate_goals_for_scenarios
from phosphobot_construct.text_to_3d import convert_scenarios_to_3d
from phosphobot_construct.sensor_generator import generate_training_data
from phosphobot_construct.reinforcement_learning import train_robot_policy

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Add file handler if configured
    config = get_config()
    if config.get("logging.log_to_file", False):
        log_file = config.get("logging.log_file", "phosphobot_construct.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)


def generate_data_command(args: argparse.Namespace) -> int:
    """
    Command to generate training data.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    config = get_config(args.config)
    
    try:
        # Create data directory
        data_dir = config.get("paths.data_dir", "data")
        scenarios_dir = os.path.join(data_dir, "scenarios")
        goals_dir = os.path.join(data_dir, "goals")
        models_dir = os.path.join(data_dir, "models")
        sensor_data_dir = os.path.join(data_dir, "sensor_data")
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(scenarios_dir, exist_ok=True)
        os.makedirs(goals_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(sensor_data_dir, exist_ok=True)
        
        if args.all or args.scenarios:
            # Generate scenarios
            num_scenarios = args.num_scenarios or config.get("data.num_scenarios", 100)
            logger.info(f"Generating {num_scenarios} scenarios")
            
            generate_tabletop_scenarios(
                num_scenarios=num_scenarios,
                output_dir=scenarios_dir
            )
        
        if args.all or args.goals:
            # Generate goals
            goals_per_scenario = args.goals_per_scenario or config.get("data.goals_per_scenario", 5)
            logger.info(f"Generating goals with {goals_per_scenario} goals per scenario")
            
            generate_goals_for_scenarios(
                scenarios_dir=scenarios_dir,
                goals_dir=goals_dir,
                goals_per_scenario=goals_per_scenario
            )
        
        if args.all or args.models:
            # Convert to 3D models
            logger.info("Converting scenarios to 3D models")
            
            convert_scenarios_to_3d(
                scenarios_dir=scenarios_dir,
                output_dir=models_dir,
                device="cuda" if args.gpu else "cpu"
            )
        
        if args.all or args.sensor:
            # Generate sensor data
            logger.info("Generating sensor data")
            
            # Load 3D models
            scene_3d_models = []
            for model_file in os.listdir(models_dir):
                if model_file.endswith(".glb") or model_file.endswith(".obj"):
                    model_path = os.path.join(models_dir, model_file)
                    scene_3d_models.append({"3d_model_path": model_path})
            
            # Generate data
            image_size = config.get("rendering.image_size", [320, 240])
            num_cameras = config.get("rendering.num_cameras", 3)
            
            generate_training_data(
                scene_3d_models=scene_3d_models,
                output_dir=sensor_data_dir,
                num_cameras=num_cameras,
                image_size=tuple(image_size)
            )
        
        logger.info("Data generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        return 1


def train_command(args: argparse.Namespace) -> int:
    """
    Command to train models.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    config = get_config(args.config)
    
    try:
        # Get paths
        models_dir = config.get("paths.models_dir", "models")
        data_dir = config.get("paths.data_dir", "data")
        output_dir = os.path.join(models_dir, args.model_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare environment data
        env_data = {
            "use_simulator": True,
            "max_steps": config.get("training.max_steps", 100),
            "reward_type": config.get("training.reward_type", "exponential"),
            "scene": {},
            "goal": {}
        }
        
        # Load scene and goal data if available
        scene_file = args.scene_file
        goal_file = args.goal_file
        
        if scene_file and os.path.exists(scene_file):
            import json
            with open(scene_file, "r") as f:
                env_data["scene"] = json.load(f)
        
        if goal_file and os.path.exists(goal_file):
            import json
            with open(goal_file, "r") as f:
                env_data["goal"] = json.load(f)
        
        # Train the model
        logger.info(f"Training model {args.model_name}")
        
        total_timesteps = args.timesteps or config.get("training.total_timesteps", 100000)
        
        model_path = train_robot_policy(
            env_data=env_data,
            output_dir=output_dir,
            total_timesteps=total_timesteps,
            use_transformer=not args.no_transformer,
            seed=args.seed,
            use_language=args.use_language
        )
        
        if model_path:
            logger.info(f"Model trained successfully and saved to {model_path}")
            return 0
        else:
            logger.error("Model training failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return 1


def run_example_command(args: argparse.Namespace) -> int:
    """
    Command to run examples.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    config = get_config(args.config)
    
    try:
        # Get example name
        example_name = args.example
        
        # Define available examples
        examples = {
            "box_stacking": "examples.box_stacking",
        }
        
        if example_name not in examples:
            logger.error(f"Unknown example: {example_name}")
            logger.info(f"Available examples: {', '.join(examples.keys())}")
            return 1
        
        # Import the example module
        import importlib
        example_module = importlib.import_module(examples[example_name])
        
        # Get the main function
        if hasattr(example_module, "main"):
            main_func = example_module.main
        else:
            # Look for a function named after the example
            main_func = getattr(example_module, f"{example_name}_demo", None)
            if main_func is None:
                logger.error(f"Example {example_name} does not have a main function")
                return 1
        
        # Run the example
        logger.info(f"Running example: {example_name}")
        result = main_func()
        
        # Check result
        if isinstance(result, dict) and "success" in result:
            if result["success"]:
                logger.info("Example completed successfully")
                return 0
            else:
                logger.warning("Example completed with errors")
                return 1
        
        logger.info("Example completed")
        return 0
        
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
        return 1


def run_command(args: argparse.Namespace) -> int:
    """
    Command to run the robot with a trained model.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code.
    """
    config = get_config(args.config)
    
    try:
        # Import required modules
        from phosphobot.api.client import PhosphoApi
        from phosphobot.camera import AllCameras
        from phosphobot_construct.models import PhosphoConstructModel
        from phosphobot_construct.perception import perception_pipeline
        from phosphobot_construct.language_understanding import language_to_goal
        from phosphobot_construct.policy import execute_policy
        
        # Connect to the robot
        server_url = args.server_url or config.get("robot.server_url", "http://localhost")
        server_port = args.server_port or config.get("robot.server_port", 80)
        
        logger.info(f"Connecting to robot at {server_url}:{server_port}")
        client = PhosphoApi(base_url=f"{server_url}:{server_port}")
        
        # Initialize cameras
        logger.info("Initializing cameras")
        cameras = AllCameras()
        time.sleep(1)  # Wait for cameras to initialize
        
        # Initialize the robot
        logger.info("Initializing robot")
        client.move.init()
        time.sleep(2)  # Wait for robot to reach initial position
        
        # Load model
        model_path = args.model_path
        logger.info(f"Loading model from {model_path}")
        model = PhosphoConstructModel(model_path=model_path)
        
        # Get instruction
        instruction = args.instruction
        logger.info(f"Using instruction: {instruction}")
        
        # Get sensor data
        logger.info("Capturing sensor data")
        images = []
        for i in range(3):  # Use up to 3 cameras if available
            try:
                images.append(cameras.get_rgb_frame(camera_id=i, resize=(240, 320)))
            except Exception as e:
                logger.warning(f"Could not get frame from camera {i}: {e}")
        
        if not images:
            logger.error("No camera images available. Exiting.")
            return 1
        
        # Get robot state
        state = client.control.read_joints()
        
        # Process scene
        logger.info("Processing scene")
        scene_3d = perception_pipeline(
            rgb_image=images[0],  # Use first camera as primary
            depth_image=None,  # Depth may not be available
            proprioception=state.angles_rad
        )
        
        # Process instruction
        logger.info("Processing instruction")
        goal_3d = language_to_goal(instruction, scene_3d)
        
        # Execute policy
        logger.info("Executing policy")
        result = execute_policy(
            model_path=model_path,
            scene_3d=scene_3d,
            goal_3d=goal_3d,
            client=client,
            max_steps=args.max_steps or config.get("robot.max_steps", 100)
        )
        
        # Check result
        if result["success"]:
            logger.info("Task completed successfully")
            return 0
        else:
            logger.warning("Task did not complete successfully")
            return 1
        
    except Exception as e:
        logger.error(f"Error running robot: {str(e)}")
        return 1


def main() -> int:
    """
    Main function for the command-line interface.
    
    Returns:
        Exit code.
    """
    # Create parent parser
    parser = create_parser()
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate data command
    generate_parser = subparsers.add_parser("generate", help="Generate training data")
    generate_parser.add_argument("--all", action="store_true", help="Generate all data types")
    generate_parser.add_argument("--scenarios", action="store_true", help="Generate scenarios")
    generate_parser.add_argument("--goals", action="store_true", help="Generate goals")
    generate_parser.add_argument("--models", action="store_true", help="Convert to 3D models")
    generate_parser.add_argument("--sensor", action="store_true", help="Generate sensor data")
    generate_parser.add_argument("--num-scenarios", type=int, help="Number of scenarios to generate")
    generate_parser.add_argument("--goals-per-scenario", type=int, help="Number of goals per scenario")
    generate_parser.add_argument("--gpu", action="store_true", help="Use GPU for rendering")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model_name", type=str, help="Name of the model to train")
    train_parser.add_argument("--scene-file", type=str, help="Path to scene JSON file")
    train_parser.add_argument("--goal-file", type=str, help="Path to goal JSON file")
    train_parser.add_argument("--timesteps", type=int, help="Total timesteps for training")
    train_parser.add_argument("--no-transformer", action="store_true", help="Disable transformer-based policy")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--use-language", action="store_true", help="Enable language conditioning")
    
    # Run example command
    example_parser = subparsers.add_parser("example", help="Run an example")
    example_parser.add_argument("example", type=str, help="Name of the example to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run robot with a trained model")
    run_parser.add_argument("model_path", type=str, help="Path to the trained model")
    run_parser.add_argument("instruction", type=str, help="Natural language instruction")
    run_parser.add_argument("--server-url", type=str, help="Robot server URL")
    run_parser.add_argument("--server-port", type=int, help="Robot server port")
    run_parser.add_argument("--max-steps", type=int, help="Maximum number of steps")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Set up logging
    setup_logging(config.get("logging.level", "INFO"))
    
    # Execute command
    if args.command == "generate":
        return generate_data_command(args)
    elif args.command == "train":
        return train_command(args)
    elif args.command == "example":
        return run_example_command(args)
    elif args.command == "run":
        return run_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Create necessary directory structure for the Phosphobot Construct project.

This script sets up the directory structure required for the project to function
properly, including data directories, model directories, and other necessary folders.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create the required directory structure for the project."""
    # Define the directory structure
    directories = [
        # Source code directories
        "src/phosphobot_construct",
        
        # Test directories
        "tests",
        
        # Data directories
        "data/scenarios",
        "data/goals",
        "data/models",
        "data/samples",
        
        # Model directories
        "models",
        "models/checkpoints",
        
        # Example directories
        "examples",
        
        # Script directories
        "scripts",
        
        # Logging directories
        "logs",
        
        # Documentation directories
        "docs",
        
        # Configuration directories
        "config"
    ]
    
    # Create each directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
        else:
            logger.info(f"Directory already exists: {dir_path}")


def create_init_files():
    """Create __init__.py files in all source code directories."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(root_dir, "src")
    
    # Walk through the src directory and create __init__.py files
    for root, dirs, files in os.walk(src_dir):
        if "__init__.py" not in files:
            init_path = os.path.join(root, "__init__.py")
            with open(init_path, "w") as f:
                package_name = os.path.basename(root)
                f.write(f'"""\n{package_name} package.\n"""\n\n')
            logger.info(f"Created __init__.py file in: {root}")


def create_sample_data():
    """Create sample data files for testing."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    samples_dir = os.path.join(root_dir, "data", "samples")
    
    # Create a sample configuration file
    config_path = os.path.join(root_dir, "config", "default_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("""{
  "simulation": {
    "gui": false,
    "timestep": 0.002,
    "gravity": [0, 0, -9.81]
  },
  "training": {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "n_epochs": 10,
    "gamma": 0.99,
    "reward_type": "exponential"
  },
  "perception": {
    "use_clip": true,
    "use_sam": true
  },
  "logging": {
    "level": "INFO",
    "save_trajectories": true
  }
}""")
        logger.info(f"Created sample configuration file: {config_path}")
    
    # Create a placeholder for sample images
    placeholder_path = os.path.join(samples_dir, "README.md")
    if not os.path.exists(placeholder_path):
        with open(placeholder_path, "w") as f:
            f.write("""# Sample Data

This directory contains sample data for testing the Phosphobot Construct.

## Contents

- Place sample images in this directory
- Use RGB or RGB-D images in JPG or PNG format
- Typical image resolution: 640x480
""")
        logger.info(f"Created placeholder for sample data: {placeholder_path}")


if __name__ == "__main__":
    logger.info("Creating directory structure for Phosphobot Construct")
    create_directory_structure()
    create_init_files()
    create_sample_data()
    logger.info("Directory structure setup complete")
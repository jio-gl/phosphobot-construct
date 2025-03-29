"""
Configuration Management for the Phosphobot Construct.

This module provides a unified way to manage configuration options
for the entire system, with support for different modes and environments.
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Union, Any, TextIO
import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for the Phosphobot Construct system.
    
    This class handles loading, saving, and accessing configuration values
    with support for hierarchical configurations and environment overrides.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file.
        """
        # Set default configuration
        self.config = self._get_default_config()
        
        # Try to load configuration from provided path
        if config_path:
            self.load_config(config_path)
        else:
            # Try to find config in common locations
            self._load_from_common_locations()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration.
        
        Returns:
            Dictionary with default configuration.
        """
        return {
            "simulation": {
                "gui": False,
                "timestep": 0.002,
                "gravity": [0, 0, -9.81],
                "use_physics": True
            },
            "training": {
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_steps": 2048,
                "n_epochs": 10,
                "gamma": 0.99,
                "reward_type": "exponential",
                "max_steps": 100,
                "total_timesteps": 100000
            },
            "perception": {
                "use_clip": True,
                "use_sam": True,
                "clip_model": "ViT-B/32",
                "sam_checkpoint": "sam_vit_h_4b8939.pth"
            },
            "language": {
                "use_openai": True,
                "model": "gpt-4o",
                "temperature": 0.2,
                "api_key_env": "OPENAI_API_KEY"
            },
            "rendering": {
                "image_size": [320, 240],
                "num_cameras": 3,
                "use_pytorch3d": True
            },
            "robot": {
                "server_url": "http://localhost",
                "server_port": 80,
                "frequency": 30,
                "use_closed_loop": True
            },
            "paths": {
                "data_dir": "data",
                "models_dir": "models",
                "logs_dir": "logs",
                "output_dir": "output"
            },
            "logging": {
                "level": "INFO",
                "save_trajectories": True,
                "log_to_file": False,
                "log_file": "phosphobot_construct.log"
            }
        }
    
    def _load_from_common_locations(self) -> bool:
        """
        Try to load configuration from common locations.
        
        Returns:
            True if configuration was loaded, False otherwise.
        """
        # List of common config locations
        config_paths = [
            "config/config.yaml",
            "config/config.json",
            "config/default_config.yaml",
            "config/default_config.json",
            os.path.expanduser("~/.config/phosphobot/config.yaml"),
            os.path.expanduser("~/.phosphobot/config.yaml")
        ]
        
        # Try each location
        for path in config_paths:
            if os.path.exists(path):
                try:
                    self.load_config(path)
                    logger.info(f"Loaded configuration from {path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load configuration from {path}: {str(e)}")
        
        logger.info("No configuration file found, using defaults")
        return False
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dictionary with loaded configuration.
        """
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return self.config
        
        try:
            # Determine file format from extension
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
            elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                with open(config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported file format: {config_path}")
                return self.config
            
            # Merge loaded config with default config
            self._merge_configs(self.config, loaded_config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self.config
    
    def _merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configurations.
        
        Args:
            base: Base configuration to merge into.
            update: Updates to apply.
            
        Returns:
            Merged configuration.
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._merge_configs(base[key], value)
            else:
                # Update or add value
                base[key] = value
        
        return base
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot notation for nested keys).
            default: Default value if key is not found.
            
        Returns:
            Configuration value or default.
        """
        # Split key into parts
        parts = key.split(".")
        
        # Traverse the config hierarchy
        value = self.config
        for part in parts:
            if part in value:
                value = value[part]
            else:
                return default
        
        return value
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key with intelligent type conversion.
        
        Args:
            key: Configuration key (dot notation for nested keys).
            value: Value to set.
        """
        # Perform intelligent type conversion
        value = self._convert_value_type(key, value)
        
        # Split key into parts
        parts = key.split(".")
        
        # Traverse the config hierarchy
        config = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            config = config[part]
        
        # Set the value
        config[parts[-1]] = value

    def _convert_value_type(self, key: str, value: Any) -> Any:
        """
        Convert value to the appropriate type based on key and existing values.
        
        Args:
            key: Configuration key.
            value: Value to convert.
            
        Returns:
            Converted value.
        """
        # Skip conversion for None or non-string values
        if value is None or not isinstance(value, str):
            return value
        
        # Get current value if exists
        current_value = self.get(key)
        
        # Try to match the type of the current value
        if current_value is not None:
            if isinstance(current_value, bool):
                # Convert to boolean
                return value.lower() in ('true', 'yes', '1', 'y', 'on')
            elif isinstance(current_value, int):
                # Convert to integer
                try:
                    return int(value)
                except ValueError:
                    pass
            elif isinstance(current_value, float):
                # Convert to float
                try:
                    return float(value)
                except ValueError:
                    pass
            elif isinstance(current_value, list):
                # Try to parse as JSON list
                try:
                    import json
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    # Try comma-separated list
                    return [item.strip() for item in value.split(',')]
        
        # Apply generic conversions for strings
        if value.lower() in ('true', 'yes', 'y', 'on'):
            return True
        elif value.lower() in ('false', 'no', 'n', 'off'):
            return False
        elif value.isdigit():
            return int(value)
        elif self._is_float(value):
            return float(value)
            
        # Keep as string if no conversion applies
        return value

    def _is_float(self, value: str) -> bool:
        """Check if a string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Determine file format from extension
            if config_path.endswith(".json"):
                with open(config_path, "w") as f:
                    json.dump(self.config, f, indent=2)
            elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
                with open(config_path, "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                logger.warning(f"Unsupported file format: {config_path}")
                return False
            
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def update_from_env(self, prefix: str = "PHOSPHOBOT_") -> None:
        """
        Update configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables.
        """
        # Get all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower().replace("__", ".")
                
                # Try to parse the value
                try:
                    # Try as JSON
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # Use as string
                    parsed_value = value
                
                # Set the value
                self.set(config_key, parsed_value)
                logger.debug(f"Updated config from environment: {config_key}={parsed_value}")
                
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Command-line arguments.
        """
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Track updates for debugging
        updated_keys = []
        
        # Update config with args
        for key, value in args_dict.items():
            if value is not None:
                # Split the key at each underscore that separates sections
                # This handles keys like simulation_use_physics correctly
                parts = key.split('_')
                
                # First part is the section (e.g., 'simulation')
                section = parts[0]
                
                # The rest is the field path (could be multiple parts)
                if len(parts) > 1:
                    field = '.'.join(parts[1:])
                    config_key = f"{section}.{field}"
                else:
                    config_key = section
                
                # Get current value for comparison
                current_value = self.get(config_key)
                
                # Only update if the value is different (prevents overwriting with the same value)
                if current_value != value:
                    # Set the value
                    self.set(config_key, value)
                    updated_keys.append((config_key, current_value, value))
                    logger.debug(f"Updated config from args: {config_key}={value} (was {current_value})")
        
        # Log summary of updates
        if updated_keys:
            logger.info(f"Updated {len(updated_keys)} configuration values from command-line arguments")
            
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Dictionary with the current configuration.
        """
        return self.config
    
    def print_config(self, file: Optional[TextIO] = None) -> None:
        """
        Print the current configuration.
        
        Args:
            file: File to print to. If None, prints to stdout.
        """
        json.dump(self.config, file or open(os.devnull, "w"), indent=2)


# Global configuration instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        ConfigManager instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance


def create_parser() -> argparse.ArgumentParser:
    """
    Create a command-line argument parser with common configuration options.
    
    Returns:
        ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Phosphobot Construct")
    
    # General options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--logging_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Logging level")
    
    # Simulation options
    parser.add_argument("--simulation_gui", action="store_true", help="Enable GUI for simulation")
    parser.add_argument("--simulation_timestep", type=float, help="Simulation timestep")
    
    # Training options
    parser.add_argument("--training_learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--training_batch_size", type=int, help="Batch size for training")
    parser.add_argument("--training_total_timesteps", type=int, help="Total timesteps for training")
    
    # Robot options
    parser.add_argument("--robot_server_url", type=str, help="Robot server URL")
    parser.add_argument("--robot_server_port", type=int, help="Robot server port")
    
    # Path options
    parser.add_argument("--paths_data_dir", type=str, help="Data directory")
    parser.add_argument("--paths_models_dir", type=str, help="Models directory")
    parser.add_argument("--paths_output_dir", type=str, help="Output directory")
    
    return parser


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.config)
    
    # Update from environment variables
    config.update_from_env()
    
    # Update from command-line arguments
    config.update_from_args(args)
    
    # Print the configuration
    print("Current Configuration:")
    config.print_config(file=None)
    
    # Example of saving configuration
    if args.config:
        config.save_config(args.config)
    else:
        config.save_config("config/generated_config.yaml")
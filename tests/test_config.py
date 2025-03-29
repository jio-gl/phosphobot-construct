"""
Unit tests for the phosphobot_construct.config module.
"""

import unittest
import os
import tempfile
import json
import yaml
from unittest.mock import patch

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phosphobot_construct.config import ConfigManager, get_config, create_parser


class TestConfigManager(unittest.TestCase):
    """Tests for the ConfigManager class."""
    
    def setUp(self):
        """Setup for tests, create sample config."""
        # Sample configuration
        self.sample_config = {
            "simulation": {
                "gui": True,
                "timestep": 0.001
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 32
            }
        }
    
    def test_init_default(self):
        """Test initialization with default config."""
        # Create config manager
        config = ConfigManager()
        
        # Check that default config was loaded
        self.assertIn("simulation", config.config)
        self.assertIn("training", config.config)
        self.assertIn("perception", config.config)
        self.assertIn("logging", config.config)
    
    def test_load_json_config(self):
        """Test loading config from JSON file."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(self.sample_config, f)
            temp_file = f.name
        
        try:
            # Create config manager with custom config
            config = ConfigManager(config_path=temp_file)
            
            # Check that config was loaded
            self.assertTrue(config.config["simulation"]["gui"])
            self.assertEqual(config.config["simulation"]["timestep"], 0.001)
            self.assertEqual(config.config["training"]["learning_rate"], 0.001)
            self.assertEqual(config.config["training"]["batch_size"], 32)
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_load_yaml_config(self):
        """Test loading config from YAML file."""
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(self.sample_config, f)
            temp_file = f.name
        
        try:
            # Create config manager with custom config
            config = ConfigManager(config_path=temp_file)
            
            # Check that config was loaded
            self.assertTrue(config.config["simulation"]["gui"])
            self.assertEqual(config.config["simulation"]["timestep"], 0.001)
            self.assertEqual(config.config["training"]["learning_rate"], 0.001)
            self.assertEqual(config.config["training"]["batch_size"], 32)
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_get_config_value(self):
        """Test getting config values."""
        # Create config manager
        config = ConfigManager()
        
        # Set some values
        config.config["test"] = {
            "nested": {
                "value": 42
            }
        }
        
        # Test getting values
        self.assertEqual(config.get("test.nested.value"), 42)
        self.assertEqual(config.get("test.nested.missing", "default"), "default")
        self.assertEqual(config.get("missing.path", 100), 100)
    
    def test_set_config_value(self):
        """Test setting config values."""
        # Create config manager
        config = ConfigManager()
        
        # Set some values
        config.set("test.nested.value", 42)
        config.set("simple", "value")
        
        # Check values
        self.assertEqual(config.get("test.nested.value"), 42)
        self.assertEqual(config.get("simple"), "value")
        
        # Update value
        config.set("test.nested.value", 43)
        self.assertEqual(config.get("test.nested.value"), 43)
    
    def test_save_config(self):
        """Test saving config to file."""
        # Create config manager
        config = ConfigManager()
        
        # Set some values
        config.set("test.value", 42)
        
        # Save to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_file = f.name
        
        # Save to YAML
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            yaml_file = f.name
        
        try:
            # Save configs
            config.save_config(json_file)
            config.save_config(yaml_file)
            
            # Check that files exist
            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(yaml_file))
            
            # Load configs
            json_config = ConfigManager(config_path=json_file)
            yaml_config = ConfigManager(config_path=yaml_file)
            
            # Check values
            self.assertEqual(json_config.get("test.value"), 42)
            self.assertEqual(yaml_config.get("test.value"), 42)
            
        finally:
            # Clean up
            os.unlink(json_file)
            os.unlink(yaml_file)
    
    @patch.dict('os.environ', {
        'PHOSPHOBOT_SIMULATION__GUI': 'false',
        'PHOSPHOBOT_TRAINING__LEARNING_RATE': '0.0005',
        'PHOSPHOBOT_PATHS__DATA_DIR': 'custom/data'
    })
    def test_update_from_env(self):
        """Test updating config from environment variables."""
        # Create config manager
        config = ConfigManager()
        
        # Initial values
        self.assertFalse(config.get("simulation.gui"))
        self.assertEqual(config.get("training.learning_rate"), 3e-4)
        
        # Update from environment
        config.update_from_env()
        
        # Check updated values
        self.assertFalse(config.get("simulation.gui"))
        self.assertEqual(config.get("training.learning_rate"), 0.0005)
        self.assertEqual(config.get("paths.data_dir"), "custom/data")
    
    def test_update_from_args(self):
        """Test updating config from command-line arguments."""
        # Create config manager
        config = ConfigManager()
        
        # Create parser
        parser = create_parser()
        args = parser.parse_args([
            "--simulation_gui",
            "--training_learning_rate", "0.001",
            "--paths_data_dir", "custom/data"
        ])
        
        # Update from args
        config.update_from_args(args)
        
        # Check updated values
        self.assertTrue(config.get("simulation.gui"))
        self.assertEqual(config.get("training.learning_rate"), 0.001)
        self.assertEqual(config.get("paths.data_dir"), "custom/data")
    
    def test_get_config_singleton(self):
        """Test the get_config singleton function."""
        # Get initial config
        config1 = get_config()
        
        # Set a value
        config1.set("test.value", 42)
        
        # Get config again
        config2 = get_config()
        
        # Check that they are the same instance
        self.assertIs(config1, config2)
        self.assertEqual(config2.get("test.value"), 42)
        
        # Force new instance with path
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(self.sample_config, f)
            temp_file = f.name
        
        try:
            # Reset singleton
            import phosphobot_construct.config
            phosphobot_construct.config._config_instance = None
            
            # Get new config
            config3 = get_config(temp_file)
            
            # Check that it loaded the file
            self.assertTrue(config3.get("simulation.gui"))
            self.assertEqual(config3.get("simulation.timestep"), 0.001)
            
        finally:
            # Clean up
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
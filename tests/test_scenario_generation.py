"""
Unit tests for the phosphobot_construct.scenario_generation module.
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phosphobot_construct.scenario_generation import ScenarioGenerator, generate_tabletop_scenarios


class TestScenarioGenerator(unittest.TestCase):
    """Tests for the ScenarioGenerator class."""
    
    def setUp(self):
        """Setup for tests, create sample data."""
        # Sample scenario for testing
        self.sample_scenario = {
            "id": "scenario_0001",
            "description": "A tabletop scene with three distinct objects for robot manipulation.",
            "workspace": "A flat wooden table surface approximately 1m x 1m in size.",
            "objects": [
                {
                    "name": "red_cube",
                    "shape": "cube",
                    "color": "red",
                    "material": "plastic",
                    "size": "5cm",
                    "mass": "50g",
                    "position": [0.2, 0.3, 0.0]
                },
                {
                    "name": "blue_cylinder",
                    "shape": "cylinder",
                    "color": "blue",
                    "material": "wood",
                    "size": "height 10cm, diameter 3cm",
                    "mass": "75g",
                    "position": [-0.1, 0.1, 0.0]
                },
                {
                    "name": "green_sphere",
                    "shape": "sphere",
                    "color": "green",
                    "material": "rubber",
                    "size": "4cm diameter",
                    "mass": "30g",
                    "position": [0.0, -0.2, 0.0]
                }
            ],
            "spatial_relations": "The red cube is positioned at the back right of the table. The blue cylinder is on the left side, and the green sphere is toward the front center.",
            "possible_tasks": [
                "Stack the objects with the cube on the bottom, cylinder in the middle, and sphere on top.",
                "Arrange the objects in a triangular formation.",
                "Sort objects by color from left to right.",
                "Group objects by material type."
            ]
        }
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('src.phosphobot_construct.scenario_generation.OpenAI')
    def test_init(self, mock_openai_class):
        """Test initialization of ScenarioGenerator."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Create generator
        generator = ScenarioGenerator(api_key="test_key")
        
        # Check OpenAI client initialization
        mock_openai_class.assert_called_once_with(api_key="test_key")
        
        # Check system prompt is set
        self.assertTrue(hasattr(generator, 'system_prompt'))
        self.assertIsInstance(generator.system_prompt, str)
        self.assertIn("robotics training data generator", generator.system_prompt.lower())
    
    @patch('src.phosphobot_construct.scenario_generation.OpenAI')
    def test_generate_scenarios_success(self, mock_openai_class):
        """Test successful scenario generation."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Create mock completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(self.sample_scenario)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create generator
        generator = ScenarioGenerator(api_key="test_key")
        
        # Generate scenarios
        scenarios = generator.generate_scenarios(num_scenarios=1, task_domain="tabletop")
        
        # Check OpenAI API was called
        mock_client.chat.completions.create.assert_called_once()
        
        # Check call arguments
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(len(kwargs["messages"]), 2)
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["response_format"]["type"], "json_object")
        
        # Check that task domain was included in the prompt
        self.assertIn("tabletop", kwargs["messages"][1]["content"].lower())
        
        # Check result
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["id"], "scenario_0001")
        self.assertEqual(scenarios[0]["description"], self.sample_scenario["description"])
    
    @patch('src.phosphobot_construct.scenario_generation.OpenAI')
    def test_generate_scenarios_batch(self, mock_openai_class):
        """Test generating multiple scenarios."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Create different scenarios for each call
        scenario1 = dict(self.sample_scenario)
        scenario1["id"] = "scenario_0001"
        
        scenario2 = dict(self.sample_scenario)
        scenario2["id"] = "scenario_0002"
        scenario2["description"] = "A kitchen countertop with cooking utensils."
        
        # Configure mock to return different responses for each call
        mock_responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(scenario1)))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(scenario2)))])
        ]
        mock_client.chat.completions.create.side_effect = mock_responses
        
        # Create generator
        generator = ScenarioGenerator(api_key="test_key")
        
        # Generate multiple scenarios
        scenarios = generator.generate_scenarios(num_scenarios=2, task_domain="mixed")
        
        # Check that API was called twice
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        
        # Check results
        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0]["id"], "scenario_0001")
        self.assertEqual(scenarios[1]["id"], "scenario_0002")
        self.assertNotEqual(scenarios[0]["description"], scenarios[1]["description"])
    
    @patch('src.phosphobot_construct.scenario_generation.OpenAI')
    def test_generate_scenarios_api_error(self, mock_openai_class):
        """Test scenario generation when API raises an error."""
        # Setup mock to raise an exception
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        # Create generator
        generator = ScenarioGenerator(api_key="test_key")
        
        # Generate scenarios (should continue and log error)
        scenarios = generator.generate_scenarios(num_scenarios=2, task_domain="tabletop")
        
        # Check that we got an empty list for the failed call
        self.assertEqual(len(scenarios), 0)
    
    # Mock the ScenarioGenerator.__init__ method to avoid API key requirement
    @patch('src.phosphobot_construct.scenario_generation.ScenarioGenerator.__init__', return_value=None)
    def test_save_scenarios(self, mock_init):
        """Test saving scenarios to files."""
        # Create generator with mocked __init__
        generator = ScenarioGenerator()
        
        # Set required attributes that would normally be set in __init__
        generator.client = MagicMock()
        
        # Create scenarios output directory
        scenarios_dir = os.path.join(self.test_dir, "scenarios")
        
        # Save scenarios
        generator.save_scenarios([self.sample_scenario], scenarios_dir)
        
        # Check that file was created
        expected_file = os.path.join(scenarios_dir, f"{self.sample_scenario['id']}.json")
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contents
        with open(expected_file, 'r') as f:
            saved_scenario = json.load(f)
        
        self.assertEqual(saved_scenario["id"], self.sample_scenario["id"])
        self.assertEqual(saved_scenario["description"], self.sample_scenario["description"])
    
    # Mock the ScenarioGenerator.__init__ method to avoid API key requirement
    @patch('src.phosphobot_construct.scenario_generation.ScenarioGenerator.__init__', return_value=None)
    def test_load_scenarios(self, mock_init):
        """Test loading scenarios from files."""
        # Create generator with mocked __init__
        generator = ScenarioGenerator()
        
        # Set required attributes that would normally be set in __init__
        generator.client = MagicMock()
        
        # Create scenarios directory
        scenarios_dir = os.path.join(self.test_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)
        
        # Save a scenario file
        scenario_file = os.path.join(scenarios_dir, f"{self.sample_scenario['id']}.json")
        with open(scenario_file, 'w') as f:
            json.dump(self.sample_scenario, f)
        
        # Load scenarios
        scenarios = generator.load_scenarios(scenarios_dir)
        
        # Check loaded scenarios
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["id"], self.sample_scenario["id"])
        self.assertEqual(scenarios[0]["description"], self.sample_scenario["description"])
    
    # Mock the ScenarioGenerator.__init__ method to avoid API key requirement
    @patch('src.phosphobot_construct.scenario_generation.ScenarioGenerator.__init__', return_value=None)
    def test_load_scenarios_multiple(self, mock_init):
        """Test loading multiple scenario files."""
        # Create generator with mocked __init__
        generator = ScenarioGenerator()
        
        # Set required attributes that would normally be set in __init__
        generator.client = MagicMock()
        
        # Create scenarios directory
        scenarios_dir = os.path.join(self.test_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)
        
        # Create two scenario variants
        scenario1 = dict(self.sample_scenario)
        scenario1["id"] = "scenario_0001"
        
        scenario2 = dict(self.sample_scenario)
        scenario2["id"] = "scenario_0002"
        scenario2["description"] = "A different scene"
        
        # Save scenario files
        with open(os.path.join(scenarios_dir, "scenario_0001.json"), 'w') as f:
            json.dump(scenario1, f)
        
        with open(os.path.join(scenarios_dir, "scenario_0002.json"), 'w') as f:
            json.dump(scenario2, f)
        
        # Load scenarios
        scenarios = generator.load_scenarios(scenarios_dir)
        
        # Check loaded scenarios
        self.assertEqual(len(scenarios), 2)
        
        # Scenarios might be loaded in any order, so we need to find them by ID
        scenario_dict = {s["id"]: s for s in scenarios}
        self.assertIn("scenario_0001", scenario_dict)
        self.assertIn("scenario_0002", scenario_dict)
        self.assertEqual(scenario_dict["scenario_0001"]["description"], scenario1["description"])
        self.assertEqual(scenario_dict["scenario_0002"]["description"], scenario2["description"])
    
    # Mock the ScenarioGenerator.__init__ method to avoid API key requirement
    @patch('src.phosphobot_construct.scenario_generation.ScenarioGenerator.__init__', return_value=None)
    def test_load_scenarios_invalid_directory(self, mock_init):
        """Test loading scenarios from non-existent directory."""
        # Create generator with mocked __init__
        generator = ScenarioGenerator()
        
        # Set required attributes that would normally be set in __init__
        generator.client = MagicMock()
        
        # Load scenarios from non-existent directory
        scenarios = generator.load_scenarios(os.path.join(self.test_dir, "nonexistent"))
        
        # Check that an empty list was returned
        self.assertEqual(scenarios, [])
    
    # Mock the ScenarioGenerator.__init__ method to avoid API key requirement
    @patch('src.phosphobot_construct.scenario_generation.ScenarioGenerator.__init__', return_value=None)
    def test_load_scenarios_invalid_file(self, mock_init):
        """Test loading scenarios with an invalid JSON file."""
        # Create generator with mocked __init__
        generator = ScenarioGenerator()
        
        # Set required attributes that would normally be set in __init__
        generator.client = MagicMock()
        
        # Create scenarios directory
        scenarios_dir = os.path.join(self.test_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)
        
        # Create an invalid JSON file
        invalid_file = os.path.join(scenarios_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("This is not valid JSON")
        
        # Create a valid JSON file
        valid_file = os.path.join(scenarios_dir, f"{self.sample_scenario['id']}.json")
        with open(valid_file, 'w') as f:
            json.dump(self.sample_scenario, f)
        
        # Load scenarios
        scenarios = generator.load_scenarios(scenarios_dir)
        
        # Check that only the valid file was loaded
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["id"], self.sample_scenario["id"])


# Standalone test function (use assert instead of self.assertEqual)
@patch('src.phosphobot_construct.scenario_generation.ScenarioGenerator')
def test_generate_tabletop_scenarios(mock_generator_class):
    """Test the generate_tabletop_scenarios function."""
    # Setup mocks
    mock_generator = MagicMock()
    mock_generator_class.return_value = mock_generator
    
    # Mock generate_scenarios and save_scenarios methods
    mock_generator.generate_scenarios.return_value = [{"id": "scenario_0001"}, {"id": "scenario_0002"}]
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as output_dir:
        # Call function
        generate_tabletop_scenarios(num_scenarios=2, output_dir=output_dir)
        
        # Check generator initialization
        mock_generator_class.assert_called_once()
        
        # Check generate_scenarios call
        mock_generator.generate_scenarios.assert_called_once_with(2, task_domain="tabletop")
        
        # Check save_scenarios call
        mock_generator.save_scenarios.assert_called_once()
        args, kwargs = mock_generator.save_scenarios.call_args
        assert args[0] == [{"id": "scenario_0001"}, {"id": "scenario_0002"}]
        assert args[1] == output_dir


if __name__ == "__main__":
    unittest.main()
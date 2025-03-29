"""
Unit tests for the phosphobot_construct.goal_generator module.
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

from src.phosphobot_construct.goal_generator import GoalGenerator, generate_goals_for_scenarios


class TestGoalGenerator(unittest.TestCase):
    """Tests for the GoalGenerator class."""
    
    def setUp(self):
        """Setup for tests, create sample data."""
        # Sample scenario for testing
        self.sample_scenario = {
            "id": "scenario_0001",
            "description": "A tabletop scene with various objects for manipulation.",
            "workspace": "A flat wooden table with a 1m x 1m surface.",
            "objects": [
                {
                    "name": "red_cube",
                    "shape": "cube",
                    "color": "red",
                    "material": "plastic",
                    "size": "small",
                    "position": [0.1, 0.2, 0.0]
                },
                {
                    "name": "blue_sphere",
                    "shape": "sphere",
                    "color": "blue",
                    "material": "foam",
                    "size": "medium",
                    "position": [-0.2, 0.1, 0.0]
                },
                {
                    "name": "green_cylinder",
                    "shape": "cylinder",
                    "color": "green",
                    "material": "wood",
                    "size": "large",
                    "position": [0.3, -0.1, 0.0]
                }
            ],
            "spatial_relations": "The red cube is to the right of the blue sphere. The green cylinder is at the far end of the table."
        }
        
        # Sample goal for testing
        self.sample_goal = {
            "id": "goal_scenario_0001_01",
            "scenario_id": "scenario_0001",
            "instruction": "Stack the red cube on top of the blue sphere.",
            "goal_state": {
                "red_cube": {
                    "position": [-0.2, 0.1, 0.05]
                },
                "blue_sphere": {
                    "position": [-0.2, 0.1, 0.0]
                },
                "green_cylinder": {
                    "position": [0.3, -0.1, 0.0]
                }
            },
            "success_criteria": "The red cube is stably positioned on top of the blue sphere.",
            "required_skills": ["grasping", "precise placement", "spatial reasoning"],
            "difficulty": "medium"
        }
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('src.phosphobot_construct.goal_generator.OpenAI')
    def test_init(self, mock_openai_class):
        """Test initialization of GoalGenerator."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Create generator
        generator = GoalGenerator(api_key="test_key")
        
        # Check OpenAI client initialization
        mock_openai_class.assert_called_once_with(api_key="test_key")
        
        # Check system prompt is set
        self.assertTrue(hasattr(generator, 'system_prompt'))
        self.assertIsInstance(generator.system_prompt, str)
        self.assertIn("robotics task planner", generator.system_prompt.lower())
    
    def test_format_scenario_description(self):
        """Test formatting scenario into a structured description."""
        # Create generator
        generator = GoalGenerator()
        
        # Format scenario
        description = generator._format_scenario_description(self.sample_scenario)
        
        # Check description contains important elements
        self.assertIn("SCENARIO DESCRIPTION", description)
        self.assertIn("WORKSPACE", description)
        self.assertIn("OBJECTS", description)
        self.assertIn("SPATIAL RELATIONS", description)
        
        # Check that object details are included
        for obj in self.sample_scenario["objects"]:
            obj_name = obj.get("name", "")
            self.assertIn(obj_name, description)
    
    @patch('src.phosphobot_construct.goal_generator.OpenAI')
    def test_generate_goals_success(self, mock_openai_class):
        """Test successful goal generation."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Create mock completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "goals": [self.sample_goal]
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create generator
        generator = GoalGenerator()
        
        # Generate goals
        goals = generator.generate_goals(self.sample_scenario, num_goals=1)
        
        # Check OpenAI API was called
        mock_client.chat.completions.create.assert_called_once()
        
        # Check call arguments
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(len(kwargs["messages"]), 2)
        self.assertEqual(kwargs["temperature"], 0.7)
        
        # Check result
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["id"], "goal_scenario_0001_01")
        self.assertEqual(goals[0]["scenario_id"], "scenario_0001")
        self.assertEqual(goals[0]["instruction"], "Stack the red cube on top of the blue sphere.")
    
    @patch('src.phosphobot_construct.goal_generator.OpenAI')
    def test_generate_goals_api_error(self, mock_openai_class):
        """Test goal generation when API raises an error."""
        # Setup mock to raise an exception
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        # Create generator
        generator = GoalGenerator()
        
        # Generate goals (should return empty list on error)
        goals = generator.generate_goals(self.sample_scenario, num_goals=1)
        
        # Check that an empty list was returned
        self.assertEqual(goals, [])
    
    def test_save_goals(self):
        """Test saving goals to files."""
        # Create generator
        generator = GoalGenerator()
        
        # Create goals output directory
        goals_dir = os.path.join(self.test_dir, "goals")
        
        # Save goals
        generator.save_goals([self.sample_goal], goals_dir)
        
        # Check that file was created
        expected_file = os.path.join(goals_dir, f"{self.sample_goal['id']}.json")
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file contents
        with open(expected_file, 'r') as f:
            saved_goal = json.load(f)
        
        self.assertEqual(saved_goal["id"], self.sample_goal["id"])
        self.assertEqual(saved_goal["instruction"], self.sample_goal["instruction"])
    
    def test_load_goals(self):
        """Test loading goals from files."""
        # Create generator
        generator = GoalGenerator()
        
        # Create goals directory
        goals_dir = os.path.join(self.test_dir, "goals")
        os.makedirs(goals_dir, exist_ok=True)
        
        # Save a goal file
        goal_file = os.path.join(goals_dir, f"{self.sample_goal['id']}.json")
        with open(goal_file, 'w') as f:
            json.dump(self.sample_goal, f)
        
        # Load goals
        goals = generator.load_goals(goals_dir)
        
        # Check loaded goals
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["id"], self.sample_goal["id"])
        self.assertEqual(goals[0]["instruction"], self.sample_goal["instruction"])
    
    def test_load_goals_with_scenario_filter(self):
        """Test loading goals with scenario filter."""
        # Create generator
        generator = GoalGenerator()
        
        # Create goals directory
        goals_dir = os.path.join(self.test_dir, "goals")
        os.makedirs(goals_dir, exist_ok=True)
        
        # Create another goal with different scenario_id
        other_goal = self.sample_goal.copy()
        other_goal["id"] = "goal_scenario_0002_01"
        other_goal["scenario_id"] = "scenario_0002"
        
        # Save both goal files
        with open(os.path.join(goals_dir, f"{self.sample_goal['id']}.json"), 'w') as f:
            json.dump(self.sample_goal, f)
        
        with open(os.path.join(goals_dir, f"{other_goal['id']}.json"), 'w') as f:
            json.dump(other_goal, f)
        
        # Load goals for specific scenario
        goals = generator.load_goals(goals_dir, scenario_id="scenario_0001")
        
        # Check only goals for the specified scenario were loaded
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["id"], self.sample_goal["id"])
        self.assertEqual(goals[0]["scenario_id"], "scenario_0001")
    
    def test_load_goals_invalid_directory(self):
        """Test loading goals from non-existent directory."""
        # Create generator
        generator = GoalGenerator()
        
        # Load goals from non-existent directory
        goals = generator.load_goals(os.path.join(self.test_dir, "nonexistent"))
        
        # Check that an empty list was returned
        self.assertEqual(goals, [])
    
    def test_load_goals_invalid_file(self):
        """Test loading goals with an invalid JSON file."""
        # Create generator
        generator = GoalGenerator()
        
        # Create goals directory
        goals_dir = os.path.join(self.test_dir, "goals")
        os.makedirs(goals_dir, exist_ok=True)
        
        # Create an invalid JSON file
        invalid_file = os.path.join(goals_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("This is not valid JSON")
        
        # Create a valid JSON file
        valid_file = os.path.join(goals_dir, f"{self.sample_goal['id']}.json")
        with open(valid_file, 'w') as f:
            json.dump(self.sample_goal, f)
        
        # Load goals
        goals = generator.load_goals(goals_dir)
        
        # Check that only the valid file was loaded
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["id"], self.sample_goal["id"])


@patch('src.phosphobot_construct.goal_generator.GoalGenerator')
def test_generate_goals_for_scenarios(mock_generator_class):
    """Test the generate_goals_for_scenarios function."""
    # Setup mocks
    mock_generator = MagicMock()
    mock_generator_class.return_value = mock_generator
    
    # Mock scenario loading and goal generation
    mock_generator.generate_goals.side_effect = [
        [{"id": "goal_1"}, {"id": "goal_2"}],
        [{"id": "goal_3"}]
    ]
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as scenarios_dir:
        with tempfile.TemporaryDirectory() as goals_dir:
            # Create mock scenario files
            with open(os.path.join(scenarios_dir, "scenario_1.json"), 'w') as f:
                json.dump({"id": "scenario_1"}, f)
            
            with open(os.path.join(scenarios_dir, "scenario_2.json"), 'w') as f:
                json.dump({"id": "scenario_2"}, f)
            
            # Call function
            generate_goals_for_scenarios(
                scenarios_dir=scenarios_dir,
                goals_dir=goals_dir,
                goals_per_scenario=2
            )
            
            # Check generator initialization
            mock_generator_class.assert_called_once()
            
            # Check generate_goals calls
            self.assertEqual(mock_generator.generate_goals.call_count, 2)
            
            # Check save_goals calls
            self.assertEqual(mock_generator.save_goals.call_count, 2)
            first_save_call = mock_generator.save_goals.call_args_list[0]
            self.assertEqual(first_save_call[0][0], [{"id": "goal_1"}, {"id": "goal_2"}])
            self.assertEqual(first_save_call[0][1], goals_dir)


if __name__ == "__main__":
    unittest.main()
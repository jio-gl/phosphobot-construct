"""
Unit tests for the phosphobot_construct.language_understanding module.
"""

import unittest
import json
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path to make imports work in testing
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phosphobot_construct.language_understanding import LanguageUnderstanding, language_to_goal


class TestLanguageUnderstanding(unittest.TestCase):
    """Tests for the LanguageUnderstanding class."""
    
    def setUp(self):
        """Setup for tests, create sample data."""
        # Sample 3D scene
        self.sample_scene = {
            "objects": [
                {
                    "id": 0,
                    "class": "a red cube",
                    "confidence": 0.92,
                    "position_3d": {"x": -0.3, "y": 0.2, "z": 0.1}
                },
                {
                    "id": 1,
                    "class": "a blue cube",
                    "confidence": 0.88,
                    "position_3d": {"x": 0.1, "y": -0.3, "z": 0.1}
                },
                {
                    "id": 2,
                    "class": "a green cube",
                    "confidence": 0.85,
                    "position_3d": {"x": 0.4, "y": 0.3, "z": 0.1}
                }
            ],
            "workspace": {
                "width": 640,
                "height": 480,
                "depth": 2.0
            }
        }
        
        # Sample instruction
        self.sample_instruction = "Stack the cubes with the red cube on the bottom, then the blue cube, and the green cube on top."
        
        # Sample goal state
        self.sample_goal = {
            "objects": [
                {
                    "id": 0,
                    "class": "a red cube",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.1}
                },
                {
                    "id": 1,
                    "class": "a blue cube",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.2}
                },
                {
                    "id": 2,
                    "class": "a green cube",
                    "position": {"x": 0.0, "y": 0.0, "z": 0.3}
                }
            ],
            "spatial_relations": "The cubes are stacked with red at the bottom, blue in the middle, and green on top.",
            "success_criteria": "All three cubes are stacked in the correct order.",
            "transformations": [
                {"object_id": 0, "action": "move", "target": {"x": 0.0, "y": 0.0, "z": 0.1}},
                {"object_id": 1, "action": "move", "target": {"x": 0.0, "y": 0.0, "z": 0.2}},
                {"object_id": 2, "action": "move", "target": {"x": 0.0, "y": 0.0, "z": 0.3}}
            ],
            "original_instruction": "Stack the cubes with the red cube on the bottom, then the blue cube, and the green cube on top."
        }
    
    @patch('src.phosphobot_construct.language_understanding.OpenAI')
    def test_init(self, mock_openai):
        """Test initialization."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create language understanding system
        language_system = LanguageUnderstanding(api_key="test_key")
        
        # Check that OpenAI was initialized with the correct API key
        mock_openai.assert_called_once_with(api_key="test_key")
    
    @patch('src.phosphobot_construct.language_understanding.OpenAI')
    def test_format_scene_description(self, mock_openai):
        """Test scene description formatting."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create language understanding system
        language_system = LanguageUnderstanding()
        
        # Format scene description
        scene_desc = language_system._format_scene_description(self.sample_scene)
        
        # Check that the description contains expected elements
        self.assertIn("OBJECTS", scene_desc)
        self.assertIn("WORKSPACE", scene_desc)
        self.assertIn("ROBOT", scene_desc)
        
        # Check that object descriptions are included
        for obj in self.sample_scene["objects"]:
            self.assertIn(obj["class"], scene_desc)
            self.assertIn(f"x={obj['position_3d']['x']:.2f}", scene_desc)
    
    @patch('src.phosphobot_construct.language_understanding.OpenAI')
    def test_instruction_to_goal_with_openai(self, mock_openai):
        """Test goal generation with OpenAI."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_choices = [MagicMock()]
        mock_choices[0].message.content = json.dumps(self.sample_goal)
        mock_response.choices = mock_choices
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create language understanding system
        language_system = LanguageUnderstanding()
        
        # Generate goal state
        goal_state = language_system.instruction_to_goal(
            self.sample_instruction,
            self.sample_scene
        )
        
        # Check that OpenAI API was called
        mock_client.chat.completions.create.assert_called_once()
        
        # Check the call arguments
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(len(kwargs["messages"]), 2)
        self.assertEqual(kwargs["temperature"], 0.2)
        
        # Check that the result is correct
        self.assertEqual(goal_state["original_instruction"], self.sample_instruction)
        self.assertEqual(len(goal_state["objects"]), 3)
        self.assertIn("spatial_relations", goal_state)
        self.assertIn("success_criteria", goal_state)
        self.assertIn("transformations", goal_state)
    
    @patch('src.phosphobot_construct.language_understanding.OpenAI')
    def test_fallback_goal_generation(self, mock_openai):
        """Test fallback goal generation."""
        # Setup mock to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        # Create language understanding system
        language_system = LanguageUnderstanding()
        
        # Generate goal state
        goal_state = language_system.instruction_to_goal(
            self.sample_instruction,
            self.sample_scene
        )
        
        # Check that the result is the fallback goal
        self.assertEqual(goal_state["original_instruction"], self.sample_instruction)
        self.assertEqual(len(goal_state["objects"]), 3)
        self.assertIn("spatial_relations", goal_state)
        self.assertIn("success_criteria", goal_state)
    
    @patch('src.phosphobot_construct.language_understanding.OpenAI')
    def test_language_to_goal_wrapper(self, mock_openai):
        """Test the language_to_goal wrapper function."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_choices = [MagicMock()]
        mock_choices[0].message.content = json.dumps(self.sample_goal)
        mock_response.choices = mock_choices
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call the wrapper function
        goal_3d = language_to_goal(
            self.sample_instruction,
            self.sample_scene,
            api_key="test_key"
        )
        
        # Check that OpenAI was initialized with the correct API key
        mock_openai.assert_called_once_with(api_key="test_key")
        
        # Check that the result is correct
        self.assertEqual(goal_3d["original_instruction"], self.sample_instruction)
        self.assertEqual(len(goal_3d["objects"]), 3)
        self.assertIn("spatial_relations", goal_3d)
        self.assertIn("success_criteria", goal_3d)


if __name__ == "__main__":
    unittest.main()
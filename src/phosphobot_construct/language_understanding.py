"""
Language Understanding and Goal Generation for the Phosphobot Construct.

This module translates natural language instructions into precise 3D goal states
for robot task execution.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI
import numpy as np

logger = logging.getLogger(__name__)


class LanguageUnderstanding:
    """
    Translates natural language instructions into 3D goal states.
    
    This class uses large language models (LLMs) to understand task instructions
    and generate structured goal descriptions.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the language understanding system.
        
        Args:
            api_key: OpenAI API key. If None, will use environment variable.
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # System prompt for consistency and quality control
        self.system_prompt = """
        You are a robot control system that converts natural language instructions into precise
        spatial goal states. Your task is to analyze the current scene and the instruction,
        then generate a structured representation of the desired goal state.
        
        Focus on:
        1. Spatial relationships between objects
        2. Required object transformations (translations, rotations)
        3. Precise 3D positions and orientations
        4. Physical feasibility of the goal state
        
        The robot is a dual-arm manipulator with parallel grippers on a fixed base.
        """
    
    def instruction_to_goal(
        self, 
        instruction: str,
        scene_3d: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert natural language instruction to 3D goal state.
        
        Args:
            instruction: Natural language instruction.
            scene_3d: Current 3D scene representation.
            
        Returns:
            3D goal state representation.
        """
        # Format the scene description
        scene_desc = self._format_scene_description(scene_3d)
        
        # Create user prompt
        user_prompt = f"""
        Scene: {scene_desc}

        Instruction: {instruction}

        Generate a precise 3D goal state description for the robot. The goal state should
        represent the scene after successfully executing the instruction.
        
        Return the goal state as a JSON object with these keys:
        - "objects": List of objects with their goal positions and orientations
        - "spatial_relations": Description of spatial relationships in the goal state
        - "success_criteria": Clear criteria for determining task completion
        - "transformations": List of required object movements to achieve the goal
        
        Ensure that all positions and orientations are physically feasible.
        """
        
        try:
            # Generate goal state using GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,  # Low temperature for deterministic results
                max_tokens=1500
            )
            
            # Parse the response
            goal_state = json.loads(response.choices[0].message.content)
            
            # Add the original instruction for reference
            goal_state["original_instruction"] = instruction
            
            logger.info(f"Generated goal state for instruction: {instruction}")
            return goal_state
            
        except Exception as e:
            logger.error(f"Error generating goal state: {str(e)}")
            return self._fallback_goal_generation(instruction, scene_3d)
    
    def _format_scene_description(self, scene_3d: Dict[str, Any]) -> str:
        """
        Format a 3D scene into a structured description for the LLM.
        
        Args:
            scene_3d: 3D scene representation.
            
        Returns:
            Formatted scene description.
        """
        # Extract objects
        objects = scene_3d.get("objects", [])
        
        # Format objects as string
        objects_str = ""
        for i, obj in enumerate(objects):
            obj_class = obj.get("class", f"Object {i+1}")
            obj_position = obj.get("position_3d", {})
            obj_desc = f"{obj_class} at position x={obj_position.get('x', 0):.2f}, y={obj_position.get('y', 0):.2f}, z={obj_position.get('z', 0):.2f}"
            objects_str += f"- {obj_desc}\n"
        
        # Get workspace dimensions
        workspace = scene_3d.get("workspace", {})
        workspace_str = f"Workspace dimensions: width={workspace.get('width', 0)}, height={workspace.get('height', 0)}, depth={workspace.get('depth', 0)}"
        
        # Get robot state if available
        robot_state = scene_3d.get("robot_state", [])
        robot_str = f"Robot state: {robot_state}" if robot_state else "Robot state: unknown"
        
        # Combine into structured description
        scene_desc = f"""
        OBJECTS:
        {objects_str}
        
        WORKSPACE:
        {workspace_str}
        
        ROBOT:
        {robot_str}
        """
        
        return scene_desc
    
    def _fallback_goal_generation(
        self, 
        instruction: str, 
        scene_3d: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback method for goal generation when LLM is not available.
        
        Args:
            instruction: Natural language instruction.
            scene_3d: Current 3D scene representation.
            
        Returns:
            Simplified goal state.
        """
        # Extract objects
        objects = scene_3d.get("objects", [])
        
        # Create a simple goal state based on keywords in the instruction
        objects_goal = []
        
        for i, obj in enumerate(objects):
            # Copy object data
            obj_goal = obj.copy()
            
            # Default is to keep the object in the same position
            obj_goal["goal_position"] = obj.get("position_3d", {}).copy()
            
            # Check for keywords related to this object
            obj_class = obj.get("class", "").lower()
            
            # Look for object mentioned in instruction
            if obj_class in instruction.lower():
                # Apply transformations based on instruction keywords
                if "pick up" in instruction.lower() or "lift" in instruction.lower():
                    # Move object up
                    obj_goal["goal_position"]["z"] += 0.2
                    obj_goal["action"] = "pick up"
                    
                elif "stack" in instruction.lower():
                    # For stacking, we'll need to reorder objects
                    # This is a very simplified approach
                    obj_goal["goal_position"]["z"] += i * 0.1
                    obj_goal["action"] = "stack"
                    
                elif "move" in instruction.lower() or "place" in instruction.lower():
                    # Move object to a new position
                    obj_goal["goal_position"]["x"] += 0.3
                    obj_goal["goal_position"]["y"] -= 0.2
                    obj_goal["action"] = "move"
                    
                elif "rotate" in instruction.lower() or "turn" in instruction.lower():
                    # Add rotation information
                    obj_goal["goal_orientation"] = {
                        "roll": 0, 
                        "pitch": 0, 
                        "yaw": 90
                    }
                    obj_goal["action"] = "rotate"
            
            objects_goal.append(obj_goal)
        
        # Create the goal state
        goal_state = {
            "objects": objects_goal,
            "original_instruction": instruction,
            "success_criteria": "Objects are in their target positions",
            "spatial_relations": "Objects arranged according to instruction",
            "transformations": [obj.get("action", "none") for obj in objects_goal if "action" in obj]
        }
        
        logger.info(f"Created fallback goal state for instruction: {instruction}")
        return goal_state


def language_to_goal(
    instruction: str,
    scene_3d: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert natural language instruction to 3D goal state (the 'imagining' operation).
    
    Args:
        instruction: Natural language instruction.
        scene_3d: Current 3D scene representation.
        api_key: OpenAI API key. If None, will use environment variable.
        
    Returns:
        3D goal state representation.
    """
    logger.info(f"Processing instruction: {instruction}")
    
    # Initialize language understanding system
    language_system = LanguageUnderstanding(api_key=api_key)
    
    # Convert instruction to goal state
    goal_3d = language_system.instruction_to_goal(instruction, scene_3d)
    
    return goal_3d


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the language understanding with a sample scene and instruction
    sample_scene = {
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
    
    sample_instruction = "Stack the cubes with the red cube on the bottom, then the blue cube, and the green cube on top."
    
    # Process the instruction
    try:
        goal_state = language_to_goal(sample_instruction, sample_scene)
        print("Generated Goal State:")
        print(json.dumps(goal_state, indent=2))
    except Exception as e:
        logger.error(f"Error in sample processing: {str(e)}")
"""
Goal generation using LLMs for the Phosphobot Construct.

This module generates diverse goal conditions for robotics scenarios
using large language models (LLMs).
"""

import json
import os
import logging
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class GoalGenerator:
    """
    Generates diverse goal conditions for robotics scenarios using LLMs.
    
    This class uses GPT-4o to generate structured goal descriptions for
    robot manipulation tasks, ensuring they are physically plausible.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the goal generator.
        
        Args:
            api_key: OpenAI API key. If None, will use environment variable.
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # System prompt for consistency and quality control
        self.system_prompt = """
        You are a robotics task planner. Your task is to generate diverse, achievable goals
        for robot manipulation scenarios.
        
        Focus on:
        1. Physical plausibility
        2. Diversity of manipulation skills
        3. Clear success criteria
        4. Precise goal state descriptions
        5. Different levels of difficulty
        
        The robot is a dual-arm manipulator with parallel grippers on a fixed base.
        """
    
    def generate_goals(self, scenario: Dict[str, Any], num_goals: int = 5) -> List[Dict[str, Any]]:
        """
        Generate diverse goals for a given scenario.
        
        Args:
            scenario: Scenario dictionary.
            num_goals: Number of goals to generate.
            
        Returns:
            List of goal dictionaries.
        """
        # Format scenario into a structured description
        scenario_desc = self._format_scenario_description(scenario)
        
        # Create user prompt
        user_prompt = f"""
        For the following scenario:
        {scenario_desc}
        
        Generate {num_goals} distinct manipulation goals that are physically possible.
        
        Each goal should include:
        - A natural language instruction (as would be given to a human)
        - A structured description of the goal state
        - Success criteria for evaluating if the goal is achieved
        - Required skills to complete the task
        - Estimated difficulty (easy, medium, hard)
        
        Return the goals as a JSON array of objects with these keys:
        - "instruction": The natural language instruction
        - "goal_state": Structured description of the desired state
        - "success_criteria": Clear criteria for success
        - "required_skills": List of skills needed
        - "difficulty": Estimated difficulty
        
        Make each goal unique and exercise different manipulation capabilities.
        """
        
        try:
            # Generate goals using GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,  # Some randomness for diversity
                max_tokens=2000
            )
            
            # Parse the response
            goals_data = json.loads(response.choices[0].message.content)
            goals = goals_data.get("goals", [])
            
            # Add unique IDs and scenario reference to each goal
            for i, goal in enumerate(goals):
                goal["id"] = f"goal_{scenario['id']}_{i+1:02d}"
                goal["scenario_id"] = scenario["id"]
            
            return goals
            
        except Exception as e:
            logger.error(f"Error generating goals: {str(e)}")
            return []
    
    def _format_scenario_description(self, scenario: Dict[str, Any]) -> str:
        """
        Format a scenario into a structured description for the LLM.
        
        Args:
            scenario: Scenario dictionary.
            
        Returns:
            Formatted scenario description.
        """
        # Extract key components
        description = scenario.get("description", "")
        workspace = scenario.get("workspace", "")
        objects = scenario.get("objects", [])
        spatial_relations = scenario.get("spatial_relations", "")
        
        # Format objects as string
        objects_str = ""
        for i, obj in enumerate(objects):
            obj_name = obj.get("name", f"Object {i+1}")
            obj_desc = ", ".join([f"{k}: {v}" for k, v in obj.items() if k != "name"])
            objects_str += f"- {obj_name}: {obj_desc}\n"
        
        # Combine into structured description
        scenario_desc = f"""
        SCENARIO DESCRIPTION:
        {description}
        
        WORKSPACE:
        {workspace}
        
        OBJECTS:
        {objects_str}
        
        SPATIAL RELATIONS:
        {spatial_relations}
        """
        
        return scenario_desc
    
    def save_goals(self, goals: List[Dict[str, Any]], output_dir: str = "goals") -> None:
        """
        Save generated goals to JSON files.
        
        Args:
            goals: List of goal dictionaries.
            output_dir: Directory to save goals to.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each goal to a separate file
        for goal in goals:
            goal_id = goal.get("id", f"goal_{len(os.listdir(output_dir)):04d}")
            output_path = os.path.join(output_dir, f"{goal_id}.json")
            
            with open(output_path, 'w') as f:
                json.dump(goal, f, indent=2)
            
            logger.info(f"Saved goal to {output_path}")
    
    def load_goals(self, input_dir: str = "goals", scenario_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load goals from JSON files.
        
        Args:
            input_dir: Directory containing goal JSON files.
            scenario_id: If provided, only load goals for this scenario.
            
        Returns:
            List of goal dictionaries.
        """
        goals = []
        
        # Check if directory exists
        if not os.path.exists(input_dir):
            logger.warning(f"Goals directory not found: {input_dir}")
            return goals
        
        # Load each goal file
        for filename in os.listdir(input_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(input_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        goal = json.load(f)
                        
                        # If scenario_id is specified, only include matching goals
                        if scenario_id is None or goal.get("scenario_id") == scenario_id:
                            goals.append(goal)
                except Exception as e:
                    logger.error(f"Error loading goal from {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(goals)} goals from {input_dir}")
        return goals


def generate_goals_for_scenarios(
    scenarios_dir: str = "data/scenarios",
    goals_dir: str = "data/goals",
    goals_per_scenario: int = 5
) -> None:
    """
    Generate goals for all scenarios in a directory.
    
    Args:
        scenarios_dir: Directory containing scenario JSON files.
        goals_dir: Directory to save goals to.
        goals_per_scenario: Number of goals to generate per scenario.
    """
    # Create goal generator
    generator = GoalGenerator()
    
    # Load scenarios
    scenarios = []
    for filename in os.listdir(scenarios_dir):
        if filename.endswith('.json'):
            with open(os.path.join(scenarios_dir, filename), 'r') as f:
                scenarios.append(json.load(f))
    
    logger.info(f"Loaded {len(scenarios)} scenarios from {scenarios_dir}")
    
    # Generate goals for each scenario
    all_goals = []
    
    for i, scenario in enumerate(scenarios):
        logger.info(f"Generating goals for scenario {i+1}/{len(scenarios)}: {scenario.get('id', 'unknown')}")
        
        goals = generator.generate_goals(scenario, num_goals=goals_per_scenario)
        all_goals.extend(goals)
        
        # Save after each scenario for fault tolerance
        generator.save_goals(goals, goals_dir)
    
    logger.info(f"Generated and saved {len(all_goals)} goals to {goals_dir}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Generate goals for test scenarios
    generate_goals_for_scenarios(
        scenarios_dir="data/test_scenarios",
        goals_dir="data/test_goals",
        goals_per_scenario=3
    )
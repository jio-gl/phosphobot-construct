"""
Scenario generation using LLMs for the Phosphobot Construct.

This module generates diverse training scenarios for the robot using
large language models (LLMs).
"""

import json
import os
import logging
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """
    Generates diverse training scenarios using large language models.
    
    This class uses GPT-4o to generate structured descriptions of
    tabletop manipulation scenarios for training robotics models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the scenario generator.
        
        Args:
            api_key: OpenAI API key. If None, will use environment variable.
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # System prompt for consistency and quality control
        self.system_prompt = """
        You are a robotics training data generator. Your task is to create realistic robot 
        manipulation scenarios with objects of different sizes, shapes, colors, and arrangements.
        
        Focus on:
        1. Physical plausibility
        2. Diversity of objects and configurations
        3. Clear spatial relationships
        4. Precise object properties (dimensions, color, material, etc.)
        5. Scenarios that exercise a range of manipulation skills
        
        The robot is a dual-arm manipulator with parallel grippers on a fixed base.
        """
    
    def generate_scenarios(self, num_scenarios: int = 10, task_domain: str = "tabletop") -> List[Dict[str, Any]]:
        """
        Generate a batch of diverse robot manipulation scenarios.
        
        Args:
            num_scenarios: Number of scenarios to generate.
            task_domain: Domain of tasks (e.g., "tabletop", "kitchen", "assembly").
            
        Returns:
            List of scenario dictionaries.
        """
        scenarios = []
        
        for i in range(num_scenarios):
            logger.info(f"Generating scenario {i+1}/{num_scenarios}")
            
            # Create specific prompt based on task domain
            user_prompt = f"""
            Generate a unique {task_domain} manipulation scenario with 3-5 objects.
            
            Include:
            - Detailed description of the workspace
            - Object properties (shape, size, color, material, mass)
            - Initial spatial configuration
            - Possible manipulation tasks for this arrangement
            
            Return the scenario as a JSON object with these keys:
            - "description": General description of the scene
            - "workspace": Description of the workspace surface and environment
            - "objects": List of objects with properties
            - "spatial_relations": Description of how objects are positioned relative to each other
            - "possible_tasks": List of manipulation tasks that could be performed
            
            Make each scenario unique with different objects and arrangements.
            """
            
            try:
                # Generate scenario using GPT-4o
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,  # Some randomness for diversity
                    max_tokens=1000
                )
                
                # Parse the response
                scenario = json.loads(response.choices[0].message.content)
                
                # Add a unique ID to the scenario
                scenario["id"] = f"scenario_{i+1:04d}"
                scenarios.append(scenario)
                
            except Exception as e:
                logger.error(f"Error generating scenario {i+1}: {str(e)}")
                continue
        
        return scenarios
    
    def save_scenarios(self, scenarios: List[Dict[str, Any]], output_dir: str = "scenarios") -> None:
        """
        Save generated scenarios to JSON files.
        
        Args:
            scenarios: List of scenario dictionaries.
            output_dir: Directory to save scenarios to.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each scenario to a separate file
        for scenario in scenarios:
            scenario_id = scenario.get("id", f"scenario_{len(os.listdir(output_dir)):04d}")
            output_path = os.path.join(output_dir, f"{scenario_id}.json")
            
            with open(output_path, 'w') as f:
                json.dump(scenario, f, indent=2)
            
            logger.info(f"Saved scenario to {output_path}")
    
    def load_scenarios(self, input_dir: str = "scenarios") -> List[Dict[str, Any]]:
        """
        Load scenarios from JSON files.
        
        Args:
            input_dir: Directory containing scenario JSON files.
            
        Returns:
            List of scenario dictionaries.
        """
        scenarios = []
        
        # Check if directory exists
        if not os.path.exists(input_dir):
            logger.warning(f"Scenarios directory not found: {input_dir}")
            return scenarios
        
        # Load each scenario file
        for filename in os.listdir(input_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(input_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        scenario = json.load(f)
                        scenarios.append(scenario)
                except Exception as e:
                    logger.error(f"Error loading scenario from {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(scenarios)} scenarios from {input_dir}")
        return scenarios


def generate_tabletop_scenarios(num_scenarios: int = 100, output_dir: str = "data/scenarios") -> None:
    """
    Generate tabletop manipulation scenarios and save them.
    
    Args:
        num_scenarios: Number of scenarios to generate.
        output_dir: Directory to save scenarios to.
    """
    # Create scenario generator
    generator = ScenarioGenerator()
    
    # Generate scenarios in batches
    batch_size = 10
    all_scenarios = []
    
    for i in range(0, num_scenarios, batch_size):
        batch_count = min(batch_size, num_scenarios - i)
        logger.info(f"Generating batch of {batch_count} scenarios ({i+1}-{i+batch_count}/{num_scenarios})")
        
        scenarios = generator.generate_scenarios(batch_count, task_domain="tabletop")
        all_scenarios.extend(scenarios)
    
    # Save all scenarios
    generator.save_scenarios(all_scenarios, output_dir)
    
    logger.info(f"Generated and saved {len(all_scenarios)} scenarios to {output_dir}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Generate a small batch of scenarios for testing
    generate_tabletop_scenarios(num_scenarios=5, output_dir="data/test_scenarios")
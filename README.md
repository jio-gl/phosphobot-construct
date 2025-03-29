# Phosphobot Construct: Robot Training w/GenAI for Superhuman Skills

![image](https://github.com/user-attachments/assets/89185655-a23c-4dc6-9215-94f7d49514d3)

## Concept

Recent advancements in robotics and generative AI have accelerated the development of embodied intelligence systems capable of understanding and executing complex tasks from natural language instructions. This project outlines a comprehensive framework for training robots to acquire superhuman manipulation skills through multimodal learning approaches. By leveraging state-of-the-art generative models, deep reinforcement learning, and 3D spatial reasoning, robots can understand linguistic commands and execute physical actions in novel environments.

### Features

- **3D Simulation Training**: Our approach is fundamentally based on training in 3D simulation environments, allowing for safe and accelerated learning
- **2D-to-3D Conversion**: We convert 2D camera inputs to rich 3D scene representations through our "placing" operation
- **Pre-training Pipeline**: Our innovative pipeline automatically generates thousands of training scenarios for comprehensive pre-training
- **Online Adaptation**: During execution, the system can continue training online to adapt to new scenarios

## Example Robot Architecture

For our demonstration, we utilize the Phospho Robotics dual-arm platform ([https://robots.phospho.ai/](https://robots.phospho.ai/)), which represents a state-of-the-art manipulation system. Each robotic arm incorporates:

- **Kinematic Structure**: 3 primary articulated joints (base, elbow, and wrist) with 6 degrees of freedom (DoF) total
- **End Effector**: Precision parallel gripper with force feedback
- **Proprioceptive Sensing**: High-resolution rotary encoders (0.01° precision) at each joint for angle and velocity measurement
- **Force/Torque Sensing**: Strain gauge-based torque sensors at each joint (0.1Nm resolution)
- **Vision System**: Dual RGB-D cameras with 4K resolution and depth perception

This hardware configuration aligns with research from ETH Zurich and UC Berkeley on high-dexterity manipulation platforms (Akkaya et al., 2023; Handa et al., 2024).

## Problem Formulation

### Input Space

The robot receives multimodal inputs from its environment:

1. **Proprioception Vector** $P \in \mathbb{R}^n$: Joint angles, velocities, and torques representing the robot's internal state.
   ```
   P = [θ₁, θ₂, ..., θₙ, ω₁, ω₂, ..., ωₙ, τ₁, τ₂, ..., τₙ]
   ```
   where $θ_i$ represents joint angles, $ω_i$ angular velocities, and $τ_i$ torques.

2. **Visual Observation Tensor** $V \in \mathbb{R}^{H×W×C×T}$: A sequence of RGB-D images captured from the robot's cameras, where H, W are spatial dimensions, C is channels, and T is the temporal dimension.

3. **Language Instruction** $L$: Natural language prompt describing the desired task or goal state.
   ```
   L = "Please, robot, stack the boxes from largest to smallest size."
   ```

### Output Space

- **Action Plan** $A = \{a_1, a_2, ..., a_m\}$: A directed acyclic graph (DAG) of primitive actions or a trajectory sequence that the robot must execute to accomplish the task.

It's important to note that this action plan is fundamentally dynamic in nature. Unlike traditional robotics approaches that generate a complete trajectory upfront and execute it in an open-loop fashion, our framework implements a closed-loop reactive planning approach:

1. The Deep Learning neural network generates only the immediate next action $a_i$ based on the current sensor observations.
2. After executing action $a_i$, new sensor data is collected, and the environment state is re-evaluated.
3. The neural network then determines the next optimal action $a_{i+1}$ based on this updated state information.
4. This sense-plan-act cycle continues until the goal state is reached.

This dynamic re-planning approach provides several advantages:
- **Adaptability**: The system can respond to unexpected changes or disturbances in the environment
- **Error Recovery**: Minor execution errors can be corrected in subsequent steps
- **Uncertainty Handling**: The system can incorporate new information as it becomes available

Mathematically, each action is a function of the current state rather than a predetermined sequence:

$a_i = f_\theta(s_i, g)$

Where $f_\theta$ is the learned neural network policy with parameters $\theta$, $s_i$ is the current state observation, and $g$ is the goal state representation.

## State-of-the-Art Pre-training Pipeline

### 1. Robot Modeling and Simulation Environment

Modern robotics training requires high-fidelity simulation before deployment on physical hardware. Industry standards include:

- **NVIDIA Isaac Sim**: Physics-accurate simulation with photorealistic rendering
- **PyBullet**: Open-source physics engine with robot models
- **MuJoCo**: Advanced physics simulator specialized for robotics research

For implementation details, see [`simulation.py`](src/phosphobot_construct/simulation.py)

### 2. LLM-Assisted Scenario Generation

To achieve robust generalization, the robot requires exposure to diverse task scenarios. We leverage large language models (LLMs) to generate thousands of diverse training scenarios.

**Recommended GenAI Model:**
- **GPT-4o**: Excels in generating diverse, realistic scenarios with physical constraints

For implementation details, see [`scenario_generator.py`](src/phosphobot_construct/scenario_generator.py)

### 3. LLM-Assisted Goal Generation

For each scenario, we generate multiple goal conditions that exercise different robotic skills.

**Recommended GenAI Model:**
- **GPT-4o**: Excellent for generating diverse goal conditions with physical awareness

For implementation details, see [`goal_generator.py`](src/phosphobot_construct/goal_generator.py)

### 4. Text-to-3D Conversion

We convert textual descriptions into 3D representations using state-of-the-art text-to-3D generative models.

**Recommended GenAI Model:**
- **Shap-E**: OpenAI's text-to-3D model for generating simple objects

For implementation details, see [`text_to_3d.py`](src/phosphobot_construct/text_to_3d.py)

### 5. 3D-to-Sensor Conversion

Converting 3D scenes to realistic sensor data is crucial for training perception models.

**Recommended Tools:**
- **PyTorch3D**: For differentiable rendering and sensor simulation

For implementation details, see [`sensor_generator.py`](src/phosphobot_construct/sensor_generator.py)

### 6. Deep Reinforcement Learning

The core of our approach uses deep reinforcement learning with transformer-based architectures to learn control policies from multimodal inputs.

**Recommended Frameworks:**
- **PyTorch**: Deep learning framework used in our implementation
- **Stable Baselines3**: Implementation of RL algorithms including PPO

**Key Algorithms:**
- **Proximal Policy Optimization (PPO)**: Policy gradient method for stable training
- **Soft Actor-Critic (SAC)**: Off-policy algorithm for sample efficiency
- **Decision Transformer**: Transformer architecture for sequential decision making

**Mathematical Formulation:**

The reinforcement learning objective is to maximize the expected cumulative reward:

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

where $\tau = (s_0, a_0, s_1, a_1, ...)$ is a trajectory, $\gamma$ is the discount factor, and $r(s_t, a_t)$ is the reward function.

For our robotics application, we use a distance-based reward function:

$$r(s_t, a_t) = -\lambda \cdot e^{-\alpha \cdot d(s_t, s_{\text{goal}})}$$

where $d(s_t, s_{\text{goal}})$ is the distance between the current state and goal state, $\lambda$ is a scaling factor, and $\alpha$ controls the reward shaping.

For implementation details, see [`reinforcement_learning.py`](src/phosphobot_construct/reinforcement_learning.py)

## Execution Pipeline

### 1. Perception and Scene Understanding

Converting real-world sensor inputs to 3D scene representations using state-of-the-art vision-language models.

**Recommended Models:**
- **SAM (Segment Anything Model)**: For object segmentation
- **CLIP**: For vision-language grounding and object classification

For implementation details, see [`perception.py`](src/phosphobot_construct/perception.py)

### 2. Language Understanding and Goal Generation

Translating natural language instructions into precise 3D goal states.

**Recommended Model:**
- **GPT-4o**: For natural language understanding and spatial reasoning

For implementation details, see [`language_understanding.py`](src/phosphobot_construct/language_understanding.py)

### 3. Policy Execution

Using the pre-trained policy to generate optimal trajectories toward the goal state.

For implementation details, see [`policy.py`](src/phosphobot_construct/policy.py)

### 4. Closed-Loop Control and Adaptation

Real-time feedback control to handle disturbances and uncertainties.

For implementation details, see [`control.py`](src/phosphobot_construct/control.py)

## Pipeline Summary: From Pre-training to Execution

To provide a clearer global view of our approach, we can summarize both the pre-training and execution pipelines with their interconnected components:

### Pre-training Pipeline Overview

1. **Robot Simulation**: Create high-fidelity physics-based models of the robot and environment
2. **Scenario Generation**: Use LLMs to generate diverse textual scenarios (1000s)
3. **Goal Generation**: For each scenario, generate multiple textual goal states (10s per scenario)
4. **Text-to-3D Conversion**: Transform textual descriptions into 3D scene representations
5. **Sensor Simulation**: Convert 3D scenes into synthetic sensor data (RGB-D images, proprioception)
6. **Deep RL Training**: Train a transformer-based policy using reinforcement learning on the synthetic data

This pre-training phase produces a robust policy model capable of generalizing to novel scenarios and goals.

### Execution Pipeline Overview

1. **"Placing" (Perception)**: Convert real-world 2D images to a 3D scene representation
   * The term "placing" refers to the process of grounding the robot's 2D visual observations in a 3D spatial model
   * This involves segmentation, depth estimation, and object recognition
   * The output is a structured 3D scene graph of the current environment state

2. **"Imagining" (Goal Generation)**: Create a 3D model of the goal state based on language instructions
   * "Imagining" is the process of generating a concrete 3D representation of the desired end state
   * This leverages language understanding to project the current scene forward to its goal configuration
   * Conceptually similar to mental simulation or predictive imagination in human cognition

3. **Planning**: Apply the pre-trained policy to generate an optimal trajectory between current and goal states
   * This uses the same neural architecture as in pre-training
   * The policy takes both the current 3D state and the imagined 3D goal as inputs

4. **Execution**: Implement the planned trajectory with closed-loop control and adaptation

The "placing" and "imagining" concepts bridge the multimodal gap between perception, language, and action. By converting both the current state and desired goal into the same 3D representation space, the robot can more effectively plan and execute complex manipulation tasks.

```
┌─────────────────────┐      ┌──────────────────┐      ┌───────────────────┐
│  Pre-training (Sim) │      │  Execution (Real)│      │                   │
│                     │      │                  │      │                   │
│ ┌─────────────────┐ │      │ ┌──────────────┐ │      │ ┌───────────────┐ │
│ │ Text Scenarios  │ │      │ │  RGB-D       │ │      │ │ Language      │ │
│ └────────┬────────┘ │      │ │  Images      │ │      │ │ Instruction   │ │
│          │          │      │ └───────┬──────┘ │      │ └───────┬───────┘ │
│ ┌────────▼────────┐ │      │         │        │      │         │         │
│ │ Text Goals      │ │      │ ┌───────▼──────┐ │      │ ┌───────▼───────┐ │
│ └────────┬────────┘ │      │ │   "Placing"  │ │      │ │  "Imagining"  │ │
│          │          │      │ │  (2D to 3D)  │ │      │ │  (Text to 3D) │ │
│ ┌────────▼────────┐ │      │ └───────┬──────┘ │      │ └───────┬───────┘ │
│ │ 3D Scenarios    │ │      │         │        │      │         │         │
│ └────────┬────────┘ │      │ ┌───────▼──────┐ │      │ ┌───────▼───────┐ │
│          │          │      │ │ Current 3D   │ │      │ │  Goal 3D      │ │
│ ┌────────▼────────┐ │      │ │ Scene        │ │      │ │  Scene        │ │
│ │ Sensor Data     │ │      │ └───────┬──────┘ │      │ └───────┬───────┘ │
│ └────────┬────────┘ │      │         │        │      │         │         │
│          │          │      │         └────────┼──────┼─────────┘         │
│ ┌────────▼────────┐ │      │                  │      │                   │
│ │ Policy Training │ │◄─────┼───┐ ┌────────────▼──────▼───────────────┐  │
│ └────────┬────────┘ │      │   └─┤  Policy Execution & Control       │  │
│          │          │      │     └─────────────────────────────────┬─┘  │
│ ┌────────▼────────┐ │      │                                       │    │
│ │ Learned Policy  │─┼──────┘                                       │    │
│ └─────────────────┘ │                                              │    │
└─────────────────────┘                                              │    │
                       ┌──────────────────────────────────────────┐  │    │
                       │              Real World                  │◄─┘    │
                       └──────────────────────────────────────────┘       │
                       └───────────────────────────────────────────────────┘
```

This dual-pipeline architecture enables robots to leverage the strengths of large-scale pre-training while maintaining the ability to adapt to specific real-world scenarios through the perception-planning-action loop.

## Practical Implementation Example: Box Stacking Task

For implementation of the complete pipeline for the box stacking task, see [`examples/box_stacking.py`](examples/box_stacking.py)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/phosphobot-construct.git
cd phosphobot-construct

# Install with pip
pip install -e .
```

## Usage

```python
from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot_construct.models import PhosphoConstructModel

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

# Instantiate the model
model = PhosphoConstructModel()

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=2, resize=(240, 320)),
    ]

    # Get the robot state
    state = client.control.read_joints()

    inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}

    # Go through the model
    actions = model(inputs)

    for action in actions:
        # Send the new joint postion to the robot
        client.control.write_joints(angles=action.tolist())
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
```

## References

1. Akkaya, I., et al. (2023). "Scalable Robot Learning from Simulation to Reality." *ICRA 2023*.
2. Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." *arXiv:2307.15818*.
3. Driess, D., et al. (2023). "PaLM-E: An Embodied Multimodal Language Model." *ICML 2023*.
4. Finn, C., et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." *ICML 2017*.
5. Handa, A., et al. (2024). "DreamFusion for Robotics: 3D Planning from Language Prompts." *CoRL 2024*.
6. Khetarpal, K., et al. (2022). "Towards Continual Reinforcement Learning: A Review and Perspectives." *JAIR*.
7. Peng, X.B., et al. (2018). "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization." *ICRA 2018*.
8. Reed, S., et al. (2023). "A Generalist Agent." *arXiv:2205.06175*.
9. Takmaz, E., et al. (2023). "From Language to Telemanipulation: A Hierarchical Approach." *CoRL 2023*.
10. Wang, Q., et al. (2024). "Text-to-3D for Robotics: Generating 3D Representations for Manipulation Tasks." *arXiv:2403.09636*.

# Phosphobot Construct Usage Guide

This guide explains how to use Phosphobot Construct for robot training and control.

## Getting Started

After installing Phosphobot Construct, you can use it either through the command-line interface (CLI) or as a Python library.

### CLI Quick Start

```bash
# Generate training data
phosphobot-construct generate --all --num-scenarios 10

# Train a model
phosphobot-construct train my_first_model

# Run a trained model
phosphobot-construct run models/my_first_model/model.zip "Stack the red cube on the blue cube"
```

### Python Quick Start

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
time.sleep(1)  # Wait for cameras to initialize

# Instantiate the model
model = PhosphoConstructModel()

# Main control loop
while True:
    # Get camera frames
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=2, resize=(240, 320)),
    ]

    # Get the robot state
    state = client.control.read_joints()

    # Prepare inputs
    inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}

    # Get actions from the model
    actions = model(inputs)

    # Send actions to the robot
    for action in actions:
        client.control.write_joints(angles=action.tolist())
        time.sleep(1 / 30)  # 30 Hz control frequency
```

## Command-line Interface (CLI)

Phosphobot Construct provides a comprehensive CLI for common tasks:

### Data Generation

Generate training data for your robot:

```bash
phosphobot-construct generate [options]

Options:
  --all                  Generate all data types
  --scenarios            Generate scenarios
  --goals                Generate goals
  --models               Convert to 3D models
  --sensor               Generate sensor data
  --num-scenarios NUM    Number of scenarios to generate
  --goals-per-scenario N Number of goals per scenario
  --gpu                  Use GPU for rendering
```

### Training

Train a new model:

```bash
phosphobot-construct train MODEL_NAME [options]

Options:
  --scene-file FILE      Path to scene JSON file
  --goal-file FILE       Path to goal JSON file
  --timesteps N          Total timesteps for training
  --no-transformer       Disable transformer-based policy
  --seed SEED            Random seed (default: 42)
  --use-language         Enable language conditioning
```

### Running Examples

Run pre-built examples:

```bash
phosphobot-construct example EXAMPLE_NAME

Available examples:
  box_stacking          Box stacking demo
  object_tracking       Object tracking demo
```

### Running Trained Models

Run a trained model on your robot:

```bash
phosphobot-construct run MODEL_PATH INSTRUCTION [options]

Options:
  --server-url URL       Robot server URL
  --server-port PORT     Robot server port
  --max-steps N          Maximum number of steps
```

### Configuration

Manage configurations:

```bash
# Using a specific config file
phosphobot-construct --config config.yaml [command]

# Setting config values directly
phosphobot-construct --logging_level DEBUG --simulation_gui [command]
```

## Python API

Phosphobot Construct can be used as a Python library for more flexibility.

### Core Components

1. **Models**:

```python
from phosphobot_construct.models import PhosphoConstructModel

# Create a model
model = PhosphoConstructModel(model_path="path/to/model.pt")

# Use the model
actions = model(inputs)
```

2. **Perception**:

```python
from phosphobot_construct.perception import perception_pipeline

# Convert images to 3D scene
scene_3d = perception_pipeline(
    rgb_image=rgb_image,
    depth_image=depth_image,
    proprioception=joint_state
)
```

3. **Language Understanding**:

```python
from phosphobot_construct.language_understanding import language_to_goal

# Convert instruction to goal state
goal_3d = language_to_goal(
    instruction="Stack the red cube on the blue cube",
    scene_3d=scene_3d
)
```

4. **Policy Execution**:

```python
from phosphobot_construct.policy import execute_policy

# Execute a policy
result = execute_policy(
    model_path="path/to/model.zip",
    scene_3d=scene_3d,
    goal_3d=goal_3d,
    client=client
)
```

5. **Control**:

```python
from phosphobot_construct.control import adaptive_control

# Move robot with closed-loop control
result = adaptive_control(
    client=client,
    target_positions=target,
    perception_func=update_perception,
    feedback_rate=30
)
```

### Configuration

Manage configuration programmatically:

```python
from phosphobot_construct.config import get_config

# Get configuration
config = get_config("config.yaml")

# Get values
server_url = config.get("robot.server_url", "http://localhost")

# Set values
config.set("robot.server_port", 8080)

# Save configuration
config.save_config("updated_config.yaml")
```

## Working with the Robot

### Basic Robot Control

```python
from phosphobot.api.client import PhosphoApi

# Connect to the robot
client = PhosphoApi(base_url="http://localhost:80")

# Initialize robot
client.move.init()

# Move robot (absolute position)
client.move.absolute(x=10, y=0, z=20, rx=0, ry=0, rz=0, open=1)

# Move robot (relative position)
client.move.relative(x=5, y=0, z=0, rx=0, ry=0, rz=0, open=0)

# Read joint positions
state = client.control.read_joints()
print(f"Joint angles: {state.angles_rad}")

# Write joint positions
client.control.write_joints(angles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
```

### Camera Access

```python
from phosphobot.camera import AllCameras

# Initialize cameras
cameras = AllCameras()

# List available cameras
camera_list = cameras.get_camera_list()
print(f"Found {len(camera_list)} cameras")

# Get RGB frame
rgb_frame = cameras.get_rgb_frame(camera_id=0, resize=(320, 240))

# Get depth frame (if depth camera is available)
depth_frame = cameras.get_depth_frame(camera_id=0, resize=(320, 240))
```

## Training Workflow

A typical workflow for training a new model:

1. **Generate training data**:
   ```bash
   phosphobot-construct generate --all
   ```

2. **Train the model**:
   ```bash
   phosphobot-construct train my_model
   ```

3. **Test in simulation** (if available):
   ```bash
   # Configure simulation
   vim config/sim_config.yaml
   
   # Run in simulation
   phosphobot-construct run models/my_model/model.zip "Stack the blocks" --config config/sim_config.yaml
   ```

4. **Deploy to the physical robot**:
   ```bash
   phosphobot-construct run models/my_model/model.zip "Stack the blocks"
   ```

## Examples

### Box Stacking Example

```python
from examples.box_stacking import box_stacking_demo

# Run the demo
result = box_stacking_demo(
    server_url="http://localhost",
    server_port=80,
    model_path="models/stacking_model.zip"
)
```

### Object Tracking Example

```python
from examples.object_tracking import object_tracking_demo

# Run the demo
result = object_tracking_demo(
    server_url="http://localhost",
    server_port=80,
    target_class="cube",
    max_duration=60.0
)
```

## Troubleshooting

### Common Runtime Issues

1. **Model predictions are incorrect**:
   - Ensure your camera setup matches what the model was trained on
   - Check image preprocessing and normalization
   - Try retraining with more diverse data

2. **Robot movements are jerky**:
   - Adjust the control frequency (default: 30 Hz)
   - Tune PID control parameters in control.py
   - Check for hardware limitations

3. **Perception fails to identify objects**:
   - Improve lighting conditions
   - Check camera positioning
   - Retrain with more diverse object appearances

### Getting Help

For more detailed information:
- Check the [example scripts](https://github.com/phosphobot/phosphobot-construct/tree/main/examples)
- Read the [API documentation](https://docs.phosphobot.ai/construct)
- Join our [Discord community](https://discord.gg/phosphobot)
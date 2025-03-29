# Installation Guide for Phosphobot Construct (WIP)

![image](https://github.com/user-attachments/assets/6ff83c49-2134-4d99-ae1d-87b2673d2d71)


This document explains how to install Phosphobot Construct and its dependencies.

## System Requirements

- **Operating System**: Linux (recommended), macOS, or Windows with WSL
- **Python**: 3.8 or higher
- **Hardware**:
  - For training: NVIDIA GPU with 8+ GB VRAM
  - For inference: CPU or GPU
  - For robot control: Compatible robot hardware (Phospho SO-100 recommended)

## Installation Methods

### 1. Installation from source (NOT AVAILABLE YET)

For the latest features or to contribute to development:

```bash
# Clone the repository
git clone https://github.com/phosphobot/phosphobot-construct.git
cd phosphobot-construct

# Install in development mode
pip install -e .
```

### 2. Installation with pip (NOT AVAILABLE YET)

The simplest way to install Phosphobot Construct is using pip:

```bash
pip install phosphobot-construct
```

To install with all optional dependencies:

```bash
pip install phosphobot-construct[full]
```

For development installation:

```bash
pip install phosphobot-construct[dev]
```

## Dependencies

Phosphobot Construct depends on several libraries:

### Core Dependencies
- numpy
- torch
- opencv-python
- phosphobot
- openai
- transformers

### Optional Dependencies
- PyTorch3D (for 3D rendering)
- Stable-Baselines3 (for reinforcement learning)
- Gymnasium (for environments)
- segment-anything (for perception)
- clip (for vision-language models)
- diffusers (for text-to-3D conversion)

## Phosphobot Hardware Setup

If you're using a Phosphobot SO-100 robot arm, follow these steps:

1. **Install the Phosphobot server**:

   ```bash
   # On macOS
   brew tap phospho-app/phosphobot
   brew install phosphobot
   
   # On Linux
   curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | sudo bash
   ```

2. **Connect hardware**:
   - Attach the SO-100 arm to a table using the provided clamps
   - Connect the power supply to the arm
   - Connect the USB cable from the arm to your computer
   - Attach cameras and position them to view the workspace

3. **Start the server**:

   ```bash
   phosphobot run
   ```

4. **Calibrate the robot**:
   - Open the dashboard at http://localhost
   - Go to the Calibration page
   - Follow the on-screen instructions

## Verify Installation

To verify your installation is working correctly:

```bash
# Check installed version
phosphobot-construct --version

# Run a simple test
phosphobot-construct example box_stacking --config config/test_config.yaml
```

## Troubleshooting

### Common Issues

1. **GPU not detected**:
   - Ensure NVIDIA drivers are installed
   - Check that PyTorch is installed with CUDA support
   - Run `torch.cuda.is_available()` in Python to verify

2. **Import errors**:
   - Ensure all dependencies are installed
   - Check Python version compatibility

3. **Robot connection issues**:
   - Verify the Phosphobot server is running
   - Check USB connections
   - Ensure the correct port is specified in configuration

### Getting Help

If you encounter problems not covered here:

- Check the [GitHub issues](https://github.com/phosphobot/phosphobot-construct/issues)
- Visit the [Phosphobot documentation](https://docs.phosphobot.ai)
- Join our [Discord community](https://discord.gg/phosphobot)

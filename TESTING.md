# Testing Guide for Phosphobot Construct

This document provides guidance on testing the Phosphobot Construct project, including running unit tests, integration tests, and evaluating test coverage.

## Testing Framework

The Phosphobot Construct project uses the following testing tools:

- **unittest**: Python's built-in unit testing framework
- **pytest**: Test runner with enhanced features
- **mock**: Library for creating mock objects (part of Python's unittest.mock)
- **coverage**: Tool for measuring code coverage

## Setting Up the Testing Environment

Install the testing dependencies:

```bash
# Install testing dependencies
pip install pytest pytest-cov
```

Install development version (NOT AVAILABLE):

```bash
# Install development version of the package
pip install -e ".[dev]"
```

## Setting Up the Testing Environment

### Creating a Virtual Environment

It's recommended to use a virtual environment for testing to avoid conflicts with other packages:

```bash
# Using venv (Python 3.3+)
python -m venv phosphobot-env
source phosphobot-env/bin/activate  # On Linux/macOS
phosphobot-env\Scripts\activate     # On Windows

# Or using virtualenv
pip install virtualenv
virtualenv phosphobot-env
source phosphobot-env/bin/activate  # On Linux/macOS
phosphobot-env\Scripts\activate     # On Windows
```

### Installing Dependencies

#### Minimal Installation (Core Testing)

```bash
# Install the package with dev dependencies
pip install -e ".[dev]"

# Install testing tools
pip install pytest pytest-cov
```

#### Full Installation (All Features)

```bash
# Install all dependencies including optional ones
pip install -e ".[dev,full]"

# Install additional dependencies for hardware testing
pip install phosphobot
```

#### Installing Specific Dependencies

If you only want to test specific modules, you can install their dependencies:

```bash
# For perception testing
pip install opencv-python clip segment-anything torch

# For 3D conversion testing
pip install diffusers trimesh torch

# For reinforcement learning testing
pip install stable-baselines3 gymnasium torch

# For simulation testing
pip install pybullet numpy
```

### Verifying Installation

To verify your installation is ready for testing:

```bash
# Check installed packages
pip list | grep -E 'phosphobot|pytest|torch'

# Run a simple test to check imports
python -c "from phosphobot_construct.models import PhosphoConstructModel; print('Import successful!')"
```

## Running Tests

### Running All Tests

To run all tests at once:

```bash
# Using pytest
pytest

# Using unittest discover
python -m unittest discover

# With code coverage
pytest --cov=phosphobot_construct
```

### Running Tests for a Specific Module

To test a single module:

```bash
# Test the models module
pytest tests/test_models.py

# Test the perception module
pytest tests/test_perception.py

# Test the control module
pytest tests/test_control.py
```

### Running a Single Test

To run a single test case or test method:

```bash
# Run a specific test class
pytest tests/test_models.py::TestPhosphoConstructModel

# Run a specific test method
pytest tests/test_models.py::TestPhosphoConstructModel::test_init
```

## Test Modules

The project includes tests for the following modules:

| Module | Test File | Description |
|--------|-----------|-------------|
| Models | `test_models.py` | Tests for the PhosphoConstructModel class and other model components |
| Simulation | `test_simulation.py` | Tests for the physics simulation environment |
| Perception | `test_perception.py` | Tests for scene understanding and 2D-to-3D conversion |
| Control | `test_control.py` | Tests for closed-loop control and trajectory execution |
| Policy | `test_policy.py` | Tests for policy execution and model loading |
| Goal Generator | `test_goal_generator.py` | Tests for generating goal states from scenarios |
| Language Understanding | `test_language_understanding.py` | Tests for processing natural language instructions |
| Reinforcement Learning | `test_reinforcement_learning.py` | Tests for RL components and training |
| Scenario Generator | `test_scenario_generator.py` | Tests for generating training scenarios |
| Sensor Generator | `test_sensor_generator.py` | Tests for generating synthetic sensor data |
| Text-to-3D | `test_text_to_3d.py` | Tests for converting text descriptions to 3D models |
| Config | `test_config.py` | Tests for configuration management |

## Common Test Commands

```bash
# Run all tests
pytest

# Run tests with detailed output
pytest -v

# Run tests and stop at first failure
pytest -xvs

# Run tests and show local variables on failure
pytest --showlocals

# Run tests matching a pattern
pytest -k "Control"

# Generate HTML coverage report
pytest --cov=phosphobot_construct --cov-report=html

# List all available tests without running them
pytest --collect-only
```

## Handling Dependencies in Tests

Some modules have optional dependencies. The tests handle these cases by:

1. Mocking external dependencies if they're not installed
2. Skipping relevant tests when required dependencies are missing
3. Providing fallback functionality for core features

Example of mocking dependencies in tests:

```python
# Mock PyTorch and related imports
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()

# Conditionally run tests depending on dependencies
@unittest.skipIf(not HAS_RL_DEPS, "RL dependencies not installed")
def test_reinforcement_learning():
    # Test code here
    pass
```

## Testing with Hardware

For tests involving physical hardware (Phosphobot SO-100 arm):

```bash
# Set environment variable to use hardware
export PHOSPHOBOT_USE_HARDWARE=1

# Run hardware-related tests
pytest tests/test_hardware_integration.py
```

Note: Hardware tests are skipped by default unless the environment variable is set.

## Testing Strategies

The project uses several testing strategies:

1. **Unit Tests**: Testing individual components in isolation
2. **Mock Tests**: Using mock objects to simulate dependencies
3. **Parametrized Tests**: Testing functions with different inputs
4. **Integration Tests**: Testing multiple components together
5. **Hardware Tests**: Testing with actual robot hardware

## Adding New Tests

When adding new features, also add corresponding tests:

1. Create a new test file in the `tests/` directory if needed
2. Follow the naming convention: `test_<module_name>.py`
3. Create test classes that inherit from `unittest.TestCase`
4. Add test methods with names starting with `test_`
5. Add docstrings to explain what each test is checking

Example template for a new test:

```python
"""
Unit tests for the phosphobot_construct.<module> module.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.phosphobot_construct.<module> import <Class>

class Test<Class>(unittest.TestCase):
    """Tests for the <Class> class."""
    
    def setUp(self):
        """Setup for tests."""
        # Setup code here
        
    def tearDown(self):
        """Clean up after tests."""
        # Cleanup code here
    
    def test_init(self):
        """Test initialization of <Class>."""
        # Test code here
        
    def test_method(self):
        """Test <method> functionality."""
        # Test code here
```

## Continuous Integration

The CI pipeline automatically runs tests on each pull request:

1. Tests run against multiple Python versions (3.8, 3.9, 3.10, 3.11)
2. Tests run with minimal dependencies and with full dependencies
3. Code coverage reports are generated
4. Tests must pass before merging is allowed

## Troubleshooting Tests

Common issues and their solutions:

### Missing Dependencies

```
ImportError: No module named 'torch'
```

Solution: Install the missing dependency or use the full installation:
```bash
pip install "phosphobot-construct[full]"
```

### Hardware Not Available

```
RuntimeError: Hardware not detected
```

Solution: Use the mock implementation:
```bash
export PHOSPHOBOT_USE_MOCK=1
```

### Test Hanging

If a test seems to hang, it might be waiting for hardware or a network resource.

Solution: Add timeout to the test or use mock objects:
```python
@timeout_decorator.timeout(5)
def test_with_timeout(self):
    # Test code here
```

## Test Coverage

To measure code coverage and generate a report:

```bash
# Generate coverage report
pytest --cov=phosphobot_construct

# Generate HTML coverage report
pytest --cov=phosphobot_construct --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## Best Practices

1. **Isolate tests**: Each test should be independent of others
2. **Mock external dependencies**: Use mocks for APIs, hardware, etc.
3. **Test edge cases**: Include tests for error conditions and edge cases
4. **Keep tests fast**: Minimize dependencies on slow resources
5. **Make assertions specific**: Test exact expected outcomes
6. **Test public interfaces**: Focus on testing the public API
7. **Follow AAA pattern**: Arrange, Act, Assert in each test

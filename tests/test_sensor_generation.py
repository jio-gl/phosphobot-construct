import unittest
import numpy as np
import os
import tempfile
import shutil
import sys
import pytest
from unittest.mock import MagicMock

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(autouse=True)
def mock_deps(monkeypatch):
    """Mock torch and pytorch3d only for this test file, per test function."""
    mock_torch = MagicMock()
    mock_torch.device = MagicMock(return_value="mock_device")
    mock_torch.cuda.is_available = MagicMock(return_value=True)
    mock_torch.ones_like = MagicMock(side_effect=lambda x: x)
    mock_torch.cat = MagicMock(side_effect=lambda tensors, dim: tensors[0])
    mock_torch.ones = MagicMock()
    mock_torch.zeros = MagicMock()
    mock_torch.tensor = MagicMock()

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", MagicMock())
    monkeypatch.setitem(sys.modules, "pytorch3d", MagicMock())
    monkeypatch.setitem(sys.modules, "pytorch3d.io", MagicMock())
    monkeypatch.setitem(sys.modules, "pytorch3d.renderer", MagicMock())
    monkeypatch.setitem(sys.modules, "pytorch3d.structures", MagicMock())

    # Import the module after mocking
    global SensorGenerator
    from src.phosphobot_construct.sensor_generation import SensorGenerator


class TestSensorGeneratorBasic(unittest.TestCase):
    def test_init(self):
        gen = SensorGenerator()
        self.assertEqual(gen.device, "mock_device")
        self.assertEqual(gen.width, 320)
        self.assertEqual(gen.height, 240)
        self.assertEqual(len(gen.camera_positions), gen.num_cameras)

    def test_generate_placeholder_images(self):
        gen = SensorGenerator(image_size=(128, 128))
        rgb, depth = gen._generate_placeholder_images()
        self.assertEqual(rgb.shape, (128, 128, 3))
        self.assertEqual(depth.shape, (128, 128))

    def test_generate_proprioception_data(self):
        gen = SensorGenerator()
        props = gen._generate_proprioception_data([{}])
        self.assertEqual(props.shape, (18,))

    def test_save_sensor_data(self):
        gen = SensorGenerator()
        temp_dir = tempfile.mkdtemp()
        try:
            data = {
                "rgb_images": [np.zeros((240, 320, 3), dtype=np.uint8)],
                "depth_images": [np.zeros((240, 320), dtype=np.uint8)],
                "proprioception": np.zeros(18)
            }
            result = gen.save_sensor_data(data, temp_dir)
            self.assertTrue(os.path.exists(result["rgb_paths"][0]))
            self.assertTrue(os.path.exists(result["depth_paths"][0]))
            self.assertTrue(os.path.exists(result["proprioception_path"]))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()

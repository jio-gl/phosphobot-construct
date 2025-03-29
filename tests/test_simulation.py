"""
Unit tests for the phosphobot_construct.simulation module.
"""

import unittest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path to make imports work in testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock pybullet since we don't want to create a physics server in tests
sys.modules['pybullet'] = MagicMock()
sys.modules['pybullet_data'] = MagicMock()

from src.phosphobot_construct.simulation import PhosphobotSimulator, create_box_stacking_env


class TestPhosphobotSimulator(unittest.TestCase):
    """Tests for the PhosphobotSimulator class."""
    
    @patch('src.phosphobot_construct.simulation.p')
    def setUp(self, mock_p):
        """Setup for tests, create simulator instance."""
        # Configure mock behavior
        mock_p.connect.return_value = 1
        mock_p.getNumJoints.return_value = 6
        
        # Create simulator instance
        self.simulator = PhosphobotSimulator(gui=False)
        
        # Check if pybullet was initialized correctly
        mock_p.connect.assert_called_once()
        mock_p.setGravity.assert_called_once()
        mock_p.setTimeStep.assert_called_once()
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_setup_cameras(self, mock_p):
        """Test camera setup."""
        # Setup cameras
        self.simulator.setup_cameras(num_cameras=2)
        
        # Check that cameras were created
        self.assertEqual(len(self.simulator.cameras), 2)
        
        # Check camera properties
        for camera in self.simulator.cameras:
            self.assertIn("position", camera)
            self.assertIn("target", camera)
            self.assertIn("up_vector", camera)
            self.assertIn("fov", camera)
            self.assertIn("aspect", camera)
            self.assertIn("near_val", camera)
            self.assertIn("far_val", camera)
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_reset_robot(self, mock_p):
        """Test robot reset."""
        # Reset robot with default positions
        self.simulator.reset_robot()
        
        # Check that resetJointState was called for each joint
        self.assertEqual(mock_p.resetJointState.call_count, self.simulator.num_joints)
        
        # Reset with custom positions
        custom_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.simulator.reset_robot(joint_positions=custom_positions)
        
        # Check that resetJointState was called with correct positions
        calls = mock_p.resetJointState.call_args_list[-len(custom_positions):]
        for i, pos in enumerate(custom_positions):
            # Extract the positional arguments instead of keyword arguments
            args, _ = calls[i]
            # The target position is typically the third argument (index 2) in resetJointState
            # resetJointState(bodyUniqueId, jointIndex, targetValue, ...)
            self.assertEqual(args[2], pos)
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_step_simulation(self, mock_p):
        """Test simulation stepping."""
        # Step simulation once
        self.simulator.step_simulation(num_steps=1)
        
        # Check that stepSimulation was called once
        mock_p.stepSimulation.assert_called_once()
        
        # Step simulation multiple times
        mock_p.stepSimulation.reset_mock()
        self.simulator.step_simulation(num_steps=5)
        
        # Check that stepSimulation was called 5 times
        self.assertEqual(mock_p.stepSimulation.call_count, 5)
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_set_joint_positions(self, mock_p):
        """Test setting joint positions."""
        # Set joint positions
        positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.simulator.set_joint_positions(positions)
        
        # Check that setJointMotorControl2 was called for each joint
        self.assertEqual(mock_p.setJointMotorControl2.call_count, len(positions))
        
        # Check that it was called with the correct parameters
        for i, pos in enumerate(positions):
            call_args = mock_p.setJointMotorControl2.call_args_list[i][1]
            self.assertEqual(call_args["bodyIndex"], self.simulator.robot_id)
            self.assertEqual(call_args["jointIndex"], i)
            self.assertEqual(call_args["controlMode"], mock_p.POSITION_CONTROL)
            self.assertEqual(call_args["targetPosition"], pos)
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_get_joint_states(self, mock_p):
        """Test getting joint states."""
        # Configure mock to return a specific joint state
        mock_p.getJointState.side_effect = [
            (0.1, 0.2, [0.3, 0.4, 0.5], 0.6),
            (0.2, 0.3, [0.4, 0.5, 0.6], 0.7),
            (0.3, 0.4, [0.5, 0.6, 0.7], 0.8),
            (0.4, 0.5, [0.6, 0.7, 0.8], 0.9),
            (0.5, 0.6, [0.7, 0.8, 0.9], 1.0),
            (0.6, 0.7, [0.8, 0.9, 1.0], 1.1),
        ]
        
        # Get joint states
        joint_states = self.simulator.get_joint_states()
        
        # Check that getJointState was called for each joint
        self.assertEqual(mock_p.getJointState.call_count, self.simulator.num_joints)
        
        # Check that the returned states have the correct shape
        self.assertEqual(joint_states["positions"].shape, (6,))
        self.assertEqual(joint_states["velocities"].shape, (6,))
        self.assertEqual(joint_states["reaction_forces"].shape, (6, 3))
        
        # Check specific values
        np.testing.assert_array_equal(
            joint_states["positions"],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        )
        np.testing.assert_array_equal(
            joint_states["velocities"],
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        )
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_capture_camera_image(self, mock_p):
        """Test capturing camera images."""
        # Configure mock to return a specific image
        width, height = 320, 240
        mock_p.getCameraImage.return_value = (
            width, 
            height, 
            np.zeros((width * height * 4,), dtype=np.uint8),  # RGBA
            np.zeros((width * height,), dtype=np.float32),    # Depth
            np.zeros((width * height,), dtype=np.uint8)       # Segmentation
        )
        
        # Capture camera image
        image_data = self.simulator.capture_camera_image(camera_idx=0)
        
        # Check that getCameraImage was called
        mock_p.getCameraImage.assert_called_once()
        
        # Check that computeViewMatrix and computeProjectionMatrixFOV were called
        mock_p.computeViewMatrix.assert_called_once()
        mock_p.computeProjectionMatrixFOV.assert_called_once()
        
        # Check that the returned images have the correct shape
        self.assertEqual(image_data["rgb"].shape, (height, width, 3))
        self.assertEqual(image_data["depth"].shape, (height, width))
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_load_objects(self, mock_p):
        """Test loading objects into the simulation."""
        # Mock os.path.exists to always return True
        with patch('os.path.exists', return_value=True):
            # Configure mock to return specific object IDs
            mock_p.loadURDF.side_effect = [101, 102, 103]
            
            # Create object configurations
            object_configs = [
                {
                    "urdf_path": "cube.urdf",
                    "position": [0.1, 0.2, 0.3],
                    "orientation": [0.0, 0.0, 0.0],
                    "scale": 1.0
                },
                {
                    "urdf_path": "sphere.urdf",
                    "position": [0.4, 0.5, 0.6],
                    "orientation": [0.1, 0.2, 0.3],
                    "scale": 0.5
                },
                {
                    "urdf_path": "cylinder.urdf",
                    "position": [0.7, 0.8, 0.9],
                    "orientation": [0.4, 0.5, 0.6],
                    "scale": 2.0
                }
            ]
            
            # Load objects
            object_ids = self.simulator.load_objects(object_configs)
            
            # Check that loadURDF was called for each object
            self.assertEqual(mock_p.loadURDF.call_count, len(object_configs))
            
            # Check that the correct object IDs were returned
            self.assertEqual(object_ids, [101, 102, 103])
            
            # Check that getQuaternionFromEuler was called for each object
            self.assertEqual(mock_p.getQuaternionFromEuler.call_count, len(object_configs))
    
    @patch('src.phosphobot_construct.simulation.p')
    def test_close(self, mock_p):
        """Test closing the simulator."""
        # Close simulator
        self.simulator.close()
        
        # Check that disconnect was called
        mock_p.disconnect.assert_called_once_with(self.simulator.physics_client)


@patch('src.phosphobot_construct.simulation.PhosphobotSimulator')
def test_create_box_stacking_env(mock_simulator_class):
    """Test creating a box stacking environment."""
    # Configure mock
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator
    mock_simulator.load_objects.return_value = [101, 102, 103]
    
    # Create environment
    sim, box_ids = create_box_stacking_env()
    
    # Check that simulator was created with gui=True
    mock_simulator_class.assert_called_once_with(gui=True)
    
    # Check that load_objects was called
    mock_simulator.load_objects.assert_called_once()
    
    # Check that step_simulation was called 100 times to let the simulation settle
    assert mock_simulator.step_simulation.call_count == 100
    
    # Check that the correct values were returned
    assert sim == mock_simulator
    assert box_ids == [101, 102, 103]


if __name__ == "__main__":
    unittest.main()
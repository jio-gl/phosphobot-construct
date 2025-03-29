"""
Unit tests for the phosphobot_construct.control module.
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock, call, ANY

# Add parent directory to path to make imports work in testing
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phosphobot_construct.control import ClosedLoopController, adaptive_control


class TestClosedLoopController(unittest.TestCase):
    """Tests for the ClosedLoopController class."""
    
    def setUp(self):
        """Setup for tests, create controller instance with mock client."""
        # Create mock client
        self.mock_client = MagicMock()
        
        # Mock read_joints method to return a specific state
        mock_state = MagicMock()
        mock_state.angles_rad = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.mock_client.control.read_joints.return_value = mock_state
        
        # Create controller
        self.controller = ClosedLoopController(
            client=self.mock_client,
            feedback_rate=100.0,  # High rate for faster tests
            error_threshold=0.01,
            max_velocity=0.5
        )
    
    def test_init(self):
        """Test controller initialization."""
        # Check controller attributes
        self.assertEqual(self.controller.client, self.mock_client)
        self.assertEqual(self.controller.dt, 0.01)  # 1/100.0
        self.assertEqual(self.controller.error_threshold, 0.01)
        self.assertEqual(self.controller.max_velocity, 0.5)
        
        # Check PID gains
        self.assertGreater(self.controller.kp, 0)
        self.assertGreater(self.controller.ki, 0)
        self.assertGreater(self.controller.kd, 0)
        
        # Check state variables
        self.assertTrue(np.all(self.controller.integral_error == 0))
        self.assertTrue(np.all(self.controller.previous_error == 0))
    
    def test_pid_control(self):
        """Test PID control law calculation."""
        # Create error vector
        error = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])
        
        # Initialize state variables
        self.controller.integral_error = np.zeros(6)
        self.controller.previous_error = np.zeros(6)
        
        # Call PID control
        control = self.controller.pid_control(error)
        
        # Check control dimensions
        self.assertEqual(control.shape, (6,))
        
        # Check control direction (should be proportional to error)
        for i in range(6):
            if error[i] > 0:
                self.assertGreater(control[i], 0)
            elif error[i] < 0:
                self.assertLess(control[i], 0)
        
        # Check integral term accumulation
        self.assertTrue(np.all(self.controller.integral_error == error * self.controller.dt))
        
        # Call PID control again with same error
        control2 = self.controller.pid_control(error)
        
        # Integral term should have increased
        self.assertTrue(np.all(self.controller.integral_error == 2 * error * self.controller.dt))
        
        # Call PID control with zero error
        zero_error = np.zeros(6)
        control3 = self.controller.pid_control(zero_error)
        
        # Control should still be non-zero due to integral term
        self.assertTrue(np.any(control3 != 0))
    
    def test_pid_control_clipping(self):
        """Test that PID control output is properly clipped."""
        # Create large error to force clipping
        large_error = np.array([10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
        
        # Call PID control
        control = self.controller.pid_control(large_error)
        
        # Check all values are within limits
        self.assertTrue(np.all(control <= self.controller.max_velocity))
        self.assertTrue(np.all(control >= -self.controller.max_velocity))
        
        # Check that some values are at the limits (clipped)
        self.assertTrue(np.any(control == self.controller.max_velocity) or 
                       np.any(control == -self.controller.max_velocity))
    
    @patch('time.sleep')
    def test_move_to_position_success(self, mock_sleep):
        """Test successful movement to target position."""
        # Target position
        target_position = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
        
        # Current position will start at a distance and converge to target
        current_positions = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # Initial
            np.array([0.13, 0.23, 0.33, 0.43, 0.53, 0.63]),  # Intermediate
            np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])   # Final
        ]
        
        # Create feedback function that returns positions in sequence
        position_iter = iter(current_positions)
        def mock_feedback():
            return next(position_iter)
        
        # Call move_to_position
        result = self.controller.move_to_position(
            target_position=target_position,
            feedback_func=mock_feedback,
            timeout=5.0
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["final_position"], current_positions[-1].tolist())
        self.assertLess(result["time_elapsed"], 5.0)
        
        # Check that write_joints was called at least once
        self.mock_client.control.write_joints.assert_called()
    
    @patch('time.sleep')
    def test_move_to_position_timeout(self, mock_sleep):
        """Test movement that times out before reaching target."""
        # Target position
        target_position = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Far away
        
        # Current position will not converge fast enough
        current_positions = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # Initial
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),  # Still far from target
            np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])   # Still far from target
        ]
        
        # Simulate timeout by having sleep raise exception on third call
        mock_sleep.side_effect = [None, None, Exception("Timeout")]
        
        # Create feedback function that returns positions in sequence
        position_iter = iter(current_positions)
        def mock_feedback():
            return next(position_iter)
        
        # Call move_to_position
        try:
            result = self.controller.move_to_position(
                target_position=target_position,
                feedback_func=mock_feedback,
                timeout=0.1  # Short timeout
            )
            
            # Should not reach here due to exception
            self.fail("Expected exception was not raised")
            
        except Exception:
            # This is expected
            pass
        
        # Check that write_joints was called at least once
        self.mock_client.control.write_joints.assert_called()
    
    @patch('time.sleep')
    def test_move_to_position_stop_condition(self, mock_sleep):
        """Test movement with custom stop condition."""
        # Target position
        target_position = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
        
        # Current position will start at a distance
        current_position = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        # Create feedback function that always returns the same position
        def mock_feedback():
            return current_position
        
        # Create stop condition that triggers after 3 calls
        call_count = 0
        def stop_condition():
            nonlocal call_count
            call_count += 1
            return call_count >= 3
        
        # Call move_to_position
        result = self.controller.move_to_position(
            target_position=target_position,
            feedback_func=mock_feedback,
            timeout=5.0,
            stop_condition=stop_condition
        )
        
        # Check result
        self.assertFalse(result["success"])  # Should not succeed due to stop condition
        self.assertEqual(result["final_position"], current_position.tolist())
        
        # Check that stop condition was called until it returned True
        self.assertEqual(call_count, 3)
    
    @patch('time.sleep')
    def test_execute_trajectory(self, mock_sleep):
        """Test trajectory execution."""
        # Create a simple trajectory
        trajectory = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ]
        
        # Create mock for move_to_position that succeeds for all waypoints
        self.controller.move_to_position = MagicMock(return_value={"success": True})
        
        # Call execute_trajectory
        result = self.controller.execute_trajectory(
            trajectory=trajectory,
            timeout=10.0
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["waypoints_reached"], 3)
        
        # Check that move_to_position was called for each waypoint
        self.assertEqual(self.controller.move_to_position.call_count, 3)
        
        # Check call arguments
        calls = [call(target_position=wp, feedback_func=None, timeout=ANY) 
                 for wp in trajectory]
        self.controller.move_to_position.assert_has_calls(calls)
    
    @patch('time.sleep')
    def test_execute_trajectory_partial_success(self, mock_sleep):
        """Test trajectory execution with some failed waypoints."""
        # Create a simple trajectory
        trajectory = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ]
        
        # Create mock for move_to_position that fails for the second waypoint
        move_results = [
            {"success": True},
            {"success": False},
            {"success": True}
        ]
        self.controller.move_to_position = MagicMock(side_effect=move_results)
        
        # Call execute_trajectory
        result = self.controller.execute_trajectory(
            trajectory=trajectory,
            timeout=10.0
        )
        
        # Check result
        self.assertFalse(result["success"])  # Overall success should be False
        self.assertEqual(result["waypoints_reached"], 2)  # But two waypoints were reached
        
        # Check that move_to_position was called for each waypoint
        self.assertEqual(self.controller.move_to_position.call_count, 3)
    
    @patch('time.sleep')
    def test_execute_trajectory_timeout(self, mock_sleep):
        """Test trajectory execution with timeout."""
        # Create a long trajectory
        trajectory = [np.array([0.1 * i] * 6) for i in range(10)]
        
        # Create mock for move_to_position that takes a long time
        def mock_move(target_position, feedback_func, timeout):
            # Simulate taking 2 seconds per waypoint
            time.sleep(2)
            return {"success": True}
        
        self.controller.move_to_position = MagicMock(side_effect=mock_move)
        
        # Mock time.sleep to advance time without actually sleeping
        original_time = time.time
        mock_time = [original_time()]
        
        def mock_time_function():
            return mock_time[0]
        
        def mock_sleep_function(seconds):
            mock_time[0] += seconds
        
        with patch('time.time', side_effect=mock_time_function):
            with patch('time.sleep', side_effect=mock_sleep_function):
                # Call execute_trajectory with short timeout
                result = self.controller.execute_trajectory(
                    trajectory=trajectory,
                    timeout=5.0  # Only enough time for 2 waypoints
                )
        
        # Check result
        self.assertFalse(result["success"])
        self.assertLess(result["waypoints_reached"], len(trajectory))


class TestAdaptiveControl(unittest.TestCase):
    """Tests for the adaptive_control function."""
    
    @patch('src.phosphobot_construct.control.ClosedLoopController')
    def test_adaptive_control_single_position(self, mock_controller_class):
        """Test adaptive control with a single target position."""
        # Create mock client
        mock_client = MagicMock()
        
        # Create mock controller
        mock_controller = MagicMock()
        mock_controller_class.return_value = mock_controller
        
        # Mock move_to_position to return success
        mock_controller.move_to_position.return_value = {"success": True}
        
        # Call adaptive_control with a single position
        target_position = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = adaptive_control(
            client=mock_client,
            target_positions=target_position,
            feedback_rate=30.0,
            timeout=5.0
        )
        
        # Check controller initialization
        mock_controller_class.assert_called_once_with(
            client=mock_client,
            feedback_rate=30.0
        )
        
        # Check move_to_position call
        mock_controller.move_to_position.assert_called_once()
        args, kwargs = mock_controller.move_to_position.call_args
        np.testing.assert_array_equal(kwargs["target_position"], target_position)
        self.assertEqual(kwargs["timeout"], 5.0)
        
        # Check result
        self.assertEqual(result, {"success": True})
    
    @patch('src.phosphobot_construct.control.ClosedLoopController')
    def test_adaptive_control_trajectory(self, mock_controller_class):
        """Test adaptive control with a trajectory."""
        # Create mock client
        mock_client = MagicMock()
        
        # Create mock controller
        mock_controller = MagicMock()
        mock_controller_class.return_value = mock_controller
        
        # Mock execute_trajectory to return success
        mock_controller.execute_trajectory.return_value = {"success": True}
        
        # Call adaptive_control with a trajectory
        trajectory = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ]
        result = adaptive_control(
            client=mock_client,
            target_positions=trajectory,
            feedback_rate=30.0,
            timeout=10.0
        )
        
        # Check controller initialization
        mock_controller_class.assert_called_once_with(
            client=mock_client,
            feedback_rate=30.0
        )
        
        # Check execute_trajectory call
        mock_controller.execute_trajectory.assert_called_once()
        args, kwargs = mock_controller.execute_trajectory.call_args
        self.assertEqual(kwargs["trajectory"], trajectory)
        self.assertEqual(kwargs["timeout"], 10.0)
        
        # Check result
        self.assertEqual(result, {"success": True})
        
    @patch('src.phosphobot_construct.control.ClosedLoopController')
    def test_adaptive_control_with_perception(self, mock_controller_class):
        """Test adaptive control with perception function."""
        # Create mock client
        mock_client = MagicMock()
        
        # Create mock controller
        mock_controller = MagicMock()
        mock_controller_class.return_value = mock_controller
        
        # Mock move_to_position to return success
        mock_controller.move_to_position.return_value = {"success": True}
        
        # Create mock perception function
        mock_perception = MagicMock(return_value={"objects": []})
        
        # Call adaptive_control with perception
        target_position = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = adaptive_control(
            client=mock_client,
            target_positions=target_position,
            perception_func=mock_perception,
            feedback_rate=30.0,
            timeout=5.0
        )
        
        # Check move_to_position call
        mock_controller.move_to_position.assert_called_once()
        args, kwargs = mock_controller.move_to_position.call_args
        
        # Extract the feedback function
        feedback_func = kwargs.get("feedback_func")
        self.assertIsNotNone(feedback_func)
        
        # Call the feedback function
        feedback_result = feedback_func()
        
        # Check that perception was called and the result was used
        mock_perception.assert_called_once()
        self.assertIsInstance(feedback_result, np.ndarray)
        
        # Check final result - only verify that 'success' is True and don't care about other keys
        self.assertTrue(result['success'])
        
        # Verify the final_scene key exists in the result
        self.assertIn('final_scene', result)

if __name__ == "__main__":
    unittest.main()
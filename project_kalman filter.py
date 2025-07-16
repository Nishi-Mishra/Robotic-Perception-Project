# -*- coding: utf-8 -*-
"""

Created on Mon Apr 21 00:06:05 2025

@author: ayush
"""

import websocket
import json
import numpy as np
import time
import math
import threading
from collections import deque

import cv2 as cv
from operator import itemgetter

#############################################################################
class Quaternion:
    """Class for quaternion operations"""
    
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def normalize(self):
        """Normalize quaternion to unit length"""
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0.00001:  # Avoid division by zero
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self
    
    def inverse(self):
        """Return the inverse of this quaternion"""
        # For unit quaternions, the inverse is the same as the conjugate
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def multiply(self, q):
        """Multiply this quaternion by another quaternion"""
        w = self.w*q.w - self.x*q.x - self.y*q.y - self.z*q.z
        x = self.w*q.x + self.x*q.w + self.y*q.z - self.z*q.y
        y = self.w*q.y - self.x*q.z + self.y*q.w + self.z*q.x
        z = self.w*q.z + self.x*q.y - self.y*q.x + self.z*q.w
        return Quaternion(w, x, y, z)
    
    def __mul__(self, other):
        """Operator overloading for multiplication"""
        return self.multiply(other)
    
    def to_rotation_matrix(self):
        """Convert quaternion to 3x3 rotation matrix"""
        # Normalize quaternion
        self.normalize()
        
        # Calculate rotation matrix elements
        xx = self.x * self.x
        xy = self.x * self.y
        xz = self.x * self.z
        xw = self.x * self.w
        yy = self.y * self.y
        yz = self.y * self.z
        yw = self.y * self.w
        zz = self.z * self.z
        zw = self.z * self.w
        
        # Construct rotation matrix
        rot_matrix = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
        ])
        
        return rot_matrix
    
    @staticmethod
    def from_angular_velocity(gyro, dt):
        """Create a quaternion from angular velocity"""
        # Extract angular velocity components
        wx, wy, wz = gyro
        
        # Calculate rotation angle
        angle = math.sqrt(wx*wx + wy*wy + wz*wz) * dt
        
        # If rotation is very small, return identity quaternion
        if angle < 0.0001:
            return Quaternion(1, 0, 0, 0)
        
        # Calculate axis of rotation
        axis_x = wx / angle
        axis_y = wy / angle
        axis_z = wz / angle
        
        # Calculate quaternion components
        half_angle = angle * 0.5
        sin_half_angle = math.sin(half_angle)
        cos_half_angle = math.cos(half_angle)
        
        return Quaternion(
            cos_half_angle,
            axis_x * sin_half_angle,
            axis_y * sin_half_angle,
            axis_z * sin_half_angle
        )
    
    @staticmethod
    def slerp(q1, q2, t):
        """Spherical linear interpolation between two quaternions"""
        # Ensure q1 and q2 are unit quaternions
        q1 = Quaternion(q1.w, q1.x, q1.y, q1.z).normalize()
        q2 = Quaternion(q2.w, q2.x, q2.y, q2.z).normalize()
        
        # Calculate cosine of angle between quaternions
        dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
        
        # If dot < 0, take the shorter path
        if dot < 0:
            q2.w = -q2.w
            q2.x = -q2.x
            q2.y = -q2.y
            q2.z = -q2.z
            dot = -dot
        
        # Clamp to avoid domain errors due to floating point imprecision
        dot = min(1.0, max(-1.0, dot))
        
        # Calculate interpolation parameters
        if dot > 0.9995:
            # Quaternions are very close - linear interpolation
            result = Quaternion(
                q1.w + t*(q2.w - q1.w),
                q1.x + t*(q2.x - q1.x),
                q1.y + t*(q2.y - q1.y),
                q1.z + t*(q2.z - q1.z)
            )
            return result.normalize()
        
        # Calculate interpolation using spherical interpolation
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        result = Quaternion(
            s0 * q1.w + s1 * q2.w,
            s0 * q1.x + s1 * q2.x,
            s0 * q1.y + s1 * q2.y,
            s0 * q1.z + s1 * q2.z
        )
        
        return result.normalize()


class PositionTracker:
    """
    Class to track phone position and orientation using sensor fusion
    """
    def __init__(self):
        # Position and orientation
        self.position = np.zeros(3)  # [x, y, z]
        self.orientation_quat = Quaternion(1, 0, 0, 0)  # Identity quaternion
        self.rotation_matrix = np.eye(3)  # Identity rotation matrix
        self.euler = np.zeros(3)
        
        # Velocities
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.angular_velocity = np.zeros(3)  # [ωx, ωy, ωz]
        
        # Sensor data storage
        self.linear_accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.rotation_vector = np.array([0, 0, 0, 1])  # Rotation vector as quaternion
        
        # Timing
        self.last_update_time = None
        self.last_gyro_time = None
        
        # For tracking relative to starting position
        self.initial_orientation = None
        self.initial_rotation_matrix = np.eye(3)
        self.relative_position = np.zeros(3)
        self.relative_orientation = Quaternion(1, 0, 0, 0)
        
        # For camera integration
        self.calibration_square_visible = False
        self.camera_position_estimate = None
        self.camera_orientation_estimate = None
        
        # Sensor fusion parameters
        self.gyro_weight = 0.85  # Weight for gyroscope in orientation fusion
        self.rotation_vector_weight = 0.15  # Weight for rotation vector sensor
        self.imu_position_weight = 0.3  # Weight for IMU-based position when camera is available
        self.imu_orientation_weight = 0.5  # Weight for IMU-based orientation when camera is available
        
        # Filter state variables
        self.filter_state = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.filter_covariance = np.eye(6)  # Covariance matrix
        
        # Status flags
        self.is_initialized = False
        self.calibration_complete = False
    
    def filter_measurements(self, measurements, timestamp):
        """
        Apply filtering to measurements
        
        PSEUDOCODE for how to use this filter function:
        
        1. This function should be called every time new data is available
           (from either IMU sensors or camera)
        
        2. The filter combines all available data sources:
           - IMU data (accelerometer, gyroscope, rotation vector)
           - Camera data (position and orientation when available)
        
        3. Ideal filter implementation would:
           a. Take current state estimate and covariance
           b. Apply prediction step using IMU data
           c. Apply update step using camera data when available
           d. Return updated state and covariance
        
        4. Try to implement as a full Extended Kalman Filter (EKF):
           - Define state transition model based on IMU physics
           - Define measurement model for camera observations
           - Calculate appropriate Jacobians and noise matrices
           - Implement standard EKF equations
        
        For now, this function uses a simple weighted average approach.
        """
        # Extract IMU and camera measurements
        imu_position = measurements.get('imu_position')
        imu_orientation = measurements.get('imu_orientation')
        camera_position = measurements.get('camera_position')
        camera_orientation = measurements.get('camera_orientation')
        
        # Get current state and covariance 
        state = self.filter_state
        covariance = self.filter_covariance
        
        # Simple prediction step (could be more sophisticated)
        # Here we just use IMU data as our prediction
        if imu_position is not None:
            predicted_position = np.array(imu_position)
        else:
            predicted_position = state[0:3]
            
        if imu_orientation is not None:
            predicted_orientation = np.array(imu_orientation)
        else:
            predicted_orientation = state[3:6]
        
        # Initialize updated state as prediction
        updated_position = predicted_position
        updated_orientation = predicted_orientation
        
        # Update step with camera data if available
        if camera_position is not None and camera_orientation is not None:
            # Simple weighted averaging (could be replaced with proper EKF update)
            if self.calibration_square_visible:
                camera_pos_weight = 1.0 - self.imu_position_weight
                camera_ori_weight = 1.0 - self.imu_orientation_weight
                
                updated_position = (self.imu_position_weight * predicted_position + 
                                  camera_pos_weight * np.array(camera_position))
                                  
                updated_orientation = (self.imu_orientation_weight * predicted_orientation +
                                     camera_ori_weight * np.array(camera_orientation))
        
        # Update state and covariance
        # (In a full EKF, covariance would be updated properly)
        updated_state = np.concatenate([updated_position, updated_orientation])
        
        # Simple covariance update (should be replaced with proper EKF formulas)
        if self.calibration_square_visible:
            # If camera data available, reduce uncertainty
            updated_covariance = covariance * 0.8
        else:
            # If using only IMU data, increase uncertainty slightly
            updated_covariance = covariance * 1.05
            
        # Ensure covariance remains positive definite
        np.fill_diagonal(updated_covariance, 
                         np.maximum(0.001, np.diag(updated_covariance)))
        
        return updated_state, updated_covariance
    
    def initialize(self, initial_accel, initial_gyro, initial_rotation):
        """Initialize the tracker with starting values"""
        self.linear_accel = np.array(initial_accel)
        self.gyro = np.array(initial_gyro)
        self.rotation_vector = np.array(initial_rotation)
        
        # Set initial orientation from rotation vector
        self.orientation_quat = Quaternion(
            initial_rotation[3], 
            initial_rotation[0], 
            initial_rotation[1], 
            initial_rotation[2]
        )
        
        # Store initial orientation for relative calculations
        self.initial_orientation = Quaternion(
            self.orientation_quat.w,
            self.orientation_quat.x,
            self.orientation_quat.y,
            self.orientation_quat.z
        )
        
        # Calculate rotation matrix
        self.rotation_matrix = self.orientation_quat.to_rotation_matrix()
        self.initial_rotation_matrix = self.rotation_matrix.copy()
        
        # Initialize filter state
        self.filter_state = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.filter_covariance = np.eye(6)
        self.filter_covariance[0:3, 0:3] *= 0.01  # Low initial position uncertainty
        self.filter_covariance[3:6, 3:6] *= 0.01  # Low initial orientation uncertainty
        
        
        # Initialize Kalman filter variables
        n = 5  # Number of state variables
        self.t = 0.01 # s
        self.x = np.zeros((n, 1))  # State vector [angular position, x, y, x_dot, y_dot]
        self.P = np.eye(n) * 1.0  # Covariance matrix
        self.F = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, self.t, 0],
            [0, 0, 1, 0, self.t],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])  # State transition matrix
        self.B = np.array([
            [self.t, 0, 0],
            [0, 1/2 * self.t**2, 0],
            [0, 0, 1/2 * self.t**2],
            [0, self.t, 0],
            [0, 0, self.t]
        ])  # Control matrix
        self.u = np.zeros((3, 1))  # Control input [ax, ay, az]
        self.Q = np.eye(n) * 0.1  # Process noise covariance
        self.R = np.eye(n) * 0.5  # Measurement noise covariance
        self.H = np.eye(n)  # Measurement matrix
    
        # Identity matrix for covariance update
        self.I = np.eye(n)
        
        self.is_initialized = True
        print("Position tracker initialized. Starting position set to origin.")
    
    def update_orientation(self, gyro_data, rotation_vector_data, dt):
        """
        Update orientation by fusing gyroscope and rotation vector data
        """
        # Use gyroscope for incremental orientation update
        gyro_delta_quat = Quaternion.from_angular_velocity(gyro_data, dt)
        gyro_orientation = self.orientation_quat * gyro_delta_quat
        
        # Get absolute orientation from rotation vector sensor
        rot_vector_orientation = Quaternion(
            rotation_vector_data[3],
            rotation_vector_data[0],
            rotation_vector_data[1],
            rotation_vector_data[2]
        )
        
        # Fuse the two using complementary filter
        # This weights gyro data more for short-term accuracy and rotation vector for long-term stability
        fused_orientation = Quaternion.slerp(gyro_orientation, rot_vector_orientation, self.rotation_vector_weight)
        self.orientation_quat = fused_orientation.normalize()
        
        # Update rotation matrix
        self.rotation_matrix = self.orientation_quat.to_rotation_matrix()
        
        # Calculate relative orientation to starting position
        # This gives us orientation relative to where the phone started
        inv_initial = self.initial_orientation.inverse()
        self.relative_orientation = inv_initial * self.orientation_quat
        
        # Extract angular velocity from quaternion change
        # This is a rough approximation
        dq = self.orientation_quat.inverse() * gyro_orientation
        self.angular_velocity = np.array([
            2 * dq.x / dt if dq.w > 0 else -2 * dq.x / dt,
            2 * dq.y / dt if dq.w > 0 else -2 * dq.y / dt,
            2 * dq.z / dt if dq.w > 0 else -2 * dq.z / dt
        ])
        
        # Extract Euler angles for filter state
        # This is a simplified conversion that assumes small angles
        roll = math.atan2(2 * (self.orientation_quat.w * self.orientation_quat.x + 
                             self.orientation_quat.y * self.orientation_quat.z),
                       1 - 2 * (self.orientation_quat.x**2 + self.orientation_quat.y**2))
        pitch = math.asin(2 * (self.orientation_quat.w * self.orientation_quat.y - 
                             self.orientation_quat.z * self.orientation_quat.x))
        yaw = math.atan2(2 * (self.orientation_quat.w * self.orientation_quat.z + 
                            self.orientation_quat.x * self.orientation_quat.y),
                      1 - 2 * (self.orientation_quat.y**2 + self.orientation_quat.z**2))
        
        euler_angles = np.array([roll, pitch, yaw])
        self.euler = euler_angles
        
        return self.rotation_matrix, self.angular_velocity, euler_angles
    
    def update_position(self, linear_accel, dt):
        """
        Update position using acceleration and time delta
        """
        # Transform acceleration from device frame to world frame
        world_accel = self.rotation_matrix @ linear_accel
        
        # Simple velocity integration with high-pass filtering to reduce drift
        alpha = 0.95  # High-pass filter coefficient
        
        # Update velocity (high-pass filtered to reduce drift)
        self.velocity = alpha * (self.velocity + world_accel * dt)
        
        # Update position
        position_delta = self.velocity * dt
        self.position += position_delta
        
        # Update relative position (from starting point)
        self.relative_position = self.position.copy()
        
        return self.position, self.velocity
    
    def integrate_camera_data(self, camera_data):
        """
        Integrate camera-based position and orientation data if available
        """
        if camera_data is None:
            return self.position, self.orientation_quat
            
        calibration_visible = camera_data.get("calibration_visible", False)
        camera_position = camera_data.get("position")
        camera_orientation = camera_data.get("orientation")
        
        self.calibration_square_visible = calibration_visible
        
        if calibration_visible and camera_position is not None:
            self.camera_position_estimate = np.array(camera_position)
            
            if camera_orientation is not None:
                self.camera_orientation_estimate = np.array(camera_orientation)
                
                # Create camera orientation quaternion
                camera_quat = Quaternion(
                    camera_orientation[3], 
                    camera_orientation[0],
                    camera_orientation[1],
                    camera_orientation[2]
                )
                
                # Use filter to fuse IMU and camera orientation
                # The filtering happens in filter_measurements() function
        
        return self.position, self.orientation_quat
    
    def update(self, linear_accel_data, gyro_data, rotation_vector_data, timestamp, camera_data=None):
        """
        Main update function that processes all sensor data
        """
        # Initialize if needed
        if not self.is_initialized:
            self.initialize(linear_accel_data, gyro_data, rotation_vector_data)
            self.last_update_time = timestamp
            return {
                "position": self.position.tolist(),
                "orientation": [self.orientation_quat.x, self.orientation_quat.y, self.orientation_quat.z, self.orientation_quat.w],
                "velocity": self.velocity.tolist(),
                "angular_velocity": self.angular_velocity.tolist(),
                "relative_position": self.relative_position.tolist()
            }
        
        # Calculate time step
        dt = timestamp - self.last_update_time
        self.last_update_time = timestamp
        
        # Ensure valid time step
        if dt <= 0 or dt > 1.0:
            dt = 0.01  # Use reasonable default if time difference is weird
        
        # Update orientation using gyro and rotation vector
        _, _, euler_angles = self.update_orientation(gyro_data, rotation_vector_data, dt)
        
        # Update position (already using linear_acceleration, no gravity compensation needed)
        position, velocity = self.update_position(linear_accel_data, dt)
        
        # Prepare measurements for filtering
        measurements = {
            'imu_position': position,
            'imu_orientation': euler_angles,
            'camera_position': None,
            'camera_orientation': None
        }
        
        # Integrate camera data if available
        if camera_data is not None:
            self.integrate_camera_data(camera_data)
            
            if self.calibration_square_visible:
                measurements['camera_position'] = self.camera_position_estimate
                measurements['camera_orientation'] = self.camera_orientation_estimate
        
        # Apply filtering to combine IMU and camera data
        filtered_state, filtered_covariance = self.filter_measurements(measurements, timestamp)
        
        # Update state with filtered results
        self.filter_state = filtered_state
        self.filter_covariance = filtered_covariance
        
        # Update position from filtered state
        self.position = filtered_state[0:3]
        
        # Create result dictionary
        result = {
            "position": self.position.tolist(),
            "orientation": [self.orientation_quat.x, self.orientation_quat.y, self.orientation_quat.z, self.orientation_quat.w],
            "velocity": self.velocity.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "relative_position": self.relative_position.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "filter_state": self.filter_state.tolist(),
            "filter_uncertainty": np.diag(self.filter_covariance).tolist()
        }
        
        return result

    def predict(self):
        """Perform Kalman filter prediction step"""
        self.x = self.F @ self.x + self.B @ self.u  # Project state forward
        self.P = self.F @ self.P @ self.F.T + self.Q  # Project uncertainty forward
        print("Prediction Step Complete")
        print("Predicted State (x):", self.x.flatten())
        print("Predicted Covariance (P):\n", self.P)
        
    def update_kalman(self, z):
        """
        Perform Kalman filter update step
        Args:
            z: Measurement vector (observed state)
        """
        # Innovation (residual)
        y = z - (self.H @ self.x)  # Difference between measurement and prediction
    
        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R  # Uncertainty of residual
    
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Optimal blending factor
    
        # Update state estimate
        self.x = self.x + K @ y
    
        # Update covariance estimate
        self.P = (self.I - K @ self.H) @ self.P
    
        print("Update Step Complete")
        print("Updated State (x):", self.x.flatten())
        print("Updated Covariance (P):\n", self.P)

        
class CameraInterface:
    """
    Interface for webcam processing (mostly pseudocode)
    """
    def __init__(self):
        self.is_initialized = False
        self.camera_id = 0
        self.calibration_visible = False
        self.camera_data = None
    
    def initialize(self, camera_id=0):
        """Initialize webcam"""
        # PSEUDOCODE:
        # 1. Open the camera connection
        # 2. Set camera parameters (resolution, etc.)
        # 3. Check if camera is working properly
        # 4. Load camera calibration parameters if available
        # 5. Set up the markers/pattern detection system
        
        self.is_initialized = True  # Set to True if initialization succeeds
        print(f"Camera initialized with ID: {camera_id}")
        return True
    
    def get_camera_data(self):
        """Process frame and detect calibration square"""
        if not self.is_initialized:
            return None
            
        # PSEUDOCODE:
        # 1. Capture a frame from the camera
        # 2. Process the image to detect calibration square/markers:
        #    - Use OpenCV's ArUco markers or similar approach
        #    - Detect corners of calibration pattern
        # 3. If calibration pattern is detected:
        #    a. Use solvePnP or similar algorithm to get position AND orientation 
        #    b. Convert rotation matrix/vector to quaternion
        #    c. Transform to phone's coordinate system (important!)
        #    d. Apply any necessary refinements (e.g., filtering)
        # 4. Return both position and orientation data
        
        # Return dummy data for testing
        import random
        self.calibration_visible = random.random() > 0.7  # 30% chance to be visible
        
        if self.calibration_visible:
            # Simulate position
            position = [random.uniform(-0.1, 0.1), 
                       random.uniform(-0.1, 0.1), 
                       random.uniform(0.5, 0.7)]
            
            # Simulate orientation as quaternion [x, y, z, w]
            # Generate small random rotations
            angle = random.uniform(0, math.pi/8)  # Small random angle
            axis = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
            axis_norm = math.sqrt(sum(x*x for x in axis))
            axis = [x/axis_norm for x in axis]
            
            qx = axis[0] * math.sin(angle/2)
            qy = axis[1] * math.sin(angle/2)
            qz = axis[2] * math.sin(angle/2)
            qw = math.cos(angle/2)
            
            orientation = [qx, qy, qz, qw]
            
            camera_data = {
                "calibration_visible": True,
                "position": position,
                "orientation": orientation
            }
        else:
            camera_data = {
                "calibration_visible": False,
                "position": None,
                "orientation": None
            }
        
        return camera_data
    
    def release(self):
        """Release camera resources"""
        # PSEUDOCODE:
        # 1. Close camera connection
        # 2. Free any allocated resources
        
        self.is_initialized = False
        print("Camera released")


# Global tracker instance
position_tracker = PositionTracker()
camera_interface = CameraInterface()

# Flag for controlling the WebSocket
ws_running = True

def camera_thread():
    """Thread to handle camera processing"""
    print("Initializing camera...")
    camera_initialized = camera_interface.initialize()
    
    if not camera_initialized:
        print("Camera not available. Using IMU-only tracking.")
    
    # Main camera processing loop
    while ws_running and camera_initialized:
        # Get camera data periodically
        position_tracker.camera_data = camera_interface.get_camera_data()
        time.sleep(0.1)  # 10Hz update rate


def user_input_thread():
    """Thread to handle user input for control"""
    global ws_running
    
    print("\n===== POSITION TRACKING SYSTEM =====")
    print("- Press 'r' to reset position to origin")
    print("- Press 'q' to quit")
    
    while ws_running:
        user_input = input().strip().lower()
        
        if user_input == 'q':
            print("Shutting down...")
            ws_running = False
            break
            
        elif user_input == 'r':
            # Reset position to origin but keep orientation
            position_tracker.position = np.zeros(3)
            position_tracker.velocity = np.zeros(3)
            position_tracker.relative_position = np.zeros(3)
            print("Position reset to origin")
        
        time.sleep(0.1)


# WebSocket handlers
def on_message(ws, message):
    if not ws_running:
        ws.close()
        return
        
    data = json.loads(message)
    sensor_type = data['type']
    values = data['values']
    timestamp = time.time()
    
    # Process different sensor types
    if sensor_type == "android.sensor.linear_acceleration":
        linear_accel = values  # Extract accelerometer data
        position_tracker.linear_accel = linear_accel  # Store in tracker

    elif sensor_type == "android.sensor.gyroscope":
        gyro_data = values  # Extract gyroscope data
        position_tracker.gyro_data = gyro_data  # Store in tracker

    # Ensure we have both gyro and accel data before prediction
    if hasattr(position_tracker, 'linear_accel') and hasattr(position_tracker, 'gyro_data'):
        # Control input for prediction
        position_tracker.u = np.array([
            [position_tracker.gyro_data[2]],  # Angular velocity about z-axis
            [position_tracker.linear_accel[0]],  # x acceleration
            [position_tracker.linear_accel[1]]   # y acceleration
        ])
        position_tracker.predict()  # Perform predict step
    
        # Prepare measurement vector
        z_imu = np.array([
            [position_tracker.gyro_data[2]],   # Angular position (z-axis)
            [position_tracker.position[0]],   # x position
            [position_tracker.position[1]],   # y position
            [position_tracker.velocity[0]],   # x velocity
            [position_tracker.velocity[1]]    # y velocity
        ])
        position_tracker.update_kalman(z_imu)  # Update using IMU data
    
        # Optionally update with camera data
# =============================================================================
#         if position_tracker.camera_data and position_tracker.camera_data.get('calibration_visible', False):
#             z_camera = np.array([
#                 [0],
#                 [position_tracker.camera_data["position"][0]],
#                 [position_tracker.camera_data["position"][1]],
#                 [0],
#                 [0]
#             ])
#             position_tracker.update_kalman(z_camera)
# =============================================================================



def on_error(ws, error):
    print("Error occurred:")
    print(error)


def on_close(ws, close_code, reason):
    print("Connection closed:")
    print(f"Close code: {close_code}")
    print(f"Reason: {reason}")
    
    # Clean up camera resources
    if camera_interface.is_initialized:
        camera_interface.release()


def on_open(ws):
    print("WebSocket connected!")
    print("Sensor data streaming started.")
    print("Waiting for initial measurements...")


def connect(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,        
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Start the user input thread
    input_thread = threading.Thread(target=user_input_thread)
    input_thread.daemon = True
    input_thread.start()
    
    # Start the camera processing thread
    camera_thread_handle = threading.Thread(target=camera_thread)
    camera_thread_handle.daemon = True
    camera_thread_handle.start()
    
    print("Connecting to sensor WebSocket...")
    ws.run_forever()
    
    # If we get here, the WebSocket has closed
    print("WebSocket connection ended.")


if __name__ == "__main__":
    # Connect to the phone's sensors - using linear acceleration, gyroscope, and rotation vector
    url = 'ws://10.244.125.246:8080/sensors/connect?types=["android.sensor.linear_acceleration","android.sensor.gyroscope","android.sensor.rotation_vector"]'
    connect(url)

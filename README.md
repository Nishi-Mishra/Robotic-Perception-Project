# 2D Tracking of a Smartphone Using Sensor Fusion and Computer Vision

This project, completed as part of a robotic perception course, focused on implementing a 2D tracking system for a smartphone using two complementary data sources: on-board inertial sensors (accelerometer, gyroscope, and rotation vector) and computer vision via a webcam. The phone's motion was tracked on a flat plane, with real-time IMU data streamed to a computer over a WebSocket connection and processed using OpenCV to localize a checkerboard marker attached to the device. A Kalman filter was used to fuse the IMU and vision signals for robust state estimation.

Although the full 3D system was not pursued, the project successfully demonstrated reliable 2D tracking (x, y, yaw) using sensor fusion. This simplified version provides a strong foundation for real-time filtering and motion inference using smartphone data.


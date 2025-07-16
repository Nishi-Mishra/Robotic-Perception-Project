# -*- coding: utf-8 -*-
"""
2D IMU Tracker - Gyroscope + Accelerometer
Tracks x, y, and yaw using planar IMU data
"""

import websocket
import json
import numpy as np
import time
import math
import threading

# State variables
x = 0.0
y = 0.0
yaw = 0.0  # In radians
vx = 0.0
vy = 0.0
last_time = None
ws_running = True
sensor_data_ready = {"gyro": False, "accel": False}

latest_gyro = [0.0, 0.0, 0.0]
latest_accel = [0.0, 0.0, 0.0]

def reset():
    global x, y, yaw, vx, vy, last_time
    x = 0.0
    y = 0.0
    yaw = 0.0
    vx = 0.0
    vy = 0.0
    last_time = None
    print("System reset to origin.")

def update_state(accel_data, gyro_z, dt):
    global x, y, yaw, vx, vy

    # Update yaw from gyroscope z-axis
    yaw += gyro_z * dt

    # Rotate body-frame acceleration to world-frame
    ax, ay = accel_data[0], accel_data[1]
    ax_world = ax * math.cos(yaw) - ay * math.sin(yaw)
    ay_world = ax * math.sin(yaw) + ay * math.cos(yaw)

    # Integrate velocity
    vx += ax_world * dt
    vy += ay_world * dt

    # Integrate position
    x += vx * dt
    y += vy * dt

    return x, y, yaw, vx, vy

def user_input_thread():
    global ws_running
    print("\n===== 2D IMU TRACKING SYSTEM =====")
    print("- Press 'r' to reset position to origin")
    print("- Press 'q' to quit")
    while ws_running:
        user_input = input().strip().lower()
        if user_input == 'q':
            print("Shutting down...")
            ws_running = False
            break
        elif user_input == 'r':
            reset()
        time.sleep(0.1)

def on_message(ws, message):
    global latest_gyro, latest_accel, last_time

    data = json.loads(message)
    sensor_type = data['type']
    values = data['values']
    timestamp = time.time()

    if sensor_type == "android.sensor.gyroscope":
        latest_gyro = values
        sensor_data_ready["gyro"] = True

    elif sensor_type == "android.sensor.linear_acceleration":
        latest_accel = values
        sensor_data_ready["accel"] = True

    if all(sensor_data_ready.values()):
        if last_time is None:
            last_time = timestamp
            return

        dt = timestamp - last_time
        last_time = timestamp

        if dt <= 0 or dt > 1.0:
            dt = 0.01

        x_out, y_out, yaw_out, vx_out, vy_out = update_state(latest_accel, latest_gyro[2], dt)

        print("\n-- IMU Update --")
        print(f"x: {x_out:.2f} m, y: {y_out:.2f} m, yaw: {math.degrees(yaw_out):.1f} deg")
        print(f"vx: {vx_out:.2f} m/s, vy: {vy_out:.2f} m/s")

def on_error(ws, error):
    print("Error occurred:", error)

def on_close(ws, close_code, reason):
    print("Connection closed.")
    print("Close code:", close_code)
    print("Reason:", reason)

def on_open(ws):
    print("WebSocket connected. Starting 2D IMU tracking...")

def connect(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    threading.Thread(target=user_input_thread, daemon=True).start()
    ws.run_forever()
    print("WebSocket connection ended.")

if __name__ == "__main__":
    connect('ws://10.245.53.174:8080/sensors/connect?types=["android.sensor.linear_acceleration","android.sensor.gyroscope"]')

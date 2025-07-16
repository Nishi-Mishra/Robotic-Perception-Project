# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:10:18 2025

@author: nishi
"""

import websocket
import json
import numpy as np
import time

# Storage for sensor data
accel_data = []
gyro_data = []
rotvec_data = []

# Timing
start_time = None
duration = 10  # seconds

def on_message(ws, message):
    global start_time

    data = json.loads(message)
    sensor_type = data["type"]
    values = data["values"]
    now = time.time()

    if start_time is None:
        start_time = now
        print("Started capturing data...")

    elapsed = now - start_time
    if elapsed > duration:
        ws.close()
        return

    if sensor_type == "android.sensor.accelerometer":
        accel_data.append(values[:3])
    elif sensor_type == "android.sensor.gyroscope":
        gyro_data.append(values[:3])
    elif sensor_type == "android.sensor.rotation_vector":
        rotvec_data.append(values[:4])  # Includes x, y, z, w

def on_error(ws, error):
    print("Error occurred:", error)

def on_close(ws, close_code, reason):
    print("Connection closed.")
    print("Close code:", close_code)
    print("Reason:", reason)

    def summarize(name, data):
        arr = np.array(data)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        print(f"\n{name} SUMMARY:")
        print(f"  Mean: {mean}")
        print(f"  Std Dev (noise): {std}")

    summarize("Accelerometer", accel_data)
    summarize("Gyroscope", gyro_data)
    summarize("Rotation Vector (Quaternion: x, y, z, w)", rotvec_data)

def on_open(ws):
    print("WebSocket connected. Collecting data for 10 seconds...")

def connect(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# Run the WebSocket connection
connect('ws://10.244.125.246:8080/sensors/connect?types=["android.sensor.accelerometer","android.sensor.gyroscope","android.sensor.rotation_vector"]')

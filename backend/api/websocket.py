"""
KINEX WebSocket Telemetry Endpoint
Handles real-time bi-directional streaming between frontend dashboard / ESP32 glove
and the inference backend.
"""

import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.inference.engine import InferenceEngine

ws_router = APIRouter()
engine = InferenceEngine()


@ws_router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    Bi-directional streaming endpoint:
      Client -> Server: Telemetry frame (flex sensor + IMU) or 21 keypoints
      Server -> Client: Predicted ISL gesture, confidence, and system status
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
                continue

            msg_type = payload.get("type", "telemetry")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))
                continue

            # Check if keypoints provided
            keypoints = payload.get("keypoints")
            sensor_readings = payload.get("sensors")

            if keypoints and len(keypoints) == 21:
                res = engine.process_keypoints(keypoints, sensor_fusion=sensor_readings)
                response = {
                    "type": "display_update",
                    "gesture": res["gesture"],
                    "confidence": res["confidence"],
                    "probabilities": res["probabilities"],
                    "latency_ms": res["latency_ms"],
                    "pipeline": "yolo26_pose",
                    "status": res["status"],
                    "timestamp_ms": time.time() * 1000.0,
                }
            elif sensor_readings:
                # Telemetry only: mock or evaluate glove kinematics
                # Generate synthetic canonical feature vector and fuse with glove readings
                canonical_kpts, _, _ = engine._generate_canonical_hand(640, 480)
                res = engine.process_keypoints(canonical_kpts, sensor_fusion=sensor_readings)
                response = {
                    "type": "display_update",
                    "gesture": res["gesture"],
                    "confidence": res["confidence"],
                    "probabilities": res["probabilities"],
                    "latency_ms": res["latency_ms"],
                    "pipeline": "esp32_glove_fusion",
                    "status": res["status"],
                    "timestamp_ms": time.time() * 1000.0,
                }
            else:
                response = {
                    "type": "telemetry_ack",
                    "status": "idle",
                    "timestamp_ms": time.time() * 1000.0,
                }

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass
    except Exception:
        pass

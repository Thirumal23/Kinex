"""
KINEX YOLO Pose Inference & Testing Script
Runs inference on an image, video, or live webcam.
Visualizes:
  - 1 hand bounding box
  - 21 hand keypoints and kinematic skeleton
  - Inference latency (ms) and FPS
  - Hand detected / Not detected status

STRICT ENFORCEMENT:
- Displays ONLY "KINEX YOLO26 POSE"
- NEVER displays gesture classification labels (e.g. THANK YOU, BYE)
- ZERO MediaPipe dependencies.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Kinematic connections for 21 hand keypoints (wrist=0)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


def draw_pose(frame, pred):
    h, w = frame.shape[:2]
    # Header overlay
    cv2.putText(frame, "KINEX YOLO26 POSE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if not pred["hand_detected"]:
        cv2.putText(frame, "HAND: NOT DETECTED", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"LATENCY: {pred['inference_time_ms']} ms", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return frame

    # Status overlay
    cv2.putText(frame, f"HAND: DETECTED ({pred['confidence']*100:.1f}%)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"LATENCY: {pred['inference_time_ms']} ms", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "21 KEYPOINTS", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Draw Bbox
    bbox = pred["bbox"]
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Keypoints
    kpts = pred["keypoints"]
    pts = []
    for k in kpts:
        kx, ky = int(k["x"]), int(k["y"])
        vis = k.get("visibility", 2.0)
        pts.append((kx, ky, vis))
        if vis > 0.5:
            cv2.circle(frame, (kx, ky), 4, (0, 0, 255), -1)

    # Skeleton bones
    for p1_idx, p2_idx in SKELETON_EDGES:
        if p1_idx < len(pts) and p2_idx < len(pts):
            x_a, y_a, v_a = pts[p1_idx]
            x_b, y_b, v_b = pts[p2_idx]
            if v_a > 0.5 and v_b > 0.5:
                cv2.line(frame, (x_a, y_a), (x_b, y_b), (255, 0, 0), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Test trained KINEX YOLO26 Pose model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained best.pt")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    from vision.yolo_pose.inference import YOLOPoseEstimator

    estimator = YOLOPoseEstimator(checkpoint_path=args.checkpoint, device=args.device)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not read image at {args.image}")
            return
        pred = estimator.predict_frame(frame)
        out = draw_pose(frame, pred)
        out_path = PROJECT_ROOT / "reports" / "test_pose_output.jpg"
        cv2.imwrite(str(out_path), out)
        print(f"Inference complete. Output saved to: {out_path}")

    elif args.webcam or args.video:
        src = 0 if args.webcam else args.video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Error: Could not open video source {src}")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pred = estimator.predict_frame(frame)
            out = draw_pose(frame, pred)
            cv2.imshow("KINEX YOLO26 POSE TEST", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

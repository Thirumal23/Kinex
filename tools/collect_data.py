#!/usr/bin/env python3
"""
KINEX Data Collection CLI — Phase 2 Production Tool
Interactive and automated camera utility for capturing ISL gesture datasets with
real-time quality gating, perceptual duplicate filtering, and session tracking.

Controls (Live Mode):
    [SPACE] : Manual capture
    [A]     : Toggle automatic interval capture
    [N]     : Switch to next gesture in vocabulary
    [P]     : Pause / Resume capture
    [Q/ESC] : Conclude session, validate integrity, and exit

Usage:
    python tools/collect_data.py --subject S01 --session 01 --gesture THANK_YOU
    python tools/collect_data.py --live-test-session  # Controlled 30-frame live webcam test
    python tools/collect_data.py --synthetic-test     # Automated headless test
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.data_collection.collector import DataCollector


def run_synthetic_test(collector: DataCollector):
    """Executes a headless test using synthetic frames to verify pipeline without a camera."""
    print("[TEST] Running synthetic data collection test...")
    session_dir = collector.start_session(
        subject_id="S01_SYNTH",
        session_id="TEST_01",
        gesture="THANK_YOU",
        hand="RIGHT",
        lighting="INDOOR_NORMAL",
        camera_distance="MEDIUM",
        background="PLAIN_WALL",
    )
    print(f"[TEST] Session initialized at: {session_dir}")

    # Generate 5 varying synthetic frames
    for i in range(5):
        frame = np.full((480, 640, 3), fill_value=(40 + i * 30, 100, 150), dtype=np.uint8)
        # Add high-contrast circle to satisfy sharpness threshold
        cv2 = sys.modules.get("cv2")
        if cv2:
            cv2.circle(frame, (100 + i * 60, 200), 50, (255, 255, 255), -1)
            cv2.putText(frame, f"TEST {i}", (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        saved, reason, record = collector.capture_frame(frame, force=(i == 0))
        print(f"  Frame {i+1}: saved={saved}, reason={reason}")
        time.sleep(0.05)

    summary = collector.end_session()
    print(f"[TEST] Session finished. Total accepted: {summary.get('total_accepted')}")
    assert summary.get("total_accepted", 0) > 0, "Synthetic test failed: 0 frames accepted"
    assert summary.get("validation", {}).get("passed", False), "Session validation failed"
    print("[TEST] Synthetic data collection test PASSED successfully!")
    return True


def run_live_test_session(collector: DataCollector, camera_id: int = 0, target_frames: int = 30):
    """Executes a controlled live webcam test session (S00_TEST / TEST_SESSION / THANK_YOU)."""
    import cv2
    print("\n" + "=" * 60)
    print("KINEX CONTROLLED LIVE WEBCAM TEST SESSION")
    print(f"Subject: S00_TEST | Session: TEST_SESSION | Gesture: THANK_YOU | Target: {target_frames} frames")
    print("=" * 60)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Live camera {camera_id} could not be opened.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    collector.start_session(
        subject_id="S00_TEST",
        session_id="TEST_SESSION",
        gesture="THANK_YOU",
        hand="RIGHT",
        lighting="INDOOR_NORMAL",
        camera_distance="MEDIUM",
        background="PLAIN_WALL",
    )

    accepted = 0
    start_time = time.time()
    last_cap_time = 0.0

    print("[INFO] Capturing test frames from live webcam...")
    while accepted < target_frames and (time.time() - start_time) < 30.0:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        now = time.time()
        if (now - last_cap_time) >= 0.20:
            saved, reason, record = collector.capture_frame(frame)
            if saved:
                accepted = collector.session_accepted_count
                print(f"  [ACCEPTED {accepted:02d}/{target_frames}] {record['filename']} | Sharpness: {record['sharpness']} | Brightness: {record['mean_brightness']}")
                last_cap_time = now
            elif "Duplicate" in reason:
                print(f"  [REJECTED - DUPLICATE] Sim: {record['similarity_score'] if record else 'N/A'}")
            time.sleep(0.05)

    cap.release()
    summary = collector.end_session()
    print("\n" + "=" * 60)
    print("TEST SESSION SUMMARY")
    print(f"Captured: {summary['total_captured']} | Accepted: {summary['total_accepted']} | Rejected: {summary['total_rejected']}")
    print(f"Duplicates: {summary['rejections']['duplicates']} | Blur: {summary['rejections']['blur']} | Exposure: {summary['rejections']['exposure']}")
    print(f"Validation Passed: {summary['validation']['passed']}")
    print("=" * 60 + "\n")
    return summary['validation']['passed'] and summary['total_accepted'] >= 10


def main():
    parser = argparse.ArgumentParser(description="KINEX ISL Hand Sign Data Collection Tool")
    parser.add_argument("--subject", default="S01", help="Subject ID (e.g. S01, S02)")
    parser.add_argument("--session", default="01", help="Session ID (e.g. 01, 02)")
    parser.add_argument("--gesture", default="THANK_YOU", help="Initial gesture class name")
    parser.add_argument("--hand", default="RIGHT", choices=["RIGHT", "LEFT", "BOTH"], help="Target hand")
    parser.add_argument("--lighting", default="INDOOR_NORMAL", choices=["INDOOR_NORMAL", "INDOOR_BRIGHT", "INDOOR_DIM", "NATURAL_LIGHT"], help="Lighting condition")
    parser.add_argument("--distance", default="MEDIUM", choices=["NEAR", "MEDIUM", "FAR"], help="Camera distance")
    parser.add_argument("--background", default="PLAIN_WALL", choices=["PLAIN_WALL", "DESK", "ROOM", "OFFICE", "LAB"], help="Background environment")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index")
    parser.add_argument("--auto-interval", type=float, default=0.20, help="Auto-capture interval in seconds")
    parser.add_argument("--similarity-threshold", type=float, default=0.95, help="Threshold to reject duplicate frames")
    parser.add_argument("--min-sharpness", type=float, default=35.0, help="Minimum Laplacian variance for blur filter")
    parser.add_argument("--min-brightness", type=float, default=30.0, help="Minimum brightness threshold")
    parser.add_argument("--max-brightness", type=float, default=235.0, help="Maximum brightness threshold")
    parser.add_argument("--target-count", type=int, default=100, help="Target number of accepted frames for batch capture")
    parser.add_argument("--synthetic-test", action="store_true", help="Run automated test with synthetic frames")
    parser.add_argument("--live-test-session", action="store_true", help="Run automated live webcam test session (S00_TEST)")
    args = parser.parse_args()

    collector = DataCollector(
        gestures_config_path="configs/gestures.yaml",
        raw_output_root="datasets/hand_pose/raw",
        metadata_output_root="datasets/hand_pose/metadata",
        rejected_output_root="datasets/hand_pose/rejected",
        similarity_threshold=args.similarity_threshold,
        min_capture_interval_s=args.auto_interval,
        min_sharpness=args.min_sharpness,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
    )

    if args.synthetic_test:
        success = run_synthetic_test(collector)
        sys.exit(0 if success else 1)

    if args.live_test_session:
        success = run_live_test_session(collector, camera_id=args.camera_id, target_frames=args.target_count)
        sys.exit(0 if success else 1)

    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV (cv2) is required to run live camera collection.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with device ID {args.camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    gesture_list = list(collector.gesture_classes.values())
    current_gesture_idx = gesture_list.index(args.gesture.upper()) if args.gesture.upper() in gesture_list else 0
    current_gesture = gesture_list[current_gesture_idx]

    formatted_session_id = f"SESSION_{int(args.session):02d}" if args.session.isdigit() else args.session.upper()

    collector.start_session(
        subject_id=args.subject,
        session_id=formatted_session_id,
        gesture=current_gesture,
        hand=args.hand,
        lighting=args.lighting,
        camera_distance=args.distance,
        background=args.background,
    )

    print("\n" + "=" * 60)
    print("KINEX DATA COLLECTION — LIVE SESSION")
    print(f"Subject: {args.subject} | Session: {formatted_session_id} | Gesture: {current_gesture}")
    print("HUD Controls:")
    print("  [SPACE] : Manual Capture")
    print("  [A]     : Toggle Auto Interval Capture")
    print("  [N]     : Switch to Next Gesture")
    print("  [Q/ESC] : Conclude Session and Save Manifest")
    print("=" * 60 + "\n")

    auto_capture = False
    last_auto_capture_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            display_frame = frame.copy()
            now = time.time()

            # Automatic capture check
            if auto_capture and (now - last_auto_capture_time) >= args.auto_interval:
                saved, reason, _ = collector.capture_frame(frame)
                if saved:
                    last_auto_capture_time = now

            # On-screen HUD (semi-transparent overlay)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (480, 185), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

            cv2.putText(display_frame, f"KINEX COLLECTOR: {current_gesture}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Subject: {args.subject} | Session: {formatted_session_id}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Hand: {args.hand} | Light: {args.lighting} | Dist: {args.distance}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_frame, f"Accepted: {collector.session_accepted_count} | Rejected: {collector.session_rejected_count} (Dup: {collector.session_duplicates_count})", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Auto: {'ACTIVE (' + str(args.auto_interval) + 's)' if auto_capture else 'PAUSED'}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if auto_capture else (0, 165, 255), 2)
            cv2.putText(display_frame, "[SPACE] Cap | [A] Auto | [N] Next | [Q] Quit", (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("KINEX Data Collector", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                saved, reason, _ = collector.capture_frame(frame, force=True)
                print(f"[CAPTURE] Manual: {reason} (Accepted: {collector.session_accepted_count})")
            elif key in (ord('a'), ord('A')):
                auto_capture = not auto_capture
                print(f"[AUTO] Auto-capture toggled: {auto_capture}")
            elif key in (ord('n'), ord('N')):
                summary = collector.end_session()
                print(f"[GESTURE COMPLETED] Finished {current_gesture}: {summary['total_accepted']} accepted frames.")
                current_gesture_idx = (current_gesture_idx + 1) % len(gesture_list)
                current_gesture = gesture_list[current_gesture_idx]
                collector.start_session(
                    subject_id=args.subject,
                    session_id=formatted_session_id,
                    gesture=current_gesture,
                    hand=args.hand,
                    lighting=args.lighting,
                    camera_distance=args.distance,
                    background=args.background,
                )
                print(f"[NEXT GESTURE] Switched to: {current_gesture}")

    finally:
        summary = collector.end_session()
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("FINAL SESSION SUMMARY:")
        print(f"Total Accepted: {summary['total_accepted']}")
        print(f"Total Rejected: {summary['total_rejected']}")
        print(f"Validation: {summary['validation']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

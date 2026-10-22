"""
KINEX Data Collection GUI (Tkinter Desktop Application) — Phase 2 Production
Interactive graphical workbench for acquiring ISL hand sign datasets with quality gating,
perceptual similarity filtering, rejected frame tracking, and live session manifest logging.
"""

import os
import sys
import time
import yaml
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.data_collection.collector import DataCollector


class DataCollectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KINEX — ISL Hand Sign Data Collector (Phase 2)")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1E1E2E")

        self.collector = DataCollector(
            gestures_config_path="configs/gestures.yaml",
            raw_output_root="datasets/hand_pose/raw",
            metadata_output_root="datasets/hand_pose/metadata",
            rejected_output_root="datasets/hand_pose/rejected",
        )
        self.gestures = list(self.collector.gesture_classes.values())

        self.cap = None
        self.is_camera_running = False
        self.is_recording = False
        self.current_frame = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Top Header
        header = tk.Frame(self.root, bg="#181825", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        title_lbl = tk.Label(
            header,
            text="KINEX — ISL DATA COLLECTION WORKBENCH",
            font=("Helvetica", 16, "bold"),
            fg="#89B4FA",
            bg="#181825",
            pady=10,
        )
        title_lbl.pack()

        # Main Body: Left Panel (Controls) & Right Panel (Camera Preview)
        body = tk.Frame(self.root, bg="#1E1E2E")
        body.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Control Panel
        ctrl_frame = tk.LabelFrame(
            body,
            text="Session Controls & Metadata",
            font=("Helvetica", 11, "bold"),
            fg="#CDD6F4",
            bg="#1E1E2E",
            padx=12,
            pady=10,
        )
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        # Form Fields
        row = 0
        tk.Label(ctrl_frame, text="Subject ID:", fg="#BAC2DE", bg="#1E1E2E").grid(row=row, column=0, sticky="w", pady=3)
        self.subject_var = tk.StringVar(value="S01")
        tk.Entry(ctrl_frame, textvariable=self.subject_var, width=15).grid(row=row, column=1, pady=3)

        row += 1
        tk.Label(ctrl_frame, text="Session ID:", fg="#BAC2DE", bg="#1E1E2E").grid(row=row, column=0, sticky="w", pady=3)
        self.session_var = tk.StringVar(value="01")
        tk.Entry(ctrl_frame, textvariable=self.session_var, width=15).grid(row=row, column=1, pady=3)

        row += 1
        tk.Label(ctrl_frame, text="Gesture Class:", fg="#BAC2DE", bg="#1E1E2E").grid(row=row, column=0, sticky="w", pady=3)
        self.gesture_var = tk.StringVar(value=self.gestures[0] if self.gestures else "THANK_YOU")
        self.gesture_cb = ttk.Combobox(ctrl_frame, textvariable=self.gesture_var, values=self.gestures, state="readonly", width=14)
        self.gesture_cb.grid(row=row, column=1, pady=3)

        row += 1
        tk.Label(ctrl_frame, text="Hand:", fg="#BAC2DE", bg="#1E1E2E").grid(row=row, column=0, sticky="w", pady=3)
        self.hand_var = tk.StringVar(value="RIGHT")
        ttk.Combobox(ctrl_frame, textvariable=self.hand_var, values=["RIGHT", "LEFT", "BOTH"], state="readonly", width=14).grid(row=row, column=1, pady=3)

        row += 1
        tk.Label(ctrl_frame, text="Lighting:", fg="#BAC2DE", bg="#1E1E2E").grid(row=row, column=0, sticky="w", pady=3)
        self.light_var = tk.StringVar(value="INDOOR_NORMAL")
        ttk.Combobox(ctrl_frame, textvariable=self.light_var, values=["INDOOR_NORMAL", "INDOOR_BRIGHT", "INDOOR_DIM", "NATURAL_LIGHT"], state="readonly", width=14).grid(row=row, column=1, pady=3)

        row += 1
        tk.Label(ctrl_frame, text="Camera Dist:", fg="#BAC2DE", bg="#1E1E2E").grid(row=row, column=0, sticky="w", pady=3)
        self.dist_var = tk.StringVar(value="MEDIUM")
        ttk.Combobox(ctrl_frame, textvariable=self.dist_var, values=["NEAR", "MEDIUM", "FAR"], state="readonly", width=14).grid(row=row, column=1, pady=3)

        row += 1
        tk.Label(ctrl_frame, text="Background:", fg="#BAC2DE", bg="#1E1E2E").grid(row=row, column=0, sticky="w", pady=3)
        self.bg_var = tk.StringVar(value="PLAIN_WALL")
        ttk.Combobox(ctrl_frame, textvariable=self.bg_var, values=["PLAIN_WALL", "DESK", "ROOM", "OFFICE", "LAB"], state="readonly", width=14).grid(row=row, column=1, pady=3)

        # Buttons
        row += 1
        sep = ttk.Separator(ctrl_frame, orient="horizontal")
        sep.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)

        row += 1
        self.btn_cam = tk.Button(ctrl_frame, text="START CAMERA", bg="#A6E3A1", fg="#11111B", font=("Helvetica", 10, "bold"), width=20, command=self.toggle_camera)
        self.btn_cam.grid(row=row, column=0, columnspan=2, pady=4)

        row += 1
        self.btn_rec = tk.Button(ctrl_frame, text="START RECORDING", bg="#FAB387", fg="#11111B", font=("Helvetica", 10, "bold"), width=20, command=self.toggle_recording, state=tk.DISABLED)
        self.btn_rec.grid(row=row, column=0, columnspan=2, pady=4)

        row += 1
        self.btn_cap = tk.Button(ctrl_frame, text="CAPTURE FRAME", bg="#89B4FA", fg="#11111B", font=("Helvetica", 10, "bold"), width=20, command=self.manual_capture, state=tk.DISABLED)
        self.btn_cap.grid(row=row, column=0, columnspan=2, pady=4)

        row += 1
        nav_box = tk.Frame(ctrl_frame, bg="#1E1E2E")
        nav_box.grid(row=row, column=0, columnspan=2, pady=4)
        self.btn_prev = tk.Button(nav_box, text="< PREV GESTURE", bg="#CBA6F7", fg="#11111B", font=("Helvetica", 8, "bold"), width=11, command=self.prev_gesture)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.btn_next = tk.Button(nav_box, text="NEXT GESTURE >", bg="#CBA6F7", fg="#11111B", font=("Helvetica", 8, "bold"), width=11, command=self.next_gesture)
        self.btn_next.pack(side=tk.RIGHT, padx=2)

        row += 1
        self.btn_end = tk.Button(ctrl_frame, text="END SESSION", bg="#F38BA8", fg="#11111B", font=("Helvetica", 10, "bold"), width=20, command=self.end_session, state=tk.DISABLED)
        self.btn_end.grid(row=row, column=0, columnspan=2, pady=4)

        # Live Status Box
        row += 1
        self.status_box = tk.Label(
            ctrl_frame,
            text="Status: Ready\nCaptured: 0 | Accepted: 0\nRejected: 0 (Dup: 0, Blur: 0)",
            font=("Courier", 8),
            fg="#A6ADC8",
            bg="#11111B",
            justify=tk.LEFT,
            padx=8,
            pady=8,
            relief=tk.SUNKEN,
        )
        self.status_box.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # Camera Display Panel
        preview_frame = tk.LabelFrame(
            body,
            text="Live Webcam Feed",
            font=("Helvetica", 11, "bold"),
            fg="#CDD6F4",
            bg="#1E1E2E",
            padx=10,
            pady=10,
        )
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas_label = tk.Label(preview_frame, bg="#11111B", text="Camera Inactive\nClick 'START CAMERA' to begin", font=("Helvetica", 12), fg="#6C7086")
        self.canvas_label.pack(fill=tk.BOTH, expand=True)

    def toggle_camera(self):
        try:
            import cv2
        except ImportError:
            messagebox.showerror("Error", "OpenCV (cv2) is not installed.")
            return

        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Unable to open webcam.")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.is_camera_running = True
            self.btn_cam.config(text="STOP CAMERA", bg="#F38BA8")
            self.btn_rec.config(state=tk.NORMAL)
            self.btn_cap.config(state=tk.NORMAL)
            self._update_feed()
        else:
            self.is_camera_running = False
            if self.is_recording:
                self.toggle_recording()
            if self.cap:
                self.cap.release()
            self.btn_cam.config(text="START CAMERA", bg="#A6E3A1")
            self.btn_rec.config(state=tk.DISABLED)
            self.btn_cap.config(state=tk.DISABLED)
            self.canvas_label.config(image="", text="Camera Inactive")

    def toggle_recording(self):
        if not self.is_recording:
            try:
                formatted_session_id = f"SESSION_{int(self.session_var.get()):02d}" if self.session_var.get().isdigit() else self.session_var.get().upper()
                self.collector.start_session(
                    subject_id=self.subject_var.get(),
                    session_id=formatted_session_id,
                    gesture=self.gesture_var.get(),
                    hand=self.hand_var.get(),
                    lighting=self.light_var.get(),
                    camera_distance=self.dist_var.get(),
                    background=self.bg_var.get(),
                )
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return

            self.is_recording = True
            self.btn_rec.config(text="PAUSE RECORDING", bg="#F9E2AF")
            self.btn_end.config(state=tk.NORMAL)
            self._update_status("Recording started")
        else:
            self.is_recording = False
            self.btn_rec.config(text="RESUME RECORDING", bg="#FAB387")
            self._update_status("Recording paused")

    def manual_capture(self):
        if self.current_frame is not None and self.collector.session_active:
            saved, reason, _ = self.collector.capture_frame(self.current_frame, force=True)
            self._update_status(f"Manual: {reason}")

    def next_gesture(self):
        if self.collector.session_active:
            self.end_session()
        idx = self.gestures.index(self.gesture_var.get())
        next_idx = (idx + 1) % len(self.gestures)
        self.gesture_var.set(self.gestures[next_idx])
        self._update_status(f"Switched to {self.gestures[next_idx]}")

    def prev_gesture(self):
        if self.collector.session_active:
            self.end_session()
        idx = self.gestures.index(self.gesture_var.get())
        prev_idx = (idx - 1) % len(self.gestures)
        self.gesture_var.set(self.gestures[prev_idx])
        self._update_status(f"Switched to {self.gestures[prev_idx]}")

    def end_session(self):
        if self.collector.session_active:
            summary = self.collector.end_session()
            self.is_recording = False
            self.btn_rec.config(text="START RECORDING", bg="#FAB387")
            self.btn_end.config(state=tk.DISABLED)
            messagebox.showinfo(
                "Session Concluded",
                f"Session finished!\nAccepted: {summary.get('total_accepted')}\n"
                f"Rejected: {summary.get('total_rejected')}\nValidation: {summary.get('validation', {}).get('passed')}"
            )
            self._update_status("Session concluded and verified.")

    def _update_status(self, custom_msg=""):
        txt = (
            f"Subject: {self.subject_var.get()} | Gesture: {self.gesture_var.get()}\n"
            f"Captured: {self.collector.session_captured_count} | Accepted: {self.collector.session_accepted_count}\n"
            f"Rejected: {self.collector.session_rejected_count} (Dup: {self.collector.session_duplicates_count}, Blur: {self.collector.session_blur_count}, Exp: {self.collector.session_exposure_count})\n"
            f"State: {'RECORDING' if self.is_recording else 'PAUSED'}"
        )
        if custom_msg:
            txt += f"\nEvent: {custom_msg}"
        self.status_box.config(text=txt)

    def _update_feed(self):
        if self.is_camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                if self.is_recording:
                    saved, reason, _ = self.collector.capture_frame(frame)
                    if saved:
                        self._update_status()

                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (640, 480))
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.canvas_label.config(image=img, text="")
                self.canvas_label.image = img

            self.root.after(30, self._update_feed)


def main():
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

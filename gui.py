import os
import cv2
import sys
import math
import numpy as np
from datetime import datetime
from collections import deque

import torch
import torch.nn.functional as F
from ultralytics import YOLO  # Spatial contextual vehicle tracker

# PyQt6 Core UI elements
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, 
    QHBoxLayout, QPushButton, QComboBox, QLineEdit, QSlider, QTextEdit, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# Backend Network imports
from models.slowfast_model import SuspiciousActivityModel
from utils import pack_pathway_output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Normal", "Violence", "Theft", "Property Damage", "Physical Abuse/Aggression"]
VEHICLE_CLASSES = [2, 3, 5, 7]  # YOLO COCO indexes: Car, Motorcycle, Bus, Truck

# ============================================================
# 🧵 WORKER THREAD: SEPARATES AI BACKEND FROM MAIN GUI
# ============================================================
class VideoProcessingThread(QThread):
    # Asynchronous signal pipelines to pass assets back to UI thread safely
    change_pixmap_signal = pyqtSignal(np.ndarray)
    add_log_signal = pyqtSignal(str)

    def __init__(self, source, max_width, freq):
        super().__init__()
        # Try to convert source to integer if it's a webcam index, otherwise keep as string path
        try:
            self.source = int(source)
        except ValueError:
            self.source = source
            
        self.max_width = max_width
        self.freq = freq
        self._run_flag = True

    def run(self):
        # 1. Pipeline initialization and network attachment
        self.add_log_signal.emit("Deploying background CUDA kernels...")
        
        # Initialize 3D CNN Core
        sf_model = SuspiciousActivityModel(num_classes=len(CLASSES))
        checkpoint_path = "checkpoints/slowfast_dcsass_e8.pth"
        if os.path.exists(checkpoint_path):
            sf_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            self.add_log_signal.emit(f"Successfully pinned production weights: {checkpoint_path}")
        else:
            self.add_log_signal.emit("CRITICAL WARNING: Checkpoint weights missing! Using native weights.")
        
        sf_model.to(DEVICE).eval()

        # Initialize Upstream YOLO Module
        yolo_model = YOLO("yolov8n.pt")

        # 2. Establish video pipeline
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.add_log_signal.emit(f"ERROR: Video source target '{self.source}' could not be initialized.")
            return

        self.add_log_signal.emit("Surveillance framework online. Processing stream...")

        # Setup rolling structural context caches
        frame_buffer = deque(maxlen=32)
        prediction_history = deque(maxlen=15)
        yolo_cooldown_counter = 0
        smoothed_display_prediction = "Normal"

        frame_count = 0
        current_prediction = "Normal"
        confidence_pct = 100.0
        kidnap_flag_active = False

        # 3. Target execution loop
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.add_log_signal.emit("End of media stream sequence detected or camera disconnected.")
                break

            frame_count += 1

            # Transform raw frame arrays for 3D network dimensions (224x224x3 Normalized RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_ai = cv2.resize(rgb_frame, (224, 224))
            normalized_ai = resized_ai.astype(np.float32) / 255.0
            tensor_frame = torch.tensor(normalized_ai).permute(2, 0, 1)
            frame_buffer.append(tensor_frame)

            # Evaluate slow and fast spatial pathways concurrently if frequency step hits match conditions
            if len(frame_buffer) == 32 and (frame_count % self.freq == 0):
                # Stack individual tensors along temporal axis (C, T, H, W)
                video_tensor = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(DEVICE)
                inputs = pack_pathway_output(video_tensor)

                with torch.no_grad():
                    outputs = sf_model(inputs)
                    probabilities = F.softmax(outputs, dim=1)
                
                confidence, predicted = torch.max(probabilities, 1)
                current_prediction = CLASSES[predicted.item()]
                confidence_pct = confidence.item() * 100

                prediction_history.append(current_prediction)

                # Temporal label smoothing: aggregate active threat markers
                active_violence_signals = prediction_history.count("Violence") + prediction_history.count("Physical Abuse/Aggression")
                
                if active_violence_signals >= 2 and confidence_pct > 70.0:
                    if yolo_cooldown_counter == 0:
                        self.add_log_signal.emit(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] Threat signature verified: {current_prediction.upper()} ({confidence_pct:.1f}%)")
                    yolo_cooldown_counter = 45  # 45 frame structural hysteresis cooldown latch
                    if active_violence_signals > prediction_history.count("Normal"):
                        smoothed_display_prediction = current_prediction

                kidnap_flag_active = False

                # Spatial Geometric Multi-Modal Fusion Engine
                if yolo_cooldown_counter > 0:
                    yolo_results = yolo_model(frame, device=DEVICE, verbose=False)[0]
                    people_centers = []
                    vehicle_boxes = []

                    for box in yolo_results.boxes:
                        cls_idx = int(box.cls[0].item())
                        coords = box.xyxy[0].cpu().numpy().astype(int)

                        if cls_idx == 0:  # Human Centroid Vector calculation
                            cx = int((coords[0] + coords[2]) / 2)
                            cy = int((coords[1] + coords[3]) / 2)
                            people_centers.append((cx, cy))
                        elif cls_idx in VEHICLE_CLASSES:
                            vehicle_boxes.append(coords)

                    # Euclidean Distance Boundary Vectoring Rule
                    for (px, py) in people_centers:
                        for vx1, vy1, vx2, vy2 in vehicle_boxes:
                            dx = max(vx1 - px, 0, px - vx2)
                            dy = max(vy1 - py, 0, py - vy2)
                            spatial_distance = math.sqrt(dx**2 + dy**2)

                            if spatial_distance < 120:  # Proximity threshold trigger parameter
                                kidnap_flag_active = True
                                break
                        if kidnap_flag_active:
                            break

                    if kidnap_flag_active:
                        self.add_log_signal.emit(f"🚨 ALERT [{datetime.now().strftime('%H:%M:%S')}] HIGHEST THREAT VECTOR DETECTED: SUSPECTED ABDUCTION EVENT ACTIVE")

                    yolo_cooldown_counter -= 1

                if yolo_cooldown_counter == 0 and current_prediction == "Normal":
                    smoothed_display_prediction = "Normal"

            # Render display properties according to runtime statuses
            orig_h, orig_w = frame.shape[:2]
            if orig_w > self.max_width:
                scale = self.max_width / orig_w
                display_w = self.max_width
                display_h = int(orig_h * scale)
                frame = cv2.resize(frame, (display_w, display_h))
            else:
                display_w = orig_w
                display_h = orig_h

            # Generate graphics bar layout text overlay
            if len(frame_buffer) < 32:
                box_color = (0, 255, 255)
                overlay_text = f"BUFFERING TEMPORAL PIPELINE ({len(frame_buffer)}/32)"
            elif kidnap_flag_active:
                box_color = (0, 0, 255)
                overlay_text = f"CRITICAL: SECURITY BREACH / POSSIBLE VEHICLE ABDUCTION [{confidence_pct:.1f}%]"
            elif smoothed_display_prediction != "Normal":
                box_color = (0, 165, 255)
                overlay_text = f"WARNING: {smoothed_display_prediction.upper()} SIGNATURE DETECTED [{confidence_pct:.1f}%]"
            else:
                box_color = (0, 255, 0)
                overlay_text = "SURVEILLANCE ENGINE STATUS: NORMAL OPERATIONS ACTIVE"

            # Draw top overlay status banner bar
            cv2.rectangle(frame, (0, 0), (display_w, 45), (15, 15, 15), -1)
            cv2.putText(frame, overlay_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

            # Signal processed frame arrays back to the main UI slot loop
            self.change_pixmap_signal.emit(frame)

        cap.release()
        self.add_log_signal.emit("Surveillance stream released. Thread engine safe.")

    def stop(self):
        self._run_flag = False
        self.wait()


# ============================================================
# 🖥️ MAIN USER INTERFACE WORKSPACE: PYQT6 CONTROL PANEL
# ============================================================
class IntelligentCCTVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Stream Intelligent CCTV Console - Professional Deployment")
        self.setGeometry(100, 100, 1400, 850)
        self.setStyleSheet("background-color: #0e1117; color: #f1f5f9; font-family: 'Segoe UI';")

        self.thread = None

        # Base central container layout definitions
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Column Layout: Controls and configuration widgets panel
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(15, 15, 15, 15)
        left_panel.setSpacing(12)

        title_lbl = QLabel("Surveillance Core Engine Parameters")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #38bdf8; padding-bottom: 5px;")
        left_panel.addWidget(title_lbl)

        # Input source path panel controls (UPDATED: Added Explorer Navigation Browse Shortcut)
        path_lbl = QLabel("Target Source Feed Input (File Path or Webcam Index):")
        path_lbl.setStyleSheet("font-weight: bold; color: #94a3b8;")
        left_panel.addWidget(path_lbl)
        
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setText("0")  # Defaulting fallback choice index to default web camera identifier
        self.path_input.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; padding: 6px; border-radius: 4px; color: white;")
        path_layout.addWidget(self.path_input)

        self.btn_browse = QPushButton("📁 Browse")
        self.btn_browse.setStyleSheet("background-color: #334155; font-weight: bold; padding: 6px 12px; border-radius: 4px; color: #f1f5f9;")
        self.btn_browse.clicked.connect(self.browse_file)
        path_layout.addWidget(self.btn_browse)
        left_panel.addLayout(path_layout)

        # Max resolution processing scale modifiers
        width_lbl = QLabel("Max Dynamic Screen Resolution Scaling Width:")
        width_lbl.setStyleSheet("font-weight: bold; color: #94a3b8;")
        left_panel.addWidget(width_lbl)

        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setMinimum(640)
        self.width_slider.setMaximum(1920)
        self.width_slider.setValue(1024)
        self.width_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.width_slider.setTickInterval(256)
        self.width_slider.setStyleSheet("height: 25px;")
        left_panel.addWidget(self.width_slider)

        self.width_val_lbl = QLabel("Active Target Value Boundary: 1024 pixels")
        self.width_val_lbl.setStyleSheet("color: #38bdf8; font-size: 11px;")
        self.width_slider.valueChanged.connect(lambda v: self.width_val_lbl.setText(f"Active Target Value Boundary: {v} pixels"))
        left_panel.addWidget(self.width_val_lbl)

        # Inference skip rate configurations
        freq_lbl = QLabel("3D Temporal Inference Evaluator Step Skipping Interval:")
        freq_lbl.setStyleSheet("font-weight: bold; color: #94a3b8;")
        left_panel.addWidget(freq_lbl)

        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(1)
        self.freq_slider.setMaximum(10)
        self.freq_slider.setValue(2)
        self.freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_slider.setTickInterval(1)
        left_panel.addWidget(self.freq_slider)

        self.freq_val_lbl = QLabel("Evaluate Network Quantization Every: 2 frames")
        self.freq_val_lbl.setStyleSheet("color: #38bdf8; font-size: 11px;")
        self.freq_slider.valueChanged.connect(lambda v: self.freq_val_lbl.setText(f"Evaluate Network Quantization Every: {v} frames"))
        left_panel.addWidget(self.freq_val_lbl)

        # Operational master engine control button initialization
        self.btn_action = QPushButton("⚡ Start CCTV Analyzer")
        self.btn_action.setStyleSheet("background-color: #d9534f; font-weight: bold; font-size: 14px; padding: 12px; border-radius: 6px; color: white;")
        self.btn_action.clicked.connect(self.toggle_analysis_state)
        left_panel.addWidget(self.btn_action)

        # System Logging widget screen console output layout definitions
        log_lbl = QLabel("System Operation Logs Console Outputs:")
        log_lbl.setStyleSheet("font-weight: bold; color: #94a3b8; padding-top: 10px;")
        left_panel.addWidget(log_lbl)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #090d16; border: 1px solid #1e293b; font-family: 'Consolas'; font-size: 11px; color: #a7f3d0; border-radius: 4px;")
        left_panel.addWidget(self.log_box)

        # Right Column Layout: Active Canvas Screen display elements output frame views
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 10, 10, 10)

        screen_header = QLabel("🔴 REAL-TIME AI SURVEILLANCE DESKTOP MONITOR FEED")
        screen_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #ef4444; padding-bottom: 2px;")
        right_panel.addWidget(screen_header)

        self.video_display = QLabel("Surveillance Core Engine Asleep. Initialize Frame Loop Stream to Awaken.")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
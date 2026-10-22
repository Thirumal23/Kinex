# KINEX — Current System Architecture Specification

```
   ██╗  ██╗██╗███╗   ██╗███████╗██╗  ██╗
   ██║ ██╔╝██║████╗  ██║██╔════╝╚██╗██╔╝
   █████╔╝ ██║██╔██╗ ██║█████╗   ╚███╔╝ 
   ██╔═██╗ ██║██║╚██╗██║██╔══╝   ██╔██╗ 
   ██║  ██╗██║██║ ╚████║███████╗██╔╝ ██╗
   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
   Hand Sign Processing, Recognition & Embedded Edge Translation
```

---

## 1. Executive Summary & Core Philosophy

**KINEX** is an end-to-end Indian Sign Language (ISL) recognition and assistive communication system. It bridges deep computer vision running on host workstations with low-power wearable sensory telemetry (ESP32 + Flex Sensors + IMU) and decentralized edge visual displays (ST7735 TFT).

### 1.1 The Decoupling Principle
A foundational tenet of KINEX is avoiding monolithic end-to-end classification:
- **YOLO26 Pose is NOT the Sign Classifier**: The deep vision backbone is strictly a spatial landmark detector answering: *"Where is the hand, and where are its 21 anatomical joints?"*
- **The Downstream Classifier answers**: *"What ISL sign do these geometric and physical relationships represent?"*

```
                     ┌────────────────────────────────┐
                     │          Camera Frame          │
                     └───────────────┬────────────────┘
                                     │ 640x640 RGB
                                     ▼
                     ┌────────────────────────────────┐
                     │       YOLO26 Pose Engine       │  <-- Spatial localization
                     │   (Random Weights / Scratch)   │      (21 Keypoints + BBox)
                     └───────────────┬────────────────┘
                                     │ Raw Joint Coordinates (x, y, v)
                                     ▼
                     ┌────────────────────────────────┐
                     │  Geometric Feature Normalizer  │  <-- Scale, translation &
                     │     (Wrist-Origin Invariant)   │      angle invariance (V4)
                     └───────────────┬────────────────┘
                                     │ 126-D Feature Vector
                                     ▼
                     ┌────────────────────────────────┐
                     │   Semantic Gesture Classifier  │  <-- Lightweight MLP / NN
                     │     (7 Classes + Reject Gate)  │      (Independent of backbone)
                     └───────────────┬────────────────┘
                                     │ Class Probabilities
                                     ▼
                     ┌────────────────────────────────┐
                     │  Temporal Smoothing & Buffering│  <-- Dwell time & voting
                     │   (16-Frame Queue / GRU Engine)│      (Flicker elimination)
                     └───────────────┬────────────────┘
                                     │ Verified ISL Sign
                                     ▼
                     ┌───────────────┴────────────────┐
                     ▼                                ▼
            Laptop UI / FastAPI API           ESP32 ST7735 TFT Screen
```

### 1.2 Architectural Benefits of Decoupling
1. **Modular Vocabulary Expansion**: New signs can be introduced by simply updating `configs/gestures.yaml` and training the lightweight MLP classifier in seconds, without retraining the deep pose backbone.
2. **Mathematical Invariance**: Raw pixel coordinates are converted into position-invariant and distance-invariant geometric vectors before classification.
3. **Multimodal Fusion Readiness**: Standardized geometric features can be concatenated directly with physical sensor telemetry (flex resistance and 6-axis IMU pitch/roll).
4. **Verifiable Quality Gating**: Low-confidence or degenerate hand keypoints are trapped before reaching the classifier, preventing erratic predictions.

---

## 2. End-to-End System Architecture Diagram

```mermaid
flowchart TD
    %% INPUT SUBSYSTEM
    subgraph S1["1. Data Acquisition Layer"]
        CAM["Laptop Webcam (720p/1080p BGR)"]
        GLOVE["Wearable Sensor Glove\n(5x Flex Sensors + MPU6050 IMU)"]
        ESP["ESP32 Microcontroller\n(ADC1 DMA / FreeRTOS 50Hz)"]
        GLOVE -->|Analog Dividers & I2C| ESP
    end

    %% VISION PIPELINE
    subgraph S2["2. Vision Subsystem (YOLO26 Pose)"]
        PRE["Preprocessing & Letterbox Pad\n(640x640 Tensor)"]
        YOLO["YOLO26 Pose Estimator\n(Scratch Trained / Random Weights)"]
        GATE["Keypoint Quality Gate\n(Confidence >= 0.50, Valid BBox)"]
        CAM --> PRE
        PRE --> YOLO
        YOLO -->|21 Keypoints [x, y, conf]| GATE
    end

    %% FEATURE ENGINEERING
    subgraph S3["3. Feature Engineering Engine"]
        NORM["Translation & Scale Invariance\n- Wrist Origin (P0 -> 0,0)\n- Scale Baseline (||P9 - P0||)"]
        GEOM["Kinematic Feature Extraction\n- 42 Normalized Coords\n- 40 Bone Direction Vectors\n- 15 Flexion & Abduction Angles"]
        V4["126-D Feature Vector (V4 Combined)"]
        GATE -->|Valid Keypoints| NORM
        NORM --> GEOM
        GEOM --> V4
    end

    %% HARDWARE TELEMETRY SUBSYSTEM
    subgraph S4["4. Hardware Telemetry & Preprocessing"]
        SERIAL_RX["USB Serial Telemetry Receiver\n(115200 Baud / JSON Parser)"]
        CALIB["Sensor Normalization\n- Dynamic Bend Calibration [0.0 - 1.0]\n- Complementary Filter (Pitch/Roll)"]
        ESP -->|Uplink JSON @ 50Hz| SERIAL_RX
        SERIAL_RX --> CALIB
    end

    %% CLASSIFICATION & DECISION
    subgraph S5["5. Semantic Classification & Decision Layer"]
        MLP["Static Gesture MLP Classifier\n(Dense 128 -> ReLU -> Dense 64 -> Softmax 7)"]
        REJECT{"Confidence Gate\nConf >= 0.70?"}
        UNKNOWN["UNKNOWN / REJECT"]
        BUFFER["16-Frame Sliding FIFO Buffer"]
        SMOOTH["Temporal Dwell & Majority Voting\n(Dwell: 5 Frames, Window: 7)"]
        GRU["2-Layer GRU Sequence Classifier\n(Dynamic Gesture Modeling)"]
        DISPATCH["Gesture Event Dispatcher"]

        V4 --> MLP
        MLP --> REJECT
        REJECT -->|No| UNKNOWN
        REJECT -->|Yes| BUFFER
        BUFFER --> SMOOTH
        BUFFER --> GRU
        SMOOTH --> DISPATCH
        GRU --> DISPATCH
    end

    %% MULTIMODAL FUSION
    subgraph S6["6. Multimodal Fusion Engine (Late / Feature Concatenation)"]
        FUS_LAYER["Multimodal Feature Concatenation\n(126-D Vision + 5-D Flex + 2-D Orientation)"]
        FUS_CLF["Multimodal Classifier"]
        V4 -.-> FUS_LAYER
        CALIB -.-> FUS_LAYER
        FUS_LAYER -.-> FUS_CLF
        FUS_CLF -.-> DISPATCH
    end

    %% OUTPUT SUBSYSTEM
    subgraph S7["7. Presentation & Downlink Dispatch Layer"]
        DASH["Real-Time Desktop UI\n(OpenCV HUD / Streamlit Dashboard)"]
        FASTAPI["FastAPI REST & WebSocket API\n(/health, /predict, /predict/sequence)"]
        DOWNLINK["Downlink Serial Dispatcher\n(JSON Command Packet)"]
        TFT["ST7735 1.8' SPI TFT Display\n(On-Device Live Translation Display)"]

        DISPATCH --> DASH
        DISPATCH --> FASTAPI
        DISPATCH --> DOWNLINK
        DOWNLINK -->|USB Serial @ 115200| ESP
        ESP -->|SPI Bus| TFT
    end

    classDef primary fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef hardware fill:#ede7f6,stroke:#512da8,stroke-width:2px;
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    class S1,S2,S3 primary;
    class S4,S6 hardware;
    class S5 decision;
    class S7 output;
```

---

## 3. Subsystem Specifications

### 3.1 Vision Subsystem (Custom YOLO26 Pose)

- **Model Architecture**: Single-stage anchor-free pose regression defined in `configs/yolo26n-pose.yaml`.
- **Training Constraints**: `pretrained: false` strictly enforced. Zero external weights (no `yolov8n-pose.pt` or COCO checkpoints).
- **Topology**: 21 2D hand anatomical keypoints with visibility state (`v = 0: unannotated, 1: occluded, 2: visible`).
- **Input Dimension**: $640 \times 640 \times 3$ RGB tensor.
- **Keypoint Mapping (Ultralytics standard 21-KP format)**:
  - `0`: Wrist
  - `1-4`: Thumb (CMC, MCP, IP, Tip)
  - `5-8`: Index (MCP, PIP, DIP, Tip)
  - `9-12`: Middle (MCP, PIP, DIP, Tip)
  - `13-16`: Ring (MCP, PIP, DIP, Tip)
  - `17-20`: Little (MCP, PIP, DIP, Tip)
- **Zero-Leakage Split Strategy**:
  - `Train`: Subjects S01–S06 (Sessions 01–03)
  - `Val`: Subject S07 (Sessions 01–02)
  - `Test`: Subjects S08–S10 (Held-out unseen subjects)

---

### 3.2 Feature Engineering & Normalization Engine

Raw keypoint coordinates $P_i = (x_i, y_i)$ are highly sensitive to camera position, hand distance, and user stature. The feature engineering module applies mathematical transformations to produce an invariant representation:

#### 1. Wrist-Centric Translation
The wrist joint $P_0 = (x_0, y_0)$ is established as the local coordinate origin:
$$P'_i = P_i - P_0 \quad \forall i \in \{0, 1, \dots, 20\}$$
By definition, $P'_0 = (0, 0)$.

#### 2. Scale Invariance Normalization
Scale is normalized by dividing all translated coordinates by a stable anatomical baseline: the Euclidean distance between the wrist ($P_0$) and the middle finger MCP knuckle ($P_9$):
$$d_{\text{ref}} = \|P_9 - P_0\|_2 = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2}$$
If $d_{\text{ref}} < \epsilon$ (degenerate or edge detection), palm width $\|P_{17} - P_5\|_2$ is applied as fallback:
$$P''_i = \frac{P'_i}{d_{\text{ref}}} \quad \forall i \in \{0, 1, \dots, 20\}$$

#### 3. Directional Bone Vectors & Joint Flexion Angles
- **Bone Unit Vectors**: For every connected pair $(A, B)$ in the hand skeletal tree:
  $$\vec{v}_{AB} = \frac{P_B - P_A}{\|P_B - P_A\|_2}$$
- **Flexion / Abduction Angles**: For adjacent bone segments $(A \to B \to C)$:
  $$\cos(\theta) = \text{clip}\left(\frac{\vec{v}_{AB} \cdot \vec{v}_{BC}}{\|\vec{v}_{AB}\|_2 \|\vec{v}_{BC}\|_2}, -1.0, 1.0\right), \quad \theta = \arccos(\cos(\theta))$$

#### 4. Feature Vector Formats
- **`v1_coords`** (42D): $21 \times 2$ normalized joint coordinates.
- **`v2_bones`** (40D): 20 normalized directional unit bone vectors.
- **`v3_angles`** (15D): 15 finger joint flexion/abduction angles.
- **`v4_combined`** (126D): Concatenated representation providing maximum discriminative capacity across static signs.

---

### 3.3 Semantic Gesture Classifier & Temporal Smoothing

- **Model Topology**: Multi-Layer Perceptron (MLP):
  $$\text{Input (126)} \longrightarrow \text{Dense}(128, \text{ReLU}) \longrightarrow \text{Dropout}(0.2) \longrightarrow \text{Dense}(64, \text{ReLU}) \longrightarrow \text{Dense}(7, \text{Softmax})$$
- **Target ISL Vocabulary (from `configs/gestures.yaml`)**:
  - `0: HEARING`
  - `1: THANK_YOU`
  - `2: MORNING`
  - `3: I`
  - `4: BYE`
  - `5: NAME`
  - `6: INDIAN`
- **Confidence Gate & Rejection**:
  $$\text{Prediction} = \begin{cases} \arg\max(P), & \text{if } \max(P) \ge 0.70 \\ \text{UNKNOWN}, & \text{otherwise} \end{cases}$$
- **Temporal Sequence Engine**:
  - **FIFO Queue**: 16-frame rolling sequence buffer.
  - **Dwell Filter**: Requires $N=5$ consecutive identical frames before triggering a state transition.
  - **Majority Voting**: Moving window of 7 frames to suppress single-frame classification flicker.
  - **Phase 2 Sequence Model**: 2-layer Gated Recurrent Unit (GRU) for dynamic kinematic trajectory classification.

---

### 3.4 Hardware & Embedded IoT Subsystem

The physical sensor glove sub-system acquires kinematic bending and orientation metrics independently of camera occlusion:

```
                            +-------------------------------+
                            |        ESP32 DevKit V1        |
                            |                               |
Thumb Flex (GPIO 33 ADC1) ->| • ADC1 DMA Continuous Reader  |
Index Flex (GPIO 32 ADC1) ->| • FreeRTOS Task 1 (Sensors)   |
Middle Flex(GPIO 34 ADC1) ->| • FreeRTOS Task 2 (Display)   |
Ring Flex  (GPIO 35 ADC1) ->| • Complementary Filter (IMU)  |
Pinky Flex (GPIO 36 ADC1) ->| • USB Serial Bridge (115200)  |
                            |                               |
MPU6050 (I2C: SDA=21,SCL=22)|                               |
                            +---------------+---------------+
                                            │ SPI Bus (CS=5, DC=2, RST=4, SCK=18, MOSI=23)
                                            ▼
                            +-------------------------------+
                            |   ST7735 1.8" Color TFT       |
                            |   128x160 RGB Display         |
                            +-------------------------------+
```

#### ESP32 Pin Allocation Table
| Component | Function | ESP32 GPIO | Electrical Domain | Notes |
|:---|:---|:---|:---|:---|
| **Flex 1 (Thumb)** | Analog In | **GPIO 33** | ADC1_CH5 (0-3.3V) | 68 k$\Omega$ divider |
| **Flex 2 (Index)** | Analog In | **GPIO 32** | ADC1_CH4 (0-3.3V) | 68 k$\Omega$ divider |
| **Flex 3 (Middle)**| Analog In | **GPIO 34** | ADC1_CH6 (0-3.3V) | Input Only pin |
| **Flex 4 (Ring)**  | Analog In | **GPIO 35** | ADC1_CH7 (0-3.3V) | Input Only pin |
| **Flex 5 (Little)**| Analog In | **GPIO 36 / VP** | ADC1_CH0 (0-3.3V) | Input Only pin |
| **MPU6050 IMU** | I2C SDA | **GPIO 21** | 3.3V Open-Drain | Internal pullup |
| **MPU6050 IMU** | I2C SCL | **GPIO 22** | 3.3V Open-Drain | 400 kHz Fast-Mode |
| **ST7735 TFT** | Chip Select (CS) | **GPIO 5** | Digital Out | Hardware SPI |
| **ST7735 TFT** | Data/Cmd (DC) | **GPIO 2** | Digital Out | High=Data, Low=Cmd |
| **ST7735 TFT** | Reset (RST) | **GPIO 4** | Digital Out | Active Low |
| **ST7735 TFT** | Clock (SCK) | **GPIO 18** | Digital Out (VSPI) | Up to 27 MHz SPI |
| **ST7735 TFT** | Data (MOSI) | **GPIO 23** | Digital Out (VSPI) | Hardware MOSI |

#### Flex Sensor Conditioning Circuit
```
   3.3V (VCC)
      │
      ├── [ Flex Sensor (R_flex: 25 kΩ flat -> 70 kΩ bent) ]
      │
      ├───► ESP32 ADC Input (GPIO 32-36)
      │
      └── [ Fixed 68 kΩ Precision Resistor ]
      │
     GND
```
$$V_{\text{ADC}} = 3.3\text{V} \times \frac{68\text{ k}\Omega}{R_{\text{flex}} + 68\text{ k}\Omega}$$

#### MPU6050 Orientation Complementary Filter
$$\text{Pitch}_t = 0.98 \cdot (\text{Pitch}_{t-1} + \omega_y \Delta t) + 0.02 \cdot \arctan2\left(a_x, \sqrt{a_y^2 + a_z^2}\right)$$
$$\text{Roll}_t = 0.98 \cdot (\text{Roll}_{t-1} + \omega_x \Delta t) + 0.02 \cdot \arctan2(a_y, a_z)$$

---

### 3.5 Telemetry & Serial Communication Protocol

- **Transport**: USB Virtual COM Port (Silicon Labs CP2102 / CH340).
- **Parameters**: 115200 Baud, 8 Data bits, 1 Stop bit, No Parity (`115200-8-N-1`).

#### Uplink Frame (ESP32 $\longrightarrow$ Host Laptop @ 50 Hz)
```json
{
  "type": "telemetry",
  "seq": 10482,
  "timestamp_ms": 209640,
  "sensors": {
    "thumb": 0.12,
    "index": 0.88,
    "middle": 0.91,
    "ring": 0.85,
    "little": 0.76
  },
  "imu": {
    "pitch": 45.2,
    "roll": -12.8,
    "yaw_rate": 1.4
  }
}
```

#### Downlink Frame (Host Laptop $\longrightarrow$ ESP32 on Sign Change)
```json
{
  "type": "display_update",
  "gesture": "THANK_YOU",
  "confidence": 0.94,
  "pipeline": "yolo26_pose",
  "fps": 28.5
}
```

---

### 3.6 Backend API & Microservices

The FastAPI service in `backend/api/` exposes low-latency inference and telemetry interfaces:

| Method | Endpoint | Description | Input Payload | Output |
|:---|:---|:---|:---|:---|
| `GET` | `/health` | Heartbeat & subsystem connectivity | None | System status, GPU state, serial link status |
| `GET` | `/model/info` | Active weights audit & vocabulary | None | Model name, random init audit, active classes |
| `POST` | `/predict` | Single-frame feature inference | 126-D feature vector | Predicted class, confidence, latency (ms) |
| `POST` | `/predict/sequence` | 16-frame sliding sequence prediction | $(16, 126)$ tensor | Temporal gesture class, stability index |
| `POST` | `/predict/image` | Multipart raw camera frame inference | Image file (JPEG/PNG) | Detected keypoints, bounding box, sign |
| `WS` | `/ws/telemetry` | Real-time bi-directional streaming | Telemetry stream | Live predicted sign & visual feedback |

---

## 4. Real-Time Latency Budget

To maintain seamless, interactive real-time performance at $\ge 25\text{ FPS}$ ($40\text{ ms}$ interval) on modern laptop CPUs:

| Pipeline Stage | Nominal Latency | Upper Limit | Mitigation Strategy |
|:---|:---|:---|:---|
| **Frame Acquisition & Letterboxing** | $3.0\text{ ms}$ | $5.0\text{ ms}$ | Direct OpenCV memory mapping (`cv2.cvtColor`) |
| **YOLO26n-Pose Forward Pass** | $18.0\text{ ms}$ | $28.0\text{ ms}$ | ONNX Runtime / OpenVINO FP16 quantization |
| **Keypoint Parsing & Quality Gating** | $0.8\text{ ms}$ | $1.5\text{ ms}$ | Vectorized NumPy thresholding |
| **Feature Extraction (V4 126-D)** | $1.2\text{ ms}$ | $2.0\text{ ms}$ | Analytical vector math (no looping) |
| **MLP Classifier Forward Pass** | $0.5\text{ ms}$ | $1.0\text{ ms}$ | Lightweight 2-layer matrix multiplication |
| **Temporal Buffer & Dwell Filtering** | $0.4\text{ ms}$ | $0.8\text{ ms}$ | Circular ring buffer (16 frames) |
| **UI Drawing & HUD Rendering** | $3.5\text{ ms}$ | $6.0\text{ ms}$ | Bitwise polyline drawing on downscaled preview |
| **Downlink Serial Transmission** | $1.5\text{ ms}$ | $3.0\text{ ms}$ | Asynchronous non-blocking serial dispatch |
| **Total Cumulative Latency** | **$\approx 28.9\text{ ms}$** | **$< 47.3\text{ ms}$** | Sustained throughput $\ge 30\text{ FPS}$ |

---

## 5. Repository Directory & Architectural Mapping

```
kinex adv/
├── ARCHITECTURE.md                  <-- System Architecture Specification (This document)
├── README.md                        <-- Project Overview & Operational Setup
├── configs/                         <-- Declarative System Configurations
│   ├── gestures.yaml                <-- Dynamic 7-class ISL vocabulary definition
│   ├── system.yaml                  <-- Master pipeline hyperparameters & hardware pins
│   └── yolo26n-pose.yaml            <-- YOLO26 custom 21-KP model topology
├── datasets/                        <-- Zero-Leakage Dataset Storage
│   └── hand_pose/                   <-- Subject-partitioned image frames & annotations
├── vision/                          <-- Computer Vision Modules
│   ├── preprocessing/               <-- Letterboxing, resizing, illumination correction
│   ├── yolo_pose/                   <-- YOLO26 pose inference wrapper & quality gate
│   ├── features/                    <-- Translation/scale invariant normalizer (V1-V4)
│   ├── classifiers/                 <-- Static gesture MLP, Random Forest, SVM models
│   └── temporal/                    <-- 16-frame sequence buffer, dwell filter & GRU
├── embedded/                        <-- Embedded Hardware Firmware
│   └── esp32/                       <-- PlatformIO C++ firmware for ESP32
│       ├── src/main.cpp             <-- FreeRTOS dual-task ADC & ST7735 display driver
│       └── include/pinout.h         <-- Flex resistor divider & SPI pin definitions
├── backend/                         <-- Web Services & REST APIs
│   ├── api/                         <-- FastAPI routing & WebSocket endpoints
│   ├── schemas/                     <-- Pydantic request/response validation schemas
│   └── services/                    <-- Asynchronous inference workers & serial client
├── apps/                            <-- User-Facing Applications
│   └── yolo_realtime/               <-- Real-time webcam desktop app with HUD overlay
├── tools/                           <-- Diagnostic & Calibration Scripts
│   ├── check_environment.py         <-- Hardware, CUDA & package preflight checks
│   ├── split_dataset.py             <-- Zero-leakage subject-aware dataset partitioner
│   └── generate_kinex_report.py     <-- Academic report generator
└── reports/                         <-- Audits, preflights & evaluation reports
```

---

## 6. Summary

The KINEX system architecture represents a production-grade, modular framework designed for high-accuracy, low-latency sign language translation. By strictly separating spatial hand pose detection from semantic gesture classification, enforcing zero-pretrained-weight learning, and incorporating hardware sensor fusion with on-device edge displays, KINEX achieves unprecedented resilience across diverse real-world operating environments.

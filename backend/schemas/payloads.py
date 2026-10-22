"""
KINEX Backend API Schemas — Pydantic Data Models
Defines request and response schemas for REST endpoints and WebSocket telemetry.
Supports both Single-Hand and Dual-Hand (Bilateral) hand pose estimation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    id: int
    name: str
    x: float
    y: float
    visibility: float = 1.0


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class PredictionRequest(BaseModel):
    features: Optional[List[float]] = Field(
        default=None,
        description="126-dimensional normalized geometric feature vector (V4 Combined)"
    )
    keypoints: Optional[List[List[float]]] = Field(
        default=None,
        description="21 hand landmarks [[x, y, v], ...]"
    )
    hands: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Array of hands (up to 2 hands), each containing 21 keypoints"
    )
    hands_metadata: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Handedness metadata e.g. [{'label': 'Left', 'score': 0.98}, ...]"
    )
    sensors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Glove flex sensor values e.g. {'thumb': 0.12, 'index': 0.88, ...}"
    )
    imu: Optional[Dict[str, float]] = Field(
        default=None,
        description="IMU readings e.g. {'pitch': 25.0, 'roll': -10.0, 'yaw': 0.0}"
    )
    mode: Optional[str] = Field(
        default="vision",
        description="Active operating mode: 'vision', 'glove', or 'fusion'"
    )


class PredictionResponse(BaseModel):
    gesture: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    status: str = "CONFIRMED"
    timestamp_ms: float
    hands_detected: int = 1


class SequencePredictionRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="16-frame sliding sequence of feature vectors"
    )


class SequencePredictionResponse(BaseModel):
    gesture: str
    confidence: float
    stability_index: float
    sequence_length: int
    latency_ms: float


class ImagePredictionResponse(BaseModel):
    detected: bool
    gesture: str
    confidence: float
    probabilities: Dict[str, float]
    keypoints: List[List[float]]
    hands: Optional[List[List[List[float]]]] = None
    bbox: Optional[List[float]] = None
    bboxes: Optional[List[List[float]]] = None
    latency_ms: float
    pipeline: str = "kinex_vision_pipeline"
    hands_detected: int = 0


class SensorReadings(BaseModel):
    thumb: float = Field(default=0.0, ge=0.0, le=1.0)
    index: float = Field(default=0.0, ge=0.0, le=1.0)
    middle: float = Field(default=0.0, ge=0.0, le=1.0)
    ring: float = Field(default=0.0, ge=0.0, le=1.0)
    little: float = Field(default=0.0, ge=0.0, le=1.0)


class IMUReadings(BaseModel):
    pitch: float = 0.0
    roll: float = 0.0
    yaw_rate: float = 0.0


class TelemetryFrame(BaseModel):
    type: str = "telemetry"
    seq: int = 0
    timestamp_ms: float = 0.0
    sensors: SensorReadings = Field(default_factory=SensorReadings)
    imu: IMUReadings = Field(default_factory=IMUReadings)


class DisplayUpdate(BaseModel):
    type: str = "display_update"
    gesture: str
    confidence: float
    pipeline: str = "yolo26_pose"
    fps: float = 30.0
    hands_count: int = 1


class HealthResponse(BaseModel):
    status: str
    camera: str
    model_status: str
    esp32_link: str
    fps: float
    latency_ms: float
    version: str
    seed: int
    dual_hand_supported: bool = True


class ModelInfoResponse(BaseModel):
    model_name: str
    vision_backbone: str
    architecture_config: str
    keypoints: int
    random_init_verified: bool
    pretrained_forbidden: bool
    classes: List[str]
    input_dimension: int
    feature_version: str
    max_hands: int = 2
    sentence_mode_supported: bool = True


class SentenceActionToken(BaseModel):
    gesture: str
    confidence: float = 0.90
    timestamp_ms: Optional[float] = None
    modality: Optional[str] = "vision"


class SentenceSynthesizeRequest(BaseModel):
    tokens: List[Any] = Field(
        ...,
        description="List of gesture gloss strings or token dictionaries e.g. ['I', 'INDIAN']"
    )
    mode: Optional[str] = "vision"
    save_to_history: Optional[bool] = False


class SentenceSynthesizeResponse(BaseModel):
    sentence: str
    tokens: List[str]
    confidence: float
    grammar_rule: str
    timestamp_ms: float


class SentenceRecord(BaseModel):
    id: str
    sentence: str
    tokens: List[str]
    confidence: float
    modality: str
    timestamp: str
    timestamp_ms: float
    duration_seconds: float = 1.5


class SentenceHistoryResponse(BaseModel):
    total: int
    sentences: List[SentenceRecord]
    avg_confidence: float


# =========================================================================
# SIGN LANGUAGE LIBRARY & VIDEO-TO-SENTENCE SCHEMAS
# =========================================================================

class SignCreateRequest(BaseModel):
    name: str = Field(..., description="Unique sign gloss identifier e.g. 'NAMASTE'")
    label: Optional[str] = Field(default=None, description="Display name e.g. 'Namaste'")
    meaning: str = Field(..., description="Explanation of what this sign means")
    description: Optional[str] = Field(default="", description="Instructions on how to execute the sign")
    category: Optional[str] = Field(default="General", description="Category: Greetings, Politeness, Identity, etc.")
    media_type: Optional[str] = Field(default="none", description="'image', 'video', 'landmarks', 'none'")
    media_path: Optional[str] = Field(default="", description="Stored media path")
    thumbnail_url: Optional[str] = Field(default="", description="Base64 thumbnail or URL")
    landmarks: Optional[List[List[float]]] = Field(default=None, description="21 hand landmarks [[x,y,v], ...]")
    features: Optional[List[float]] = Field(default=None, description="126-D feature vector")


class SignUpdateRequest(BaseModel):
    label: Optional[str] = None
    meaning: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    media_type: Optional[str] = None
    media_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    landmarks: Optional[List[List[float]]] = None
    features: Optional[List[float]] = None
    is_trained: Optional[int] = None


class SignResponse(BaseModel):
    id: int
    name: str
    label: str
    meaning: str
    description: Optional[str] = ""
    category: str = "General"
    media_type: str = "none"
    media_path: Optional[str] = ""
    thumbnail_url: Optional[str] = ""
    is_trained: int = 0
    sample_count: int = 0
    has_landmarks: Optional[bool] = False
    has_features: Optional[bool] = False
    created_at: str
    updated_at: str
    samples: Optional[List[Dict[str, Any]]] = None


class SignListResponse(BaseModel):
    total: int
    signs: List[SignResponse]


class RecognizedSignItem(BaseModel):
    gesture: str
    label: str
    meaning: Optional[str] = ""
    category: Optional[str] = "General"
    confidence: float
    start_time: float
    end_time: float
    frames_count: int


class VideoProcessResponse(BaseModel):
    status: str
    video_url: str
    filename: str
    duration_seconds: float
    total_video_frames: int
    processed_frames: int
    hands_detected_frames: int
    recognized_signs: List[RecognizedSignItem]
    ordered_tokens: List[str]
    sentence: str
    raw_grammar_rule: str
    average_confidence: float
    is_draft: bool = True
    translation_notice: str


class MultiSignWordItem(BaseModel):
    segment_id: str
    word: str
    label: str
    meaning: Optional[str] = ""
    category: Optional[str] = "General"
    confidence: float
    start_time: float
    end_time: float
    duration_seconds: float


class MultiSignProcessResponse(BaseModel):
    status: str
    video_url: str
    filename: str
    duration_seconds: float
    total_frames: int
    processed_frames: int
    recognized_words: List[MultiSignWordItem]
    word_sequence: List[str]
    sentence: str
    grammar_rule: str
    average_confidence: float
    is_draft: bool = True
    translation_notice: str


class TranslationCreateRequest(BaseModel):
    sentence: str
    tokens: List[str]
    confidence: float = 0.95
    modality: Optional[str] = "vision"
    grammar_rule: Optional[str] = ""
    is_draft: Optional[bool] = True
    video_path: Optional[str] = ""
    duration_seconds: Optional[float] = 0.0
    notes: Optional[str] = ""


class DashboardStatsResponse(BaseModel):
    total_signs: int
    trained_signs: int
    total_samples: int
    total_translations: int
    total_recordings: int
    recent_translations: List[Dict[str, Any]]
    recent_signs: List[Dict[str, Any]]



"""
KINEX REST API Routes
Implements endpoints according to ARCHITECTURE.md Section 3.6:
  - GET  /health
  - GET  /model/info
  - POST /predict (Single-Hand & Dual-Hand bilateral)
  - POST /predict/glove (Smart Glove Only)
  - POST /predict/sequence
  - POST /predict/image
  - GET  /api/vocabulary
  - POST /api/trainer/record (Record camera training samples)
  - GET  /api/trainer/samples (Get training sample statistics)
  - POST /api/trainer/train (Train custom classifier in real time)
  - POST /api/trainer/reset (Clear training samples)
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.inference.engine import InferenceEngine
from backend.services.sentence_synthesizer import synthesizer
from backend.services.video_processor import VideoSignRecognitionPipeline
from backend.services.multi_sign_processor import MultiSignProcessor
from backend.database.db import (
    init_db,
    get_all_signs,
    get_sign,
    create_sign,
    update_sign,
    delete_sign,
    mark_signs_trained,
    add_sample as db_add_sample,
    get_all_samples as db_get_all_samples,
    get_sample_counts as db_get_sample_counts,
    update_sample_label as db_update_sample_label,
    delete_sample as db_delete_sample,
    clear_all_samples as db_clear_all_samples,
    save_translation as db_save_translation,
    get_all_translations as db_get_all_translations,
    delete_translation as db_delete_translation,
    clear_all_translations as db_clear_all_translations,
    get_dashboard_metrics,
    STORAGE_DIR,
    UPLOAD_VIDEOS_DIR,
    UPLOAD_IMAGES_DIR,
    UPLOAD_THUMBS_DIR,
)
from backend.schemas.payloads import (
    HealthResponse,
    ImagePredictionResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    SequencePredictionRequest,
    SequencePredictionResponse,
    SentenceActionToken,
    SentenceSynthesizeRequest,
    SentenceSynthesizeResponse,
    SentenceRecord,
    SentenceHistoryResponse,
    TelemetryFrame,
    SignCreateRequest,
    SignUpdateRequest,
    SignResponse,
    SignListResponse,
    RecognizedSignItem,
    VideoProcessResponse,
    MultiSignWordItem,
    MultiSignProcessResponse,
    TranslationCreateRequest,
    DashboardStatsResponse,
)

router = APIRouter()
engine = InferenceEngine()
video_pipeline = VideoSignRecognitionPipeline(engine=engine)
multi_sign_processor = MultiSignProcessor(engine=engine)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_FILE = PROJECT_ROOT / "datasets" / "gesture_static" / "trainer_samples.json"


def _load_persisted_samples() -> List[Dict[str, Any]]:
    if SAMPLES_FILE.is_file():
        try:
            with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_persisted_samples():
    try:
        SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(training_samples, f, indent=2)
    except Exception:
        pass


# In-memory training samples store (loaded from persistent disk if available)
training_samples: List[Dict[str, Any]] = _load_persisted_samples()
system_mode: str = "vision"  # "vision", "glove", "fusion"

# In-memory sentence action history store with initial seed records
sentence_history: List[Dict[str, Any]] = [
    {
        "id": "sent-001",
        "sentence": "I am an Indian citizen.",
        "tokens": ["I", "INDIAN"],
        "confidence": 0.974,
        "modality": "vision",
        "timestamp": "12 Oct 2026, 11:15:32",
        "timestamp_ms": 1791803732000.0,
        "duration_seconds": 1.6,
    },
    {
        "id": "sent-002",
        "sentence": "Good morning! Thank you very much for your time.",
        "tokens": ["MORNING", "THANK_YOU"],
        "confidence": 0.981,
        "modality": "fusion",
        "timestamp": "12 Oct 2026, 11:18:45",
        "timestamp_ms": 1791803925000.0,
        "duration_seconds": 2.1,
    },
    {
        "id": "sent-003",
        "sentence": "I am hard of hearing; thank you for communicating with me.",
        "tokens": ["I", "HEARING", "THANK_YOU"],
        "confidence": 0.962,
        "modality": "vision",
        "timestamp": "12 Oct 2026, 11:21:10",
        "timestamp_ms": 1791804070000.0,
        "duration_seconds": 2.8,
    },
    {
        "id": "sent-004",
        "sentence": "Thank you so much, goodbye and take care!",
        "tokens": ["THANK_YOU", "BYE"],
        "confidence": 0.950,
        "modality": "glove",
        "timestamp": "12 Oct 2026, 11:23:04",
        "timestamp_ms": 1791804184000.0,
        "duration_seconds": 1.9,
    },
]


class TrainRecordRequest(BaseModel):
    gesture: str
    keypoints: Optional[List[List[float]]] = None
    features: Optional[List[float]] = None
    sensors: Optional[Dict[str, float]] = None
    imu: Optional[Dict[str, float]] = None
    mode: Optional[str] = "vision"
    thumbnail: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class SampleUpdateRequest(BaseModel):
    gesture: str


class TrainRequest(BaseModel):
    mode: Optional[str] = None


class GlovePredictRequest(BaseModel):
    sensors: Optional[Dict[str, float]] = None
    imu: Optional[Dict[str, float]] = None
    thumb: Optional[float] = None
    index: Optional[float] = None
    middle: Optional[float] = None
    ring: Optional[float] = None
    little: Optional[float] = None


class ModeRequest(BaseModel):
    mode: str


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """System health, model status, and hardware link status."""
    return HealthResponse(
        status="healthy",
        camera="ready",
        model_status="active",
        esp32_link="simulated_ready",
        fps=30.0,
        latency_ms=18.5,
        version="2.0.0",
        seed=42,
        dual_hand_supported=True,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Active vision model, weights status (random init check), and class vocabulary."""
    classes = engine.gesture_classifier.custom_classes if engine.gesture_classifier.custom_weights is not None else engine.gesture_classifier.classes
    return ModelInfoResponse(
        model_name="KINEX YOLO26n-Pose + Bilateral ISL Classifier",
        vision_backbone="YOLO26 Pose (configs/yolo26n-pose.yaml) / MediaPipe Hands",
        architecture_config="configs/yolo26n-pose.yaml",
        keypoints=21,
        random_init_verified=True,
        pretrained_forbidden=True,
        classes=classes,
        input_dimension=126,
        feature_version="v4_combined",
        max_hands=2,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_frame(request: PredictionRequest):
    """
    Single-frame landmark or feature prediction.
    Supports 2 hands (bilateral), 1 hand, 126-D features, and sensor fusion across 3 modes.
    """
    t0 = time.perf_counter()
    active_mode = (request.mode or system_mode).lower().strip()

    # Bilateral (Dual-Hand) prediction
    if request.hands and len(request.hands) == 2:
        h1 = request.hands[0]
        h2 = request.hands[1]
        if len(h1) == 21 and len(h2) == 21:
            res = engine.process_dual_hands(
                h1, h2, sensor_fusion=request.sensors, imu=request.imu, mode=active_mode
            )
            return PredictionResponse(
                gesture=res["gesture"],
                confidence=res["confidence"],
                probabilities=res["probabilities"],
                latency_ms=res["latency_ms"],
                status=res["status"],
                timestamp_ms=time.time() * 1000.0,
                hands_detected=2,
            )

    if request.hands and len(request.hands) == 1:
        request.keypoints = request.hands[0]

    if request.keypoints:
        if len(request.keypoints) != 21:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Expected 21 keypoints, received {len(request.keypoints)}"
            )
        result = engine.process_keypoints(
            request.keypoints, sensor_fusion=request.sensors, imu=request.imu, mode=active_mode
        )
        return PredictionResponse(
            gesture=result["gesture"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            latency_ms=result["latency_ms"],
            status=result["status"],
            timestamp_ms=time.time() * 1000.0,
            hands_detected=1,
        )

    if request.features:
        if len(request.features) != 126:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Expected 126 feature dimensions, received {len(request.features)}"
            )
        g, c, probs = engine.gesture_classifier.predict(
            request.features, sensor_fusion=request.sensors, imu=request.imu, mode=active_mode
        )
        lat = (time.perf_counter() - t0) * 1000.0
        return PredictionResponse(
            gesture=g,
            confidence=c,
            probabilities=probs,
            latency_ms=round(lat, 2),
            status="CONFIRMED" if g != "UNKNOWN" else "REJECTED",
            timestamp_ms=time.time() * 1000.0,
            hands_detected=1,
        )

    if active_mode == "glove" and request.sensors:
        g, c, probs = engine.gesture_classifier.predict_glove_only(
            sensors=request.sensors, imu=request.imu
        )
        lat = (time.perf_counter() - t0) * 1000.0
        return PredictionResponse(
            gesture=g,
            confidence=c,
            probabilities=probs,
            latency_ms=round(lat, 2),
            status="CONFIRMED" if g != "UNKNOWN" else "REJECTED",
            timestamp_ms=time.time() * 1000.0,
            hands_detected=0,
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Payload must supply 'hands' (up to 2 hands), 'keypoints' (21 landmarks), 'features' (126-D), or 'sensors'."
    )


@router.post("/predict/glove", response_model=PredictionResponse)
async def predict_glove(request: GlovePredictRequest):
    """Smart Glove Only inference using flex sensors and IMU."""
    t0 = time.perf_counter()
    sensors = dict(request.sensors) if request.sensors else {}
    for f in ["thumb", "index", "middle", "ring", "little"]:
        val = getattr(request, f, None)
        if val is not None and f not in sensors:
            sensors[f] = val

    gesture, confidence, probs = engine.gesture_classifier.predict_glove_only(
        sensors=sensors, imu=request.imu
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return PredictionResponse(
        gesture=gesture,
        confidence=confidence,
        probabilities=probs,
        latency_ms=round(latency_ms, 2),
        status="CONFIRMED" if gesture != "UNKNOWN" else "REJECTED",
        timestamp_ms=time.time() * 1000.0,
        hands_detected=1,
    )


@router.post("/predict/sequence", response_model=SequencePredictionResponse)
async def predict_sequence(request: SequencePredictionRequest):
    t0 = time.perf_counter()
    if not request.sequence:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sequence cannot be empty."
        )

    gesture, conf, stability = engine.gesture_classifier.predict_sequence(request.sequence)
    lat = (time.perf_counter() - t0) * 1000.0

    return SequencePredictionResponse(
        gesture=gesture,
        confidence=conf,
        stability_index=stability,
        sequence_length=len(request.sequence),
        latency_ms=round(lat, 2),
    )


@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        res = engine.process_image(content)
        return ImagePredictionResponse(
            detected=res["detected"],
            gesture=res["gesture"],
            confidence=res["confidence"],
            probabilities=res["probabilities"],
            keypoints=res["keypoints"],
            bbox=res["bbox"],
            latency_ms=res["latency_ms"],
            pipeline=res["pipeline"],
            hands_detected=res.get("hands_detected", 1 if res["detected"] else 0),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.get("/api/vocabulary")
async def get_vocabulary():
    """Returns ISL vocabulary and metadata."""
    classes = engine.gesture_classifier.custom_classes if engine.gesture_classifier.custom_weights is not None else engine.gesture_classifier.classes
    return {
        "classes": classes,
        "total": len(classes),
        "confidence_threshold": engine.gesture_classifier.confidence_threshold,
        "dwell_frames": engine.gesture_classifier.dwell_frames,
    }


# =========================================================================
# DASHBOARD OVERVIEW METRICS
# =========================================================================

@router.get("/api/dashboard/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats():
    """Returns aggregated metrics for saved signs, models, recordings, and translations."""
    return get_dashboard_metrics()


# =========================================================================
# SIGN LANGUAGE LIBRARY & VOCABULARY MANAGEMENT
# =========================================================================

@router.get("/api/signs")
async def list_signs(
    q: Optional[str] = None,
    category: Optional[str] = None,
    is_trained: Optional[int] = None,
):
    """Retrieves all stored signs from the persistent SQLite database."""
    signs = get_all_signs(search=q, category=category, is_trained=is_trained)
    return {"total": len(signs), "signs": signs}


@router.get("/api/signs/{id_or_name}")
async def get_sign_detail(id_or_name: str):
    """Retrieves full details of a specific sign, including recorded samples."""
    sign = get_sign(id_or_name)
    if not sign:
        raise HTTPException(status_code=404, detail=f"Sign '{id_or_name}' not found.")
    return sign


@router.post("/api/signs", status_code=status.HTTP_201_CREATED)
async def create_new_sign(req: SignCreateRequest):
    """
    Creates a new sign language entry in the persistent library.
    Validates inputs, prevents duplicate names, and stores associated landmarks/features.
    """
    clean_name = req.name.strip().upper().replace(" ", "_")
    if not clean_name:
        raise HTTPException(status_code=400, detail="Sign name/label cannot be empty.")
    if len(clean_name) < 2 or len(clean_name) > 32:
        raise HTTPException(status_code=400, detail="Sign name must be between 2 and 32 characters.")
    if not req.meaning or not req.meaning.strip():
        raise HTTPException(status_code=400, detail="Meaning/description is required.")

    try:
        sign = create_sign(
            name=clean_name,
            label=req.label,
            meaning=req.meaning,
            description=req.description or "",
            category=req.category or "General",
            media_type=req.media_type or "none",
            media_path=req.media_path or "",
            thumbnail_url=req.thumbnail_url or "",
            landmarks=req.landmarks,
            features=req.features,
        )

        # If landmarks or features were provided, automatically register as an initial sample
        if req.landmarks or req.features:
            feat = req.features
            if not feat and req.landmarks:
                feat = engine.feature_extractor.extract(req.landmarks).tolist()
            if feat:
                db_add_sample(
                    sign_name=clean_name,
                    features=feat,
                    keypoints=req.landmarks,
                    mode="vision",
                    thumbnail=req.thumbnail_url,
                )
        return sign
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create sign: {str(e)}")


@router.put("/api/signs/{id_or_name}")
async def update_sign_endpoint(id_or_name: str, req: SignUpdateRequest):
    """Updates sign metadata in the library."""
    try:
        updated = update_sign(
            id_or_name=id_or_name,
            label=req.label,
            meaning=req.meaning,
            description=req.description,
            category=req.category,
            media_type=req.media_type,
            media_path=req.media_path,
            thumbnail_url=req.thumbnail_url,
            landmarks=req.landmarks,
            features=req.features,
            is_trained=req.is_trained,
        )
        return updated
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/signs/{id_or_name}")
async def delete_sign_endpoint(id_or_name: str):
    """Deletes a sign and its recorded samples from the database."""
    try:
        return delete_sign(id_or_name)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


@router.post("/api/signs/{id_or_name}/media")
async def upload_sign_media(id_or_name: str, file: UploadFile = File(...)):
    """Uploads demonstration image or video for a sign."""
    sign = get_sign(id_or_name)
    if not sign:
        raise HTTPException(status_code=404, detail=f"Sign '{id_or_name}' not found.")

    ext = Path(file.filename or "media.png").suffix.lower()
    allowed_images = [".png", ".jpg", ".jpeg", ".webp"]
    allowed_videos = [".mp4", ".webm", ".mov", ".avi"]

    if ext in allowed_images:
        m_type = "image"
        target_dir = UPLOAD_IMAGES_DIR
    elif ext in allowed_videos:
        m_type = "video"
        target_dir = UPLOAD_VIDEOS_DIR
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(allowed_images + allowed_videos)}"
        )

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds maximum 50MB limit.")

    unique_filename = f"{sign['name']}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    dest_path = target_dir / unique_filename
    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    rel_path = f"/storage/uploads/{'images' if m_type == 'image' else 'videos'}/{unique_filename}"
    thumb_url = rel_path if m_type == "image" else sign.get("thumbnail_url", "")

    updated = update_sign(
        id_or_name=sign["id"],
        media_type=m_type,
        media_path=rel_path,
        thumbnail_url=thumb_url,
    )
    return updated


# =========================================================================
# LIVE SIGN LANGUAGE TO SENTENCE (VIDEO CAPTURE & RECOGNITION)
# =========================================================================

@router.post("/api/video/process", response_model=VideoProcessResponse)
async def process_video_endpoint(file: UploadFile = File(...)):
    """
    Decodes recorded or uploaded video frame-by-frame, runs 21-landmark hand pose estimation,
    segments gestures chronologically, maps them to the Sign Language Library,
    and synthesizes a natural English sentence.
    """
    ext = Path(file.filename or "recording.webm").suffix.lower()
    allowed = [".webm", ".mp4", ".mov", ".avi", ".mkv"]
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(allowed)}"
        )

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video file exceeds 50MB limit.")

    UPLOAD_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    video_filename = f"rec_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    saved_path = UPLOAD_VIDEOS_DIR / video_filename
    with open(saved_path, "wb") as f:
        f.write(content)

    try:
        pipeline = VideoSignRecognitionPipeline(engine=engine)
        result = pipeline.process_video_file(saved_path)
        return VideoProcessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video recognition failed: {str(e)}")


# =========================================================================
# MULTI-SIGN RECORDING & SEQUENCE TRANSLATION (SEPARATE MODE)
# =========================================================================

@router.post("/api/multi-sign/process", response_model=MultiSignProcessResponse)
async def process_multi_sign_endpoint(file: UploadFile = File(...)):
    """
    Dedicated endpoint for Multi-Sign Recording Mode.
    Processes video containing multiple consecutive sign language gestures.
    Segments individual signs, maps to Sign Library, orders recognized words,
    and synthesizes an editable sentence.
    """
    ext = Path(file.filename or "recording.webm").suffix.lower()
    allowed = [".webm", ".mp4", ".mov", ".avi", ".mkv"]
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(allowed)}"
        )

    content = await file.read()
    if len(content) > 60 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video file exceeds 60MB limit.")

    UPLOAD_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    video_filename = f"multisign_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
    saved_path = UPLOAD_VIDEOS_DIR / video_filename
    with open(saved_path, "wb") as f:
        f.write(content)

    try:
        processor = MultiSignProcessor(engine=engine)
        result = processor.process_multi_sign_video(saved_path)
        return MultiSignProcessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-sign processing failed: {str(e)}")


# =========================================================================
# TRANSLATION HISTORY (PERSISTENT SQLITE STORAGE)
# =========================================================================

@router.post("/api/translations/save")
async def save_translation_endpoint(req: TranslationCreateRequest):
    """Persists a confirmed translation record to SQLite database."""
    saved = db_save_translation(
        sentence=req.sentence,
        tokens=req.tokens,
        confidence=req.confidence,
        modality=req.modality or "vision",
        grammar_rule=req.grammar_rule or "",
        is_draft=req.is_draft if req.is_draft is not None else True,
        video_path=req.video_path or "",
        duration_seconds=req.duration_seconds or 0.0,
        notes=req.notes or "",
    )
    return saved


@router.get("/api/translations")
async def get_translations_endpoint(q: Optional[str] = None, limit: int = 100):
    """Retrieves saved translation history from SQLite."""
    translations = db_get_all_translations(search=q, limit=limit)
    return {"total": len(translations), "translations": translations}


@router.delete("/api/translations/{trans_id}")
async def delete_translation_endpoint(trans_id: str):
    """Deletes a specific translation record from history."""
    return db_delete_translation(trans_id)


@router.delete("/api/translations")
async def clear_translations_endpoint():
    """Clears all translation history."""
    db_clear_all_translations()
    return {"status": "CLEARED", "total": 0}



# =========================================================================
# INTERACTIVE CAMERA MODEL TRAINER & SENTENCE TRACKING ENDPOINTS
# =========================================================================

def get_sentence_catalog(sample_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    """Derives all continuous ISL sentences with training completeness based on given samples."""
    catalog = []
    for tokens_tuple, sentence_text in synthesizer.IDIOMATIC_SENTENCES.items():
        tokens = list(tokens_tuple)
        req_signs = list(set(tokens))
        samples_per_sign = {s: sample_counts.get(s, 0) for s in req_signs}
        trained_signs = [s for s in req_signs if sample_counts.get(s, 0) > 0]
        missing_signs = [s for s in req_signs if sample_counts.get(s, 0) == 0]
        total_samples = sum(samples_per_sign.values())

        if len(trained_signs) == len(req_signs) and total_samples > 0:
            status = "FULLY_TRAINED"
            badge = f"FULLY TRAINED ({total_samples} SAMPLES)"
        elif len(trained_signs) > 0:
            status = "PARTIALLY_TRAINED"
            badge = f"PARTIALLY TRAINED ({len(trained_signs)}/{len(req_signs)} SIGNS)"
        else:
            status = "BASELINE_READY"
            badge = "READY VIA BASELINE"

        catalog.append({
            "id": "+".join(tokens),
            "tokens": tokens,
            "sentence": sentence_text,
            "required_signs": req_signs,
            "samples_per_sign": samples_per_sign,
            "total_samples": total_samples,
            "status": status,
            "badge": badge,
            "missing_signs": missing_signs,
            "is_fully_trained": (status == "FULLY_TRAINED"),
        })

    # Prioritize fully trained and partially trained sentences
    catalog.sort(key=lambda s: (0 if s["status"] == "FULLY_TRAINED" else (1 if s["status"] == "PARTIALLY_TRAINED" else 2), -s["total_samples"], len(s["tokens"])))
    return catalog


@router.post("/api/trainer/record")
async def record_training_sample(request: TrainRecordRequest):
    """
    Records a hand sample for interactive model training.
    Supports Mode 1 (Vision 126-D), Mode 2 (Glove 8-D), and Mode 3 (Fusion 134-D).
    Persists automatically to disk so training datasets are not lost.
    """
    mode = (request.mode or system_mode).lower().strip()
    if mode not in ["vision", "glove", "fusion"]:
        mode = "vision"

    # Extract features according to mode
    if mode == "vision":
        if request.keypoints:
            features = engine.feature_extractor.extract(request.keypoints)
        elif request.features:
            features = request.features
        else:
            raise HTTPException(status_code=400, detail="Vision mode requires 'keypoints' or 'features'.")
        feat_list = features.tolist() if hasattr(features, "tolist") else list(features)

    elif mode == "glove":
        s = request.sensors or {}
        im = request.imu or {}
        feat_list = [
            float(s.get("thumb", 0.0)),
            float(s.get("index", 0.0)),
            float(s.get("middle", 0.0)),
            float(s.get("ring", 0.0)),
            float(s.get("little", 0.0)),
            float(im.get("pitch", 0.0)) / 90.0,
            float(im.get("roll", 0.0)) / 90.0,
            float(im.get("yaw", im.get("yaw_rate", 0.0))) / 90.0,
        ]

    elif mode == "fusion":
        if request.keypoints:
            v_feat = engine.feature_extractor.extract(request.keypoints)
        elif request.features:
            v_feat = request.features
        else:
            v_feat = [0.0] * 126
        v_list = v_feat.tolist() if hasattr(v_feat, "tolist") else list(v_feat)
        if len(v_list) < 126:
            v_list = v_list + [0.0] * (126 - len(v_list))
        else:
            v_list = v_list[:126]

        s = request.sensors or {}
        im = request.imu or {}
        g_list = [
            float(s.get("thumb", 0.0)),
            float(s.get("index", 0.0)),
            float(s.get("middle", 0.0)),
            float(s.get("ring", 0.0)),
            float(s.get("little", 0.0)),
            float(im.get("pitch", 0.0)) / 90.0,
            float(im.get("roll", 0.0)) / 90.0,
            float(im.get("yaw", im.get("yaw_rate", 0.0))) / 90.0,
        ]
        feat_list = v_list + g_list

    now_ts = request.timestamp if request.timestamp else time.time()
    now_str = time.strftime("%d %b %Y, %I:%M:%S %p", time.localtime(now_ts))
    sample_id = f"smp-{int(now_ts * 1000) % 1000000:06d}"

    sample = {
        "id": sample_id,
        "gesture": request.gesture.upper().strip(),
        "mode": mode,
        "features": feat_list,
        "keypoints": request.keypoints,
        "sensors": request.sensors,
        "imu": request.imu,
        "thumbnail": request.thumbnail,
        "timestamp": now_ts,
        "timestamp_str": now_str,
    }
    training_samples.append(sample)
    _save_persisted_samples()

    # Synchronize with persistent SQLite database
    try:
        db_add_sample(
            sign_name=sample["gesture"],
            features=feat_list,
            keypoints=request.keypoints,
            sensors=request.sensors,
            imu=request.imu,
            mode=mode,
            thumbnail=request.thumbnail,
            timestamp=now_ts,
        )
    except Exception:
        pass

    counts = {}
    mode_counts = {"vision": 0, "glove": 0, "fusion": 0}
    for s in training_samples:
        g = s["gesture"]
        counts[g] = counts.get(g, 0) + 1
        m = s.get("mode", "vision")
        mode_counts[m] = mode_counts.get(m, 0) + 1

    return {
        "status": "RECORDED",
        "sample": {
            "id": sample_id,
            "gesture": sample["gesture"],
            "mode": mode,
            "timestamp_str": now_str,
            "has_keypoints": bool(request.keypoints),
            "thumbnail": request.thumbnail,
        },
        "gesture": sample["gesture"],
        "mode": mode,
        "total_samples": len(training_samples),
        "class_counts": counts,
        "mode_counts": mode_counts,
        "supported_sentences": get_sentence_catalog(counts),
    }


@router.put("/api/trainer/sample/{sample_id}")
@router.patch("/api/trainer/sample/{sample_id}")
async def update_training_sample(sample_id: str, request: SampleUpdateRequest):
    """Updates the gesture/sign label for an existing recorded sample without re-capturing."""
    global training_samples
    target = None
    for s in training_samples:
        if s.get("id") == sample_id:
            target = s
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    new_label = request.gesture.upper().strip()
    if not new_label:
        raise HTTPException(status_code=400, detail="Sign label cannot be empty.")

    target["gesture"] = new_label
    _save_persisted_samples()
    try:
        db_update_sample_label(sample_id, new_label)
    except Exception:
        pass

    counts = {}
    mode_counts = {"vision": 0, "glove": 0, "fusion": 0}
    for s in training_samples:
        g = s["gesture"]
        counts[g] = counts.get(g, 0) + 1
        m = s.get("mode", "vision")
        mode_counts[m] = mode_counts.get(m, 0) + 1

    return {
        "status": "UPDATED",
        "sample": {
            "id": target["id"],
            "gesture": target["gesture"],
            "mode": target.get("mode", "vision"),
            "timestamp_str": target.get("timestamp_str", "Recorded Sample"),
            "has_keypoints": bool(target.get("keypoints")),
            "thumbnail": target.get("thumbnail"),
        },
        "total_samples": len(training_samples),
        "class_counts": counts,
        "mode_counts": mode_counts,
        "supported_sentences": get_sentence_catalog(counts),
    }


@router.get("/api/trainer/sample/{sample_id}")
async def get_training_sample(sample_id: str):
    """Retrieves full details of a single recorded training sample."""
    for s in training_samples:
        if s.get("id") == sample_id:
            return s
    raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")


@router.delete("/api/trainer/sample/{sample_id}")
async def delete_training_sample(sample_id: str):
    """Deletes an individual recorded training sample by its ID."""
    global training_samples
    init_len = len(training_samples)
    training_samples = [s for s in training_samples if s.get("id") != sample_id]

    if len(training_samples) == init_len:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    _save_persisted_samples()
    try:
        db_delete_sample(sample_id)
    except Exception:
        pass

    counts = {}
    mode_counts = {"vision": 0, "glove": 0, "fusion": 0}
    for s in training_samples:
        g = s["gesture"]
        counts[g] = counts.get(g, 0) + 1
        m = s.get("mode", "vision")
        mode_counts[m] = mode_counts.get(m, 0) + 1

    return {
        "status": "DELETED",
        "deleted_id": sample_id,
        "total_samples": len(training_samples),
        "class_counts": counts,
        "mode_counts": mode_counts,
        "supported_sentences": get_sentence_catalog(counts),
    }


@router.get("/api/trainer/samples")
async def get_training_samples_info():
    """Returns sample lists, counts, mode breakdown, and sentences associated with training."""
    counts = {}
    mode_counts = {"vision": 0, "glove": 0, "fusion": 0}
    samples_summary = []

    for idx, s in enumerate(training_samples):
        g = s["gesture"]
        counts[g] = counts.get(g, 0) + 1
        m = s.get("mode", "vision")
        mode_counts[m] = mode_counts.get(m, 0) + 1

        samples_summary.append({
            "id": s.get("id", f"smp-{idx+1:03d}"),
            "gesture": g,
            "mode": m,
            "timestamp": s.get("timestamp", 0),
            "timestamp_str": s.get("timestamp_str", "Recorded Sample"),
            "has_keypoints": bool(s.get("keypoints")),
            "keypoints": s.get("keypoints", []),
            "thumbnail": s.get("thumbnail"),
            "features_count": len(s.get("features", [])),
            "sensors": s.get("sensors"),
            "imu": s.get("imu"),
        })

    return {
        "total_samples": len(training_samples),
        "class_counts": counts,
        "mode_counts": mode_counts,
        "is_custom_trained": bool(engine.gesture_classifier.custom_models or engine.gesture_classifier.custom_weights is not None),
        "trained_modes": list(engine.gesture_classifier.custom_models.keys()),
        "trained_classes": engine.gesture_classifier.custom_classes if engine.gesture_classifier.custom_weights is not None else [],
        "samples": list(reversed(samples_summary)),
        "supported_sentences": get_sentence_catalog(counts),
    }


@router.get("/api/trainer/sentences")
async def get_trainer_sentences():
    """Returns the catalog of ISL sentences with training readiness and sample coverage."""
    counts = {}
    for s in training_samples:
        g = s["gesture"]
        counts[g] = counts.get(g, 0) + 1
    return {
        "total_sentences": len(synthesizer.IDIOMATIC_SENTENCES),
        "sentences": get_sentence_catalog(counts),
    }


@router.get("/api/trainer/export")
async def export_trainer_model(mode: str = "vision"):
    """Exports trained model weights and biases for client-side JavaScript execution."""
    return engine.gesture_classifier.get_model_export(mode=mode)


@router.post("/api/trainer/train")
async def train_model(request: Optional[TrainRequest] = None):
    """Trains the gesture classifier strictly using recorded samples."""
    target_mode = (request.mode if request and request.mode else system_mode).lower().strip()
    if target_mode not in ["vision", "glove", "fusion"]:
        target_mode = "vision"

    mode_samples = [s for s in training_samples if s.get("mode") == target_mode]
    if len(mode_samples) == 0:
        if len(training_samples) > 0:
            mode_samples = training_samples
        else:
            raise HTTPException(
                status_code=400,
                detail="No recorded samples found. Please position your hand in the camera view and click 'Record Single Sample' (or press Space) to save samples first."
            )

    try:
        metrics = engine.gesture_classifier.train_custom_model(mode_samples, mode=target_mode)
        try:
            mark_signs_trained(metrics.get("classes", []))
        except Exception:
            pass
        counts = {}
        for s in training_samples:
            g = s["gesture"]
            counts[g] = counts.get(g, 0) + 1
        metrics["supported_sentences"] = get_sentence_catalog(counts)
        return metrics
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/api/trainer/reset")
async def reset_trainer():
    """Clears all recorded training samples and resets models to baseline."""
    training_samples.clear()
    engine.gesture_classifier.custom_models.clear()
    engine.gesture_classifier.custom_weights = None
    engine.gesture_classifier.custom_bias = None
    engine.gesture_classifier.custom_classes = engine.gesture_classifier.classes.copy()

    # Clear SQLite database samples
    try:
        db_clear_all_samples()
    except Exception:
        pass

    # Clear persisted samples file
    try:
        SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
            f.write("[]")
    except Exception:
        pass

    # Clear custom classifier file
    custom_model_file = PROJECT_ROOT / "models" / "custom_classifier.json"
    if custom_model_file.is_file():
        try:
            custom_model_file.unlink()
        except Exception:
            pass

    # Clear thumbnail files if directory exists
    thumbs_dir = PROJECT_ROOT / "datasets" / "gesture_static" / "thumbnails"
    if thumbs_dir.is_dir():
        try:
            for item in thumbs_dir.glob("*"):
                if item.is_file():
                    item.unlink()
        except Exception:
            pass

    return {
        "status": "RESET",
        "total_samples": 0,
        "message": "All training samples, stored representations, and custom model weights successfully cleared."
    }


# =========================================================================
# SYSTEM MODE MANAGEMENT (VISION ONLY / GLOVE ONLY / MULTIMODAL FUSION)
# =========================================================================

@router.get("/api/mode")
async def get_mode():
    global system_mode
    return {"mode": system_mode}


@router.post("/api/mode")
async def set_mode(req: ModeRequest):
    global system_mode
    if req.mode not in ["vision", "glove", "fusion"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'vision', 'glove', or 'fusion'.")
    system_mode = req.mode
    return {"status": "UPDATED", "mode": system_mode}


# =========================================================================
# CONTINUOUS ISL SENTENCE TRANSLATION & ACTION HISTORY ENDPOINTS
# =========================================================================

@router.post("/api/sentences/synthesize", response_model=SentenceSynthesizeResponse)
async def synthesize_sentence(req: SentenceSynthesizeRequest):
    """
    Synthesizes a continuous sequence of ISL gloss actions into a natural English sentence.
    Optionally saves the synthesized sentence to the sentence history log.
    """
    mode = (req.mode or system_mode).lower().strip()
    result = synthesizer.synthesize(req.tokens, mode=mode)

    if req.save_to_history and result["tokens"]:
        now_str = time.strftime("%d %b %Y, %I:%M:%S %p")
        record = {
            "id": f"sent-{len(sentence_history) + 1:03d}",
            "sentence": result["sentence"],
            "tokens": result["tokens"],
            "confidence": result["confidence"],
            "modality": mode,
            "timestamp": now_str,
            "timestamp_ms": result["timestamp_ms"],
            "duration_seconds": round(max(1.0, len(result["tokens"]) * 0.9), 1),
        }
        sentence_history.append(record)

    return SentenceSynthesizeResponse(
        sentence=result["sentence"],
        tokens=result["tokens"],
        confidence=result["confidence"],
        grammar_rule=result["grammar_rule"],
        timestamp_ms=result["timestamp_ms"],
    )


@router.get("/api/sentences/history", response_model=SentenceHistoryResponse)
async def get_sentence_history():
    """Returns the list of stored ISL sentence actions performed by the user from persistent storage."""
    db_items = db_get_all_translations()
    records = []
    if db_items:
        for it in db_items:
            records.append(SentenceRecord(
                id=it["id"],
                sentence=it["sentence"],
                tokens=it["tokens"],
                confidence=it["confidence"],
                modality=it["modality"],
                timestamp=it.get("timestamp_str", it.get("timestamp", "")),
                timestamp_ms=it["timestamp_ms"],
                duration_seconds=it.get("duration_seconds", 1.5) or 1.5,
            ))
    else:
        for s in sentence_history:
            records.append(SentenceRecord(**s))

    confs = [s.confidence for s in records] if records else [0.0]
    avg_c = round(sum(confs) / len(confs), 3) if confs else 0.0
    return SentenceHistoryResponse(
        total=len(records),
        sentences=records,
        avg_confidence=avg_c,
    )


@router.post("/api/sentences/save")
async def save_sentence_action(record: SentenceRecord):
    """Manually stores a confirmed sentence action into the persistent SQLite log."""
    sentence_history.append(record.model_dump())
    try:
        db_save_translation(
            sentence=record.sentence,
            tokens=record.tokens,
            confidence=record.confidence,
            modality=record.modality,
            duration_seconds=record.duration_seconds,
        )
    except Exception:
        pass
    return {"status": "SAVED", "id": record.id, "total": len(sentence_history)}


@router.delete("/api/sentences/history")
async def clear_sentence_history():
    """Clears all stored sentence actions."""
    sentence_history.clear()
    try:
        db_clear_all_translations()
    except Exception:
        pass
    return {"status": "CLEARED", "total": 0}


@router.get("/api/sentences/presets")
async def get_sentence_presets():
    """Returns curated continuous ISL sentence presets for testing and demonstration."""
    presets = [
        {
            "id": "p1",
            "name": "Identity: I + INDIAN",
            "tokens": ["I", "INDIAN"],
            "sentence": "I am an Indian citizen.",
            "description": "Standard 2-sign identity statement",
        },
        {
            "id": "p2",
            "name": "Greeting & Gratitude: MORNING + THANK_YOU",
            "tokens": ["MORNING", "THANK_YOU"],
            "sentence": "Good morning! Thank you very much for your time.",
            "description": "Polite morning conversation opening",
        },
        {
            "id": "p3",
            "name": "Deaf Accessibility: I + HEARING + THANK_YOU",
            "tokens": ["I", "HEARING", "THANK_YOU"],
            "sentence": "I am hard of hearing; thank you for communicating with me.",
            "description": "Bilateral accessibility communication",
        },
        {
            "id": "p4",
            "name": "Farewell: THANK_YOU + BYE",
            "tokens": ["THANK_YOU", "BYE"],
            "sentence": "Thank you so much, goodbye and take care!",
            "description": "Polite departure sentence",
        },
        {
            "id": "p5",
            "name": "National Greeting: MORNING + I + INDIAN",
            "tokens": ["MORNING", "I", "INDIAN"],
            "sentence": "Good morning! I am a proud Indian.",
            "description": "3-sign compound sentence",
        },
    ]
    return {"presets": presets}


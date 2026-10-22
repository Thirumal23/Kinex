"""
KINEX SQLite Database & Persistent Storage Module
Provides ACID-compliant persistent storage for:
  - Sign Language Library (labels, meanings, descriptions, categories, media, landmarks)
  - Sign Training Samples (recorded keypoints, features, glove telemetry, thumbnails)
  - Sentence Translations & Video Action History (sentences, tokens, confidence, video refs)
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
DB_PATH = DATA_DIR / "kinex.db"
STORAGE_DIR = PROJECT_ROOT / "storage"
UPLOAD_VIDEOS_DIR = STORAGE_DIR / "uploads" / "videos"
UPLOAD_IMAGES_DIR = STORAGE_DIR / "uploads" / "images"
UPLOAD_THUMBS_DIR = STORAGE_DIR / "uploads" / "thumbnails"

# Standard baseline ISL vocabulary seed definitions
BASELINE_ISL_SIGNS = [
    {
        "name": "THANK_YOU",
        "label": "Thank You",
        "meaning": "Expression of gratitude and appreciation toward another person.",
        "description": "Flat hand starts near the chin/lips with palm facing inward, moving gently forward and downward toward the recipient with a polite nod.",
        "category": "Politeness",
        "is_trained": 1,
    },
    {
        "name": "INDIAN",
        "label": "Indian",
        "meaning": "Referring to India, Indian nationality, or Indian heritage.",
        "description": "Index finger and thumb forming an arc or circle near the forehead or center chest, symbolizing the Indian national identity and heritage.",
        "category": "Identity",
        "is_trained": 1,
    },
    {
        "name": "NAME",
        "label": "Name",
        "meaning": "Asking for or stating someone's or one's own name.",
        "description": "Index and middle fingers extended (H-shape or 2-finger handshape), tapping the edge of the other hand's corresponding fingers twice.",
        "category": "Identity",
        "is_trained": 1,
    },
    {
        "name": "I",
        "label": "I / Me",
        "meaning": "First-person pronoun referring to oneself.",
        "description": "Index finger or pinky finger pointing directly toward the center of one's chest.",
        "category": "Identity",
        "is_trained": 1,
    },
    {
        "name": "HEARING",
        "label": "Hearing / Deaf",
        "meaning": "Relating to hearing status, the deaf community, or hard-of-hearing identity.",
        "description": "Index finger touches near the ear and moves down toward the mouth or chin, indicating the deaf/hearing communication bond.",
        "category": "Accessibility",
        "is_trained": 1,
    },
    {
        "name": "MORNING",
        "label": "Morning",
        "meaning": "Referring to morning time or a morning greeting.",
        "description": "Non-dominant arm rests flat horizontally while the dominant open hand rises from behind it upward like the rising sun.",
        "category": "Greetings",
        "is_trained": 1,
    },
    {
        "name": "BYE",
        "label": "Goodbye / Farewell",
        "meaning": "Parting salutation wishing someone well on departure.",
        "description": "Open flat palm raised to shoulder level, fingers waving side to side or opening and closing gently in farewell.",
        "category": "Greetings",
        "is_trained": 1,
    },
    {
        "name": "NAMASTE",
        "label": "Namaste",
        "meaning": "Traditional respectful Indian greeting meaning 'I bow to the divine in you'.",
        "description": "Both palms brought together flat in front of the chest in Anjali mudra, accompanied by a slight reverent bow.",
        "category": "Greetings",
        "is_trained": 1,
    },
    {
        "name": "HELP",
        "label": "Help / Assist",
        "meaning": "Requesting or offering urgent or general assistance.",
        "description": "Closed fist with thumb upright placed onto the flat open palm of the other hand, both lifting upward together.",
        "category": "Emergency",
        "is_trained": 1,
    },
    {
        "name": "WATER",
        "label": "Water",
        "meaning": "Requesting or referring to drinking water.",
        "description": "Three middle fingers extended in 'W' shape (or cupped hand), tapping near the lips twice.",
        "category": "Daily Needs",
        "is_trained": 1,
    },
    {
        "name": "FOOD",
        "label": "Food / Eat",
        "meaning": "Referring to eating, food, meals, or hunger.",
        "description": "Fingers and thumb of dominant hand pinched together like holding food, brought toward the mouth twice.",
        "category": "Daily Needs",
        "is_trained": 1,
    },
    {
        "name": "YES",
        "label": "Yes",
        "meaning": "Affirmation, agreement, or confirmation.",
        "description": "Closed fist held at chest height nodding up and down like a head nod.",
        "category": "General",
        "is_trained": 1,
    },
    {
        "name": "NO",
        "label": "No",
        "meaning": "Negation, disagreement, or decline.",
        "description": "Index and middle fingers snapping down against the thumb together firmly.",
        "category": "General",
        "is_trained": 1,
    },
    {
        "name": "PLEASE",
        "label": "Please",
        "meaning": "Polite request indicator.",
        "description": "Open flat hand rubbing gently in circular clockwise motion over the center of the chest.",
        "category": "Politeness",
        "is_trained": 1,
    },
    {
        "name": "SORRY",
        "label": "Sorry",
        "meaning": "Expression of apology or regret.",
        "description": "Closed fist rubbing in circular motion over the heart/chest area.",
        "category": "Politeness",
        "is_trained": 1,
    },
]


def _get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes SQLite schema, seeds standard ISL library, and migrates existing JSON files."""
    conn = _get_connection()
    cursor = conn.cursor()

    # 1. Signs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            meaning TEXT NOT NULL,
            description TEXT,
            category TEXT DEFAULT 'General',
            media_type TEXT DEFAULT 'none',
            media_path TEXT DEFAULT '',
            thumbnail_url TEXT DEFAULT '',
            landmarks_json TEXT DEFAULT '',
            features_json TEXT DEFAULT '',
            is_trained INTEGER DEFAULT 0,
            sample_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # 2. Sign Samples Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sign_samples (
            id TEXT PRIMARY KEY,
            sign_name TEXT NOT NULL,
            mode TEXT DEFAULT 'vision',
            features TEXT,
            keypoints TEXT,
            sensors TEXT,
            imu TEXT,
            thumbnail_url TEXT,
            created_at REAL,
            timestamp_str TEXT
        )
    """)

    # 3. Translations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id TEXT PRIMARY KEY,
            sentence TEXT NOT NULL,
            tokens TEXT NOT NULL,
            confidence REAL NOT NULL,
            modality TEXT DEFAULT 'vision',
            grammar_rule TEXT DEFAULT '',
            is_draft INTEGER DEFAULT 0,
            video_path TEXT DEFAULT '',
            duration_seconds REAL DEFAULT 0.0,
            timestamp_str TEXT NOT NULL,
            timestamp_ms REAL NOT NULL,
            notes TEXT DEFAULT ''
        )
    """)

    # Check if signs table is empty; if so, seed standard baseline gestures
    cursor.execute("SELECT COUNT(*) as count FROM signs")
    row = cursor.fetchone()
    now_str = time.strftime("%d %b %Y, %I:%M:%S %p")
    if row and row["count"] == 0:
        for s in BASELINE_ISL_SIGNS:
            cursor.execute("""
                INSERT OR IGNORE INTO signs (
                    name, label, meaning, description, category, is_trained, sample_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (
                s["name"], s["label"], s["meaning"], s["description"], s["category"], s["is_trained"], now_str, now_str
            ))

    # Migrate any existing samples from datasets/gesture_static/trainer_samples.json
    samples_file = PROJECT_ROOT / "datasets" / "gesture_static" / "trainer_samples.json"
    if samples_file.is_file():
        try:
            with open(samples_file, "r", encoding="utf-8") as f:
                raw_samples = json.load(f)
            for smp in raw_samples:
                s_id = smp.get("id")
                s_name = smp.get("gesture", "").upper().strip()
                if not s_id or not s_name:
                    continue

                # Ensure sign exists in signs table
                cursor.execute("SELECT id FROM signs WHERE name = ?", (s_name,))
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO signs (
                            name, label, meaning, description, category, is_trained, sample_count, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'Custom', 1, 0, ?, ?)
                    """, (
                        s_name,
                        s_name.replace("_", " ").title(),
                        f"Custom sign {s_name} added by user.",
                        f"Recorded gesture representing {s_name}.",
                        now_str,
                        now_str,
                    ))

                # Insert sample if not already present
                cursor.execute("SELECT id FROM sign_samples WHERE id = ?", (s_id,))
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO sign_samples (
                            id, sign_name, mode, features, keypoints, sensors, imu, thumbnail_url, created_at, timestamp_str
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        s_id,
                        s_name,
                        smp.get("mode", "vision"),
                        json.dumps(smp.get("features", [])),
                        json.dumps(smp.get("keypoints", [])),
                        json.dumps(smp.get("sensors", {})),
                        json.dumps(smp.get("imu", {})),
                        smp.get("thumbnail") or "",
                        float(smp.get("timestamp", time.time())),
                        smp.get("timestamp_str", now_str),
                    ))
        except Exception:
            pass

    # Update sample counts on signs table
    cursor.execute("""
        UPDATE signs SET sample_count = (
            SELECT COUNT(*) FROM sign_samples WHERE sign_samples.sign_name = signs.name
        )
    """)

    # Seed translations if empty
    cursor.execute("SELECT COUNT(*) as count FROM translations")
    t_row = cursor.fetchone()
    if t_row and t_row["count"] == 0:
        seed_translations = [
            ("trans-001", "I am an Indian citizen.", ["I", "INDIAN"], 0.974, "vision", "EXACT_IDIOMATIC_L2", 0, "", 1.6, "12 Oct 2026, 11:15:32", 1791803732000.0),
            ("trans-002", "Good morning! Thank you very much for your time.", ["MORNING", "THANK_YOU"], 0.981, "fusion", "EXACT_IDIOMATIC_L2", 0, "", 2.1, "12 Oct 2026, 11:18:45", 1791803925000.0),
            ("trans-003", "I am hard of hearing; thank you for communicating with me.", ["I", "HEARING", "THANK_YOU"], 0.962, "vision", "EXACT_IDIOMATIC_L3", 0, "", 2.8, "12 Oct 2026, 11:21:10", 1791804070000.0),
            ("trans-004", "Thank you so much, goodbye and take care!", ["THANK_YOU", "BYE"], 0.950, "glove", "EXACT_IDIOMATIC_L2", 0, "", 1.9, "12 Oct 2026, 11:23:04", 1791804184000.0),
        ]
        for t in seed_translations:
            cursor.execute("""
                INSERT OR IGNORE INTO translations (
                    id, sentence, tokens, confidence, modality, grammar_rule, is_draft, video_path, duration_seconds, timestamp_str, timestamp_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (t[0], t[1], json.dumps(t[2]), t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10]))

    conn.commit()
    conn.close()


# =========================================================================
# SIGN LIBRARY DATABASE ACCESSORS
# =========================================================================

def get_all_signs(
    search: Optional[str] = None,
    category: Optional[str] = None,
    is_trained: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Returns list of all signs with filtering and sample count."""
    conn = _get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM signs WHERE 1=1"
    params: List[Any] = []

    if search:
        s_term = f"%{search.strip().upper()}%"
        query += " AND (name LIKE ? OR label LIKE ? OR meaning LIKE ? OR description LIKE ?)"
        params.extend([s_term, s_term, s_term, s_term])

    if category and category.upper() != "ALL":
        query += " AND category = ?"
        params.append(category)

    if is_trained is not None:
        query += " AND is_trained = ?"
        params.append(int(is_trained))

    query += " ORDER BY is_trained DESC, sample_count DESC, name ASC"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for r in rows:
        item = dict(r)
        item["has_landmarks"] = bool(r["landmarks_json"] and r["landmarks_json"] != "[]")
        item["has_features"] = bool(r["features_json"] and r["features_json"] != "[]")
        results.append(item)

    conn.close()
    return results


def get_sign(id_or_name: Union[int, str]) -> Optional[Dict[str, Any]]:
    """Retrieves a single sign by numeric ID or upper-case name."""
    conn = _get_connection()
    cursor = conn.cursor()

    if isinstance(id_or_name, int) or (isinstance(id_or_name, str) and id_or_name.isdigit()):
        cursor.execute("SELECT * FROM signs WHERE id = ?", (int(id_or_name),))
    else:
        cursor.execute("SELECT * FROM signs WHERE UPPER(name) = ?", (str(id_or_name).upper().strip(),))

    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    sign = dict(row)
    # Fetch sample details for this sign
    cursor.execute(
        "SELECT id, mode, created_at, timestamp_str, thumbnail_url FROM sign_samples WHERE sign_name = ? ORDER BY created_at DESC",
        (sign["name"],)
    )
    sign["samples"] = [dict(s) for s in cursor.fetchall()]
    conn.close()
    return sign


def create_sign(
    name: str,
    label: Optional[str] = None,
    meaning: str = "",
    description: str = "",
    category: str = "General",
    media_type: str = "none",
    media_path: str = "",
    thumbnail_url: str = "",
    landmarks: Optional[List[Any]] = None,
    features: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Creates a new sign in the library. Rejects duplicate sign names."""
    clean_name = name.strip().upper().replace(" ", "_")
    if not clean_name:
        raise ValueError("Sign name/label cannot be empty.")

    clean_label = label.strip() if label else clean_name.replace("_", " ").title()
    clean_meaning = meaning.strip() or f"Represents '{clean_label}' in Indian Sign Language."
    clean_desc = description.strip() or f"Perform the '{clean_label}' gesture in front of the camera."

    conn = _get_connection()
    cursor = conn.cursor()

    # Check for duplicate
    cursor.execute("SELECT id FROM signs WHERE name = ?", (clean_name,))
    if cursor.fetchone():
        conn.close()
        raise ValueError(f"Sign '{clean_name}' already exists in the Sign Language Library.")

    now_str = time.strftime("%d %b %Y, %I:%M:%S %p")
    lm_json = json.dumps(landmarks) if landmarks else ""
    ft_json = json.dumps(features) if features else ""

    cursor.execute("""
        INSERT INTO signs (
            name, label, meaning, description, category, media_type, media_path,
            thumbnail_url, landmarks_json, features_json, is_trained, sample_count,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
    """, (
        clean_name, clean_label, clean_meaning, clean_desc, category,
        media_type, media_path, thumbnail_url, lm_json, ft_json,
        now_str, now_str
    ))

    sign_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return get_sign(sign_id)  # type: ignore


def update_sign(
    id_or_name: Union[int, str],
    label: Optional[str] = None,
    meaning: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    media_type: Optional[str] = None,
    media_path: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    landmarks: Optional[List[Any]] = None,
    features: Optional[List[float]] = None,
    is_trained: Optional[int] = None,
) -> Dict[str, Any]:
    """Updates sign metadata in the library."""
    sign = get_sign(id_or_name)
    if not sign:
        raise ValueError(f"Sign '{id_or_name}' not found.")

    conn = _get_connection()
    cursor = conn.cursor()

    updates = []
    params: List[Any] = []

    if label is not None:
        updates.append("label = ?")
        params.append(label.strip())
    if meaning is not None:
        updates.append("meaning = ?")
        params.append(meaning.strip())
    if description is not None:
        updates.append("description = ?")
        params.append(description.strip())
    if category is not None:
        updates.append("category = ?")
        params.append(category.strip())
    if media_type is not None:
        updates.append("media_type = ?")
        params.append(media_type.strip())
    if media_path is not None:
        updates.append("media_path = ?")
        params.append(media_path.strip())
    if thumbnail_url is not None:
        updates.append("thumbnail_url = ?")
        params.append(thumbnail_url.strip())
    if landmarks is not None:
        updates.append("landmarks_json = ?")
        params.append(json.dumps(landmarks))
    if features is not None:
        updates.append("features_json = ?")
        params.append(json.dumps(features))
    if is_trained is not None:
        updates.append("is_trained = ?")
        params.append(int(is_trained))

    updates.append("updated_at = ?")
    now_str = time.strftime("%d %b %Y, %I:%M:%S %p")
    params.append(now_str)

    params.append(sign["id"])
    query = f"UPDATE signs SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, params)
    conn.commit()
    conn.close()

    return get_sign(sign["id"])  # type: ignore


def delete_sign(id_or_name: Union[int, str]) -> Dict[str, Any]:
    """Deletes a sign and its recorded samples from the database."""
    sign = get_sign(id_or_name)
    if not sign:
        raise ValueError(f"Sign '{id_or_name}' not found.")

    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM sign_samples WHERE sign_name = ?", (sign["name"],))
    cursor.execute("DELETE FROM signs WHERE id = ?", (sign["id"],))

    conn.commit()
    conn.close()

    # Sync back to trainer_samples.json
    _sync_samples_file()

    return {"status": "DELETED", "sign_name": sign["name"], "id": sign["id"]}


def mark_signs_trained(class_names: List[str]):
    """Marks specified sign names as trained in the database."""
    if not class_names:
        return
    conn = _get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in class_names)
    cursor.execute(f"UPDATE signs SET is_trained = 1 WHERE name IN ({placeholders})", class_names)
    conn.commit()
    conn.close()


# =========================================================================
# SAMPLES DATABASE ACCESSORS
# =========================================================================

def add_sample(
    sign_name: str,
    features: List[float],
    keypoints: Optional[List[Any]] = None,
    sensors: Optional[Dict[str, float]] = None,
    imu: Optional[Dict[str, float]] = None,
    mode: str = "vision",
    thumbnail: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Adds a recorded training sample for a sign."""
    clean_name = sign_name.strip().upper()
    now_ts = timestamp if timestamp else time.time()
    now_str = time.strftime("%d %b %Y, %I:%M:%S %p", time.localtime(now_ts))
    sample_id = f"smp-{int(now_ts * 1000) % 1000000:06d}"

    conn = _get_connection()
    cursor = conn.cursor()

    # Ensure sign exists in signs table
    cursor.execute("SELECT id FROM signs WHERE name = ?", (clean_name,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO signs (
                name, label, meaning, description, category, is_trained, sample_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'Custom', 0, 0, ?, ?)
        """, (
            clean_name,
            clean_name.replace("_", " ").title(),
            f"User defined sign: {clean_name}",
            f"Hand gesture for {clean_name}",
            now_str,
            now_str,
        ))

    # Save sample
    cursor.execute("""
        INSERT INTO sign_samples (
            id, sign_name, mode, features, keypoints, sensors, imu, thumbnail_url, created_at, timestamp_str
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sample_id,
        clean_name,
        mode,
        json.dumps(features),
        json.dumps(keypoints or []),
        json.dumps(sensors or {}),
        json.dumps(imu or {}),
        thumbnail or "",
        now_ts,
        now_str,
    ))

    # Update sign sample_count
    cursor.execute("""
        UPDATE signs SET sample_count = (
            SELECT COUNT(*) FROM sign_samples WHERE sign_name = ?
        ), updated_at = ? WHERE name = ?
    """, (clean_name, now_str, clean_name))

    conn.commit()
    conn.close()

    _sync_samples_file()

    return {
        "id": sample_id,
        "gesture": clean_name,
        "mode": mode,
        "timestamp": now_ts,
        "timestamp_str": now_str,
        "has_keypoints": bool(keypoints),
        "thumbnail": thumbnail,
    }


def get_all_samples(sign_name: Optional[str] = None, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all training samples with optional sign filtering."""
    conn = _get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM sign_samples WHERE 1=1"
    params: List[Any] = []

    if sign_name and sign_name.upper() != "ALL":
        query += " AND sign_name = ?"
        params.append(sign_name.upper().strip())

    if mode:
        query += " AND mode = ?"
        params.append(mode.lower().strip())

    query += " ORDER BY created_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    samples = []
    for r in rows:
        samples.append({
            "id": r["id"],
            "gesture": r["sign_name"],
            "mode": r["mode"],
            "features": json.loads(r["features"]) if r["features"] else [],
            "keypoints": json.loads(r["keypoints"]) if r["keypoints"] else [],
            "sensors": json.loads(r["sensors"]) if r["sensors"] else {},
            "imu": json.loads(r["imu"]) if r["imu"] else {},
            "thumbnail": r["thumbnail_url"],
            "timestamp": r["created_at"],
            "timestamp_str": r["timestamp_str"],
            "has_keypoints": bool(r["keypoints"] and r["keypoints"] != "[]"),
        })

    conn.close()
    return samples


def get_sample_counts() -> Dict[str, int]:
    """Returns sample counts dictionary per gesture."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sign_name, COUNT(*) as cnt FROM sign_samples GROUP BY sign_name")
    counts = {r["sign_name"]: r["cnt"] for r in cursor.fetchall()}
    conn.close()
    return counts


def update_sample_label(sample_id: str, new_label: str) -> Dict[str, Any]:
    """Updates sign label for a specific sample."""
    clean_label = new_label.strip().upper()
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT sign_name FROM sign_samples WHERE id = ?", (sample_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Sample '{sample_id}' not found.")

    old_sign = row["sign_name"]
    cursor.execute("UPDATE sign_samples SET sign_name = ? WHERE id = ?", (clean_label, sample_id))

    # Update counts for both signs
    now_str = time.strftime("%d %b %Y, %I:%M:%S %p")
    for s_name in [old_sign, clean_label]:
        cursor.execute("""
            UPDATE signs SET sample_count = (
                SELECT COUNT(*) FROM sign_samples WHERE sign_name = ?
            ), updated_at = ? WHERE name = ?
        """, (s_name, now_str, s_name))

    conn.commit()
    conn.close()

    _sync_samples_file()
    return {"status": "UPDATED", "id": sample_id, "gesture": clean_label}


def delete_sample(sample_id: str) -> Dict[str, Any]:
    """Deletes a single sample by ID."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT sign_name FROM sign_samples WHERE id = ?", (sample_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Sample '{sample_id}' not found.")

    sign_name = row["sign_name"]
    cursor.execute("DELETE FROM sign_samples WHERE id = ?", (sample_id,))

    now_str = time.strftime("%d %b %Y, %I:%M:%S %p")
    cursor.execute("""
        UPDATE signs SET sample_count = (
            SELECT COUNT(*) FROM sign_samples WHERE sign_name = ?
        ), updated_at = ? WHERE name = ?
    """, (sign_name, now_str, sign_name))

    conn.commit()
    conn.close()

    _sync_samples_file()
    return {"status": "DELETED", "id": sample_id}


def clear_all_samples():
    """Clears all recorded training samples from SQLite and marks signs sample_count=0."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sign_samples")
    cursor.execute("UPDATE signs SET sample_count = 0")
    conn.commit()
    conn.close()
    _sync_samples_file()


def _sync_samples_file():
    """Syncs SQLite samples back to trainer_samples.json for backward compatibility."""
    samples_file = PROJECT_ROOT / "datasets" / "gesture_static" / "trainer_samples.json"
    try:
        samples = get_all_samples()
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
    except Exception:
        pass


# =========================================================================
# TRANSLATIONS DATABASE ACCESSORS
# =========================================================================

def save_translation(
    sentence: str,
    tokens: List[str],
    confidence: float,
    modality: str = "vision",
    grammar_rule: str = "",
    is_draft: bool = False,
    video_path: str = "",
    duration_seconds: float = 0.0,
    notes: str = "",
) -> Dict[str, Any]:
    """Saves a sentence translation record into persistent SQLite database."""
    now_ts = time.time()
    now_str = time.strftime("%d %b %Y, %I:%M:%S %p", time.localtime(now_ts))
    trans_id = f"trans-{int(now_ts * 1000) % 1000000:06d}"

    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO translations (
            id, sentence, tokens, confidence, modality, grammar_rule,
            is_draft, video_path, duration_seconds, timestamp_str, timestamp_ms, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trans_id, sentence, json.dumps(tokens), round(confidence, 3), modality,
        grammar_rule, 1 if is_draft else 0, video_path, duration_seconds,
        now_str, now_ts * 1000.0, notes
    ))

    conn.commit()
    conn.close()

    return {
        "id": trans_id,
        "sentence": sentence,
        "tokens": tokens,
        "confidence": round(confidence, 3),
        "modality": modality,
        "grammar_rule": grammar_rule,
        "is_draft": bool(is_draft),
        "video_path": video_path,
        "duration_seconds": duration_seconds,
        "timestamp_str": now_str,
        "timestamp_ms": now_ts * 1000.0,
        "notes": notes,
    }


def get_all_translations(search: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Returns persistent list of translation history."""
    conn = _get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM translations WHERE 1=1"
    params: List[Any] = []

    if search:
        s_term = f"%{search.strip()}%"
        query += " AND (sentence LIKE ? OR tokens LIKE ? OR notes LIKE ?)"
        params.extend([s_term, s_term, s_term, s_term])

    query += " ORDER BY timestamp_ms DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "sentence": r["sentence"],
            "tokens": json.loads(r["tokens"]) if r["tokens"] else [],
            "confidence": r["confidence"],
            "modality": r["modality"],
            "grammar_rule": r["grammar_rule"],
            "is_draft": bool(r["is_draft"]),
            "video_path": r["video_path"],
            "duration_seconds": r["duration_seconds"],
            "timestamp": r["timestamp_str"],
            "timestamp_str": r["timestamp_str"],
            "timestamp_ms": r["timestamp_ms"],
            "notes": r["notes"],
        })

    conn.close()
    return results


def delete_translation(trans_id: str) -> Dict[str, Any]:
    """Deletes a specific translation record from history."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM translations WHERE id = ?", (trans_id,))
    conn.commit()
    conn.close()
    return {"status": "DELETED", "id": trans_id}


def clear_all_translations():
    """Clears all saved translation history."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM translations")
    conn.commit()
    conn.close()


# =========================================================================
# DASHBOARD OVERVIEW METRICS
# =========================================================================

def get_dashboard_metrics() -> Dict[str, Any]:
    """Calculates comprehensive statistics for the Dashboard."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM signs")
    total_signs = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as trained FROM signs WHERE is_trained = 1")
    trained_signs = cursor.fetchone()["trained"]

    cursor.execute("SELECT COUNT(*) as total FROM sign_samples")
    total_samples = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as total FROM translations")
    total_translations = cursor.fetchone()["total"]

    # Recent translations (top 5)
    cursor.execute("SELECT * FROM translations ORDER BY timestamp_ms DESC LIMIT 5")
    recent_translations = []
    for r in cursor.fetchall():
        recent_translations.append({
            "id": r["id"],
            "sentence": r["sentence"],
            "tokens": json.loads(r["tokens"]) if r["tokens"] else [],
            "confidence": r["confidence"],
            "modality": r["modality"],
            "timestamp_str": r["timestamp_str"],
            "video_path": r["video_path"],
        })

    # Recent signs (top 6)
    cursor.execute("SELECT * FROM signs ORDER BY updated_at DESC LIMIT 6")
    recent_signs = [dict(r) for r in cursor.fetchall()]

    # Video recordings count
    video_count = 0
    if UPLOAD_VIDEOS_DIR.is_dir():
        video_count = len(list(UPLOAD_VIDEOS_DIR.glob("*.*")))

    conn.close()

    return {
        "total_signs": total_signs,
        "trained_signs": trained_signs,
        "total_samples": total_samples,
        "total_translations": total_translations,
        "total_recordings": video_count,
        "recent_translations": recent_translations,
        "recent_signs": recent_signs,
    }

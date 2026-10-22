#!/usr/bin/env python3
"""
Full End-to-End Workflow Verification for KINEX Improvements:
1. Clear All Samples & Verification
2. Sample Recording & Metadata (21 Keypoints & Camera Thumbnails)
3. Sample Management: Retrieval, Label Editing (PUT/PATCH), Individual Deletion
4. Model Training strictly on actual samples & Metrics Verification
5. Real-time Inference using the custom trained model
6. Final Reset & Verification
"""

import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://localhost:8000"


def request(method, path, data=None):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode("utf-8") if data is not None else None
    try:
        with urllib.request.urlopen(req, data=body, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        try:
            return e.code, json.loads(content)
        except Exception:
            return e.code, {"error": content}


def make_dummy_keypoints(pose_type="open"):
    kpts = [[0.5, 0.8, 0.0]]  # wrist
    if pose_type == "open":
        for i in range(1, 21):
            kpts.append([0.3 + (i % 5) * 0.1, 0.7 - (i // 5) * 0.15, 0.0])
    elif pose_type == "fist":
        for i in range(1, 21):
            kpts.append([0.48 + (i % 3) * 0.02, 0.75 - (i // 4) * 0.03, 0.0])
    elif pose_type == "peace":
        for i in range(1, 21):
            if i in [5, 6, 7, 8, 9, 10, 11, 12]:
                kpts.append([0.4 + (i % 2) * 0.2, 0.2 + (i // 4) * 0.1, 0.0])
            else:
                kpts.append([0.5, 0.75, 0.0])
    return kpts


def run_tests():
    print("=" * 60)
    print("KINEX END-TO-END VERIFICATION SUITE")
    print("=" * 60)

    # 1. Reset / Clear All Samples
    print("[1/6] Testing Clear All Samples (POST /api/trainer/reset)...")
    code, res = request("POST", "/api/trainer/reset")
    assert code == 200, f"Reset failed: {res}"
    assert res.get("total_samples") == 0, f"Expected 0 samples, got {res.get('total_samples')}"
    print("  -> PASS: All samples and custom models cleared.")

    # 2. Check Samples Empty State
    print("[2/6] Verifying empty state (GET /api/trainer/samples)...")
    code, res = request("GET", "/api/trainer/samples")
    assert code == 200
    assert res.get("total_samples") == 0
    assert len(res.get("samples", [])) == 0
    assert res.get("is_custom_trained") is False
    print("  -> PASS: Initial state is completely clean.")

    # 3. Record Samples with Keypoints & Thumbnails
    print("[3/6] Recording custom sign samples (OPEN_PALM & FIST)...")
    sample_ids = []
    for i in range(4):
        code, res = request("POST", "/api/trainer/record", {
            "gesture": "OPEN_PALM",
            "keypoints": make_dummy_keypoints("open"),
            "thumbnail": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "sensors": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
        })
        assert code == 200
        sample_ids.append(res["sample"]["id"])

    for i in range(4):
        code, res = request("POST", "/api/trainer/record", {
            "gesture": "FIST",
            "keypoints": make_dummy_keypoints("fist"),
            "thumbnail": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "sensors": {"thumb": 0.9, "index": 0.9, "middle": 0.9, "ring": 0.9, "little": 0.9},
        })
        assert code == 200
        sample_ids.append(res["sample"]["id"])

    code, res = request("GET", "/api/trainer/samples")
    assert res.get("total_samples") == 8
    assert res.get("class_counts", {}).get("OPEN_PALM") == 4
    assert res.get("class_counts", {}).get("FIST") == 4
    first_sample = res["samples"][0]
    assert first_sample.get("has_keypoints") is True
    assert len(first_sample.get("keypoints", [])) == 21
    assert first_sample.get("thumbnail") is not None
    print(f"  -> PASS: 8 samples recorded with 21 keypoints and camera thumbnails. Counts: {res.get('class_counts')}")

    # 4. Sample Management: Edit Label & Delete
    print("[4/6] Testing Sample Management: Edit Label & Delete Individual...")
    target_id = sample_ids[0]  # One of OPEN_PALM
    code, res = request("PUT", f"/api/trainer/sample/{target_id}", {"gesture": "PEACE"})
    assert code == 200, f"Edit label failed: {res}"
    assert res["sample"]["gesture"] == "PEACE"

    code, res = request("GET", "/api/trainer/samples")
    assert res["class_counts"].get("PEACE") == 1
    assert res["class_counts"].get("OPEN_PALM") == 3
    print("  -> PASS: Sample label updated to 'PEACE' successfully and counts updated.")

    # Delete the newly renamed sample
    code, res = request("DELETE", f"/api/trainer/sample/{target_id}")
    assert code == 200
    assert res.get("total_samples") == 7

    code, res = request("GET", "/api/trainer/samples")
    assert res.get("total_samples") == 7
    assert "PEACE" not in res.get("class_counts", {})
    print("  -> PASS: Sample deleted successfully with updated counts.")

    # Add 1 more OPEN_PALM to keep balance for training
    request("POST", "/api/trainer/record", {
        "gesture": "OPEN_PALM",
        "keypoints": make_dummy_keypoints("open"),
        "sensors": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
    })

    # 5. Model Training & Custom Inference
    print("[5/6] Testing Model Training strictly on saved samples...")
    code, res = request("POST", "/api/trainer/train", {})
    assert code == 200, f"Training failed: {res}"
    assert res.get("status") == "SUCCESS"
    assert "OPEN_PALM" in res.get("classes", [])
    assert "FIST" in res.get("classes", [])
    assert res.get("accuracy", 0) > 85.0
    print(f"  -> PASS: Model trained! Classes: {res.get('classes')}, Accuracy: {res.get('accuracy')}%, Augmented samples: {res.get('augmented_samples')}")

    # Check Export
    code, exp = request("GET", "/api/trainer/export")
    assert code == 200
    assert exp.get("is_custom") is True
    assert len(exp.get("weights", [])) == len(exp.get("classes", []))
    print(f"  -> PASS: Model exported with unnormalized weights for client-side inference.")

    # Live Prediction using the custom model
    print("[6/6] Testing Live Prediction with Custom Trained Model...")
    code, pred = request("POST", "/predict", {
        "keypoints": make_dummy_keypoints("open"),
        "mode": "vision"
    })
    assert code == 200, f"Predict failed: {pred}"
    print(f"  -> Prediction result for Open Hand: {pred.get('gesture')} (confidence: {pred.get('confidence'):.2f}, status: {pred.get('status')})")
    assert pred.get("gesture") == "OPEN_PALM", f"Expected OPEN_PALM, got {pred.get('gesture')}"

    code, pred = request("POST", "/predict", {
        "keypoints": make_dummy_keypoints("fist"),
        "mode": "vision"
    })
    assert code == 200, f"Predict failed: {pred}"
    print(f"  -> Prediction result for Fist Hand: {pred.get('gesture')} (confidence: {pred.get('confidence'):.2f}, status: {pred.get('status')})")
    assert pred.get("gesture") == "FIST", f"Expected FIST, got {pred.get('gesture')}"

    # Final Clean Reset as requested by user ("Remove all currently stored sample data from the website so I can start fresh")
    print("\n[FINAL CLEANUP] Resetting dataset to start completely fresh...")
    request("POST", "/api/trainer/reset")
    code, res = request("GET", "/api/trainer/samples")
    assert res.get("total_samples") == 0
    print("  -> Stored samples cleared to 0. Custom classifier wiped. System is fresh!")

    print("\n" + "=" * 60)
    print("ALL 6 VERIFICATION PHASES PASSED WITH ZERO ERRORS!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

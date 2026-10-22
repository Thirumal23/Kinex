"""
Unit Tests for KINEX Backend API & Inference Services
Validates:
  1. GET /health
  2. GET /model/info
  3. POST /predict with 21 keypoints
  4. POST /predict with 126-D features
  5. POST /predict/sequence
  6. Static file mounting (GET / returns index.html)
"""

import unittest
from fastapi.testclient import TestClient
from backend.api.main import app


class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_01_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "2.0.0")

    def test_02_model_info_endpoint(self):
        res = self.client.get("/model/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["random_init_verified"])
        self.assertTrue(data["pretrained_forbidden"])
        self.assertEqual(data["keypoints"], 21)
        self.assertEqual(data["input_dimension"], 126)
        self.assertIn("THANK_YOU", data["classes"])

    def test_03_predict_keypoints(self):
        # 21 mock hand keypoints
        keypoints = [[100.0 + i * 5, 200.0 + i * 10, 1.0] for i in range(21)]
        res = self.client.post("/predict", json={"keypoints": keypoints})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("gesture", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)

    def test_04_predict_features(self):
        features = [0.1] * 126
        res = self.client.post("/predict", json={"features": features})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("gesture", data)
        self.assertIn("confidence", data)

    def test_05_predict_sequence(self):
        sequence = [[0.1] * 126 for _ in range(16)]
        res = self.client.post("/predict/sequence", json={"sequence": sequence})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("gesture", data)
        self.assertEqual(data["sequence_length"], 16)

    def test_06_dashboard_served_at_root(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("KINEX", res.text)
        self.assertIn("skeletonCanvas", res.text)

    def test_07_predict_dual_hands(self):
        hand1 = [[100.0 + i * 5, 200.0 + i * 10, 1.0] for i in range(21)]
        hand2 = [[120.0 + i * 5, 200.0 + i * 10, 1.0] for i in range(21)]
        res = self.client.post("/predict", json={"hands": [hand1, hand2]})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["hands_detected"], 2)
        self.assertIn("gesture", data)
        self.assertIn("confidence", data)

    def test_08_mode_management_endpoints(self):
        # GET default mode
        res = self.client.get("/api/mode")
        self.assertEqual(res.status_code, 200)
        self.assertIn("mode", res.json())

        # Switch to all 3 modes
        for mode in ["vision", "glove", "fusion"]:
            res = self.client.post("/api/mode", json={"mode": mode})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["mode"], mode)

        # Invalid mode fails cleanly
        res = self.client.post("/api/mode", json={"mode": "invalid_mode"})
        self.assertEqual(res.status_code, 400)

        # Switch back to vision
        self.client.post("/api/mode", json={"mode": "vision"})

    def test_09_predict_glove_mode(self):
        glove_payload = {
            "sensors": {"thumb": 0.1, "index": 0.15, "middle": 0.15, "ring": 0.15, "little": 0.15},
            "imu": {"pitch": 25.0, "roll": 0.0, "yaw": 0.0}
        }
        res = self.client.post("/predict/glove", json=glove_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("gesture", data)
        self.assertIn("probabilities", data)
        self.assertGreaterEqual(data["confidence"], 0.0)

    def test_10_multimodal_trainer_flow(self):
        # 1. Reset trainer
        res = self.client.post("/api/trainer/reset")
        self.assertEqual(res.status_code, 200)

        # 2. Record samples in Mode 1 (Vision)
        for i in range(3):
            kpts = [[100.0 + i, 200.0 + j, 1.0] for j in range(21)]
            res = self.client.post("/api/trainer/record", json={
                "gesture": "THANK_YOU",
                "keypoints": kpts,
                "mode": "vision"
            })
            self.assertEqual(res.status_code, 200)

        for i in range(3):
            kpts = [[150.0 + i, 220.0 + j, 1.0] for j in range(21)]
            res = self.client.post("/api/trainer/record", json={
                "gesture": "INDIAN",
                "keypoints": kpts,
                "mode": "vision"
            })
            self.assertEqual(res.status_code, 200)

        # 3. Check samples info
        res = self.client.get("/api/trainer/samples")
        self.assertEqual(res.status_code, 200)
        info = res.json()
        self.assertEqual(info["total_samples"], 6)
        self.assertEqual(info["class_counts"]["THANK_YOU"], 3)
        self.assertEqual(info["class_counts"]["INDIAN"], 3)
        self.assertEqual(info["mode_counts"]["vision"], 6)

        # 4. Train model
        res = self.client.post("/api/trainer/train", json={"mode": "vision"})
        self.assertEqual(res.status_code, 200)
        train_res = res.json()
        self.assertEqual(train_res["status"], "SUCCESS")
        self.assertGreaterEqual(train_res["accuracy"], 50.0)
        self.assertIn("THANK_YOU", train_res["classes"])

        # 5. Clean up with reset
        res = self.client.post("/api/trainer/reset")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["total_samples"], 0)

    def test_11_sentence_synthesis(self):
        # 1. Synthesize 2-token sentence
        res = self.client.post("/api/sentences/synthesize", json={
            "tokens": ["I", "INDIAN"],
            "mode": "vision"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["sentence"], "I am an Indian citizen.")
        self.assertEqual(data["tokens"], ["I", "INDIAN"])
        self.assertGreater(data["confidence"], 0.8)

        # 2. Synthesize with save_to_history=True
        res = self.client.post("/api/sentences/synthesize", json={
            "tokens": ["MORNING", "THANK_YOU"],
            "save_to_history": True
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn("Good morning", res.json()["sentence"])

    def test_12_sentence_history_and_presets(self):
        # 1. Query sentence history
        res = self.client.get("/api/sentences/history")
        self.assertEqual(res.status_code, 200)
        history = res.json()
        self.assertGreater(history["total"], 0)
        self.assertTrue(len(history["sentences"]) > 0)
        first_item = history["sentences"][0]
        self.assertIn("sentence", first_item)
        self.assertIn("tokens", first_item)

        # 2. Query presets
        res_presets = self.client.get("/api/sentences/presets")
        self.assertEqual(res_presets.status_code, 200)
        presets_data = res_presets.json()
        self.assertIn("presets", presets_data)
        self.assertGreaterEqual(len(presets_data["presets"]), 3)

        # 3. Save manual record
        rec = {
            "id": "sent-test-99",
            "sentence": "Namaste, warm greetings to you.",
            "tokens": ["NAMASTE"],
            "confidence": 0.99,
            "modality": "vision",
            "timestamp": "12 Oct 2026, 11:30:00",
            "timestamp_ms": 1791804600000.0,
            "duration_seconds": 1.2
        }
        res_save = self.client.post("/api/sentences/save", json=rec)
        self.assertEqual(res_save.status_code, 200)
        self.assertEqual(res_save.json()["status"], "SAVED")

    def test_13_trainer_samples_management_and_sentences(self):
        # 1. Reset
        self.client.post("/api/trainer/reset")

        # 2. Record samples
        for i in range(2):
            kpts = [[100.0 + i, 200.0 + j, 1.0] for j in range(21)]
            self.client.post("/api/trainer/record", json={"gesture": "I", "keypoints": kpts})
        for i in range(2):
            kpts = [[120.0 + i, 210.0 + j, 1.0] for j in range(21)]
            self.client.post("/api/trainer/record", json={"gesture": "INDIAN", "keypoints": kpts})

        # 3. Get samples and check sentence readiness
        res = self.client.get("/api/trainer/samples")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_samples"], 4)
        self.assertIn("samples", data)
        self.assertIn("supported_sentences", data)

        # 4. Check sentences endpoint
        res_sent = self.client.get("/api/trainer/sentences")
        self.assertEqual(res_sent.status_code, 200)
        sentences = res_sent.json()["sentences"]
        # 'I + INDIAN' should be fully trained
        i_indian = next((s for s in sentences if s["id"] == "I+INDIAN"), None)
        self.assertIsNotNone(i_indian)
        self.assertTrue(i_indian["is_fully_trained"])

        # 5. Delete one sample
        first_id = data["samples"][0]["id"]
        del_res = self.client.delete(f"/api/trainer/sample/{first_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["total_samples"], 3)

        # 6. Train model and verify export
        train_res = self.client.post("/api/trainer/train", json={"mode": "vision"})
        self.assertEqual(train_res.status_code, 200)
        self.assertIn("weights", train_res.json())

        exp_res = self.client.get("/api/trainer/export")
        self.assertEqual(exp_res.status_code, 200)
        self.assertTrue(exp_res.json()["is_custom"])

        # Cleanup
        self.client.post("/api/trainer/reset")

    def test_14_sample_edit_and_management(self):
        # 1. Reset
        self.client.post("/api/trainer/reset")

        # 2. Record a sample with keypoints and thumbnail
        kpts = [[100.0 + j, 150.0 + j, 1.0] for j in range(21)]
        thumb_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
        rec_res = self.client.post("/api/trainer/record", json={
            "gesture": "OLD_SIGN",
            "keypoints": kpts,
            "thumbnail": thumb_data
        })
        self.assertEqual(rec_res.status_code, 200)
        sample_id = rec_res.json()["sample"]["id"]

        # 3. Get single sample
        get_res = self.client.get(f"/api/trainer/sample/{sample_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["gesture"], "OLD_SIGN")
        self.assertEqual(get_res.json()["thumbnail"], thumb_data)

        # 4. Edit sample label
        edit_res = self.client.put(f"/api/trainer/sample/{sample_id}", json={"gesture": "NEW_SIGN"})
        self.assertEqual(edit_res.status_code, 200)
        self.assertEqual(edit_res.json()["status"], "UPDATED")
        self.assertEqual(edit_res.json()["sample"]["gesture"], "NEW_SIGN")

        # 5. Verify sample list shows updated label
        list_res = self.client.get("/api/trainer/samples")
        self.assertEqual(list_res.status_code, 200)
        samples = list_res.json()["samples"]
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["gesture"], "NEW_SIGN")
        self.assertIn("NEW_SIGN", list_res.json()["class_counts"])
        self.assertNotIn("OLD_SIGN", list_res.json()["class_counts"])

        # 6. Cleanup
        self.client.post("/api/trainer/reset")

    def test_15_clear_all_samples_and_reset(self):
        # Record multiple samples
        for i in range(3):
            kpts = [[100.0, 150.0, 1.0] for _ in range(21)]
            self.client.post("/api/trainer/record", json={"gesture": f"TEST_{i}", "keypoints": kpts})

        res_before = self.client.get("/api/trainer/samples")
        self.assertEqual(res_before.json()["total_samples"], 3)

        # Clear all samples
        reset_res = self.client.post("/api/trainer/reset")
        self.assertEqual(reset_res.status_code, 200)
        self.assertEqual(reset_res.json()["status"], "RESET")
        self.assertEqual(reset_res.json()["total_samples"], 0)

        # Verify completely empty
        res_after = self.client.get("/api/trainer/samples")
        self.assertEqual(res_after.json()["total_samples"], 0)
        self.assertEqual(len(res_after.json()["samples"]), 0)
        self.assertEqual(len(res_after.json()["class_counts"]), 0)


if __name__ == "__main__":
    unittest.main()



"""
Unit tests for Sign Language Library, Video Processing, and Translation Persistence
"""

import io
import json
import unittest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.database.db import get_all_signs, get_sign, delete_sign


class TestLibraryAndVideoAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_01_dashboard_stats(self):
        res = self.client.get("/api/dashboard/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_signs", data)
        self.assertIn("trained_signs", data)
        self.assertIn("total_translations", data)
        self.assertGreaterEqual(data["total_signs"], 7)

    def test_02_get_all_signs(self):
        res = self.client.get("/api/signs")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("signs", data)
        names = [s["name"] for s in data["signs"]]
        self.assertIn("THANK_YOU", names)
        self.assertIn("INDIAN", names)

    def test_03_create_and_duplicate_sign(self):
        # Create unique test sign
        unique_name = "TEST_SIGN_XYZ"
        # Ensure it doesn't already exist
        try:
            delete_sign(unique_name)
        except Exception:
            pass

        create_res = self.client.post("/api/signs", json={
            "name": unique_name,
            "label": "Test Sign XYZ",
            "meaning": "Special test gesture for verification.",
            "description": "Show open hand to camera.",
            "category": "Test",
        })
        self.assertEqual(create_res.status_code, 201)
        created_data = create_res.json()
        self.assertEqual(created_data["name"], unique_name)
        self.assertEqual(created_data["meaning"], "Special test gesture for verification.")

        # Duplicate submission must return 409 Conflict
        dup_res = self.client.post("/api/signs", json={
            "name": unique_name,
            "label": "Test Sign XYZ Duplicate",
            "meaning": "Another meaning.",
        })
        self.assertEqual(dup_res.status_code, 409)

        # Get sign details
        detail_res = self.client.get(f"/api/signs/{unique_name}")
        self.assertEqual(detail_res.status_code, 200)
        self.assertEqual(detail_res.json()["name"], unique_name)

        # Update sign
        update_res = self.client.put(f"/api/signs/{unique_name}", json={
            "meaning": "Updated meaning text for test sign.",
            "category": "UpdatedCategory",
        })
        self.assertEqual(update_res.status_code, 200)
        self.assertEqual(update_res.json()["meaning"], "Updated meaning text for test sign.")
        self.assertEqual(update_res.json()["category"], "UpdatedCategory")

        # Delete sign
        del_res = self.client.delete(f"/api/signs/{unique_name}")
        self.assertEqual(del_res.status_code, 200)

        # Verify 404 after deletion
        after_del = self.client.get(f"/api/signs/{unique_name}")
        self.assertEqual(after_del.status_code, 404)

    def test_04_translations_persistence(self):
        save_res = self.client.post("/api/translations/save", json={
            "sentence": "Namaste! I am an Indian citizen.",
            "tokens": ["NAMASTE", "I", "INDIAN"],
            "confidence": 0.985,
            "modality": "vision",
            "grammar_rule": "EXACT_IDIOMATIC_L3",
            "notes": "Test persistence verification",
        })
        self.assertEqual(save_res.status_code, 200)
        saved = save_res.json()
        trans_id = saved["id"]

        # Fetch translations
        list_res = self.client.get("/api/translations")
        self.assertEqual(list_res.status_code, 200)
        translations = list_res.json()["translations"]
        found = any(t["id"] == trans_id for t in translations)
        self.assertTrue(found)

        # Delete translation
        del_res = self.client.delete(f"/api/translations/{trans_id}")
        self.assertEqual(del_res.status_code, 200)

    def test_05_input_validation(self):
        # Empty name
        res1 = self.client.post("/api/signs", json={
            "name": "",
            "meaning": "Some meaning"
        })
        self.assertEqual(res1.status_code, 400)

        # Empty meaning
        res2 = self.client.post("/api/signs", json={
            "name": "VALID_NAME",
            "meaning": ""
        })
        self.assertEqual(res2.status_code, 400)


if __name__ == "__main__":
    unittest.main()

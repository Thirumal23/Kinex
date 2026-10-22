"""
KINEX Feature Engineering Service (V4 Combined 126-D)
Transforms 21 raw hand pose keypoints into scale, translation, and rotation invariant
geometric feature vectors ready for semantic gesture classification.

Complies strictly with ARCHITECTURE.md Section 3.3.
"""

import math
from typing import List, Tuple, Union
import numpy as np


class FeatureExtractor:
    """
    Extracts 126-dimensional geometric features from 21 hand pose keypoints:
      - 42 values: Wrist-relative normalized landmark coordinates (x, y)
      - 40 values: Inter-joint unit bone direction vectors (20 bones * 2)
      - 15 values: Flexion and abduction joint angles
      - 21 values: Keypoint detection confidences / visibility scores
      - 8 values:  Palm geometry, span ratios, and fingertip distances
      Total: 126 dimensions
    """

    BONE_EDGES = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    FINGER_TRIPLETS = [
        # Thumb joints
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        # Index joints
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        # Middle joints
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        # Ring joints
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        # Pinky joints
        (0, 17, 18), (17, 18, 19), (18, 19, 20),
    ]

    FINGERTIP_INDICES = [4, 8, 12, 16, 20]

    def __init__(self, target_dim: int = 126):
        self.target_dim = target_dim

    def extract(self, keypoints: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Converts 21 keypoints [[x, y, v], ...] or [21, 2] into a 126-D feature vector.
        """
        kpts = np.asarray(keypoints, dtype=np.float32)
        if kpts.shape[0] != 21:
            raise ValueError(f"Expected 21 keypoints, received shape: {kpts.shape}")

        coords = kpts[:, :2]
        visibilities = kpts[:, 2] if kpts.shape[1] >= 3 else np.ones(21, dtype=np.float32)

        # 1. Wrist-origin translation (P0 -> origin [0,0])
        wrist = coords[0]
        centered = coords - wrist

        # 2. Scale normalization: distance from wrist (0) to middle MCP (9)
        scale = np.linalg.norm(centered[9])
        if scale < 1e-4:
            # Fallback to maximum coordinate distance if scale is degenerate
            dists = np.linalg.norm(centered, axis=1)
            scale = np.max(dists) if np.max(dists) > 1e-4 else 1.0

        normalized_coords = centered / scale
        coords_flat = normalized_coords.flatten()  # 42 values

        # 3. Inter-joint unit bone direction vectors (20 bones * 2 = 40 values)
        bone_vectors = []
        for a, b in self.BONE_EDGES:
            vec = coords[b] - coords[a]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                unit_vec = vec / norm
            else:
                unit_vec = np.zeros(2, dtype=np.float32)
            bone_vectors.extend(unit_vec.tolist())
        bone_vecs = np.asarray(bone_vectors, dtype=np.float32)  # 40 values

        # 4. Joint angles (15 flexion angles)
        angles = []
        for a, b, c in self.FINGER_TRIPLETS:
            v1 = coords[a] - coords[b]
            v2 = coords[c] - coords[b]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cosine = np.dot(v1, v2) / (norm1 * norm2)
                cosine = np.clip(cosine, -1.0, 1.0)
                angle_rad = np.arccos(cosine)
                normalized_angle = angle_rad / np.pi  # Normalize to [0, 1]
            else:
                normalized_angle = 1.0
            angles.append(normalized_angle)
        angles_arr = np.asarray(angles, dtype=np.float32)  # 15 values

        # 5. Keypoint confidences / visibilities (21 values)
        conf_arr = np.asarray(visibilities, dtype=np.float32)  # 21 values

        # 6. Palm geometric features (8 values)
        # 6a. Normalized distances from wrist to each fingertip (5 values)
        fingertip_dists = [np.linalg.norm(normalized_coords[tip]) for tip in self.FINGERTIP_INDICES]

        # 6b. Palm span (distance between thumb CMC #1 and pinky MCP #17)
        palm_width = np.linalg.norm(normalized_coords[1] - normalized_coords[17])

        # 6c. Hand aspect ratio
        bbox_w = np.max(normalized_coords[:, 0]) - np.min(normalized_coords[:, 0])
        bbox_h = np.max(normalized_coords[:, 1]) - np.min(normalized_coords[:, 1])
        aspect_ratio = bbox_w / (bbox_h + 1e-6)

        # 6d. Hand main orientation angle relative to vertical
        v_middle = normalized_coords[12] - normalized_coords[0]
        orientation = (math.atan2(v_middle[1], v_middle[0]) + math.pi) / (2.0 * math.pi)

        palm_features = np.asarray(
            fingertip_dists + [palm_width, aspect_ratio, orientation],
            dtype=np.float32
        )  # 8 values

        # Concatenate into 126-D feature vector
        v4_combined = np.concatenate([
            coords_flat,    # 42
            bone_vecs,      # 40
            angles_arr,     # 15
            conf_arr,       # 21
            palm_features   # 8
        ])

        if len(v4_combined) < self.target_dim:
            padding = np.zeros(self.target_dim - len(v4_combined), dtype=np.float32)
            v4_combined = np.concatenate([v4_combined, padding])
        elif len(v4_combined) > self.target_dim:
            v4_combined = v4_combined[:self.target_dim]

        return v4_combined

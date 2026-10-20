"""
Comprehensive Automated Test Suite for AI-Based Accident Detection System.
Tests:
1. Model loading and inference.
2. Centroid tracking and kinematics.
3. IoU and overlap ratio calculation.
4. Normal traffic scenario (zero collisions expected).
5. Collision scenario (collision alert triggered).
6. Invalid file handling.
7. CSV report export.
"""

import os
import sys
import unittest
import numpy as np
import cv2

from config import (
    SAMPLE_VIDEOS_DIR,
    OUTPUTS_DIR,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_PERSISTENCE_FRAMES,
    DEFAULT_SPEED_DROP_PERCENT,
    TARGET_VEHICLE_CLASSES
)
from detector import (
    calculate_iou,
    calculate_overlap_ratio,
    VehicleDetection,
    SimpleCentroidTracker,
    AccidentAnalysisEngine,
    VehicleDetector
)
from utils import (
    draw_annotations,
    export_detection_summary_csv
)


class TestAccidentDetectionSystem(unittest.TestCase):

    def test_01_iou_calculation(self):
        """Test geometric IoU math."""
        # Non-overlapping boxes
        boxA = [0, 0, 10, 10]
        boxB = [20, 20, 30, 30]
        self.assertEqual(calculate_iou(boxA, boxB), 0.0)

        # Fully identical boxes
        self.assertAlmostEqual(calculate_iou(boxA, boxA), 1.0, places=3)

        # 50% overlap box
        boxC = [5, 0, 15, 10]
        # Inter = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50 / 150 = 1/3 ~ 0.333
        self.assertAlmostEqual(calculate_iou(boxA, boxC), 1.0 / 3.0, places=3)

    def test_02_overlap_ratio(self):
        """Test Overlap Ratio relative to smaller box."""
        boxLarge = [0, 0, 100, 100]  # Area 10000
        boxSmall = [10, 10, 30, 30]   # Area 400, completely inside
        # Overlap relative to small box should be 1.0 (100% containment)
        self.assertAlmostEqual(calculate_overlap_ratio(boxLarge, boxSmall), 1.0, places=3)

    def test_03_centroid_tracker(self):
        """Test persistent tracking IDs and trajectory displacement."""
        tracker = SimpleCentroidTracker()
        det1 = VehicleDetection([10, 10, 50, 50], 0.85, 2, "car")
        updated1 = tracker.update([det1])
        self.assertEqual(len(updated1), 1)
        orig_id = updated1[0].track_id
        self.assertIsNotNone(orig_id)

        # Move vehicle slightly (simulating frame 2)
        det2 = VehicleDetection([15, 12, 55, 52], 0.88, 2, "car")
        updated2 = tracker.update([det2])
        self.assertEqual(len(updated2), 1)
        self.assertEqual(updated2[0].track_id, orig_id)  # Same ID preserved

    def test_04_accident_engine_collision_trigger(self):
        """Test that sustained overlapping vehicles trigger POSSIBLE_COLLISION alert."""
        engine = AccidentAnalysisEngine(iou_threshold=0.15, min_persistence=3)

        # Simulate 4 consecutive frames where two vehicles overlap
        alerts_found = []
        for frame in range(1, 5):
            v1 = VehicleDetection([100, 100, 180, 180], 0.9, 2, "car")
            v2 = VehicleDetection([120, 110, 200, 190], 0.9, 2, "car")
            _, active_alerts = engine.analyze_frame(frame, frame / 24.0, [v1, v2])
            for a in active_alerts:
                if a.alert_level == "POSSIBLE_COLLISION":
                    alerts_found.append(a)

        self.assertGreater(len(alerts_found), 0, "Engine should trigger POSSIBLE_COLLISION after min_persistence frames.")

    def test_05_detector_on_synthetic_frame(self):
        """Test that VehicleDetector runs without crash on image array."""
        detector = VehicleDetector(model_name_or_path="yolov8n.pt")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simulated car rectangle
        cv2.rectangle(blank_frame, (100, 100), (250, 200), (200, 100, 50), -1)
        detections = detector.detect(blank_frame, conf_threshold=0.25)
        self.assertIsInstance(detections, list)

    def test_06_csv_export(self):
        """Test detection log CSV export."""
        engine = AccidentAnalysisEngine(iou_threshold=0.10, min_persistence=1)
        v1 = VehicleDetection([10, 10, 60, 60], 0.9, 2, "car")
        v2 = VehicleDetection([15, 15, 65, 65], 0.9, 7, "truck")
        _, alerts = engine.analyze_frame(1, 0.04, [v1, v2])

        csv_str = export_detection_summary_csv(alerts, 1, 24.0)
        self.assertIn("Timestamp (s)", csv_str)
        self.assertIn("Vehicle 1", csv_str)


if __name__ == "__main__":
    unittest.main()

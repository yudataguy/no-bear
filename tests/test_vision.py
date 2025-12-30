"""Tests for vision modules."""

import pytest
import numpy as np

from drone_flight_core.vision.detector import (
    BoundingBox,
    Detection,
    DetectionResult,
    DetectionClass,
    ObjectDetector,
    SimpleHeatDetector,
    SimpleMotionDetector,
)
from drone_flight_core.vision.tracker import (
    TrackedObject,
    TrackingState,
    ObjectTracker,
)
from drone_flight_core.vision.fusion import (
    SensorFusion,
    FusedDetection,
)
from drone_flight_core.vision.camera_manager import CameraType, Frame
from drone_flight_core.core.config import VisionConfig


class TestBoundingBox:
    """Tests for BoundingBox."""

    def test_center(self):
        """Center should be calculated correctly."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        assert bbox.center == (5, 5)

    def test_area(self):
        """Area should be calculated correctly."""
        bbox = BoundingBox(x=0, y=0, width=10, height=20)
        assert bbox.area == 200

    def test_corners(self):
        """Corner coordinates should be correct."""
        bbox = BoundingBox(x=5, y=10, width=20, height=30)
        assert bbox.x2 == 25
        assert bbox.y2 == 40

    def test_iou_identical(self):
        """IoU of identical boxes should be 1."""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        bbox2 = BoundingBox(x=0, y=0, width=10, height=10)
        assert bbox1.iou(bbox2) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0."""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        bbox2 = BoundingBox(x=20, y=20, width=10, height=10)
        assert bbox1.iou(bbox2) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        """IoU of partially overlapping boxes should be correct."""
        bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
        bbox2 = BoundingBox(x=5, y=5, width=10, height=10)

        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        expected_iou = 25 / 175
        assert bbox1.iou(bbox2) == pytest.approx(expected_iou)

    def test_to_xyxy(self):
        """Conversion to xyxy format should be correct."""
        bbox = BoundingBox(x=10, y=20, width=30, height=40)
        assert bbox.to_xyxy() == (10, 20, 40, 60)

    def test_to_xywh(self):
        """Conversion to xywh format should be correct."""
        bbox = BoundingBox(x=10, y=20, width=30, height=40)
        assert bbox.to_xywh() == (10, 20, 30, 40)


class TestDetection:
    """Tests for Detection."""

    def test_creation(self):
        """Detection should be created with required fields."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        detection = Detection(
            bbox=bbox,
            confidence=0.9,
            class_id=DetectionClass.PERSON,
            class_name="person",
        )

        assert detection.confidence == 0.9
        assert detection.class_id == DetectionClass.PERSON

    def test_to_dict(self):
        """to_dict should return complete data."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        detection = Detection(
            bbox=bbox,
            confidence=0.9,
            class_id=DetectionClass.PERSON,
            class_name="person",
        )

        data = detection.to_dict()
        assert "bbox" in data
        assert "confidence" in data
        assert data["confidence"] == 0.9


class TestSimpleHeatDetector:
    """Tests for SimpleHeatDetector."""

    def test_detect_heat_signature(self):
        """Should detect bright regions as heat signatures."""
        detector = SimpleHeatDetector(threshold=128, min_area=50)

        # Create test image with bright spot
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255  # 20x20 bright region

        detections = detector.detect(image)

        assert len(detections) >= 1
        assert detections[0].class_id == DetectionClass.HEAT_SIGNATURE

    def test_no_detection_below_threshold(self):
        """Should not detect regions below threshold."""
        detector = SimpleHeatDetector(threshold=200, min_area=50)

        # Create low-intensity image
        image = np.full((100, 100), 100, dtype=np.uint8)

        detections = detector.detect(image)
        assert len(detections) == 0


class TestObjectTracker:
    """Tests for ObjectTracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker for testing."""
        config = VisionConfig()
        return ObjectTracker(config)

    def test_create_track(self, tracker):
        """New detection should create track."""
        bbox = BoundingBox(x=50, y=50, width=20, height=20)
        detection = Detection(
            bbox=bbox,
            confidence=0.9,
            class_id=DetectionClass.PERSON,
        )

        tracks = tracker.update([detection])

        assert len(tracks) == 1
        assert tracks[0].state == TrackingState.TENTATIVE

    def test_update_track(self, tracker):
        """Matched detection should update track."""
        bbox1 = BoundingBox(x=50, y=50, width=20, height=20)
        detection1 = Detection(bbox=bbox1, confidence=0.9)

        tracker.update([detection1])

        # Slightly moved detection
        bbox2 = BoundingBox(x=52, y=52, width=20, height=20)
        detection2 = Detection(bbox=bbox2, confidence=0.9)

        tracks = tracker.update([detection2])

        assert len(tracks) == 1
        assert tracks[0].hits == 2

    def test_confirm_track(self, tracker):
        """Track should be confirmed after enough hits."""
        for _ in range(5):
            bbox = BoundingBox(x=50, y=50, width=20, height=20)
            detection = Detection(bbox=bbox, confidence=0.9)
            tracker.update([detection])

        confirmed = tracker.confirmed_tracks
        assert len(confirmed) >= 1
        assert confirmed[0].state == TrackingState.CONFIRMED

    def test_track_lost(self, tracker):
        """Track should be lost without detections."""
        bbox = BoundingBox(x=50, y=50, width=20, height=20)
        detection = Detection(bbox=bbox, confidence=0.9)

        # Create and confirm track
        for _ in range(5):
            tracker.update([detection])

        # Update without detection many times
        for _ in range(15):
            tracker.update([])

        tracks = tracker.tracks
        lost_tracks = [t for t in tracks if t.state == TrackingState.LOST]
        assert len(lost_tracks) >= 1 or len(tracks) == 0

    def test_primary_target(self, tracker):
        """Primary target should be settable."""
        bbox = BoundingBox(x=50, y=50, width=20, height=20)
        detection = Detection(bbox=bbox, confidence=0.9)

        tracker.update([detection])
        tracks = tracker.tracks

        if tracks:
            tracker.set_primary_target(tracks[0].track_id)
            assert tracker.primary_target is not None


class TestTrackedObject:
    """Tests for TrackedObject."""

    def test_velocity_calculation(self):
        """Velocity should be calculated from history."""
        bbox1 = BoundingBox(x=50, y=50, width=20, height=20)
        track = TrackedObject(track_id=1, bbox=bbox1)

        bbox2 = BoundingBox(x=55, y=55, width=20, height=20)
        detection = Detection(bbox=bbox2, confidence=0.9)
        track.update(detection)

        # Velocity should be (5, 5) - movement from center (60,60) to (65,65)
        assert track.velocity == (5, 5)

    def test_predict(self):
        """Prediction should use velocity."""
        bbox = BoundingBox(x=50, y=50, width=20, height=20)
        track = TrackedObject(track_id=1, bbox=bbox)
        track.velocity = (10, 5)

        prediction = track.predict()
        predicted_center = prediction.position

        # Original center is (60, 60), predicted should be (70, 65)
        assert predicted_center == (70, 65)


class TestSensorFusion:
    """Tests for SensorFusion."""

    @pytest.fixture
    def fusion(self):
        """Create fusion module for testing."""
        config = VisionConfig(fusion_mode="weighted", fusion_weight_thermal=0.5)
        return SensorFusion(config)

    def test_fuse_matching_detections(self, fusion):
        """Matching detections should be fused."""
        rgb_bbox = BoundingBox(x=50, y=50, width=20, height=20)
        rgb_det = Detection(
            bbox=rgb_bbox,
            confidence=0.8,
            class_id=DetectionClass.PERSON,
            class_name="person",
            source=CameraType.RGB,
        )
        rgb_result = DetectionResult(
            detections=[rgb_det],
            frame_number=1,
            processing_time_ms=10,
            source=CameraType.RGB,
        )

        # Thermal detection at same location (scaled)
        thermal_bbox = BoundingBox(x=25, y=25, width=10, height=10)
        thermal_det = Detection(
            bbox=thermal_bbox,
            confidence=0.9,
            class_id=DetectionClass.HEAT_SIGNATURE,
            source=CameraType.THERMAL,
        )
        thermal_result = DetectionResult(
            detections=[thermal_det],
            frame_number=1,
            processing_time_ms=5,
            source=CameraType.THERMAL,
        )

        fused = fusion.fuse(rgb_result, thermal_result)

        assert len(fused) >= 1

    def test_fuse_rgb_only(self, fusion):
        """RGB-only detections should be included."""
        rgb_bbox = BoundingBox(x=50, y=50, width=20, height=20)
        rgb_det = Detection(
            bbox=rgb_bbox,
            confidence=0.8,
            source=CameraType.RGB,
        )
        rgb_result = DetectionResult(
            detections=[rgb_det],
            frame_number=1,
            processing_time_ms=10,
            source=CameraType.RGB,
        )

        fused = fusion.fuse(rgb_result, None)

        assert len(fused) == 1
        assert fused[0].has_rgb
        assert not fused[0].has_thermal

    def test_fuse_thermal_only(self, fusion):
        """Thermal-only detections should be included."""
        thermal_bbox = BoundingBox(x=50, y=50, width=20, height=20)
        thermal_det = Detection(
            bbox=thermal_bbox,
            confidence=0.9,
            source=CameraType.THERMAL,
        )
        thermal_result = DetectionResult(
            detections=[thermal_det],
            frame_number=1,
            processing_time_ms=5,
            source=CameraType.THERMAL,
        )

        fused = fusion.fuse(None, thermal_result)

        assert len(fused) == 1
        assert fused[0].has_thermal
        assert not fused[0].has_rgb

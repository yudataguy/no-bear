"""
Object detection for RGB and thermal imagery.

Provides detection interface with support for various
backend detection models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
import structlog

import numpy as np
import cv2

from drone_flight_core.vision.camera_manager import Frame, CameraType
from drone_flight_core.core.config import VisionConfig

logger = structlog.get_logger(__name__)


class DetectionClass(Enum):
    """Object detection classes."""

    UNKNOWN = auto()
    PERSON = auto()
    VEHICLE = auto()
    ANIMAL = auto()
    AIRCRAFT = auto()
    BOAT = auto()
    BUILDING = auto()
    HEAT_SIGNATURE = auto()  # Thermal-specific


@dataclass
class BoundingBox:
    """Bounding box for detected object."""

    x: int  # Top-left x
    y: int  # Top-left y
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        """Center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of bounding box in pixels."""
        return self.width * self.height

    @property
    def x2(self) -> int:
        """Bottom-right x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom-right y coordinate."""
        return self.y + self.height

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x2, self.y2)

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Convert to (x, y, width, height) format."""
        return (self.x, self.y, self.width, self.height)

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    """Single object detection result."""

    bbox: BoundingBox
    confidence: float
    class_id: DetectionClass = DetectionClass.UNKNOWN
    class_name: str = ""
    source: CameraType = CameraType.RGB
    timestamp: datetime = field(default_factory=datetime.now)
    features: np.ndarray | None = None  # For re-identification
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox.to_xywh(),
            "confidence": self.confidence,
            "class_id": self.class_id.name,
            "class_name": self.class_name,
            "source": self.source.name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DetectionResult:
    """Complete detection result for a frame."""

    detections: list[Detection]
    frame_number: int
    processing_time_ms: float
    source: CameraType
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def count(self) -> int:
        """Number of detections."""
        return len(self.detections)

    def filter_by_confidence(self, min_confidence: float) -> list[Detection]:
        """Get detections above confidence threshold."""
        return [d for d in self.detections if d.confidence >= min_confidence]

    def filter_by_class(self, class_id: DetectionClass) -> list[Detection]:
        """Get detections of specific class."""
        return [d for d in self.detections if d.class_id == class_id]


class DetectorBackend(ABC):
    """Abstract base class for detection backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Perform object detection on frame.

        Args:
            frame: Input image as numpy array.

        Returns:
            List of detections.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load detection model.

        Args:
            model_path: Path to model file.

        Returns:
            True if loaded successfully.
        """
        pass


class SimpleHeatDetector(DetectorBackend):
    """
    Simple thermal/heat signature detector.

    Uses thresholding and contour detection for thermal imagery.
    Suitable for basic heat signature detection.
    """

    def __init__(
        self,
        threshold: int = 200,
        min_area: int = 100,
        max_area: int = 50000,
    ) -> None:
        self._threshold = threshold
        self._min_area = min_area
        self._max_area = max_area

    def load_model(self, model_path: str) -> bool:
        """No model needed for this detector."""
        return True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect heat signatures in thermal image."""
        detections = []

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply threshold
        _, binary = cv2.threshold(gray, self._threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if self._min_area <= area <= self._max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate confidence based on intensity
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]
                confidence = mean_intensity / 255.0

                detection = Detection(
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    confidence=confidence,
                    class_id=DetectionClass.HEAT_SIGNATURE,
                    class_name="heat_signature",
                    source=CameraType.THERMAL,
                    metadata={"area": area, "mean_intensity": mean_intensity},
                )
                detections.append(detection)

        return detections


class SimpleMotionDetector(DetectorBackend):
    """
    Simple motion-based detector.

    Uses frame differencing for motion detection.
    Useful as a pre-filter before more expensive detection.
    """

    def __init__(
        self,
        threshold: int = 25,
        min_area: int = 500,
    ) -> None:
        self._threshold = threshold
        self._min_area = min_area
        self._prev_frame: np.ndarray | None = None
        self._background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )

    def load_model(self, model_path: str) -> bool:
        """No model needed for this detector."""
        return True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect motion in frame."""
        detections = []

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply background subtraction
        fg_mask = self._background_subtractor.apply(gray)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self._min_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Confidence based on area relative to min threshold
                confidence = min(1.0, area / (self._min_area * 10))

                detection = Detection(
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    confidence=confidence,
                    class_id=DetectionClass.UNKNOWN,
                    class_name="motion",
                    source=CameraType.RGB,
                    metadata={"area": area},
                )
                detections.append(detection)

        return detections


class ObjectDetector:
    """
    Main object detector combining multiple detection backends.

    Supports pluggable detection backends for different use cases.
    """

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._rgb_detector: DetectorBackend | None = None
        self._thermal_detector: DetectorBackend | None = None
        self._detection_count = 0
        self._total_processing_time = 0.0

        # Initialize default detectors
        self._init_default_detectors()

        logger.info("Object detector initialized")

    def _init_default_detectors(self) -> None:
        """Initialize default detection backends."""
        # Motion detector for RGB
        self._rgb_detector = SimpleMotionDetector()
        self._rgb_detector.load_model("")

        # Heat detector for thermal
        self._thermal_detector = SimpleHeatDetector()
        self._thermal_detector.load_model("")

    def set_rgb_detector(self, detector: DetectorBackend) -> None:
        """Set custom RGB detector backend."""
        self._rgb_detector = detector

    def set_thermal_detector(self, detector: DetectorBackend) -> None:
        """Set custom thermal detector backend."""
        self._thermal_detector = detector

    def detect_rgb(self, frame: Frame) -> DetectionResult:
        """
        Run detection on RGB frame.

        Args:
            frame: RGB camera frame.

        Returns:
            Detection result with all detections.
        """
        import time
        start_time = time.time()

        detections = []
        if self._rgb_detector:
            raw_detections = self._rgb_detector.detect(frame.data)

            # Filter by confidence
            detections = [
                d for d in raw_detections
                if d.confidence >= self._config.detection_confidence_threshold
            ]

            # Set source
            for d in detections:
                d.source = CameraType.RGB

        processing_time = (time.time() - start_time) * 1000
        self._detection_count += len(detections)
        self._total_processing_time += processing_time

        return DetectionResult(
            detections=detections,
            frame_number=frame.frame_number,
            processing_time_ms=processing_time,
            source=CameraType.RGB,
        )

    def detect_thermal(self, frame: Frame) -> DetectionResult:
        """
        Run detection on thermal frame.

        Args:
            frame: Thermal camera frame.

        Returns:
            Detection result with all detections.
        """
        import time
        start_time = time.time()

        detections = []
        if self._thermal_detector:
            raw_detections = self._thermal_detector.detect(frame.data)

            # Filter by confidence
            detections = [
                d for d in raw_detections
                if d.confidence >= self._config.detection_confidence_threshold
            ]

            # Set source
            for d in detections:
                d.source = CameraType.THERMAL

        processing_time = (time.time() - start_time) * 1000
        self._detection_count += len(detections)
        self._total_processing_time += processing_time

        return DetectionResult(
            detections=detections,
            frame_number=frame.frame_number,
            processing_time_ms=processing_time,
            source=CameraType.THERMAL,
        )

    def detect_both(
        self, rgb_frame: Frame | None, thermal_frame: Frame | None
    ) -> tuple[DetectionResult | None, DetectionResult | None]:
        """
        Run detection on both camera frames.

        Args:
            rgb_frame: RGB camera frame (optional).
            thermal_frame: Thermal camera frame (optional).

        Returns:
            Tuple of (rgb_result, thermal_result).
        """
        rgb_result = None
        thermal_result = None

        if rgb_frame is not None:
            rgb_result = self.detect_rgb(rgb_frame)

        if thermal_frame is not None:
            thermal_result = self.detect_thermal(thermal_frame)

        return rgb_result, thermal_result

    @property
    def average_processing_time(self) -> float:
        """Average processing time in milliseconds."""
        if self._detection_count == 0:
            return 0.0
        return self._total_processing_time / self._detection_count

    def to_dict(self) -> dict[str, Any]:
        """Export detector status."""
        return {
            "detection_count": self._detection_count,
            "average_processing_time_ms": self.average_processing_time,
            "confidence_threshold": self._config.detection_confidence_threshold,
            "has_rgb_detector": self._rgb_detector is not None,
            "has_thermal_detector": self._thermal_detector is not None,
        }

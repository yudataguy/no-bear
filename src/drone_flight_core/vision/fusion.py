"""
Sensor fusion for combining RGB and thermal detections.

Fuses detections from multiple cameras to improve detection
accuracy and reduce false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import structlog

import numpy as np

from drone_flight_core.vision.detector import Detection, DetectionResult, BoundingBox, DetectionClass
from drone_flight_core.vision.camera_manager import CameraType
from drone_flight_core.core.config import VisionConfig

logger = structlog.get_logger(__name__)


@dataclass
class FusedDetection:
    """Detection result from fusing RGB and thermal data."""

    bbox: BoundingBox
    confidence: float
    class_id: DetectionClass
    class_name: str

    # Source detections
    rgb_detection: Detection | None = None
    thermal_detection: Detection | None = None

    # Fusion metadata
    fusion_type: str = "single"  # single, matched, thermal_only, rgb_only
    thermal_confidence: float = 0.0
    rgb_confidence: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_thermal(self) -> bool:
        """Check if thermal detection contributed."""
        return self.thermal_detection is not None

    @property
    def has_rgb(self) -> bool:
        """Check if RGB detection contributed."""
        return self.rgb_detection is not None

    @property
    def is_fused(self) -> bool:
        """Check if detection is from multiple sources."""
        return self.has_thermal and self.has_rgb

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox.to_xywh(),
            "confidence": self.confidence,
            "class_id": self.class_id.name,
            "class_name": self.class_name,
            "fusion_type": self.fusion_type,
            "has_thermal": self.has_thermal,
            "has_rgb": self.has_rgb,
            "thermal_confidence": self.thermal_confidence,
            "rgb_confidence": self.rgb_confidence,
        }


class SensorFusion:
    """
    Fuses detections from RGB and thermal cameras.

    Strategies:
    - thermal_priority: Prefer thermal detections, use RGB for classification
    - rgb_priority: Prefer RGB detections, use thermal for validation
    - weighted: Combine both based on configured weights
    """

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._fusion_count = 0

        # Calibration for aligning thermal and RGB
        # These should be calibrated for actual camera setup
        self._thermal_to_rgb_scale = (
            config.rgb_resolution[0] / config.thermal_resolution[0],
            config.rgb_resolution[1] / config.thermal_resolution[1],
        )
        self._thermal_to_rgb_offset = (0, 0)

        logger.info(
            "Sensor fusion initialized",
            mode=config.fusion_mode,
            thermal_weight=config.fusion_weight_thermal,
        )

    def fuse(
        self,
        rgb_result: DetectionResult | None,
        thermal_result: DetectionResult | None,
    ) -> list[FusedDetection]:
        """
        Fuse detections from RGB and thermal cameras.

        Args:
            rgb_result: Detection result from RGB camera.
            thermal_result: Detection result from thermal camera.

        Returns:
            List of fused detections.
        """
        if rgb_result is None and thermal_result is None:
            return []

        # Get detections
        rgb_detections = rgb_result.detections if rgb_result else []
        thermal_detections = thermal_result.detections if thermal_result else []

        # Transform thermal detections to RGB coordinate space
        aligned_thermal = [
            self._align_thermal_to_rgb(d) for d in thermal_detections
        ]

        # Fuse based on mode
        if self._config.fusion_mode == "thermal_priority":
            fused = self._fuse_thermal_priority(rgb_detections, aligned_thermal)
        elif self._config.fusion_mode == "rgb_priority":
            fused = self._fuse_rgb_priority(rgb_detections, aligned_thermal)
        else:  # weighted
            fused = self._fuse_weighted(rgb_detections, aligned_thermal)

        self._fusion_count += len(fused)
        return fused

    def _align_thermal_to_rgb(self, detection: Detection) -> Detection:
        """
        Transform thermal detection to RGB coordinate space.

        This is a simplified alignment - real implementation would
        use proper camera calibration matrices.
        """
        bbox = detection.bbox
        scale_x, scale_y = self._thermal_to_rgb_scale
        offset_x, offset_y = self._thermal_to_rgb_offset

        aligned_bbox = BoundingBox(
            x=int(bbox.x * scale_x + offset_x),
            y=int(bbox.y * scale_y + offset_y),
            width=int(bbox.width * scale_x),
            height=int(bbox.height * scale_y),
        )

        # Create new detection with aligned bbox
        return Detection(
            bbox=aligned_bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            source=detection.source,
            timestamp=detection.timestamp,
            features=detection.features,
            metadata=detection.metadata,
        )

    def _fuse_thermal_priority(
        self,
        rgb_detections: list[Detection],
        thermal_detections: list[Detection],
    ) -> list[FusedDetection]:
        """
        Fuse with thermal priority.

        Thermal detections are primary; RGB provides classification.
        """
        fused = []
        matched_rgb = set()

        for thermal in thermal_detections:
            # Find matching RGB detection
            best_match = None
            best_iou = 0.3  # Minimum IOU for match

            for i, rgb in enumerate(rgb_detections):
                if i in matched_rgb:
                    continue

                iou = thermal.bbox.iou(rgb.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (i, rgb)

            if best_match:
                idx, rgb = best_match
                matched_rgb.add(idx)

                # Use thermal bbox but RGB classification
                fused_det = FusedDetection(
                    bbox=thermal.bbox,
                    confidence=max(thermal.confidence, rgb.confidence),
                    class_id=rgb.class_id if rgb.class_id != DetectionClass.UNKNOWN else thermal.class_id,
                    class_name=rgb.class_name or thermal.class_name,
                    rgb_detection=rgb,
                    thermal_detection=thermal,
                    fusion_type="matched",
                    thermal_confidence=thermal.confidence,
                    rgb_confidence=rgb.confidence,
                )
            else:
                # Thermal only
                fused_det = FusedDetection(
                    bbox=thermal.bbox,
                    confidence=thermal.confidence,
                    class_id=thermal.class_id,
                    class_name=thermal.class_name,
                    thermal_detection=thermal,
                    fusion_type="thermal_only",
                    thermal_confidence=thermal.confidence,
                )

            fused.append(fused_det)

        return fused

    def _fuse_rgb_priority(
        self,
        rgb_detections: list[Detection],
        thermal_detections: list[Detection],
    ) -> list[FusedDetection]:
        """
        Fuse with RGB priority.

        RGB detections are primary; thermal validates/boosts confidence.
        """
        fused = []
        matched_thermal = set()

        for rgb in rgb_detections:
            # Find matching thermal detection
            best_match = None
            best_iou = 0.3

            for i, thermal in enumerate(thermal_detections):
                if i in matched_thermal:
                    continue

                iou = rgb.bbox.iou(thermal.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (i, thermal)

            if best_match:
                idx, thermal = best_match
                matched_thermal.add(idx)

                # Boost confidence if thermal confirms
                boosted_confidence = min(1.0, rgb.confidence + thermal.confidence * 0.3)

                fused_det = FusedDetection(
                    bbox=rgb.bbox,
                    confidence=boosted_confidence,
                    class_id=rgb.class_id,
                    class_name=rgb.class_name,
                    rgb_detection=rgb,
                    thermal_detection=thermal,
                    fusion_type="matched",
                    thermal_confidence=thermal.confidence,
                    rgb_confidence=rgb.confidence,
                )
            else:
                # RGB only - lower confidence without thermal confirmation
                fused_det = FusedDetection(
                    bbox=rgb.bbox,
                    confidence=rgb.confidence * 0.8,
                    class_id=rgb.class_id,
                    class_name=rgb.class_name,
                    rgb_detection=rgb,
                    fusion_type="rgb_only",
                    rgb_confidence=rgb.confidence,
                )

            fused.append(fused_det)

        # Add unmatched thermal detections
        for i, thermal in enumerate(thermal_detections):
            if i not in matched_thermal:
                fused.append(FusedDetection(
                    bbox=thermal.bbox,
                    confidence=thermal.confidence * 0.9,
                    class_id=thermal.class_id,
                    class_name=thermal.class_name,
                    thermal_detection=thermal,
                    fusion_type="thermal_only",
                    thermal_confidence=thermal.confidence,
                ))

        return fused

    def _fuse_weighted(
        self,
        rgb_detections: list[Detection],
        thermal_detections: list[Detection],
    ) -> list[FusedDetection]:
        """
        Fuse with weighted combination.

        Both sources contribute based on configured weights.
        """
        fused = []
        thermal_weight = self._config.fusion_weight_thermal
        rgb_weight = 1.0 - thermal_weight

        matched_rgb = set()
        matched_thermal = set()

        # Find all matching pairs
        matches = []
        for i, rgb in enumerate(rgb_detections):
            for j, thermal in enumerate(thermal_detections):
                iou = rgb.bbox.iou(thermal.bbox)
                if iou > 0.3:
                    matches.append((i, j, iou))

        # Sort by IOU and process best matches first
        matches.sort(key=lambda x: x[2], reverse=True)

        for rgb_idx, thermal_idx, iou in matches:
            if rgb_idx in matched_rgb or thermal_idx in matched_thermal:
                continue

            matched_rgb.add(rgb_idx)
            matched_thermal.add(thermal_idx)

            rgb = rgb_detections[rgb_idx]
            thermal = thermal_detections[thermal_idx]

            # Weighted average of bounding boxes
            fused_bbox = BoundingBox(
                x=int(rgb.bbox.x * rgb_weight + thermal.bbox.x * thermal_weight),
                y=int(rgb.bbox.y * rgb_weight + thermal.bbox.y * thermal_weight),
                width=int(rgb.bbox.width * rgb_weight + thermal.bbox.width * thermal_weight),
                height=int(rgb.bbox.height * rgb_weight + thermal.bbox.height * thermal_weight),
            )

            # Weighted confidence
            fused_confidence = (
                rgb.confidence * rgb_weight + thermal.confidence * thermal_weight
            )

            # Prefer RGB classification if available
            class_id = rgb.class_id if rgb.class_id != DetectionClass.UNKNOWN else thermal.class_id
            class_name = rgb.class_name or thermal.class_name

            fused.append(FusedDetection(
                bbox=fused_bbox,
                confidence=fused_confidence,
                class_id=class_id,
                class_name=class_name,
                rgb_detection=rgb,
                thermal_detection=thermal,
                fusion_type="matched",
                thermal_confidence=thermal.confidence,
                rgb_confidence=rgb.confidence,
            ))

        # Add unmatched RGB
        for i, rgb in enumerate(rgb_detections):
            if i not in matched_rgb:
                fused.append(FusedDetection(
                    bbox=rgb.bbox,
                    confidence=rgb.confidence * rgb_weight,
                    class_id=rgb.class_id,
                    class_name=rgb.class_name,
                    rgb_detection=rgb,
                    fusion_type="rgb_only",
                    rgb_confidence=rgb.confidence,
                ))

        # Add unmatched thermal
        for i, thermal in enumerate(thermal_detections):
            if i not in matched_thermal:
                fused.append(FusedDetection(
                    bbox=thermal.bbox,
                    confidence=thermal.confidence * thermal_weight,
                    class_id=thermal.class_id,
                    class_name=thermal.class_name,
                    thermal_detection=thermal,
                    fusion_type="thermal_only",
                    thermal_confidence=thermal.confidence,
                ))

        return fused

    def set_calibration(
        self,
        scale: tuple[float, float],
        offset: tuple[int, int],
    ) -> None:
        """
        Set thermal-to-RGB calibration parameters.

        Args:
            scale: (scale_x, scale_y) to transform thermal to RGB.
            offset: (offset_x, offset_y) in RGB pixels.
        """
        self._thermal_to_rgb_scale = scale
        self._thermal_to_rgb_offset = offset
        logger.info("Calibration updated", scale=scale, offset=offset)

    def to_dict(self) -> dict[str, Any]:
        """Export fusion status."""
        return {
            "fusion_mode": self._config.fusion_mode,
            "thermal_weight": self._config.fusion_weight_thermal,
            "total_fused": self._fusion_count,
            "calibration": {
                "scale": self._thermal_to_rgb_scale,
                "offset": self._thermal_to_rgb_offset,
            },
        }

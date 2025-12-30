"""
Object tracking for continuous target following.

Maintains track state across frames and handles track
lifecycle (creation, update, deletion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
import structlog

import numpy as np

from drone_flight_core.vision.detector import Detection, BoundingBox, DetectionClass
from drone_flight_core.vision.camera_manager import CameraType
from drone_flight_core.core.config import VisionConfig

logger = structlog.get_logger(__name__)


class TrackingState(Enum):
    """State of a tracked object."""

    TENTATIVE = auto()  # New track, not yet confirmed
    CONFIRMED = auto()  # Track confirmed and active
    LOST = auto()  # Track temporarily lost
    DELETED = auto()  # Track deleted


@dataclass
class TrackPrediction:
    """Predicted state of tracked object."""

    position: tuple[float, float]  # Predicted center (x, y)
    velocity: tuple[float, float]  # Velocity (vx, vy) pixels/frame
    bbox: BoundingBox
    confidence: float


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""

    track_id: int
    bbox: BoundingBox
    state: TrackingState = TrackingState.TENTATIVE
    class_id: DetectionClass = DetectionClass.UNKNOWN
    class_name: str = ""

    # Track history
    history: list[BoundingBox] = field(default_factory=list)
    max_history: int = 100

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    frames_since_seen: int = 0

    # Confirmation
    hits: int = 0
    age: int = 0
    time_since_update: int = 0

    # Motion estimation
    velocity: tuple[float, float] = (0.0, 0.0)

    # Features for re-identification
    features: np.ndarray | None = None

    # Source camera
    source: CameraType = CameraType.RGB

    # Confidence
    confidence: float = 0.0

    def update(self, detection: Detection) -> None:
        """Update track with new detection."""
        # Store history
        self.history.append(self.bbox)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Calculate velocity
        if len(self.history) >= 2:
            prev_center = self.history[-1].center
            curr_center = detection.bbox.center
            self.velocity = (
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1],
            )

        # Update state
        self.bbox = detection.bbox
        self.last_seen = datetime.now()
        self.frames_since_seen = 0
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.confidence = detection.confidence

        if detection.features is not None:
            self.features = detection.features

        # Confirm track after enough hits
        if self.state == TrackingState.TENTATIVE and self.hits >= 3:
            self.state = TrackingState.CONFIRMED
            logger.debug("Track confirmed", track_id=self.track_id)

    def predict(self) -> TrackPrediction:
        """Predict next state based on motion model."""
        # Simple linear prediction
        center = self.bbox.center
        predicted_center = (
            center[0] + self.velocity[0],
            center[1] + self.velocity[1],
        )

        predicted_bbox = BoundingBox(
            x=int(predicted_center[0] - self.bbox.width / 2),
            y=int(predicted_center[1] - self.bbox.height / 2),
            width=self.bbox.width,
            height=self.bbox.height,
        )

        # Confidence decreases with time since update
        confidence = max(0.1, self.confidence * (0.9 ** self.time_since_update))

        return TrackPrediction(
            position=predicted_center,
            velocity=self.velocity,
            bbox=predicted_bbox,
            confidence=confidence,
        )

    def mark_missed(self) -> None:
        """Mark track as missed in current frame."""
        self.frames_since_seen += 1
        self.age += 1
        self.time_since_update += 1

        if self.state == TrackingState.CONFIRMED and self.frames_since_seen > 10:
            self.state = TrackingState.LOST
            logger.debug("Track lost", track_id=self.track_id)

    @property
    def time_alive_seconds(self) -> float:
        """Time since track was created in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def time_since_seen_seconds(self) -> float:
        """Time since track was last seen in seconds."""
        return (datetime.now() - self.last_seen).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_xywh(),
            "state": self.state.name,
            "class_id": self.class_id.name,
            "confidence": self.confidence,
            "velocity": self.velocity,
            "hits": self.hits,
            "age": self.age,
            "time_alive_seconds": self.time_alive_seconds,
            "time_since_seen_seconds": self.time_since_seen_seconds,
        }


class ObjectTracker:
    """
    Multi-object tracker using simple IOU-based association.

    Maintains a set of active tracks and associates new
    detections with existing tracks based on overlap.
    """

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._tracks: dict[int, TrackedObject] = {}
        self._next_track_id = 1
        self._deleted_tracks: list[TrackedObject] = []

        # Tracking parameters
        self._iou_threshold = 0.3
        self._max_age = 30  # Frames before track is deleted
        self._min_hits = 3  # Hits before track is confirmed

        # Primary target
        self._primary_target_id: int | None = None

        logger.info("Object tracker initialized")

    @property
    def tracks(self) -> list[TrackedObject]:
        """Get all active tracks."""
        return list(self._tracks.values())

    @property
    def confirmed_tracks(self) -> list[TrackedObject]:
        """Get confirmed tracks only."""
        return [t for t in self._tracks.values() if t.state == TrackingState.CONFIRMED]

    @property
    def primary_target(self) -> TrackedObject | None:
        """Get the primary tracking target."""
        if self._primary_target_id is not None:
            return self._tracks.get(self._primary_target_id)
        return None

    def set_primary_target(self, track_id: int) -> bool:
        """Set the primary tracking target."""
        if track_id in self._tracks:
            self._primary_target_id = track_id
            logger.info("Primary target set", track_id=track_id)
            return True
        return False

    def clear_primary_target(self) -> None:
        """Clear the primary tracking target."""
        self._primary_target_id = None

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of new detections.

        Returns:
            List of updated tracks.
        """
        # Predict new locations for all tracks
        predictions = {
            track_id: track.predict()
            for track_id, track in self._tracks.items()
        }

        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate(
            detections, predictions
        )

        # Update matched tracks
        for track_id, detection_idx in matches:
            self._tracks[track_id].update(detections[detection_idx])

        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self._tracks[track_id].mark_missed()

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            self._create_track(detection)

        # Delete old tracks
        self._cleanup_tracks()

        return self.tracks

    def _associate(
        self,
        detections: list[Detection],
        predictions: dict[int, TrackPrediction],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Associate detections with predicted track positions.

        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if not detections or not predictions:
            return [], list(range(len(detections))), list(predictions.keys())

        # Build IOU matrix
        track_ids = list(predictions.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, track_id in enumerate(track_ids):
            pred_bbox = predictions[track_id].bbox
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = pred_bbox.iou(detection.bbox)

        # Greedy matching (could use Hungarian algorithm for optimal matching)
        matches = []
        matched_tracks = set()
        matched_detections = set()

        while True:
            # Find best match
            if iou_matrix.size == 0:
                break

            max_iou = np.max(iou_matrix)
            if max_iou < self._iou_threshold:
                break

            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = max_idx

            track_id = track_ids[track_idx]
            matches.append((track_id, det_idx))
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)

            # Remove matched row/column
            iou_matrix[track_idx, :] = -1
            iou_matrix[:, det_idx] = -1

        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        unmatched_tracks = [t for t in track_ids if t not in matched_tracks]

        return matches, unmatched_detections, unmatched_tracks

    def _create_track(self, detection: Detection) -> TrackedObject:
        """Create new track from detection."""
        track = TrackedObject(
            track_id=self._next_track_id,
            bbox=detection.bbox,
            class_id=detection.class_id,
            class_name=detection.class_name,
            source=detection.source,
            confidence=detection.confidence,
            features=detection.features,
        )

        self._tracks[self._next_track_id] = track
        self._next_track_id += 1

        logger.debug("New track created", track_id=track.track_id)
        return track

    def _cleanup_tracks(self) -> None:
        """Remove old/lost tracks."""
        tracks_to_delete = []

        for track_id, track in self._tracks.items():
            # Delete tentative tracks that haven't been confirmed
            if track.state == TrackingState.TENTATIVE and track.time_since_update > self._max_age // 2:
                tracks_to_delete.append(track_id)

            # Delete lost tracks after max_age
            elif track.time_since_update > self._max_age:
                tracks_to_delete.append(track_id)

        for track_id in tracks_to_delete:
            track = self._tracks.pop(track_id)
            track.state = TrackingState.DELETED
            self._deleted_tracks.append(track)

            # Clear primary target if deleted
            if self._primary_target_id == track_id:
                self._primary_target_id = None

            logger.debug("Track deleted", track_id=track_id)

        # Limit deleted track history
        if len(self._deleted_tracks) > 1000:
            self._deleted_tracks = self._deleted_tracks[-1000:]

    def get_track(self, track_id: int) -> TrackedObject | None:
        """Get track by ID."""
        return self._tracks.get(track_id)

    def delete_track(self, track_id: int) -> bool:
        """Manually delete a track."""
        if track_id in self._tracks:
            track = self._tracks.pop(track_id)
            track.state = TrackingState.DELETED
            self._deleted_tracks.append(track)

            if self._primary_target_id == track_id:
                self._primary_target_id = None

            return True
        return False

    def clear_all(self) -> None:
        """Clear all tracks."""
        for track in self._tracks.values():
            track.state = TrackingState.DELETED
            self._deleted_tracks.append(track)

        self._tracks.clear()
        self._primary_target_id = None
        logger.info("All tracks cleared")

    def get_closest_to_center(
        self, frame_width: int, frame_height: int
    ) -> TrackedObject | None:
        """Get the track closest to frame center."""
        if not self.confirmed_tracks:
            return None

        frame_center = (frame_width // 2, frame_height // 2)

        def distance_to_center(track: TrackedObject) -> float:
            center = track.bbox.center
            return ((center[0] - frame_center[0]) ** 2 +
                    (center[1] - frame_center[1]) ** 2) ** 0.5

        return min(self.confirmed_tracks, key=distance_to_center)

    def get_largest(self) -> TrackedObject | None:
        """Get the track with largest bounding box."""
        if not self.confirmed_tracks:
            return None

        return max(self.confirmed_tracks, key=lambda t: t.bbox.area)

    def to_dict(self) -> dict[str, Any]:
        """Export tracker status."""
        return {
            "active_tracks": len(self._tracks),
            "confirmed_tracks": len(self.confirmed_tracks),
            "primary_target_id": self._primary_target_id,
            "tracks": [t.to_dict() for t in self.tracks],
        }

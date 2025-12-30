"""
Flight data recorder for comprehensive mission logging.

Records all flight data including telemetry, video frames,
detections, and events for post-flight analysis.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import threading
import structlog

import numpy as np
import cv2

from drone_flight_core.core.config import LoggingConfig
from drone_flight_core.communication.telemetry import TelemetryData
from drone_flight_core.vision.camera_manager import Frame
from drone_flight_core.vision.tracker import TrackedObject

logger = structlog.get_logger(__name__)


@dataclass
class FlightRecord:
    """Metadata for a recorded flight."""

    flight_id: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # Counts
    telemetry_samples: int = 0
    frames_recorded: int = 0
    events_recorded: int = 0
    detections_recorded: int = 0

    # Flight summary
    max_altitude_m: float = 0.0
    max_speed_ms: float = 0.0
    distance_traveled_m: float = 0.0
    home_position: tuple[float, float] | None = None

    # Storage paths
    base_path: Path | None = None
    telemetry_file: Path | None = None
    events_file: Path | None = None
    video_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flight_id": self.flight_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "telemetry_samples": self.telemetry_samples,
            "frames_recorded": self.frames_recorded,
            "events_recorded": self.events_recorded,
            "detections_recorded": self.detections_recorded,
            "max_altitude_m": self.max_altitude_m,
            "max_speed_ms": self.max_speed_ms,
            "distance_traveled_m": self.distance_traveled_m,
            "home_position": self.home_position,
        }

    def save_metadata(self) -> None:
        """Save flight metadata to file."""
        if self.base_path:
            metadata_file = self.base_path / "flight_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)


class FlightRecorder:
    """
    Records comprehensive flight data.

    Features:
    - Telemetry logging at configurable rate
    - Video frame capture (optional)
    - Event logging
    - Detection logging
    - Automatic file management
    """

    def __init__(
        self,
        config: LoggingConfig,
        flight_id: str | None = None,
    ) -> None:
        self._config = config
        self._flight_id = flight_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create recording directory
        self._base_path = Path(config.log_dir) / "flights" / self._flight_id
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Initialize record
        self._record = FlightRecord(
            flight_id=self._flight_id,
            start_time=datetime.now(),
            base_path=self._base_path,
        )

        # File handles
        self._telemetry_file: Path | None = None
        self._telemetry_handle = None
        self._events_file: Path | None = None
        self._events_handle = None

        # Video writer
        self._video_writer: cv2.VideoWriter | None = None
        self._video_file: Path | None = None

        # Threading
        self._write_lock = threading.Lock()
        self._running = False

        # Rate limiting for telemetry
        self._telemetry_interval = 1.0 / config.telemetry_log_rate_hz
        self._last_telemetry_time = 0.0

        # Previous position for distance calculation
        self._prev_position: tuple[float, float] | None = None

        self._init_files()
        logger.info("Flight recorder initialized", flight_id=self._flight_id)

    def _init_files(self) -> None:
        """Initialize log files."""
        # Telemetry file
        self._telemetry_file = self._base_path / "telemetry.jsonl"
        self._telemetry_handle = open(self._telemetry_file, "w")
        self._record.telemetry_file = self._telemetry_file

        # Events file
        self._events_file = self._base_path / "events.jsonl"
        self._events_handle = open(self._events_file, "w")
        self._record.events_file = self._events_file

    @property
    def flight_id(self) -> str:
        """Current flight ID."""
        return self._flight_id

    @property
    def record(self) -> FlightRecord:
        """Current flight record."""
        return self._record

    @property
    def recording_path(self) -> Path:
        """Path to recording directory."""
        return self._base_path

    def start(self) -> None:
        """Start recording."""
        self._running = True
        self._record.start_time = datetime.now()
        logger.info("Recording started", flight_id=self._flight_id)

    def stop(self) -> None:
        """Stop recording and finalize files."""
        self._running = False
        self._record.end_time = datetime.now()
        self._record.duration_seconds = (
            self._record.end_time - self._record.start_time
        ).total_seconds()

        self._finalize()
        logger.info(
            "Recording stopped",
            flight_id=self._flight_id,
            duration=self._record.duration_seconds,
        )

    def _finalize(self) -> None:
        """Finalize and close all files."""
        with self._write_lock:
            if self._telemetry_handle:
                self._telemetry_handle.close()
                self._telemetry_handle = None

            if self._events_handle:
                self._events_handle.close()
                self._events_handle = None

            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None

        # Save metadata
        self._record.save_metadata()

    def log_telemetry(self, telemetry: TelemetryData) -> bool:
        """
        Log telemetry data.

        Args:
            telemetry: Telemetry data to log.

        Returns:
            True if logged (rate limit may skip).
        """
        import time

        if not self._running:
            return False

        current_time = time.time()
        if current_time - self._last_telemetry_time < self._telemetry_interval:
            return False

        with self._write_lock:
            if not self._telemetry_handle:
                return False

            data = telemetry.to_dict()
            data["_seq"] = self._record.telemetry_samples

            self._telemetry_handle.write(json.dumps(data) + "\n")
            self._telemetry_handle.flush()

        self._record.telemetry_samples += 1
        self._last_telemetry_time = current_time

        # Update statistics
        self._update_flight_stats(telemetry)

        return True

    def _update_flight_stats(self, telemetry: TelemetryData) -> None:
        """Update flight statistics from telemetry."""
        # Max altitude
        if telemetry.gps.altitude_rel > self._record.max_altitude_m:
            self._record.max_altitude_m = telemetry.gps.altitude_rel

        # Max speed
        if telemetry.velocity.ground_speed > self._record.max_speed_ms:
            self._record.max_speed_ms = telemetry.velocity.ground_speed

        # Distance traveled
        current_pos = (telemetry.gps.latitude, telemetry.gps.longitude)
        if self._prev_position and current_pos != (0, 0):
            dist = self._haversine_distance(self._prev_position, current_pos)
            self._record.distance_traveled_m += dist
        self._prev_position = current_pos

        # Home position
        if self._record.home_position is None and current_pos != (0, 0):
            self._record.home_position = current_pos

    def _haversine_distance(
        self,
        pos1: tuple[float, float],
        pos2: tuple[float, float],
    ) -> float:
        """Calculate distance between two GPS coordinates in meters."""
        import math

        R = 6371000  # Earth's radius in meters

        lat1, lon1 = map(math.radians, pos1)
        lat2, lon2 = map(math.radians, pos2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def log_event(
        self,
        event_type: str,
        data: dict[str, Any],
        severity: str = "info",
    ) -> None:
        """Log an event."""
        if not self._running:
            return

        with self._write_lock:
            if not self._events_handle:
                return

            event = {
                "_timestamp": datetime.now().isoformat(),
                "_seq": self._record.events_recorded,
                "event_type": event_type,
                "severity": severity,
                "data": data,
            }

            self._events_handle.write(json.dumps(event) + "\n")
            self._events_handle.flush()

        self._record.events_recorded += 1

    def log_detection(
        self,
        track: TrackedObject,
        frame_number: int,
    ) -> None:
        """Log a detection/tracking event."""
        self.log_event(
            "detection",
            {
                "track_id": track.track_id,
                "bbox": track.bbox.to_xywh(),
                "confidence": track.confidence,
                "class_name": track.class_name,
                "state": track.state.name,
                "velocity": track.velocity,
                "frame_number": frame_number,
            },
            severity="debug",
        )
        self._record.detections_recorded += 1

    def log_frame(
        self,
        frame: Frame,
        annotations: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        """
        Log a video frame.

        Args:
            frame: Frame to log.
            annotations: Optional list of bounding boxes to draw.
        """
        if not self._running or not self._config.log_video_frames:
            return

        # Initialize video writer if needed
        if self._video_writer is None:
            self._init_video_writer(frame)

        if self._video_writer is None:
            return

        # Draw annotations if provided
        frame_data = frame.data.copy()
        if annotations:
            for x, y, w, h in annotations:
                cv2.rectangle(frame_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add timestamp overlay
        timestamp_str = frame.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(
            frame_data,
            timestamp_str,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        with self._write_lock:
            self._video_writer.write(frame_data)

        self._record.frames_recorded += 1

    def _init_video_writer(self, frame: Frame) -> None:
        """Initialize video writer with frame dimensions."""
        self._video_file = self._base_path / "video.mp4"
        self._record.video_file = self._video_file

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 10.0  # Recording FPS

        self._video_writer = cv2.VideoWriter(
            str(self._video_file),
            fourcc,
            fps,
            (frame.width, frame.height),
        )

    def get_summary(self) -> dict[str, Any]:
        """Get recording summary."""
        return {
            "flight_id": self._flight_id,
            "recording": self._running,
            "duration_seconds": (
                datetime.now() - self._record.start_time
            ).total_seconds() if self._running else self._record.duration_seconds,
            "telemetry_samples": self._record.telemetry_samples,
            "frames_recorded": self._record.frames_recorded,
            "events_recorded": self._record.events_recorded,
            "detections_recorded": self._record.detections_recorded,
            "storage_path": str(self._base_path),
        }


class FlightArchive:
    """
    Manages archived flight recordings.

    Provides access to historical flight data for analysis.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir) / "flights"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def list_flights(self) -> list[str]:
        """List all recorded flight IDs."""
        return [
            d.name for d in self._base_dir.iterdir()
            if d.is_dir() and (d / "flight_metadata.json").exists()
        ]

    def get_flight_metadata(self, flight_id: str) -> FlightRecord | None:
        """Get metadata for a specific flight."""
        metadata_file = self._base_dir / flight_id / "flight_metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            data = json.load(f)

        return FlightRecord(
            flight_id=data["flight_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            duration_seconds=data.get("duration_seconds", 0),
            telemetry_samples=data.get("telemetry_samples", 0),
            frames_recorded=data.get("frames_recorded", 0),
            events_recorded=data.get("events_recorded", 0),
            detections_recorded=data.get("detections_recorded", 0),
            max_altitude_m=data.get("max_altitude_m", 0),
            max_speed_ms=data.get("max_speed_ms", 0),
            distance_traveled_m=data.get("distance_traveled_m", 0),
            home_position=tuple(data["home_position"]) if data.get("home_position") else None,
            base_path=self._base_dir / flight_id,
        )

    def load_telemetry(self, flight_id: str) -> list[dict[str, Any]]:
        """Load telemetry data for a flight."""
        telemetry_file = self._base_dir / flight_id / "telemetry.jsonl"

        if not telemetry_file.exists():
            return []

        telemetry = []
        with open(telemetry_file) as f:
            for line in f:
                if line.strip():
                    telemetry.append(json.loads(line))

        return telemetry

    def load_events(self, flight_id: str) -> list[dict[str, Any]]:
        """Load events for a flight."""
        events_file = self._base_dir / flight_id / "events.jsonl"

        if not events_file.exists():
            return []

        events = []
        with open(events_file) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        return events

    def delete_flight(self, flight_id: str) -> bool:
        """Delete a flight recording."""
        flight_dir = self._base_dir / flight_id

        if not flight_dir.exists():
            return False

        shutil.rmtree(flight_dir)
        logger.info("Flight deleted", flight_id=flight_id)
        return True

    def get_disk_usage(self) -> dict[str, Any]:
        """Get disk usage statistics."""
        total_size = 0
        flight_count = 0

        for flight_dir in self._base_dir.iterdir():
            if flight_dir.is_dir():
                flight_count += 1
                for file in flight_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "flight_count": flight_count,
            "base_dir": str(self._base_dir),
        }

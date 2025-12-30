"""
Camera management for RGB and thermal cameras.

Handles camera initialization, frame capture, and adaptive
frame rate control based on processing capacity.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Any
import threading
import structlog

import cv2
import numpy as np

from drone_flight_core.core.config import VisionConfig

logger = structlog.get_logger(__name__)


class CameraType(Enum):
    """Type of camera."""

    RGB = auto()
    THERMAL = auto()


class CameraState(Enum):
    """State of camera."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    STREAMING = auto()
    ERROR = auto()


@dataclass
class Frame:
    """Represents a captured frame."""

    data: np.ndarray
    camera_type: CameraType
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0
    exposure_time_ms: float = 0.0
    gain: float = 0.0

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self.data.shape[1] if len(self.data.shape) >= 2 else 0

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self.data.shape[0] if len(self.data.shape) >= 1 else 0

    @property
    def channels(self) -> int:
        """Number of color channels."""
        return self.data.shape[2] if len(self.data.shape) >= 3 else 1


@dataclass
class CameraStats:
    """Camera performance statistics."""

    frames_captured: int = 0
    frames_dropped: int = 0
    current_fps: float = 0.0
    average_latency_ms: float = 0.0
    last_frame_time: datetime | None = None


class Camera:
    """
    Manages a single camera device.

    Supports both standard RGB cameras (via OpenCV) and
    thermal cameras (with appropriate drivers).
    """

    def __init__(
        self,
        camera_id: int,
        camera_type: CameraType,
        resolution: tuple[int, int] = (1280, 720),
    ) -> None:
        self._camera_id = camera_id
        self._camera_type = camera_type
        self._resolution = resolution
        self._state = CameraState.DISCONNECTED

        self._capture: cv2.VideoCapture | None = None
        self._stats = CameraStats()
        self._frame_number = 0

        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Frame | None = None

        self._fps_timestamps: list[float] = []

        logger.info(
            "Camera initialized",
            camera_id=camera_id,
            type=camera_type.name,
            resolution=resolution,
        )

    @property
    def state(self) -> CameraState:
        """Current camera state."""
        return self._state

    @property
    def stats(self) -> CameraStats:
        """Camera statistics."""
        return self._stats

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._state in (CameraState.CONNECTED, CameraState.STREAMING)

    def connect(self) -> bool:
        """
        Connect to the camera.

        Returns:
            True if connection successful.
        """
        self._state = CameraState.CONNECTING

        try:
            if self._camera_type == CameraType.RGB:
                self._capture = cv2.VideoCapture(self._camera_id)
            else:
                # For thermal cameras, might need different initialization
                # This is a placeholder - actual implementation depends on hardware
                self._capture = cv2.VideoCapture(self._camera_id)

            if not self._capture.isOpened():
                logger.error("Failed to open camera", camera_id=self._camera_id)
                self._state = CameraState.ERROR
                return False

            # Set resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

            # Verify connection with test frame
            ret, _ = self._capture.read()
            if not ret:
                logger.error("Failed to read test frame", camera_id=self._camera_id)
                self._state = CameraState.ERROR
                return False

            self._state = CameraState.CONNECTED
            logger.info("Camera connected", camera_id=self._camera_id)
            return True

        except Exception as e:
            logger.error("Camera connection error", error=str(e))
            self._state = CameraState.ERROR
            return False

    def disconnect(self) -> None:
        """Disconnect from camera."""
        self.stop_streaming()

        if self._capture:
            self._capture.release()
            self._capture = None

        self._state = CameraState.DISCONNECTED
        logger.info("Camera disconnected", camera_id=self._camera_id)

    def start_streaming(self) -> None:
        """Start continuous frame capture in background thread."""
        if self._running:
            return

        if not self.is_connected:
            if not self.connect():
                return

        self._running = True
        self._state = CameraState.STREAMING
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"camera-{self._camera_id}",
        )
        self._capture_thread.start()
        logger.info("Camera streaming started", camera_id=self._camera_id)

    def stop_streaming(self) -> None:
        """Stop continuous frame capture."""
        self._running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        if self._state == CameraState.STREAMING:
            self._state = CameraState.CONNECTED

        logger.info("Camera streaming stopped", camera_id=self._camera_id)

    def _capture_loop(self) -> None:
        """Background thread for frame capture."""
        while self._running and self._capture:
            try:
                ret, frame_data = self._capture.read()

                if ret:
                    self._frame_number += 1
                    frame = Frame(
                        data=frame_data,
                        camera_type=self._camera_type,
                        timestamp=datetime.now(),
                        frame_number=self._frame_number,
                    )

                    with self._frame_lock:
                        self._latest_frame = frame

                    self._update_stats()
                else:
                    self._stats.frames_dropped += 1

            except Exception as e:
                logger.error("Frame capture error", error=str(e))
                time.sleep(0.1)

    def _update_stats(self) -> None:
        """Update camera statistics."""
        now = time.time()
        self._fps_timestamps.append(now)

        # Keep only last second of timestamps
        self._fps_timestamps = [t for t in self._fps_timestamps if now - t < 1.0]

        self._stats.frames_captured = self._frame_number
        self._stats.current_fps = len(self._fps_timestamps)
        self._stats.last_frame_time = datetime.now()

    def get_frame(self) -> Frame | None:
        """
        Get the latest captured frame.

        Returns:
            Latest frame or None if no frame available.
        """
        with self._frame_lock:
            return self._latest_frame

    def capture_frame(self) -> Frame | None:
        """
        Capture a single frame synchronously.

        Returns:
            Captured frame or None on failure.
        """
        if not self._capture:
            return None

        try:
            ret, frame_data = self._capture.read()

            if ret:
                self._frame_number += 1
                return Frame(
                    data=frame_data,
                    camera_type=self._camera_type,
                    timestamp=datetime.now(),
                    frame_number=self._frame_number,
                )
            return None

        except Exception as e:
            logger.error("Frame capture error", error=str(e))
            return None


class CameraManager:
    """
    Manages multiple cameras with synchronized capture.

    Handles adaptive frame rate control based on processing
    capacity to balance detection quality with compute resources.
    """

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._cameras: dict[CameraType, Camera] = {}
        self._running = False
        self._current_fps = config.processing_fps

        # Frame callbacks
        self._frame_callbacks: list[Callable[[Frame, Frame | None], None]] = []

        # Processing timing
        self._processing_times: list[float] = []
        self._max_processing_history = 30

        logger.info("Camera manager initialized", config=config.model_dump())

    def initialize_cameras(self) -> bool:
        """
        Initialize all configured cameras.

        Returns:
            True if at least one camera initialized successfully.
        """
        success = False

        if self._config.enable_rgb:
            rgb_camera = Camera(
                camera_id=self._config.rgb_camera_id,
                camera_type=CameraType.RGB,
                resolution=self._config.rgb_resolution,
            )
            if rgb_camera.connect():
                self._cameras[CameraType.RGB] = rgb_camera
                success = True

        if self._config.enable_thermal:
            thermal_camera = Camera(
                camera_id=self._config.thermal_camera_id,
                camera_type=CameraType.THERMAL,
                resolution=self._config.thermal_resolution,
            )
            if thermal_camera.connect():
                self._cameras[CameraType.THERMAL] = thermal_camera
                success = True

        return success

    def start_streaming(self) -> None:
        """Start streaming from all cameras."""
        for camera in self._cameras.values():
            camera.start_streaming()
        self._running = True

    def stop_streaming(self) -> None:
        """Stop streaming from all cameras."""
        self._running = False
        for camera in self._cameras.values():
            camera.stop_streaming()

    def shutdown(self) -> None:
        """Shutdown all cameras."""
        self.stop_streaming()
        for camera in self._cameras.values():
            camera.disconnect()
        self._cameras.clear()
        logger.info("Camera manager shutdown")

    def get_camera(self, camera_type: CameraType) -> Camera | None:
        """Get camera by type."""
        return self._cameras.get(camera_type)

    def get_synchronized_frames(self) -> tuple[Frame | None, Frame | None]:
        """
        Get synchronized frames from RGB and thermal cameras.

        Returns:
            Tuple of (rgb_frame, thermal_frame). Either may be None.
        """
        rgb_frame = None
        thermal_frame = None

        rgb_camera = self._cameras.get(CameraType.RGB)
        if rgb_camera:
            rgb_frame = rgb_camera.get_frame()

        thermal_camera = self._cameras.get(CameraType.THERMAL)
        if thermal_camera:
            thermal_frame = thermal_camera.get_frame()

        return rgb_frame, thermal_frame

    def on_frame(self, callback: Callable[[Frame, Frame | None], None]) -> None:
        """
        Register callback for frame pairs.

        Args:
            callback: Function called with (rgb_frame, thermal_frame).
        """
        self._frame_callbacks.append(callback)

    def report_processing_time(self, processing_time_ms: float) -> None:
        """
        Report frame processing time for adaptive FPS control.

        Args:
            processing_time_ms: Time taken to process frame in milliseconds.
        """
        self._processing_times.append(processing_time_ms)

        if len(self._processing_times) > self._max_processing_history:
            self._processing_times = self._processing_times[-self._max_processing_history:]

        if self._config.adaptive_fps_enabled:
            self._adjust_fps()

    def _adjust_fps(self) -> None:
        """Adjust processing FPS based on processing times."""
        if len(self._processing_times) < 5:
            return

        avg_processing_time = sum(self._processing_times) / len(self._processing_times)
        max_fps_for_processing = 1000.0 / avg_processing_time if avg_processing_time > 0 else 30.0

        # Target 80% of maximum capacity
        target_fps = max_fps_for_processing * 0.8

        # Clamp to configured limits
        target_fps = max(self._config.min_processing_fps, target_fps)
        target_fps = min(self._config.max_processing_fps, target_fps)

        # Smooth adjustment (avoid sudden changes)
        self._current_fps = 0.9 * self._current_fps + 0.1 * target_fps

        logger.debug(
            "FPS adjusted",
            current_fps=self._current_fps,
            avg_processing_ms=avg_processing_time,
        )

    @property
    def current_fps(self) -> float:
        """Current processing FPS."""
        return self._current_fps

    @property
    def frame_interval_sec(self) -> float:
        """Current frame interval in seconds."""
        return 1.0 / self._current_fps if self._current_fps > 0 else 0.1

    async def process_loop(
        self,
        processor: Callable[[Frame, Frame | None], Any],
    ) -> None:
        """
        Main processing loop with adaptive frame rate.

        Args:
            processor: Function to process frame pair.
        """
        logger.info("Processing loop started", initial_fps=self._current_fps)

        while self._running:
            start_time = time.time()

            # Get frames
            rgb_frame, thermal_frame = self.get_synchronized_frames()

            if rgb_frame or thermal_frame:
                # Process frames
                process_start = time.time()
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, processor, rgb_frame, thermal_frame
                    )
                except Exception as e:
                    logger.error("Frame processing error", error=str(e))

                processing_time_ms = (time.time() - process_start) * 1000
                self.report_processing_time(processing_time_ms)

            # Wait for next frame interval
            elapsed = time.time() - start_time
            wait_time = self.frame_interval_sec - elapsed

            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def to_dict(self) -> dict[str, Any]:
        """Export camera manager status."""
        return {
            "cameras": {
                cam_type.name: {
                    "state": camera.state.name,
                    "stats": {
                        "frames_captured": camera.stats.frames_captured,
                        "frames_dropped": camera.stats.frames_dropped,
                        "current_fps": camera.stats.current_fps,
                    },
                }
                for cam_type, camera in self._cameras.items()
            },
            "current_processing_fps": self._current_fps,
            "adaptive_fps_enabled": self._config.adaptive_fps_enabled,
        }

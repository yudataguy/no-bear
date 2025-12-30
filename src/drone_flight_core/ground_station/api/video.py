"""
Video streaming API.

Provides endpoints for accessing live video feeds
and recorded video from drone cameras.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/video", tags=["video"])


class StreamConfig(BaseModel):
    """Video stream configuration."""

    camera: str = "rgb"  # rgb, thermal
    resolution: str = "720p"  # 480p, 720p, 1080p
    fps: int = 15
    quality: int = 80  # JPEG quality 1-100


class StreamStatus(BaseModel):
    """Video stream status."""

    camera: str
    streaming: bool
    fps: float
    resolution: tuple[int, int]
    clients: int
    bytes_sent: int


# Stream state (would be connected to actual camera in production)
_stream_status: dict[str, StreamStatus] = {
    "rgb": StreamStatus(
        camera="rgb",
        streaming=False,
        fps=0.0,
        resolution=(1280, 720),
        clients=0,
        bytes_sent=0,
    ),
    "thermal": StreamStatus(
        camera="thermal",
        streaming=False,
        fps=0.0,
        resolution=(640, 480),
        clients=0,
        bytes_sent=0,
    ),
}


@router.get("/status")
async def get_stream_status() -> dict[str, StreamStatus]:
    """Get status of all video streams."""
    return _stream_status


@router.get("/status/{camera}")
async def get_camera_status(camera: str) -> StreamStatus:
    """Get status of specific camera stream."""
    if camera not in _stream_status:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera}")
    return _stream_status[camera]


@router.post("/start/{camera}")
async def start_stream(camera: str, config: StreamConfig | None = None) -> dict:
    """Start video stream for camera."""
    if camera not in _stream_status:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera}")

    _stream_status[camera].streaming = True
    logger.info("Video stream started", camera=camera)

    return {
        "message": f"Stream started for {camera}",
        "camera": camera,
        "config": config.model_dump() if config else None,
    }


@router.post("/stop/{camera}")
async def stop_stream(camera: str) -> dict:
    """Stop video stream for camera."""
    if camera not in _stream_status:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera}")

    _stream_status[camera].streaming = False
    _stream_status[camera].clients = 0
    logger.info("Video stream stopped", camera=camera)

    return {"message": f"Stream stopped for {camera}", "camera": camera}


@router.get("/snapshot/{camera}")
async def get_snapshot(camera: str) -> Response:
    """
    Get a single frame snapshot from camera.

    Returns JPEG image.
    """
    if camera not in _stream_status:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera}")

    # In real implementation, this would capture from actual camera
    # For now, return a placeholder
    try:
        import numpy as np
        import cv2

        # Generate placeholder image
        width, height = _stream_status[camera].resolution
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some text
        text = f"{camera.upper()} - {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(img, text, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", img)
        image_bytes = buffer.tobytes()

        return Response(content=image_bytes, media_type="image/jpeg")

    except ImportError:
        # If OpenCV not available, return error
        raise HTTPException(status_code=503, detail="Camera not available")


async def generate_mjpeg_stream(camera: str):
    """
    Generate MJPEG stream for camera.

    Yields JPEG frames with multipart boundaries.
    """
    try:
        import numpy as np
        import cv2

        width, height = _stream_status[camera].resolution
        frame_count = 0

        while _stream_status[camera].streaming:
            # Generate frame (in real implementation, get from camera)
            img = np.zeros((height, width, 3), dtype=np.uint8)

            # Add timestamp and frame info
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(img, f"{camera.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(img, f"Frame: {frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Encode as JPEG
            _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            # Yield with multipart boundary
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            frame_count += 1
            _stream_status[camera].bytes_sent += len(frame_bytes)

            # Control frame rate
            await asyncio.sleep(1.0 / 15.0)  # 15 FPS

    except ImportError:
        yield b"--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available\r\n"


@router.get("/stream/{camera}")
async def stream_mjpeg(camera: str) -> StreamingResponse:
    """
    Get MJPEG video stream from camera.

    Returns multipart MJPEG stream that can be displayed in <img> tag.
    """
    if camera not in _stream_status:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera}")

    if not _stream_status[camera].streaming:
        # Auto-start stream
        _stream_status[camera].streaming = True

    _stream_status[camera].clients += 1

    return StreamingResponse(
        generate_mjpeg_stream(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/recordings")
async def list_recordings() -> list[dict]:
    """List available video recordings."""
    # In real implementation, list files from storage
    return [
        {
            "id": "rec_001",
            "filename": "flight_20240101_120000.mp4",
            "camera": "rgb",
            "duration_sec": 300,
            "size_mb": 150.5,
            "created_at": "2024-01-01T12:00:00",
        }
    ]


@router.get("/recordings/{recording_id}")
async def get_recording_info(recording_id: str) -> dict:
    """Get information about a specific recording."""
    # In real implementation, fetch from storage
    return {
        "id": recording_id,
        "filename": "flight_20240101_120000.mp4",
        "camera": "rgb",
        "duration_sec": 300,
        "size_mb": 150.5,
        "created_at": "2024-01-01T12:00:00",
        "flight_id": "flight_001",
    }


@router.delete("/recordings/{recording_id}")
async def delete_recording(recording_id: str) -> dict:
    """Delete a video recording."""
    # In real implementation, delete from storage
    logger.info("Recording deleted", recording_id=recording_id)
    return {"message": "Recording deleted", "recording_id": recording_id}

"""
Configuration management for drone flight core.

Provides validated configuration using Pydantic with support for
environment variables, YAML files, and runtime overrides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeofenceConfig(BaseModel):
    """Geofencing configuration for safety boundaries."""

    enabled: bool = True
    max_altitude_m: float = Field(default=120.0, ge=0, le=500)
    max_distance_m: float = Field(default=1000.0, ge=0)
    home_latitude: float | None = None
    home_longitude: float | None = None
    no_fly_zones: list[dict] = Field(default_factory=list)


class SafetyConfig(BaseModel):
    """Safety system configuration."""

    low_battery_threshold_percent: float = Field(default=20.0, ge=5, le=50)
    critical_battery_threshold_percent: float = Field(default=10.0, ge=3, le=30)
    signal_loss_timeout_sec: float = Field(default=5.0, ge=1, le=30)
    rth_on_signal_loss: bool = True
    rth_on_low_battery: bool = True
    emergency_land_on_critical_battery: bool = True
    geofence: GeofenceConfig = Field(default_factory=GeofenceConfig)


class CommunicationConfig(BaseModel):
    """Communication system configuration."""

    mavlink_connection: str = Field(default="udp:127.0.0.1:14550")
    mavlink_baud_rate: int = Field(default=57600)
    heartbeat_interval_sec: float = Field(default=1.0, ge=0.1, le=5.0)
    telemetry_rate_hz: float = Field(default=10.0, ge=1, le=50)
    command_timeout_sec: float = Field(default=5.0, ge=1, le=30)
    enable_encryption: bool = False
    encryption_key: str | None = None


class VisionConfig(BaseModel):
    """Vision system configuration."""

    rgb_camera_id: int = 0
    thermal_camera_id: int = 1
    rgb_resolution: tuple[int, int] = (1280, 720)
    thermal_resolution: tuple[int, int] = (640, 480)
    processing_fps: float = Field(default=10.0, ge=1, le=30)
    min_processing_fps: float = Field(default=3.0, ge=1, le=10)
    max_processing_fps: float = Field(default=30.0, ge=10, le=60)
    adaptive_fps_enabled: bool = True
    detection_confidence_threshold: float = Field(default=0.6, ge=0.1, le=1.0)
    tracking_lost_timeout_sec: float = Field(default=5.0, ge=1, le=30)
    enable_thermal: bool = True
    enable_rgb: bool = True
    fusion_mode: Literal["thermal_priority", "rgb_priority", "weighted"] = "weighted"
    fusion_weight_thermal: float = Field(default=0.6, ge=0, le=1)

    @field_validator("fusion_weight_thermal")
    @classmethod
    def validate_fusion_weight(cls, v: float) -> float:
        """Ensure fusion weight is valid."""
        if not 0 <= v <= 1:
            raise ValueError("Fusion weight must be between 0 and 1")
        return v


class NavigationConfig(BaseModel):
    """Navigation system configuration."""

    default_altitude_m: float = Field(default=50.0, ge=5, le=400)
    max_speed_ms: float = Field(default=15.0, ge=1, le=30)
    approach_speed_ms: float = Field(default=5.0, ge=1, le=15)
    waypoint_acceptance_radius_m: float = Field(default=5.0, ge=1, le=20)
    heading_tolerance_deg: float = Field(default=5.0, ge=1, le=30)


class TrackingConfig(BaseModel):
    """Object tracking configuration."""

    default_pattern: Literal["hover", "orbit"] = "hover"
    orbit_radius_m: float = Field(default=30.0, ge=10, le=100)
    orbit_speed_ms: float = Field(default=5.0, ge=1, le=15)
    hover_altitude_m: float = Field(default=30.0, ge=10, le=150)
    energy_aware_pattern_selection: bool = True
    wind_threshold_for_hover_ms: float = Field(default=5.0, ge=1, le=15)
    min_tracking_duration_sec: float = Field(default=10.0, ge=5)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("logs")
    max_log_size_mb: int = Field(default=100, ge=1, le=1000)
    max_log_files: int = Field(default=10, ge=1, le=100)
    log_telemetry: bool = True
    log_video_frames: bool = False
    telemetry_log_rate_hz: float = Field(default=1.0, ge=0.1, le=10)


class GroundStationConfig(BaseModel):
    """Ground station API configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1024, le=65535)
    api_prefix: str = "/api/v1"
    enable_websocket: bool = True
    websocket_path: str = "/ws"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_key: str | None = None
    enable_auth: bool = False


class DroneConfig(BaseSettings):
    """
    Main configuration for drone flight core.

    Configuration can be loaded from:
    - Environment variables (prefixed with DRONE_)
    - JSON/YAML configuration files
    - Runtime overrides
    """

    model_config = SettingsConfigDict(
        env_prefix="DRONE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Drone identification
    drone_id: str = Field(default="drone-001")
    drone_name: str = Field(default="Drone Alpha")

    # Subsystem configurations
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ground_station: GroundStationConfig = Field(default_factory=GroundStationConfig)

    # Simulation mode
    simulation_mode: bool = False

    def save(self, path: Path) -> None:
        """Save configuration to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def from_file(cls, path: Path) -> DroneConfig:
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def load_config(config_path: Path | str | None = None) -> DroneConfig:
    """
    Load configuration from file or environment.

    Args:
        config_path: Optional path to configuration file.
                    If None, loads from environment variables.

    Returns:
        Validated DroneConfig instance.
    """
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            return DroneConfig.from_file(path)

    return DroneConfig()

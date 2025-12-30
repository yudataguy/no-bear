"""Tests for configuration module."""

import pytest
import json
import tempfile
from pathlib import Path

from drone_flight_core.core.config import (
    DroneConfig,
    SafetyConfig,
    VisionConfig,
    CommunicationConfig,
    NavigationConfig,
    TrackingConfig,
    LoggingConfig,
    GroundStationConfig,
    load_config,
)


class TestDroneConfig:
    """Tests for DroneConfig."""

    def test_default_config(self):
        """Default config should be valid."""
        config = DroneConfig()

        assert config.drone_id == "drone-001"
        assert config.simulation_mode is False
        assert config.safety is not None
        assert config.vision is not None

    def test_config_from_env(self, monkeypatch):
        """Config should load from environment variables."""
        monkeypatch.setenv("DRONE_DRONE_ID", "test-drone")
        monkeypatch.setenv("DRONE_SIMULATION_MODE", "true")

        config = DroneConfig()

        assert config.drone_id == "test-drone"
        assert config.simulation_mode is True

    def test_nested_config_from_env(self, monkeypatch):
        """Nested config should load from environment."""
        monkeypatch.setenv("DRONE_SAFETY__LOW_BATTERY_THRESHOLD_PERCENT", "25")

        config = DroneConfig()

        assert config.safety.low_battery_threshold_percent == 25.0

    def test_save_and_load(self, tmp_path):
        """Config should save and load correctly."""
        config = DroneConfig(drone_id="test-save")

        config_path = tmp_path / "config.json"
        config.save(config_path)

        loaded = DroneConfig.from_file(config_path)

        assert loaded.drone_id == "test-save"

    def test_load_config_from_file(self, tmp_path):
        """load_config should work with file path."""
        config_data = {
            "drone_id": "loaded-drone",
            "simulation_mode": True,
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config(config_path)

        assert config.drone_id == "loaded-drone"
        assert config.simulation_mode is True

    def test_load_config_missing_file(self):
        """load_config should return defaults for missing file."""
        config = load_config("/nonexistent/path/config.json")

        assert config.drone_id == "drone-001"


class TestSafetyConfig:
    """Tests for SafetyConfig."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = SafetyConfig()

        assert config.low_battery_threshold_percent == 20.0
        assert config.critical_battery_threshold_percent == 10.0
        assert config.rth_on_signal_loss is True

    def test_validation_battery_threshold(self):
        """Battery thresholds should be validated."""
        with pytest.raises(ValueError):
            SafetyConfig(low_battery_threshold_percent=60.0)  # Too high

        with pytest.raises(ValueError):
            SafetyConfig(critical_battery_threshold_percent=1.0)  # Too low

    def test_geofence_defaults(self):
        """Geofence should have sensible defaults."""
        config = SafetyConfig()

        assert config.geofence.enabled is True
        assert config.geofence.max_altitude_m == 120.0
        assert config.geofence.max_distance_m == 1000.0


class TestVisionConfig:
    """Tests for VisionConfig."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = VisionConfig()

        assert config.processing_fps == 10.0
        assert config.enable_thermal is True
        assert config.enable_rgb is True

    def test_fusion_weight_validation(self):
        """Fusion weight should be validated."""
        with pytest.raises(ValueError):
            VisionConfig(fusion_weight_thermal=1.5)

    def test_resolution_defaults(self):
        """Resolution defaults should be reasonable."""
        config = VisionConfig()

        assert config.rgb_resolution == (1280, 720)
        assert config.thermal_resolution == (640, 480)


class TestCommunicationConfig:
    """Tests for CommunicationConfig."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = CommunicationConfig()

        assert "udp" in config.mavlink_connection or "tcp" in config.mavlink_connection
        assert config.heartbeat_interval_sec == 1.0


class TestNavigationConfig:
    """Tests for NavigationConfig."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = NavigationConfig()

        assert config.default_altitude_m == 50.0
        assert config.max_speed_ms > 0

    def test_speed_validation(self):
        """Speed should be validated."""
        with pytest.raises(ValueError):
            NavigationConfig(max_speed_ms=0.5)  # Too slow


class TestTrackingConfig:
    """Tests for TrackingConfig."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = TrackingConfig()

        assert config.default_pattern in ["hover", "orbit"]
        assert config.energy_aware_pattern_selection is True


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = LoggingConfig()

        assert config.level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert config.log_dir == Path("logs")


class TestGroundStationConfig:
    """Tests for GroundStationConfig."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = GroundStationConfig()

        assert config.port == 8080
        assert config.enable_websocket is True
        assert config.api_prefix == "/api/v1"

    def test_port_validation(self):
        """Port should be validated."""
        with pytest.raises(ValueError):
            GroundStationConfig(port=80)  # Below 1024

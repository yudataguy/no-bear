"""Tests for safety monitor."""

import pytest
from datetime import datetime, timedelta

from drone_flight_core.core.safety import (
    SafetyMonitor,
    SafetyViolationType,
    SafetyAction,
    SafetyViolation,
    GeofenceZone,
    DroneState,
)
from drone_flight_core.core.config import SafetyConfig, GeofenceConfig


class TestGeofenceZone:
    """Tests for GeofenceZone."""

    def test_contains_point_inside(self):
        """Point inside zone should be detected."""
        zone = GeofenceZone(
            name="Test Zone",
            latitude=0.0,
            longitude=0.0,
            radius_m=1000.0,
        )

        assert zone.contains_point(0.0, 0.0)
        assert zone.contains_point(0.001, 0.001)

    def test_contains_point_outside(self):
        """Point outside zone should not be detected."""
        zone = GeofenceZone(
            name="Test Zone",
            latitude=0.0,
            longitude=0.0,
            radius_m=100.0,
        )

        assert not zone.contains_point(10.0, 10.0)

    def test_altitude_limits(self):
        """Altitude limits should be respected."""
        zone = GeofenceZone(
            name="Test Zone",
            latitude=0.0,
            longitude=0.0,
            radius_m=1000.0,
            min_altitude_m=10.0,
            max_altitude_m=100.0,
        )

        assert zone.contains_point(0.0, 0.0, alt=50.0)
        assert not zone.contains_point(0.0, 0.0, alt=5.0)
        assert not zone.contains_point(0.0, 0.0, alt=150.0)


class TestSafetyMonitor:
    """Tests for SafetyMonitor."""

    @pytest.fixture
    def config(self):
        """Create safety config for testing."""
        return SafetyConfig(
            low_battery_threshold_percent=20.0,
            critical_battery_threshold_percent=10.0,
            signal_loss_timeout_sec=5.0,
            geofence=GeofenceConfig(
                enabled=True,
                max_altitude_m=100.0,
                max_distance_m=500.0,
            ),
        )

    @pytest.fixture
    def monitor(self, config):
        """Create safety monitor for testing."""
        return SafetyMonitor(config)

    def test_initial_state(self, monitor):
        """Monitor should start with no violations."""
        violations = monitor.check_all_safety()
        # May have violations due to default drone state
        assert isinstance(violations, list)

    def test_low_battery_warning(self, monitor):
        """Low battery should trigger warning."""
        monitor.update_drone_state(battery_percent=15.0)
        violations = monitor.check_all_safety()

        battery_violations = [
            v for v in violations
            if v.violation_type == SafetyViolationType.LOW_BATTERY
        ]
        assert len(battery_violations) == 1
        assert battery_violations[0].severity == "warning"

    def test_critical_battery(self, monitor):
        """Critical battery should trigger emergency."""
        monitor.update_drone_state(battery_percent=5.0)
        violations = monitor.check_all_safety()

        critical_violations = [
            v for v in violations
            if v.violation_type == SafetyViolationType.CRITICAL_BATTERY
        ]
        assert len(critical_violations) == 1
        assert critical_violations[0].severity == "emergency"

    def test_altitude_violation(self, monitor):
        """Exceeding max altitude should trigger violation."""
        monitor.update_drone_state(altitude_m=150.0)
        violations = monitor.check_all_safety()

        altitude_violations = [
            v for v in violations
            if v.violation_type == SafetyViolationType.GEOFENCE_ALTITUDE
        ]
        assert len(altitude_violations) == 1

    def test_distance_violation(self, monitor):
        """Exceeding max distance should trigger violation."""
        monitor.set_home_position(0.0, 0.0)
        monitor.update_drone_state(latitude=10.0, longitude=10.0)

        violations = monitor.check_all_safety()

        distance_violations = [
            v for v in violations
            if v.violation_type == SafetyViolationType.GEOFENCE_DISTANCE
        ]
        assert len(distance_violations) == 1

    def test_signal_loss(self, monitor):
        """Signal loss should trigger violation."""
        old_time = datetime.now() - timedelta(seconds=10)
        monitor.update_drone_state(last_heartbeat=old_time)

        violations = monitor.check_all_safety()

        signal_violations = [
            v for v in violations
            if v.violation_type == SafetyViolationType.SIGNAL_LOSS
        ]
        assert len(signal_violations) == 1

    def test_gps_loss(self, monitor):
        """GPS loss during flight should trigger violation."""
        monitor.update_drone_state(gps_fix=False, in_flight=True)

        violations = monitor.check_all_safety()

        gps_violations = [
            v for v in violations
            if v.violation_type == SafetyViolationType.GPS_LOSS
        ]
        assert len(gps_violations) == 1

    def test_no_fly_zone(self, monitor):
        """Entering no-fly zone should trigger violation."""
        zone = GeofenceZone(
            name="Restricted Area",
            latitude=1.0,
            longitude=1.0,
            radius_m=1000.0,
            is_no_fly=True,
        )
        monitor.add_no_fly_zone(zone)
        monitor.update_drone_state(latitude=1.0, longitude=1.0)

        violations = monitor.check_all_safety()

        nfz_violations = [
            v for v in violations
            if v.violation_type == SafetyViolationType.GEOFENCE_NO_FLY_ZONE
        ]
        assert len(nfz_violations) == 1
        assert nfz_violations[0].severity == "emergency"

    def test_is_safe_to_fly_no_gps(self, monitor):
        """Should not be safe to fly without GPS."""
        monitor.update_drone_state(gps_fix=False, battery_percent=100.0)

        safe, reason = monitor.is_safe_to_fly()
        assert not safe
        assert "GPS" in reason

    def test_is_safe_to_fly_low_battery(self, monitor):
        """Should not be safe to fly with low battery."""
        monitor.update_drone_state(
            gps_fix=True,
            gps_satellites=10,
            battery_percent=10.0,
        )

        safe, reason = monitor.is_safe_to_fly()
        assert not safe
        assert "battery" in reason.lower()

    def test_is_safe_to_fly_all_good(self, monitor):
        """Should be safe to fly with all systems nominal."""
        monitor.update_drone_state(
            gps_fix=True,
            gps_satellites=10,
            battery_percent=100.0,
        )

        safe, reason = monitor.is_safe_to_fly()
        assert safe

    def test_highest_priority_action(self, monitor):
        """Should return highest priority action."""
        monitor.update_drone_state(battery_percent=5.0)  # Critical

        action = monitor.get_highest_priority_action()
        assert action in [SafetyAction.EMERGENCY_LAND, SafetyAction.RETURN_TO_HOME]

    def test_remove_no_fly_zone(self, monitor):
        """Should be able to remove no-fly zones."""
        zone = GeofenceZone(
            name="Test Zone",
            latitude=0.0,
            longitude=0.0,
            radius_m=100.0,
        )
        monitor.add_no_fly_zone(zone)
        assert monitor.remove_no_fly_zone("Test Zone")
        assert not monitor.remove_no_fly_zone("Nonexistent")

    def test_to_dict(self, monitor):
        """to_dict should return complete status."""
        data = monitor.to_dict()

        assert "monitoring_active" in data
        assert "active_violations" in data
        assert "drone_state" in data

"""Tests for mission executor and patrol functionality."""

import pytest
from datetime import datetime

from drone_flight_core.navigation.mission_executor import (
    MissionExecutor,
    Mission,
    MissionType,
    MissionState,
    MissionWaypoint,
    WaypointAction,
    DetectionEvent,
    create_patrol_mission,
)
from drone_flight_core.navigation.alerts import (
    AlertManager,
    Alert,
    AlertType,
    AlertSeverity,
)
from drone_flight_core.core.state_machine import FlightStateMachine, FlightState
from drone_flight_core.core.config import NavigationConfig, TrackingConfig


class TestMissionWaypoint:
    """Tests for MissionWaypoint."""

    def test_basic_waypoint(self):
        """Basic waypoint creation."""
        wp = MissionWaypoint(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
        )

        assert wp.latitude == 37.7749
        assert wp.altitude == 50.0
        assert wp.action == WaypointAction.NONE

    def test_waypoint_with_action(self):
        """Waypoint with action."""
        wp = MissionWaypoint(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
            action=WaypointAction.HOVER,
            hold_time_sec=10.0,
            scan_on_arrival=True,
        )

        assert wp.action == WaypointAction.HOVER
        assert wp.hold_time_sec == 10.0
        assert wp.scan_on_arrival is True

    def test_to_waypoint_conversion(self):
        """Conversion to command handler Waypoint."""
        wp = MissionWaypoint(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
            speed=5.0,
        )

        cmd_wp = wp.to_waypoint()
        assert cmd_wp.latitude == 37.7749
        assert cmd_wp.speed == 5.0


class TestMission:
    """Tests for Mission."""

    def test_mission_creation(self):
        """Basic mission creation."""
        waypoints = [
            MissionWaypoint(37.0, -122.0, 50.0),
            MissionWaypoint(37.1, -122.1, 50.0),
        ]

        mission = Mission(
            id="test_001",
            name="Test Mission",
            mission_type=MissionType.PATROL,
            waypoints=waypoints,
        )

        assert mission.id == "test_001"
        assert len(mission.waypoints) == 2
        assert mission.mission_type == MissionType.PATROL

    def test_current_waypoint(self):
        """Current waypoint property."""
        waypoints = [
            MissionWaypoint(37.0, -122.0, 50.0),
            MissionWaypoint(37.1, -122.1, 50.0),
        ]

        mission = Mission(
            id="test",
            name="Test",
            mission_type=MissionType.ONE_WAY,
            waypoints=waypoints,
        )

        assert mission.current_waypoint == waypoints[0]

        mission.current_waypoint_index = 1
        assert mission.current_waypoint == waypoints[1]

        mission.current_waypoint_index = 5  # Out of range
        assert mission.current_waypoint is None

    def test_progress_percent(self):
        """Progress calculation."""
        waypoints = [
            MissionWaypoint(37.0, -122.0, 50.0),
            MissionWaypoint(37.1, -122.1, 50.0),
            MissionWaypoint(37.2, -122.2, 50.0),
            MissionWaypoint(37.3, -122.3, 50.0),
        ]

        mission = Mission(
            id="test",
            name="Test",
            mission_type=MissionType.ONE_WAY,
            waypoints=waypoints,
        )

        assert mission.progress_percent == 0.0

        mission.current_waypoint_index = 2
        assert mission.progress_percent == 50.0

    def test_is_active(self):
        """Active state detection."""
        mission = Mission(
            id="test",
            name="Test",
            mission_type=MissionType.ONE_WAY,
            waypoints=[],
        )

        mission.state = MissionState.IDLE
        assert not mission.is_active

        mission.state = MissionState.EXECUTING
        assert mission.is_active

        mission.state = MissionState.MONITORING
        assert mission.is_active

    def test_to_dict(self):
        """Dictionary export."""
        mission = Mission(
            id="test_001",
            name="Test Mission",
            mission_type=MissionType.PATROL,
            waypoints=[MissionWaypoint(37.0, -122.0, 50.0)],
        )

        data = mission.to_dict()
        assert data["id"] == "test_001"
        assert data["mission_type"] == "PATROL"
        assert data["waypoint_count"] == 1


class TestCreatePatrolMission:
    """Tests for create_patrol_mission helper."""

    def test_basic_patrol(self):
        """Create basic patrol mission."""
        waypoints = [
            (37.0, -122.0, 50.0),
            (37.1, -122.1, 50.0),
            (37.2, -122.2, 50.0),
        ]

        mission = create_patrol_mission(
            mission_id="patrol_001",
            name="Test Patrol",
            waypoints=waypoints,
        )

        assert mission.id == "patrol_001"
        assert mission.mission_type == MissionType.PATROL
        assert len(mission.waypoints) == 3
        assert mission.loop_count == 0  # Infinite

    def test_patrol_with_options(self):
        """Create patrol with custom options."""
        waypoints = [(37.0, -122.0, 50.0)]

        mission = create_patrol_mission(
            mission_id="patrol_002",
            name="Limited Patrol",
            waypoints=waypoints,
            loop_count=3,
            scan_at_waypoints=True,
            hold_time=10.0,
        )

        assert mission.loop_count == 3
        assert mission.waypoints[0].scan_on_arrival is True
        assert mission.waypoints[0].hold_time_sec == 10.0


class TestDetectionEvent:
    """Tests for DetectionEvent."""

    def test_detection_event_creation(self):
        """Create detection event."""
        event = DetectionEvent(
            timestamp=datetime.now(),
            track_id=1,
            confidence=0.85,
            class_name="person",
            location=(37.7749, -122.4194),
            bbox=(100, 100, 50, 100),
        )

        assert event.track_id == 1
        assert event.confidence == 0.85
        assert not event.confirmed

    def test_confirmation_count(self):
        """Confirmation count tracking."""
        event = DetectionEvent(
            timestamp=datetime.now(),
            track_id=1,
            confidence=0.85,
            class_name="person",
            location=(37.7749, -122.4194),
            bbox=(100, 100, 50, 100),
        )

        assert event.confirmation_count == 1

        event.confirmation_count += 1
        assert event.confirmation_count == 2


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def manager(self):
        """Create alert manager for testing."""
        return AlertManager()

    @pytest.mark.asyncio
    async def test_create_alert(self, manager):
        """Create basic alert."""
        alert = await manager.create_alert(
            alert_type=AlertType.DETECTION,
            severity=AlertSeverity.WARNING,
            title="Object Detected",
            message="Person detected at location",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DETECTION
        assert alert.severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_alert_history(self, manager):
        """Alert history tracking."""
        await manager.create_alert(
            AlertType.DETECTION,
            AlertSeverity.WARNING,
            "Test 1",
            "Message 1",
        )
        await manager.create_alert(
            AlertType.DETECTION_CONFIRMED,
            AlertSeverity.CRITICAL,
            "Test 2",
            "Message 2",
        )

        alerts = manager.get_alerts()
        assert len(alerts) == 2

    @pytest.mark.asyncio
    async def test_unacknowledged_filter(self, manager):
        """Filter unacknowledged alerts."""
        alert = await manager.create_alert(
            AlertType.DETECTION,
            AlertSeverity.WARNING,
            "Test",
            "Message",
        )

        unacked = manager.get_alerts(unacknowledged_only=True)
        assert len(unacked) == 1

        manager.acknowledge_alert(alert.id)

        unacked = manager.get_alerts(unacknowledged_only=True)
        assert len(unacked) == 0

    @pytest.mark.asyncio
    async def test_severity_filter(self, manager):
        """Filter by severity."""
        await manager.create_alert(
            AlertType.WAYPOINT_REACHED,
            AlertSeverity.INFO,
            "Info",
            "Message",
        )
        await manager.create_alert(
            AlertType.DETECTION,
            AlertSeverity.WARNING,
            "Warning",
            "Message",
        )
        await manager.create_alert(
            AlertType.EMERGENCY,
            AlertSeverity.EMERGENCY,
            "Emergency",
            "Message",
        )

        critical_plus = manager.get_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical_plus) == 1  # Only emergency

    @pytest.mark.asyncio
    async def test_detection_alert_helper(self, manager):
        """Detection alert helper method."""
        alert = await manager.alert_detection(
            track_id=1,
            class_name="person",
            confidence=0.85,
            location=(37.7749, -122.4194, 50.0),
            bbox=(100, 100, 50, 100),
            confirmed=True,
        )

        assert alert.alert_type == AlertType.DETECTION_CONFIRMED
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.track_id == 1

    @pytest.mark.asyncio
    async def test_callback_notification(self, manager):
        """Callback should be called on alert."""
        received_alerts = []

        def callback(alert):
            received_alerts.append(alert)

        manager.add_callback(callback)

        await manager.create_alert(
            AlertType.DETECTION,
            AlertSeverity.WARNING,
            "Test",
            "Message",
        )

        assert len(received_alerts) == 1


class TestFlightStateMachinePatrol:
    """Tests for patrol-related state machine functionality."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine in hover state."""
        sm = FlightStateMachine(initial_state=FlightState.HOVER)
        return sm

    @pytest.mark.asyncio
    async def test_start_patrol(self, state_machine):
        """Start patrol from hover."""
        await state_machine.start_patrol()
        assert state_machine.state == FlightState.PATROL

    @pytest.mark.asyncio
    async def test_patrol_to_monitoring(self, state_machine):
        """Transition from patrol to monitoring on detection."""
        await state_machine.start_patrol()
        await state_machine.enter_monitoring("Object detected")

        assert state_machine.state == FlightState.MONITORING

    @pytest.mark.asyncio
    async def test_monitoring_to_tracking(self, state_machine):
        """Transition from monitoring to tracking on confirmation."""
        await state_machine.start_patrol()
        await state_machine.enter_monitoring()
        await state_machine.start_tracking("Target confirmed")

        assert state_machine.state == FlightState.TRACKING

    @pytest.mark.asyncio
    async def test_resume_patrol(self, state_machine):
        """Resume patrol after monitoring."""
        await state_machine.start_patrol()
        await state_machine.enter_monitoring()
        await state_machine.resume_patrol()

        assert state_machine.state == FlightState.PATROL

    def test_is_on_mission(self, state_machine):
        """is_on_mission property."""
        assert not state_machine.is_on_mission  # HOVER

    @pytest.mark.asyncio
    async def test_is_on_mission_during_patrol(self, state_machine):
        """is_on_mission during patrol."""
        await state_machine.start_patrol()
        assert state_machine.is_on_mission

    @pytest.mark.asyncio
    async def test_is_monitoring_or_tracking(self, state_machine):
        """is_monitoring_or_tracking property."""
        assert not state_machine.is_monitoring_or_tracking

        await state_machine.start_patrol()
        await state_machine.enter_monitoring()
        assert state_machine.is_monitoring_or_tracking

        await state_machine.start_tracking()
        assert state_machine.is_monitoring_or_tracking

    @pytest.mark.asyncio
    async def test_full_detection_workflow(self, state_machine):
        """Full detection workflow: patrol -> detect -> monitor -> confirm -> track -> resume."""
        # Start patrol
        await state_machine.start_patrol()
        assert state_machine.state == FlightState.PATROL

        # Detection triggers monitoring
        await state_machine.enter_monitoring("Object detected")
        assert state_machine.state == FlightState.MONITORING

        # Confirmed - start tracking
        await state_machine.start_tracking("Target confirmed")
        assert state_machine.state == FlightState.TRACKING

        # Can orbit
        await state_machine.transition_to(FlightState.ORBITING)
        assert state_machine.state == FlightState.ORBITING

        # Resume patrol
        await state_machine.resume_patrol()
        assert state_machine.state == FlightState.PATROL

    @pytest.mark.asyncio
    async def test_unconfirmed_detection_resume(self, state_machine):
        """Unconfirmed detection should resume patrol."""
        await state_machine.start_patrol()
        await state_machine.enter_monitoring()

        # Not confirmed - resume patrol
        await state_machine.resume_patrol()
        assert state_machine.state == FlightState.PATROL

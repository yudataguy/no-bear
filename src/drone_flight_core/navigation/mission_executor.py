"""
Mission execution for waypoint-based autonomous flight.

Handles patrol missions, waypoint navigation, and automatic
responses to detection events.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Any
import structlog

from drone_flight_core.core.config import NavigationConfig, TrackingConfig
from drone_flight_core.core.state_machine import FlightStateMachine, FlightState
from drone_flight_core.communication.command_handler import CommandHandler, Waypoint

logger = structlog.get_logger(__name__)


class MissionType(Enum):
    """Type of mission."""

    ONE_WAY = auto()  # Fly waypoints once
    ROUND_TRIP = auto()  # Fly waypoints and return
    PATROL = auto()  # Continuous loop
    SEARCH = auto()  # Search pattern


class MissionState(Enum):
    """State of mission execution."""

    IDLE = auto()
    LOADING = auto()
    EXECUTING = auto()
    PAUSED = auto()
    MONITORING = auto()  # Interrupted for detection monitoring
    RETURNING = auto()
    COMPLETED = auto()
    ABORTED = auto()
    FAILED = auto()


class WaypointAction(Enum):
    """Action to perform at waypoint."""

    NONE = auto()
    HOVER = auto()  # Hover for specified time
    PHOTOGRAPH = auto()  # Take photo
    SCAN = auto()  # Perform area scan
    LAND = auto()


@dataclass
class MissionWaypoint:
    """Enhanced waypoint with mission-specific data."""

    latitude: float
    longitude: float
    altitude: float
    speed: float | None = None
    hold_time_sec: float = 0.0
    acceptance_radius_m: float = 5.0
    action: WaypointAction = WaypointAction.NONE
    heading: float | None = None  # None = auto (face direction of travel)

    # Patrol-specific
    scan_on_arrival: bool = False  # Perform detection scan on arrival

    def to_waypoint(self) -> Waypoint:
        """Convert to command handler Waypoint."""
        return Waypoint(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            speed=self.speed,
            hold_time_sec=self.hold_time_sec,
            acceptance_radius_m=self.acceptance_radius_m,
            yaw=self.heading,
        )


@dataclass
class Mission:
    """Complete mission definition."""

    id: str
    name: str
    mission_type: MissionType
    waypoints: list[MissionWaypoint]

    # Patrol settings
    loop_count: int = 0  # 0 = infinite
    current_loop: int = 0

    # Detection settings
    pause_on_detection: bool = True
    detection_confirmation_count: int = 3  # Detections needed to confirm
    monitoring_duration_sec: float = 60.0  # Time to monitor after detection

    # State
    state: MissionState = MissionState.IDLE
    current_waypoint_index: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Statistics
    waypoints_completed: int = 0
    detections_triggered: int = 0
    total_distance_m: float = 0.0

    @property
    def current_waypoint(self) -> MissionWaypoint | None:
        """Get current target waypoint."""
        if 0 <= self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None

    @property
    def progress_percent(self) -> float:
        """Mission progress as percentage."""
        if not self.waypoints:
            return 0.0
        return (self.current_waypoint_index / len(self.waypoints)) * 100

    @property
    def is_active(self) -> bool:
        """Check if mission is currently active."""
        return self.state in (
            MissionState.EXECUTING,
            MissionState.PAUSED,
            MissionState.MONITORING,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "mission_type": self.mission_type.name,
            "state": self.state.name,
            "waypoint_count": len(self.waypoints),
            "current_waypoint": self.current_waypoint_index,
            "progress_percent": self.progress_percent,
            "loop_count": self.loop_count,
            "current_loop": self.current_loop,
            "detections_triggered": self.detections_triggered,
        }


@dataclass
class DetectionEvent:
    """Event triggered by object detection."""

    timestamp: datetime
    track_id: int
    confidence: float
    class_name: str
    location: tuple[float, float]  # lat, lon where detected
    bbox: tuple[int, int, int, int]
    confirmed: bool = False
    confirmation_count: int = 1


class MissionExecutor:
    """
    Executes waypoint missions with detection-triggered monitoring.

    Features:
    - Patrol loop execution
    - Automatic pause on detection
    - Monitoring mode for detection confirmation
    - Ground station alerts
    - Resume after monitoring
    """

    def __init__(
        self,
        state_machine: FlightStateMachine,
        command_handler: CommandHandler,
        nav_config: NavigationConfig,
        tracking_config: TrackingConfig,
    ) -> None:
        self._state_machine = state_machine
        self._commands = command_handler
        self._nav_config = nav_config
        self._tracking_config = tracking_config

        self._current_mission: Mission | None = None
        self._running = False
        self._execution_task: asyncio.Task | None = None

        # Detection handling
        self._pending_detections: list[DetectionEvent] = []
        self._confirmed_detections: list[DetectionEvent] = []
        self._monitoring_start_time: datetime | None = None

        # Position tracking
        self._current_position: tuple[float, float, float] | None = None  # lat, lon, alt

        # Callbacks
        self._on_detection_alert: Callable[[DetectionEvent], None] | None = None
        self._on_mission_complete: Callable[[Mission], None] | None = None
        self._on_waypoint_reached: Callable[[int, MissionWaypoint], None] | None = None

        logger.info("Mission executor initialized")

    @property
    def current_mission(self) -> Mission | None:
        """Get current mission."""
        return self._current_mission

    @property
    def is_executing(self) -> bool:
        """Check if a mission is executing."""
        return self._current_mission is not None and self._current_mission.is_active

    @property
    def confirmed_detections(self) -> list[DetectionEvent]:
        """Get list of confirmed detections."""
        return self._confirmed_detections.copy()

    def set_detection_alert_callback(
        self, callback: Callable[[DetectionEvent], None]
    ) -> None:
        """Set callback for detection alerts."""
        self._on_detection_alert = callback

    def set_mission_complete_callback(
        self, callback: Callable[[Mission], None]
    ) -> None:
        """Set callback for mission completion."""
        self._on_mission_complete = callback

    def set_waypoint_reached_callback(
        self, callback: Callable[[int, MissionWaypoint], None]
    ) -> None:
        """Set callback for waypoint reached events."""
        self._on_waypoint_reached = callback

    def update_position(self, lat: float, lon: float, alt: float) -> None:
        """Update current drone position from telemetry."""
        self._current_position = (lat, lon, alt)

    async def load_mission(self, mission: Mission) -> bool:
        """
        Load a mission for execution.

        Args:
            mission: Mission to load.

        Returns:
            True if loaded successfully.
        """
        if self.is_executing:
            logger.warning("Cannot load mission while another is executing")
            return False

        if not mission.waypoints:
            logger.error("Mission has no waypoints")
            return False

        self._current_mission = mission
        mission.state = MissionState.LOADING
        mission.current_waypoint_index = 0
        mission.current_loop = 0

        logger.info(
            "Mission loaded",
            mission_id=mission.id,
            waypoints=len(mission.waypoints),
            type=mission.mission_type.name,
        )

        return True

    async def start_mission(self) -> bool:
        """
        Start executing the loaded mission.

        Returns:
            True if started successfully.
        """
        if self._current_mission is None:
            logger.error("No mission loaded")
            return False

        if not self._state_machine.is_flying:
            logger.error("Drone must be flying to start mission")
            return False

        self._running = True
        self._current_mission.state = MissionState.EXECUTING
        self._current_mission.started_at = datetime.now()

        # Transition to navigating state
        if self._state_machine.can_transition_to(FlightState.NAVIGATING):
            await self._state_machine.transition_to(
                FlightState.NAVIGATING,
                reason=f"Starting mission: {self._current_mission.name}",
            )

        # Start execution task
        self._execution_task = asyncio.create_task(self._execute_mission())

        logger.info("Mission started", mission_id=self._current_mission.id)
        return True

    async def pause_mission(self, reason: str = "User requested pause") -> bool:
        """Pause the current mission."""
        if not self.is_executing:
            return False

        self._current_mission.state = MissionState.PAUSED

        # Command hover
        await self._commands.set_mode("HOLD")

        logger.info("Mission paused", reason=reason)
        return True

    async def resume_mission(self) -> bool:
        """Resume a paused mission."""
        if self._current_mission is None:
            return False

        if self._current_mission.state not in (
            MissionState.PAUSED,
            MissionState.MONITORING,
        ):
            return False

        self._current_mission.state = MissionState.EXECUTING

        logger.info("Mission resumed")
        return True

    async def abort_mission(self, reason: str = "User aborted") -> bool:
        """Abort the current mission."""
        if self._current_mission is None:
            return False

        self._running = False
        self._current_mission.state = MissionState.ABORTED

        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass

        # Command hover
        await self._commands.set_mode("HOLD")

        logger.info("Mission aborted", reason=reason)
        return True

    async def _execute_mission(self) -> None:
        """Main mission execution loop."""
        mission = self._current_mission
        if mission is None:
            return

        try:
            while self._running and mission.state != MissionState.ABORTED:
                # Check if paused or monitoring
                if mission.state == MissionState.PAUSED:
                    await asyncio.sleep(0.5)
                    continue

                if mission.state == MissionState.MONITORING:
                    await self._handle_monitoring()
                    continue

                # Get current waypoint
                waypoint = mission.current_waypoint
                if waypoint is None:
                    # End of waypoints
                    await self._handle_mission_end()
                    break

                # Navigate to waypoint
                logger.info(
                    "Navigating to waypoint",
                    index=mission.current_waypoint_index,
                    lat=waypoint.latitude,
                    lon=waypoint.longitude,
                    alt=waypoint.altitude,
                )

                reached = await self._navigate_to_waypoint(waypoint)

                if not reached:
                    # Navigation interrupted (detection, abort, etc.)
                    continue

                # Waypoint reached
                mission.waypoints_completed += 1

                if self._on_waypoint_reached:
                    self._on_waypoint_reached(
                        mission.current_waypoint_index, waypoint
                    )

                # Execute waypoint action
                await self._execute_waypoint_action(waypoint)

                # Check for detections if scan enabled
                if waypoint.scan_on_arrival:
                    await self._perform_detection_scan()

                # Move to next waypoint
                mission.current_waypoint_index += 1

        except asyncio.CancelledError:
            logger.info("Mission execution cancelled")
        except Exception as e:
            logger.error("Mission execution error", error=str(e))
            mission.state = MissionState.FAILED
        finally:
            self._running = False

    async def _navigate_to_waypoint(self, waypoint: MissionWaypoint) -> bool:
        """
        Navigate to a waypoint.

        Returns:
            True if waypoint reached, False if interrupted.
        """
        # Set speed if specified
        if waypoint.speed:
            await self._commands.set_speed(waypoint.speed)

        # Send goto command
        await self._commands.goto(
            waypoint.latitude,
            waypoint.longitude,
            waypoint.altitude,
        )

        # Wait until reached or interrupted
        while self._running:
            if self._current_mission.state in (
                MissionState.PAUSED,
                MissionState.MONITORING,
                MissionState.ABORTED,
            ):
                return False

            # Check if reached
            if self._current_position:
                distance = self._calculate_distance(
                    self._current_position[0],
                    self._current_position[1],
                    waypoint.latitude,
                    waypoint.longitude,
                )

                if distance <= waypoint.acceptance_radius_m:
                    logger.debug(
                        "Waypoint reached",
                        index=self._current_mission.current_waypoint_index,
                    )
                    return True

            await asyncio.sleep(0.5)

        return False

    async def _execute_waypoint_action(self, waypoint: MissionWaypoint) -> None:
        """Execute action at waypoint."""
        if waypoint.action == WaypointAction.HOVER:
            logger.debug("Hovering at waypoint", duration=waypoint.hold_time_sec)
            await asyncio.sleep(waypoint.hold_time_sec)

        elif waypoint.action == WaypointAction.PHOTOGRAPH:
            # Trigger photo capture (would integrate with camera)
            logger.info("Capturing photo at waypoint")
            await asyncio.sleep(1.0)

        elif waypoint.action == WaypointAction.SCAN:
            # Perform area scan
            await self._perform_detection_scan()

    async def _perform_detection_scan(self) -> None:
        """Perform detection scan at current location."""
        logger.debug("Performing detection scan")
        # Give vision system time to detect
        await asyncio.sleep(2.0)

    async def _handle_mission_end(self) -> None:
        """Handle reaching end of waypoints."""
        mission = self._current_mission
        if mission is None:
            return

        if mission.mission_type == MissionType.PATROL:
            # Check loop count
            mission.current_loop += 1

            if mission.loop_count == 0 or mission.current_loop < mission.loop_count:
                # Continue patrol
                mission.current_waypoint_index = 0
                logger.info(
                    "Patrol loop completed, starting next",
                    loop=mission.current_loop,
                )
                return

        elif mission.mission_type == MissionType.ROUND_TRIP:
            if mission.current_loop == 0:
                # Reverse waypoints for return trip
                mission.waypoints = list(reversed(mission.waypoints))
                mission.current_waypoint_index = 0
                mission.current_loop = 1
                logger.info("Starting return trip")
                return

        # Mission complete
        mission.state = MissionState.COMPLETED
        mission.completed_at = datetime.now()

        if self._on_mission_complete:
            self._on_mission_complete(mission)

        logger.info(
            "Mission completed",
            mission_id=mission.id,
            waypoints_completed=mission.waypoints_completed,
            detections=mission.detections_triggered,
        )

    def handle_detection(
        self,
        track_id: int,
        confidence: float,
        class_name: str,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """
        Handle detection event from vision system.

        Args:
            track_id: Tracker ID of detected object.
            confidence: Detection confidence.
            class_name: Class name of detected object.
            bbox: Bounding box (x, y, w, h).
        """
        if not self.is_executing:
            return

        mission = self._current_mission
        if mission is None or not mission.pause_on_detection:
            return

        # Get current location
        location = (
            self._current_position[:2]
            if self._current_position
            else (0.0, 0.0)
        )

        # Check if this is a repeat detection of same track
        for pending in self._pending_detections:
            if pending.track_id == track_id:
                pending.confirmation_count += 1
                pending.confidence = max(pending.confidence, confidence)

                # Check if confirmed
                if pending.confirmation_count >= mission.detection_confirmation_count:
                    self._confirm_detection(pending)

                return

        # New detection
        event = DetectionEvent(
            timestamp=datetime.now(),
            track_id=track_id,
            confidence=confidence,
            class_name=class_name,
            location=location,
            bbox=bbox,
        )

        self._pending_detections.append(event)
        mission.detections_triggered += 1

        logger.info(
            "Detection event",
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
        )

        # Trigger monitoring mode
        self._trigger_monitoring(event)

    def _trigger_monitoring(self, event: DetectionEvent) -> None:
        """Trigger monitoring mode for detection."""
        if self._current_mission is None:
            return

        self._current_mission.state = MissionState.MONITORING
        self._monitoring_start_time = datetime.now()

        # Transition state machine to tracking/searching
        asyncio.create_task(self._enter_monitoring_state())

        logger.info(
            "Monitoring mode triggered",
            track_id=event.track_id,
            class_name=event.class_name,
        )

    async def _enter_monitoring_state(self) -> None:
        """Enter monitoring state in state machine."""
        if self._state_machine.can_transition_to(FlightState.SEARCHING):
            await self._state_machine.transition_to(
                FlightState.SEARCHING,
                reason="Detection triggered monitoring",
            )

    def _confirm_detection(self, event: DetectionEvent) -> None:
        """Confirm a detection after enough sightings."""
        event.confirmed = True
        self._confirmed_detections.append(event)

        # Remove from pending
        self._pending_detections = [
            p for p in self._pending_detections if p.track_id != event.track_id
        ]

        logger.info(
            "Detection confirmed",
            track_id=event.track_id,
            class_name=event.class_name,
            confirmations=event.confirmation_count,
        )

        # Alert ground station
        if self._on_detection_alert:
            self._on_detection_alert(event)

    async def _handle_monitoring(self) -> None:
        """Handle monitoring mode."""
        mission = self._current_mission
        if mission is None or self._monitoring_start_time is None:
            return

        elapsed = (datetime.now() - self._monitoring_start_time).total_seconds()

        if elapsed >= mission.monitoring_duration_sec:
            # Monitoring complete, resume mission
            logger.info("Monitoring period complete, resuming mission")

            self._monitoring_start_time = None
            self._pending_detections.clear()
            mission.state = MissionState.EXECUTING

            # Transition back to navigating
            if self._state_machine.can_transition_to(FlightState.NAVIGATING):
                await self._state_machine.transition_to(
                    FlightState.NAVIGATING,
                    reason="Resuming patrol after monitoring",
                )
        else:
            # Continue monitoring
            await asyncio.sleep(0.5)

    def _calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two GPS coordinates in meters."""
        import math

        R = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def get_status(self) -> dict[str, Any]:
        """Get executor status."""
        return {
            "executing": self.is_executing,
            "current_mission": (
                self._current_mission.to_dict()
                if self._current_mission
                else None
            ),
            "current_position": self._current_position,
            "pending_detections": len(self._pending_detections),
            "confirmed_detections": len(self._confirmed_detections),
        }


def create_patrol_mission(
    mission_id: str,
    name: str,
    waypoints: list[tuple[float, float, float]],
    loop_count: int = 0,
    scan_at_waypoints: bool = True,
    hold_time: float = 5.0,
) -> Mission:
    """
    Helper to create a patrol mission.

    Args:
        mission_id: Unique mission ID.
        name: Mission name.
        waypoints: List of (lat, lon, alt) tuples.
        loop_count: Number of loops (0 = infinite).
        scan_at_waypoints: Whether to scan for targets at each waypoint.
        hold_time: Time to hover at each waypoint.

    Returns:
        Configured Mission object.
    """
    mission_waypoints = [
        MissionWaypoint(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            hold_time_sec=hold_time,
            action=WaypointAction.HOVER,
            scan_on_arrival=scan_at_waypoints,
        )
        for lat, lon, alt in waypoints
    ]

    return Mission(
        id=mission_id,
        name=name,
        mission_type=MissionType.PATROL,
        waypoints=mission_waypoints,
        loop_count=loop_count,
        pause_on_detection=True,
    )

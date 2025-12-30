"""
Safety systems for drone flight.

Provides geofencing, battery monitoring, signal loss detection,
and other safety-critical functionality.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Any
import structlog

from drone_flight_core.core.config import SafetyConfig

logger = structlog.get_logger(__name__)


class SafetyViolationType(Enum):
    """Types of safety violations."""

    GEOFENCE_ALTITUDE = auto()
    GEOFENCE_DISTANCE = auto()
    GEOFENCE_NO_FLY_ZONE = auto()
    LOW_BATTERY = auto()
    CRITICAL_BATTERY = auto()
    SIGNAL_LOSS = auto()
    SENSOR_FAILURE = auto()
    MOTOR_FAILURE = auto()
    GPS_LOSS = auto()


class SafetyAction(Enum):
    """Actions to take on safety violations."""

    NONE = auto()
    WARN = auto()
    HOVER = auto()
    RETURN_TO_HOME = auto()
    EMERGENCY_LAND = auto()
    EMERGENCY_STOP = auto()


@dataclass
class SafetyViolation:
    """Record of a safety violation."""

    violation_type: SafetyViolationType
    timestamp: datetime
    severity: str  # "warning", "critical", "emergency"
    message: str
    recommended_action: SafetyAction
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeofenceZone:
    """Defines a geofence zone (no-fly zone or allowed zone)."""

    name: str
    latitude: float
    longitude: float
    radius_m: float
    is_no_fly: bool = True  # True = no-fly zone, False = allowed zone
    min_altitude_m: float = 0.0
    max_altitude_m: float = float("inf")

    def contains_point(self, lat: float, lon: float, alt: float = 0.0) -> bool:
        """Check if a point is within this zone."""
        distance = self._haversine_distance(lat, lon)
        in_radius = distance <= self.radius_m
        in_altitude = self.min_altitude_m <= alt <= self.max_altitude_m
        return in_radius and in_altitude

    def _haversine_distance(self, lat: float, lon: float) -> float:
        """Calculate distance between two GPS coordinates in meters."""
        R = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(lat)
        delta_lat = math.radians(lat - self.latitude)
        delta_lon = math.radians(lon - self.longitude)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


@dataclass
class DroneState:
    """Current state of the drone for safety monitoring."""

    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0
    battery_percent: float = 100.0
    battery_voltage: float = 0.0
    gps_fix: bool = False
    gps_satellites: int = 0
    signal_strength_dbm: float = 0.0
    last_heartbeat: datetime | None = None
    armed: bool = False
    in_flight: bool = False


class SafetyMonitor:
    """
    Monitors drone state and enforces safety constraints.

    Continuously checks for safety violations and triggers
    appropriate responses based on configuration.
    """

    def __init__(
        self,
        config: SafetyConfig,
        on_violation: Callable[[SafetyViolation], None] | None = None,
    ) -> None:
        self._config = config
        self._on_violation = on_violation
        self._drone_state = DroneState()
        self._home_position: tuple[float, float, float] | None = None
        self._no_fly_zones: list[GeofenceZone] = []
        self._violations: list[SafetyViolation] = []
        self._active_violations: set[SafetyViolationType] = set()
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None

        # Parse no-fly zones from config
        self._load_no_fly_zones()

        logger.info("Safety monitor initialized", config=config.model_dump())

    def _load_no_fly_zones(self) -> None:
        """Load no-fly zones from configuration."""
        for zone_data in self._config.geofence.no_fly_zones:
            zone = GeofenceZone(
                name=zone_data.get("name", "Unknown"),
                latitude=zone_data["latitude"],
                longitude=zone_data["longitude"],
                radius_m=zone_data["radius_m"],
                is_no_fly=zone_data.get("is_no_fly", True),
                min_altitude_m=zone_data.get("min_altitude_m", 0),
                max_altitude_m=zone_data.get("max_altitude_m", float("inf")),
            )
            self._no_fly_zones.append(zone)
            logger.debug("Loaded no-fly zone", zone=zone.name)

    def set_home_position(self, lat: float, lon: float, alt: float = 0.0) -> None:
        """Set the home position for geofence distance calculations."""
        self._home_position = (lat, lon, alt)
        # Also update config
        self._config.geofence.home_latitude = lat
        self._config.geofence.home_longitude = lon
        logger.info("Home position set", latitude=lat, longitude=lon, altitude=alt)

    def add_no_fly_zone(self, zone: GeofenceZone) -> None:
        """Add a no-fly zone."""
        self._no_fly_zones.append(zone)
        logger.info("No-fly zone added", zone=zone.name)

    def remove_no_fly_zone(self, zone_name: str) -> bool:
        """Remove a no-fly zone by name."""
        for i, zone in enumerate(self._no_fly_zones):
            if zone.name == zone_name:
                self._no_fly_zones.pop(i)
                logger.info("No-fly zone removed", zone=zone_name)
                return True
        return False

    def update_drone_state(self, **kwargs: Any) -> None:
        """Update current drone state."""
        for key, value in kwargs.items():
            if hasattr(self._drone_state, key):
                setattr(self._drone_state, key, value)

    def check_all_safety(self) -> list[SafetyViolation]:
        """
        Perform all safety checks and return any violations.

        Returns:
            List of current safety violations.
        """
        violations = []

        if self._config.geofence.enabled:
            violations.extend(self._check_geofence())

        violations.extend(self._check_battery())
        violations.extend(self._check_signal())
        violations.extend(self._check_gps())

        # Update active violations
        self._active_violations = {v.violation_type for v in violations}

        return violations

    def _check_geofence(self) -> list[SafetyViolation]:
        """Check geofence constraints."""
        violations = []
        state = self._drone_state

        # Check altitude
        if state.altitude_m > self._config.geofence.max_altitude_m:
            violations.append(
                SafetyViolation(
                    violation_type=SafetyViolationType.GEOFENCE_ALTITUDE,
                    timestamp=datetime.now(),
                    severity="critical",
                    message=f"Altitude {state.altitude_m:.1f}m exceeds maximum {self._config.geofence.max_altitude_m:.1f}m",
                    recommended_action=SafetyAction.RETURN_TO_HOME,
                    metadata={"current_altitude": state.altitude_m},
                )
            )

        # Check distance from home
        if self._home_position:
            home_zone = GeofenceZone(
                name="Home",
                latitude=self._home_position[0],
                longitude=self._home_position[1],
                radius_m=self._config.geofence.max_distance_m,
                is_no_fly=False,
            )
            distance = home_zone._haversine_distance(state.latitude, state.longitude)

            if distance > self._config.geofence.max_distance_m:
                violations.append(
                    SafetyViolation(
                        violation_type=SafetyViolationType.GEOFENCE_DISTANCE,
                        timestamp=datetime.now(),
                        severity="critical",
                        message=f"Distance {distance:.1f}m exceeds maximum {self._config.geofence.max_distance_m:.1f}m from home",
                        recommended_action=SafetyAction.RETURN_TO_HOME,
                        metadata={"current_distance": distance},
                    )
                )

        # Check no-fly zones
        for zone in self._no_fly_zones:
            if zone.is_no_fly and zone.contains_point(
                state.latitude, state.longitude, state.altitude_m
            ):
                violations.append(
                    SafetyViolation(
                        violation_type=SafetyViolationType.GEOFENCE_NO_FLY_ZONE,
                        timestamp=datetime.now(),
                        severity="emergency",
                        message=f"Entered no-fly zone: {zone.name}",
                        recommended_action=SafetyAction.EMERGENCY_LAND,
                        metadata={"zone_name": zone.name},
                    )
                )

        return violations

    def _check_battery(self) -> list[SafetyViolation]:
        """Check battery levels."""
        violations = []
        battery = self._drone_state.battery_percent

        if battery <= self._config.critical_battery_threshold_percent:
            violations.append(
                SafetyViolation(
                    violation_type=SafetyViolationType.CRITICAL_BATTERY,
                    timestamp=datetime.now(),
                    severity="emergency",
                    message=f"Critical battery level: {battery:.1f}%",
                    recommended_action=SafetyAction.EMERGENCY_LAND
                    if self._config.emergency_land_on_critical_battery
                    else SafetyAction.RETURN_TO_HOME,
                    metadata={"battery_percent": battery},
                )
            )
        elif battery <= self._config.low_battery_threshold_percent:
            violations.append(
                SafetyViolation(
                    violation_type=SafetyViolationType.LOW_BATTERY,
                    timestamp=datetime.now(),
                    severity="warning",
                    message=f"Low battery level: {battery:.1f}%",
                    recommended_action=SafetyAction.RETURN_TO_HOME
                    if self._config.rth_on_low_battery
                    else SafetyAction.WARN,
                    metadata={"battery_percent": battery},
                )
            )

        return violations

    def _check_signal(self) -> list[SafetyViolation]:
        """Check for signal loss."""
        violations = []

        if self._drone_state.last_heartbeat:
            time_since_heartbeat = (
                datetime.now() - self._drone_state.last_heartbeat
            ).total_seconds()

            if time_since_heartbeat > self._config.signal_loss_timeout_sec:
                violations.append(
                    SafetyViolation(
                        violation_type=SafetyViolationType.SIGNAL_LOSS,
                        timestamp=datetime.now(),
                        severity="critical",
                        message=f"Signal lost for {time_since_heartbeat:.1f} seconds",
                        recommended_action=SafetyAction.RETURN_TO_HOME
                        if self._config.rth_on_signal_loss
                        else SafetyAction.HOVER,
                        metadata={"time_since_heartbeat": time_since_heartbeat},
                    )
                )

        return violations

    def _check_gps(self) -> list[SafetyViolation]:
        """Check GPS status."""
        violations = []

        if not self._drone_state.gps_fix and self._drone_state.in_flight:
            violations.append(
                SafetyViolation(
                    violation_type=SafetyViolationType.GPS_LOSS,
                    timestamp=datetime.now(),
                    severity="critical",
                    message="GPS fix lost during flight",
                    recommended_action=SafetyAction.HOVER,
                    metadata={"satellites": self._drone_state.gps_satellites},
                )
            )

        return violations

    async def start_monitoring(self, interval_sec: float = 0.5) -> None:
        """Start continuous safety monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_sec))
        logger.info("Safety monitoring started", interval=interval_sec)

    async def stop_monitoring(self) -> None:
        """Stop continuous safety monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Safety monitoring stopped")

    async def _monitor_loop(self, interval_sec: float) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                violations = self.check_all_safety()

                for violation in violations:
                    # Only report new violations
                    if violation.violation_type not in self._active_violations:
                        self._violations.append(violation)
                        logger.warning(
                            "Safety violation detected",
                            type=violation.violation_type.name,
                            severity=violation.severity,
                            message=violation.message,
                        )

                        if self._on_violation:
                            self._on_violation(violation)

                await asyncio.sleep(interval_sec)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in safety monitor", error=str(e))
                await asyncio.sleep(interval_sec)

    def get_highest_priority_action(self) -> SafetyAction:
        """Get the highest priority action from current violations."""
        if not self._active_violations:
            return SafetyAction.NONE

        violations = self.check_all_safety()
        if not violations:
            return SafetyAction.NONE

        # Priority order (highest first)
        action_priority = [
            SafetyAction.EMERGENCY_STOP,
            SafetyAction.EMERGENCY_LAND,
            SafetyAction.RETURN_TO_HOME,
            SafetyAction.HOVER,
            SafetyAction.WARN,
            SafetyAction.NONE,
        ]

        recommended_actions = [v.recommended_action for v in violations]

        for action in action_priority:
            if action in recommended_actions:
                return action

        return SafetyAction.NONE

    def is_safe_to_fly(self) -> tuple[bool, str]:
        """
        Check if it's safe to initiate flight.

        Returns:
            Tuple of (is_safe, reason)
        """
        state = self._drone_state

        if state.battery_percent <= self._config.low_battery_threshold_percent:
            return False, f"Battery too low: {state.battery_percent:.1f}%"

        if not state.gps_fix:
            return False, "No GPS fix"

        if state.gps_satellites < 6:
            return False, f"Insufficient GPS satellites: {state.gps_satellites}"

        return True, "All preflight checks passed"

    def to_dict(self) -> dict[str, Any]:
        """Export safety status as dictionary."""
        return {
            "monitoring_active": self._monitoring,
            "home_position": self._home_position,
            "no_fly_zones_count": len(self._no_fly_zones),
            "active_violations": [v.name for v in self._active_violations],
            "total_violations_recorded": len(self._violations),
            "highest_priority_action": self.get_highest_priority_action().name,
            "drone_state": {
                "latitude": self._drone_state.latitude,
                "longitude": self._drone_state.longitude,
                "altitude_m": self._drone_state.altitude_m,
                "battery_percent": self._drone_state.battery_percent,
                "gps_fix": self._drone_state.gps_fix,
                "armed": self._drone_state.armed,
                "in_flight": self._drone_state.in_flight,
            },
        }

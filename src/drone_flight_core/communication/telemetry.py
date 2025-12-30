"""
Telemetry data management for drone flight.

Collects, processes, and distributes telemetry data from
the flight controller via MAVLink.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any
import structlog

from drone_flight_core.communication.mavlink_connection import (
    MAVLinkConnection,
    MAVLinkMessage,
)

logger = structlog.get_logger(__name__)


@dataclass
class GPSData:
    """GPS telemetry data."""

    latitude: float = 0.0  # degrees
    longitude: float = 0.0  # degrees
    altitude_msl: float = 0.0  # meters above sea level
    altitude_rel: float = 0.0  # meters above home
    ground_speed: float = 0.0  # m/s
    course: float = 0.0  # degrees
    fix_type: int = 0  # 0=no fix, 2=2D, 3=3D
    satellites_visible: int = 0
    hdop: float = 99.9
    vdop: float = 99.9
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AttitudeData:
    """Attitude telemetry data."""

    roll: float = 0.0  # radians
    pitch: float = 0.0  # radians
    yaw: float = 0.0  # radians
    roll_speed: float = 0.0  # rad/s
    pitch_speed: float = 0.0  # rad/s
    yaw_speed: float = 0.0  # rad/s
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def roll_deg(self) -> float:
        """Roll in degrees."""
        import math
        return math.degrees(self.roll)

    @property
    def pitch_deg(self) -> float:
        """Pitch in degrees."""
        import math
        return math.degrees(self.pitch)

    @property
    def yaw_deg(self) -> float:
        """Yaw in degrees."""
        import math
        return math.degrees(self.yaw)


@dataclass
class BatteryData:
    """Battery telemetry data."""

    voltage: float = 0.0  # volts
    current: float = 0.0  # amps
    remaining_percent: float = 100.0
    consumed_mah: float = 0.0
    temperature: float = 0.0  # Celsius
    cell_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VelocityData:
    """Velocity telemetry data."""

    vx: float = 0.0  # m/s (North)
    vy: float = 0.0  # m/s (East)
    vz: float = 0.0  # m/s (Down)
    ground_speed: float = 0.0  # m/s
    climb_rate: float = 0.0  # m/s (positive = ascending)
    heading: float = 0.0  # degrees
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemStatus:
    """System status data."""

    armed: bool = False
    flight_mode: str = "UNKNOWN"
    custom_mode: int = 0
    system_status: int = 0  # MAV_STATE
    health_flags: int = 0
    error_count: int = 0
    cpu_load: float = 0.0  # percent
    voltage_battery: float = 0.0
    drop_rate_comm: float = 0.0  # percent
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TelemetryData:
    """Complete telemetry snapshot."""

    gps: GPSData = field(default_factory=GPSData)
    attitude: AttitudeData = field(default_factory=AttitudeData)
    battery: BatteryData = field(default_factory=BatteryData)
    velocity: VelocityData = field(default_factory=VelocityData)
    system: SystemStatus = field(default_factory=SystemStatus)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gps": {
                "latitude": self.gps.latitude,
                "longitude": self.gps.longitude,
                "altitude_msl": self.gps.altitude_msl,
                "altitude_rel": self.gps.altitude_rel,
                "ground_speed": self.gps.ground_speed,
                "fix_type": self.gps.fix_type,
                "satellites": self.gps.satellites_visible,
            },
            "attitude": {
                "roll": self.attitude.roll_deg,
                "pitch": self.attitude.pitch_deg,
                "yaw": self.attitude.yaw_deg,
            },
            "battery": {
                "voltage": self.battery.voltage,
                "current": self.battery.current,
                "remaining": self.battery.remaining_percent,
            },
            "velocity": {
                "ground_speed": self.velocity.ground_speed,
                "climb_rate": self.velocity.climb_rate,
                "heading": self.velocity.heading,
            },
            "system": {
                "armed": self.system.armed,
                "flight_mode": self.system.flight_mode,
                "status": self.system.system_status,
            },
            "timestamp": self.timestamp.isoformat(),
        }


class TelemetryManager:
    """
    Manages telemetry data collection and distribution.

    Subscribes to MAVLink messages and maintains current
    telemetry state with update callbacks.
    """

    def __init__(self, mavlink: MAVLinkConnection) -> None:
        self._mavlink = mavlink
        self._telemetry = TelemetryData()
        self._callbacks: list[Callable[[TelemetryData], None]] = []
        self._running = False
        self._update_count = 0

        # Register MAVLink message handlers
        self._register_handlers()

        logger.info("Telemetry manager initialized")

    @property
    def current(self) -> TelemetryData:
        """Get current telemetry snapshot."""
        self._telemetry.timestamp = datetime.now()
        return self._telemetry

    @property
    def update_count(self) -> int:
        """Number of telemetry updates received."""
        return self._update_count

    def subscribe(self, callback: Callable[[TelemetryData], None]) -> None:
        """Subscribe to telemetry updates."""
        self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[TelemetryData], None]) -> None:
        """Unsubscribe from telemetry updates."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def _register_handlers(self) -> None:
        """Register handlers for telemetry messages."""
        handlers = {
            "GPS_RAW_INT": self._handle_gps_raw,
            "GLOBAL_POSITION_INT": self._handle_global_position,
            "ATTITUDE": self._handle_attitude,
            "BATTERY_STATUS": self._handle_battery,
            "SYS_STATUS": self._handle_sys_status,
            "VFR_HUD": self._handle_vfr_hud,
            "HEARTBEAT": self._handle_heartbeat,
            "LOCAL_POSITION_NED": self._handle_local_position,
        }

        for msg_type, handler in handlers.items():
            self._mavlink.register_handler(msg_type, handler)

    def _notify_callbacks(self) -> None:
        """Notify all subscribers of telemetry update."""
        self._update_count += 1
        for callback in self._callbacks:
            try:
                callback(self._telemetry)
            except Exception as e:
                logger.error("Telemetry callback error", error=str(e))

    def _handle_gps_raw(self, msg: MAVLinkMessage) -> None:
        """Handle GPS_RAW_INT message."""
        data = msg.data
        self._telemetry.gps.latitude = data.get("lat", 0) / 1e7
        self._telemetry.gps.longitude = data.get("lon", 0) / 1e7
        self._telemetry.gps.altitude_msl = data.get("alt", 0) / 1000.0
        self._telemetry.gps.fix_type = data.get("fix_type", 0)
        self._telemetry.gps.satellites_visible = data.get("satellites_visible", 0)
        self._telemetry.gps.hdop = data.get("eph", 9999) / 100.0
        self._telemetry.gps.vdop = data.get("epv", 9999) / 100.0
        self._telemetry.gps.timestamp = msg.timestamp
        self._notify_callbacks()

    def _handle_global_position(self, msg: MAVLinkMessage) -> None:
        """Handle GLOBAL_POSITION_INT message."""
        data = msg.data
        self._telemetry.gps.latitude = data.get("lat", 0) / 1e7
        self._telemetry.gps.longitude = data.get("lon", 0) / 1e7
        self._telemetry.gps.altitude_msl = data.get("alt", 0) / 1000.0
        self._telemetry.gps.altitude_rel = data.get("relative_alt", 0) / 1000.0

        # Also update velocity from this message
        self._telemetry.velocity.vx = data.get("vx", 0) / 100.0
        self._telemetry.velocity.vy = data.get("vy", 0) / 100.0
        self._telemetry.velocity.vz = data.get("vz", 0) / 100.0
        self._telemetry.velocity.heading = data.get("hdg", 0) / 100.0
        self._telemetry.velocity.timestamp = msg.timestamp

        self._notify_callbacks()

    def _handle_attitude(self, msg: MAVLinkMessage) -> None:
        """Handle ATTITUDE message."""
        data = msg.data
        self._telemetry.attitude.roll = data.get("roll", 0)
        self._telemetry.attitude.pitch = data.get("pitch", 0)
        self._telemetry.attitude.yaw = data.get("yaw", 0)
        self._telemetry.attitude.roll_speed = data.get("rollspeed", 0)
        self._telemetry.attitude.pitch_speed = data.get("pitchspeed", 0)
        self._telemetry.attitude.yaw_speed = data.get("yawspeed", 0)
        self._telemetry.attitude.timestamp = msg.timestamp
        self._notify_callbacks()

    def _handle_battery(self, msg: MAVLinkMessage) -> None:
        """Handle BATTERY_STATUS message."""
        data = msg.data
        voltages = data.get("voltages", [0])
        if voltages and voltages[0] != 65535:
            self._telemetry.battery.voltage = voltages[0] / 1000.0
            self._telemetry.battery.cell_count = sum(
                1 for v in voltages if v != 65535 and v > 0
            )

        self._telemetry.battery.current = data.get("current_battery", 0) / 100.0
        remaining = data.get("battery_remaining", -1)
        if remaining >= 0:
            self._telemetry.battery.remaining_percent = remaining
        self._telemetry.battery.consumed_mah = data.get("current_consumed", 0)
        self._telemetry.battery.temperature = data.get("temperature", 0) / 100.0
        self._telemetry.battery.timestamp = msg.timestamp
        self._notify_callbacks()

    def _handle_sys_status(self, msg: MAVLinkMessage) -> None:
        """Handle SYS_STATUS message."""
        data = msg.data
        self._telemetry.system.health_flags = data.get("onboard_control_sensors_health", 0)
        self._telemetry.system.error_count = data.get("errors_count1", 0)
        self._telemetry.system.cpu_load = data.get("load", 0) / 10.0
        self._telemetry.system.voltage_battery = data.get("voltage_battery", 0) / 1000.0
        self._telemetry.system.drop_rate_comm = data.get("drop_rate_comm", 0) / 100.0

        # Also update battery voltage from here
        if self._telemetry.battery.voltage == 0:
            self._telemetry.battery.voltage = self._telemetry.system.voltage_battery

        remaining = data.get("battery_remaining", -1)
        if remaining >= 0 and self._telemetry.battery.remaining_percent == 100:
            self._telemetry.battery.remaining_percent = remaining

        self._telemetry.system.timestamp = msg.timestamp
        self._notify_callbacks()

    def _handle_vfr_hud(self, msg: MAVLinkMessage) -> None:
        """Handle VFR_HUD message."""
        data = msg.data
        self._telemetry.velocity.ground_speed = data.get("groundspeed", 0)
        self._telemetry.velocity.climb_rate = data.get("climb", 0)
        self._telemetry.velocity.heading = data.get("heading", 0)
        self._telemetry.gps.ground_speed = data.get("groundspeed", 0)
        self._telemetry.gps.course = data.get("heading", 0)

        # Throttle could be useful
        # throttle = data.get("throttle", 0)

        self._notify_callbacks()

    def _handle_heartbeat(self, msg: MAVLinkMessage) -> None:
        """Handle HEARTBEAT message."""
        data = msg.data

        # Check if armed (base_mode bit 7)
        base_mode = data.get("base_mode", 0)
        self._telemetry.system.armed = bool(base_mode & 128)

        self._telemetry.system.custom_mode = data.get("custom_mode", 0)
        self._telemetry.system.system_status = data.get("system_status", 0)

        # Decode flight mode (simplified - actual mapping depends on autopilot)
        custom_mode = data.get("custom_mode", 0)
        mode_map = {
            0: "MANUAL",
            1: "ALTITUDE",
            2: "POSITION",
            3: "AUTO",
            4: "GUIDED",
            5: "LOITER",
            6: "RTL",
            7: "CIRCLE",
            8: "LAND",
            9: "OFFBOARD",
        }
        self._telemetry.system.flight_mode = mode_map.get(custom_mode, f"MODE_{custom_mode}")

        self._notify_callbacks()

    def _handle_local_position(self, msg: MAVLinkMessage) -> None:
        """Handle LOCAL_POSITION_NED message."""
        data = msg.data
        # Local position is NED (North, East, Down)
        self._telemetry.velocity.vx = data.get("vx", 0)
        self._telemetry.velocity.vy = data.get("vy", 0)
        self._telemetry.velocity.vz = data.get("vz", 0)
        self._notify_callbacks()

    async def request_telemetry_streams(self, rate_hz: int = 10) -> None:
        """Request telemetry data streams from flight controller."""
        from pymavlink import mavutil

        streams = [
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # Attitude
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,  # VFR_HUD
            mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
        ]

        for stream in streams:
            self._mavlink.request_data_stream(stream, rate_hz)
            await asyncio.sleep(0.1)  # Small delay between requests

        logger.info("Telemetry streams requested", rate_hz=rate_hz)

    def to_dict(self) -> dict[str, Any]:
        """Export telemetry manager status."""
        return {
            "update_count": self._update_count,
            "subscriber_count": len(self._callbacks),
            "telemetry": self._telemetry.to_dict(),
        }

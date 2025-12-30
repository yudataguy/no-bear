"""
Command handling for drone control.

Provides high-level command interface with queuing,
acknowledgment handling, and timeout management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable
import structlog

from pymavlink import mavutil

from drone_flight_core.communication.mavlink_connection import (
    MAVLinkConnection,
    MAVLinkMessage,
)

logger = structlog.get_logger(__name__)


class CommandType(Enum):
    """Types of commands."""

    ARM = auto()
    DISARM = auto()
    TAKEOFF = auto()
    LAND = auto()
    RTL = auto()
    GOTO = auto()
    SET_MODE = auto()
    SET_SPEED = auto()
    SET_ALTITUDE = auto()
    SET_HEADING = auto()
    PAUSE = auto()
    RESUME = auto()
    EMERGENCY_STOP = auto()


class CommandStatus(Enum):
    """Status of a command."""

    PENDING = auto()
    SENT = auto()
    ACKNOWLEDGED = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


@dataclass
class Command:
    """Represents a command to be sent to the drone."""

    command_type: CommandType
    params: dict[str, Any] = field(default_factory=dict)
    status: CommandStatus = CommandStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    timeout_sec: float = 5.0
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command_type": self.command_type.name,
            "params": self.params,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


@dataclass
class Waypoint:
    """Represents a navigation waypoint."""

    latitude: float
    longitude: float
    altitude: float
    speed: float | None = None
    hold_time_sec: float = 0.0
    acceptance_radius_m: float = 5.0
    yaw: float | None = None  # None = auto (face direction of travel)


class CommandHandler:
    """
    Handles command execution and acknowledgment.

    Features:
    - Command queuing
    - Acknowledgment handling
    - Timeout management
    - Retry logic
    - Command history
    """

    def __init__(
        self,
        mavlink: MAVLinkConnection,
        default_timeout: float = 5.0,
    ) -> None:
        self._mavlink = mavlink
        self._default_timeout = default_timeout

        # Command tracking
        self._pending_commands: dict[int, Command] = {}
        self._command_history: list[Command] = []
        self._command_sequence = 0
        self._max_history = 1000

        # Callbacks
        self._on_command_complete: list[Callable[[Command], None]] = []

        # Register for command acknowledgments
        self._mavlink.register_handler("COMMAND_ACK", self._handle_command_ack)

        logger.info("Command handler initialized")

    def on_complete(self, callback: Callable[[Command], None]) -> None:
        """Register callback for command completion."""
        self._on_command_complete.append(callback)

    async def execute(self, command: Command) -> Command:
        """
        Execute a command and wait for completion.

        Args:
            command: The command to execute.

        Returns:
            The command with updated status.
        """
        command.status = CommandStatus.PENDING

        try:
            # Send the command
            sent = self._send_command(command)

            if not sent:
                command.status = CommandStatus.FAILED
                command.error = "Failed to send command"
                return command

            command.status = CommandStatus.SENT
            command.sent_at = datetime.now()

            # Wait for acknowledgment with timeout
            try:
                await asyncio.wait_for(
                    self._wait_for_ack(command),
                    timeout=command.timeout_sec,
                )
            except asyncio.TimeoutError:
                if command.retries < command.max_retries:
                    command.retries += 1
                    logger.warning(
                        "Command timeout, retrying",
                        command=command.command_type.name,
                        retry=command.retries,
                    )
                    return await self.execute(command)
                else:
                    command.status = CommandStatus.TIMEOUT
                    command.error = "Command timed out"

        except Exception as e:
            command.status = CommandStatus.FAILED
            command.error = str(e)
            logger.error(
                "Command execution failed",
                command=command.command_type.name,
                error=str(e),
            )

        finally:
            command.completed_at = datetime.now()
            self._add_to_history(command)
            self._notify_complete(command)

        return command

    def _send_command(self, command: Command) -> bool:
        """Send command via MAVLink."""
        cmd_type = command.command_type
        params = command.params

        if cmd_type == CommandType.ARM:
            return self._mavlink.arm(force=params.get("force", False))

        elif cmd_type == CommandType.DISARM:
            return self._mavlink.disarm(force=params.get("force", False))

        elif cmd_type == CommandType.TAKEOFF:
            altitude = params.get("altitude", 10.0)
            return self._mavlink.takeoff(altitude)

        elif cmd_type == CommandType.LAND:
            return self._mavlink.land()

        elif cmd_type == CommandType.RTL:
            return self._mavlink.return_to_launch()

        elif cmd_type == CommandType.GOTO:
            lat = params.get("latitude", 0)
            lon = params.get("longitude", 0)
            alt = params.get("altitude", 30)
            return self._mavlink.send_set_position_target_global_int(lat, lon, alt)

        elif cmd_type == CommandType.SET_MODE:
            mode = params.get("mode", "POSITION")
            return self._mavlink.set_mode(mode)

        elif cmd_type == CommandType.SET_SPEED:
            speed = params.get("speed", 5.0)
            speed_type = params.get("type", 1)  # 1 = ground speed
            return self._mavlink.send_command_long(
                mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                param1=float(speed_type),
                param2=speed,
                param3=-1,  # No throttle change
            )

        elif cmd_type == CommandType.SET_ALTITUDE:
            altitude = params.get("altitude", 30)
            return self._mavlink.send_command_long(
                mavutil.mavlink.MAV_CMD_NAV_CONTINUE_AND_CHANGE_ALT,
                param1=0,  # Climb rate (0 = default)
                param7=altitude,
            )

        elif cmd_type == CommandType.SET_HEADING:
            heading = params.get("heading", 0)
            return self._mavlink.send_command_long(
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                param1=heading,
                param2=0,  # Yaw speed (deg/s, 0 = default)
                param3=1,  # Direction: 1=CW, -1=CCW
                param4=0,  # 0=absolute, 1=relative
            )

        elif cmd_type == CommandType.PAUSE:
            return self._mavlink.send_command_long(
                mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
                param1=0,  # 0 = pause
            )

        elif cmd_type == CommandType.RESUME:
            return self._mavlink.send_command_long(
                mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
                param1=1,  # 1 = resume
            )

        elif cmd_type == CommandType.EMERGENCY_STOP:
            # Kill motors immediately
            return self._mavlink.send_command_long(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                param1=0,  # Disarm
                param2=21196,  # Force
            )

        else:
            logger.warning("Unknown command type", command=cmd_type.name)
            return False

    async def _wait_for_ack(self, command: Command) -> None:
        """Wait for command acknowledgment."""
        # Store in pending commands
        seq = self._command_sequence
        self._command_sequence += 1
        self._pending_commands[seq] = command

        # Wait for status update
        while command.status == CommandStatus.SENT:
            await asyncio.sleep(0.1)

    def _handle_command_ack(self, msg: MAVLinkMessage) -> None:
        """Handle COMMAND_ACK message."""
        data = msg.data
        result = data.get("result", 0)

        # Find matching pending command (simplified - real implementation
        # would match on command ID)
        if self._pending_commands:
            # Get most recent pending command
            seq = max(self._pending_commands.keys())
            command = self._pending_commands.pop(seq)

            if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                command.status = CommandStatus.ACKNOWLEDGED
                command.result = "accepted"
            elif result == mavutil.mavlink.MAV_RESULT_IN_PROGRESS:
                command.status = CommandStatus.ACKNOWLEDGED
                command.result = "in_progress"
            else:
                command.status = CommandStatus.FAILED
                command.error = f"Command rejected: {result}"

    def _add_to_history(self, command: Command) -> None:
        """Add command to history."""
        self._command_history.append(command)
        if len(self._command_history) > self._max_history:
            self._command_history = self._command_history[-self._max_history:]

    def _notify_complete(self, command: Command) -> None:
        """Notify callbacks of command completion."""
        for callback in self._on_command_complete:
            try:
                callback(command)
            except Exception as e:
                logger.error("Command complete callback error", error=str(e))

    # High-level convenience methods

    async def arm(self, force: bool = False) -> Command:
        """Arm the drone."""
        cmd = Command(
            command_type=CommandType.ARM,
            params={"force": force},
        )
        return await self.execute(cmd)

    async def disarm(self, force: bool = False) -> Command:
        """Disarm the drone."""
        cmd = Command(
            command_type=CommandType.DISARM,
            params={"force": force},
        )
        return await self.execute(cmd)

    async def takeoff(self, altitude: float = 10.0) -> Command:
        """Take off to specified altitude."""
        cmd = Command(
            command_type=CommandType.TAKEOFF,
            params={"altitude": altitude},
            timeout_sec=30.0,  # Takeoff can take time
        )
        return await self.execute(cmd)

    async def land(self) -> Command:
        """Land at current position."""
        cmd = Command(
            command_type=CommandType.LAND,
            timeout_sec=60.0,  # Landing can take time
        )
        return await self.execute(cmd)

    async def return_to_launch(self) -> Command:
        """Return to launch position."""
        cmd = Command(
            command_type=CommandType.RTL,
            timeout_sec=10.0,
        )
        return await self.execute(cmd)

    async def goto(
        self,
        latitude: float,
        longitude: float,
        altitude: float = 30.0,
    ) -> Command:
        """Go to specified GPS coordinates."""
        cmd = Command(
            command_type=CommandType.GOTO,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
            },
            timeout_sec=10.0,
        )
        return await self.execute(cmd)

    async def goto_waypoint(self, waypoint: Waypoint) -> Command:
        """Navigate to a waypoint."""
        # Set speed if specified
        if waypoint.speed is not None:
            await self.set_speed(waypoint.speed)

        return await self.goto(
            waypoint.latitude,
            waypoint.longitude,
            waypoint.altitude,
        )

    async def set_mode(self, mode: str) -> Command:
        """Set flight mode."""
        cmd = Command(
            command_type=CommandType.SET_MODE,
            params={"mode": mode},
        )
        return await self.execute(cmd)

    async def set_speed(self, speed: float) -> Command:
        """Set ground speed."""
        cmd = Command(
            command_type=CommandType.SET_SPEED,
            params={"speed": speed},
        )
        return await self.execute(cmd)

    async def set_heading(self, heading: float) -> Command:
        """Set heading in degrees."""
        cmd = Command(
            command_type=CommandType.SET_HEADING,
            params={"heading": heading},
        )
        return await self.execute(cmd)

    async def emergency_stop(self) -> Command:
        """Emergency stop - kills motors immediately."""
        cmd = Command(
            command_type=CommandType.EMERGENCY_STOP,
            max_retries=0,  # Don't retry emergency stop
        )
        return await self.execute(cmd)

    def get_history(self, limit: int = 100) -> list[Command]:
        """Get command history."""
        return self._command_history[-limit:]

    def to_dict(self) -> dict[str, Any]:
        """Export command handler status."""
        return {
            "pending_commands": len(self._pending_commands),
            "total_commands": len(self._command_history),
            "recent_commands": [
                cmd.to_dict() for cmd in self._command_history[-10:]
            ],
        }

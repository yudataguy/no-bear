"""
MAVLink connection management for drone communication.

Provides abstraction over pymavlink for connecting to PX4/ArduPilot
flight controllers via various transport protocols.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Any
import structlog

from pymavlink import mavutil

from drone_flight_core.core.config import CommunicationConfig

logger = structlog.get_logger(__name__)


class ConnectionState(Enum):
    """State of MAVLink connection."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    LOST = auto()
    ERROR = auto()


@dataclass
class HeartbeatInfo:
    """Information from MAVLink heartbeat."""

    system_id: int = 0
    component_id: int = 0
    autopilot_type: int = 0
    vehicle_type: int = 0
    base_mode: int = 0
    custom_mode: int = 0
    system_status: int = 0
    mavlink_version: int = 0
    last_received: datetime | None = None


@dataclass
class MAVLinkMessage:
    """Wrapper for received MAVLink messages."""

    msg_type: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    system_id: int = 0
    component_id: int = 0


class MAVLinkConnection:
    """
    Manages MAVLink connection to flight controller.

    Supports multiple connection types:
    - Serial: /dev/ttyUSB0, /dev/ttyACM0
    - UDP: udp:ip:port
    - TCP: tcp:ip:port

    Handles heartbeat exchange, message routing, and connection monitoring.
    """

    def __init__(
        self,
        config: CommunicationConfig,
        on_message: Callable[[MAVLinkMessage], None] | None = None,
        on_state_change: Callable[[ConnectionState], None] | None = None,
    ) -> None:
        self._config = config
        self._on_message = on_message
        self._on_state_change = on_state_change

        self._connection: mavutil.mavlink_connection | None = None
        self._state = ConnectionState.DISCONNECTED
        self._heartbeat_info = HeartbeatInfo()
        self._last_heartbeat_sent: datetime | None = None
        self._last_heartbeat_received: datetime | None = None

        # Message handlers by type
        self._message_handlers: dict[str, list[Callable]] = {}

        # Threading
        self._receive_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # Stats
        self._messages_received = 0
        self._messages_sent = 0
        self._connection_attempts = 0

        logger.info(
            "MAVLink connection initialized",
            connection_string=config.mavlink_connection,
        )

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def heartbeat_info(self) -> HeartbeatInfo:
        """Get last heartbeat information."""
        return self._heartbeat_info

    @property
    def time_since_heartbeat(self) -> float | None:
        """Seconds since last heartbeat received."""
        if self._last_heartbeat_received:
            return (datetime.now() - self._last_heartbeat_received).total_seconds()
        return None

    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state and notify listeners."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            logger.info(
                "Connection state changed",
                old_state=old_state.name,
                new_state=new_state.name,
            )
            if self._on_state_change:
                self._on_state_change(new_state)

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Establish connection to flight controller.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if connection successful.
        """
        self._connection_attempts += 1
        self._set_state(ConnectionState.CONNECTING)

        try:
            logger.info(
                "Connecting to flight controller",
                connection=self._config.mavlink_connection,
            )

            self._connection = mavutil.mavlink_connection(
                self._config.mavlink_connection,
                baud=self._config.mavlink_baud_rate,
                source_system=255,  # Ground station system ID
                source_component=0,
            )

            # Wait for heartbeat
            logger.debug("Waiting for heartbeat...")
            msg = self._connection.wait_heartbeat(timeout=timeout)

            if msg:
                self._process_heartbeat(msg)
                self._set_state(ConnectionState.CONNECTED)
                self._running = True
                self._start_threads()
                logger.info(
                    "Connected to flight controller",
                    system_id=self._heartbeat_info.system_id,
                    autopilot=self._heartbeat_info.autopilot_type,
                )
                return True
            else:
                logger.warning("No heartbeat received")
                self._set_state(ConnectionState.ERROR)
                return False

        except Exception as e:
            logger.error("Connection failed", error=str(e))
            self._set_state(ConnectionState.ERROR)
            return False

    def disconnect(self) -> None:
        """Disconnect from flight controller."""
        self._running = False

        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)

        if self._connection:
            self._connection.close()
            self._connection = None

        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("Disconnected from flight controller")

    def _start_threads(self) -> None:
        """Start receive and heartbeat threads."""
        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="mavlink-receive"
        )
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="mavlink-heartbeat"
        )

        self._receive_thread.start()
        self._heartbeat_thread.start()

    def _receive_loop(self) -> None:
        """Background thread for receiving messages."""
        while self._running and self._connection:
            try:
                msg = self._connection.recv_match(blocking=True, timeout=1.0)

                if msg:
                    self._messages_received += 1
                    self._process_message(msg)

            except Exception as e:
                if self._running:
                    logger.error("Receive error", error=str(e))
                    time.sleep(0.1)

    def _heartbeat_loop(self) -> None:
        """Background thread for sending heartbeats."""
        while self._running and self._connection:
            try:
                self._send_heartbeat()

                # Check for connection loss
                if self._last_heartbeat_received:
                    time_since = (
                        datetime.now() - self._last_heartbeat_received
                    ).total_seconds()
                    if time_since > self._config.command_timeout_sec:
                        if self._state == ConnectionState.CONNECTED:
                            self._set_state(ConnectionState.LOST)
                            logger.warning(
                                "Connection lost",
                                time_since_heartbeat=time_since,
                            )

                time.sleep(self._config.heartbeat_interval_sec)

            except Exception as e:
                if self._running:
                    logger.error("Heartbeat error", error=str(e))
                    time.sleep(1.0)

    def _send_heartbeat(self) -> None:
        """Send heartbeat to flight controller."""
        if not self._connection:
            return

        self._connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,  # Ground station
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,  # base_mode
            0,  # custom_mode
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        self._last_heartbeat_sent = datetime.now()
        self._messages_sent += 1

    def _process_message(self, msg: Any) -> None:
        """Process received MAVLink message."""
        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            self._process_heartbeat(msg)
            return

        # Wrap message
        wrapped = MAVLinkMessage(
            msg_type=msg_type,
            data=msg.to_dict(),
            timestamp=datetime.now(),
            system_id=msg.get_srcSystem(),
            component_id=msg.get_srcComponent(),
        )

        # Call registered handlers
        if msg_type in self._message_handlers:
            for handler in self._message_handlers[msg_type]:
                try:
                    handler(wrapped)
                except Exception as e:
                    logger.error(
                        "Message handler error",
                        msg_type=msg_type,
                        error=str(e),
                    )

        # Call global handler
        if self._on_message:
            self._on_message(wrapped)

    def _process_heartbeat(self, msg: Any) -> None:
        """Process heartbeat message."""
        self._heartbeat_info = HeartbeatInfo(
            system_id=msg.get_srcSystem(),
            component_id=msg.get_srcComponent(),
            autopilot_type=msg.autopilot,
            vehicle_type=msg.type,
            base_mode=msg.base_mode,
            custom_mode=msg.custom_mode,
            system_status=msg.system_status,
            mavlink_version=msg.mavlink_version,
            last_received=datetime.now(),
        )
        self._last_heartbeat_received = datetime.now()

        # Restore connection if it was lost
        if self._state == ConnectionState.LOST:
            self._set_state(ConnectionState.CONNECTED)
            logger.info("Connection restored")

    def register_handler(
        self, msg_type: str, handler: Callable[[MAVLinkMessage], None]
    ) -> None:
        """
        Register a handler for a specific message type.

        Args:
            msg_type: MAVLink message type (e.g., "ATTITUDE", "GPS_RAW_INT")
            handler: Callback function to handle the message.
        """
        if msg_type not in self._message_handlers:
            self._message_handlers[msg_type] = []
        self._message_handlers[msg_type].append(handler)

    def unregister_handler(
        self, msg_type: str, handler: Callable[[MAVLinkMessage], None]
    ) -> None:
        """Unregister a message handler."""
        if msg_type in self._message_handlers:
            try:
                self._message_handlers[msg_type].remove(handler)
            except ValueError:
                pass

    def send_command_long(
        self,
        command: int,
        param1: float = 0,
        param2: float = 0,
        param3: float = 0,
        param4: float = 0,
        param5: float = 0,
        param6: float = 0,
        param7: float = 0,
        target_system: int | None = None,
        target_component: int = 0,
    ) -> bool:
        """
        Send a MAVLink COMMAND_LONG message.

        Args:
            command: MAVLink command ID
            param1-7: Command parameters
            target_system: Target system ID (default: connected system)
            target_component: Target component ID

        Returns:
            True if command was sent successfully.
        """
        if not self._connection:
            logger.error("Cannot send command: not connected")
            return False

        if target_system is None:
            target_system = self._heartbeat_info.system_id

        try:
            self._connection.mav.command_long_send(
                target_system,
                target_component,
                command,
                0,  # confirmation
                param1,
                param2,
                param3,
                param4,
                param5,
                param6,
                param7,
            )
            self._messages_sent += 1
            logger.debug("Command sent", command=command)
            return True

        except Exception as e:
            logger.error("Failed to send command", command=command, error=str(e))
            return False

    def send_set_position_target_global_int(
        self,
        lat: float,
        lon: float,
        alt: float,
        vx: float = 0,
        vy: float = 0,
        vz: float = 0,
        yaw: float = 0,
        coordinate_frame: int = 6,  # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    ) -> bool:
        """
        Send position target command.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            vx, vy, vz: Velocity components (m/s)
            yaw: Yaw angle in radians
            coordinate_frame: MAVLink coordinate frame

        Returns:
            True if command was sent successfully.
        """
        if not self._connection:
            return False

        try:
            # Convert to integer format (1e7 for lat/lon)
            lat_int = int(lat * 1e7)
            lon_int = int(lon * 1e7)

            self._connection.mav.set_position_target_global_int_send(
                0,  # time_boot_ms
                self._heartbeat_info.system_id,
                0,  # target_component
                coordinate_frame,
                0b0000111111111000,  # type_mask (position only)
                lat_int,
                lon_int,
                alt,
                vx,
                vy,
                vz,
                0,  # afx
                0,  # afy
                0,  # afz
                yaw,
                0,  # yaw_rate
            )
            self._messages_sent += 1
            return True

        except Exception as e:
            logger.error("Failed to send position target", error=str(e))
            return False

    def request_data_stream(
        self,
        stream_id: int,
        rate_hz: int,
        start: bool = True,
    ) -> bool:
        """
        Request a data stream from the flight controller.

        Args:
            stream_id: MAVLink data stream ID
            rate_hz: Desired rate in Hz
            start: True to start, False to stop

        Returns:
            True if request was sent successfully.
        """
        if not self._connection:
            return False

        try:
            self._connection.mav.request_data_stream_send(
                self._heartbeat_info.system_id,
                0,  # target_component
                stream_id,
                rate_hz,
                1 if start else 0,
            )
            self._messages_sent += 1
            return True

        except Exception as e:
            logger.error("Failed to request data stream", error=str(e))
            return False

    def arm(self, force: bool = False) -> bool:
        """
        Arm the vehicle.

        Args:
            force: Force arm even if preflight checks fail.

        Returns:
            True if arm command was sent.
        """
        param2 = 21196.0 if force else 0.0
        return self.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,  # Arm
            param2=param2,
        )

    def disarm(self, force: bool = False) -> bool:
        """
        Disarm the vehicle.

        Args:
            force: Force disarm.

        Returns:
            True if disarm command was sent.
        """
        param2 = 21196.0 if force else 0.0
        return self.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,  # Disarm
            param2=param2,
        )

    def takeoff(self, altitude: float) -> bool:
        """
        Command takeoff to specified altitude.

        Args:
            altitude: Target altitude in meters.

        Returns:
            True if takeoff command was sent.
        """
        return self.send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            param7=altitude,
        )

    def land(self) -> bool:
        """
        Command landing at current position.

        Returns:
            True if land command was sent.
        """
        return self.send_command_long(mavutil.mavlink.MAV_CMD_NAV_LAND)

    def return_to_launch(self) -> bool:
        """
        Command return to launch (home).

        Returns:
            True if RTL command was sent.
        """
        return self.send_command_long(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)

    def set_mode(self, mode: str | int) -> bool:
        """
        Set flight mode.

        Args:
            mode: Flight mode name or custom_mode integer.

        Returns:
            True if mode change command was sent.
        """
        if isinstance(mode, str):
            # Map common mode names to custom_mode values
            # These are PX4 mode values; ArduPilot uses different values
            mode_map = {
                "MANUAL": 0,
                "STABILIZED": 7,
                "ALTITUDE": 2,
                "POSITION": 3,
                "OFFBOARD": 6,
                "LAND": 4,
                "RTL": 5,
                "HOLD": 3,
            }
            mode_value = mode_map.get(mode.upper(), 0)
        else:
            mode_value = mode

        return self.send_command_long(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            param1=1.0,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            param2=float(mode_value),
        )

    def to_dict(self) -> dict[str, Any]:
        """Export connection status as dictionary."""
        return {
            "state": self._state.name,
            "is_connected": self.is_connected,
            "connection_string": self._config.mavlink_connection,
            "system_id": self._heartbeat_info.system_id,
            "autopilot_type": self._heartbeat_info.autopilot_type,
            "vehicle_type": self._heartbeat_info.vehicle_type,
            "time_since_heartbeat": self.time_since_heartbeat,
            "messages_received": self._messages_received,
            "messages_sent": self._messages_sent,
            "connection_attempts": self._connection_attempts,
        }


async def create_mavlink_connection(
    config: CommunicationConfig,
    on_message: Callable[[MAVLinkMessage], None] | None = None,
    timeout: float = 10.0,
) -> MAVLinkConnection:
    """
    Async factory function to create and connect MAVLink connection.

    Args:
        config: Communication configuration.
        on_message: Optional message callback.
        timeout: Connection timeout.

    Returns:
        Connected MAVLinkConnection instance.

    Raises:
        ConnectionError: If connection fails.
    """
    connection = MAVLinkConnection(config, on_message)

    # Run connect in thread pool since pymavlink is blocking
    loop = asyncio.get_event_loop()
    connected = await loop.run_in_executor(None, connection.connect, timeout)

    if not connected:
        raise ConnectionError("Failed to connect to MAVLink")

    return connection

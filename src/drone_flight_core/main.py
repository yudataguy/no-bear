"""
Main entry point for drone flight core.

Initializes all subsystems and runs the main control loop.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

import structlog

from drone_flight_core.core.config import DroneConfig, load_config
from drone_flight_core.core.state_machine import FlightStateMachine, FlightState
from drone_flight_core.core.safety import SafetyMonitor, SafetyAction
from drone_flight_core.communication.mavlink_connection import MAVLinkConnection
from drone_flight_core.communication.telemetry import TelemetryManager
from drone_flight_core.communication.command_handler import CommandHandler
from drone_flight_core.vision.camera_manager import CameraManager
from drone_flight_core.vision.detector import ObjectDetector
from drone_flight_core.vision.tracker import ObjectTracker
from drone_flight_core.vision.fusion import SensorFusion
from drone_flight_core.data.logger import setup_logging, EventLogger, TelemetryLogger
from drone_flight_core.data.flight_recorder import FlightRecorder
from drone_flight_core.ground_station.server import create_app, GroundStation

logger = structlog.get_logger(__name__)


class DroneFlightCore:
    """
    Main drone flight control application.

    Coordinates all subsystems:
    - State machine for flight control
    - Safety monitoring
    - MAVLink communication
    - Vision processing
    - Data logging
    - Ground station API
    """

    def __init__(self, config: DroneConfig) -> None:
        self.config = config
        self._running = False

        # Initialize logging
        setup_logging(config.logging)

        logger.info(
            "Initializing drone flight core",
            drone_id=config.drone_id,
            simulation_mode=config.simulation_mode,
        )

        # Core components
        self.state_machine = FlightStateMachine()
        self.safety_monitor = SafetyMonitor(
            config.safety,
            on_violation=self._handle_safety_violation,
        )

        # Communication (initialized on connect)
        self.mavlink: MAVLinkConnection | None = None
        self.telemetry: TelemetryManager | None = None
        self.commands: CommandHandler | None = None

        # Vision (initialized when cameras available)
        self.camera_manager: CameraManager | None = None
        self.detector: ObjectDetector | None = None
        self.tracker: ObjectTracker | None = None
        self.fusion: SensorFusion | None = None

        # Data logging
        self.event_logger = EventLogger(Path(config.logging.log_dir))
        self.telemetry_logger = TelemetryLogger(
            Path(config.logging.log_dir),
            rate_hz=config.logging.telemetry_log_rate_hz,
        )
        self.flight_recorder: FlightRecorder | None = None

        # Ground station
        self.ground_station = GroundStation(config)

        # Register state callbacks
        self._register_callbacks()

        logger.info("Drone flight core initialized")

    def _register_callbacks(self) -> None:
        """Register state machine callbacks."""
        # Log all state transitions
        self.state_machine.on_transition(self._on_state_transition)

        # State-specific callbacks
        self.state_machine.on_enter(FlightState.TAKING_OFF, self._on_takeoff)
        self.state_machine.on_enter(FlightState.LANDED, self._on_landed)
        self.state_machine.on_enter(FlightState.EMERGENCY_LAND, self._on_emergency)

    def _on_state_transition(self, transition: Any) -> None:
        """Handle state transition."""
        self.event_logger.log_state_change(
            from_state=transition.from_state.name,
            to_state=transition.to_state.name,
            reason=transition.reason,
        )

    async def _on_takeoff(self) -> None:
        """Handle takeoff state entry."""
        logger.info("Takeoff initiated")

        # Start flight recording
        if self.flight_recorder is None:
            self.flight_recorder = FlightRecorder(self.config.logging)
            self.flight_recorder.start()

    async def _on_landed(self) -> None:
        """Handle landed state entry."""
        logger.info("Landed")

        # Stop flight recording
        if self.flight_recorder:
            self.flight_recorder.stop()
            self.flight_recorder = None

    async def _on_emergency(self) -> None:
        """Handle emergency state entry."""
        logger.warning("Emergency landing initiated")

    def _handle_safety_violation(self, violation: Any) -> None:
        """Handle safety violation from monitor."""
        self.event_logger.log_safety_violation(
            violation_type=violation.violation_type.name,
            message=violation.message,
            action=violation.recommended_action.name,
        )

        # Take automatic action for serious violations
        if violation.recommended_action == SafetyAction.EMERGENCY_LAND:
            asyncio.create_task(self.state_machine.emergency_land(violation.message))
        elif violation.recommended_action == SafetyAction.RETURN_TO_HOME:
            if self.state_machine.can_transition_to(FlightState.RETURN_TO_HOME):
                asyncio.create_task(self.state_machine.return_to_home(violation.message))

    async def connect_mavlink(self) -> bool:
        """Connect to flight controller via MAVLink."""
        logger.info("Connecting to flight controller...")

        try:
            self.mavlink = MAVLinkConnection(self.config.communication)

            if not self.mavlink.connect():
                logger.error("Failed to connect to flight controller")
                return False

            # Initialize telemetry and command handlers
            self.telemetry = TelemetryManager(self.mavlink)
            self.commands = CommandHandler(self.mavlink)

            # Subscribe to telemetry updates
            self.telemetry.subscribe(self._on_telemetry_update)

            # Request data streams
            await self.telemetry.request_telemetry_streams(
                rate_hz=int(self.config.communication.telemetry_rate_hz)
            )

            logger.info("Connected to flight controller")
            return True

        except Exception as e:
            logger.error("MAVLink connection failed", error=str(e))
            return False

    def _on_telemetry_update(self, telemetry: Any) -> None:
        """Handle telemetry update."""
        # Update safety monitor with drone state
        self.safety_monitor.update_drone_state(
            latitude=telemetry.gps.latitude,
            longitude=telemetry.gps.longitude,
            altitude_m=telemetry.gps.altitude_rel,
            battery_percent=telemetry.battery.remaining_percent,
            gps_fix=telemetry.gps.fix_type >= 3,
            gps_satellites=telemetry.gps.satellites_visible,
            armed=telemetry.system.armed,
            in_flight=self.state_machine.is_flying,
        )

        # Log telemetry
        self.telemetry_logger.log(telemetry.to_dict())

        # Update ground station
        asyncio.create_task(
            self.ground_station.update_telemetry(telemetry.to_dict())
        )

        # Record if flight active
        if self.flight_recorder:
            self.flight_recorder.log_telemetry(telemetry)

    async def initialize_vision(self) -> bool:
        """Initialize vision system."""
        if self.config.simulation_mode:
            logger.info("Skipping vision initialization in simulation mode")
            return True

        logger.info("Initializing vision system...")

        try:
            self.camera_manager = CameraManager(self.config.vision)

            if not self.camera_manager.initialize_cameras():
                logger.warning("No cameras available")
                return False

            self.detector = ObjectDetector(self.config.vision)
            self.tracker = ObjectTracker(self.config.vision)
            self.fusion = SensorFusion(self.config.vision)

            self.camera_manager.start_streaming()

            logger.info("Vision system initialized")
            return True

        except Exception as e:
            logger.error("Vision initialization failed", error=str(e))
            return False

    async def run_vision_loop(self) -> None:
        """Run the vision processing loop."""
        if not self.camera_manager:
            return

        logger.info("Starting vision processing loop")

        while self._running:
            try:
                # Get frames
                rgb_frame, thermal_frame = self.camera_manager.get_synchronized_frames()

                if rgb_frame is None and thermal_frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Run detection
                rgb_result, thermal_result = self.detector.detect_both(
                    rgb_frame, thermal_frame
                )

                # Fuse detections
                fused_detections = self.fusion.fuse(rgb_result, thermal_result)

                # Update tracker
                from drone_flight_core.vision.detector import Detection
                detections = [
                    Detection(
                        bbox=fd.bbox,
                        confidence=fd.confidence,
                        class_id=fd.class_id,
                        class_name=fd.class_name,
                    )
                    for fd in fused_detections
                ]

                tracks = self.tracker.update(detections)

                # Log detections
                for track in tracks:
                    if self.flight_recorder:
                        self.flight_recorder.log_detection(
                            track,
                            frame_number=rgb_frame.frame_number if rgb_frame else 0,
                        )

                # Report processing time for adaptive FPS
                if rgb_result:
                    self.camera_manager.report_processing_time(
                        rgb_result.processing_time_ms
                    )

                # Sleep based on current FPS
                await asyncio.sleep(self.camera_manager.frame_interval_sec)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Vision processing error", error=str(e))
                await asyncio.sleep(0.1)

    async def run(self) -> None:
        """Main run loop."""
        self._running = True

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        logger.info("Starting drone flight core...")

        try:
            # Start safety monitoring
            await self.safety_monitor.start_monitoring()

            # Start ground station
            await self.ground_station.start()

            # Connect to flight controller (if not simulation)
            if not self.config.simulation_mode:
                if not await self.connect_mavlink():
                    logger.warning("Running without MAVLink connection")

                # Initialize vision
                await self.initialize_vision()

            # Create background tasks
            tasks = [
                asyncio.create_task(self._main_loop()),
            ]

            if self.camera_manager:
                tasks.append(asyncio.create_task(self.run_vision_loop()))

            logger.info("Drone flight core running")

            # Wait for tasks
            await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def _main_loop(self) -> None:
        """Main control loop."""
        while self._running:
            try:
                # Check safety
                violations = self.safety_monitor.check_all_safety()

                # Handle primary target tracking
                if (
                    self.tracker
                    and self.tracker.primary_target
                    and self.state_machine.state == FlightState.TRACKING
                ):
                    target = self.tracker.primary_target
                    # Here you would calculate and send tracking commands
                    pass

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Main loop error", error=str(e))
                await asyncio.sleep(1.0)

    async def shutdown(self) -> None:
        """Shutdown all systems."""
        logger.info("Shutting down...")
        self._running = False

        # Stop vision
        if self.camera_manager:
            self.camera_manager.shutdown()

        # Stop safety monitoring
        await self.safety_monitor.stop_monitoring()

        # Stop ground station
        await self.ground_station.stop()

        # Disconnect MAVLink
        if self.mavlink:
            self.mavlink.disconnect()

        # Close loggers
        self.event_logger.close()
        self.telemetry_logger.close()

        # Stop flight recording
        if self.flight_recorder:
            self.flight_recorder.stop()

        logger.info("Shutdown complete")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Drone Flight Core")
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "-s", "--simulation",
        action="store_true",
        help="Run in simulation mode",
    )
    parser.add_argument(
        "--ground-station-only",
        action="store_true",
        help="Run only ground station API",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.simulation:
        config.simulation_mode = True

    if args.ground_station_only:
        # Run only ground station
        from drone_flight_core.ground_station.server import main as gs_main
        gs_main()
    else:
        # Run full application
        app = DroneFlightCore(config)
        asyncio.run(app.run())


if __name__ == "__main__":
    main()

"""
Drone Flight Core - Comprehensive drone flight software with autonomous tracking.

This package provides:
- Flight state machine with safety systems
- MAVLink communication with PX4/ArduPilot
- Dual-camera object detection (thermal + RGB)
- Autonomous tracking with energy-efficient patterns
- Ground station API
- Comprehensive logging and data persistence
"""

__version__ = "0.1.0"

from drone_flight_core.core.state_machine import FlightState, FlightStateMachine
from drone_flight_core.core.config import DroneConfig, load_config

__all__ = [
    "FlightState",
    "FlightStateMachine",
    "DroneConfig",
    "load_config",
    "__version__",
]

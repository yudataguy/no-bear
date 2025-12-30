"""
Ground station server with REST and WebSocket APIs.

Provides external API access to drone control, telemetry,
and mission management.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

from drone_flight_core.core.config import DroneConfig, GroundStationConfig
from drone_flight_core.core.state_machine import FlightStateMachine, FlightState
from drone_flight_core.core.safety import SafetyMonitor

logger = structlog.get_logger(__name__)


# Request/Response Models
class StatusResponse(BaseModel):
    """System status response."""

    drone_id: str
    state: str
    is_flying: bool
    is_emergency: bool
    battery_percent: float
    gps_fix: bool
    timestamp: str


class TelemetryResponse(BaseModel):
    """Telemetry data response."""

    gps: dict
    attitude: dict
    battery: dict
    velocity: dict
    system: dict
    timestamp: str


class CommandRequest(BaseModel):
    """Generic command request."""

    command: str
    params: dict = Field(default_factory=dict)


class CommandResponse(BaseModel):
    """Command execution response."""

    command: str
    status: str
    message: str
    timestamp: str


class GotoRequest(BaseModel):
    """Go to coordinate request."""

    latitude: float
    longitude: float
    altitude: float = 30.0


class MissionRequest(BaseModel):
    """Mission definition request."""

    waypoints: list[dict]
    name: str = "Unnamed Mission"


class TrackingRequest(BaseModel):
    """Tracking control request."""

    action: str  # start, stop, set_target
    target_id: int | None = None
    pattern: str | None = None  # hover, orbit


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected", total=len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected", total=len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        """Send message to specific client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)


class GroundStation:
    """
    Ground station controller.

    Manages drone connection, commands, and provides API endpoints.
    """

    def __init__(self, config: DroneConfig) -> None:
        self._config = config
        self._state_machine = FlightStateMachine()
        self._safety_monitor = SafetyMonitor(config.safety)
        self._connection_manager = ConnectionManager()

        # Telemetry cache
        self._telemetry: dict = {}
        self._telemetry_lock = asyncio.Lock()

        # Background tasks
        self._broadcast_task: asyncio.Task | None = None

        logger.info("Ground station initialized", drone_id=config.drone_id)

    @property
    def state_machine(self) -> FlightStateMachine:
        """Get state machine."""
        return self._state_machine

    @property
    def safety_monitor(self) -> SafetyMonitor:
        """Get safety monitor."""
        return self._safety_monitor

    @property
    def connection_manager(self) -> ConnectionManager:
        """Get WebSocket connection manager."""
        return self._connection_manager

    async def start(self) -> None:
        """Start ground station services."""
        self._broadcast_task = asyncio.create_task(self._telemetry_broadcast_loop())
        logger.info("Ground station started")

    async def stop(self) -> None:
        """Stop ground station services."""
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("Ground station stopped")

    async def update_telemetry(self, telemetry: dict) -> None:
        """Update cached telemetry data."""
        async with self._telemetry_lock:
            self._telemetry = telemetry

    async def get_telemetry(self) -> dict:
        """Get cached telemetry data."""
        async with self._telemetry_lock:
            return self._telemetry.copy()

    async def _telemetry_broadcast_loop(self) -> None:
        """Periodically broadcast telemetry to WebSocket clients."""
        while True:
            try:
                telemetry = await self.get_telemetry()
                if telemetry:
                    await self._connection_manager.broadcast({
                        "type": "telemetry",
                        "data": telemetry,
                    })
                await asyncio.sleep(0.1)  # 10 Hz broadcast rate

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Broadcast error", error=str(e))
                await asyncio.sleep(1.0)

    def get_status(self) -> dict:
        """Get current system status."""
        return {
            "drone_id": self._config.drone_id,
            "state": self._state_machine.state.name,
            "is_flying": self._state_machine.is_flying,
            "is_emergency": self._state_machine.is_emergency,
            "battery_percent": self._telemetry.get("battery", {}).get("remaining", 0),
            "gps_fix": self._telemetry.get("system", {}).get("gps_fix", False),
            "timestamp": datetime.now().isoformat(),
        }


def create_app(
    config: DroneConfig | None = None,
    ground_station: GroundStation | None = None,
) -> FastAPI:
    """
    Create FastAPI application for ground station.

    Args:
        config: Drone configuration.
        ground_station: Optional pre-created ground station.

    Returns:
        Configured FastAPI application.
    """
    if config is None:
        config = DroneConfig()

    if ground_station is None:
        ground_station = GroundStation(config)

    gs_config = config.ground_station

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        await ground_station.start()
        yield
        await ground_station.stop()

    app = FastAPI(
        title="Drone Flight Core API",
        description="REST and WebSocket API for drone control and monitoring",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=gs_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store ground station in app state
    app.state.ground_station = ground_station
    app.state.config = config

    # Dependency to get ground station
    def get_gs() -> GroundStation:
        return app.state.ground_station

    # ============== REST API Endpoints ==============

    @app.get(f"{gs_config.api_prefix}/status", response_model=StatusResponse)
    async def get_status(gs: GroundStation = Depends(get_gs)) -> StatusResponse:
        """Get current system status."""
        status = gs.get_status()
        return StatusResponse(**status)

    @app.get(f"{gs_config.api_prefix}/telemetry", response_model=TelemetryResponse)
    async def get_telemetry(gs: GroundStation = Depends(get_gs)) -> TelemetryResponse:
        """Get current telemetry data."""
        telemetry = await gs.get_telemetry()
        if not telemetry:
            raise HTTPException(status_code=503, detail="No telemetry available")
        return TelemetryResponse(**telemetry, timestamp=datetime.now().isoformat())

    @app.get(f"{gs_config.api_prefix}/state")
    async def get_state(gs: GroundStation = Depends(get_gs)) -> dict:
        """Get detailed state machine information."""
        return gs.state_machine.to_dict()

    @app.get(f"{gs_config.api_prefix}/safety")
    async def get_safety(gs: GroundStation = Depends(get_gs)) -> dict:
        """Get safety monitor status."""
        return gs.safety_monitor.to_dict()

    @app.post(f"{gs_config.api_prefix}/command", response_model=CommandResponse)
    async def execute_command(
        request: CommandRequest,
        gs: GroundStation = Depends(get_gs),
    ) -> CommandResponse:
        """Execute a drone command."""
        command = request.command.upper()
        params = request.params

        try:
            if command == "ARM":
                await gs.state_machine.arm()
            elif command == "DISARM":
                await gs.state_machine.disarm()
            elif command == "TAKEOFF":
                await gs.state_machine.takeoff()
            elif command == "LAND":
                await gs.state_machine.land()
            elif command == "RTH":
                await gs.state_machine.return_to_home()
            elif command == "EMERGENCY_LAND":
                await gs.state_machine.emergency_land()
            elif command == "EMERGENCY_STOP":
                await gs.state_machine.emergency_stop()
            else:
                raise HTTPException(status_code=400, detail=f"Unknown command: {command}")

            return CommandResponse(
                command=command,
                status="success",
                message=f"Command {command} executed",
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error("Command failed", command=command, error=str(e))
            return CommandResponse(
                command=command,
                status="error",
                message=str(e),
                timestamp=datetime.now().isoformat(),
            )

    @app.post(f"{gs_config.api_prefix}/goto")
    async def goto_position(
        request: GotoRequest,
        gs: GroundStation = Depends(get_gs),
    ) -> CommandResponse:
        """Navigate to GPS coordinates."""
        try:
            # Validate state allows navigation
            if not gs.state_machine.is_flying:
                raise HTTPException(
                    status_code=400,
                    detail="Drone must be flying to navigate",
                )

            # Here you would send the actual navigation command
            # For now, just acknowledge
            logger.info(
                "Goto command received",
                lat=request.latitude,
                lon=request.longitude,
                alt=request.altitude,
            )

            return CommandResponse(
                command="GOTO",
                status="success",
                message=f"Navigating to ({request.latitude}, {request.longitude}) at {request.altitude}m",
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post(f"{gs_config.api_prefix}/tracking")
    async def control_tracking(
        request: TrackingRequest,
        gs: GroundStation = Depends(get_gs),
    ) -> CommandResponse:
        """Control object tracking."""
        action = request.action.lower()

        if action == "start":
            message = "Tracking started"
        elif action == "stop":
            message = "Tracking stopped"
        elif action == "set_target":
            if request.target_id is None:
                raise HTTPException(status_code=400, detail="target_id required")
            message = f"Target set to track ID {request.target_id}"
        elif action == "set_pattern":
            if request.pattern not in ["hover", "orbit"]:
                raise HTTPException(status_code=400, detail="Invalid pattern")
            message = f"Tracking pattern set to {request.pattern}"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

        return CommandResponse(
            command=f"TRACKING_{action.upper()}",
            status="success",
            message=message,
            timestamp=datetime.now().isoformat(),
        )

    @app.get(f"{gs_config.api_prefix}/config")
    async def get_config(gs: GroundStation = Depends(get_gs)) -> dict:
        """Get current configuration."""
        return app.state.config.model_dump()

    @app.get(f"{gs_config.api_prefix}/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }

    # ============== WebSocket Endpoint ==============

    @app.websocket(gs_config.websocket_path)
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time updates.

        Sends:
        - Telemetry updates at 10 Hz
        - State change notifications
        - Detection events
        """
        gs = app.state.ground_station
        await gs.connection_manager.connect(websocket)

        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_text()

                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "")

                    if msg_type == "ping":
                        await gs.connection_manager.send_personal(
                            websocket,
                            {"type": "pong", "timestamp": datetime.now().isoformat()},
                        )

                    elif msg_type == "subscribe":
                        # Handle subscription requests
                        topics = message.get("topics", [])
                        await gs.connection_manager.send_personal(
                            websocket,
                            {"type": "subscribed", "topics": topics},
                        )

                    elif msg_type == "command":
                        # Handle commands via WebSocket
                        command = message.get("command", "")
                        params = message.get("params", {})

                        # Process command (simplified)
                        await gs.connection_manager.send_personal(
                            websocket,
                            {
                                "type": "command_ack",
                                "command": command,
                                "status": "received",
                            },
                        )

                except json.JSONDecodeError:
                    await gs.connection_manager.send_personal(
                        websocket,
                        {"type": "error", "message": "Invalid JSON"},
                    )

        except WebSocketDisconnect:
            gs.connection_manager.disconnect(websocket)

    return app


def main() -> None:
    """Run ground station server."""
    import uvicorn

    config = DroneConfig()
    app = create_app(config)

    uvicorn.run(
        app,
        host=config.ground_station.host,
        port=config.ground_station.port,
    )


if __name__ == "__main__":
    main()

"""
Mission planning and management API.

Provides endpoints for creating, managing, and executing
flight missions with waypoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/missions", tags=["missions"])


class Waypoint(BaseModel):
    """Mission waypoint definition."""

    latitude: float
    longitude: float
    altitude: float
    speed: float | None = None
    hold_time_sec: float = 0.0
    action: str | None = None  # Optional action at waypoint


class Mission(BaseModel):
    """Mission definition."""

    id: str | None = None
    name: str
    description: str = ""
    waypoints: list[Waypoint]
    created_at: str | None = None
    status: str = "draft"  # draft, ready, running, completed, failed


class MissionCreate(BaseModel):
    """Request to create a new mission."""

    name: str
    description: str = ""
    waypoints: list[Waypoint] = Field(min_length=1)


class MissionUpdate(BaseModel):
    """Request to update a mission."""

    name: str | None = None
    description: str | None = None
    waypoints: list[Waypoint] | None = None


# In-memory mission storage (would be database in production)
_missions: dict[str, Mission] = {}
_mission_counter = 0


def _generate_mission_id() -> str:
    """Generate unique mission ID."""
    global _mission_counter
    _mission_counter += 1
    return f"mission_{_mission_counter:04d}"


@router.get("", response_model=list[Mission])
async def list_missions() -> list[Mission]:
    """List all missions."""
    return list(_missions.values())


@router.post("", response_model=Mission)
async def create_mission(request: MissionCreate) -> Mission:
    """Create a new mission."""
    mission_id = _generate_mission_id()

    mission = Mission(
        id=mission_id,
        name=request.name,
        description=request.description,
        waypoints=request.waypoints,
        created_at=datetime.now().isoformat(),
        status="draft",
    )

    _missions[mission_id] = mission
    logger.info("Mission created", mission_id=mission_id, waypoints=len(request.waypoints))

    return mission


@router.get("/{mission_id}", response_model=Mission)
async def get_mission(mission_id: str) -> Mission:
    """Get a specific mission."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")
    return _missions[mission_id]


@router.put("/{mission_id}", response_model=Mission)
async def update_mission(mission_id: str, request: MissionUpdate) -> Mission:
    """Update a mission."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")

    mission = _missions[mission_id]

    if mission.status == "running":
        raise HTTPException(status_code=400, detail="Cannot update running mission")

    if request.name is not None:
        mission.name = request.name
    if request.description is not None:
        mission.description = request.description
    if request.waypoints is not None:
        mission.waypoints = request.waypoints

    return mission


@router.delete("/{mission_id}")
async def delete_mission(mission_id: str) -> dict:
    """Delete a mission."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")

    if _missions[mission_id].status == "running":
        raise HTTPException(status_code=400, detail="Cannot delete running mission")

    del _missions[mission_id]
    logger.info("Mission deleted", mission_id=mission_id)

    return {"message": "Mission deleted", "mission_id": mission_id}


@router.post("/{mission_id}/start")
async def start_mission(mission_id: str) -> dict:
    """Start executing a mission."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")

    mission = _missions[mission_id]

    if mission.status == "running":
        raise HTTPException(status_code=400, detail="Mission already running")

    if not mission.waypoints:
        raise HTTPException(status_code=400, detail="Mission has no waypoints")

    mission.status = "running"
    logger.info("Mission started", mission_id=mission_id)

    return {
        "message": "Mission started",
        "mission_id": mission_id,
        "waypoint_count": len(mission.waypoints),
    }


@router.post("/{mission_id}/pause")
async def pause_mission(mission_id: str) -> dict:
    """Pause a running mission."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")

    mission = _missions[mission_id]

    if mission.status != "running":
        raise HTTPException(status_code=400, detail="Mission not running")

    mission.status = "paused"
    logger.info("Mission paused", mission_id=mission_id)

    return {"message": "Mission paused", "mission_id": mission_id}


@router.post("/{mission_id}/resume")
async def resume_mission(mission_id: str) -> dict:
    """Resume a paused mission."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")

    mission = _missions[mission_id]

    if mission.status != "paused":
        raise HTTPException(status_code=400, detail="Mission not paused")

    mission.status = "running"
    logger.info("Mission resumed", mission_id=mission_id)

    return {"message": "Mission resumed", "mission_id": mission_id}


@router.post("/{mission_id}/abort")
async def abort_mission(mission_id: str) -> dict:
    """Abort a running mission."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")

    mission = _missions[mission_id]

    if mission.status not in ["running", "paused"]:
        raise HTTPException(status_code=400, detail="Mission not active")

    mission.status = "aborted"
    logger.info("Mission aborted", mission_id=mission_id)

    return {"message": "Mission aborted", "mission_id": mission_id}


@router.get("/{mission_id}/progress")
async def get_mission_progress(mission_id: str) -> dict:
    """Get mission execution progress."""
    if mission_id not in _missions:
        raise HTTPException(status_code=404, detail="Mission not found")

    mission = _missions[mission_id]

    # In real implementation, this would track actual progress
    return {
        "mission_id": mission_id,
        "status": mission.status,
        "total_waypoints": len(mission.waypoints),
        "current_waypoint": 0,
        "progress_percent": 0.0,
        "estimated_time_remaining_sec": 0,
    }

"""
Flight state machine for drone control.

Manages transitions between flight states with safety checks
and event-driven state changes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Any
import structlog

logger = structlog.get_logger(__name__)


class FlightState(Enum):
    """Enumeration of all possible flight states."""

    # Ground states
    DISARMED = auto()
    ARMED = auto()
    PREFLIGHT_CHECK = auto()

    # Takeoff/Landing
    TAKING_OFF = auto()
    LANDING = auto()
    LANDED = auto()

    # Flight modes
    MANUAL = auto()
    HOVER = auto()
    NAVIGATING = auto()
    WAYPOINT = auto()

    # Mission states
    PATROL = auto()  # Autonomous patrol flight
    SEARCHING = auto()  # Searching for targets
    TRACKING = auto()  # Actively tracking target
    ORBITING = auto()  # Orbiting around target
    MONITORING = auto()  # Monitoring detected target for confirmation

    # Safety states
    RETURN_TO_HOME = auto()
    EMERGENCY_LAND = auto()
    EMERGENCY_STOP = auto()

    # Error state
    FAULT = auto()


@dataclass
class StateTransition:
    """Represents a state transition event."""

    from_state: FlightState
    to_state: FlightState
    timestamp: datetime
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Valid state transitions map
VALID_TRANSITIONS: dict[FlightState, set[FlightState]] = {
    FlightState.DISARMED: {
        FlightState.ARMED,
        FlightState.PREFLIGHT_CHECK,
        FlightState.FAULT,
    },
    FlightState.PREFLIGHT_CHECK: {
        FlightState.ARMED,
        FlightState.DISARMED,
        FlightState.FAULT,
    },
    FlightState.ARMED: {
        FlightState.TAKING_OFF,
        FlightState.DISARMED,
        FlightState.MANUAL,
        FlightState.FAULT,
        FlightState.EMERGENCY_STOP,
    },
    FlightState.TAKING_OFF: {
        FlightState.HOVER,
        FlightState.MANUAL,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.HOVER: {
        FlightState.MANUAL,
        FlightState.NAVIGATING,
        FlightState.WAYPOINT,
        FlightState.PATROL,
        FlightState.SEARCHING,
        FlightState.TRACKING,
        FlightState.ORBITING,
        FlightState.LANDING,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.MANUAL: {
        FlightState.HOVER,
        FlightState.NAVIGATING,
        FlightState.PATROL,
        FlightState.LANDING,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.EMERGENCY_STOP,
        FlightState.FAULT,
    },
    FlightState.NAVIGATING: {
        FlightState.HOVER,
        FlightState.WAYPOINT,
        FlightState.PATROL,
        FlightState.SEARCHING,
        FlightState.MONITORING,
        FlightState.MANUAL,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.WAYPOINT: {
        FlightState.HOVER,
        FlightState.NAVIGATING,
        FlightState.PATROL,
        FlightState.SEARCHING,
        FlightState.MONITORING,
        FlightState.MANUAL,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.PATROL: {
        FlightState.HOVER,
        FlightState.NAVIGATING,
        FlightState.SEARCHING,
        FlightState.MONITORING,  # Detection triggered
        FlightState.MANUAL,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.SEARCHING: {
        FlightState.TRACKING,
        FlightState.MONITORING,
        FlightState.HOVER,
        FlightState.NAVIGATING,
        FlightState.PATROL,  # Resume patrol
        FlightState.MANUAL,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.MONITORING: {
        FlightState.TRACKING,  # Confirmed, start tracking
        FlightState.PATROL,  # Not confirmed, resume patrol
        FlightState.NAVIGATING,  # Resume navigation
        FlightState.HOVER,
        FlightState.MANUAL,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.TRACKING: {
        FlightState.ORBITING,
        FlightState.HOVER,
        FlightState.SEARCHING,
        FlightState.MONITORING,
        FlightState.PATROL,  # Resume patrol after tracking
        FlightState.MANUAL,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.ORBITING: {
        FlightState.TRACKING,
        FlightState.HOVER,
        FlightState.SEARCHING,
        FlightState.PATROL,  # Resume patrol
        FlightState.MANUAL,
        FlightState.RETURN_TO_HOME,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.RETURN_TO_HOME: {
        FlightState.HOVER,
        FlightState.LANDING,
        FlightState.MANUAL,
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.LANDING: {
        FlightState.LANDED,
        FlightState.HOVER,  # Abort landing
        FlightState.EMERGENCY_LAND,
        FlightState.FAULT,
    },
    FlightState.LANDED: {
        FlightState.DISARMED,
        FlightState.ARMED,
        FlightState.FAULT,
    },
    FlightState.EMERGENCY_LAND: {
        FlightState.LANDED,
        FlightState.FAULT,
    },
    FlightState.EMERGENCY_STOP: {
        FlightState.DISARMED,
        FlightState.FAULT,
    },
    FlightState.FAULT: {
        FlightState.DISARMED,  # Only way out of fault is full reset
    },
}


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self, from_state: FlightState, to_state: FlightState, reason: str = ""
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            f"Invalid transition from {from_state.name} to {to_state.name}: {reason}"
        )


class FlightStateMachine:
    """
    Manages flight state transitions with validation and callbacks.

    The state machine ensures that only valid transitions occur and
    provides hooks for state entry/exit actions and transition callbacks.
    """

    def __init__(self, initial_state: FlightState = FlightState.DISARMED) -> None:
        self._state = initial_state
        self._previous_state: FlightState | None = None
        self._state_entry_time = datetime.now()
        self._transition_history: list[StateTransition] = []
        self._max_history_size = 1000

        # Callbacks
        self._on_enter_callbacks: dict[FlightState, list[Callable]] = {}
        self._on_exit_callbacks: dict[FlightState, list[Callable]] = {}
        self._transition_callbacks: list[Callable[[StateTransition], None]] = []

        # Lock for thread-safe transitions
        self._lock = asyncio.Lock()

        logger.info("State machine initialized", state=initial_state.name)

    @property
    def state(self) -> FlightState:
        """Current flight state."""
        return self._state

    @property
    def previous_state(self) -> FlightState | None:
        """Previous flight state."""
        return self._previous_state

    @property
    def state_duration_seconds(self) -> float:
        """Time spent in current state in seconds."""
        return (datetime.now() - self._state_entry_time).total_seconds()

    @property
    def transition_history(self) -> list[StateTransition]:
        """History of state transitions."""
        return self._transition_history.copy()

    @property
    def is_flying(self) -> bool:
        """Check if drone is currently in a flying state."""
        ground_states = {
            FlightState.DISARMED,
            FlightState.ARMED,
            FlightState.PREFLIGHT_CHECK,
            FlightState.LANDED,
        }
        return self._state not in ground_states

    @property
    def is_emergency(self) -> bool:
        """Check if drone is in an emergency state."""
        emergency_states = {
            FlightState.EMERGENCY_LAND,
            FlightState.EMERGENCY_STOP,
            FlightState.FAULT,
        }
        return self._state in emergency_states

    def can_transition_to(self, target_state: FlightState) -> bool:
        """Check if transition to target state is valid."""
        valid_targets = VALID_TRANSITIONS.get(self._state, set())
        return target_state in valid_targets

    def get_valid_transitions(self) -> set[FlightState]:
        """Get all valid transitions from current state."""
        return VALID_TRANSITIONS.get(self._state, set()).copy()

    async def transition_to(
        self,
        target_state: FlightState,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> StateTransition:
        """
        Transition to a new state.

        Args:
            target_state: The state to transition to.
            reason: Human-readable reason for the transition.
            metadata: Optional metadata to attach to the transition.
            force: If True, skip validation (use with caution).

        Returns:
            The StateTransition record.

        Raises:
            StateTransitionError: If the transition is not valid.
        """
        async with self._lock:
            # Validate transition
            if not force and not self.can_transition_to(target_state):
                raise StateTransitionError(
                    self._state,
                    target_state,
                    f"Not in valid transitions: {self.get_valid_transitions()}",
                )

            # Create transition record
            transition = StateTransition(
                from_state=self._state,
                to_state=target_state,
                timestamp=datetime.now(),
                reason=reason,
                metadata=metadata or {},
            )

            # Execute exit callbacks for current state
            await self._execute_exit_callbacks(self._state)

            # Update state
            self._previous_state = self._state
            self._state = target_state
            self._state_entry_time = datetime.now()

            # Record transition
            self._transition_history.append(transition)
            if len(self._transition_history) > self._max_history_size:
                self._transition_history = self._transition_history[-self._max_history_size :]

            logger.info(
                "State transition",
                from_state=transition.from_state.name,
                to_state=transition.to_state.name,
                reason=reason,
            )

            # Execute enter callbacks for new state
            await self._execute_enter_callbacks(target_state)

            # Execute general transition callbacks
            await self._execute_transition_callbacks(transition)

            return transition

    def on_enter(self, state: FlightState, callback: Callable) -> None:
        """Register a callback to be called when entering a state."""
        if state not in self._on_enter_callbacks:
            self._on_enter_callbacks[state] = []
        self._on_enter_callbacks[state].append(callback)

    def on_exit(self, state: FlightState, callback: Callable) -> None:
        """Register a callback to be called when exiting a state."""
        if state not in self._on_exit_callbacks:
            self._on_exit_callbacks[state] = []
        self._on_exit_callbacks[state].append(callback)

    def on_transition(self, callback: Callable[[StateTransition], None]) -> None:
        """Register a callback to be called on any state transition."""
        self._transition_callbacks.append(callback)

    async def _execute_enter_callbacks(self, state: FlightState) -> None:
        """Execute all enter callbacks for a state."""
        callbacks = self._on_enter_callbacks.get(state, [])
        for callback in callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Enter callback failed", state=state.name, error=str(e))

    async def _execute_exit_callbacks(self, state: FlightState) -> None:
        """Execute all exit callbacks for a state."""
        callbacks = self._on_exit_callbacks.get(state, [])
        for callback in callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Exit callback failed", state=state.name, error=str(e))

    async def _execute_transition_callbacks(self, transition: StateTransition) -> None:
        """Execute all transition callbacks."""
        for callback in self._transition_callbacks:
            try:
                result = callback(transition)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Transition callback failed", error=str(e))

    # Convenience methods for common transitions
    async def arm(self, reason: str = "Arm command received") -> StateTransition:
        """Arm the drone."""
        return await self.transition_to(FlightState.ARMED, reason)

    async def disarm(self, reason: str = "Disarm command received") -> StateTransition:
        """Disarm the drone."""
        return await self.transition_to(FlightState.DISARMED, reason)

    async def takeoff(self, reason: str = "Takeoff command received") -> StateTransition:
        """Initiate takeoff."""
        return await self.transition_to(FlightState.TAKING_OFF, reason)

    async def land(self, reason: str = "Land command received") -> StateTransition:
        """Initiate landing."""
        return await self.transition_to(FlightState.LANDING, reason)

    async def return_to_home(
        self, reason: str = "Return to home initiated"
    ) -> StateTransition:
        """Initiate return to home."""
        return await self.transition_to(FlightState.RETURN_TO_HOME, reason)

    async def emergency_land(
        self, reason: str = "Emergency landing initiated"
    ) -> StateTransition:
        """Initiate emergency landing."""
        return await self.transition_to(FlightState.EMERGENCY_LAND, reason, force=True)

    async def emergency_stop(
        self, reason: str = "Emergency stop activated"
    ) -> StateTransition:
        """Activate emergency stop (kill switch)."""
        return await self.transition_to(FlightState.EMERGENCY_STOP, reason, force=True)

    async def enter_fault(self, reason: str = "Fault detected") -> StateTransition:
        """Enter fault state."""
        return await self.transition_to(FlightState.FAULT, reason, force=True)

    async def start_patrol(
        self, reason: str = "Patrol mission started"
    ) -> StateTransition:
        """Start patrol flight mode."""
        return await self.transition_to(FlightState.PATROL, reason)

    async def enter_monitoring(
        self, reason: str = "Detection triggered monitoring"
    ) -> StateTransition:
        """Enter monitoring mode for detection confirmation."""
        return await self.transition_to(FlightState.MONITORING, reason)

    async def start_tracking(
        self, reason: str = "Target confirmed, tracking started"
    ) -> StateTransition:
        """Start tracking a confirmed target."""
        return await self.transition_to(FlightState.TRACKING, reason)

    async def resume_patrol(
        self, reason: str = "Resuming patrol"
    ) -> StateTransition:
        """Resume patrol after monitoring/tracking."""
        return await self.transition_to(FlightState.PATROL, reason)

    @property
    def is_on_mission(self) -> bool:
        """Check if drone is executing an autonomous mission."""
        mission_states = {
            FlightState.PATROL,
            FlightState.WAYPOINT,
            FlightState.NAVIGATING,
            FlightState.SEARCHING,
            FlightState.MONITORING,
            FlightState.TRACKING,
            FlightState.ORBITING,
        }
        return self._state in mission_states

    @property
    def is_monitoring_or_tracking(self) -> bool:
        """Check if drone is in detection response mode."""
        return self._state in {
            FlightState.MONITORING,
            FlightState.TRACKING,
            FlightState.ORBITING,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export state machine status as dictionary."""
        return {
            "current_state": self._state.name,
            "previous_state": self._previous_state.name if self._previous_state else None,
            "state_duration_seconds": self.state_duration_seconds,
            "is_flying": self.is_flying,
            "is_emergency": self.is_emergency,
            "is_on_mission": self.is_on_mission,
            "is_monitoring_or_tracking": self.is_monitoring_or_tracking,
            "valid_transitions": [s.name for s in self.get_valid_transitions()],
            "transition_count": len(self._transition_history),
        }

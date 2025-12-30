"""Tests for flight state machine."""

import pytest

from drone_flight_core.core.state_machine import (
    FlightState,
    FlightStateMachine,
    StateTransitionError,
    VALID_TRANSITIONS,
)


class TestFlightState:
    """Tests for FlightState enum."""

    def test_all_states_have_transitions(self):
        """Every state should have defined transitions."""
        for state in FlightState:
            assert state in VALID_TRANSITIONS, f"{state} has no transitions defined"

    def test_emergency_states_exist(self):
        """Emergency states should be defined."""
        assert FlightState.EMERGENCY_LAND in FlightState
        assert FlightState.EMERGENCY_STOP in FlightState
        assert FlightState.FAULT in FlightState


class TestFlightStateMachine:
    """Tests for FlightStateMachine."""

    @pytest.fixture
    def state_machine(self):
        """Create a fresh state machine for each test."""
        return FlightStateMachine()

    def test_initial_state(self, state_machine):
        """State machine should start in DISARMED state."""
        assert state_machine.state == FlightState.DISARMED

    def test_custom_initial_state(self):
        """State machine should accept custom initial state."""
        sm = FlightStateMachine(initial_state=FlightState.ARMED)
        assert sm.state == FlightState.ARMED

    @pytest.mark.asyncio
    async def test_valid_transition(self, state_machine):
        """Valid transitions should succeed."""
        await state_machine.arm()
        assert state_machine.state == FlightState.ARMED

    @pytest.mark.asyncio
    async def test_invalid_transition(self, state_machine):
        """Invalid transitions should raise error."""
        with pytest.raises(StateTransitionError):
            await state_machine.transition_to(FlightState.HOVER)

    @pytest.mark.asyncio
    async def test_forced_transition(self, state_machine):
        """Forced transitions should bypass validation."""
        await state_machine.transition_to(FlightState.HOVER, force=True)
        assert state_machine.state == FlightState.HOVER

    @pytest.mark.asyncio
    async def test_emergency_transitions(self, state_machine):
        """Emergency transitions should always succeed."""
        await state_machine.arm()
        await state_machine.takeoff()

        # Emergency land should work from any state
        await state_machine.emergency_land()
        assert state_machine.state == FlightState.EMERGENCY_LAND

    @pytest.mark.asyncio
    async def test_transition_history(self, state_machine):
        """Transitions should be recorded in history."""
        await state_machine.arm()
        await state_machine.takeoff()

        history = state_machine.transition_history
        assert len(history) == 2
        assert history[0].to_state == FlightState.ARMED
        assert history[1].to_state == FlightState.TAKING_OFF

    def test_is_flying_property(self, state_machine):
        """is_flying should correctly identify ground vs flying states."""
        assert not state_machine.is_flying

    def test_is_emergency_property(self, state_machine):
        """is_emergency should identify emergency states."""
        assert not state_machine.is_emergency

    @pytest.mark.asyncio
    async def test_is_emergency_in_emergency_state(self, state_machine):
        """is_emergency should be True in emergency states."""
        await state_machine.enter_fault()
        assert state_machine.is_emergency

    def test_can_transition_to(self, state_machine):
        """can_transition_to should correctly check valid transitions."""
        assert state_machine.can_transition_to(FlightState.ARMED)
        assert not state_machine.can_transition_to(FlightState.HOVER)

    def test_get_valid_transitions(self, state_machine):
        """get_valid_transitions should return correct set."""
        valid = state_machine.get_valid_transitions()
        assert FlightState.ARMED in valid
        assert FlightState.HOVER not in valid

    @pytest.mark.asyncio
    async def test_previous_state(self, state_machine):
        """previous_state should track correctly."""
        assert state_machine.previous_state is None

        await state_machine.arm()
        assert state_machine.previous_state == FlightState.DISARMED

    @pytest.mark.asyncio
    async def test_state_duration(self, state_machine):
        """state_duration_seconds should track time."""
        import asyncio

        await asyncio.sleep(0.1)
        duration = state_machine.state_duration_seconds
        assert duration >= 0.1

    @pytest.mark.asyncio
    async def test_transition_callbacks(self, state_machine):
        """Callbacks should be called on transitions."""
        callback_called = []

        def on_enter():
            callback_called.append("enter")

        def on_exit():
            callback_called.append("exit")

        def on_transition(transition):
            callback_called.append(f"transition:{transition.to_state.name}")

        state_machine.on_enter(FlightState.ARMED, on_enter)
        state_machine.on_exit(FlightState.DISARMED, on_exit)
        state_machine.on_transition(on_transition)

        await state_machine.arm()

        assert "enter" in callback_called
        assert "exit" in callback_called
        assert "transition:ARMED" in callback_called

    def test_to_dict(self, state_machine):
        """to_dict should return complete state."""
        data = state_machine.to_dict()

        assert "current_state" in data
        assert "is_flying" in data
        assert "is_emergency" in data
        assert "valid_transitions" in data

    @pytest.mark.asyncio
    async def test_full_flight_sequence(self, state_machine):
        """Test a complete flight sequence."""
        # Pre-flight
        await state_machine.transition_to(FlightState.PREFLIGHT_CHECK)
        assert state_machine.state == FlightState.PREFLIGHT_CHECK

        # Arm and takeoff
        await state_machine.arm()
        await state_machine.takeoff()
        assert state_machine.is_flying

        # Hover
        await state_machine.transition_to(FlightState.HOVER)

        # Navigate
        await state_machine.transition_to(FlightState.NAVIGATING)

        # Search
        await state_machine.transition_to(FlightState.SEARCHING)

        # Track
        await state_machine.transition_to(FlightState.TRACKING)

        # Return home
        await state_machine.return_to_home()
        assert state_machine.state == FlightState.RETURN_TO_HOME

        # Land
        await state_machine.land()
        assert state_machine.state == FlightState.LANDING

        # Complete
        await state_machine.transition_to(FlightState.LANDED)
        await state_machine.disarm()
        assert state_machine.state == FlightState.DISARMED

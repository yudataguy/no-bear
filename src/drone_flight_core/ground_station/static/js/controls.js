/**
 * Control Panel Functions
 *
 * Handles sending commands to drone and
 * managing control UI interactions.
 */

// Command history for debugging
const commandHistory = [];

/**
 * Send a command to the drone
 */
function sendCommand(command, params = {}) {
    console.log(`Sending command: ${command}`, params);

    // Record in history
    commandHistory.push({
        command: command,
        params: params,
        timestamp: new Date().toISOString()
    });

    // Send via WebSocket
    if (wsClient && wsClient.isConnected) {
        wsClient.sendCommand(command, params);
        showToast(`Command sent: ${command}`, 'info');
    } else {
        showToast('Not connected to drone', 'error');
    }
}

/**
 * Show takeoff altitude dialog
 */
function showTakeoffDialog() {
    const content = `
        <div class="form-row">
            <label>Takeoff Altitude:</label>
            <input type="number" id="takeoff-alt" value="10" min="2" max="100" step="1">
            <span class="unit">m</span>
        </div>
    `;

    showDialog('Takeoff', content, [
        { text: 'Cancel', action: closeDialog },
        {
            text: 'Takeoff',
            action: () => {
                const alt = parseFloat(document.getElementById('takeoff-alt').value) || 10;
                sendCommand('TAKEOFF', { altitude: alt });
                closeDialog();
            },
            primary: true
        }
    ]);
}

/**
 * Confirm emergency stop
 */
function confirmEmergency() {
    const content = `
        <p class="warning-text">
            Are you sure you want to activate EMERGENCY STOP?
        </p>
        <p>
            This will immediately stop all motors and the drone will fall.
            Only use this if there is an immediate danger.
        </p>
    `;

    showDialog('EMERGENCY STOP', content, [
        { text: 'Cancel', action: closeDialog },
        {
            text: 'CONFIRM EMERGENCY STOP',
            action: () => {
                sendCommand('EMERGENCY_STOP');
                closeDialog();
            },
            danger: true
        }
    ]);
}

/**
 * Go to specified position
 */
function goToPosition() {
    const lat = parseFloat(document.getElementById('goto-lat').value);
    const lon = parseFloat(document.getElementById('goto-lon').value);
    const alt = parseFloat(document.getElementById('goto-alt').value) || 50;

    if (isNaN(lat) || isNaN(lon)) {
        showToast('Please enter valid coordinates', 'error');
        return;
    }

    // Validate coordinate ranges
    if (lat < -90 || lat > 90) {
        showToast('Latitude must be between -90 and 90', 'error');
        return;
    }
    if (lon < -180 || lon > 180) {
        showToast('Longitude must be between -180 and 180', 'error');
        return;
    }

    sendCommand('GOTO', {
        latitude: lat,
        longitude: lon,
        altitude: alt
    });
}

/**
 * Load selected mission
 */
function loadMission() {
    const select = document.getElementById('mission-select');
    const missionId = select.value;

    if (!missionId) {
        showToast('Please select a mission', 'warning');
        return;
    }

    sendCommand('LOAD_MISSION', { mission_id: missionId });
}

/**
 * Start loaded mission
 */
function startMission() {
    sendCommand('START_MISSION');
}

/**
 * Pause current mission
 */
function pauseMission() {
    sendCommand('PAUSE_MISSION');
}

/**
 * Abort current mission
 */
function abortMission() {
    const content = `
        <p>Are you sure you want to abort the current mission?</p>
        <p>The drone will hold position after aborting.</p>
    `;

    showDialog('Abort Mission', content, [
        { text: 'Cancel', action: closeDialog },
        {
            text: 'Abort Mission',
            action: () => {
                sendCommand('ABORT_MISSION');
                closeDialog();
            },
            danger: true
        }
    ]);
}

/**
 * Set tracking pattern
 */
function setTrackingPattern() {
    const pattern = document.getElementById('tracking-pattern').value;
    sendCommand('SET_TRACKING_PATTERN', { pattern: pattern });
}

/**
 * Stop tracking
 */
function stopTracking() {
    sendCommand('STOP_TRACKING');
}

/**
 * Switch camera feed
 */
function switchCamera(camera) {
    // Update button states
    document.getElementById('btn-rgb').classList.remove('active');
    document.getElementById('btn-thermal').classList.remove('active');
    document.getElementById(`btn-${camera}`).classList.add('active');

    // Request new stream
    if (wsClient) {
        wsClient.requestVideoStream(camera);
    }

    currentCamera = camera;
}

/**
 * Toggle Picture-in-Picture mode
 */
let pipEnabled = false;
let currentCamera = 'rgb';

function togglePiP() {
    pipEnabled = !pipEnabled;
    const pipElement = document.getElementById('video-pip');
    const pipButton = document.getElementById('btn-pip');

    if (pipEnabled) {
        pipElement.classList.remove('hidden');
        pipButton.classList.add('active');

        // Request secondary stream
        const secondaryCamera = currentCamera === 'rgb' ? 'thermal' : 'rgb';
        // In a real implementation, we'd request a secondary stream
    } else {
        pipElement.classList.add('hidden');
        pipButton.classList.remove('active');
    }
}

/**
 * Fetch available missions from API
 */
async function fetchMissions() {
    try {
        const response = await fetch('/api/missions');
        if (!response.ok) throw new Error('Failed to fetch missions');

        const missions = await response.json();
        const select = document.getElementById('mission-select');

        // Clear existing options
        select.innerHTML = '<option value="">Select Mission</option>';

        // Add missions
        missions.forEach(mission => {
            const option = document.createElement('option');
            option.value = mission.id;
            option.textContent = mission.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to fetch missions:', error);
    }
}

/**
 * Update button states based on flight state
 */
function updateControlStates(telemetry) {
    const armed = telemetry.armed;
    const isFlying = telemetry.is_flying;
    const state = telemetry.state;

    // ARM/DISARM buttons
    const armBtn = document.getElementById('btn-arm');
    const disarmBtn = document.getElementById('btn-disarm');
    if (armBtn) armBtn.disabled = armed;
    if (disarmBtn) disarmBtn.disabled = !armed || isFlying;

    // Takeoff button - only when armed and on ground
    const takeoffBtn = document.getElementById('btn-takeoff');
    if (takeoffBtn) {
        takeoffBtn.disabled = !armed || isFlying;
    }

    // Land/RTH buttons - only when flying
    const landBtn = document.getElementById('btn-land');
    const rthBtn = document.getElementById('btn-rth');
    const holdBtn = document.getElementById('btn-hold');
    if (landBtn) landBtn.disabled = !isFlying;
    if (rthBtn) rthBtn.disabled = !isFlying;
    if (holdBtn) holdBtn.disabled = !isFlying;

    // Mission controls
    const missionBtns = document.querySelectorAll('.mission-buttons button');
    missionBtns.forEach(btn => {
        // Enable based on current mission status
        if (btn.textContent === 'Start') {
            btn.disabled = !isFlying || telemetry.mission_status === 'IN_PROGRESS';
        } else if (btn.textContent === 'Pause') {
            btn.disabled = telemetry.mission_status !== 'IN_PROGRESS';
        } else if (btn.textContent === 'Abort') {
            btn.disabled = !['IN_PROGRESS', 'PAUSED'].includes(telemetry.mission_status);
        }
    });

    // Go To button - only when flying
    const gotoBtn = document.querySelector('.btn-goto');
    if (gotoBtn) gotoBtn.disabled = !isFlying;

    // Tracking controls
    const trackingStatus = document.getElementById('tracking-status');
    const isTracking = telemetry.is_monitoring_or_tracking;
    if (trackingStatus) {
        trackingStatus.textContent = isTracking ? 'Active' : 'Inactive';
        trackingStatus.className = 'value ' + (isTracking ? 'active' : '');
    }
}

/**
 * Show dialog modal
 */
function showDialog(title, content, actions) {
    const overlay = document.getElementById('dialog-overlay');
    const titleEl = document.getElementById('dialog-title');
    const contentEl = document.getElementById('dialog-content');
    const actionsEl = document.getElementById('dialog-actions');

    titleEl.textContent = title;
    contentEl.innerHTML = content;

    // Clear and add action buttons
    actionsEl.innerHTML = '';
    actions.forEach(action => {
        const btn = document.createElement('button');
        btn.className = 'btn';
        if (action.primary) btn.classList.add('btn-primary');
        if (action.danger) btn.classList.add('btn-danger');
        btn.textContent = action.text;
        btn.onclick = action.action;
        actionsEl.appendChild(btn);
    });

    overlay.classList.remove('hidden');
}

/**
 * Close dialog modal
 */
function closeDialog() {
    document.getElementById('dialog-overlay').classList.add('hidden');
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);

    // Remove after duration
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Handle command acknowledgment
 */
function handleCommandAck(ack) {
    if (ack.success) {
        showToast(`${ack.command}: ${ack.message || 'Success'}`, 'success');
    } else {
        showToast(`${ack.command} failed: ${ack.error || 'Unknown error'}`, 'error');
    }
}

// Close dialog on overlay click
document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('dialog-overlay');
    if (overlay) {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeDialog();
            }
        });
    }

    // Close dialog on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeDialog();
        }
    });
});

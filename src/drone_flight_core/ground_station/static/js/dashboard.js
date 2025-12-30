/**
 * Main Dashboard Controller
 *
 * Initializes all components and manages
 * the overall ground station dashboard.
 */

// Global state
let currentDetections = new Map();
let alertHistory = [];

/**
 * Initialize dashboard on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Ground Station Dashboard...');

    // Initialize components
    initTelemetry();
    initClock();
    initWebSocket();
    fetchMissions();

    console.log('Dashboard initialized');
});

/**
 * Initialize clock display
 */
function initClock() {
    function updateClock() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
        const clockEl = document.getElementById('clock');
        if (clockEl) {
            clockEl.textContent = timeStr;
        }
    }

    updateClock();
    setInterval(updateClock, 1000);
}

/**
 * Initialize WebSocket connections and handlers
 */
function initWebSocket() {
    // Connection status handler
    wsClient.on('connection', (status) => {
        updateConnectionStatus(status);
    });

    // Telemetry handler
    wsClient.on('telemetry', (data) => {
        if (telemetryManager) {
            telemetryManager.update(data);
        }
        updateControlStates(data);
    });

    // Video frame handler
    wsClient.on('video', (data) => {
        handleVideoFrame(data);
    });

    // Alert handler
    wsClient.on('alert', (alert) => {
        handleAlert(alert);
    });

    // Detection handler
    wsClient.on('detection', (detection) => {
        handleDetection(detection);
    });

    // Command acknowledgment handler
    wsClient.on('command_ack', (ack) => {
        handleCommandAck(ack);
    });

    // Connect to WebSocket endpoints
    wsClient.connect();
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(status) {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;

    const dotEl = statusEl.querySelector('.status-dot');
    const textEl = statusEl.querySelector('.status-text');

    if (status.connected) {
        dotEl.className = 'status-dot connected';
        textEl.textContent = 'Connected';
    } else {
        dotEl.className = 'status-dot disconnected';
        textEl.textContent = 'Disconnected';

        // Clear telemetry display
        if (telemetryManager) {
            telemetryManager.clear();
        }
    }
}

/**
 * Handle incoming video frames
 */
let lastFrameUrl = null;

function handleVideoFrame(data) {
    if (data.type === 'frame') {
        const videoEl = document.getElementById('video-main');
        if (videoEl) {
            // Revoke old URL to prevent memory leak
            if (lastFrameUrl) {
                URL.revokeObjectURL(lastFrameUrl);
            }
            lastFrameUrl = data.url;
            videoEl.src = data.url;
        }
    } else if (data.type === 'info') {
        // Update video info overlay
        const fpsEl = document.getElementById('video-fps');
        const resEl = document.getElementById('video-resolution');

        if (fpsEl && data.data.fps) {
            fpsEl.textContent = `${data.data.fps} FPS`;
        }
        if (resEl && data.data.width && data.data.height) {
            resEl.textContent = `${data.data.width}x${data.data.height}`;
        }
    }
}

/**
 * Handle detection updates
 */
function handleDetection(detection) {
    // Update or add detection
    currentDetections.set(detection.track_id, {
        ...detection,
        lastSeen: Date.now()
    });

    // Update detection overlay
    updateDetectionOverlay();

    // Update detections list
    updateDetectionsList();

    // Remove stale detections
    cleanStaleDetections();
}

/**
 * Update SVG overlay with detection bounding boxes
 */
function updateDetectionOverlay() {
    const overlay = document.getElementById('detection-overlay');
    if (!overlay) return;

    // Clear existing elements
    overlay.innerHTML = '';

    const videoEl = document.getElementById('video-main');
    if (!videoEl) return;

    // Get video dimensions
    const videoWidth = videoEl.naturalWidth || 640;
    const videoHeight = videoEl.naturalHeight || 480;
    const displayWidth = videoEl.clientWidth;
    const displayHeight = videoEl.clientHeight;

    // Scale factors
    const scaleX = displayWidth / videoWidth;
    const scaleY = displayHeight / videoHeight;

    currentDetections.forEach((detection, trackId) => {
        if (!detection.bbox) return;

        const [x1, y1, x2, y2] = detection.bbox;

        // Create bounding box
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x1 * scaleX);
        rect.setAttribute('y', y1 * scaleY);
        rect.setAttribute('width', (x2 - x1) * scaleX);
        rect.setAttribute('height', (y2 - y1) * scaleY);
        rect.setAttribute('class', `detection-box ${detection.confirmed ? 'confirmed' : ''}`);
        overlay.appendChild(rect);

        // Create label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x1 * scaleX);
        text.setAttribute('y', y1 * scaleY - 5);
        text.setAttribute('class', 'detection-label');
        text.textContent = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`;
        overlay.appendChild(text);
    });
}

/**
 * Update detections list panel
 */
function updateDetectionsList() {
    const listEl = document.getElementById('detections-list');
    if (!listEl) return;

    // Sort by confidence
    const sorted = Array.from(currentDetections.values())
        .sort((a, b) => b.confidence - a.confidence);

    if (sorted.length === 0) {
        listEl.innerHTML = '<div class="no-detections">No active detections</div>';
        return;
    }

    listEl.innerHTML = sorted.map(det => `
        <div class="detection-item ${det.confirmed ? 'confirmed' : ''}" data-track-id="${det.track_id}">
            <div class="detection-header">
                <span class="detection-class">${det.class_name}</span>
                <span class="detection-confidence">${(det.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="detection-info">
                <span>Track ID: ${det.track_id}</span>
                ${det.confirmed ? '<span class="confirmed-badge">Confirmed</span>' : ''}
            </div>
        </div>
    `).join('');
}

/**
 * Clean up stale detections (older than 5 seconds)
 */
function cleanStaleDetections() {
    const now = Date.now();
    const staleThreshold = 5000;

    currentDetections.forEach((detection, trackId) => {
        if (now - detection.lastSeen > staleThreshold) {
            currentDetections.delete(trackId);
        }
    });
}

/**
 * Handle incoming alerts
 */
function handleAlert(alert) {
    // Add to history
    alertHistory.unshift(alert);
    if (alertHistory.length > 100) {
        alertHistory = alertHistory.slice(0, 100);
    }

    // Update alerts display
    updateAlertsDisplay();

    // Update alert count badge
    updateAlertCount();

    // Show toast for important alerts
    if (alert.severity === 'CRITICAL' || alert.severity === 'EMERGENCY') {
        showToast(alert.title + ': ' + alert.message, 'error', 5000);
    } else if (alert.severity === 'WARNING') {
        showToast(alert.title, 'warning');
    }

    // Play sound for critical alerts
    if (alert.severity === 'EMERGENCY') {
        playAlertSound();
    }
}

/**
 * Update alerts display
 */
function updateAlertsDisplay() {
    const container = document.getElementById('alerts-container');
    if (!container) return;

    const unacknowledged = alertHistory.filter(a => !a.acknowledged);

    if (unacknowledged.length === 0) {
        container.innerHTML = '<div class="no-alerts">No alerts</div>';
        return;
    }

    container.innerHTML = unacknowledged.slice(0, 20).map(alert => `
        <div class="alert-item alert-${alert.severity.toLowerCase()}" data-alert-id="${alert.id}">
            <div class="alert-header">
                <span class="alert-title">${alert.title}</span>
                <span class="alert-time">${formatTime(alert.timestamp)}</span>
            </div>
            <div class="alert-message">${alert.message}</div>
            <button class="btn btn-sm alert-ack" onclick="acknowledgeAlert('${alert.id}')">
                Acknowledge
            </button>
        </div>
    `).join('');
}

/**
 * Update alert count badge
 */
function updateAlertCount() {
    const badge = document.getElementById('alert-count');
    if (!badge) return;

    const unacknowledgedCount = alertHistory.filter(a => !a.acknowledged).length;
    badge.textContent = unacknowledgedCount;
    badge.className = 'alert-badge' + (unacknowledgedCount > 0 ? ' has-alerts' : '');
}

/**
 * Acknowledge an alert
 */
function acknowledgeAlert(alertId) {
    const alert = alertHistory.find(a => a.id === alertId);
    if (alert) {
        alert.acknowledged = true;
        updateAlertsDisplay();
        updateAlertCount();

        // Send acknowledgment to server
        fetch(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' })
            .catch(err => console.error('Failed to acknowledge alert:', err));
    }
}

/**
 * Clear all acknowledged alerts
 */
function clearAlerts() {
    alertHistory = alertHistory.filter(a => !a.acknowledged);
    updateAlertsDisplay();
    updateAlertCount();
}

/**
 * Play alert sound
 */
function playAlertSound() {
    try {
        // Create a simple beep using Web Audio API
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 880;
        oscillator.type = 'square';
        gainNode.gain.value = 0.3;

        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.3);
    } catch (e) {
        console.log('Could not play alert sound:', e);
    }
}

/**
 * Format timestamp for display
 */
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour12: false });
}

/**
 * Update tracking target display
 */
function updateTrackingTarget(target) {
    const targetEl = document.getElementById('tracking-target');
    if (targetEl) {
        if (target) {
            targetEl.textContent = `${target.class_name} (ID: ${target.track_id})`;
        } else {
            targetEl.textContent = 'None';
        }
    }
}

// Periodically clean stale detections
setInterval(cleanStaleDetections, 1000);

// Periodically update detection overlay (in case video resizes)
setInterval(updateDetectionOverlay, 500);

// Handle window resize
window.addEventListener('resize', () => {
    updateDetectionOverlay();
});

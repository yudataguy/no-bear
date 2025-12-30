/**
 * Telemetry Display Manager
 *
 * Updates UI elements with real-time telemetry data
 * and manages attitude indicator visualization.
 */

class TelemetryManager {
    constructor() {
        this.lastUpdate = null;
        this.updateRate = 0;
        this.updateCount = 0;
        this.rateInterval = null;

        // Cache DOM elements
        this.elements = {
            // Flight status
            flightState: document.getElementById('flight-state'),
            armedStatus: document.getElementById('armed-status'),
            flightMode: document.getElementById('flight-mode'),
            missionStatus: document.getElementById('mission-status'),

            // Position
            gpsLat: document.getElementById('gps-lat'),
            gpsLon: document.getElementById('gps-lon'),
            altitude: document.getElementById('altitude'),
            gpsSats: document.getElementById('gps-sats'),

            // Attitude
            roll: document.getElementById('roll'),
            pitch: document.getElementById('pitch'),
            yaw: document.getElementById('yaw'),
            horizon: document.getElementById('horizon'),
            attitudeIndicator: document.getElementById('attitude-indicator'),

            // Velocity
            groundSpeed: document.getElementById('ground-speed'),
            climbRate: document.getElementById('climb-rate'),
            heading: document.getElementById('heading'),

            // Battery
            batteryBar: document.getElementById('battery-bar'),
            batteryPercent: document.getElementById('battery-percent'),
            batteryVoltage: document.getElementById('battery-voltage'),
            batteryCurrent: document.getElementById('battery-current'),

            // Status
            telemetryRate: document.getElementById('telemetry-rate'),
            droneId: document.getElementById('drone-id')
        };

        // Start rate calculation
        this.startRateCalculation();
    }

    /**
     * Start calculating update rate
     */
    startRateCalculation() {
        this.rateInterval = setInterval(() => {
            this.updateRate = this.updateCount;
            this.updateCount = 0;
            if (this.elements.telemetryRate) {
                this.elements.telemetryRate.textContent = `${this.updateRate} Hz`;
            }
        }, 1000);
    }

    /**
     * Update all telemetry displays
     */
    update(telemetry) {
        this.updateCount++;
        this.lastUpdate = Date.now();

        // Update each section
        this.updateFlightStatus(telemetry);
        this.updatePosition(telemetry);
        this.updateAttitude(telemetry);
        this.updateVelocity(telemetry);
        this.updateBattery(telemetry);

        // Update drone ID if present
        if (telemetry.drone_id && this.elements.droneId) {
            this.elements.droneId.textContent = telemetry.drone_id;
        }
    }

    /**
     * Update flight status section
     */
    updateFlightStatus(telemetry) {
        if (telemetry.state && this.elements.flightState) {
            const stateEl = this.elements.flightState;
            const valueEl = stateEl.querySelector('.value');
            if (valueEl) {
                valueEl.textContent = telemetry.state;
            }

            // Update state indicator class
            stateEl.className = 'status-item state-indicator';
            if (telemetry.is_emergency) {
                stateEl.classList.add('emergency');
            } else if (telemetry.is_flying) {
                stateEl.classList.add('flying');
            } else if (telemetry.armed) {
                stateEl.classList.add('armed');
            }
        }

        if (this.elements.armedStatus) {
            const valueEl = this.elements.armedStatus.querySelector('.value');
            if (valueEl) {
                valueEl.textContent = telemetry.armed ? 'YES' : 'NO';
                valueEl.className = 'value ' + (telemetry.armed ? 'armed' : '');
            }
        }

        if (telemetry.mode && this.elements.flightMode) {
            const valueEl = this.elements.flightMode.querySelector('.value');
            if (valueEl) {
                valueEl.textContent = telemetry.mode;
            }
        }

        if (telemetry.mission_status && this.elements.missionStatus) {
            const valueEl = this.elements.missionStatus.querySelector('.value');
            if (valueEl) {
                valueEl.textContent = telemetry.mission_status;
            }
        }
    }

    /**
     * Update position section
     */
    updatePosition(telemetry) {
        if (telemetry.latitude !== undefined && this.elements.gpsLat) {
            this.elements.gpsLat.textContent = telemetry.latitude.toFixed(6);
        }

        if (telemetry.longitude !== undefined && this.elements.gpsLon) {
            this.elements.gpsLon.textContent = telemetry.longitude.toFixed(6);
        }

        if (telemetry.altitude !== undefined && this.elements.altitude) {
            this.elements.altitude.textContent = `${telemetry.altitude.toFixed(1)} m`;
        }

        if (telemetry.gps_satellites !== undefined && this.elements.gpsSats) {
            this.elements.gpsSats.textContent = telemetry.gps_satellites;
            // Add warning class if low satellite count
            this.elements.gpsSats.className = 'value' + (telemetry.gps_satellites < 6 ? ' warning' : '');
        }
    }

    /**
     * Update attitude section with visual indicator
     */
    updateAttitude(telemetry) {
        const roll = telemetry.roll || 0;
        const pitch = telemetry.pitch || 0;
        const yaw = telemetry.yaw || 0;

        // Update text values
        if (this.elements.roll) {
            this.elements.roll.textContent = `${roll.toFixed(1)}°`;
        }
        if (this.elements.pitch) {
            this.elements.pitch.textContent = `${pitch.toFixed(1)}°`;
        }
        if (this.elements.yaw) {
            this.elements.yaw.textContent = `${yaw.toFixed(1)}°`;
        }

        // Update attitude indicator visualization
        if (this.elements.horizon) {
            // Pitch affects vertical position (scale: 2px per degree)
            const pitchOffset = pitch * 2;
            // Roll rotates the horizon
            this.elements.horizon.style.transform =
                `translateY(${pitchOffset}px) rotate(${-roll}deg)`;
        }

        // Update roll indicator
        if (this.elements.attitudeIndicator) {
            const rollIndicator = this.elements.attitudeIndicator.querySelector('.roll-indicator');
            if (rollIndicator) {
                rollIndicator.style.transform = `rotate(${-roll}deg)`;
            }
        }
    }

    /**
     * Update velocity section
     */
    updateVelocity(telemetry) {
        if (telemetry.ground_speed !== undefined && this.elements.groundSpeed) {
            this.elements.groundSpeed.textContent = `${telemetry.ground_speed.toFixed(1)} m/s`;
        }

        if (telemetry.climb_rate !== undefined && this.elements.climbRate) {
            const rate = telemetry.climb_rate;
            this.elements.climbRate.textContent = `${rate >= 0 ? '+' : ''}${rate.toFixed(1)} m/s`;
            this.elements.climbRate.className = 'value' + (rate > 0 ? ' climbing' : rate < 0 ? ' descending' : '');
        }

        if (telemetry.heading !== undefined && this.elements.heading) {
            this.elements.heading.textContent = `${telemetry.heading.toFixed(0)}°`;
        }
    }

    /**
     * Update battery section
     */
    updateBattery(telemetry) {
        const percent = telemetry.battery_percent || 0;
        const voltage = telemetry.battery_voltage || 0;
        const current = telemetry.battery_current || 0;

        if (this.elements.batteryBar) {
            this.elements.batteryBar.style.width = `${percent}%`;

            // Update color based on level
            this.elements.batteryBar.className = 'battery-bar';
            if (percent <= 20) {
                this.elements.batteryBar.classList.add('critical');
            } else if (percent <= 40) {
                this.elements.batteryBar.classList.add('low');
            }
        }

        if (this.elements.batteryPercent) {
            this.elements.batteryPercent.textContent = `${percent.toFixed(0)}%`;
        }

        if (this.elements.batteryVoltage) {
            this.elements.batteryVoltage.textContent = `${voltage.toFixed(1)} V`;
        }

        if (this.elements.batteryCurrent) {
            this.elements.batteryCurrent.textContent = `${current.toFixed(1)} A`;
        }
    }

    /**
     * Update mission progress
     */
    updateMissionProgress(progress) {
        const progressBar = document.getElementById('mission-progress');
        const waypointText = document.getElementById('mission-waypoint');

        if (progressBar) {
            const percent = (progress.current_waypoint / progress.total_waypoints) * 100;
            progressBar.style.width = `${percent}%`;
        }

        if (waypointText) {
            waypointText.textContent = `Waypoint: ${progress.current_waypoint}/${progress.total_waypoints}`;
        }
    }

    /**
     * Clear telemetry display (on disconnect)
     */
    clear() {
        // Reset all values to placeholder
        Object.values(this.elements).forEach(el => {
            if (el) {
                const valueEl = el.querySelector ? el.querySelector('.value') : null;
                if (valueEl) {
                    valueEl.textContent = '--';
                } else if (el.id && !el.id.includes('bar')) {
                    el.textContent = '--';
                }
            }
        });

        // Reset battery bar
        if (this.elements.batteryBar) {
            this.elements.batteryBar.style.width = '0%';
        }
    }

    /**
     * Cleanup
     */
    destroy() {
        if (this.rateInterval) {
            clearInterval(this.rateInterval);
        }
    }
}

// Global telemetry manager instance
let telemetryManager = null;

function initTelemetry() {
    telemetryManager = new TelemetryManager();
    return telemetryManager;
}

# Drone Flight Core

Comprehensive drone flight software with autonomous object tracking capabilities.

## Features

- **Flight Control**: State machine-based flight control with PX4/ArduPilot integration via MAVLink
- **Safety Systems**: Geofencing, battery monitoring, signal loss detection, automatic RTH
- **Dual Camera Vision**: RGB + Thermal camera fusion for object detection
- **Object Tracking**: Continuous tracking with adaptive patterns (hover/orbit)
- **Ground Station API**: REST and WebSocket APIs for remote control
- **Data Logging**: Comprehensive flight recording and telemetry logging

## Architecture

```
drone-flight-core/
├── src/drone_flight_core/
│   ├── core/              # State machine, safety, config
│   ├── communication/     # MAVLink, telemetry, commands
│   ├── vision/            # Camera, detection, tracking, fusion
│   ├── navigation/        # Waypoints, path planning (TODO)
│   ├── sensors/           # Sensor abstraction (TODO)
│   ├── ground_station/    # REST/WebSocket API
│   ├── data/              # Logging, storage, recording
│   └── simulation/        # SITL support (TODO)
├── rust_src/              # Performance-critical Rust code
├── config/                # Configuration files
├── tests/                 # Unit tests
└── logs/                  # Log output
```

## Installation

### Prerequisites

- Python 3.10+
- Rust 1.70+ (for Rust components)
- OpenCV (for vision)

### Install

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Build Rust components
maturin develop
```

## Usage

### Run Full Application

```bash
# With configuration file
drone-core -c config/default.json

# In simulation mode
drone-core --simulation

# Ground station only
drone-core --ground-station-only
```

### Ground Station API

The ground station exposes a REST API at `http://localhost:8080/api/v1/`:

- `GET /status` - System status
- `GET /telemetry` - Current telemetry
- `POST /command` - Execute command
- `POST /goto` - Navigate to coordinates
- `WS /ws` - WebSocket for real-time updates

### Example: ARM and Takeoff

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"

# Arm
requests.post(f"{BASE_URL}/command", json={"command": "ARM"})

# Takeoff
requests.post(f"{BASE_URL}/command", json={"command": "TAKEOFF"})

# Navigate to coordinates
requests.post(f"{BASE_URL}/goto", json={
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 50.0
})
```

## Configuration

Key configuration options in `config/default.json`:

```json
{
  "safety": {
    "low_battery_threshold_percent": 20.0,
    "geofence": {
      "max_altitude_m": 120.0,
      "max_distance_m": 1000.0
    }
  },
  "vision": {
    "processing_fps": 10.0,
    "fusion_mode": "weighted",
    "fusion_weight_thermal": 0.6
  },
  "tracking": {
    "default_pattern": "hover",
    "energy_aware_pattern_selection": true
  }
}
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=drone_flight_core

# Run specific test file
pytest tests/test_state_machine.py
```

## Development

### Project Structure

- **Core**: Flight state machine and safety systems
- **Communication**: MAVLink protocol handling
- **Vision**: Dual camera processing and object tracking
- **Ground Station**: External API for control and monitoring
- **Data**: Logging, recording, and storage

### Adding Features

1. Create module in appropriate package
2. Add configuration in `core/config.py`
3. Write tests in `tests/`
4. Update `__init__.py` exports

## License

MIT

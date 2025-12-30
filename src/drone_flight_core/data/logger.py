"""
Structured logging configuration for drone flight core.

Provides consistent logging across all modules with support
for file and console output with automatic rotation.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import Processor

from drone_flight_core.core.config import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> None:
    """
    Configure structured logging for the application.

    Args:
        config: Logging configuration. Uses defaults if None.
    """
    if config is None:
        config = LoggingConfig()

    # Create log directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up standard logging
    log_level = getattr(logging, config.level, logging.INFO)

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    log_file = log_dir / f"drone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_log_size_mb * 1024 * 1024,
        backupCount=config.max_log_files,
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure structlog
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if sys.stdout.isatty():
        # Pretty printing for console
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # JSON for file/production
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger for a module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


class TelemetryLogger:
    """
    Specialized logger for high-frequency telemetry data.

    Writes telemetry to a separate file with rate limiting
    to avoid overwhelming storage.
    """

    def __init__(
        self,
        log_dir: Path,
        rate_hz: float = 1.0,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._rate_hz = rate_hz
        self._min_interval = 1.0 / rate_hz
        self._last_log_time = 0.0

        # Create telemetry log file
        self._log_file = self._log_dir / f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._file_handle = open(self._log_file, "a")

        self._log_count = 0

    def log(self, data: dict[str, Any]) -> bool:
        """
        Log telemetry data if rate limit allows.

        Args:
            data: Telemetry data dictionary.

        Returns:
            True if data was logged.
        """
        import time
        import json

        current_time = time.time()

        if current_time - self._last_log_time < self._min_interval:
            return False

        # Add timestamp
        data["_timestamp"] = datetime.now().isoformat()
        data["_seq"] = self._log_count

        # Write to file
        self._file_handle.write(json.dumps(data) + "\n")
        self._file_handle.flush()

        self._last_log_time = current_time
        self._log_count += 1

        return True

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()

    @property
    def log_count(self) -> int:
        """Number of telemetry entries logged."""
        return self._log_count

    @property
    def log_file(self) -> Path:
        """Path to current log file."""
        return self._log_file


class EventLogger:
    """
    Logger for discrete events (state changes, commands, etc).

    Events are logged with full detail for post-flight analysis.
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._log_file = self._log_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._file_handle = open(self._log_file, "a")

        self._event_count = 0
        self._logger = structlog.get_logger("events")

    def log_event(
        self,
        event_type: str,
        data: dict[str, Any],
        severity: str = "info",
    ) -> None:
        """
        Log an event.

        Args:
            event_type: Type of event (e.g., "state_change", "command").
            data: Event data.
            severity: Event severity (debug, info, warning, error).
        """
        import json

        event = {
            "_timestamp": datetime.now().isoformat(),
            "_seq": self._event_count,
            "event_type": event_type,
            "severity": severity,
            "data": data,
        }

        # Write to file
        self._file_handle.write(json.dumps(event) + "\n")
        self._file_handle.flush()

        # Also log via structlog
        log_method = getattr(self._logger, severity, self._logger.info)
        log_method(event_type, **data)

        self._event_count += 1

    def log_state_change(
        self,
        from_state: str,
        to_state: str,
        reason: str = "",
    ) -> None:
        """Log a state change event."""
        self.log_event(
            "state_change",
            {
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
            },
        )

    def log_command(
        self,
        command: str,
        params: dict[str, Any],
        result: str = "pending",
    ) -> None:
        """Log a command event."""
        self.log_event(
            "command",
            {
                "command": command,
                "params": params,
                "result": result,
            },
        )

    def log_safety_violation(
        self,
        violation_type: str,
        message: str,
        action: str,
    ) -> None:
        """Log a safety violation event."""
        self.log_event(
            "safety_violation",
            {
                "violation_type": violation_type,
                "message": message,
                "action": action,
            },
            severity="warning",
        )

    def log_detection(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        confidence: float,
        class_name: str,
    ) -> None:
        """Log a detection event."""
        self.log_event(
            "detection",
            {
                "track_id": track_id,
                "bbox": bbox,
                "confidence": confidence,
                "class_name": class_name,
            },
            severity="debug",
        )

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()

    @property
    def event_count(self) -> int:
        """Number of events logged."""
        return self._event_count


# Import logging.handlers for RotatingFileHandler
import logging.handlers

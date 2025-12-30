"""
Alert system for detection events and mission status.

Provides notifications to ground station via WebSocket and
optional external integrations (webhooks, SMS, etc.).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable
import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Severity level of alerts."""

    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class AlertType(Enum):
    """Type of alert."""

    DETECTION = auto()  # Object detected
    DETECTION_CONFIRMED = auto()  # Detection confirmed after multiple sightings
    DETECTION_LOST = auto()  # Lost track of detected object
    MISSION_STARTED = auto()
    MISSION_COMPLETED = auto()
    MISSION_PAUSED = auto()
    MISSION_ABORTED = auto()
    WAYPOINT_REACHED = auto()
    LOW_BATTERY = auto()
    SIGNAL_LOST = auto()
    GEOFENCE_WARNING = auto()
    EMERGENCY = auto()


@dataclass
class Alert:
    """Alert notification."""

    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # Location where alert was triggered
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None

    # For detection alerts
    track_id: int | None = None
    detection_class: str | None = None
    confidence: float | None = None
    bbox: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.alert_type.name,
            "severity": self.severity.name,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
            }
            if self.latitude is not None
            else None,
            "detection": {
                "track_id": self.track_id,
                "class": self.detection_class,
                "confidence": self.confidence,
                "bbox": self.bbox,
            }
            if self.track_id is not None
            else None,
            "metadata": self.metadata,
        }


class AlertManager:
    """
    Manages alert generation and distribution.

    Features:
    - Alert history
    - WebSocket broadcast
    - Webhook callbacks
    - Alert acknowledgment
    - Severity filtering
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._alerts: list[Alert] = []
        self._max_history = max_history
        self._alert_counter = 0

        # Subscribers
        self._ws_broadcast: Callable[[dict], None] | None = None
        self._webhooks: list[str] = []
        self._callbacks: list[Callable[[Alert], None]] = []

        # Filtering
        self._min_severity = AlertSeverity.INFO

        logger.info("Alert manager initialized")

    def set_websocket_broadcast(self, broadcast_fn: Callable[[dict], None]) -> None:
        """Set WebSocket broadcast function."""
        self._ws_broadcast = broadcast_fn

    def add_webhook(self, url: str) -> None:
        """Add webhook URL for alert notifications."""
        self._webhooks.append(url)

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications."""
        self._callbacks.append(callback)

    def set_minimum_severity(self, severity: AlertSeverity) -> None:
        """Set minimum severity for alerts to be processed."""
        self._min_severity = severity

    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"alert_{self._alert_counter:06d}"

    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        **kwargs,
    ) -> Alert:
        """
        Create and distribute an alert.

        Args:
            alert_type: Type of alert.
            severity: Alert severity.
            title: Alert title.
            message: Alert message.
            **kwargs: Additional alert fields.

        Returns:
            Created Alert object.
        """
        # Check severity filter
        if severity.value < self._min_severity.value:
            return None

        alert = Alert(
            id=self._generate_id(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            **kwargs,
        )

        # Store in history
        self._alerts.append(alert)
        if len(self._alerts) > self._max_history:
            self._alerts = self._alerts[-self._max_history:]

        logger.info(
            "Alert created",
            id=alert.id,
            type=alert_type.name,
            severity=severity.name,
        )

        # Distribute alert
        await self._distribute_alert(alert)

        return alert

    async def _distribute_alert(self, alert: Alert) -> None:
        """Distribute alert to all subscribers."""
        alert_data = {
            "type": "alert",
            "data": alert.to_dict(),
        }

        # WebSocket broadcast
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(alert_data)
            except Exception as e:
                logger.error("WebSocket broadcast failed", error=str(e))

        # Webhooks (async, fire and forget)
        for webhook_url in self._webhooks:
            asyncio.create_task(self._send_webhook(webhook_url, alert))

        # Callbacks
        for callback in self._callbacks:
            try:
                result = callback(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))

    async def _send_webhook(self, url: str, alert: Alert) -> None:
        """Send alert to webhook URL."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=alert.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status >= 400:
                        logger.warning(
                            "Webhook failed",
                            url=url,
                            status=response.status,
                        )
        except ImportError:
            logger.warning("aiohttp not installed, webhook disabled")
        except Exception as e:
            logger.error("Webhook error", url=url, error=str(e))

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.debug("Alert acknowledged", id=alert_id)
                return True
        return False

    def get_alerts(
        self,
        unacknowledged_only: bool = False,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        """
        Get alerts with optional filtering.

        Args:
            unacknowledged_only: Only return unacknowledged alerts.
            severity: Filter by severity (and above).
            alert_type: Filter by type.
            limit: Maximum alerts to return.

        Returns:
            List of matching alerts.
        """
        alerts = self._alerts.copy()

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity.value >= severity.value]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        # Return most recent first
        return list(reversed(alerts[-limit:]))

    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alerts."""
        return sum(1 for a in self._alerts if not a.acknowledged)

    def clear_acknowledged(self) -> int:
        """Clear all acknowledged alerts from history."""
        before = len(self._alerts)
        self._alerts = [a for a in self._alerts if not a.acknowledged]
        cleared = before - len(self._alerts)
        logger.debug("Cleared acknowledged alerts", count=cleared)
        return cleared

    # Convenience methods for common alerts

    async def alert_detection(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        location: tuple[float, float, float],
        bbox: tuple[int, int, int, int],
        confirmed: bool = False,
    ) -> Alert:
        """Create detection alert."""
        severity = AlertSeverity.WARNING if not confirmed else AlertSeverity.CRITICAL
        alert_type = (
            AlertType.DETECTION_CONFIRMED if confirmed else AlertType.DETECTION
        )

        title = f"{'Confirmed: ' if confirmed else ''}Object Detected"
        message = (
            f"Detected {class_name} (confidence: {confidence:.1%}) "
            f"at ({location[0]:.6f}, {location[1]:.6f})"
        )

        return await self.create_alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            latitude=location[0],
            longitude=location[1],
            altitude=location[2],
            track_id=track_id,
            detection_class=class_name,
            confidence=confidence,
            bbox=bbox,
        )

    async def alert_mission_status(
        self,
        status: str,
        mission_name: str,
        details: str = "",
    ) -> Alert:
        """Create mission status alert."""
        type_map = {
            "started": AlertType.MISSION_STARTED,
            "completed": AlertType.MISSION_COMPLETED,
            "paused": AlertType.MISSION_PAUSED,
            "aborted": AlertType.MISSION_ABORTED,
        }

        alert_type = type_map.get(status.lower(), AlertType.MISSION_STARTED)
        severity = (
            AlertSeverity.INFO
            if status in ["started", "completed"]
            else AlertSeverity.WARNING
        )

        return await self.create_alert(
            alert_type=alert_type,
            severity=severity,
            title=f"Mission {status.capitalize()}",
            message=f"Mission '{mission_name}' {status}. {details}".strip(),
            metadata={"mission_name": mission_name, "status": status},
        )

    async def alert_waypoint_reached(
        self,
        waypoint_index: int,
        total_waypoints: int,
        location: tuple[float, float, float],
    ) -> Alert:
        """Create waypoint reached alert."""
        return await self.create_alert(
            alert_type=AlertType.WAYPOINT_REACHED,
            severity=AlertSeverity.INFO,
            title="Waypoint Reached",
            message=f"Reached waypoint {waypoint_index + 1} of {total_waypoints}",
            latitude=location[0],
            longitude=location[1],
            altitude=location[2],
            metadata={
                "waypoint_index": waypoint_index,
                "total_waypoints": total_waypoints,
            },
        )

    async def alert_emergency(
        self,
        reason: str,
        action_taken: str,
        location: tuple[float, float, float] | None = None,
    ) -> Alert:
        """Create emergency alert."""
        return await self.create_alert(
            alert_type=AlertType.EMERGENCY,
            severity=AlertSeverity.EMERGENCY,
            title="EMERGENCY",
            message=f"{reason}. Action: {action_taken}",
            latitude=location[0] if location else None,
            longitude=location[1] if location else None,
            altitude=location[2] if location else None,
            metadata={"reason": reason, "action": action_taken},
        )

    def to_dict(self) -> dict[str, Any]:
        """Export alert manager status."""
        return {
            "total_alerts": len(self._alerts),
            "unacknowledged": self.get_unacknowledged_count(),
            "webhooks_configured": len(self._webhooks),
            "min_severity": self._min_severity.name,
        }

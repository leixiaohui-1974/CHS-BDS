"""
Alert and Notification System
==============================

Monitoring alert system with multiple notification channels.
Supports threshold-based alerts, anomaly detection, and notifications.
"""

import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from enum import Enum
from pathlib import Path

from .logger import get_logger
from .exceptions import AlertError


logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class Alert:
    """
    Alert object containing alert information.
    """

    def __init__(
        self,
        alert_id: str,
        title: str,
        message: str,
        level: AlertLevel,
        module: str,
        data: Optional[Dict] = None
    ):
        """
        Initialize alert.

        Args:
            alert_id: Unique alert identifier.
            title: Alert title.
            message: Alert message.
            level: Alert severity level.
            module: Source module name.
            data: Additional alert data.
        """
        self.alert_id = alert_id
        self.title = title
        self.message = message
        self.level = level
        self.module = module
        self.data = data or {}
        self.timestamp = datetime.now()
        self.status = AlertStatus.ACTIVE

    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'message': self.message,
            'level': self.level.value,
            'module': self.module,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value
        }

    def __repr__(self) -> str:
        return f"Alert({self.level.value.upper()}: {self.title})"


class AlertRule:
    """
    Rule for triggering alerts based on conditions.
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        condition: Callable,
        level: AlertLevel,
        message_template: str
    ):
        """
        Initialize alert rule.

        Args:
            rule_id: Unique rule identifier.
            name: Rule name.
            condition: Function that returns True when alert should trigger.
            level: Alert level for this rule.
            message_template: Message template (can include {placeholders}).
        """
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.level = level
        self.message_template = message_template
        self.enabled = True

    def check(self, data: Dict) -> Optional[Alert]:
        """
        Check if alert should be triggered.

        Args:
            data: Data to check against condition.

        Returns:
            Alert if condition met, None otherwise.
        """
        if not self.enabled:
            return None

        try:
            if self.condition(data):
                # Generate alert
                message = self.message_template.format(**data)
                alert = Alert(
                    alert_id=f"{self.rule_id}_{datetime.now().timestamp()}",
                    title=self.name,
                    message=message,
                    level=self.level,
                    module=data.get('module', 'unknown'),
                    data=data
                )
                return alert
        except Exception as e:
            logger.error(f"Error checking rule '{self.name}': {e}")

        return None


class NotificationChannel:
    """
    Base class for notification channels.
    """

    def send(self, alert: Alert) -> bool:
        """
        Send alert notification.

        Args:
            alert: Alert to send.

        Returns:
            True if successful, False otherwise.
        """
        raise NotImplementedError


class ConsoleNotifier(NotificationChannel):
    """Console notification channel."""

    def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        level_icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨"
        }

        icon = level_icons.get(alert.level, "📢")
        print(f"\n{icon} {alert.level.value.upper()} ALERT")
        print(f"Module: {alert.module}")
        print(f"Title: {alert.title}")
        print(f"Message: {alert.message}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)

        return True


class FileNotifier(NotificationChannel):
    """File-based notification channel."""

    def __init__(self, output_dir: str = './alerts'):
        """
        Initialize file notifier.

        Args:
            output_dir: Directory for alert files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> bool:
        """Save alert to file."""
        try:
            filename = f"{alert.alert_id}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(alert.to_dict(), f, indent=2)

            logger.debug(f"Alert saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
            return False


class EmailNotifier(NotificationChannel):
    """
    Email notification channel (placeholder).

    Note: Requires email configuration and SMTP setup.
    """

    def __init__(self, smtp_config: Optional[Dict] = None):
        """
        Initialize email notifier.

        Args:
            smtp_config: SMTP configuration dictionary.
        """
        self.smtp_config = smtp_config or {}
        logger.warning("Email notifications not implemented - requires SMTP configuration")

    def send(self, alert: Alert) -> bool:
        """Send email notification."""
        # Placeholder for actual email implementation
        logger.info(f"Would send email for alert: {alert.title}")
        return False


class AlertManager:
    """
    Central alert management system.

    Handles alert rules, notifications, and alert history.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.rules: Dict[str, AlertRule] = {}
        self.channels: List[NotificationChannel] = []
        self.alert_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}

        logger.info("AlertManager initialized")

    def add_rule(self, rule: AlertRule):
        """
        Add alert rule.

        Args:
            rule: Alert rule to add.
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str):
        """Remove alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")

    def add_channel(self, channel: NotificationChannel):
        """
        Add notification channel.

        Args:
            channel: Notification channel to add.
        """
        self.channels.append(channel)
        logger.info(f"Added notification channel: {type(channel).__name__}")

    def check_all_rules(self, data: Dict) -> List[Alert]:
        """
        Check all rules against data.

        Args:
            data: Data to check.

        Returns:
            List of triggered alerts.
        """
        triggered_alerts = []

        for rule in self.rules.values():
            alert = rule.check(data)
            if alert:
                triggered_alerts.append(alert)
                self._handle_alert(alert)

        return triggered_alerts

    def _handle_alert(self, alert: Alert):
        """Handle a triggered alert."""
        # Add to history
        self.alert_history.append(alert)

        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert

        # Send notifications
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {type(channel).__name__}: {e}")

        logger.info(f"Alert triggered: {alert.title} ({alert.level.value})")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            logger.info(f"Alert {alert_id} acknowledged")

    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.status = AlertStatus.RESOLVED
            logger.info(f"Alert {alert_id} resolved")

    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """
        Get active alerts.

        Args:
            level: Filter by alert level.

        Returns:
            List of active alerts.
        """
        alerts = list(self.active_alerts.values())
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts

    def get_alert_summary(self) -> Dict:
        """Get summary of alerts."""
        return {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts.values()
                                   if a.level == AlertLevel.CRITICAL]),
            'warning_alerts': len([a for a in self.active_alerts.values()
                                  if a.level == AlertLevel.WARNING]),
            'info_alerts': len([a for a in self.active_alerts.values()
                               if a.level == AlertLevel.INFO])
        }


# Predefined alert rules for GNSS monitoring
class GNSSAlertRules:
    """Predefined alert rules for GNSS systems."""

    @staticmethod
    def create_deformation_rule() -> AlertRule:
        """Alert rule for large deformation."""
        return AlertRule(
            rule_id="deformation_critical",
            name="Critical Deformation Detected",
            condition=lambda data: data.get('displacement', 0) > 0.05,  # 50mm
            level=AlertLevel.CRITICAL,
            message_template="Displacement of {displacement:.3f}m exceeds critical threshold of 0.05m"
        )

    @staticmethod
    def create_pwv_rule() -> AlertRule:
        """Alert rule for high PWV."""
        return AlertRule(
            rule_id="pwv_high",
            name="High PWV Detected",
            condition=lambda data: data.get('pwv', 0) > 40,  # mm
            level=AlertLevel.WARNING,
            message_template="PWV of {pwv:.1f}mm indicates high atmospheric moisture"
        )

    @staticmethod
    def create_quality_rule() -> AlertRule:
        """Alert rule for poor data quality."""
        return AlertRule(
            rule_id="quality_poor",
            name="Poor Data Quality",
            condition=lambda data: data.get('quality_score', 100) < 60,
            level=AlertLevel.WARNING,
            message_template="Data quality score of {quality_score:.1f} is below acceptable threshold"
        )


if __name__ == '__main__':
    # Test alert system
    print("Testing Alert and Notification System...")
    print("-" * 70)

    # Create alert manager
    manager = AlertManager()

    # Add notification channels
    manager.add_channel(ConsoleNotifier())
    manager.add_channel(FileNotifier(output_dir='./output/alerts'))

    # Add rules
    manager.add_rule(GNSSAlertRules.create_deformation_rule())
    manager.add_rule(GNSSAlertRules.create_pwv_rule())
    manager.add_rule(GNSSAlertRules.create_quality_rule())

    print("\n1. Testing deformation alert...")
    data1 = {'module': 'deformation', 'displacement': 0.06}
    alerts = manager.check_all_rules(data1)
    print(f"   Triggered {len(alerts)} alert(s)")

    print("\n2. Testing PWV alert...")
    data2 = {'module': 'pwv', 'pwv': 45}
    alerts = manager.check_all_rules(data2)
    print(f"   Triggered {len(alerts)} alert(s)")

    print("\n3. Testing quality alert...")
    data3 = {'module': 'quality', 'quality_score': 55}
    alerts = manager.check_all_rules(data3)
    print(f"   Triggered {len(alerts)} alert(s)")

    print("\n4. Alert summary:")
    summary = manager.get_alert_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ Alert system test completed!")

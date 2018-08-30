"""
The engine that records alerts and pushes notifications.
"""

class AlertsBoard():
    """Collects alerts."""

    def __init__(self):
        self.clear()

    def clear(self):
        """Resets the alerts board."""
        self.active = set()

    def add(self, new_alerts):
        """Add new alerts."""
        self.active.update(new_alerts)

    def remove(self, expired_alerts):
        """Remove expired alerts."""
        self.active.difference_update(expired_alerts)

    def save(self):
        """Saves the alerts board in the database."""
        print("Not yet implemented.")

    def update(self, study_list):
        """Updates the entire alerts board based on a study list."""
        new_study = study_list[-1]
        alerts_on = new_study.alerts - self.active
        alerts_off = self.active - new_study.alerts
        self.add(alerts_on)
        self.remove(alerts_off)
        print(
            "New Alerts: {0}\n"
            "Expired Alerts: {1}".format(alerts_on, alerts_off)
        )

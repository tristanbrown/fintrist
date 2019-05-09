"""Notifications"""

class Notification():
    def __init__(self, actions, **kwargs):
        for action in actions:
            getattr(self, action)(**kwargs)

    def printhead(self, study, alerts, **kwargs):
        """Print the head of the study's data."""
        print(study.data.head(5))

    def log(self, study, alerts, **kwargs):
        """Create a log entry for the alert."""
        print("'log' not implemented")

    def email(self, study, alerts, **kwargs):
        """Send an email about the alert."""
        print("'email' not implemented")

    def sms(self, study, alerts, **kwargs):
        """Send an sms text about the alert."""
        print("'sms' not implemented")

    def trade(self, study, alerts, **kwargs):
        """Initiate a trade based on the alert."""
        print("'trade' not implemented")

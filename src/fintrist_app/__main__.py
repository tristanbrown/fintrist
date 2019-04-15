"""Main module for the Flask app for Fintrist."""
from fintrist.scheduling import scheduler
from fintrist_app import app

def run():
    """Main method to run the app."""
    scheduler.start()
    try:
        app.run(debug=True)
    finally:
        scheduler.shutdown(wait=False)

if __name__ == "__main__":
    run()

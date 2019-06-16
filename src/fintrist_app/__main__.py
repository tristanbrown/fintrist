"""Main module for running the Flask app for Fintrist."""
from fintrist.scheduling import scheduler
from fintrist_app import app

def run():
    """Main method to run the app."""
    scheduler.start()
    app.run(debug=True)

if __name__ == "__main__":
    run()

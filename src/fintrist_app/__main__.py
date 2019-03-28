"""Main module for the Flask app for Fintrist."""
from fintrist_app import app

def run():
    """Main method to run the app."""
    app.run(debug=True)

if __name__ == "__main__":
    run()

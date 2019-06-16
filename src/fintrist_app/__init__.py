"""Set up Flask app"""
from flask import Flask, render_template

import fintrist
from fintrist_app.settings import Config
from fintrist_app.version2.views import dash_blueprint
from fintrist_app.studies.views import studies_blueprint

# Set up Flask
app = Flask(__name__)
app.config.from_object(Config())

app.register_blueprint(studies_blueprint, url_prefix="/studies")
app.register_blueprint(dash_blueprint, url_prefix="/version2")

@app.route("/")
def index():
    return render_template('home.html')

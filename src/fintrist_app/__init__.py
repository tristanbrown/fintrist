import os
from flask import Flask, render_template

import fintrist
from fintrist_app.settings import Config
from fintrist_app.streams.views import streams_blueprint
from fintrist_app.studies.views import studies_blueprint

# Set up Flask
app = Flask(__name__)
app.config.from_object(Config())

app.register_blueprint(streams_blueprint, url_prefix="/streams")
app.register_blueprint(studies_blueprint, url_prefix='/studies')

@app.route("/")
def index():
    return render_template('home.html')

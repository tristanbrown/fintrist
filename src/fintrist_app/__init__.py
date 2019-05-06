import os
from flask import Flask, render_template
from flask_apscheduler import APScheduler

import fintrist
from fintrist_app.settings import Config
from fintrist_app.streams.views import streams_blueprint
from fintrist_app.studies.views import studies_blueprint

# Set up Flask
app = Flask(__name__)
app.config.from_object(Config())

app.register_blueprint(streams_blueprint, url_prefix="/streams")
app.register_blueprint(studies_blueprint, url_prefix='/studies')

# Set up Scheduler
scheduler = APScheduler()
scheduler.init_app(app)

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/appdata")
def appdata():
    return app.config['APPDATA']

@app.route("/streamslist")
def streamslist():
    return ', '.join([stream.name for stream in fintrist.Stream.objects])

@app.route("/<name>")
def hello_name(name):
    return "Hello "+ name

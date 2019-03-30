import os
from flask import Flask, render_template

import fintrist
from fintrist_app import settings
from fintrist_app.streams.views import streams_blueprint
from fintrist_app.studies.views import studies_blueprint

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

app.register_blueprint(streams_blueprint,url_prefix="/streams")
app.register_blueprint(studies_blueprint,url_prefix='/studies')

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/appdata")
def appdata():
    return settings.APPDATA

@app.route("/streamslist")
def streamslist():
    return ', '.join([stream.name for stream in fintrist.Stream.objects])

@app.route("/<name>")
def hello_name(name):
    return "Hello "+ name

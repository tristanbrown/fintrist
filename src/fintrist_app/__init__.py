"""Set up Flask app"""
from flask import Flask, render_template, redirect
import dash
import dash_html_components as html

import fintrist
from fintrist_app.settings import Config
# from fintrist_app.studies.views import studies_blueprint
# from fintrist_app.version2.views import dash_blueprint

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Set up Flask
app = dash.Dash(__name__)
server = app.server
server.config.from_object(Config())

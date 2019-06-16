"""HTTP views related to the new Dash version"""
import dash
import dash_html_components as html
from flask import Blueprint, redirect
from fintrist_app import app_dash

# dash_blueprint = Blueprint('version2',
#                               __name__,
#                               template_folder='templates/version2')

# app_dash = dash.Dash(__name__, server=dash_blueprint, url_base_pathname='/pathname/')

# @dash_blueprint.route('/plotly_dashboard') 
# def render_dashboard():
#     return redirect('/pathname/')

app_dash.layout = html.Div("My Dash app")

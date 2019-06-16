"""URL routing for the Dash/Flask app"""
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask import render_template

from fintrist_app import app, server
from fintrist_app.version2 import app1, app2
from fintrist_app.studies.views import studies_blueprint


server.register_blueprint(studies_blueprint, url_prefix="/studies")

@server.route("/")
def index():
    return render_template('home.html')

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/app1':
        return app1.layout
    elif pathname == '/apps/app2':
        return app2.layout
    else:
        return '404'

def launch(debug=False):
    app.run_server(debug)

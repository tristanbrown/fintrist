"""HTTP views related to the new Dash version"""

from flask import Blueprint

dash_blueprint = Blueprint('version2',
                           __name__,
                           template_folder='templates/version2')

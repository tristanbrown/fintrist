from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField

class AddForm(FlaskForm):

    name = StringField('Name of Stream:')
    refresh = IntegerField('Refresh Interval (s):')
    submit = SubmitField('Add Stream')

class DelForm(FlaskForm):

    name = StringField('Name of Stream to Remove:')
    submit = SubmitField('Remove Stream')

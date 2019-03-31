from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, SelectMultipleField

from fintrist import Stream

class AddForm(FlaskForm):

    name = StringField('Name of Stream:')
    refresh = IntegerField('Refresh Interval (s):')
    submit = SubmitField('Add Stream')

class DelForm(FlaskForm):

    name = StringField('Name of Stream to Remove:')
    submit = SubmitField('Remove Stream')

class SelForm(FlaskForm):
    choices = []
    selections = SelectMultipleField('Available Streams', choices=choices)
    submit = SubmitField('Choose Streams')

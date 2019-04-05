from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, SelectField

from fintrist import Stream

class AddForm(FlaskForm):

    name = StringField('Name of Stream:')
    refresh = IntegerField('Refresh Interval (s):')
    submit = SubmitField('Add Stream')

class DelForm(FlaskForm):

    name = StringField('Name of Stream to Remove:')
    submit = SubmitField('Remove Stream')

class StreamSelForm(FlaskForm):
    choices = []
    selections = SelectField('Available Streams', choices=choices)
    choose = SubmitField('Choose Streams')
    edit = SubmitField('Edit Streams')
    delete = SubmitField('Delete Streams')

class StudySelForm(FlaskForm):
    choices = []
    selections = SelectField('Associated Studies', choices=choices)

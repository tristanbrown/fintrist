from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField

class AddForm(FlaskForm):

    name = StringField('Name of Study:')
    analysis = StringField('Name of Process:')
    submit = SubmitField('Add Study')

class DelForm(FlaskForm):

    name = StringField('Name of Study to Remove:')
    submit = SubmitField('Remove Study')

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

def sel_form(label):
    """Generate a selection form."""
    class SelForm(FlaskForm):
        default_choices = []
        selections = SelectField(f'Available {label}', choices=default_choices)
        choose = SubmitField(f'Choose {label}')
        edit = SubmitField(f'Edit {label}')
        delete = SubmitField(f'Delete {label}')
    return SelForm()

def subsel_form(label):
    class SelForm(FlaskForm):
        choices = []
        selections = SelectField(f'Associated {label}', choices=choices)
    return SelForm()

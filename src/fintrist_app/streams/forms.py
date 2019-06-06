"""Forms allowing CRUD operations for Streams"""

from flask_wtf import FlaskForm
from wtforms import (StringField, IntegerField, SubmitField, SelectField,
                     SelectMultipleField, validators)

class AddForm(FlaskForm):
    name = StringField('Name of Stream:')
    refresh = IntegerField('Refresh Interval (sec):', validators=[validators.Optional(),])
    submit = SubmitField('Save')

def sel_form(label):
    """Generate a selection form."""
    class SelForm(FlaskForm):
        default_choices = []
        selections = SelectField(f'Available {label}', choices=default_choices)
        choose = SubmitField(f'Choose {label}')
        moveright = SubmitField('>>')
        moveleft = SubmitField('<<')
        edit = SubmitField(f'Edit {label}')
        delete = SubmitField(f'Delete {label}')
        clear = SubmitField('Clear Selections')
        activate = SubmitField(f'Activate {label}')
        deactivate = SubmitField(f'Deactivate {label}')
    return SelForm(prefix=label)

def subsel_form(label):
    class SubSelForm(FlaskForm):
        choices = []
        selections = SelectField(f'Associated {label}', choices=choices)
        movefirst = SubmitField('First')
        moveup = SubmitField('^')
        movedown = SubmitField('v')
        movelast = SubmitField('Last')
        remove = SubmitField(f'Remove {label}')
    return SubSelForm(prefix=label + '_sub')

def multisel_form(label):
    """Generate a selection form."""
    class MultiSelForm(FlaskForm):
        default_choices = []
        selections = SelectMultipleField(f'Available {label}', choices=default_choices)
        choose = SubmitField(f'Choose {label}')
        moveright = SubmitField('>>')
        moveleft = SubmitField('<<')
        edit = SubmitField(f'Edit {label}')
        delete = SubmitField(f'Delete {label}')
        clear = SubmitField('Clear Selections')
        activate = SubmitField(f'Activate {label}')
        deactivate = SubmitField(f'Deactivate {label}')
        runonce = SubmitField(f'Run Once')
    return MultiSelForm(prefix=label + '_multi')

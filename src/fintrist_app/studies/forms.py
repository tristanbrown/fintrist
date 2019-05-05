from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, SelectField

class AddForm(FlaskForm):

    name = StringField('Name of Study:')
    analysis = StringField('Name of Process:')
    submit = SubmitField('Add Study')

class DelForm(FlaskForm):

    name = StringField('Name of Study to Remove:')
    submit = SubmitField('Remove Study')

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
    return SelForm()

def subsel_form(label):
    class SubSelForm(FlaskForm):
        choices = []
        selections = SelectField(f'Associated {label}', choices=choices)
        movefirst = SubmitField('First')
        moveup = SubmitField('^')
        movedown = SubmitField('v')
        movelast = SubmitField('Last')
        remove = SubmitField(f'Remove {label}')
    return SubSelForm()

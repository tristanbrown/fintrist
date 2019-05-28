from flask_wtf import FlaskForm
from wtforms import (StringField, IntegerField, SubmitField, SelectField,
    SelectMultipleField, BooleanField,)

from fintrist import Study, Trigger
from fintrist_app import util

def add_form(label):
    """Generate a selection form."""
    class AddForm(FlaskForm):

        entry = StringField(label)
        submit = SubmitField('Save')
    return AddForm(prefix=label)

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

def mini_sel_form(label, def_choices):
    """Generate a selection form."""
    class MiniSelForm(FlaskForm):
        selections = SelectField(label, choices=def_choices)
    return MiniSelForm(prefix=label)

def multisel_form(label):
    """Generate a selection form."""
    class MultiSelForm(FlaskForm):
        default_choices = []
        selections = SelectMultipleField(f'Associated {label}', choices=default_choices)
        choose = SubmitField(f'Choose {label}')
        moveright = SubmitField('>>')
        moveleft = SubmitField('<<')
        edit = SubmitField(f'Edit {label}')
        delete = SubmitField(f'Delete {label}')
        clear = SubmitField('Clear Selections')
        activate = SubmitField(f'Activate {label}')
        deactivate = SubmitField(f'Deactivate {label}')
        runonce = SubmitField(f'Run Once')
    return MultiSelForm(prefix=label)

def trigger_build(sel_trigger):
    """Generate a boolean form."""
    alerttypes = simplechoices(Trigger.alert_types)
    conds = simplechoices(Trigger.match_if)

    if sel_trigger:
        def_type = sel_trigger.on
        def_cond = sel_trigger.condition
        def_actions = sel_trigger.actions
    else:
        def_type = def_cond = None
        def_actions = []

    class TriggerForm(FlaskForm):
        actions = Trigger.action_choices
        matchtext = StringField('Trigger match text')
        submit = SubmitField('Save')
        alerttype = SelectField('Alert type', choices=alerttypes, default=def_type)
        condition = SelectField('Match condition', choices=conds, default=def_cond)

    for action in TriggerForm.actions:
        checked = action in def_actions
        setattr(TriggerForm, action, BooleanField(label=action, default=checked))
    return TriggerForm()

def inputs_build(parents, params):
    """Generate inputs entry fields."""
    class InputsForm(FlaskForm):
        parent_keys = parents
        param_keys = params
        submit = SubmitField('Save')
    db_objects = util.get_choices(Study.objects())
    for key in parents:
        setattr(InputsForm, key, SelectField(key, choices=db_objects))
    for key in params:
        setattr(InputsForm, key, StringField(key))
    return InputsForm()

def simplechoices(iterable):
    """Convert an iterable into a list of duplicated tuples."""
    return zip(iterable, iterable)

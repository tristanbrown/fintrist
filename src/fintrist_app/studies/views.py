from flask import Blueprint, render_template, redirect, url_for, session
from fintrist import Study, Process, Trigger
from fintrist_app.studies.forms import (
    sel_form, mini_sel_form, multisel_form, add_form, trigger_build)
from fintrist_app import util

studies_blueprint = Blueprint('studies',
                              __name__,
                              template_folder='templates/studies')

@studies_blueprint.route('/edit', methods=['GET', 'POST'])
def edit():
    """Edit Studies.

    Allows modification of Studies.
    """
    # Set up All Studies selection list
    studyform = sel_form('Studies')
    db_objects = util.get_choices(Study.objects())
    studyform.selections.choices = db_objects

    # Set up the Study to edit and associated inputs
    editstudy_id = session.get('editstudy')
    inputsform = multisel_form('Inputs')
    alltriggers = sel_form('Triggers')

    if editstudy_id:
        editstudy = Study.objects(id=editstudy_id).get()
        studyname = editstudy.name
        procname = editstudy.process.name
        parents = editstudy.all_parents
        params = editstudy.all_params
        inputsform.selections.choices = inputchoices(parents, params)
        parentforms = make_selforms(parents)
        paramforms = make_entryforms(params)
        alltriggers.selections.choices = triggerchoices(editstudy.triggers)
        sel_trigger = editstudy[session.get('sel_trigger')]
    else:
        editstudy = None
        studyname = ''
        procname = ''
        parentforms = {}
        paramforms = {}
        sel_trigger = None

    # Set up All Processes selection list
    procform = sel_form('Processes')
    procform.selections.choices = util.get_choices(Process.objects())

    # Set up Add/Edit
    addform = add_form('Study Name')
    saveinputs = add_form('Inputs')

    # Trigger components
    triggerform = trigger_build(sel_trigger)

    # Clear the selections
    if studyform.clear.data:
        session['editstudy'] = None
        return redirect(url_for('studies.edit'))

    # Submit buttons for Studies selection list
    if studyform.validate_on_submit():
        # Select a Study
        selections = studyform.selections.data
        editstudy = Study.objects(id=selections).get()
        # Edit Study
        if studyform.edit.data:
            session['editstudy'] = str(editstudy.id)
        # Delete Study
        elif studyform.delete.data:
            editstudy.delete()
            session['editstudy'] = None
        return redirect(url_for('studies.edit'))

    # Update Study or create new
    if addform.validate_on_submit() and addform.submit.data:
        name = addform.entry.data
        newproc = Process.objects(pk=procform.selections.data)
        if editstudy:
            if name:
                editstudy.rename(name)
            if newproc:
                editstudy.set_process(newproc.get().name)
        else:
            new_study = Study(name=name)
            new_study.set_process(newproc.get().name)
            new_study.reload()
            session['editstudy'] = str(new_study.id)  #pylint: disable=no-member
        return redirect(url_for('studies.edit'))

    # Delete Study-associated Inputs
    if editstudy and inputsform.validate_on_submit():
        selections = inputsform.selections.data
        editstudy.remove_inputs(selections)
        return redirect(url_for('studies.edit'))

    # Save Study-associated Inputs
    if editstudy and saveinputs.is_submitted():
        newparents = {key: form.selections.data
                      for key, form in parentforms.items() if form.selections.data != 'None'}
        editstudy.add_parents(newparents)
        newparams = {key: form.entry.data
                     for key, form in paramforms.items() if form.entry.data}
        editstudy.add_params(newparams)
        return redirect(url_for('studies.edit'))

    # Submit buttons for Triggers selection list
    if sel_trigger and alltriggers.validate_on_submit():
        selections = alltriggers.selections.data
        if alltriggers.delete.data:
            # TODO: Somehow id and remove the study
            pass
        elif alltriggers.choose.data:
            # TODO: Somehow set the selected trigger to the session data
            pass
        elif alltriggers.clear.data:
            session['sel_trigger'] = None
        return redirect(url_for('studies.edit'))

    # Save Study-associated Triggers
    if triggerform.validate_on_submit():
        # TODO: Add or update Triggers
        pass

    return render_template(
        'edit_study.html',
        studyname=studyname,
        studyform=studyform,
        procname=procname,
        addform=addform,
        procsel=procform,
        inputsform=inputsform,
        parentforms=parentforms,
        paramforms=paramforms,
        saveinputs=saveinputs,
        alltriggers=alltriggers,
        triggerform=triggerform,
        )

def inputchoices(parents, params):
    """Convert the parents and params into selection lists."""
    parentnames = [(key, parent.name) if parent else (key, None) for key, parent in parents.items()]
    return [(key, f"{key}: {val}") for key, val in parentnames + list(params.items())]

def simplechoices(iterable):
    """Convert an iterable into a list of duplicated tuples."""
    return zip(iterable, iterable)

def triggerchoices(triggers):
    """Convert the triggers into selection lists."""
    triggervals = [(trigger.matchtext, trigger.condition, trigger.on) for trigger in triggers]
    return [(on + cond + match, f"{match} ({cond}, {on})") for match, cond, on in triggervals]

def make_selforms(parents):
    """Take a list or dict of object references and return a dict of forms."""
    db_objects = util.get_choices(Study.objects())
    return {key: mini_sel_form(key, db_objects) for key in parents}

def make_entryforms(params):
    """Take a list or dict of parameter names and return a dict of forms."""
    return {key: add_form(key) for key in params}

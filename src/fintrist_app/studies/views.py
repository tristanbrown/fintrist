"""HTTP views related to Studies"""

from flask import Blueprint, render_template, redirect, url_for, session
from fintrist import Study, Process
from fintrist.scheduling import scheduler
from fintrist_app.studies.forms import (
    sel_form, multisel_form, AddForm, trigger_build, inputs_build)
from fintrist_app import util

studies_blueprint = Blueprint('studies',
                              __name__,
                              template_folder='templates/studies')

@studies_blueprint.route('/manage', methods=['GET', 'POST'])
def manage():
    """Manage the running of Studies.
    """
    active_jobs = [job.id for job in scheduler.get_jobs()]
    # Set up Inactive studies selection list
    inactiveform = multisel_form('Studies')
    inactive_studies = Study.objects(id__not__in=active_jobs)
    inactive_choices = util.get_choices(inactive_studies())
    inactiveform.selections.choices = inactive_choices
    # Set up Active studies selection list
    activeform = multisel_form('Studies')
    active_studies = Study.objects(id__in=active_jobs)
    active_choices = util.get_choices(active_studies())
    activeform.selections.choices = active_choices
    # Activate studies
    if inactiveform.validate_on_submit() and inactiveform.moveright.data:
        selections = inactiveform.selections.data
        editstudies = Study.objects(id__in=selections)
        for study in editstudies:
            study.activate()
        return redirect(url_for('studies.manage'))
    # Deactivate studies
    elif activeform.validate_on_submit() and activeform.moveleft.data:
        selections = activeform.selections.data
        editstudies = Study.objects(id__in=selections)
        for study in editstudies:
            study.deactivate()
        return redirect(url_for('studies.manage'))
    # Run selected studies once
    elif inactiveform.validate_on_submit() and inactiveform.runonce.data:
        selections = inactiveform.selections.data
        editstudies = Study.objects(id__in=selections)
        for study in editstudies:
            study.run_study_once()
    elif activeform.validate_on_submit() and activeform.runonce.data:
        selections = activeform.selections.data
        editstudies = Study.objects(id__in=selections)
        for study in editstudies:
            study.run_study_once()

    return render_template(
        'manage_study.html',
        activeform=activeform,
        inactiveform=inactiveform,
        )

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
    trig_id = session.get('sel_trigger')

    try:
        editstudy = Study.objects(id=editstudy_id).get()
        studyname = editstudy.name
        procname = editstudy.process.name
        seltimevalid = int(editstudy.valid_age)
        parents = editstudy.all_parents
        params = editstudy.all_params
        inputsform.selections.choices = inputchoices(parents, params)
        alltriggers.selections.choices = simplechoices(editstudy.triggers.keys())
        sel_trigger = editstudy.get_trigger(trig_id)
    except Exception as ex:  #pylint: disable=broad-except
        editstudy = None
        studyname = ''
        procname = ''
        seltimevalid = 0
        parents = {}
        params = {}
        sel_trigger = None
        print(ex)

    # Set up All Processes selection list
    procform = sel_form('Processes')
    procform.selections.choices = util.get_choices(Process.objects())

    # Set up Add/Edit
    addform = AddForm(prefix='add')

    # Trigger components
    triggerform = trigger_build(sel_trigger)
    parentparams = inputs_build(parents, params)

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
        name = addform.name.data
        newage = addform.timevalid.data
        newproc = Process.objects(pk=procform.selections.data)
        if editstudy:
            if name:
                editstudy.rename(name)
            if isinstance(newage, int):
                editstudy.update_valid_age(newage)
            if newproc:
                editstudy.set_process(newproc.get().name)
        else:
            new_study = Study(name=name, valid_age=newage)
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
    if editstudy and parentparams.validate_on_submit():
        newparents = {key: parentparams[key].data
                      for key in parentparams.parent_keys if parentparams[key].data != 'None'}
        editstudy.add_parents(newparents)
        newparams = {key: parentparams[key].data
                     for key in parentparams.param_keys if parentparams[key].data}
        editstudy.add_params(newparams)
        return redirect(url_for('studies.edit'))

    # Submit buttons for Triggers selection list
    if alltriggers.validate_on_submit():
        selection = alltriggers.selections.data
        if alltriggers.delete.data:
            editstudy.del_trigger(selection)
        elif alltriggers.choose.data:
            session['sel_trigger'] = selection
        elif alltriggers.clear.data:
            session['sel_trigger'] = None
        return redirect(url_for('studies.edit'))

    # Save Study-associated Triggers
    if editstudy and triggerform.validate_on_submit():
        editstudy.add_trigger(
            triggerform.matchtext.data,
            on=triggerform.alerttype.data,
            condition=triggerform.condition.data,
            actions=[action for action in triggerform.actions if triggerform[action].data]
            )
        return redirect(url_for('studies.edit'))

    return render_template(
        'edit_study.html',
        studyname=studyname,
        studyform=studyform,
        procname=procname,
        addform=addform,
        timevalid=seltimevalid,
        procsel=procform,
        inputsform=inputsform,
        parentparams=parentparams,
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

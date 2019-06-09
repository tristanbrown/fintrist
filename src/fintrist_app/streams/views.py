"""HTTP views related to Streams"""

from flask import Blueprint, render_template, redirect, url_for, session
from fintrist import Stream, Study
from fintrist.scheduling import scheduler
from fintrist_app.streams.forms import AddForm, sel_form, subsel_form, multisel_form
from fintrist_app import util

streams_blueprint = Blueprint('streams',
                              __name__,
                              template_folder='templates/streams')

@streams_blueprint.route('/manage', methods=['GET','POST'])
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
        return redirect(url_for('streams.manage'))
    # Deactivate studies
    elif activeform.validate_on_submit() and activeform.moveleft.data:
        selections = activeform.selections.data
        editstudies = Study.objects(id__in=selections)
        for study in editstudies:
            study.deactivate()
        return redirect(url_for('streams.manage'))
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
        'manage_stream.html',
        activeform=activeform,
        inactiveform=inactiveform,
        )

@streams_blueprint.route('/edit', methods=['GET','POST'])
def edit():
    """Edit Streams.

    Allows modification of the associated Study list, as well as the refresh
    interval.
    """
    # Set up All Streams selection list
    streamform = sel_form('Streams')
    db_objects = util.get_choices(Stream.objects())
    streamform.selections.choices = db_objects
    # Set up Stream-associated Studies list
    assocform = subsel_form('Studies')
    # The Stream to edit
    editstream_id = session.get('editstream')
    try:
        editstream = Stream.objects(id=editstream_id).get()
        streamname = editstream.name
        selrefresh = int(editstream.refresh)
        new_objects = util.get_choices(editstream.studies)
        assocform.selections.choices = new_objects
    except Exception as ex:  #pylint: disable=broad-except
        editstream = None
        streamname = ''
        selrefresh = None
        print(ex)

    # Set up All Studies selection list
    allform = sel_form('Studies')
    allform.selections.choices = util.get_choices(Study.objects())

    # Set up refresh interval edit
    addform = AddForm()

    # Clear the selections
    if streamform.clear.data:
        session['editstream'] = None
        return redirect(url_for('streams.edit'))

    # Submit buttons for Streams selection list
    if streamform.validate_on_submit():
        # Select a Stream
        selections = streamform.selections.data
        editstream = Stream.objects(id=selections).get()
        # Edit Stream
        if streamform.edit.data:
            session['editstream'] = str(editstream.id)
        # Delete Stream
        elif streamform.delete.data:
            editstream.delete()
            session['editstream'] = None
        return redirect(url_for('streams.edit'))

    # Submit buttons for All Studies selection list
    if allform.validate_on_submit():
        selections = allform.selections.data
        selected_study = Study.objects(id=selections).get()
        if editstream and allform.moveleft.data:
            editstream.add_study(selected_study)
            return redirect(url_for('streams.edit'))
        elif allform.edit.data:
            session['editstudy'] = str(selected_study.id)
            return redirect(url_for('studies.edit'))
    # Submit buttons for Stream-associated Studies
    if editstream and assocform.validate_on_submit():
        selections = assocform.selections.data
        selected_study = Study.objects(id=selections).get()
        if assocform.remove.data:
            editstream.remove_study(selected_study)
        elif assocform.moveup.data:
            editstream.move_study_earlier(selected_study)
        elif assocform.movedown.data:
            editstream.move_study_later(selected_study)
        elif assocform.movefirst.data:
            editstream.move_study_first(selected_study)
        elif assocform.movelast.data:
            editstream.move_study_last(selected_study)
        return redirect(url_for('streams.edit'))
    # Update refresh interval for the Stream
    if addform.validate_on_submit() and addform.submit.data:
        name = addform.name.data
        newrefresh = addform.refresh.data
        if editstream:
            if name:
                editstream.rename(name)
            if newrefresh:
                editstream.update_refresh(newrefresh)
        else:
            new_stream = Stream(name, newrefresh)
            new_stream.save()
        return redirect(url_for('streams.edit'))

    return render_template(
        'edit_stream.html',
        streamform=streamform,
        streamname=streamname,
        allsel=allform,
        assoc=assocform,
        refresh=selrefresh,
        addform=addform,
        )

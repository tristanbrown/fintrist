import time

from flask import Blueprint, render_template, redirect, url_for, session
from fintrist import Stream, Study
from fintrist.scheduling import scheduler
from fintrist_app.streams.forms import AddForm, DelForm, sel_form, subsel_form, multisel_form
from fintrist_app import util

streams_blueprint = Blueprint('streams',
                              __name__,
                              template_folder='templates/streams')

@streams_blueprint.route('/manage', methods=['GET','POST'])
def manage():
    """Manage the running of Streams.
    """
    active_jobs = [job.id for job in scheduler.get_jobs()]
    # Set up Inactive Streams selection list
    inactiveform = multisel_form('Streams')
    inactive_streams = Stream.objects(id__not__in=active_jobs)
    inactive_choices = util.get_choices(inactive_streams())
    inactiveform.selections.choices = inactive_choices
    # Set up Active Streams selection list
    activeform = multisel_form('Streams')
    active_streams = Stream.objects(id__in=active_jobs)
    active_choices = util.get_choices(active_streams())
    activeform.selections.choices = active_choices
    # Activate Streams
    if inactiveform.validate_on_submit() and inactiveform.moveright.data:
        selections = inactiveform.selections.data
        editstreams = Stream.objects(id__in=selections)
        for stream in editstreams:
            stream.activate()
        return redirect(url_for('streams.manage'))
    # Deactivate Streams
    elif activeform.validate_on_submit() and activeform.moveleft.data:
        selections = activeform.selections.data
        editstreams = Stream.objects(id__in=selections)
        for stream in editstreams:
            stream.deactivate()
        return redirect(url_for('streams.manage'))
    # Run selected Streams once
    elif inactiveform.validate_on_submit() and inactiveform.runonce.data:
        selections = inactiveform.selections.data
        editstreams = Stream.objects(id__in=selections)
        for stream in editstreams:
            stream.run_stream_once()
    elif activeform.validate_on_submit() and activeform.runonce.data:
        selections = activeform.selections.data
        editstreams = Stream.objects(id__in=selections)
        for stream in editstreams:
            stream.run_stream_once()

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
    # The Stream to edit
    editstream_id = session.get('editstream')
    if editstream_id:
        editstream = Stream.objects(id=editstream_id).get()
        streamname = editstream.name
        selrefresh = editstream.refresh
    else:
        editstream = None
        streamname = ''
        selrefresh = None
    # Set up All Studies selection list
    allform = sel_form('Studies')
    allform.selections.choices = util.get_choices(Study.objects())
    # Set up Stream-associated Studies list
    assocform = subsel_form('Studies')
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
        session['editstream'] = str(editstream.id)
        streamname = editstream.name
        # Edit Stream
        if streamform.edit.data:
            new_objects = util.get_choices(editstream.studies)
            assocform.selections.choices = new_objects
            selrefresh = int(editstream.refresh)
        # Delete Stream
        elif streamform.delete.data:
            editstream.delete()
            session['editstream'] = None
            return redirect(url_for('streams.edit'))

    # Submit buttons for All Studies selection list
    if editstream and allform.validate_on_submit():
        selections = allform.selections.data
        selected_study = Study.objects(id=selections).get()
        if allform.moveleft.data:
            editstream.add_study(selected_study)
        assocform.selections.choices = util.get_choices(editstream.studies)
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
        assocform.selections.choices = util.get_choices(editstream.studies)
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
            new_stream.reload()
            # session['editstream'] = str(new_stream.id)
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

import time

from flask import Blueprint, render_template, redirect, url_for, session
from fintrist import Stream, Study
from fintrist_app.streams.forms import AddForm, DelForm, sel_form, subsel_form

streams_blueprint = Blueprint('streams',
                              __name__,
                              template_folder='templates/streams')

@streams_blueprint.route('/add', methods=['GET', 'POST'])
def add():

    form = AddForm()
    if form.validate_on_submit():
        name = form.name.data
        refresh = form.refresh.data
        # Add new stream to database
        new_stream = Stream(name, refresh)
        new_stream.save()

        return redirect(url_for('streams.list'))
    return render_template('add_stream.html', form=form)

@streams_blueprint.route('/list')
def list():
    # Grab a list of studies from database.
    streams = Stream.objects()
    return render_template('list_streams.html', streams=streams)

@streams_blueprint.route('/select', methods=['GET','POST'])
def select():
    # Grab a selectable list of studies from database.
    selform = sel_form('Streams')
    selform2 = subsel_form('Studies')
    db_objects = get_choices(Stream.objects())
    selform.selections.choices = db_objects
    selrefresh = ''
    if selform.validate_on_submit():
        # Select a Stream
        selections = selform.selections.data
        selected_stream = Stream.objects(id=selections).get()
        # View associated studies and refresh interval
        if selform.choose.data:
            selrefresh = selected_stream.refresh
            new_objects = get_choices(selected_stream.studies)
            selform2.selections.choices = new_objects
        # Delete Stream
        elif selform.delete.data:
            selected_stream.delete()
        # Edit Stream
        elif selform.edit.data:
            session['editstream'] = str(selected_stream.id)
            return redirect(url_for('streams.edit'))
    # Add a new stream to the database
    addform = AddForm()
    if addform.validate_on_submit() and addform.submit.data:
        name = addform.name.data
        newrefresh = addform.refresh.data
        new_stream = Stream(name, newrefresh)
        new_stream.save()
        return redirect(url_for('streams.select'))
    return render_template(
        'select_streams.html', 
        selform=selform,
        selform2=selform2,
        addform=addform,
        refresh=selrefresh,
        )

@streams_blueprint.route('/edit', methods=['GET','POST'])
def edit():
    """Edit a specific stream.

    Allows modification of the associated Study list, as well as the refresh
    interval.
    """
    # The Stream to edit
    editstream = Stream.objects(id=session['editstream']).get()
    streamname = editstream.name
    # Set up All Studies selection list
    allform = sel_form('Studies')
    allform.selections.choices = get_choices(Study.objects())
    # Set up Stream-associated Studies list
    assocform = subsel_form('Studies')
    assocform.selections.choices = get_choices(editstream.studies)
    # Set up refresh interval edit
    refreshform = AddForm()
    # Submit buttons for All Studies selection list
    if allform.validate_on_submit():
        selections = allform.selections.data
        selected_study = Study.objects(id=selections).get()
        if allform.moveright.data:
            editstream.add_study(selected_study)
        assocform.selections.choices = get_choices(editstream.studies)
    # Submit buttons for Stream-associated Studies
    if assocform.validate_on_submit():
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
        assocform.selections.choices = get_choices(editstream.studies)
    # Update refresh interval for the Stream
    if refreshform.validate_on_submit() and refreshform.submit.data:
        newrefresh = refreshform.refresh.data
        print(newrefresh)
        editstream.update_refresh(newrefresh)
    return render_template(
        'edit_stream.html',
        streamname=streamname,
        allsel=allform,
        assoc=assocform,
        refresh=int(editstream.refresh),
        refreshform=refreshform,
        )

def get_choices(query):
    """Get a list of selection choices from a query object."""
    return [(str(item.id), item.name) for item in query]

@streams_blueprint.route('/delete', methods=['GET', 'POST'])
def delete():

    form = DelForm()

    if form.validate_on_submit():
        name = form.name.data
        stream = Stream.objects(name=name).get()
        stream.delete()

        return redirect(url_for('streams.list'))
    return render_template('delete_stream.html', form=form)

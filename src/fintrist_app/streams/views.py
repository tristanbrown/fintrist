from flask import Blueprint, render_template, redirect, url_for, session
from fintrist import Stream
from fintrist_app.streams.forms import AddForm, DelForm, StreamSelForm, StudySelForm

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
    selform = StreamSelForm()
    selform2 = StudySelForm()
    db_objects = [(str(stream.id), stream.name) for stream in Stream.objects()]
    selform.selections.choices = db_objects
    if selform.validate_on_submit():
        # Select a Stream and view associated Studies
        selections = selform.selections.data
        selected_stream = Stream.objects(id=selections).get()
        if selform.choose.data:
            new_objects = [(str(stream.id), stream.name) for stream in selected_stream.studies]
            selform2.selections.choices = new_objects
            selform.selections.data = selform2.selections.data = []
        # Delete Stream
        elif selform.delete.data:
            selected_stream.delete()
        # Edit Stream
        elif selform.edit.data:
            session['editstream'] = selected_stream
            return redirect(url_for('streams.edit'))
    # Add a new stream to the database
    addform = AddForm()
    if addform.validate_on_submit() and addform.submit.data:
        name = addform.name.data
        refresh = addform.refresh.data
        # Add new stream to database
        new_stream = Stream(name, refresh)
        new_stream.save()
        return redirect(url_for('streams.select'))
    return render_template('select_streams.html', selform=selform, selform2=selform2, addform=addform)

@streams_blueprint.route('/delete', methods=['GET', 'POST'])
def delete():

    form = DelForm()

    if form.validate_on_submit():
        name = form.name.data
        stream = Stream.objects(name=name).get()
        stream.delete()

        return redirect(url_for('streams.list'))
    return render_template('delete_stream.html', form=form)

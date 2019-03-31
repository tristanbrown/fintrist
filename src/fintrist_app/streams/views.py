from flask import Blueprint, render_template, redirect, url_for
from fintrist import Stream
from fintrist_app.streams.forms import AddForm, DelForm, SelForm

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
    form = SelForm()
    form2 = SelForm()
    db_objects = [(str(stream.id), stream.name) for stream in Stream.objects()]
    form.selections.choices = db_objects
    if form.validate_on_submit():
        form2 = SelForm()
        selections = form.selections.data
        new_objects = [(str(stream.id), stream.name) for stream in Stream.objects(id__in=selections)]
        form2.selections.choices = new_objects
        form.selections.data = form2.selections.data = []
    return render_template('select_streams.html', form=form, form2=form2)

@streams_blueprint.route('/delete', methods=['GET', 'POST'])
def delete():

    form = DelForm()

    if form.validate_on_submit():
        name = form.name.data
        stream = Stream.objects(name=name).get()
        stream.delete()

        return redirect(url_for('streams.list'))
    return render_template('delete_stream.html', form=form)

from flask import Blueprint, render_template, redirect, url_for
from fintrist import Stream
from fintrist_app.streams.forms import AddForm, DelForm

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
    return render_template('add_stream.html',form=form)

@streams_blueprint.route('/list')
def list():
    # Grab a list of studies from database.
    streams = Stream.objects()
    return render_template('list_streams.html', streams=streams)

@streams_blueprint.route('/delete', methods=['GET', 'POST'])
def delete():

    form = DelForm()

    if form.validate_on_submit():
        name = form.name.data
        stream = Stream.objects(name=name).get()
        stream.delete()

        return redirect(url_for('streams.list'))
    return render_template('delete_stream.html', form=form)

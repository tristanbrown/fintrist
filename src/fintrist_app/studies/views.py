from flask import Blueprint, render_template, redirect, url_for
from fintrist import Study
from fintrist_app.studies.forms import AddForm, DelForm

studies_blueprint = Blueprint('studies',
                              __name__,
                              template_folder='templates/studies')

@studies_blueprint.route('/add', methods=['GET', 'POST'])
def add():

    form = AddForm()
    if form.validate_on_submit():
        name = form.name.data
        process = form.analysis.data
        # Add new study to database
        new_study = Study(name, process)
        new_study.save()

        return redirect(url_for('studies.list'))
    return render_template('add.html',form=form)

@studies_blueprint.route('/list')
def list():
    # Grab a list of studies from database.
    studies = Study.objects()
    return render_template('list.html', studies=studies)

@studies_blueprint.route('/delete', methods=['GET', 'POST'])
def delete():

    form = DelForm()

    if form.validate_on_submit():
        name = form.name.data
        study = Study.objects(name=name).get()
        study.delete()

        return redirect(url_for('studies.list'))
    return render_template('delete.html', form=form)

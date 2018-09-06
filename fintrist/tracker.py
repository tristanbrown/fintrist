"""
The Tracking Engine that monitors the wall clock and any alerts, triggering
events such as data refresh.
"""
import os
import time
import pickle
import bson

from fintrist import settings
from fintrist.alerts import AlertsBoard
from fintrist.processes.scrapers.base import Scraper
from fintrist.models import AlertsBoard, Process, Stream, Study
from pymongo import MongoClient
from mongoengine import connect

class TrackingEngine():
    """Tracking engine that monitors and updates data acquired by scrapers.

    Each scraper must return a dataframe.

    :param source: identifies the scraper to use to gather the data.
    :type source: str
    :param inputs: contains any parameters necessary for the scraper to run.
    :type inputs: dict
    """
    def __init__(self, stream_name, source, analysis_name, inputs):
        connect(settings.DATABASE_NAME)
        self.study_list = []
        self.alerts_board = AlertsBoard()
        self.source = source
        self.scraper = Scraper(source, inputs).get()
        self.analysis = Analysis(analysis_name, inputs)
        self.track(settings.REFRESH_INTERVAL)

    def track(self, interval):
        """Periodically update the data and run any desired analyses."""
        try:
            os.mkdir(self.basedir)
        except FileExistsError:
            pass

        while True:
            print("{0}: Updating {1}".format(time.asctime(), self.source))
            self.update()
            time.sleep(interval)
        

    def update(self):
        """Update all of the tracked data."""
        self.scraper.refresh()
        data = self.scraper.get_data()
        newstudy = self.create_study(data)
        self.analysis.analyze(newstudy)
        self.cycle_studies(newstudy)
        self.alerts_board.update(self.study_list)
        print(newstudy.data)
        # Test MongoDB
        client = MongoClient()
        db = client.FintristTest
        collection = db.studies
        study_data = {
            'type': 'stock',
            'name': newstudy.name,
            'data': bson.binary.Binary(pickle.dumps(newstudy.data, protocol=2)),
        }
        result = collection.insert_one(study_data)
        print('One study: {0}'.format(result.inserted_id))

    def create_study(self, data):
        """Create a study object."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        study = Study(timestamp, self.basedir, data)
        study.save()
        return study

    def cycle_studies(self, study):
        """Add to the study list, but don't let it get too long."""
        self.study_list.append(study)
        if len(self.study_list) > settings.KEEP_STUDIES:
            self.study_list[0].delete()
            self.study_list = self.study_list[1:]

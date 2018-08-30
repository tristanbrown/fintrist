"""
The Tracking Engine that monitors the wall clock and any alerts, triggering
events such as data refresh.
"""
import os
import time

from fintrist import settings
from fintrist.alerts import AlertsBoard
from fintrist.analysis import Analysis, Study
from fintrist.scrapers.base import Scraper

class TrackingEngine():
    """Tracking engine that monitors and updates data acquired by scrapers.

    Each scraper must return a dataframe.

    :param source: identifies the scraper to use to gather the data.
    :type source: str
    :param inputs: contains any parameters necessary for the scraper to run.
    :type inputs: dict
    """
    def __init__(self, source, analysis_name, inputs):
        self.basedir = self.generate_base_dir()
        self.study_list = []
        self.alerts_board = AlertsBoard()
        self.source = source
        self.scraper = Scraper(source, inputs).get()
        self.analysis = Analysis(analysis_name, inputs)
        self.track(settings.REFRESH_INTERVAL)

    def track(self, interval):
        """Periodically update the data and run any desired analyses."""
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

    def generate_base_dir(self):
        """Choose the directory in which to save the studies."""
        datadir = settings.DATA_DIR
        i = 0
        while os.path.exists(os.path.join(datadir, str(i))):
            i += 1
        return os.path.join(datadir, str(i))

    def create_study(self, data):
        """Create a study object."""
        study = Study(data)
        study.save()
        return study

    def cycle_studies(self, study):
        """Add to the study list, but don't let it get too long."""
        self.study_list.append(study)
        if len(self.study_list) > settings.KEEP_STUDIES:
            self.study_list[0].delete()
            self.study_list = self.study_list[1:]

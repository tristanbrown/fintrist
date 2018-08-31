"""
The Tracking Engine that monitors the wall clock and any alerts, triggering
events such as data refresh.
"""
import os
import time

from fintrist import settings
from fintrist.alerts import AlertsBoard
from fintrist.analysis import Analysis
from fintrist.scrapers.base import Scraper
from fintrist.study import Study

class Stream():
    """Tracking engine that monitors and updates data acquired by scrapers.

    Each scraper must return a dataframe.

    :param source: identifies the scraper to use to gather the data.
    :type source: str
    :param inputs: contains any parameters necessary for the scraper to run.
    :type inputs: dict
    """
    def __init__(self, stream_name, source, analysis_name, inputs):
        self.basedir = os.path.join(settings.DATA_DIR, stream_name)
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

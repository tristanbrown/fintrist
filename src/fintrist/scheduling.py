"""Scheduler"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.jobstores.mongodb import MongoDBJobStore

from fintrist.settings import Config

def create_scheduler():
    """Create a configured scheduler."""
    jobstores = {
        'default': MongoDBJobStore(
            database=Config.DATABASE_NAME,
            collection=f"jobs_{Config.APP_HOST}",
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            username=Config.USERNAME,
            password=Config.PASSWORD,
            authSource='admin',
            )}
    executors = {'default': ProcessPoolExecutor(max_workers=4)}
    job_defaults = {'coalesce': True, 'max_instances': 1, 'misfire_grace_time': 5,}

    return BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
    )

scheduler = create_scheduler()

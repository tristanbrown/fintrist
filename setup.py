from setuptools import setup, find_packages
import os

import __about__

REQUIREMENTS = [
    'pandas',
    'matplotlib',
    'flask',
    'flask_wtf',
    'wtforms',
    'dash',
    'python-dotenv',
    'mongoengine',
    'blinker',
    'dask[distributed]',
    'bokeh>=0.13.0',
    'jupyter-server-proxy',
    'cloudpickle',
    'pymc3',
    'finta',
    'tulipy',
    'ruamel.yaml',
    'pandas-datareader',
    'arrow',
    'torch>=1.6.*',
    'pandas-market-calendars',
    f"alpaca_management @ git+https://{os.environ['GITHUB_TOKEN']}:x-oauth-basic@github.com/tristanbrown/alpaca_management.git@v0.1.2#egg=alpaca_management"
]

setup(
    name=__about__.__name__,
    version=__about__.__version__,
    author=__about__.__author__,
    author_email=__about__.__email__,
    description=__about__.__desc__,
    license=__about__.__license__,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=REQUIREMENTS,
    python_requires='>=3.6, !=3.7.2',
    entry_points={
        'console_scripts': [
            'fintrist_app = fintrist_app.__main__:run',
        ]
    },
)

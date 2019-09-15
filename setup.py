from setuptools import setup, find_packages

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
    'apscheduler',
    'dask[distributed]',
    'cloudpickle',
    'alpha_vantage>=2.0.0',
    'pymc3',
    'finta',
    'tulipy',
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

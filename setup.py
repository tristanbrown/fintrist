from setuptools import setup, find_packages

REQUIREMENTS = [
    'pandas',
    'matplotlib',
    'flask',
    'python-dotenv',
    'pymongo',
    'mongoengine',
    'alpha_vantage>=2.0.0',
]

setup(
    name='fintrist',
    version='0.0.1',
    author="Tristan R. Brown",
    author_email="brown.tristan.r@gmail.com",
    description=("A personal financial analysis package. "),
    license="Private",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=REQUIREMENTS,
    entry_points={
        'console_scripts': [
            # 'fintrist = fintrist.__main__:main',
        ]
    },
)

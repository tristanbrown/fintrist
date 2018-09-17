from setuptools import setup, find_packages

setup(
    name='fintrist',
    version='0.0.1',
    author="Tristan R. Brown",
    author_email="brown.tristan.r@gmail.com",
    description=("A personal financial analysis package. "),
    license="Private",
    packages=find_packages(),
    install_requires=['pandas', 'matplotlib', 'alpha_vantage'],
    entry_points={
        'console_scripts': [
            'fintrist = fintrist.__main__:main',
            'manage.py = scripts.manage:main',
        ]
    },
)

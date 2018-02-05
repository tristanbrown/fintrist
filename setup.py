from setuptools import setup, find_packages

setup(name='mypkg',
    version='0.0.1',
    author = "Tristan R. Brown",
    author_email = "brown.tristan.r@gmail.com",
    description = ("A package template. "),
    url = 'https://github.com/tristanbrown/template-pkg',
    license = "MIT",
    packages = find_packages(),
    install_requires = [''],
    entry_points = {
        'console_scripts': [
            'mypkg = mypkg.__main__:main'
        ]
    },
    )
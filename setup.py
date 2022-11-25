#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='528_project',
    version='0.0.0',
    description='Tools and analysis for decoding behavioral/experimental variables from neural data',
    packages=find_packages(exclude=[]),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "sklearn", 
        "nilearn",
        "matplotlib", 
    ]
)
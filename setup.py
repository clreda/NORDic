#!/usr/bin/env python

from setuptools import setup

NAME = "NORDic"
VERSION = "1.0.0"

setup(name=NAME,
    version=VERSION,
    author="Clémence Réda",
    author_email="clemence.reda@inserm.fr",
    url="https://github.com/clreda/NORDic",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords='',
    description="Network Oriented Repurposing of Drugs (NORDic): network identification and master regulator detection",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "potassco",
        "clingo",
        "graphviz",
        "bonesis",
        "cmapPy",
        "matplotlib",
        "scikit_learn",
        "scipy",
        "qnorm",
        "tqdm",
        "mpbn-sim"
    ],
    entry_points={},
    packages=[NAME],
)

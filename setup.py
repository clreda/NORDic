#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from subprocess import check_call

def postinstall():
    check_call("git clone https://github.com/cmap/cmapPy".split())
    check_call("sed -i 's/temp_array = temp_array.astype(\"str\")/temp_array = np.core.defchararray.decode(temp_array, \"utf8\")  # <- introduced for Python3 compatibility/' cmapPy/cmapPy/pandasGEXpress/parse_gctx.py", shell=True)
    check_call("python3 -m pip install cmapPy/".split())
    check_call("rm -rf cmapPy/".split())
    check_call("python3 -m pip install git+https://github.com/bnediction/mpbn-sim.git@5f919c5c62e111628136d62357902966404b988e".split())

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        postinstall()

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        postinstall()


class PostEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        postinstall()

NAME = "NORDic"
VERSION = "1.0.4"

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
    packages=find_packages(where="src"),
    package_dir={'':"src"},
    python_requires='~=3.8.5',
    install_requires=[
        "clingo==5.6.1",
        "graphviz==0.20.1",
        "bonesis==0.4.91",
        "matplotlib==3.3.4",
        "scikit_learn==1.1.2",
        "scipy==1.6.2",
        "qnorm==0.5.1",
        "tqdm==4.62.3",
    ],
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
        'egg_info': PostEggInfoCommand,
    },
    entry_points={},
)

from setuptools import setup, find_packages

NAME = "NORDic"
VERSION = "9999"

setup(name=NAME,
    version=VERSION,
    author="Clémence Réda",
    author_email="clemence.reda@inserm.fr",
    url="https://github.com/clreda/NORDic",
    license_files = ('LICENSE'),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords='',
    description="Network Oriented Repurposing of Drugs (NORDic): network identification / master regulator detection / drug effect simulator / drug repurposing",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={'':"src"},
    python_requires='>=3.8.5',
    install_requires=[
        "pandas>=1.5.1",
        "numpy>=1.22.4,<2.0.0",
        "clingo>=5.6.1",
        "graphviz>=0.20.1",
        "bonesis>=0.4.91",
        "mpbn>=2.0",
        "matplotlib>=3.3.4",
        "scikit_learn>=1.1.2",
        "scipy>=1.6.2",
        "qnorm>=0.5.1",
        "tqdm>=4.62.3",
        "cmapPy>=4.0.1",
	"openpyxl>=3.0.10",
	"quadprog>=0.1.11",
	"seaborn>=0.12.1",
	"omnipath>=1.0.6",
	"maboss>=0.8.4"
    ],
    entry_points={},
)

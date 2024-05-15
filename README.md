# Network Oriented Repurposing of Drugs (NORDic)
[![Anaconda version](https://anaconda.org/creda/nordic/badges/version.svg)](https://anaconda.org/creda/nordic) [![PyPI version](https://badge.fury.io/py/nordic.svg)](https://badge.fury.io/py/nordic) [![Zenodo version](https://zenodo.org/badge/DOI/10.5281/zenodo.7239047.svg)](https://doi.org/10.5281/zenodo.7239047) [![publication](https://joss.theoj.org/papers/a8173d7864bf1bc8dd074c7ce80d6d7d/status.svg)](https://joss.theoj.org/papers/a8173d7864bf1bc8dd074c7ce80d6d7d)

## Statement of need

Being able to build in an automated and reproducible way a model of gene interactions and their influences on gene activity will allow to consider more complex diseases and biological phenomena, on a larger set of genes. These models might speed up the understanding of the gene regulation hierarchy by bioinformaticians and biologists, allow to predict novel drugs or gene targets which might be investigated later for healthcare purposes. In particular, the network-oriented approach allow to predict off-targets, which are non-specific drug targets which might lead to otherwise unexpected toxic side effects.

[NORDic](https://github.com/clreda/NORDic) is an open-source package which allows to focus on a network-oriented approach to identify regulatory mechanisms linked to a disease, to detect master regulators in a diseased transcriptomic context, to simulate drug effects on a patient through a network, and adaptively test drugs to perform sample-efficient, error-bound drug repurposing. As such, it is comprised of four distinct submodules:
- **NORDic NI** identifies a disease-associated gene regulatory network (as a *Boolean network*) with its dynamics combining several biological sources and methods. The main contribution is that this inference can be performed even in the absence of previously curated experiments and prior knowledge networks.
- **NORDic PMR** detects master regulators in a Boolean network, given examples of diseased transcriptomic profiles. In contrast to prior works, the score assigned to (groups of) master regulators takes into account the network topology as well as its dynamics with respect to the diseased profiles.
- **NORDic DS** (since version 2.0.0) scores the effect of a treatment on a patient (the higher the score, the most promising the treatment) based on a Boolean network. This approach computes the similarity of a predicted *treated* patient profile to control profiles to output a *signature reversal score* associated with the considered drug. The *signature reversion* approach has already been applied with some success.
- **NORDic DR** (since version 2.0.0) uses the routine in **NORDic DS** and a bandit algorithm to adaptively test treatments and perform drug repurposing. This novel approach allows to get recommendations with a bounded probability of false discovery, while remaining sample efficient.

## Install the latest release

### Supported platforms

The package has been developed and mainly tested on a Linux platform. Issues when using it on Windows or Macs can be reported on this GitHub repository.

### Dependencies

It is strongly advised to create a virtual environment using Conda (python>=3.8)

```bash
conda create -n test_NORDic python=3.8
conda activate test_NORDic
```

The complete list of dependencies can be found at [requirements.txt](https://raw.githubusercontent.com/clreda/NORDic/main/pip/requirements.txt) or [meta.yaml](https://raw.githubusercontent.com/clreda/NORDic/main/conda/meta.yaml).

### Using pip (package hosted on PyPI)

We need to install missing dependencies from PyPI:

```bash
apt-get install graphviz # for Debian distributions, check the correct command for your own distribution
conda install -c colomoto -y -q maboss
pip install NORDic 
```

### Using conda (package hosted on Anaconda.org)

All dependencies (except for clingo) are retrievable from Anaconda:

```bash
conda install -c potassco clingo

conda install -c creda -y -q nordic
conda install -c bioconda -y -q nordic
```

### Using [CoLoMoTo-Docker](https://github.com/colomoto/colomoto-docker) (since March 1st, 2023)

```bash
pip install -U colomoto-docker
colomoto-docker
```

## Quick access to NORDic

The easiest way not to having to deal with environment configuration is to use the CoLoMoTo-Docker. First ensure that [Docker](https://docs.docker.com/engine/install/) is installed for your distribution:

```bash
service docker start
docker run hello-world # downloads a test image, runs it in a container (prints a confirmation message), exits
```

Then install the [CoLoMoTo-Docker](https://github.com/colomoto/colomoto-docker):

```bash
conda create -n nordic_colomoto python=3.10 -y
conda activate nordic_colomoto
pip install -U colomoto-docker
mkdir notebooks
colomoto-docker -v notebooks:local-notebooks ## or any version later than 2023-03-01
```

In the Jupyter browser, you will see a ``local-notebooks`` directory which is bound to your ``notebooks`` directory, where you can find all tutorial notebooks in CoLoMoTo, the one for NORDic included (``NORDic-demo.ipynb``).

## Example usage

Once installed, to import **NORDic**

```python
import NORDic 
```

Please check out the associated Jupyter notebooks in folder [*notebooks/*](https://github.com/clreda/NORDic/tree/main/notebooks), starting with [this short notebook](https://github.com/clreda/NORDic/blob/main/notebooks/NORDic%20CoLoMoTo.ipynb). All functions are documented, so one can check out the inputs and outputs of a function *func* by typing

```python
help(func)
```

The documentation website is up at [this page](https://clreda.github.io/NORDic).

## Cite

If you use **NORDic** in academic research, please cite the following JOSS paper:

[![publication](https://joss.theoj.org/papers/a8173d7864bf1bc8dd074c7ce80d6d7d/status.svg)](https://joss.theoj.org/papers/a8173d7864bf1bc8dd074c7ce80d6d7d)

+ Formatted citation:

> Réda et al., (2023). NORDic: a Network-Oriented package for the Repurposing of Drugs. Journal of Open Source Software, 8(90), 5532, https://doi.org/10.21105/joss.05532

+ BibTeX citation:

```bash
@article{Réda2023, 
    doi = {10.21105/joss.05532}, 
    url = {https://doi.org/10.21105/joss.05532}, 
    year = {2023}, 
    publisher = {The Open Journal}, 
    volume = {8}, 
    number = {90}, 
    pages = {5532}, 
    author = {Clémence Réda and Andrée Delahaye-Duriez}, 
    title = {NORDic: a Network-Oriented package for the Repurposing of Drugs}, 
    journal = {Journal of Open Source Software} 
} 
```

## License

This code is under [OSI-approved](https://opensource.org/licenses/) [MIT license](https://raw.githubusercontent.com/clreda/NORDic/main/LICENSE).

## Community guidelines with respect to contributions, issue reporting, and support

[Pull requests](https://github.com/clreda/NORDic/pulls) and [issue flagging](https://github.com/clreda/NORDic/issues) are welcome, and can be made through the GitHub interface. Support can be provided by reaching out to clemence.reda [at]() uni-rostock.de. However, please note that contributors and users **must** abide by the [Code of Conduct](https://raw.githubusercontent.com/clreda/NORDic/main/CODE%20OF%20CONDUCT).

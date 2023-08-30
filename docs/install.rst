Installation
------------

Supported platforms
:::::::::::::::::::

The package has been developed and mainly tested on a Linux platform. Issues when using it on Windows or Macs can be reported on this GitHub repository.

Dependencies
::::::::::::

It is strongly advised to create a virtual environment using Conda (python>=3.8) ::

    $ conda create -n test_NORDic python=3.8
    $ conda activate test_NORDic

The complete list of dependencies can be found at `requirements.txt <https://raw.githubusercontent.com/clreda/NORDic/main/pip/requirements.txt>`_ or `meta.yaml <https://raw.githubusercontent.com/clreda/NORDic/main/conda/meta.yaml>`_.

Using pip (package hosted on PyPI)
::::::::::::::::::::::::::::::::::

We need to install missing dependencies from PyPI: ::

    $ apt-get install graphviz # for Debian distributions, check the correct command for your own distribution
    $ conda install -c colomoto -y -q maboss
    $ pip install NORDic 

Using conda (package hosted on Anaconda.org)
::::::::::::::::::::::::::::::::::::::::::::

All dependencies are retrievable from Anaconda: ::

    $ conda install -c creda -y -q nordic

Using `CoLoMoTo-Docker <https://github.com/colomoto/colomoto-docker>`_ (since March 1st, 2023)
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Run the following command lines: ::

    $ pip install -U colomoto-docker
    $ colomoto-docker

From Source Files
:::::::::::::::::

Download the `tar.gz file from PyPI <https://pypi.python.org/pypi/nordic/>`_ and extract it.  The library consists of a directory named `NORDic` containing several Python modules.
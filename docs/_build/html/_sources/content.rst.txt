Statement of need 
-----------------

Being able to build in an automated and reproducible way a model of gene interactions and their influences on gene activity will allow to consider more complex diseases and biological phenomena, on a larger set of genes. These models might speed up the understanding of the gene regulation hierarchy by bioinformaticians and biologists, allow to predict novel drugs or gene targets which might be investigated later for healthcare purposes. In particular, the network-oriented approach allow to predict off-targets, which are non-specific drug targets which might lead to otherwise unexpected toxic side effects.

`NORDic <https://github.com/clreda/NORDic>`_ is an open-source package which allows to focus on a network-oriented approach to identify regulatory mechanisms linked to a disease, to detect master regulators in a diseased transcriptomic context, to simulate drug effects on a patient through a network, and adaptively test drugs to perform sample-efficient, error-bound drug repurposing. As such, it is comprised of four distinct submodules:

- NORDic NI identifies a disease-associated gene regulatory network (as a Boolean network) with its dynamics combining several biological sources and methods. The main contribution is that this inference can be performed even in the absence of previously curated experiments and prior knowledge networks.

- NORDic PMR detects master regulators in a Boolean network, given examples of diseased transcriptomic profiles. In contrast to prior works, the score assigned to (groups of) master regulators takes into account the network topology as well as its dynamics with respect to the diseased profiles.

- NORDic DS (since version 2.0.0) scores the effect of a treatment on a patient (the higher the score, the most promising the treatment) based on a Boolean network. This approach computes the similarity of a predicted treated patient profile to control profiles to output a signature reversal score associated with the considered drug. The signature reversion approach has already been applied with some success.

- NORDic DR (since version 2.0.0) uses the routine in NORDic DS and a bandit algorithm to adaptively test treatments and perform drug repurposing. This novel approach allows to get recommendations with a bounded probability of false discovery, while remaining sample efficient.

Usage
-----

Quick access to NORDic
::::::::::::::::::::::::

The easiest way not to having to deal with environment configuration is to use the CoLoMoTo-Docker. First ensure that `Docker <https://docs.docker.com/engine/install/>`_ is installed for your distribution: ::

 $ service docker start
 $ docker run hello-world # downloads a test image, runs it in a container (prints a confirmation message), exits

Then install the `CoLoMoTo-Docker <https://github.com/colomoto/colomoto-docker>`_: ::

 $ conda create -n nordic_colomoto python=3.10 -y
 $ conda activate nordic_colomoto
 $ pip install -U colomoto-docker
 $ mkdir notebooks
 $ colomoto-docker -v notebooks:local-notebooks ## or any version later than 2023-03-01

In the Jupyter browser, you will see a local-notebooks directory which is bound to your notebooks directory, where you can find all tutorial notebooks in CoLoMoTo, the one for NORDic included (NORDic-demo.ipynb).

Environment
:::::::::::

In order to run notebook `Introduction to NORDic.ipynb <https://github.com/clreda/NORDic/blob/main/notebooks/NORDic%20CoLoMoTo.ipynb>`__, it is strongly advised to create a virtual environment using Conda (python>=3.8): ::

 $ conda create -n test_NORDic python=3.8 -y
 $ conda activate test_NORDic
 $ conda install -c creda -y -q nordic
 $ python3 -m pip install notebook>=6.5.4 markupsafe==2.0.1 ## packages for Jupyter notebook
 $ conda deactivate ## refresh the virtual environment
 $ conda activate test_NORDic
 $ cd notebooks/ && jupyter notebook

The complete list of dependencies for NORDic can be found at `requirements.txt <https://raw.githubusercontent.com/clreda/NORDic/main/pip/requirements.txt>`__ (pip) or `meta.yaml <https://raw.githubusercontent.com/clreda/NORDic/main/conda/meta.yaml>`__ (conda).

Example usage
:::::::::::::

Once installed, to import NORDic into your Python code: ::

 $ import NORDic

Please check out notebook `Introduction to NORDic.ipynb <https://github.com/clreda/NORDic/blob/main/notebooks/NORDic%20CoLoMoTo.ipynb>`__. All functions are documented, so one can check out the inputs and outputs of a function func by typing: ::

$ > help(func)
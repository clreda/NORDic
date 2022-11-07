# Network Oriented Repurposing of Drugs (NORDic)
(c) Clémence Réda, 2022.

[NORDic](https://github.com/clreda/NORDic) is an open-source package which allows to focus on a network-oriented approach to identify regulatory mechanisms linked to a disease, master regulators, and to simulate drug effects on a network, and adaptively test drugs to perform drug repurposing. As such, it is comprised of four distinct parts:
- **NORDic NI** identifies a disease-associated gene regulatory network (as a *Boolean network*) with its dynamics from several biological sources.
- **NORDic PMR** detects master regulators in a Boolean network.
- **NORDic DS** (since version 2.0.0) scores the effect of a treatment on a patient (the higher the score, the most promising the treatment) based on a Boolean network.
- **NORDic DR** (since version 2.0.0) uses the routine in **NORDic DS** and a bandit algorithm to adaptively test treatments and perform drug repurposing.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7239047.svg)](https://doi.org/10.5281/zenodo.7239047)

To learn how to use the different methods, please check out the associated Jupyter notebooks.

## Citation

If you use NORDic in published research, please cite the following preliminary work:

> Réda, C., & Delahaye-Duriez, A. (2022). Prioritization of Candidate Genes Through Boolean Networks. In *International Conference on Computational Methods in Systems Biology* (pp. 89-121). Springer, Cham.

Due to the presence of copyrighted databases, the license for this code is [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Installation

```bash
pip install NORDic # latest version
```

## Pull requests, issues, suggestions?

clemence.reda@inserm.fr

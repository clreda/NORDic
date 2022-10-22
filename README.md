# Network Oriented Repurposing of Drugs (NORDic) package
(c) Clémence Réda, 2022.

Due to the presence of copyrighted databases, the license for this code is [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Installation

```bash
pip install NORDic
```

## Using the "refractory epilepsy" application

Import the initial states from Mirza et al., 2017 and the M30 genes from Delahaye-Duriez et al., 2016

```bash
conda activate NORDic_env
python3 download_Refractory_Epilepsy_Data.py
conda deactivate
```

## Building a Boolean network

You need to register to the [LINCS L1000 database](https://clue.io/developer-resources#apisection) and the [DisGeNet database](https://www.disgenet.org/) and write up the corresponding credentials and API keys to files 

```python
from NORDic.NORDic_NI.functions import network_identification
solution = network_identification(file_folder, taxon_id, path_to_genes, ...)
```

The final network solution is written to *<file_folder>solution.bnet*.

## Detection of master regulators

Using the filename in .bnet "network_name", the size k of the set of master regulators and the set of initial states states

```python
from NORDic.NORDic_PMR.functions import greedy
S, spreads = greedy(network_name, k, states, ...)
```

The result file is named *application_regulators.csv*.

## Network analysis with Cytoscape

Network analyses are performed with Cytoscape 3.8.0. You need to download the module CytoCtrlAnalyser (version 1.0.0). Then run

```python
from NORDic.NORDic_NI.functions import solution2cytoscape
solution2cytoscape(solution, file_folder+"solution_minimal_cytoscape")
```

which will create a style file (in .xml) and a network file readable by Cytoscape (in .sif). 

## Citation

If you use NORDic in published research, please cite the following preliminary work:

> Réda, C., & Delahaye-Duriez, A. (2022). Prioritization of Candidate Genes Through Boolean Networks. In *International Conference on Computational Methods in Systems Biology* (pp. 89-121). Springer, Cham.

## Pull requests, issues, suggestions?

clemence.reda@inserm.fr

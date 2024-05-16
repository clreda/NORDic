# NORDic (Boolean network; end-to-end inference)

Note that NORDic relies on [BoneSiS](https://github.com/bioasp/bonesis) for the inference of Boolean networks. In those synthetic examples, we consider the Boolean trajectories, an appropriate prior knowledge network (or the complete undirected unsigned graph if we have no biological information), and then run the inference + model selection pipeline to get a single model. Note that NORDic can only consider an initial and final (steady) states (no intermediary ones) and a single gene perturbation at a time. Then, some (parts) of the Boolean trajectories are actually ignored in these notebooks.

## Installation

```bash
## create virtual env
conda create -n nordic_env python=3.8 -y
conda activate nordic_env

conda install -c colomoto -y -q maboss
python3 -m pip install nordic jupyter

## clean up (if needed)
python3 -m pip cache purge
conda clean -a -y
```

## Launch notebook(s)

```bash
conda activate nordic_env
jupyter notebook toy_star_1.ipynb
jupyter notebook toy_reprogramming_1.ipynb
jupyter notebook synthetic_random_diff3-1.ipynb
```

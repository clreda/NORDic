#!/bin/bash

conda create -n NORDic_env -c potassco clingo python=3.8.5
conda activate NORDic_env
python3 -m graphviz
python3 -m pip install git+https://github.com/bioasp/bonesis.git@64e88178816f86ff112cd9c9e423cf40e5029c5c
git clone https://github.com/cmap/cmapPy
sed -i "s/temp_array = temp_array.astype('str')/temp_array = np.core.defchararray.decode(temp_array, 'utf8')  # <- introduced for Python3 compatibility/" cmapPy/cmapPy/pandasGEXpress/parse_gctx.py
python3 -m pip install cmapPy/
rm -rf cmapPy/
python3 -m pip install matplotlib==3.3.4 scikit_learn==1.1.2 scipy==1.6.2 qnorm==0.5.1 tqdm==4.62.3
python3 -m pip install git+https://github.com/bnediction/mpbn-sim.git@5f919c5c62e111628136d62357902966404b988e
sed -i 's/sys.exit(cli())/sys.exit()/' ~/miniconda3/envs/NORDic_env/bin/mpbn_sim
conda deactivate


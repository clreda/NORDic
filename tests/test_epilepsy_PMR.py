# coding:utf-8

import imports
import os
from subprocess import call as sbcall
from glob import glob

## Download epilepsy-related data
from download_Refractory_Epilepsy_Data import *

from NORDic.NORDic_PMR.functions import greedy
from NORDic.UTILS.utils_state import binarize_experiments
from NORDic.UTILS.utils_grn import load_grn

seed_number=0
root="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/"
k=1
solution_fname=file_folder+"solution.bnet"

from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2)

## 1. Get M30 genes
with open(solution_fname, "r") as f:
    network = str(f.read())
genes = [x.split(" <- ")[0] for x in network.split("\n")[:-1]]
gene_outputs = [x.split(" <- ")[0] for x in network.split("\n")[:-1] if (x.split(" <- ")[1] not in [x.split(" <- ")[0], "0", "1"])]
## 2. Get epileptic hippocampi profiles (private communication of normalized count matrix from the raw count data in ArrayExpress)
if (not os.path.exists(path_to_initial_states)):
    raise ValueError("File does not exist.")
## df: normalized gene expression DataFrame (rows=genes, columns=samples)
df = pd.read_csv(path_to_initial_states, index_col=0)
df = df.loc[~df.index.duplicated()]
## 3. Aggregate and binarize initial states
states = binarize_experiments(df, thres=0.5, method="binary").astype(int)
states.columns = range(states.shape[1])
## 4. Convert to an accepted format for GRNs
grn_fname=file_folder+"grn.bnet"
if (not os.path.exists(grn_fname)):
    fname_ls = glob(solution_fname)
    assert len(fname_ls)>0
    sbcall("sed 's/ <- /, /g' "+fname_ls[0]+" > "+grn_fname, shell=True)

IM_params = {
    "seed": seed_number,
    "njobs": min(5, njobs),
    "gene_inputs": genes, # genes to be perturbed
    "gene_outputs": gene_outputs # genes to be observed
}
SIMU_params = {
    'nb_sims': 100,
    'rates': "fully_asynchronous",
    'thread_count': njobs,
    'depth': "constant_unitary",
}

#########################################
## Test for master regulator detection ##
#########################################

S, spreads = greedy(grn_fname, k, states, IM_params, SIMU_params, save_folder=file_folder)

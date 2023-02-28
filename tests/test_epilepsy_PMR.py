# coding:utf-8

import imports
import os
from subprocess import call as sbcall
from glob import glob
import pandas as pd
import numpy as np

LINCS_args = {
        "path_to_lincs": "../lincs/",
}

## Download epilepsy-related data
from download_Refractory_Epilepsy_Data import get_EPILEPSY_initial_states, file_folder, path_to_initial_states
get_EPILEPSY_initial_states(LINCS_args["path_to_lincs"])
file_folder="refractory_epilepsy3/" ## "refractory_epilepsy/"

from NORDic.NORDic_PMR.functions import greedy
from NORDic.UTILS.utils_state import binarize_experiments
from NORDic.UTILS.utils_grn import load_grn

seed_number=0
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
## df: normalized gene expression DataFrame (rows=genes, columns=samples)
df = pd.read_csv(path_to_initial_states, index_col=0)
df = df.loc[~df.index.duplicated()]
## 3. Aggregate and binarize initial states
states = binarize_experiments(df, thres=0.5, method="binary").astype(int)
states.columns = ["State%d" % i for i in range(states.shape[1])]
states = states[states.columns[:10]] ##
## 4. Convert to an accepted format for GRNs
grn_fname=file_folder+"grn.bnet"
if (not os.path.exists(grn_fname)):
    fname_ls = glob(solution_fname)
    assert len(fname_ls)>0
    sbcall("sed 's/ <- /, /g' "+fname_ls[0]+" > "+grn_fname, shell=True)

IM_params = {
    "seed": seed_number,
    "gene_inputs": genes, # genes to be perturbed
    "gene_outputs": gene_outputs # genes to be observed
}
SIMU_params = {
    'nb_sims': 1000,
    'rates': "fully_asynchronous",
    'thread_count': njobs,
    'depth': "constant_unitary",
}

#########################################
## Test for master regulator detection ##
#########################################

#S, spreads = greedy(grn_fname, k, states, IM_params, SIMU_params, save_folder=file_folder)

##https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

n_gene,n_state=njobs,1
spreads_df = []
for it, state in enumerate(list(range(states.shape[1]))):
    spreads_state, gene_lst_state = [], []
    for ig, gene_lst in enumerate(chunks(genes, n_gene)):
        if (not os.path.exists("spreads_ngene=%d-_nstate=%d.csv" % (ig*n_gene, it*n_state))):
            print((gene_lst, ig, state))
            IM_params = {
                "seed": seed_number,
                "gene_inputs": gene_lst, # genes to be perturbed
                "gene_outputs": gene_outputs # genes to be observed
            }
            S, spreads = greedy(grn_fname, k, states[[states.columns[state]]], IM_params, SIMU_params, save_folder=file_folder)
            sbcall("mv %s/application_regulators.csv spreads_ngene=%d-_nstate=%d.csv" % (file_folder, ig*n_gene, it*n_state), shell=True)
            sbcall("rm -f "+file_folder+"application_regulators.json", shell=True)
        else:
            spreads = pd.read_csv("spreads_ngene=%d-_nstate=%d.csv" % (ig*n_gene, it*n_state), index_col=0)
        spreads_state += list(spreads[spreads.columns[0]])
        gene_lst_state += list(spreads.index)
    spreads_df.append(pd.DataFrame(spreads_state, index=gene_lst_state, columns=[states.columns[state]]))
from scipy.stats.mstats import gmean
spreads = spreads_df[0].join(spreads_df[1:], how="outer")
gmeans = [(gmean([(sg+1) for sg in list(spreads.loc[g])])-1) for g in spreads.index]
spreads = pd.DataFrame(gmeans, index=spreads.index, columns=["result"])
vals = list(spreads[spreads.columns[0]])
ids_max = np.argwhere(np.array(vals)==np.max(vals))
S = [genes[x[0]] for x in ids_max]
spreads.columns = ["["+(", ".join(["["+genes[x[0]]+"]" for x in ids_max]))+"]"]

print("ANSWER:\n%s" % str(S))

print(spreads.sort_values(by=spreads.columns[0], ascending=False))

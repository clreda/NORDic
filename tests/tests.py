#coding: utf-8

## Run all the code in NORDic/notebooks/NORDic CoLoMoTo.ipynb

import NORDic
import pandas as pd
from multiprocessing import cpu_count
import os
import numpy as np

## I. Building a small regulatory model of CCHS (**NORDic NI**)

## Registration to databases
DisGeNET_credentials = "../tests/credentials_DISGENET.txt"
STRING_credentials = "../tests/credentials_STRING.txt"
LINCS_credentials = "../tests/credentials_LINCS.txt"
## Parameters
seed_number=123456
njobs=max(1,cpu_count()-2) ## all available threads but 2
file_folder="ToyOndine/"
taxon_id=9606 # human species
disease_cids=["C1275808"] ## Concept ID of Ondine syndrome
cell_lines=["NPC", "SHSY5Y"] # brain cell lines in LINCS L1000
## Information about the disease
DISGENET_args = {"credentials": DisGeNET_credentials, "disease_cids": disease_cids}
## Selection of parameters relative to the prior knowledge network 
STRING_args = {"credentials": STRING_credentials, "score": 0}
EDGE_args = {"tau": 0, "filter": True, "connected": True}
accept_nonRNA=True
preserve_network_sign=True
## Selection of parameters relative to experimental constraints
LINCS_args = {"path_to_lincs": "../lincs/", "credentials": LINCS_credentials, "cell_lines": cell_lines, 
              "thres_iscale": None}
SIG_args = {"bin_thres": 0.5}
force_experiments=False
## Selection of parameters relative to the inference of networks
BONESIS_args = {"limit": 1, "exact": True, "max_maxclause": 3}
## Advanced
DESIRABILITY = {"DS": 3, "CL": 3, "Centr": 3, "GT": 1}

## The undirected, unsigned network from STRING
network_content = pd.DataFrame([], index=["preferredName_A", "preferredName_B", "sign", "directed", "score"])
network_content[0] = ["PHOX2B", "BDNF", 2, 0, 0.342]
network_content[1] = ["PHOX2B", "GDNF", 2, 0, 0.572]
network_content[2] = ["PHOX2B", "RET", 2, 0, 0.605]
network_content[3] = ["PHOX2B", "EDN3", 2, 0, 0.607]
network_content[4] = ["PHOX2B", "ASCL1", 2, 0, 0.676]
network_content[5] = ["ASCL1", "RET", 2, 0, 0.397]
network_content[6] = ["ASCL1", "EDN3", 2, 0, 0.433]
network_content[7] = ["ASCL1", "GDNF", 2, 0, 0.47]
network_content[8] = ["ASCL1", "BDNF", 2, 0, 0.519]
network_content[9] = ["EDN3", "BDNF", 2, 0, 0.15]
network_content[10] = ["EDN3", "RET", 2, 0, 0.622]
network_content[11] = ["EDN3", "GDNF", 2, 0, 0.634]
network_content[12] = ["RET", "BDNF", 2, 0, 0.438]
network_content[12] = ["RET", "GDNF", 2, 0, 0.999]
network_content[12] = ["GDNF", "BDNF", 2, 0, 0.95]
network_content.T.to_csv("network.tsv", sep="\t", index=None)

## An experiment retrieved from LINCS L1000 involving those genes
index=["PHOX2B","EDN3","RET","GDNF","BDNF","ASCL1","cell_line","annotation","perturbed","perturbation","sigid"]
experiments_content = pd.DataFrame([], index=index)
experiments_content["BDNF_KD_SHSY5Y"] = [1,1,1,0,0,0,"Cell","2","BDNF","KD","Sig1"] ## mutated profile
experiments_content["initial_SHSY5Y"] = [1,1,1,1,0,0,"Cell","1","None","None","Sig2"] ## control/initial profile
experiments_content.to_csv("experiments.csv")

from NORDic.NORDic_NI.functions import network_identification
solution = network_identification(file_folder, taxon_id, path_to_genes=None, 
    network_fname="network.tsv", experiments_fname="experiments.csv", 
    disgenet_args=DISGENET_args, string_args=STRING_args, edge_args=EDGE_args, lincs_args=LINCS_args, 
    sig_args=SIG_args, bonesis_args=BONESIS_args, weights=DESIRABILITY, seed=seed_number, njobs=njobs,
    force_experiments=force_experiments, accept_nonRNA=accept_nonRNA, preserve_network_sign=preserve_network_sign)

assert os.path.exists(file_folder+'inferred_max_criterion_solution.png')

with open(file_folder+"solution.bnet", "r") as f:
    network = f.read().split("\n")
assert len(network)==6

## II. Prioritization of master regulators (**NORDic PMR**)

np.random.seed(12345)
with open(file_folder+"solution.bnet", "r") as f:
    genes = [line.split(", ")[0] for line in f.read().split("\n") if (len(line)>0)]
state_len = 10 # number of patient states to generate
states = pd.DataFrame(
  [np.random.choice([0,1], p=[0.5,0.5], size=len(genes)).tolist() for _ in range(state_len)]
  , columns=genes, index=["Patient %d" % (i+1) for i in range(state_len)]).T
seed_number=12345
njobs=min(5,max(1,cpu_count()-2))
k=1
IM_params = {"seed": seed_number, "gene_inputs": genes, "gene_outputs": genes}
SIMU_params = {'nb_sims': 1000, 'rates': "fully_asynchronous", 'thread_count': njobs, 'depth': "constant_unitary"}

from NORDic.NORDic_PMR.functions import greedy
S, spreads = greedy(file_folder+"solution.bnet", k, states, IM_params, SIMU_params, save_folder=file_folder)
assert spreads.shape[0]==6

from subprocess import call
call("rm -rf "+file_folder, shell=True)
call("rm -f experiments.csv network.tsv", shell=True)from subprocess import call
call("rm -rf "+file_folder, shell=True)
call("rm -f experiments.csv network.tsv", shell=True)
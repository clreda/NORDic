#coding: utf-8

import imports

import pandas as pd
from subprocess import call as sbcall
import os

data_folder="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/NetworkOrientedRepurposingofDrugs/"
root_folder="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/MDD/"
file_folder=root_folder+"MDDMale_JURKAT/"
if ("MDDFemale_JURKAT/" in file_folder):
    module_fname=root_folder+"PourClemence/FEMALE_ME_129.txt"
if ("MDDMale_JURKAT/" in file_folder):
    module_fname=root_folder+"PourClemence/MALE_ME_48.txt"

sbcall("mkdir -p "+file_folder, shell=True)

## Create network accepted by NORDic
network = pd.read_csv(module_fname, index_col=0, sep=",")
network = network[[c for c in network.columns if (c!="Origin")]]
network.columns = ["preferredName_A", "preferredName_B", "score"]
network["sign"] = [(-1)**int(network.loc[i]["score"]<0) for i in network.index]
network["score"] = network["score"].abs()
network["directed"] = 1
network = network[["preferredName_A", "preferredName_B", "sign", "directed", "score"]]

network_fname = file_folder+"network.tsv"
network.to_csv(network_fname, index=None, sep="\t")

## Registration to databases
DisGeNET_credentials = data_folder+"tests/credentials_DISGENET.txt"
STRING_credentials = data_folder+"tests/credentials_STRING.txt"
LINCS_credentials = data_folder+"tests/credentials_LINCS.txt"
path_to_genes = None
network_fname = file_folder+"network.tsv"
cell_lines = ["JURKAT"]

## Parameters
seed_number=123456
from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2)
taxon_id=9606 # human species
disease_cids=["C1269683"] # major depressive disorder

## Information about the disease
DISGENET_args = {"credentials": DisGeNET_credentials, "disease_cids": disease_cids}

## Selection of parameters relative to the prior knowledge network
STRING_args = {"credentials": STRING_credentials, "score": 0}
EDGE_args = {"tau": 0, "filter": True, "connected": True}

## Selection of parameters relative to experimental constraints
LINCS_args = {"path_to_lincs": data_folder+"lincs/", "credentials": LINCS_credentials,
        "cell_lines": cell_lines, "thres_iscale": 0}
SIG_args = {"bin_thres": 0.5}

## Selection of parameters relative to the inference of networks
BONESIS_args = {"limit": 1, "exact": True, "max_maxclause": 3}

## Advanced
DESIRABILITY = {"DS": 3, "CL": 3, "Centr": 3, "GT": 1}

if ("MDDFemale_JURKAT/" in file_folder):
    STRING_args.update({"score": 0, "beta": 1})
    EDGE_args.update({"tau": 0, "filter": False, "connected": False})
    SIG_args.update({"bin_thres": 0.27})
    LINCS_args.update({"thres_iscale": 0.2})
    BONESIS_args.update({"limit": 1000, "exact": False, "max_maxclause": 20})

if ("MDDMale_JURKAT" in file_folder):
    STRING_args.update({"score": 0, "beta": 1})
    EDGE_args.update({"tau": 0, "filter": False, "connected": False})
    SIG_args.update({"bin_thres": 0.11}) 
    LINCS_args.update({"thres_iscale": 0})
    BONESIS_args.update({"limit": 50, "exact": True, "max_maxclause": 10}) 

from NORDic.NORDic_NI.functions import network_identification

MME48_solution = network_identification(file_folder, taxon_id,
            path_to_genes=None, disgenet_args=DISGENET_args,
            string_args=STRING_args, lincs_args=LINCS_args, edge_args=EDGE_args,
            sig_args=SIG_args, bonesis_args=BONESIS_args, weights=DESIRABILITY,
            seed=seed_number, network_fname=network_fname, njobs=njobs, force_experiments=True, accept_nonRNA=True)

if (os.path.exists(file_folder+"solution.bnet")):
    with open(file_folder+"solution.bnet", "r") as f:
        network = f.read().split("\n")
    print(network)

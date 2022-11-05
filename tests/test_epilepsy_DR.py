#coding:utf-8

import os
from subprocess import call as sbcall
import pandas as pd

LINCS_args = {
        "path_to_lincs": "../lincs/",
        "credentials": "credentials_LINCS.txt",
        "selection": "distil_ss", 
        "nsigs": 2,
}

## Download epilepsy-related data
from download_Refractory_Epilepsy_Data import get_EPILEPSY_phenotypes, file_folder, path_to_phenotypes, dataset_folder
get_EPILEPSY_phenotypes(LINCS_args["path_to_lincs"])

solution_fname=file_folder+"solution.bnet"
drug_dataset_fname=dataset_folder+"small_drug_dataset.txt"
file_folder, save_folder=file_folder+"DR/", file_folder+"DS/"
sbcall("mkdir -p "+file_folder, shell=True)

## 1. Get drug names
with open(drug_dataset_fname, "r") as f:
    drug_names = f.read().split("\n")[:-1]

## 2. Get drug signatures (which will guide drug testing)
from NORDic.NORDic_DS.get_drug_signatures import drugname2pubchem, compute_drug_signatures_L1000

signature_fname=save_folder+"reduced_signatures.csv"
if (not os.path.exists(signature_fname)):
    if (not os.path.exists(pubchem_fname)):
        pubchem_df = pd.DataFrame({"PubChemCID": drugname2pubchem(drug_names, LINCS_args)})
        pubchem_df.to_csv(pubchem_fname)
    pubchem_df = pd.read_csv(pubchem_fname, index_col=0).dropna()
    pubchem_cids = list(pubchem_df["PubChemCID"].astype(int))
    signatures = compute_drug_signatures_L1000(list(pd.read_csv(pubchem_fname, index_col=0).dropna()["PubChemCID"].astype(int)), LINCS_args)
    signatures.columns = [list(pubchem_df.loc[pubchem_df["PubChemCID"]==p].index)[0] for p in signatures.columns]
    signatures.to_csv(signature_fname)
signatures = pd.read_csv(signature_fname, index_col=0)

## 3. Get drug targets (for the simulation)
from NORDic.NORDic_DS.get_drug_targets import retrieve_drug_targets

## All MINERVA maps are listed at https://minerva.pages.uni.lu/doc/
## To download the data from DrugBank, register (for free) to DrugBank and download the complete database (full_database.xml) and protein database
target_args = {
    "DrugBank": {"path_to_drugbank": "../DrugBank/", "drug_fname": "COMPLETE DATABASE/full database.xml", 
        "target_fname": "PROTEIN IDENTIFIERS/Drug Target Identifiers/all.csv"},
    "LINCS": LINCS_args,
}

## Get M30 genes
with open(solution_fname, "r") as f:
    network = str(f.read())
genes = [x.split(" <- ")[0] for x in network.split("\n")[:-1]]

targets = retrieve_drug_targets(save_folder, drug_names, TARGET_args=target_args, gene_list=genes, quiet=True)
targets = targets[signatures.columns]

## 4. Get patient phenotypes and model to classify states
from NORDic.NORDic_DS.functions import compute_frontier
from copy import deepcopy
binary_phenotypes = pd.read_csv(path_to_phenotypes, index_col=0)
samples = deepcopy(binary_phenotypes.loc["annotation"])
binary_phenotypes[binary_phenotypes>0] = 1
binary_phenotypes[binary_phenotypes<0] = -1
binary_phenotypes[binary_phenotypes==0] = 0
dfdata = binary_phenotypes.loc[list(set([g for g in genes if (g in binary_phenotypes.index)]))]
frontier = compute_frontier(dfdata, samples, quiet=False)
patients = dfdata[[c for ic, c in enumerate(dfdata.columns) if (samples[ic]==2)]]

targets = targets.loc[patients.index]

## 5. Run adaptive testing
from NORDic.NORDic_DR.functions import adaptive_testing
from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2)

SIMU_params = {
    'nb_sims': 1000,
    'rates': "fully_asynchronous",
    'thread_count': njobs,
    'depth': "constant_unitary",
}

## Test if the model is almost linear (with heatmap)

BANDIT_args = {
    'bandit': 'LinGapE', #MisLid #type of algorithm, (greedy) LinGapE is faster but more prone to errors (assumes that the model is linear)
    'seed': 0,
    'delta': 0.1, #error rate
    'nsimu': 500, #number of repeats
    'm': 4, #number of recommendations to make

    'c': 1, #parameter to tune for MisLid (if the model is linear, set to 0)
    'sigma': 1,
    'beta': "heuristic",
    'epsilon': 0,
    'tracking_type': "D",
    'gain_type': 'empirical',
    'learner': "AdaHedge",
    'subsample': 0,
    'geometric_factor': 1.3,
}

if (not os.path.exists(file_folder+"recommendation.csv")):
    recommendation = adaptive_testing(solution_fname, signatures, targets, frontier, 
		patients, SIMU_params, BANDIT_args, reward_fname=save_folder+"scores.csv", quiet=False).T
    recommendation.to_csv(file_folder+"recommendation.csv")
recommendation = pd.read_csv(file_folder+"recommendation.csv", index_col=0)
recommendation = recommendation.loc[recommendation["Frequency"]>0]

## 6. Compare to ground truth scores (1: treating, -1: mimicking the disease)
ground_truth_scores = pd.read_csv(dataset_folder+"scores.csv", index_col=1)[["score"]]
ground_truth_scores.columns = ["Ground Truth"]

recommendation = recommendation.join(ground_truth_scores, how="inner")
print(recommendation)

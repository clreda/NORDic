#coding:utf-8

import imports

import pandas as pd
import os
from subprocess import call as sbcall
from copy import deepcopy

LINCS_args = {
        "path_to_lincs": "../lincs/",
        "credentials": "credentials_LINCS.txt",
        "selection": "distil_ss", 
        "nsigs": 2,
}

## Download epilepsy-related data
from download_Refractory_Epilepsy_Data import get_EPILEPSY_phenotypes, file_folder, path_to_phenotypes, dataset_folder
get_EPILEPSY_phenotypes(LINCS_args["path_to_lincs"])

from NORDic.NORDic_DS.get_drug_signatures import drugname2pubchem, compute_drug_signatures_L1000
from NORDic.UTILS.LINCS_utils import binarize_via_CD
from NORDic.NORDic_DS.functions import baseline, compute_metrics, simulate
from NORDic.NORDic_DS.get_drug_targets import retrieve_drug_targets

seed_number=0
solution_fname=file_folder+"solution.bnet"
drug_dataset_fname=dataset_folder+"smallest_drug_dataset.txt"
file_folder=file_folder+"DS/"
sbcall("mkdir -p "+file_folder, shell=True)

## Get drug names
with open(drug_dataset_fname, "r") as f:
    drug_names = f.read().split("\n")[:-1]

## Get M30 genes
with open(solution_fname, "r") as f:
    network = str(f.read())
genes = [x.split(" <- ")[0] for x in network.split("\n")[:-1]]

############################
## BASELINE (L1000 CDS^2) ##
############################

pubchem_fname=file_folder+"pubchemcids_"+drug_dataset_fname.split(dataset_folder)[-1].split(".txt")[0]+".csv"
signature_fname=file_folder+"signatures_"+drug_dataset_fname.split(dataset_folder)[-1].split(".txt")[0]+".csv"
diffpheno_fname=file_folder+"diffpheno_"+drug_dataset_fname.split(dataset_folder)[-1].split(".txt")[0]+".csv"

## 1. Get drug signatures
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

## 2. Get differential phenotype
phenotypes = pd.read_csv(path_to_phenotypes, index_col=0)
if (not os.path.exists(diffpheno_fname)):
    differential_phenotype = binarize_via_CD(phenotypes.loc[[idx for idx in phenotypes.index if (idx!="annotation")]], samples=list(phenotypes.loc["annotation"]), binarize=0, nperm=int(1e4))
    differential_phenotype.to_csv(diffpheno_fname)
differential_phenotype = pd.read_csv(diffpheno_fname, index_col=0)

print(signatures.join(differential_phenotype, how="inner"))

## 3. Ground truth scores (1: treating, -1: mimicking the disease)
ground_truth_scores = pd.read_csv(dataset_folder+"scores.csv", index_col=1)[["score"]]
ground_truth_scores.columns = ["Ground Truth"]

## 4. Compute scores
scores = baseline(signatures, differential_phenotype, is_binary=True)
scores = scores.sort_values(by="Cosine Score", ascending=False)
scores = scores.join(ground_truth_scores, how="inner")
print(scores)

## 5. Evaluate the accuracy
res_di = compute_metrics(scores["Cosine Score"], scores["Ground Truth"], K=[2,5,10], nperms=100)
print(pd.DataFrame({"Baseline": res_di}))

############################
## SIMULATOR              ##
############################

## All MINERVA maps are listed at https://minerva.pages.uni.lu/doc/
## To download the data from DrugBank, register (for free) to DrugBank and download the complete database (full_database.xml) and protein database
target_args = {
    "DrugBank": {"path_to_drugbank": "../DrugBank/", "drug_fname": "COMPLETE DATABASE/full database.xml", "target_fname": "PROTEIN IDENTIFIERS/Drug Target Identifiers/all.csv"},
    "LINCS": LINCS_args,
}

from multiprocessing import cpu_count
njobs=1#max(1,cpu_count()-2)

SIMU_params = {
    'nb_sims': 1000,
    'rates': "fully_asynchronous",
    'thread_count': njobs,
    'depth': "constant_unitary",
}

## 1. Get drug targets
targets = retrieve_drug_targets(file_folder, drug_names, TARGET_args=target_args, gene_list=genes, quiet=True)
targets = targets.drop_duplicates() # restrict to genes in M30
print(targets)

## 2. Get binary patient/control phenotypes
binary_phenotypes = deepcopy(phenotypes)
binary_phenotypes[binary_phenotypes>0] = 1
binary_phenotypes[binary_phenotypes<0] = -1
binary_phenotypes[binary_phenotypes==0] = 0
binary_phenotypes.loc["annotation"] = phenotypes.loc["annotation"]
print(binary_phenotypes)

## 3. Score the effect of each drug in each patient phenotype using the network
scores = simulate(solution_fname, targets, phenotypes, SIMU_params, nbseed=0)
scores = pd.DataFrame(scores.mean(axis=0), index=scores.index, columns=["Simulator Score"])
scores = scores.sort_values(by="Simulator Score", ascending=False)
scores = scores.join(ground_truth_scores, how="inner")
print(scores)

## 4. Evaluate the accuracy
res_di = compute_metrics(scores["Simulator Score"], scores["Ground Truth"], K=[2,5,10], nperms=100)
print(pd.DataFrame({"Method": res_di}))

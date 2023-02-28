#coding:utf-8

import os
from subprocess import call as sbcall
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
drug_dataset_fname=dataset_folder+"small_drug_dataset.txt" ## DO IT ON THE WHOLE SUBSET?
file_folder, save_folder=file_folder+"DR/", file_folder+"DS/"
sbcall("mkdir -p "+file_folder, shell=True)

## 1. Get drug names
with open(drug_dataset_fname, "r") as f:
    drug_names = f.read().split("\n")[:-1]

## 2. Get drug signatures (which will guide drug testing)
from NORDic.NORDic_DS.get_drug_signatures import drugname2pubchem, compute_drug_signatures_L1000

n_components=9
X_fname=save_folder+"reduced_signatures.csv"
pubchem_fname=save_folder+"pubchemcids_"+drug_dataset_fname.split(dataset_folder)[-1].split(".txt")[0]+".csv"
signature_fname=save_folder+"signatures_"+drug_dataset_fname.split(dataset_folder)[-1].split(".txt")[0]+".csv"
if (not os.path.exists(signature_fname)):
    if (not os.path.exists(pubchem_fname)):
        pubchem_df = pd.DataFrame({"PubChemCID": drugname2pubchem(drug_names, LINCS_args)})
        pubchem_df.to_csv(pubchem_fname)
    pubchem_df = pd.read_csv(pubchem_fname, index_col=0).dropna()
    pubchem_cids = list(pubchem_df["PubChemCID"].astype(int))
    signatures = compute_drug_signatures_L1000(list(pd.read_csv(pubchem_fname, index_col=0).dropna()["PubChemCID"].astype(int)), LINCS_args)
    signatures.columns = [list(pubchem_df.loc[pubchem_df["PubChemCID"]==p].index)[0] for p in signatures.columns]
    signatures.to_csv(signature_fname)
    X = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(signatures.values.T)).T
    X = pd.DataFrame(X, index=["PCA%d" % (i+1) for i in range(n_components)], columns=signatures.columns)
    X.to_csv(X_fname)
X = pd.read_csv(X_fname, index_col=0)
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
score = lambda attrs : (frontier.predict(attrs.values.T)==1).astype(int)  #classifies into 1:control, 2:patient
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
    'bandit': 'LinGapE', #type of algorithm, (greedy) LinGapE is faster but more prone to errors (assumes that the model is linear)
    'seed': 0,
    'delta': 0.1, #error rate
    'nsimu': 500, #number of repeats
    'm': 4, #number of recommendations to make

    'c': 0, #nonnegative parameter to tune for MisLid (if the model is linear, set to 0
    ## To speed up the algorithm, decrease
    ## To ensure correctness of the recommendation, increase
    'sigma': 1,
    'beta': "heuristic",
    'epsilon': 0.001,
    'tracking_type': "D",
    'gain_type': "empirical",
    'learner': "AdaHedge"
}

## 6. Compare to ground truth scores (1: treating, -1: mimicking the disease)
ground_truth_scores = pd.read_csv(dataset_folder+"scores.csv", index_col=1)[["score"]]
ground_truth_scores.columns = ["Ground Truth"]

drug_columns = [s for s in list(signatures.columns) if (s in ground_truth_scores.index)]
rewards = pd.read_csv(save_folder+"scores.csv", index_col=0)[drug_columns]
df = pd.DataFrame(rewards.mean(axis=0).sort_values(ascending=False), columns=["Score"])
df = df.join(ground_truth_scores, how="inner")
import numpy as np
df["Gap"] = [np.nan]+[np.round(a-list(df["Score"])[i+1],3) for i,a in enumerate(list(df["Score"])[:-1])]
print(df)

## Use a subset of drugs
drug_columns = [s for s in list(signatures.columns) if (s in ground_truth_scores.index)]
assert all([(d in signatures.columns and d in targets.columns) for d in drug_columns])
BANDIT_args.update({'bandit': 'LinGapE', 'nsimu': 1, 'm': 2})
assert BANDIT_args["m"]<len(drug_columns)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
## Check that d < K and that features are not collinear when using MisLid
if ((BANDIT_args["bandit"]=="MisLid") and (len(drug_columns)!=signatures.shape[1])):
    pca = PCA(n_components=min(X.shape[0],len(drug_columns)))
    scaler = StandardScaler()
    sigs = signatures[drug_columns]
    X = pca.fit_transform(scaler.fit_transform(sigs.values.T)).T
    X = pd.DataFrame(X, index=["PCA%d" % (i+1) for i in range(X.shape[0])], columns=drug_columns)
else:
    X = X[drug_columns]

targets = targets[drug_columns]
rewards_fname=save_folder+"scores.csv"
if (not os.path.exists(file_folder+"recommendation.csv")):
    recommendation = adaptive_testing(solution_fname, X, targets, score, 
		patients, SIMU_params, BANDIT_args, reward_fname=rewards_fname, quiet=False).T
    recommendation.to_csv(file_folder+"recommendation.csv")
recommendation = pd.read_csv(file_folder+"recommendation.csv", index_col=0)
recommendation = recommendation.loc[recommendation["Frequency"]>0]

recommendation = recommendation.join(ground_truth_scores, how="inner")
print(recommendation)

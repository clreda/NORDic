#coding:utf-8

import imports

import pandas as pd
import os
from subprocess import call as sbcall
from copy import deepcopy
from tqdm import tqdm
import numpy as np

LINCS_args = {
        "path_to_lincs": "../lincs/",
        "credentials": "credentials_LINCS.txt",
        "selection": "distil_ss", 
        "nsigs": 2,
}

## Download epilepsy-related data
from download_Refractory_Epilepsy_Data import get_EPILEPSY_phenotypes, file_folder, path_to_phenotypes, dataset_folder
get_EPILEPSY_phenotypes(LINCS_args["path_to_lincs"])
file_folder="refractory_epilepsy2/"

from NORDic.NORDic_DS.get_drug_signatures import drugname2pubchem, compute_drug_signatures_L1000
from NORDic.UTILS.LINCS_utils import binarize_via_CD
from NORDic.NORDic_DS.functions import baseline, compute_metrics, simulate, compute_frontier
from NORDic.NORDic_DS.get_drug_targets import retrieve_drug_targets

from NORDic.UTILS.utils_state import binarize_experiments ##

seed_number=0
solution_fname=file_folder+"solution.bnet"
drug_dataset_fname=dataset_folder+"small_drug_dataset.txt"
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
    dfsdata = phenotypes.loc[[idx for idx in phenotypes.index if (idx!="annotation")]]
    differential_phenotype = binarize_via_CD(dfsdata, samples=list(phenotypes.loc["annotation"]), binarize=0, nperm=int(1e4))
    differential_phenotype.to_csv(diffpheno_fname)
differential_phenotype = pd.read_csv(diffpheno_fname, index_col=0)

print(signatures.join(differential_phenotype, how="inner").head())

## 3. Compute scores
scores = baseline(signatures, differential_phenotype, is_binary=False)
scores = scores.sort_values(by="Cosine Score", ascending=False)
print(scores.head())

############################
## SIMULATOR              ##
############################

## All MINERVA maps are listed at https://minerva.pages.uni.lu/doc/
## To download the data from DrugBank, register (for free) to DrugBank and download the complete database (full_database.xml) and protein database
target_args = {
    "DrugBank": {
        "path_to_drugbank": "../DrugBank/", 
        "drug_fname": "COMPLETE DATABASE/full database.xml", 
        "target_fname": "PROTEIN IDENTIFIERS/Drug Target Identifiers/all.csv"
    },
    "LINCS": LINCS_args,
}

from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2)

SIMU_params = {
    'nb_sims': 1000,
    'rates': "fully_asynchronous",
    'thread_count': njobs,
    'depth': "constant_unitary",
}

## 1. Get drug targets
targets = retrieve_drug_targets(file_folder, drug_names, TARGET_args=target_args, gene_list=genes, quiet=False)
targets = targets.drop_duplicates() # restrict to genes in M30

from numpy import nan
targets[targets==0] = nan
targets[targets<0] = 0
targets[targets>0] = 1
print(targets.head())

## 2. Get binary patient/control phenotypes
binary_phenotypes = deepcopy(phenotypes)
binary_phenotypes[binary_phenotypes>0] = 1
binary_phenotypes[binary_phenotypes<0] = -1
binary_phenotypes[binary_phenotypes==0] = 0
binary_phenotypes.loc["annotation"] = phenotypes.loc["annotation"].astype(int)
print(binary_phenotypes.head())

with open(solution_fname, "r") as f:
    network = str(f.read())
if (", " in network):
    genes = [x.split(", ")[0] for x in network.split("\n") if (len(x)>0)]
else:
    genes = [x.split(" <- ")[0] for x in network.split("\n") if (len(x)>0)]

## Model/score to identify attractors
dfdata = binary_phenotypes.loc[list(set([g for g in genes if (g in binary_phenotypes.index)]))]
samples = binary_phenotypes.loc["annotation"]
patients = dfdata[[c for c in dfdata if (samples[c]==2)]]
frontier = compute_frontier(dfdata, samples)
score = lambda attrs : (frontier.predict(attrs.values.T)==1).astype(int)  #classifies into 1:control, 2:patient

## 3. Score the effect of each drug in each patient phenotype using the network
scores_fname = file_folder+"scores_targets.csv"
if (not os.path.exists(scores_fname)):
    scores = simulate(solution_fname, targets, patients, score, simu_params=SIMU_params, nbseed=0)
    scores.to_csv(scores_fname)
scores_all = pd.read_csv(scores_fname, index_col=0).T

scores = pd.DataFrame(scores_all.mean(axis=1), columns=["Simulator Score"])
scores = scores.sort_values(by="Simulator Score", ascending=False)
print(scores.head())

############################
## VISUALIZATION          ##
############################
from NORDic.UTILS.utils_plot import plot_boxplots, plot_heatmap, plot_roc_curve, plot_precision_recall

## Boxplots
plot_boxplots(scores, scores_all, ground_truth=ground_truth_scores, fname=file_folder+"boxplots.pdf")

## Heatmap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans
import numpy as np
true = np.ravel(ground_truth_scores.join(scores, how="outer")[ground_truth_scores.columns[0]].loc[signatures.columns].fillna(0).astype(int).values)
argmax_ncomponents, max_ari = 0, -float("inf")
ncomponents_lst = range(2, 21, 1)
nclusters=len(np.unique(ground_truth_scores.values))
for ncomponents in tqdm(ncomponents_lst):
    X = PCA(n_components=ncomponents, random_state=0).fit_transform(StandardScaler().fit_transform(signatures.values.T)).T
    X = pd.DataFrame(X, index=["PCA%d" % (i+1) for i in range(ncomponents)], columns=signatures.columns)
    clust = KMeans(n_clusters=nclusters, random_state=0).fit(X.T, true)
    ari = ARI(true, clust.labels_)
    if (ari>max_ari):
        max_ari = ari
        argmax_ncomponents = ncomponents
print("%d components: ARI=%.5f (%d clusters)" % (argmax_ncomponents, ari, 2))
if (len(ncomponents_lst)>1):
    X = PCA(n_components=argmax_ncomponents).fit_transform(StandardScaler().fit_transform(signatures.values.T)).T
    X = pd.DataFrame(X, index=["PCA%d" % (i+1) for i in range(argmax_ncomponents)], columns=signatures.columns)
    plot_heatmap(X, ground_truth=ground_truth_scores, fname=file_folder+"heatmap.pdf")
X.to_csv(file_folder+"reduced_signatures.csv")

## Take into account possible variation to patients
plot_roc_curve(scores_["Simulator Score"], scores_all, scores_["Ground Truth"], fname=file_folder+"ROC.pdf")
plot_precision_recall(scores_["Simulator Score"], scores_all, scores_["Ground Truth"], beta=beta, fname=file_folder+"PRC.pdf")

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

path_to_full_phenotypes=dataset_folder+"pool_full.csv" ##
signature_full_fname=dataset_folder+"signatures_L1000CDS2.csv"## DRUG SIGNATURES ON THE FULL GENOME
diffpheno_full_fname=dataset_folder+"full_bin_compare.csv" ## DIFFERENTIAL PHENOTYPE ON THE FULL GENOME
path_to_phenotypes=dataset_folder+"pool.csv" ##
signature_fname=dataset_folder+"signatures.csv"#"signatures_drugs_binarized2_m30.csv" ##
diffpheno_fname=dataset_folder+"pool.csv" ##

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

signatures = pd.read_csv(signature_full_fname, index_col=0) ##

## 2. Get differential phenotype
phenotypes = pd.read_csv(path_to_phenotypes, index_col=0)
if (not os.path.exists(diffpheno_fname)):
    dfsdata = phenotypes.loc[[idx for idx in phenotypes.index if (idx!="annotation")]]
    differential_phenotype = binarize_via_CD(dfsdata, samples=list(phenotypes.loc["annotation"]), binarize=0, nperm=int(1e4))
    differential_phenotype.to_csv(diffpheno_fname)
differential_phenotype = pd.read_csv(diffpheno_fname, index_col=0)

differential_phenotype = pd.read_csv(diffpheno_full_fname, index_col=0).dropna() ##

print(signatures.join(differential_phenotype, how="inner").head())

## 3. Ground truth scores (1: treating, -1: mimicking the disease)
#ground_truth_scores = pd.read_csv(dataset_folder+"scores_%s.csv" % drug_dataset_fname.split(dataset_folder)[-1].split(".txt")[0], index_col=1)[["score"]]
ground_truth_scores = pd.read_csv(dataset_folder+"scores.csv", index_col=1)[["score"]] ##
ground_truth_scores_convert_names = pd.read_csv(dataset_folder+"scores.csv", index_col=0)[["drug_name"]] ##
ground_truth_scores.columns = ["Ground Truth"]

signatures.columns = [ground_truth_scores_convert_names.loc[int(i)]["drug_name"] for i in signatures.columns]

## 4. Compute scores
scores = baseline(signatures, differential_phenotype, is_binary=False)

#scores in the thesis (sanity check)
#scores_ = pd.read_csv(dataset_folder+"treatments_scores_baseline_THESIS.csv", index_col=0) ##
#scores_.columns = ["Cosine Score2"] ##
#scores_.index = [ground_truth_scores_convert_names.loc[int(i)]["drug_name"] for i in scores_.index] ##
#print(scores.join(scores_, how="inner"))

scores = scores.join(ground_truth_scores, how="inner")
scores = scores.sort_values(by="Cosine Score", ascending=False)
print(scores.head())

## 5. Evaluate the accuracy
res_di = compute_metrics(scores["Cosine Score"], scores["Ground Truth"], K=[2,3,5,10], thres=0.)#thres=0.5, nperms=100)
print(pd.DataFrame({"Baseline": res_di}))

#scores_ = scores_.join(ground_truth_scores, how="inner") ##
#scores_ = scores_.sort_values(by="Cosine Score2", ascending=False) ##
#res_di = compute_metrics(scores_["Cosine Score2"], scores_["Ground Truth"], K=[2,3,5,10], thres=0.)#thres=0.5, nperms=100) ##
#print(pd.DataFrame({"Baseline_old": res_di})) ##

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

#targets = pd.read_csv(signature_fname, index_col=0) ##
#targets.columns = [ground_truth_scores_convert_names.loc[int(i)]["drug_name"] for i in targets.columns] ##

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

ctrl, ptt = [[c for c in binary_phenotypes.columns if (float(binary_phenotypes.loc["annotation"][c])==v)] for v in [1,2]]

pool = pd.read_csv(path_to_phenotypes, index_col=0) ##
pool = pool.loc[~pool.index.duplicated()] ##
samples = list(pool.loc["annotation"]) ##
binary_phenotypes = binarize_experiments(pool.iloc[:-1]) ##
binary_phenotypes.loc["annotation"] = samples ##

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

#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#import random
#random.seed(seed_number)
#np.random.seed(seed_number)
#n_components=2
#frontier, scaler = PCA(n_components=n_components), StandardScaler()
#ctrl_profiles = binary_phenotypes[ctrl]
#for i, j in np.argwhere(ctrl_profiles.values==0):
#    ctrl_profiles.loc[list(ctrl_profiles.index)[i]][list(ctrl_profiles.columns)[j]] = -1
#ptt_profiles = binary_phenotypes[ptt]
#for i, j in np.argwhere(ptt_profiles.values==0):
#    ptt_profiles.loc[list(ptt_profiles.index)[i]][list(ptt_profiles.columns)[j]] = -1
#ones_genes = pd.DataFrame(np.ones((len(genes),1)), index=genes, columns=["NULL"])
#CP = ctrl_profiles.join(ptt_profiles, how="outer")
#CP = CP.loc[~CP.index.duplicated()]
#CP = CP.loc[[g for g in genes if (g in CP.index)]]
### Fit PCA on the genes in the network
#X = ones_genes.join(CP, how="outer")[ctrl+ptt].T.fillna(0).values
#scaler.fit(X)
#frontier.fit(scaler.transform(X))
#XX, SS = ctrl_profiles.join(ptt_profiles,how="inner"), [2]*ctrl_profiles.shape[1]+[1]*ptt_profiles.shape[1]
#compare_to = binarize_via_CD(XX, samples=SS, binarize=0, nperm=10000)
#compare_to.columns = ["CD[Healthy||Patients]"]
#compare_to.to_csv(file_folder+"bin_compare.csv")
#CC = compare_to.loc[~compare_to.index.duplicated()]
#CC = CC.loc[[g for g in genes if (g in CC.index)]]
#sig_cd = ones_genes.join(CC, how="outer")[CC.columns].T.fillna(0).values
#sig_pca = frontier.transform(scaler.transform(sig_cd))
#a, b = sig_pca.flatten().tolist()[:2]

#def score_nonorient(attrs, frontier_normal=[sig_pca,sig_cd][int(n_components==0)], orient=1):
#    attrs = attrs.loc[~attrs.index.duplicated()]
#    attrs = attrs.loc[[g for g in genes if (g in attrs.index)]]
#    A = ones_genes.join(attrs, how="outer")
#    A[A==0] = -1
#    A = A.fillna(0)[attrs.columns]
#    if (n_components==0):
#        X_p = A.T.values
#    else:
#        X_p = frontier.transform(scaler.transform(A.T.values))
#    ## Projection onto Frontier
#    #proj_X_p = X_p-(X_p.dot(frontier_normal.T)/(np.linalg.norm(frontier_normal, 2)**2))*frontier_normal
#    proj_X_p = (X_p.dot(frontier_normal.T)/(np.linalg.norm(frontier_normal, 2)**2))*frontier_normal
#    #scores = orient*np.sum([(X_p[:,i]-proj_X_p[:,i]) for i in range(X_p.shape[1])], axis=0) #signed l1-norm + orientation of control samples to decreasing values of x,y
#    scores = orient*(X_p[:,0]-proj_X_p[:,0]+X_p[:,1]-proj_X_p[:,1])##equivalent
#    return scores
#scores_ctrl_sign = score_nonorient(ctrl_profiles, orient=1)
#def score(attrs):
#    return score_nonorient(attrs, orient=np.sign(np.median(scores_ctrl_sign)))

## 3. Score the effect of each drug in each patient phenotype using the network
scores_fname = file_folder+"scores_targets.csv" #"scores_signatures_PCA.csv"
if (not os.path.exists(scores_fname)):
    scores = simulate(solution_fname, targets, patients, score, simu_params=SIMU_params, nbseed=0)
    scores.to_csv(scores_fname)
scores_all = pd.read_csv(scores_fname, index_col=0).T

scores = pd.DataFrame(scores_all.mean(axis=1), columns=["Simulator Score"])
scores_ = scores.join(ground_truth_scores, how="inner")
scores_ = scores_.sort_values(by="Simulator Score", ascending=False)
print(scores_.head())

## 4. Evaluate the accuracy
beta=1
res_di = compute_metrics(scores_["Simulator Score"], scores_["Ground Truth"], K=[2,3,5,10], beta=beta, thres=0.5, nperms=100)
print(pd.DataFrame({"Method": res_di}))

#scores = pd.read_csv(dataset_folder+"treatments_scores_FINAL2.csv", index_col=0) ##
#scores.index = [ground_truth_scores_convert_names.loc[int(i)]["drug_name"] for i in scores.index] ##
#scores = pd.DataFrame(scores.mean(axis=1), columns=["Simulator Score"]) ##
#scores_ = scores.join(ground_truth_scores, how="inner") ##
#scores_ = scores_.sort_values(by="Simulator Score", ascending=False) ##
#print(scores_.head())
#res_di = compute_metrics(scores_["Simulator Score"], scores_["Ground Truth"], K=[2,3,5,10], beta=beta, thres=0.5, nperms=100) ##
#print(pd.DataFrame({"Old one": res_di})) ##

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

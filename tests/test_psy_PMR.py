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

data_folder="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/NetworkOrientedRepurposingofDrugs/"
root_folder="/media/kali/1b80f30d-2803-4260-a792-9ae206084252/Code/M30/MDD/"
file_folder=root_folder+"MDDFemale_JURKAT/"
if ("MDDFemale_JURKAT/" in file_folder):
    sex="F"
if ("MDDMale_JURKAT/" in file_folder):
    sex="M"
module_path=root_folder+"PourClemence/"

seed_number=0
k=1
solution_fname=file_folder+"solution.bnet"

from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2)

from NORDic.NORDic_PMR.functions import greedy
from NORDic.UTILS.utils_state import binarize_experiments
from NORDic.UTILS.utils_grn import load_grn
from NORDic.UTILS.utils_data import request_biodbnet

## 1. Get genes
with open(solution_fname, "r") as f:
    network = str(f.read())
genes = [x.split(" <- ")[0] for x in network.split("\n")]
gene_outputs = [x.split(" <- ")[0] for x in network.split("\n") if (x.split(" <- ")[1] not in [x.split(" <- ")[0], "0", "1"])]

## 2. Create states (mRNA, miRNA, conditions)
if (not os.path.exists(root_folder+"MDD_omics_"+sex+".csv")):
    cmd_contents = "; ".join([
        "MDD_omics <- readRDS(\""+module_path+"ALL_MDD_omic_miRNAmRNADNAm.RDS\")",
        "df <- merge(x=MDD_omics$miRNA, y=MDD_omics$mRNA, by=\"row.names\", all=TRUE)", 
        "rownames(df) <- rownames(MDD_omics$miRNA)",
        "metadata <- read.csv(\""+module_path+"CovariatesCommon.csv\", sep=\";\", row.names=1)", 
        "metadata <- metadata[metadata$SEX==\""+sex+"\",]", 
        "df <- df[rownames(metadata),]",
        "df$annotation <- sapply(metadata[rownames(df),]$GROUP, function(x) 1+as.integer(x==\"MDD\"))",
        "df$Row.names <- NULL",
        "write.csv(t(df), \""+root_folder+"MDD_omics_"+sex+".csv\")",
    ])
    sbcall('R -e \''+cmd_contents+'\'', shell=True)
MDD_omics = pd.read_csv(root_folder+"MDD_omics_"+sex+".csv", index_col=0, dtype=str)
samples = list(MDD_omics.loc["annotation"].astype(int))
MDD_omics = MDD_omics.loc[[a for a in MDD_omics.index if (a not in ["annotation"])]].apply(pd.to_numeric)
MDD_omics = MDD_omics.loc[~MDD_omics.index.duplicated()]
states = binarize_experiments(MDD_omics, thres=0.5, method="binary").astype(int)
states = states[[c for ic, c in enumerate(states.columns) if (samples[ic]==2)]]
states.columns = ["State%d" % i for i in range(states.shape[1])]
states.index = [".".join(g.split("-")) for g in states.index]
if (not os.path.exists(root_folder+"matches.csv")):
    probes = request_biodbnet(["-".join(g.split(".")) for g in list(states.index)], from_="Ensembl Gene ID", to_="Gene Symbol", taxon_id=9606)
    probes.to_csv(root_folder+"matches.csv")
probes = pd.read_csv(root_folder+"matches.csv", index_col=0)
probes.loc["ENSG00000163156"] = "SCNM1"
probes.loc["ENSG00000137767"] = "SQOR"
probes.loc["ENSG00000234506"] = "LINC01506"
probes.loc["ENSG00000164032"] = "H2AZ1"
probes.index = [".".join(g.split("-")) for g in probes.index]
probes = probes.loc[(np.vectorize(lambda x : "miR" in x)(list(probes.index)))|(probes["Gene Symbol"]!="-")]
states = states.loc[probes.index]
states.index = [".".join(x.split("-")) if ("miR" in x) else ".".join(probes.loc[x]["Gene Symbol"].split("-")) for x in states.index]
states = states.loc[[".".join(g.split("-")) for g in genes if (".".join(g.split("-")) in states.index)]]
if ("MDDMale_JURKAT/" in file_folder):
    states.loc["SQRDL"] = states.loc["SQOR"]
else:
    states.loc["H2AFZ"] = states.loc["H2AZ1"]

states.index = ["/".join(g.split("-")) for g in states.index]
genes = [".".join(g.split("-")) for g in genes]
gene_outputs = [".".join(g.split("-")) for g in gene_outputs]

## 3. Convert to an accepted format for GRNs
grn_fname=file_folder+"grn.bnet"
if (not os.path.exists(grn_fname)):
    fname_ls = glob(solution_fname)
    assert len(fname_ls)>0
    sbcall("sed 's/ <- /, /g' "+fname_ls[0]+" | sed 's/-/./g' > "+grn_fname, shell=True)

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

S, spreads = greedy(grn_fname, k, states, IM_params, SIMU_params, save_folder=file_folder)

print("ANSWER:\n%s" % str(S))

print(spreads.sort_values(by=spreads.columns[0], ascending=False))

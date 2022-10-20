# coding:utf-8

import subprocess as sb
import pandas as pd
import numpy as np
import os

import imports

dataset_folder = "Refractory_Epilepsy_Data/"
path_to_genes = dataset_folder+"S1_Delahayeetal2016_M30genes.txt"
path_to_initial_states = dataset_folder+"EMTAB3123.csv"
file_folder="refractory_epilepsy/"

################################################
## Import M30 genes [Delahaye et al., 2016]   ##
################################################
if (not os.path.exists(path_to_genes)):
    m30_url="https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-016-1097-7/MediaObjects/13059_2016_1097_MOESM1_ESM.xlsx"
    gene_modules = pd.read_excel(m30_url, sheet_name=1)
    id_M30 = np.argwhere(np.array(list(gene_modules[gene_modules.columns[1]]))=="M30")[0,0]+2
    M30 = gene_modules.iloc[id_M30:(id_M30+320),:]
    M30[[M30.columns[1]]].to_csv(path_to_genes, header=None, index=None)

################################################
## Import initial states [Mirza et al., 2017] ##
################################################

## epileptic hippocampi 
## (private communication of normalized count matrix from the raw count data in ArrayExpress using limma R package)
## (correcting for batch effect using ComBat in SVA R package)
if (not os.path.exists(dataset_folder+"EMTAB3123.csv")):
    genes = list(pd.read_csv(path_to_genes, index_col=0).index)
    if (not os.path.exists(dataset_folder+"EMTAB_3123ensg.csv")):
        cmd_R = "load(\""+dataset_folder+"EMTAB_3123ensg.Rdata\");"
        cmd_R += "write.csv(expr, \""+dataset_folder+"EMTAB_3123ensg.csv\");"
        sb.call("R -e \'"+cmd_R+"\'", shell=True)
    data = pd.read_csv(dataset_folder+"EMTAB_3123ensg.csv", index_col=0)
    if (not os.path.exists(dataset_folder+"matches.csv")):
        from utils_data import request_biodbnet
        matches = request_biodbnet(list(data.index), from_="Ensembl Gene ID", to_="Gene Symbol", taxonId=taxon_id)
        matches.to_csv(dataset_folder+"matches.csv")
    matches = pd.read_csv(dataset_folder+"matches.csv", index_col=0)
    matches = matches.loc[matches["Gene Symbol"] != "-"]
    data = data.loc[matches.index]
    data.index = matches["Gene Symbol"]
    #data = data.loc[[g for g in genes if (g in data.index)]]
    sb.call("wget -c -qO "+dataset_folder+"E-MTAB-3123.sdrf.txt https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3123/E-MTAB-3123.sdrf.txt", shell=True)
    metadata = pd.read_csv(dataset_folder+"E-MTAB-3123.sdrf.txt",sep="\t",index_col=0)[["Characteristics[phenotype]"]]
    metadata.index = [idx.split(".txt")[0] for idx in metadata.index]
    metadata = metadata.loc[metadata["Characteristics[phenotype]"]=="case"]
    data = data[metadata.index]
    data = data.groupby(level=0).median()
    ## Convert and restrict gene symbols to those existing in LINCS L1000
    entrezgene_fname=dataset_folder+"pool_entrezgenes_ids.csv"
    import sys
    sys.path.insert(1,"utils/")
    sys.path.insert(1,"utils/credentials/")
    if (not os.path.exists(entrezgene_fname)):
        from utils_data import request_biodbnet
        probes = request_biodbnet([x for y in list(data.index) for x in y.split("; ")], from_="Gene Symbol and Synonyms", to_="Gene ID", taxonId=taxon_id, chunksize=100)
        probes.to_csv(entrezgene_fname)
    probes = pd.read_csv(entrezgene_fname,index_col=0)
    probes = probes[probes["Gene ID"]!="-"]
    probes = probes.loc[~probes.index.duplicated()]
    ## convert EntrezGene ids back to gene symbols in LINCS L1000
    from LINCS_utils import download_lincs_files, path_to_lincs
    gene_files, sig_files, _, _ = download_lincs_files(path_to_lincs, which_lvl=[3])
    probes_ids = pd.Index([x for y in list(probes["Gene ID"]) for x in y.split("; ")]).astype(int)
    gene_selection = {}
    for idf, sf in enumerate(sig_files):
        df = pd.read_csv(path_to_lincs+gene_files[idf], sep="\t", engine='python', index_col=0)
        df = df.loc[[s for s in df.index if (s in probes_ids)]]
        gene_selection.update({str(s): df.loc[s]["pr_gene_symbol"] for s in df.index})
    probes["Data Symbol"] = probes.index
    probes.index = probes["Gene ID"]
    probes = probes.loc[[p for p in probes.index if (any([y in gene_selection for y in p.split("; ")]))]]
    probes["Gene Symbol"] = [list(sorted([gene_selection[y] for y in p.split("; ") if (y in gene_selection)], key=lambda x : len(x)))[0] for p in probes.index]
    probes.index = probes["Data Symbol"]
    probes = probes.loc[~probes.index.duplicated()]
    data = data.loc[[d for d in data.index if (any([y in probes.index for y in d.split("; ")]))]]
    data.index = ["; ".join([probes.loc[y]["Gene Symbol"] for y in set(d.split("; ")).intersection(list(probes.index))]) for d in data.index]
    data = data.loc[data.index!=""]
    data.index = [d.split("; ")[0] for d in data.index]
    data = data.loc[[g for g in genes if (g in data.index)]]
    ## mat: normalized gene expression matrix (rows=genes, columns=samples)
    data.to_csv(dataset_folder+"EMTAB3123.csv")

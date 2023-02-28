# coding:utf-8

import subprocess as sb
import pandas as pd
import numpy as np
import os

import imports

dataset_folder = "Refractory_Epilepsy_Data/"
path_to_genes = dataset_folder+"S1_Delahayeetal2016_M30genes.txt"
path_to_initial_states = dataset_folder+"EMTAB3123.csv"
path_to_phenotypes = dataset_folder+"EMTAB3123_control_case.csv"
file_folder="refractory_epilepsy/"
taxon_id=9606

################################################
## Import M30 genes [Delahaye et al., 2016]   ##
################################################
def get_EPILEPSY_genes():
    if (not os.path.exists(path_to_genes)):
        m30_url="https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-016-1097-7/MediaObjects/13059_2016_1097_MOESM1_ESM.xlsx"
        gene_modules = pd.read_excel(m30_url, sheet_name=1)
        id_M30 = np.argwhere(np.array(list(gene_modules[gene_modules.columns[1]]))=="M30")[0,0]+2
        M30 = gene_modules.iloc[id_M30:(id_M30+320),:]
        M30[[M30.columns[1]]].to_csv(path_to_genes, header=None, index=None)
    return None

################################################
## Import initial states [Mirza et al., 2017] ##
################################################

## epileptic hippocampi 
## (private communication of normalized count matrix from the raw count data in ArrayExpress using limma R package)
## (correcting for batch effect using ComBat in SVA R package)
def get_EPILEPSY_initial_states(path_to_lincs):
    if (not os.path.exists(dataset_folder+"EMTAB3123.csv")):
        genes = list(pd.read_csv(path_to_genes, index_col=0).index)
        if (not os.path.exists(dataset_folder+"EMTAB_3123ensg.csv")):
            cmd_R = "load(\""+dataset_folder+"EMTAB_3123ensg.Rdata\");"
            cmd_R += "write.csv(expr, \""+dataset_folder+"EMTAB_3123ensg.csv\");"
            sb.call("R -e \'"+cmd_R+"\'", shell=True)
        data = pd.read_csv(dataset_folder+"EMTAB_3123ensg.csv", index_col=0)
        if (not os.path.exists(dataset_folder+"matches.csv")):
            from NORDic.UTILS.utils_data import request_biodbnet
            matches = request_biodbnet(list(data.index), "Ensembl Gene ID", "Gene Symbol", taxon_id)
            matches.to_csv(dataset_folder+"matches.csv")
        matches = pd.read_csv(dataset_folder+"matches.csv", index_col=0)
        matches = matches.loc[matches["Gene Symbol"] != "-"]
        data = data.loc[matches.index]
        data.index = matches["Gene Symbol"]
        sb.call("wget -c -qO "+dataset_folder+"E-MTAB-3123.sdrf.txt https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3123/E-MTAB-3123.sdrf.txt", shell=True)
        metadata = pd.read_csv(dataset_folder+"E-MTAB-3123.sdrf.txt",sep="\t",index_col=0)[["Characteristics[phenotype]"]]
        metadata.index = [idx.split(".txt")[0] for idx in metadata.index]
        metadata = metadata.loc[metadata["Characteristics[phenotype]"]=="case"]
        data = data[metadata.index]
        data = data.groupby(level=0).median()
        ## Convert and restrict gene symbols to those existing in LINCS L1000
        entrezgene_fname=dataset_folder+"pool_entrezgenes_ids.csv"
        if (not os.path.exists(entrezgene_fname)):
            from NORDic.UTILS.utils_data import request_biodbnet
            probes = request_biodbnet([x for y in list(data.index) for x in y.split("; ")], "Gene Symbol and Synonyms", "Gene ID", taxon_id, chunksize=100)
            probes.to_csv(entrezgene_fname)
        probes = pd.read_csv(entrezgene_fname,index_col=0)
        probes = probes[probes["Gene ID"]!="-"]
        probes = probes.loc[~probes.index.duplicated()]
        ## convert EntrezGene ids back to gene symbols in LINCS L1000
        from NORDic.UTILS.LINCS_utils import download_lincs_files
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
    return None

##########################################################
## Import control/patient profiles [Mirza et al., 2017] ##
##########################################################

## (private communication of normalized count matrix from the raw count data in ArrayExpress using limma R package)
## (correcting for batch effect using ComBat in SVA R package)
def get_EPILEPSY_phenotypes(path_to_lincs):
    if (not os.path.exists(dataset_folder+"EMTAB3123_control_case.csv")):
        genes = list(pd.read_csv(path_to_genes, index_col=0).index)
        if (not os.path.exists(dataset_folder+"EMTAB_3123ensg.csv")):
            cmd_R = "load(\""+dataset_folder+"EMTAB_3123ensg.Rdata\");"
            cmd_R += "write.csv(expr, \""+dataset_folder+"EMTAB_3123ensg.csv\");"
            b.call("R -e \'"+cmd_R+"\'", shell=True)
        data = pd.read_csv(dataset_folder+"EMTAB_3123ensg.csv", index_col=0)
        if (not os.path.exists(dataset_folder+"matches.csv")):
            from NORDic.UTILS.utils_data import request_biodbnet
            matches = request_biodbnet(list(data.index), "Ensembl Gene ID", "Gene Symbol", taxon_id)
            matches.to_csv(dataset_folder+"matches.csv")
        matches = pd.read_csv(dataset_folder+"matches.csv", index_col=0)
        matches = matches.loc[matches["Gene Symbol"] != "-"]
        data = data.loc[matches.index]
        data.index = matches["Gene Symbol"]
        sb.call("wget -c -qO "+dataset_folder+"E-MTAB-3123.sdrf.txt https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3123/E-MTAB-3123.sdrf.txt", shell=True)
        metadata = pd.read_csv(dataset_folder+"E-MTAB-3123.sdrf.txt",sep="\t",index_col=0)[["Characteristics[phenotype]"]]
        metadata.index = [idx.split(".txt")[0] for idx in metadata.index]
        metadata_case = metadata.loc[metadata["Characteristics[phenotype]"]=="case"]
        metadata_control = metadata.loc[metadata["Characteristics[phenotype]"]=="control"]
        data_case = data[metadata_case.index]
        data_case = data_case.groupby(level=0).median()
        data_control = data[metadata_control.index]
        data_control = data_case.groupby(level=0).median()
        ## Convert and restrict gene symbols to those existing in LINCS L1000
        entrezgene_fname=dataset_folder+"pool_entrezgenes_ids.csv"
        if (not os.path.exists(entrezgene_fname)):
            from NORDic.UTILS.utils_data import request_biodbnet
            probes = request_biodbnet([x for y in list(data.index) for x in y.split("; ")], "Gene Symbol and Synonyms", "Gene ID", taxon_id, chunksize=100)
            probes.to_csv(entrezgene_fname)
        probes = pd.read_csv(entrezgene_fname,index_col=0)
        probes = probes[probes["Gene ID"]!="-"]
        robes = probes.loc[~probes.index.duplicated()]
        ## convert EntrezGene ids back to gene symbols in LINCS L1000
        from NORDic.UTILS.LINCS_utils import download_lincs_files
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
        data_lst = []
        for idt, dt in enumerate([data_case, data_control]):
            dt = dt.loc[[d for d in dt.index if (any([y in probes.index for y in d.split("; ")]))]]
            dt.index = ["; ".join([probes.loc[y]["Gene Symbol"] for y in set(d.split("; ")).intersection(list(probes.index))]) for d in dt.index]
            dt = dt.loc[dt.index!=""]
            dt.index = [d.split("; ")[0] for d in dt.index]
            dt = dt.loc[[g for g in genes if (g in dt.index)]]
            dt.loc["annotation"] = [int(idt==0)+1]*dt.shape[1]
            dt.columns = ["Case%d" %i for i in range(dt.shape[1])] if (idt==0) else ["Control%d" %i for i in range(dt.shape[1])]
            data_lst.append(dt)
        ## mat: normalized gene expression matrix (rows=genes, columns=samples)
        data_lst[0].join(data_lst[1], how="outer").to_csv(dataset_folder+"EMTAB3123_control_case.csv")
    return None


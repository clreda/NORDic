#coding: utf-8

import numpy as np
import pandas as pd
import json
from subprocess import check_output as sbcheck_output
from time import sleep
import os
from glob import glob
import io
import requests
import pickle

from NORDic.UTILS.STRING_utils import get_protein_names_from_STRING
from NORDic.UTILS.LINCS_utils import build_url, post_request, lincs_api_url

#https://biodbnet-abcc.ncifcrf.gov/webServices/RestWebService.php

def request_biodbnet(probe_list, from_, to_, taxon_id, chunksize=500, quiet=False):
    '''
        Converts gene identifier from from_ to to_ in a given species
        @param\tprobe_list\tPython character string list: list of probes to convert (of type from_)
        @param\tfrom_\tPython character string: an identifier type as recognized by BioDBnet
        @param\tto_\tPython character string: an identifier type as recognized by BioDBnet
        @param\ttaxonId\tPython integer: NCBI taxonomy ID
        @param\tchunksize\tPython integer[default=500]: 1 chunk per request
        @return\tres_df\tPandas DataFrame: rows/["InputValue"/from_] x columns[to_] ("-" if the identifier has not been found)
    '''
    chunk_probes=[probe_list[i:i+chunksize] for i in range(0,len(probe_list),chunksize)]
    res_list = []
    biodbnet_url="https://biodbnet-abcc.ncifcrf.gov/webServices/rest.php/biodbnetRestApi.json"
    for i, chunk in enumerate(chunk_probes):
        if (not quiet):
            print("<BioDBNet> %d/%d" % (i+1, len(chunk_probes)))
        args_di = {
            "method":"db2db",
            "format": "row",
            "inputValues":",".join(chunk),
            "input":from_,
            "outputs":to_,
            "taxonId":str(taxon_id),
        }
        #https://biodbnet-abcc.ncifcrf.gov/webServices/RestSampleCode.php
        query=biodbnet_url+"?"+"&".join([k+"="+args_di[k] for k in args_di])
        res_list += json.loads("; ".join(sbcheck_output("wget -qO- \""+query+"\"", shell=True).decode("utf-8").split("//")))
        sleep(1)
    res_df = pd.DataFrame(res_list)
    res_df.index = res_df["InputValue"]
    res_df = res_df.drop(columns=["InputValue"])
    return res_df

EntrezGene_missing_genes = {
    "ENSP00000451560": "TPPP2",
    "C11ORF74": "IFTAP",
    "RP11-566K11.2": "TUBB4",
}
def convert_genes_EntrezGene(gene_list, taxon_id, app_name, chunksize=100, missing_genes=EntrezGene_missing_genes,quiet=False):
    '''
        Convert gene symbols into EntrezGene CID
        @param\tgene_list\tPython character string list: list of genes
        @param\ttaxon_id\tPython character string
        @param\tapp_name\tPython character string
        @param\tmissing_genes\tPython dictionary of character string x character string: known conversions
        @param\tchunksize\tPython integer[default=100]: 1 chunk per request
        @param\tquiet\tPython bool[default=False]
        @return\tres_df\tPandas DataFrame: rows/["InputValue"] x columns/["Gene ID"/might be separated by "; "] ("-" if they do not exist) or None if no identifier has been found
    '''
    if (not quiet):
        print("<UTILS_DATA> Gene Symbol -> Gene ID (%d probes)" % len(gene_list))
    probes = request_biodbnet(gene_list, "Gene Symbol and Synonyms", "Gene ID", taxon_id, chunksize=chunksize)
    other_ids = list(probes[probes["Gene ID"]=="-"].index)
    probes = probes[probes["Gene ID"]!="-"]
    df_list = [probes]
    if (len(other_ids)>0):
        if (not quiet):
            print("<UTILS_DATA> Ensembl Gene ID -> Gene ID (%d probes)" % len(other_ids))
        other_probes = request_biodbnet(other_ids, "Ensembl Gene ID", "Gene ID", taxon_id, chunksize=chunksize)
        other_ids = list(other_probes[other_probes["Gene ID"]=="-"].index)
        other_probes = other_probes[other_probes["Gene ID"]!="-"]
        df_list.append(other_probes)
    if (len(other_ids)>0):
        if (not quiet):
            print("<UTILS_DATA> HGNC ID -> Gene ID (%d probes)" % len(other_ids))
        other_other_probes = request_biodbnet(other_ids, "HGNC ID", "Gene ID", taxon_id, chunksize=chunksize)
        other_ids = list(other_other_probes[other_other_probes["Gene ID"]=="-"].index)
        other_other_probes = other_other_probes[other_other_probes["Gene ID"]!="-"]
        df_list.append(other_other_probes)
    if (len(other_ids)>0 and app_name is not None):
        if (not quiet):
            print("<UTILS_DATA> STRING ID -> Gene ID (%d probes)" % len(other_ids))
        res_df = get_protein_names_from_STRING(other_ids, taxon_id, app_name)
        ## if can't be found automatically...
        if (res_df is not None):
            other_ids = []
            for idx in list(res_df["queryItem"]):
                other_ids.append(missing_genes.get(idx, idx))
            res_df = request_biodbnet(other_ids, "Gene Symbol and Synonyms", "Gene ID", taxon_id, chunksize=chunksize, quiet=quiet)
            df_list.append(res_df)
    if (len(other_ids)>0):
        df_list.append(pd.DataFrame([], index=other_ids, columns=["Gene ID"]).fillna("-"))
    if (len(df_list)>0):
        probes = pd.concat(tuple(df_list), axis=0)
    else:
        probes = None
        return probes
    probes = probes.sort_index(ascending=True)
    if (not quiet):
        print("%d probes (successful %d, unsuccessful %d)" % (len(probes), len(probes.loc[probes["Gene ID"]!='-']), len(probes.loc[probes["Gene ID"]=="-"])))
    return probes

def convert_EntrezGene_LINCSL1000(file_folder, EntrezGenes, user_key, quiet=False):
    '''
    Converts EntrezIDs to Gene Symbols present in LINCS L1000
    @param\tfile_folder\tPython character string: path to folder of intermediate results
    @param\tEntrezGenes\tPython character string list: list of EntrezGene IDs
    @param\tuser_key\tPython character string: LINCS L1000 user key
    @param\tquiet\tPython bool[default=False]
    @return\tPandas\tPandas DataFrame: rows/[EntrezID] x columns/["Gene Symbol","Entrez ID"] ("-" if they do not exist)
    '''
    assert user_key
    gene_file = file_folder+"entrezGene_LINCSL1000.pck"
    if (not os.path.exists(gene_file)):
        pert_inames = [None]*len(EntrezGenes)
        entrez_ids = [None]*len(EntrezGenes)
        seen_genes = []
    else:
        with open(gene_file, "rb") as f:
            results = pickle.load(f)
        pert_inames, entrez_ids, seen_genes = results["pert_inames"], results["entrez_ids"], results["seen_genes"]
    for ig, g in enumerate(EntrezGenes):
        if (g in seen_genes):
                continue
        if (not quiet):
            print("<UTILS_DATA> Entrez ID -> LINCS L1000 (%d/%d)" % (ig+1,len(EntrezGenes)))
        endpoint="genes"
        method="filter"
        all_entrezid = str(g).split("; ")
        for entrezid in all_entrezid:
            params = {"where":{"entrez_id": str(entrezid)},"fields":["gene_symbol"]}
            request_url = build_url(endpoint, method, params, user_key=user_key)
            data = post_request(request_url, quiet=True, pause_time=0.3)
            if (len(data)==0):
                continue
            else:
                pert_inames[ig] = data[0]["gene_symbol"]
                entrez_ids[ig] = entrezid
                if (pert_inames[ig]==g):
                    break
        if (pert_inames[ig] is not None):
            print("\t".join([g, pert_inames[ig], str(ig+1), str(len(EntrezGenes))]))
        seen_genes.append(g)
        with open(gene_file, "wb") as f:
            pickle.dump({"pert_inames":pert_inames, "entrez_ids": entrez_ids, "seen_genes": seen_genes}, f)
    pert_inames_ = [p if (p is not None) else "-" for p in pert_inames]
    entrez_ids_ = [entrez_ids[i] if (pert_inames[i] is not None) else "-" for i in range(len(EntrezGenes))]
    pert_df = pd.DataFrame([pert_inames_, entrez_ids], columns=EntrezGenes, index=["Gene Symbol", "Entrez ID"]).T
    pert_df = pert_df.sort_values(by="Gene Symbol", ascending=True)
    pert_df = pert_df.loc[~pert_df.duplicated()]
    return pert_df

def get_all_celllines(pert_inames, user_key, quiet=False):
    '''
        Get all cell lines in which one gene in the input list has been specifically perturbed (genetic perturbation)
        @param\tpert_inames\tPython character string: List of genes (symbols from LINCS L1000)
        @param\tuser_key\tPython character string: user key from LINCS L1000 CLUE API
        @param\tquiet\tPython bool[default=False]
        @return\tcell_lines\tPython character string list: list of cell lines in which at least one gene from pert_inames has been perturbed
    '''
    assert user_key
    endpoint = "sigs"
    method = "filter"
    cell_lines = []
    for ig, g in enumerate(pert_inames):
        if (not quiet):
            print("<UTILS_DATA> Perturbagen %s (%d/%d)" % (g, ig+1, len(pert_inames)), end=": ")
        params = {"where": {"pert_iname": g}, "fields": ["cell_id"]}
        request_url = build_url(endpoint, method, params, user_key=user_key)
        response = requests.get(request_url)
        assert response.status_code == 200
        data = json.loads(response.text)
        cell_lines_gene = list(set([d["cell_id"] for d in data]))
        if (not quiet):
            print("%d unique cells" % len(cell_lines_gene))
        cell_lines = list(set(cell_lines_gene+cell_lines))
    return cell_lines

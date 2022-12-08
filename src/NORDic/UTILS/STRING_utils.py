#coding: utf-8

from time import sleep
import os
import requests
import pandas as pd
import numpy as np
from io import StringIO
from copy import deepcopy
from subprocess import check_output as sbcheck_output
from subprocess import call as sbcall

string_api_url = "https://string-db.org/api"

def get_app_name_STRING(fname):
    '''
        Retrieves app name from STRING to interact with the API
        @param\tfname\tPython character string: path to file with a unique line = email adress
        @return\tapp_name\tPython character string: identifier for the STRING API
    '''
    with open(fname, "r") as f:
        app_name = f.read().split("\n")[0]
    return app_name 

def get_protein_names_from_STRING(gene_list, taxon_id, app_name=None, quiet=False):
    '''
        Retrieves protein IDs in STRING associated with input genes in the correct species
        @param\tgenes_list\tPython character list: list of gene symbols
        @param\ttaxon_id\tPython integer: taxon ID from NCBI
        @param\tapp_name\tPython character string
        @param\tquiet\tPython bool[default=False]
        @returns\tres_df\tPandas DataFrame: rows/[row number] x columns/["queryItem", "stringId", "preferredName", "annotation"]
    '''
    assert app_name
    assert taxon_id
    if (not quiet):
        print("<STRING> Getting the STRING name mapping for genes")
    output_format = "tsv"
    method = "get_string_ids"
    params = {
        "identifiers" : "\r".join(gene_list), # your protein list
        "species" : taxon_id, # species NCBI identifier
        "limit" : 1, # only one (best) identifier per input protein
        "echo_query" : 1, # see your input identifiers in the output
        "caller_identity" : app_name # your app name
    }
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params).text
    sleep(1)
    from io import StringIO
    res_df = pd.read_csv(StringIO(results), sep="\t")
    if ("Error" in res_df.columns):
        return None
    return res_df[["queryItem", "stringId", "preferredName", "annotation"]]

def get_network_from_STRING(gene_list, taxon_id, min_score=0, app_name=None, quiet=False):
    '''
        @param\tgene_list\tPython character string list: list of gene symbols
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @param\tmin_score\tPython integer[default=0]: minimum STRING combined edge score in [0,1000]
        @param\tapp_name\tPython character string
        @param\tquiet\tPython bool[default=False]
        @return\tnetwork\tPandas DataFrame: rows/[row number] x columns/["preferredName_A","preferredName_B","score","directed"]
    '''
    assert app_name
    assert taxon_id
    assert min_score >= 0 and min_score <= 1000
    results = get_protein_names_from_STRING(gene_list, taxon_id, app_name=app_name, quiet=quiet)
    id_di = {}
    my_genes = []
    for line in range(len(results.index)):
        input_identifier, string_identifier = results["queryItem"][line], results["stringId"][line]
        my_genes.append(string_identifier)
        id_di.setdefault(string_identifier, input_identifier)
    if (not quiet):
        print("<STRING> Getting the STRING network interactions")
    output_format = "tsv"
    method = "network"
    request_url = "/".join([string_api_url, output_format, method])
    params = {
        "identifiers" : "%0d".join(my_genes), # your protein
        "species" : taxon_id, # species NCBI identifier 
        "required_score" : min_score, # in 0 - 1000, 0 : get all edges
        "caller_identity" : app_name # your app name
    }
    if (not quiet):
        print("<STRING> Getting the STRING network interactions")
    response = requests.post(request_url, data=params).text
    sleep(1)
    network = pd.read_csv(StringIO(response), sep="\t")
    network["preferredName_A"] = [id_di.get(x, x) for x in list(network["preferredName_A"])]
    network["preferredName_B"] = [id_di.get(x, x) for x in list(network["preferredName_B"])]
    network = network[["preferredName_A","preferredName_B","score"]]
    network["sign"] = [2]*network.shape[0]
    network["directed"] = [0]*network.shape[0]
    network = network[["preferredName_A","preferredName_B","sign","directed","score"]]
    network = network.drop_duplicates(keep="first")
    return network

def get_interactions_from_STRING(gene_list, taxon_id, min_score=0, app_name=None, file_folder=None, version="v10.5", quiet=False):
    '''
        Retrieves (un)directed interactions from the STRING database
        @param\tgene_list\tPython character string list: list of genes
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @param\tmin_score\tPython integer[default=0]: in [0,1000] STRING combined score
        @param\tapp_name\tPython character string
        @param\tfile_folder\tPython character string[default=None]: where to save the file from STRING (if None, the file is not saved)
        @param\tversion\tPython character string[default="v10.5"]: STRING database version
        @param\tquiet\tPython bool[default=False]
        @return\tres_df\tPandas Dataframe: rows/[] x columns/[]
    '''
    assert app_name
    assert taxon_id
    protein_action_fname = (file_folder if (file_folder) else "")+"protein_action.tsv"
    species = str(taxon_id)
    ftype ="protein.actions"
    fname_ = ftype+"."+version+".txt"
    alpha, beta = version[1:].split(".")
    STRING_url = "https://version-"+alpha+"-"+beta+".string-db.org/download/"+ftype+"."+version+"/"+species+"."+fname_+".gz"
    if (not quiet):
        print("<STRING> Retrieving the file from STRING", end="... ")
    if (file_folder is not None and not os.path.exists(protein_action_fname)):
        sbcall("wget -qO- \""+STRING_url+"\" | gzip -d -c > "+protein_action_fname, shell=True)
    if (file_folder is not None):
        if (not quiet):
            print("Saved at %s" % protein_action_fname)
        df = pd.read_csv(protein_action_fname, sep="\t")
    else:
        if (not quiet):
            print("Downloaded online")
        df = pd.read_csv(StringIO(sbcheck_output("wget -qO- \""+STRING_url+"\" | gzip -d -c", shell=True).decode("utf-8")), sep="\t")
    df = df.loc[df["score"]>=min_score]
    df["score"] /= 1000
    res_df = get_protein_names_from_STRING(gene_list, taxon_id, app_name=app_name, quiet=quiet)
    res_df.index = res_df["stringId"]
    res_df = res_df["preferredName"].to_dict()
    df1 = deepcopy(df)
    df1["item_id_a"] = [res_df.get(a, "-") for a in df1["item_id_a"]]
    df1["item_id_b"] = [res_df.get(b, "-") for b in df1["item_id_b"]]
    df = df.loc[(df1["item_id_a"]!="-")|(df1["item_id_b"]!="-")]
    genes = list(set(list(df["item_id_a"])+list(df["item_id_b"])))
    res_df = get_protein_names_from_STRING([x.split(str(taxon_id)+".")[-1] for x in genes], taxon_id, app_name=app_name, quiet=quiet)
    res_df.index = res_df["stringId"]
    res_df = res_df["preferredName"].to_dict()
    df["item_id_a"] = [res_df.get(a, "-") for a in df["item_id_a"]]
    df["item_id_b"] = [res_df.get(b, "-") for b in df["item_id_b"]]
    df = df.loc[(df["item_id_a"]!="-")&(df["item_id_b"]!="-")]
    ## if sign is known
    df["sign"] = [{"activation": 1, "inhibition": -1}.get(x, 2) for x in df["action"]]
    ## if direction is known
    df["directed"] = [int(x=="t") for x in df["is_directional"]]
    df_A = df.loc[df["a_is_acting"]=="f"]
    df_B = df.loc[df["a_is_acting"]=="t"]
    network = pd.DataFrame([], index=range(df.shape[0]))
    network["preferredName_A"] = list(df_B["item_id_a"])+list(df_A["item_id_b"])
    network["preferredName_B"] = list(df_B["item_id_b"])+list(df_A["item_id_a"])
    network["sign"] = list(df_B["sign"])+list(df_A["sign"])
    network["directed"] = list(df_B["directed"])+list(df_A["directed"])
    network["score"] = list(df_B["score"])+list(df_A["score"])
    ## remove redundant
    network = network.drop_duplicates(keep="first")
    return network

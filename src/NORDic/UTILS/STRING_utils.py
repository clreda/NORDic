#coding: utf-8

from time import sleep
import requests
import pandas as pd
import numpy as np
from io import StringIO
from copy import deepcopy
from subprocess import check_output as sbcheck_output

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

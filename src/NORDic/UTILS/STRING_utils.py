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

string_api_url = lambda v : "https://version-"+"-".join(v.split("."))+".string-db.org/api"

def get_app_name_STRING(fname):
    '''
        Retrieves app name from STRING to interact with the API
        @param\tfname\tPython character string: path to file with a unique line = email adress
        @return\tapp_name\tPython character string: identifier for the STRING API
    '''
    with open(fname, "r") as f:
        app_name = f.read().split("\n")[0]
    return app_name 

def get_protein_names_from_STRING(gene_list, taxon_id, app_name=None, version="11.5", quiet=False):
    '''
        Retrieves protein IDs in STRING associated with input genes in the correct species
        @param\tgenes_list\tPython character list: list of gene symbols
        @param\ttaxon_id\tPython integer: taxon ID from NCBI
        @param\tversion\tPython character string[default="11.5"]: STRING version
        @param\tapp_name\tPython character string[default=None]
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
        "echo_query": 1,
        "limit" : 1, # only one (best) identifier per input protein
        "echo_query" : 1, # see your input identifiers in the output
        "caller_identity" : app_name # your app name
    }
    request_url = "/".join([string_api_url(version), output_format, method])
    results = requests.post(request_url, data=params).text
    sleep(1)
    from io import StringIO
    res_df = pd.read_csv(StringIO(results), sep="\t")
    if ("Error" in res_df.columns):
        print("<STRING_utils> Error from STRING: %s" % str(res_df["ErrorMessage"]))
        return None
    return res_df[["queryItem", "stringId", "preferredName", "annotation"]]

def get_image_from_STRING(my_genes, taxon_id, file_name="network.png", min_score=0, network_flavor="evidence", network_type="functional", app_name=None, version="11.5", quiet=False):
    '''
        Retrieves protein IDs in STRING associated with input genes in the correct species
        @param\tgenes_list\tPython character list: list of gene symbols
        @param\ttaxon_id\tPython integer: taxon ID from NCBI
        @param\tfile_name\tPython character string[default="network.png"]: image file name
        @param\tmin_score\tPython float[default=0]: confidence lower threshold (in [0,1])
        @param\tnetwork_flavor\tPython character string[default="evidence"]: show links related to ["confidence", "action", "evidence"]
        @param\tnetwork_type\tPython character string[default="functional"]: show "functional" or "physical" network
        @param\tapp_name\tPython character string
        @param\tquiet\tPython bool[default=False]
        @returns\tNone\t
    '''
    assert app_name
    assert taxon_id
    assert min_score>=0 and min_score<=1
    assert network_flavor in ["confidence", "action", "evidence"]
    assert network_type in ["functional", "physical"]
    if (not quiet):
        print("<STRING> Getting an image of the STRING network")
    output_format = "highres_image"
    method = "network"
    request_url = "/".join([string_api_url(version), output_format, method])
    params = {"identifiers" : "%0d".join(my_genes), 
        "species" : taxon_id,
        "add_white_nodes": 15, # add 15 white nodes to my protein 
        "network_flavor": network_flavor, 
        "network_type": network_type, 
        "required_score": int(min_score*1000),
        "hide_disconnected_nodes": 1,
        "hide_node_labels": 0,
        "show_query_node_labels": 0,
        "caller_identity" : app_name, 
    }
    response = requests.post(request_url, data=params)
    if (not quiet):
        print("Saving interaction network to %s" % file_name)
    with open(file_name, 'wb') as fh:
        fh.write(response.content)
    sleep(1)

def get_network_from_STRING(gene_list, taxon_id, min_score=0, network_type="functional", add_nodes=0, app_name=None, version="11.5", quiet=False):
    '''
        Retrieves undirected and unsigned interactions from the STRING database
        @param\tgene_list\tPython character string list: list of gene symbols
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @param\tmin_score\tPython integer[default=0]: minimum STRING combined edge score in [0,1]
        @param\tnetwork_type\tPython character string[default="functional"]: returns "functional" or "physical" network
        @param\tadd_nodes\tPython integer[default=0]: add nodes *in the closest interaction neighborhood* involved with the genes in @gene_list if set to 1
        @param\tapp_name\tPython character string
        @param\tversion\tPython character string[default="11.5"]: STRING version
        @param\tquiet\tPython bool[default=False]
        @return\tnetwork\tPandas DataFrame: rows/[row number] x columns/["preferredName_A","preferredName_B","score","directed"]
    '''
    assert app_name
    assert taxon_id
    assert min_score >= 0 and min_score <= 1
    results = get_protein_names_from_STRING(gene_list, taxon_id, app_name=app_name, version=version, quiet=quiet)
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
    request_url = "/".join([string_api_url(version), output_format, method])
    params = {
        "identifiers" : "%0d".join(my_genes), # your protein
        "species" : taxon_id, # species NCBI identifier 
        "required_score" : int(min_score*1000), # in 0 - 1000, 0 : get all edges
        "network_type": network_type,
        "add_nodes": add_nodes,
        "show_query_node_labels": 0,
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
    network.sort_values(by="score", ascending=False)
    return network

def get_interactions_partners_from_STRING(gene_list, taxon_id, min_score=0, network_type="functional", add_nodes=0, limit=5, app_name=None, version="11.5", quiet=False):
    '''
        Retrieves undirected and unsigned interactions from the STRING database
        @param\tgene_list\tPython character string list: list of gene symbols
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @param\tmin_score\tPython integer[default=0]: minimum STRING combined edge score in [0,1]
        @param\tnetwork_type\tPython character string[default="functional"]: returns "functional" or "physical" network
        @param\tlimit\tPython integer[default=5]: limits the number of interaction partners retrieved per protein (most confident interactions come first)
        @param\tapp_name\tPython character string
        @param\tversion\tPython character string[default="11.5"]: STRING version
        @param\tquiet\tPython bool[default=False]
        @return\tnetwork\tPandas DataFrame: rows/[row number] x columns/["preferredName_A","preferredName_B","score","directed"]
    '''
    assert app_name
    assert taxon_id
    assert min_score >= 0 and min_score <= 1
    results = get_protein_names_from_STRING(gene_list, taxon_id, app_name=app_name, version=version, quiet=quiet)
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
    request_url = "/".join([string_api_url(version), output_format, method])
    params = {
        "identifiers" : "%0d".join(my_genes), # your protein
        "species" : taxon_id, # species NCBI identifier 
        "limit" : limit, # limits the number of interaction partners retrieved per protein
        "required_score" : int(min_score*1000), # in 0 - 1000, 0 : get all edges
        "network_type": network_type,
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
    network.sort_values(by="score", ascending=False)
    return network

def get_interactions_from_STRING(gene_list, taxon_id, min_score=0, app_name=None, file_folder=None, version="11.0", strict=False, quiet=False):
    '''
        Retrieves (un)directed and (un)signed physical interactions from the STRING database
        @param\tgene_list\tPython character string list: list of genes
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @param\tmin_score\tPython integer[default=0]: in [0,1] STRING combined score
        @param\tapp_name\tPython character string
        @param\tfile_folder\tPython character string[default=None]: where to save the file from STRING (if None, the file is not saved)
        @param\tversion\tPython character string[default="v11.0"]: STRING database version
        @param\tstrict\tPython bool[default=False]: if set to True, only keep interactions involving genes BOTH in @gene_list
        @param\tquiet\tPython bool[default=False]
        @return\tres_df\tPandas Dataframe: rows/[] x columns/[]
    '''
    assert version!="11.5"
    assert app_name
    assert taxon_id
    assert min_score<=1 and min_score>=0
    protein_action_fname = (file_folder if (file_folder) else "")+"protein_action_v"+version+".tsv"
    species = str(taxon_id)
    ftype ="protein.actions"
    fname_ = ftype+".v"+version+".txt"
    alpha, beta = version.split(".")
    STRING_url = "https://version-"+alpha+"-"+beta+".string-db.org/download/"+ftype+".v"+version+"/"+species+"."+fname_+".gz"
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
    df = df.loc[df["score"]>=int(min_score*1000)]
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
    if (not quiet):
        print("... Protein identifier matching")
    ## if sign is known
    df["sign"] = [{"activation": 1, "inhibition": -1}.get(x, 2) for x in df["action"]]
    if (not quiet):
        print("... Signed")
    ## if direction is known
    df["directed"] = [int(x=="t") for x in df["is_directional"]]
    df_A = df.loc[df["a_is_acting"]=="f"]
    df_B = df.loc[df["a_is_acting"]=="t"]
    if (not quiet):
        print("... Directed")
    network = pd.DataFrame([], index=range(df.shape[0]))
    network["preferredName_A"] = list(df_B["item_id_a"])+list(df_A["item_id_b"])
    network["preferredName_B"] = list(df_B["item_id_b"])+list(df_A["item_id_a"])
    network["sign"] = list(df_B["sign"])+list(df_A["sign"])
    network["directed"] = list(df_B["directed"])+list(df_A["directed"])
    network["score"] = list(df_B["score"])+list(df_A["score"])
    if (not quiet):
        print("... Aggregate info")
    ## remove redundant, solve paradoxes
    if (strict):
        f = np.vectorize(lambda x : x in gene_list)
        network = network.loc[f(network["preferredName_A"])&f(network["preferredName_B"])]
    network.index = ["--".join(x) for x in zip(list(network["preferredName_A"]), list(network["preferredName_B"]))]
    network = network.sort_values(by="score", ascending=False) #sort by decreasing score
    solve_conflicts_directed = {x: int(network.loc[x]["directed"].max()) for x in network.index} #if reported once directed, considered directed
    if (not quiet):
        print("... Solve conflicts on direction")
    is_unsigned = lambda sign : ((sign==2).all() or ((sign==-1).any() and (sign==1).any()))
    #for a fixed directed edge, if several different signs are reported, if both signs are present, consider it unsigned
    solve_conflicts_sign = {x: int(network.loc[x]["sign"]) if ("numpy.int64" in str(type(network.loc[x]["sign"]))) else (2 if (is_unsigned(network.loc[x]["sign"])) else (-1)**int((network.loc[x]["sign"]==-1).all())) for x in network.index}
    if (not quiet):
        print("... Solve conflicts on sign for directed edges")
    network = network.loc[~network.index.duplicated(keep="first")] #keep the duplicate with highest score
    network["directed"] = [solve_conflicts_directed[x] for x in network.index]
    network["sign"] = [solve_conflicts_sign[x] for x in network.index]
    final_indices = list(set([idx if (network.loc[idx]["directed"]==1) else ("--".join(list(sorted(idx.split("--"))))) for idx in network.index]))
    network = network.loc[final_indices]
    if (not quiet):
        print("... Aggregate info again")
    directed,indices=[],[]
    for idx in network.index:
        if (("--".join(list(sorted(idx.split("--"))))) in indices):
            continue
        if (("--".join(list(sorted(idx.split("--"))))) in network.index and (network.loc[idx]["directed"]==1)):
            directed.append(0)
        else:
            directed.append(network.loc[idx]["directed"])
        indices.append(idx)
    network = network.loc[indices]
    network["directed"] = directed
    if (not quiet):
        print("... Remove multiple undirected edges")
    #for a fixed UNDIRECTED edge, if several different signs are reported, if both signs are present, consider it unsigned
    network.index = ["--".join(list(sorted([network.loc[x]["preferredName_A"],network.loc[x]["preferredName_B"]]))) for x in network.index] 
    solve_conflicts_sign_undirected = {x: int(network.loc[x]["sign"]) if ((network.loc[x]["directed"]!=0).any() or ("numpy.int64" in str(type(network.loc[x]["sign"])))) else (2 if (is_unsigned(network.loc[x]["sign"])) else (-1)**int((network.loc[x]["sign"]==-1).all())) for x in network.index}
    if (not quiet):
        print("... Solve conflicts on sign for undirected edges")
    for ix, x in enumerate(network.index):
        values = list(network.iloc[ix,:].values.flatten()) 
        values[list(network.columns).index("sign")] = int(solve_conflicts_sign_undirected[x])
        values[list(network.columns).index("directed")] = int(values[list(network.columns).index("directed")])
        values[list(network.columns).index("score")] = float(values[list(network.columns).index("score")])
        network.loc[x] = values
    if (not quiet):
        print("... Aggregate info once again")
    network.index = range(network.shape[0])
    network.sort_values(by="score", ascending=False)
    return network
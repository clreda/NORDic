# coding: utf-8

## source: https://www.disgenet.org/static/disgenet_rest/example_scripts/disgenet_api_request.py

import requests
import os
import sys
import json
import pandas as pd

api_host = "https://www.disgenet.org/api/"

def get_user_key_DISGENET(fname):
    '''
        Retrieves the user key from DisGeNET to call the API
        @param\tfname\tPython character string: path of text file containing on the first line the email, the second the password
        @return\tuser_key\tPython character string: from DisGeNET
    '''
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        email, password = f.read().split("\n")
    user_key = None
    auth_params = {"email": email, "password": password}
    with requests.Session() as s:
        try:
            r = s.post(api_host+'/auth/', data=auth_params)
            if (r.status_code == 200):
                json_response = r.json()
                user_key = json_response.get("token")
            else:
                print(r.status_code)
                print(r.text)
        except requests.exceptions.RequestException as req_ex:
            print(req_ex)
            print("<DISGENET> Something went wrong with the request.")
    return user_key

def get_genes_proteins_from_DISGENET(disease_list, limit=3000, source="CURATED", min_score=0, min_ei=0, min_dsi=0.25, min_dpi=0, chunksize=100, user_key=None, quiet=False):
    '''
        Retrieves a list of protein names (and associated gene names) related to the input disease CIDs
        @param\tdisease_list\tPython character string list: list of Concept IDs (CID) from Medgen for each disease
        @param\tlimit\tPython integer[default=3000]: max. number of proteins
        @param\tsource\tPython character string[default="CURATED"]: DisGeNET data sources ["CURATED","ANIMAL MODELS","INFERRED","ALL"] (see https://www.disgenet.org/dbinfo)
        @param\tmin_score\tPython float[default=0]: minimum global score
        @param\tmin_ei\tPython float[default=0]: minimimum Evidence Index
        @param\tmin_dsi\tPython float[default=0.25]: minimum Disease Specificity Index
        @param\tmin_dpi\tPython float[default=0]: minimum Disease Pleiotropy Index
        @param\tchunksize\tPython integer[default=100]: size of chunks (1 chunk per request)
        @param\tuser_key\tPython character string
        @param\tquiet\tPython bool[default=False]
        @returns\tres_df\tPandas DataFrame: rows/[Disease CID] x columns/["Protein", "Gene Name"] or None if Not found.
    '''
    assert user_key
    assert source in ['CURATED', 'ANIMAL MODELS', 'INFERRED', 'ALL']
    assert chunksize <= 100 and chunksize > 1
    res_df = []
    for i in range(0, len(disease_list), chunksize):
        if (not quiet):
            print("<DISGENET> Retrieving genes... %d/%d" % (i+1, len(disease_list)))
        request_url = api_host+"gda/disease/"+",".join(disease_list[i:(i+chunksize)])
        params = {
                "source": source, 
                "format": "json", 
                "limit": limit, 
                "min_score": min_score, #GDA Score
                "max_score": 1, 
                "min_ei": min_ei, #Evidence Level
                "max_ei": 1,
                "min_dsi": min_dsi, #Disease Specificity Index: min 0.25
                "max_dsi": 1,
                "min_dpi": min_dpi, #Disease Pleiotropy Index
                "max_dpi": 1,
        }
        headers = {"Authorization": "Bearer %s" % user_key}
        request_url += "?"+"&".join([p+"="+str(params[p]) for p in params])
        r = requests.get(request_url, headers=headers)
        if (r.status_code not in [200, 404]):
            print("<DISGENET> %s [request=%s]" % (r.text, request_url))
            raise ValueError("<DISGENET> Request failed.")
        if (r.status_code == 404):
            raise ValueError("<DISGENET> Not found.")
        res = pd.DataFrame(json.loads(r.text))
        res.index = res["diseaseid"]
        vals_genes = res[["gene_symbol"]].groupby(level=0).apply(lambda x : "; ".join(list(sorted(set(list(x.values.flatten()))))))
        vals_proteins = res[["uniprotid"]].groupby(level=0).apply(lambda x : "; ".join(list(map(str,set(x.values.flatten())))))
        res = pd.concat([vals_proteins, vals_genes], axis=1)
        res_df.append(res)
    res_df = pd.concat(res_df)
    res_df.columns = ["Protein", "Gene Name"]
    return res_df

def get_genes_evidences_from_DISGENET(gene_list, disease, limit=3000, source="CURATED", min_score=0, chunksize=100, user_key=None, quiet=False):
    '''
        Retrieves the references for the association between each gene in the list and the disease 
        @param\tgene_list\tPython character string list: list of associated genes
        @param\tdisease\tPython character string: Concept ID (CID) from MedGen
        @param\tlimit\tPython integer[default=3000]: limit of the number of references
        @param\tsource\tPython character string[default="CURATED"]: DisGeNET data sources ["CURATED","ANIMAL MODELS","INFERRED","ALL"] (see https://www.disgenet.org/dbinfo)
        @param\tmin_score\tPython float[default=0]: minimum evidence score
        @param\tchunksize\tPython integer[default=100]: size of chunks (1 chunk per request)
        @param\tuser_key\tPython character string
        @param\tquiet\tPython bool[default=False]
        @returns\tres_df\tPandas DataFrame: rows/[row number] x columns/["gene_symbol", "sentence", "associationtype", "pmid", "year", "score"]
    '''
    assert user_key
    assert source in ['CURATED', 'ANIMAL MODELS', 'INFERRED', 'ALL']
    assert chunksize <= 100 and chunksize > 1
    res_df = []
    for i in range(0, len(gene_list), chunksize):
        if (not quiet):
            print("<DISGENET> Retrieving evidence... %d/%d" % (i+1, len(gene_list)))
        request_url = api_host+"gda/evidences/gene/"+",".join(gene_list[i:(i+chunksize)])
        params = {
                "disease": disease,
                "source": source, 
                "format": "json", 
                "limit": limit, 
                "min_score": min_score, #GDA Score
                "max_score": 1, 
        }
        headers = {"Authorization": "Bearer %s" % user_key}
        request_url += "?"+"&".join([p+"="+str(params[p]) for p in params])
        r = requests.get(request_url, headers=headers)
        if (r.status_code not in [200, 404]):
            print("<DISGENET> %s [request=%s]" % (r.text, request_url))
            raise ValueError("<DISGENET> Request failed.")
        if (r.status_code == 404):
            raise ValueError("<DISGENET> Not found.")
        res = pd.DataFrame(json.loads(r.text))
        ls = list(res["results"])
        res = pd.DataFrame({d: lss for d,lss in enumerate(list(res["results"]))}).T
        res = res.loc[(res["score"]>=min_score)&(res["disease_id"]==disease)]
        if (len(res)>0):
            res_df.append(res)
    if (len(res_df)>0):
        res_df = pd.concat(res_df)
        res_df = res_df[["gene_symbol","sentence","associationtype","pmid","year","score"]]
    else:
        res_df = None
    return res_df

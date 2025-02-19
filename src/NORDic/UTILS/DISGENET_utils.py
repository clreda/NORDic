# coding: utf-8

## source: https://www.disgenet.org/static/disgenet_rest/example_scripts/disgenet_api_request.py

import requests
import os
import sys
import json
import pandas as pd
from time import sleep

api_host = "https://api.disgenet.com/api/v1/"

def get_user_key_DISGENET(fname):
    '''
    Retrieves the user key from DisGeNET to call the API

    ...

    Parameters
    ----------
    fname : Python character string
        path of text file containing on the first line the email, the second the password

    Returns
    ----------
    user_key : Python character string
        from DisGeNET
    '''
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        user_key = f.read().split("\n")[2]
    return user_key

def get_genes_proteins_from_DISGENET(disease_list, limit=3000, source="CURATED", min_score=0, min_dsi=0.25, min_dpi=0, min_pli=0.25, chunksize=10, user_key=None, quiet=False):
    '''
    Retrieves a list of protein names (and associated gene names) related to the input disease CIDs

    ...

    Parameters
    ----------
    disease_list : Python character string list
        list of Concept IDs (CID) from Medgen for each disease
    limit : Python integer
        [default=3000] : max. number of proteins
    source : Python character string
        [default="CURATED"] : DisGeNET data sources ALL, CLINICALTRIALS, CLINGEN, CLINVAR, CURATED, GWASCAT, HPO, INFERRED, MGD_HUMAN, MGD_MOUSE, MODELS, ORPHANET, PHEWASCAT, PSYGENET, RGD_HUMAN, RGD_RAT, TEXTMINING_HUMAN, TEXTMINING_MODELS, UNIPROT (see https://www.disgenet.org/dbinfo)
    min_score : Python float
        [default=0] : minimum global score
    min_dsi : Python float
        [default=0.25] : minimum Disease Specificity Index
    min_dpi : Python float
        [default=0] : minimum Disease Pleiotropy Index
    min_pli : Python float
        [default=0.25] : minimum Loss Intolerance probability
    chunksize : Python integer
        [default=100] : size of chunks (1 chunk per request)
    user_key : Python character string or None
        [default=None] : API key from DisGeNET
    quiet : Python bool
        [default=False] : prints out verbose

    Returns
    ----------
    res_df : Pandas DataFrame
        rows/[Disease CID] x columns/["Protein", "Gene Name"] or None if Not found.
    '''
    assert user_key
    assert source in "ALL, CLINICALTRIALS, CLINGEN, CLINVAR, CURATED, GWASCAT, HPO, INFERRED, MGD_HUMAN, MGD_MOUSE, MODELS, ORPHANET, PHEWASCAT, PSYGENET, RGD_HUMAN, RGD_RAT, TEXTMINING_HUMAN, TEXTMINING_MODELS, UNIPROT".split(", ")
    assert chunksize <= 10 and chunksize > 1
    res_df = []
    for i in range(0, len(disease_list), chunksize):
        if (not quiet):
            print("<DISGENET> Retrieving genes... %d/%d" % (i+1, len(disease_list)))
        request_url = api_host+"gda/summary/"
        nlimit, page = 0, 0
        while ((nlimit<limit) and (page<100)):
            params = {
        	"disease": ",".join(["UMLS_"+x for x in disease_list[i:(i+chunksize)]]),
                "source": source,
                "min_score": min_score, #GDA Score
                "max_score": 1,  
                "min_dsi": min_dsi, #Disease Specificity Index: min 0.25
                "max_dsi": 1,
                "min_dpi": min_dpi, #Disease Pleiotropy Index: min 0.
                "max_dpi": 1,
                "min_pli": min_pli, #Loss Intolerance probability: min 0.25
                "max_pli": 1,
                "type": "disease",
                "page_number": str(page),
                "order_by": "score"
            }
            headers = {"Authorization": user_key, "accept": "application/json"}
            r = requests.get(request_url, params=params, headers=headers, verify=False)
            if not r.ok:
                if r.status_code == 429:
                    while r.ok is False:
                        print("<DISGENET> You have reached a query limit for your user. Please wait {} seconds until next query".format(\
                            r.headers['x-rate-limit-retry-after-seconds']))
                        sleep(int(r.headers['x-rate-limit-retry-after-seconds']))
                        print("<DISGENET> Your rate limit is now restored")
                        # Repeat your query
                        r = requests.get(request_url, params=params, headers=headers, verify=False)
                        if response.ok is True:
                            break
                        else:
                            continue
            response_parsed = json.loads(r.text)
            npage = response_parsed["paging"]["totalElementsInPage"]
            ntotal = response_parsed["paging"]["totalElements"]
            print('<DISGENET> Number of results retrieved by current call (page number {} {}): {} (total number {}/total={},lim={}'.format(\
      response_parsed["paging"]["currentPageNumber"], page, npage, nlimit+npage, ntotal, limit))
            res = pd.DataFrame(response_parsed['payload'])
            res.index = res['diseaseUMLSCUI']
            res = res[['geneProteinStrIDs',"symbolOfGene"]]
            N = min(npage, limit-nlimit)
            if (N < npage):
                res = res.loc[res.index[:N]] ## ordered by score
            if (len(res.index)>0):
                res_df.append(res)
            nlimit += npage
            page += 1
            if (nlimit==ntotal):
                break
    if (len(res_df)>0):
        res_df = pd.concat(res_df)
        res_df.columns = ["Protein", "Gene Name"]
    else:
        res_df = None
    return res_df

def get_genes_evidences_from_DISGENET(gene_list, disease, limit=3000, source="CURATED", min_score=0, min_dsi=0.25, min_dpi=0, min_pli=0.25, chunksize=100, user_key=None, quiet=False):
    '''
    Retrieves the references for the association between each gene in the list and the disease 

    ...

    Parameters
    ----------
    gene_list : Python character string list
        list of associated genes 
    disease : Python character string 
        Concept ID (CID) from MedGen
    limit : Python integer
        [default=3000] : limit of the number of references
    source : Python character string
        [default="CURATED"] : DisGeNET data sources ALL, CLINICALTRIALS, CLINGEN, CLINVAR, CURATED, GWASCAT, HPO, INFERRED, MGD_HUMAN, MGD_MOUSE, MODELS, ORPHANET, PHEWASCAT, PSYGENET, RGD_HUMAN, RGD_RAT, TEXTMINING_HUMAN, TEXTMINING_MODELS, UNIPROT (see https://www.disgenet.org/dbinfo)
    min_score : Python float
        [default=0] : minimum evidence score
    min_dsi : Python float
        [default=0.25] : minimum Disease Specificity Index
    min_dpi : Python float
        [default=0] : minimum Disease Pleiotropy Index
    min_pli : Python float
        [default=0.25] : minimum Loss Intolerance probability
    chunksize : Python integer
        [default=100] : size of chunks (1 chunk per request)
    user_key : Python character string or None
        [default=None] : API key from DisGeNET
    quiet : Python bool
        [default=False] : prints out verbose

    Returns
    ----------
    res_df : Pandas DataFrame
        rows/[row number] x columns/["gene_symbol", "sentence", "associationtype", "pmid", "year", "score"]
    '''
    assert user_key
    assert source in "ALL, CLINICALTRIALS, CLINGEN, CLINVAR, CURATED, GWASCAT, HPO, INFERRED, MGD_HUMAN, MGD_MOUSE, MODELS, ORPHANET, PHEWASCAT, PSYGENET, RGD_HUMAN, RGD_RAT, TEXTMINING_HUMAN, TEXTMINING_MODELS, UNIPROT".split(", ")
    assert chunksize <= 100 and chunksize > 1
    res_df = []
    for i in range(0, len(gene_list), chunksize):
        if (not quiet):
            print("<DISGENET> Retrieving evidence... %d/%d" % (i+1, len(gene_list)))
        request_url = api_host+"gda/summary"
        nlimit, page = 0, 0
        while ((nlimit<limit) and (page<100)):
            params = {
        	"gene_symbol": ",".join(gene_list[i:(i+chunksize)]),
                "source": source,
                "min_score": min_score, #GDA Score
                "max_score": 1,  
                "min_dsi": min_dsi, #Disease Specificity Index: min 0.25
                "max_dsi": 1,
                "min_dpi": min_dpi, #Disease Pleiotropy Index: min 0.
                "max_dpi": 1,
                "min_pli": min_pli, #Loss Intolerance probability: min 0.25
                "max_pli": 1,
                "type": "disease",
                "page_number": str(page),
                "order_by": "score"
            }
            headers = {"Authorization": user_key, "accept": "application/json"}
            r = requests.get(request_url, params=params, headers=headers, verify=False)
            if not r.ok:
                if r.status_code == 429:
                    while r.ok is False:
                        print("<DISGENET> You have reached a query limit for your user. Please wait {} seconds until next query".format(\
                            r.headers['x-rate-limit-retry-after-seconds']))
                        sleep(int(r.headers['x-rate-limit-retry-after-seconds']))
                        print("<DISGENET> Your rate limit is now restored")
                        # Repeat your query
                        r = requests.get(request_url, params=params, headers=headers, verify=False)
                        if response.ok is True:
                            break
                        else:
                            continue                  
            response_parsed = json.loads(r.text)
            res = pd.DataFrame(response_parsed['payload'])
            res = res.loc[res['diseaseUMLSCUI']==disease]
            npage = res.shape[0]
            if (npage==0):
                break  
            ntotal = response_parsed["paging"]["totalElements"]
            print('<DISGENET> Number of results retrieved by current call (page number {} {}): {} (total number {}/total={},lim={}'.format(\
      response_parsed["paging"]["currentPageNumber"], page, npage, nlimit+npage, ntotal, limit))
            res["year"] = ["-".join(list(map(lambda x: str(int(x)), list(res.loc[x][["yearInitial","yearFinal"]])))) for x in res.index]
            res = res[['symbolOfGene','geneProteinClassNames', 'diseaseVocabularies', 'numPMIDs',  'year', 'score']]
            N = min(npage, limit-nlimit)
            if (N < npage):
                res = res.loc[res.index[:N]] ## ordered by score
            if (len(res.index)>0):
                res_df.append(res)
            nlimit += npage
            page += 1
            if (nlimit==ntotal):
                break          
    if (len(res_df)>0):
        res_df = pd.concat(res_df)
        res_df.columns = ["gene_symbol","protein_classes","disease_vocabulary","#pmid","year","score"]
    else:
        res_df = None
    return res_df

if __name__=="__main__":
	user_key = get_user_key_DISGENET("../../../tests/credentials_DISGENET.txt")
	print("")
	res_df = get_genes_proteins_from_DISGENET(["C1275808"], limit=3000, source="CURATED", min_score=0, min_dsi=0.25, min_dpi=0, min_pli=0.25, chunksize=10, user_key=user_key, quiet=False)
	print(res_df)
	print("")
	res_df = get_genes_evidences_from_DISGENET("PHOX2B,RET,BDNF,LBX1,EDN3,GDNF,ASCL1".split(","), "C1275808", limit=3000, source="CURATED", min_score=0., min_dsi=0, min_dpi=0, min_pli=0, chunksize=100, user_key=user_key, quiet=False)
	print(res_df)

# coding:utf-8

## https://maayanlab.cloud/L1000CDS2/help/#cosine

import pandas as pd
import requests
import json
import numpy as np

from NORDic.UTILS.LINCS_utils import build_url, get_user_key, binarize_via_CD, download_lincs_files, create_restricted_drug_signatures
from NORDic.UTILS.utils_data import get_all_celllines

def get_ranking(CD):
    '''
        Retrieve ranking (50 first drugs) from L1000 CDS^2 search engine
        @param\tCD\tPandas DataFrame: rows/[genes] x column/[value] differential phenotype
        @return\tresuls\tPandas DataFrame: rows/[drug names] x column/["L1000 CDS2"] ranking 
        of the drugs according to their ability to reverse the phenotype
    '''
    ## # The app uses uppercase gene symbols. So it is crucial to perform this step
    CD.index = [g.upper() for g in CD.index]
    l1000cds2_url = 'https://maayanlab.cloud/L1000CDS2/query'
    ## https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-debian/
    ## https://maayanlab.cloud/public/L1000CDS_download/#genesMeta
    data = {"genes":list(CD.index),"vals":list(CD[CD.columns[0]])}
    config = {"aggravate":False,"searchMethod":"CD","share":False,"combination":False,"db-version":"latest","limit":100}
    headers = {'content-type': 'application/json'}
    results = {'topMeta': {}, 'uniqInput': []}
    metadata = [{"key":"Tag","value":"CD baseline test"}]
    payload = {"data":data,"config":config,"meta":metadata}
    r = requests.post(l1000cds2_url,data=json.dumps(payload),headers=headers,stream=True)
    assert r.status_code==200
    resCD = r.json()
    assert "err" not in resCD
    results["topMeta"].update({res["pert_desc"]: res["score"] for res in resCD["topMeta"]})# if (res["cell_id"] in ["NPC","SH-SY5Y"])}) 
    results["uniqInput"] = resCD["uniqInput"]
    return pd.DataFrame({"L1000 CDS2": results["topMeta"]}).sort_values(by="L1000 CDS2", ascending=False)

def drugname2pubchem(drug_names, lincs_args):
    '''
        Convert drug names into PubChem CIDs
        @param\tdrug_names\tPython character string list: list of drug names
        @param\tlincs_args\tPython dictionary: additional arguments for LINCS L1000 requests
        @return\tpubchem_cids\tPython dictionary: (keys=drug names, values=PubChem CIDs)
    '''
    endpoint = "perts"
    method = "filter"
    pubchem_cids = {}
    assert lincs_args and "credentials" in lincs_args
    user_key = get_user_key(lincs_args["credentials"])
    for ip, drug_name in enumerate(drug_names):
        params = {"where": {"pert_iname": drug_name.lower()}, "fields": ["pubchem_cid"]}
        request_url = build_url(endpoint, method, params=params, user_key=user_key)
        response = requests.get(request_url)
        assert response.status_code == 200
        data = json.loads(response.text)
        cid = int(data[0]["pubchem_cid"]) if (len(data)>0) else np.nan
        pubchem_cids.setdefault(drug_name, cid)
    return pubchem_cids

def pubchem2drugname(pubchem_cids, lincs_args):
    '''
        Convert drug names into PubChem CIDs
        @param\tpubchem_cids\tPython integer list: list of drug PubChem CIDs
        @param\tlincs_args\tPython dictionary: additional arguments for LINCS L1000 requests
        @return\tpert_inames\tPython dictionary: (keys=PubChem CIDs, values=drug names)
    '''
    endpoint = "perts"
    method = "filter"
    pert_inames = {}
    assert lincs_args and "credentials" in lincs_args
    user_key = get_user_key(lincs_args["credentials"])
    for ip, pubchem_cid in enumerate(pubchem_cids):
        params = {"where": {"pubchem_cid": pubchem_cid}, "fields": ["pert_iname"]}
        request_url = build_url(endpoint, method, params=params, user_key=user_key)
        response = requests.get(request_url)
        assert response.status_code == 200
        data = json.loads(response.text)
        drug_name = data[0]["pert_iname"] if (len(data)>0) else np.nan
        pert_inames.setdefault(pubchem_cid, drug_name)
    return pert_inames

def retrieve_drug_signature(pubchem_cid, cell_ids, gene_list, lincs_args, quiet=False):
    '''
        Retrieve control & treated samples from LINCS L1000 and compute the corresponding drug signature
        @param\tpubchem_cid\tPython integer: drug PubChem CID
        @param\tcell_ids\tPython character string list: list of candidate cell lines in LINCS L1000
        @param\tgene_list\tPython integer list: list of EntrezID genes
        @param\tlincs_args\tPython dictionary: additional arguments for LINCS L1000 requests
        @param\tquiet\tPython bool[default=False]
        @return\tsig\tPandas DataFrame: rows/[genes] x column/[drug PubChem]
    '''
    endpoint = "sigs"
    method = "filter"
    pert_iname = pubchem2drugname([pubchem_cid], lincs_args)[pubchem_cid]
    assert lincs_args and "credentials" in lincs_args
    user_key = get_user_key(lincs_args["credentials"])
    selection = lincs_args.get("selection", "distil_ss")
    data = []
    if (not quiet):
        print("* PubChemCID %d (name %s)" % (pubchem_cid, pert_iname))
    for cell_id in cell_ids:
        ## 1. Build request to get distil_id of relevant treated samples
        where = {"pert_type": "trt_cp", "pert_iname": pert_iname, "cell_id": cell_id}
        params = {
            "where": where,
            "fields": ["cell_id", "distil_id", "brew_prefix"]+[selection]
        }
        request_url = build_url(endpoint, method, params=params, user_key=user_key)
        response = requests.get(request_url)
        assert response.status_code == 200
        data += json.loads(response.text)
    if (len(data)==0):
        return None
    ## At least @nsigs replicates and maximize the criterion
    data_treated = [dt for dt in data if (len(dt["distil_id"])>lincs_args.get("nsigs", 2))]
    data_treated = data_treated[np.argmax([dt[selection] for dt in data_treated])]
    brew_prefix, cell_id, data_treated = data_treated["brew_prefix"], data_treated["cell_id"], data_treated["distil_id"]
    ## 2. Get corresponding control samples from the same plate
    where = {'pert_type': "ctl_vehicle", "cell_id": cell_id, "brew_prefix": brew_prefix}
    params = {"where": where,
        "fields": ["distil_id"]+[selection],
    }
    request_url = build_url(endpoint, method, params=params, user_key=user_key)
    response = requests.get(request_url)
    assert response.status_code == 200
    data = json.loads(response.text)
    if (len(data)==0):
        return None
    ## At least @nsigs replicates and maximize the criterion
    data_control = [dt for dt in data if (len(dt["distil_id"])>lincs_args.get("nsigs", 2))]
    data_control = data_control[np.argmax([dt[selection] for dt in data_control])]
    data_control = data_control["distil_id"]
    nsamples = len(data_treated+data_control)
    if (not quiet):
        print("%d samples" % nsamples)
    sigs = create_restricted_drug_signatures(data_treated+data_control, gene_list, which_lvl=[3], strict=True, path_to_lincs=lincs_args["path_to_lincs"])
    if (sigs is None):
        return None
    assert sigs.shape[1] == nsamples
    sig = binarize_via_CD(sigs, samples=[2]*len(data_treated)+[1]*len(data_control), binarize=0, nperm=10000)
    sig.columns = [pubchem_cid]
    return sig

def compute_drug_signatures_L1000(pubchem_cids, lincs_args, chunksize=10):
    '''
        Get drug signatures from LINCS L1000
        @param\tpubchem_cids\tPython integer list: list of drug PubChem CIDs
        @param\tlincs_args\tPython dictionary: additional arguments for LINCS L1000 requests
        @return\tsigs\tPandas DataFrame: rows/[genes] x columns/[drug names]
    '''
    assert lincs_args and "credentials" in lincs_args and "path_to_lincs" in lincs_args
    user_key = get_user_key(lincs_args["credentials"])
    path_to_lincs = lincs_args["path_to_lincs"]
    pert_inames = pubchem2drugname(pubchem_cids, lincs_args)
    gene_files, _, _, _ = download_lincs_files(path_to_lincs, which_lvl=[3])
    gene_df = pd.read_csv(path_to_lincs+gene_files[0], sep="\t", engine='python', index_col=0)
    gene_list, gene_name_list = list(gene_df.index), list(gene_df["pr_gene_symbol"])
    sigs_list = [retrieve_drug_signature(pubchem_cid, get_all_celllines([pert_inames[pubchem_cid]], user_key) if (len(lincs_args.get("cell_lines", []))==0) else lincs_args["cell_lines"], gene_list, lincs_args) for pubchem_cid in pubchem_cids]
    sigs = sigs_list[0].join(sigs_list[1:], how="outer")
    sigs.index = [gene_df.loc[i]["pr_gene_symbol"] for i in sigs.index]
    return sigs
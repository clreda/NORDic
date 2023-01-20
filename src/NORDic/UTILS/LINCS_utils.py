#coding: utf-8

from subprocess import check_output as sbcheck_output
from subprocess import call as sbcall
import json
from time import sleep
import os
import pandas as pd
import numpy as np
import cmapPy.pandasGEXpress.parse_gctx as parse_gctx
from functools import reduce

def get_user_key(fname):
    '''
        Retrieves user key for interacting with LINCS L1000 CLUE API
        @param\tfname\tPython character string: path to file containing credentials for LINCS L1000 (first line: username, second line: password, third line: user key)
        @return\tuser_key\tPython character string: identifier for the LINCS L1000 CLUE API
    '''
    with open(fname, "r") as f:
        user_key = f.read().split("\n")[2]
    return user_key

def convert_ctrlgenes_EntrezGene(taxon_id):
    '''
        Retrieves EntrezID from control genes in LINCS L1000 [1]
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @return\tlincs_specific_ctl_genes\tPython character string list: list of EntrezGene IDs for all genes in input list
        [1] doi.org/10.1002/psp4.12107 
    '''
    lincs_specific_ctl_genes = ["ACTB", "CHMP2A", "EEF1A1", "EMC7", "GAPDH", "GPI", "PSMB2", "PSMB4", "RAB7A", "REEP5", "SNRPD3", "TUBA1A", "VCP"]
    from NORDic.UTILS.utils_data import request_biodbnet
    lincs_specific_ctl_genes_ = [np.min(list(map(int, s.split("; ")))) for s in request_biodbnet(lincs_specific_ctl_genes, from_="Gene Symbol and Synonyms", to_="Gene ID", taxon_id=taxon_id, chunksize=100)["Gene ID"]]
    lincs_specific_ctl_genes = {lincs_specific_ctl_genes[ix]: int(x) for ix, x in enumerate(lincs_specific_ctl_genes_)}
    return lincs_specific_ctl_genes

####################
## API CALLS      ##
####################

lincs_api_url = "https://api.clue.io/api"

def build_url(endpoint, method, params, user_key=None):
    '''
        Builds the request to CLUE API
        @param\tendpoint\tPython character string: in ["sigs", "cells", "genes", "perts", "plates", "profiles", "rep_drugs", "rep_drug_indications", "pcls"]
        @param\tmethod\tPython character string: in ["count", "filter", "distinct"]
        @param\tparams\tPython dictionary: additional arguments for the request
        @param\tuser_key\tPython character string
        @return\turl\tPython character string: URL of request
    '''
    assert user_key
    assert endpoint in ["sigs", "cells", "genes", "perts", "plates", "profiles", "rep_drugs", "rep_drug_indications", "pcls"]
    assert method in ["count", "filter", "distinct"]
    request_url = "/".join([lincs_api_url, endpoint])
    convert_dict = lambda lst : '"'.join(str(lst).split("\'"))
    if (len(params) > 0):
        params_concat = "&".join([k+"="+convert_dict(params[k]) for k in list(params.keys())])
    else:
        params_concat = ""
    if (method == "count"):
        if (len(params) > 0):
            params_concat = "&".join([k+"="+convert_dict(params[k]) for k in list(params.keys())])
        else:
            params_concat = ""
        request_url += "/"+method+"?"+params_concat
    elif (method == "filter"):
        if (len(params) > 0):
            params_concat = "{"+(",".join(['"'+k+'":'+convert_dict(params[k]) for k in list(params.keys())]))+"}"
        else:
            params_concat = ""
        request_url += "?"+method+"="+params_concat
    elif (method == "distinct"):
        assert params["field"]
        if ("where" in list(params.keys())):
            request_url += "/"+method+"?where="+convert_dict(params["where"])+"&field="+params["field"]
        else:
            request_url += "/"+method+"?field="+params["field"]
    else:
        raise ValueError("<LINCS> Request method could not be identified.")
    return request_url+"&user_key="+user_key

def post_request(url, quiet=True, pause_time=1):
    '''
        Post request to API
        @param\turl\tPython character string: URL formatted as in build_url
        @param\tquiet\tPython bool[default=True]
        @param\tpause_time\tPython integer[default=1]: minimum time in seconds between each request
        @return\tdata\tPython dictionary (JSON) / Python character string list [if request was method="distinct"]
    '''
    if (not quiet):
        print("> POST "+url.split("&")[0])
    response = sbcheck_output("wget -O - -q \'" + url + "\'", shell=True)
    data = json.loads(response)
    if ("/distinct?field=" in url):
        data = list(set([y for x in data for y in x]))
    sleep(pause_time)
    return data

##<><><><><><><><><><><><><><><><><><><><><><><><><><><><>##
##   Binarization of Level 3 signatures                   ##
##<><><><><><><><><><><><><><><><><><><><><><><><><><><><>##

def binarize_via_CD(df, samples, binarize=1, nperm=10000, quiet=False):
    '''
        Run a differential expression analysis on a dataframe using Characteristic Direction (CD) [1] (implementation: www.maayanlab.net/CD/)
        @param\tdf\tPandas DataFrame: 1 transcriptional profile per column (/!\ if #genes>25,000, then the 25,000 genes with highest variance will be considered)
        @param\tsamples\tPython integer list: indicates which columns correspond to control (=1) / treated (=2) samples
        @param\tbinarize\tPython integer[default=1]: whether to return a binary signature or a real-valued column ~magnitude of change in expression
        @param\tnperm\tPython integer[default=10000]: number of iterations to build the null distribution on which p-values will be computed
        @param\tquiet\tPython bool[default=False]
        @return\tsignature\tPandas DataFrame: rows/[gene index] x columns/["aggregated"]: 0=down-regulated (DR), 1=up-regulated (UR) (if binarize=1) else <0=DR, >0=UR
        [1] doi.org/10.1186/1471-2105-15-79
    '''
    sbcheck_output("pip install git+https://github.com/Maayanlab/geode.git", shell=True)
    from geode import chdir
    assert all([s in [1,2] for s in samples])
    assert binarize in [0,1]
    assert nperm>1
    assert len(df.columns) >= 4
    assert len(df.columns) == len(samples)
    assert len(np.argwhere(np.array(samples) == 1).tolist()) > 1
    assert len(np.argwhere(np.array(samples) == 2).tolist()) > 1
    df = df.dropna()
    ## To decrease computational cost (in terms of memory)
    if (df.shape[0] > 25000):
        selected_genes = list(df.var(axis=0).sort_values(ascending=False).index)[:25000]
        if (not quiet):
            print("/!\ The number of genes has been trimmed down to the 25,000 genes with highest variance (down from %d). Consider filtering your genes beforehand!" % df.shape[0])
        df = df.loc[selected_genes]
    # @param mat NumPy matrix
    # @param samples Python integer list
    # @param calculate_sig Python integer: computes p-values if set to 1
    # @param nnull Python integer
    # @param sig_only Python integer: returns only significant genes if set to 1
    # @returns chdir_res: list of tuples (magnitude, gene, *) sorted by decreasing *absolute* magnitude
    chdir_res = chdir(df.values, samples, list(df.index), calculate_sig=binarize, nnull=nperm, sig_only=binarize)
    sign_genes = {x[1] : x[0] for x in chdir_res}
    signature = pd.DataFrame([[sign_genes.get(g, np.nan)] for g in df.index], index=df.index, columns=["aggregated"])
    if (binarize>0):
        signature = (signature > 0).astype(int)-(signature < 0).astype(int)
        signature[signature==0] = np.nan
        signature[signature<0] = 0
    return signature

####################
## DOWNLOAD FILES ##
####################

def download_file(path_to_lincs, file_name, base_url, file_sha, check_SHA=True, quiet=False):
    '''
        Downloads automatically LINCS L1000-related files from Gene Expression Omnibus (GEO) (/!\ can be time-consuming: expect waiting times up to 20 min with a good Internet connection)
        @param\tpath_to_lincs\tPython character string: path to local LINCS L1000 folder in which the files will be downloaded
        @param\tfile_name\tPython character string: file name to download on GEO
        @param\tbase_url\tPython character string: path to GEO repository
        @param\tfile_sha\tPython character string: file name of corresponding SHA hash to check file integrity
        @param\tcheck_SHA\tPython bool[default=True]: whether to check the file integrity
        @params\tquiet\tPython bool[default=False]
        @return\t0\tmeaning that the download was successful
    '''
    if (not os.path.exists(path_to_lincs+file_name)):
        if (not os.path.exists(path_to_lincs+file_name+".gz")):
            cmd = "wget -c "+("" if (not quiet) else "-q ")+"-O "+path_to_lincs+file_name+".gz "+base_url+file_name+".gz"
            sbcall(cmd, shell=True)
            if (check_SHA):
                ## Checks file integrity
                sha_table = pd.read_csv(path_to_lincs+file_sha, sep="  ", names=["sha", "file"], engine='python')
                true_sha_id = list(sha_table[sha_table["file"] == file_name+".gz"]["sha"])[0]
                sha_id = sb.check_output("sha512sum "+path_to_lincs+file_name+".gz", shell=True).decode("utf-8").split("  ")[0]
                assert sha_id == true_sha_id
        sbcall("gzip -df "+path_to_lincs+file_name+".gz", shell=True)
        if (not quiet):
            print("<LINCS_utils> "+file_name+" successfully downloaded in "+path_to_lincs)
    else:
        ## Checks file integrity
        if (check_SHA and os.path.exists(path_to_lincs+file_name+".gz")):
            sha_table = pd.read_csv(path_to_lincs+file_sha, sep="  ", names=["sha", "file"], engine='python')
            true_sha_id = list(sha_table[sha_table["file"] == file_name+".gz"]["sha"])[0]
            sha_id = sb.check_output("sha512sum "+path_to_lincs+file_name+".gz", shell=True).decode("utf-8").split("  ")[0]
            if (not (sha_id == true_sha_id)):
                cmd = "wget "+("" if (not quiet) else "-q ")+"-c -O "+path_to_lincs+file_name+".gz "+base_url+file_name+".gz"
                sbcall(cmd, shell=True)
                if (check_SHA):
                    ## Checks file integrity
                    sha_table = pd.read_csv(path_to_lincs+file_sha, sep="  ", names=["sha", "file"], engine='python')
                    true_sha_id = list(sha_table[sha_table["file"] == file_name+".gz"]["sha"])[0]
                    sha_id = sb.check_output("sha512sum "+path_to_lincs+file_name+".gz", shell=True).decode("utf-8").split("  ")[0]
                    assert sha_id == true_sha_id
                sbcall("gzip -df "+path_to_lincs+file_name+".gz", shell=True)
                if (not quiet):
                    print("<LINCS_UTILS> "+file_name+" successfully (re)downloaded in "+path_to_lincs)
    return 0

def download_lincs_files(path_to_lincs, which_lvl):
    '''
        Returns and downloads the proper LINCS L1000 files from Gene Expression Omnibus (GEO)
        @param\tpath_to_lincs\tPython character string: path to folder in which LINCS L1000-related files will be locally stored
        @param\twhich_lvl\tPython integer list: LINCS L1000 Level to download (either [3] -normalized gene expression-, [5] -binary experimental signatures-, [3,5])
        @return\tfile_list\tList of 4 Python character string lists: gene_files, sig_files, lvl3_files, lvl5_files Python lists of character strings
    '''
    assert all([x in [3,5] for x in which_lvl])
    ## Latest versions of LINCS L1000 as of September 2022
    lincs_gse = {
		"phase1": {
			"acc": "GSE92742",
			"file_lvl5": "Level5_COMPZ.MODZ_n473647x12328.gctx",
			"file_lvl3": "Level3_INF_mlr12k_n1319138x12328.gctx",
			"gene_file": "gene_info.txt",
			"sig_file": "sig_info.txt"
		},
		"phase2": {
			"acc": "GSE70138",
			"file_lvl5": "Level5_COMPZ_n118050x12328_2017-03-06.gctx",
			"file_lvl3": "Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx",
			"gene_file": "gene_info_2017-03-06.txt",
			"sig_file": "sig_info_2017-03-06.txt"
		}
	}
    keys = ["gene_file", "sig_file", "file_lvl3", "file_lvl5"]
    file_di = {}
    for k in keys:
        file_di.setdefault(k, [])
    for key in list(lincs_gse.keys()):
        acc = lincs_gse[key]["acc"]
        base_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/"+acc[:5]+"nnn/"+acc+"/suppl/"
        for k in keys:
            file_di[k] = file_di[k]+[acc+"_Broad_LINCS_"+lincs_gse[key][k]]
        file_sha = acc+"_SHA512SUMS.txt"
        download_file(path_to_lincs, file_sha, base_url, None, check_SHA=False)
        # 5GB! ~20 minutes with a good Internet connection (level 5)
        # 48.8GB and 12.6GB! (level 3)
        for lvl in which_lvl:
            download_file(path_to_lincs, file_di["file_lvl"+str(lvl)][-1], base_url, file_sha)
        for k in keys[:2]:
            download_file(path_to_lincs, file_di[k][-1], base_url, file_sha)	
    return [file_di[k] for k in keys]

####################
## RETRIEVE DATA  ##
####################

def create_restricted_drug_signatures(sig_ids, entrezid_list, path_to_lincs, which_lvl=[3], strict=True, quiet=False):
    '''
        Create dataframe of drug signatures from LINCS L1000 from a subset of signature and gene IDs
        @param\tsig_ids\tPython character string list: list of signature IDs from LINCS L1000 (Level 3: "distil_id", Level 5: "sig_id")
        @param\tentrezid_list\tPython character string list: list of EntrezIDs 
        @param\tpath_to_lincs\tPython character string: folder in which LINCS L1000-related files are stored
        @param\twhich_lvl\tPython integer list: [3] for Level 3, [5] for Level 5 
        @param\tstrict\tPython bool[default=True]: if set to True, if not all signatures are retrieved, then return None. If set to False, return the (sub)set of retrievable signatures
        @param\tquiet\tPython bool[default=False]
        @return\tsigs\tPandas DataFrame: rows/[] x columns/[]
    '''
    assert all([x in [3,5] for x in which_lvl])
    assert len(which_lvl)==1
    gene_files, sig_files, lvl3_files, lvl5_files = download_lincs_files(path_to_lincs, which_lvl=which_lvl)
    sigs = []
    cid = ("sig_id" if (5 in which_lvl) else "distil_id")
    for idf, sf in enumerate(sig_files):
        df = pd.read_csv(path_to_lincs+sf, sep="\t", engine='python')
        if (3 in which_lvl):
            sig_selection = [sig for sig in list(set(sig_ids)) if any([sig in x.split("|") for x in list(df[cid])])]
        else:
            sig_selection = [sig for sig in list(set(sig_ids)) if (sig in list(df[cid]))]
        if (len(sig_selection) > 0):
            df = pd.read_csv(path_to_lincs+gene_files[idf], sep="\t", engine='python', dtype=str)
            if (entrezid_list is None):
                gene_selection = list(df.index)
            else:
                gene_selection = [g for g in list(df["pr_gene_id"]) if (int(g) in entrezid_list)]
            try:
                # https://github.com/cmap/cmapPy/blob/master/cmapPy/pandasGEXpress/parse_gctx.py
                dataset = parse_gctx.parse(path_to_lincs+(lvl5_files if (5 in which_lvl) else lvl3_files)[idf], cid=sig_selection, rid=gene_selection).data_df
            except Exception as e:
                # means "not in metadata for the considered GCTX file"
                print("<LINCS> "+str(e))
                continue
            dataset = dataset.loc[gene_selection][sig_selection]
            dataset.index = [int(x) for x in gene_selection]
            assert dataset.shape[0]==len(gene_selection)
            assert dataset.shape[1]==len(sig_selection)
            sigs.append(dataset)
    if (len(sigs)>0):
        sigs = sigs[0].join(sigs[1:], how="outer")
    else:
        if (not quiet):
            print("\n<LINCS> /!\ None of the following signature/profile ids was found -> {"+"  ".join(sig_ids)+"}")
        if (strict):
            raise ValueError("<LINCS> No signature/profile found.")
        else:
            return None
    assert (len(sig_ids)==sigs.shape[1]) or (not strict)
    if (len(sigs)>0 and ((len(sigs.columns) < len(sig_ids)) and strict)):
        if (not quiet):
            print("\n<LINCS> /!\ Some of the following signature/profile ids were not found -> {"+"  ".join([x for x in sig_ids if (x not in sigs.columns, sig_ids)])+"}")
        return None
    return sigs

def select_best_sig(params, filters, user_key, selection="distil_ss", nsigs=2, same_plate=True, iunit=None, quiet=False):
    '''
        Select "best" set of profiles ("experiment") (in terms of quality, or criterion "selection") according to filters
        @param\tparams\tPython dictionary: additional arguments for the request
        @param\tfilters\tPython dictionary: additional arguments for filtering the results of the request (defined with params)
        @param\tselection\tPython character string[default="distil_ss"]: name of the metric in LINCS L1000 to define the best signature
        @param\tnsigs\tPython integer[default=2]: minimum number of signatures to retrieve
        @param\tsame_plate\tPython bool[default=True]: retrieve signatures from the same plate or not
        @param\tiunit\tPython character string
        @param\tquiet\tPython bool[default=False]
        @return\tdata\tPython dictionary list:
    '''
    assert len(selection) == 0 or (selection in list(filters.keys()) or selection in params.get("fields", []))
    id_fields = ["brew_prefix"]
    if (any([not f in params.get("fields", []) for f in id_fields]) or (len(params.get("fields", [])) == 0 or (len(filters) > 0 and any([not (f in params.get("fields", [])) for f in list(filters.keys())])))):
        fields = params.get("fields", [])
        fields += list(filters.keys())
        params["fields"] = list(set(fields+id_fields))
    endpoint = "sigs"
    method = "filter"
    request_url = build_url(endpoint, method, params=params, user_key=user_key)
    ## GET EXPERIMENTS MATCHING THE PARAMS
    data = post_request(request_url, quiet=True, pause_time=0.3)
    ## 1. Test if enough signatures were found
    if (nsigs > 0 and (len(data) == 0 or (same_plate and all([len(d["distil_id"]) < nsigs for d in data])) or (not same_plate and sum([len(d["distil_id"]) for d in data]) < nsigs))):
        raise ValueError("<LINCS> (1) No (enough) signatures (%d instead of min. %d) retrieved via LINCS L1000 API.\n%s\n" % (sum([len(d["distil_id"]) for d in data]), nsigs, request_url))
    if (not quiet):
        if ("where" not in params):
            cond = ""
        else:
            if ("pert_type" in params["where"]):
                cond = "TREATED" if ("trt_" == params["where"]["pert_type"][:4]) else "CONTROL"
            else:
                cond = ""
        print("<LINCS> %d %s experiments ('brew_prefix') (%d profiles 'distil_id' in average)" % (len(data), cond, np.mean([len(d["distil_id"]) for d in data])))
    ## 2. Filter signatures according to the filters
    if (len(filters) > 0):
        data_ = [x for x in data if (all([x.get(k, filters[k]+1) > filters[k] for k in list(filters.keys())]))]
        if (not quiet):
            print("<LINCS> %d/%d filtered experiments" % (len(data_),len(data)))
        data = data_
    ## 3. Check if matches the specified (if exists) dose
    if ("pert_dose_unit" in params["fields"] and iunit):
        data = [x for x in data if (x["pert_dose_unit"] == iunit.decode("utf-8"))]
    ## 4. Keep only valid profile IDs ('distil_id')
    if ("distil_id" in params["fields"]):
        data_ = []
        for i in range(len(data)):
            distil_ids = [x for x in data[i]["distil_id"] if (len(x.split(":")) == 2)]
            if (len(distil_ids) > 0):
                d = data[i]
                d["distil_id"] = distil_ids
                data_.append(d)
        data = data_
    ## 5. Select the best set of profiles ("experiment") based on the criterion "selection"
    if (same_plate and "distil_id" in params["fields"] and len(data) > 0):
        data = [x for x in data if (len(list(set(x["distil_id"]))) >= nsigs)]
        argmax_rank = int(np.argmax([float(x.get(selection, 0)) for x in data]))
        data_rank = data[argmax_rank]
        if (not quiet):
            print("<LINCS> Selected "+selection+" = "+str(data_rank[selection]))
        data = [{"distil_id": distil_id} for distil_id in list(set(data_rank["distil_id"]))]
        for f in [x for x in params["fields"] if (x != "distil_id")]:
            for i in range(len(data)):
                data[i].setdefault(f, data_rank[f])
        if (not quiet):
            print("<LINCS> %d same-plate profiles" % len(data))
    elif ("distil_id" in params["fields"]):
        for i in range(len(data)):
            data[i]["distil_id"] = data[i]["distil_id"][0]
    return data

def compute_interference_scale(sigs, samples, entrez_id, is_oe, taxon_id, lincs_specific_ctl_genes, quiet=True, eps=2e-7):
    '''
        Computes the interference scale [1] which determines whether a genetic perturbation was successful
	@param\tsigs\tPandas DataFrame: rows/[genes] x columns/[control and treated samples]
	@param\tsamples\tPython integer list: contains 1 for control samples, 2 for treated ones for each column of @sigs
	@param\tentrez_id\tPython integer: EntrezID of the perturbed gene 
	@param\tis_oe\tPython bool: is the experiment an overexpression of the perturbed gene (is_oe=True) or a knockdown
	@param\tquiet\tPython bool[default=True]
	@param\teps\tPython float[default=2e-7]
	@return\tiscale\tPython float: interference scale for the input experiment
        [1] doi.org/10.1002/psp4.12107 
    '''
    assert len(sigs.columns) == len(samples)
    assert all([s in [1,2] for s in samples])
    assert entrez_id in sigs.index
    ## "Housekeeping genes" selected in PMC5192966
    treated = sigs.columns[np.array(samples)==2]
    control = sigs.columns[np.array(samples)==1] 
    exp_ratio = np.mean(sigs[treated].loc[entrez_id])/float(np.mean(sigs[control].loc[entrez_id])+eps)
    ctl_genes = [g for g in list(lincs_specific_ctl_genes.values()) if (g in sigs.index)]
    assert len(ctl_genes) > 0
    hk_inv_ratios = [np.mean(sigs[control].loc[g])/float(np.mean(sigs[treated].loc[g])+eps) for g in ctl_genes]
    ## The most stable (ratio ~ 1) between control and treated groups
    best_hk_gene_id = int(np.argmin(np.abs(np.array(hk_inv_ratios)-1)))
    iscale = exp_ratio*hk_inv_ratios[best_hk_gene_id]
    assert not pd.isna(iscale)
    iscale = iscale-1 if (is_oe) else 1-iscale
    if (not quiet):
        print("<INTERFERENCE SCALE> %s Perturbed %d = %.5f || Most stable control %s = %.5f" % ("OE" if (is_oe) else "KD", entrez_id,exp_ratio,ctl_genes[best_hk_gene_id],1./hk_inv_ratios[best_hk_gene_id]))
        print("<INTERFERENCE SCALE> %.5f" % iscale)
    return iscale

def get_treated_control_dataset(treatment, pert_type, cell, filters, entrez_ids, taxon_id, user_key, path_to_lincs, entrez_id=None, selection="distil_ss", dose=None, iunit=None, itime=None, which_lvl=[[3,5][1]], nsigs=2, same_plate=True, quiet=False, trim_w_interference_scale=True, return_metrics=[]):
    '''
        Retrieve set of experimental profiles, with at least @nsigs treated and control sample
        @param\ttreatment\tPython character string: HUGO gene symbol
        @param\tpert_type\tPython character string: type of perturbation as accepted by LINCS L1000
        @param\tcell\tPython character string: cell line existing in LINCS L1000
        @param\tfilters\tPython dictionary: additional parameters for the LINCS L1000 requests
        @param\tentrez_ids\tPython integer list: EntrezID genes
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @param\tuser_key\tPython character string: LINCS L1000 user API key
        @param\tpath_to_lincs\tPython character string: path where LINCS L1000 files are locally stored
        @param\tentrez_id\tPython integer: EntrezID identifier for HUGO gene symbol @treatment
        @param\tselection\tPython character string[default="distil_ss"]: LINCS L1000 metric which is maximized by a given experiment
        @param\tdose\tPython character string or None[default=None]
        @param\tiunit\tPython character string or None[default=None]
        @param\titime\tPython character string or None[default=None]
        @param\twhich_lvl\tPython integer list[default=[3]]
        @param\tnsigs\tPython integer[default=2]: minimal number of samples of each condition in each experiment
        @param\tsame_plate\tPython bool[default=True]: select samples from the same plate for each experiment and condition
        @param\tquiet\tPython bool[default=True]
        @param\ttrim_w_interference_scale\tPython bool[default=True]: computes the interference scale criteria for further trimming
        @param\treturn_metrics\tPython character string list: list of LINCS L1000 metrics to return as the same time as the profiles
        @return\tsigs\tPandas DataFrame: rows/[genes+"annotation"+"signame"+"sigid"] x columns/[profiles] or None
    '''
    assert all([x in [3,5] for x in which_lvl])
    assert len(which_lvl)==1
    assert user_key
    assert nsigs > 1
    if (pert_type == "trt_cp"):
        trim_w_interference_scale = False
    lincs_specific_ctl_genes = convert_ctrlgenes_EntrezGene(taxon_id)
    genes, genes_controls = list(set(entrez_ids)), list(set(list(lincs_specific_ctl_genes.values())+entrez_ids))
    ## 1. Build LINCS L1000 request to determine IDs of suitable experimental profiles
    where = {"cell_id": cell, "pert_type": pert_type}
    if (len(treatment) > 0):
        where.setdefault("pert_iname", treatment)
    if (dose is not None):
        where.setdefault("pert_dose", dose)
    if (itime is not None):
        where.setdefault("pert_itime", itime)
    # https://www.biostars.org/p/211896/
    cid = ("sig_id" if (which_lvl == 5) else "distil_id")
    fields = list(set(list(filters.keys())+[cid]+([selection] if (len(selection) > 0) else [])+return_metrics))
    params = { "where": where, "fields": fields+["pert_dose_unit"] }
    ## 2. Retrieve IDs of treated experiments (maximizing criterion @selection)
    data_treated = select_best_sig(params, filters, user_key, selection=selection, nsigs=nsigs, same_plate=same_plate, quiet=quiet, iunit=iunit)
    assert len(data_treated) > 0
    where = {"cell_id": cell, "pert_type": ("ctl_vehicle" if (pert_type == "trt_cp") else "ctl_vector")}
    if (same_plate):
        where.setdefault("brew_prefix", str(data_treated[0]["brew_prefix"][0]))
    ## 3. Retrieve IDs of associated control experiments 
    params = { "where": where, "fields": fields }
    data_control = select_best_sig(params, filters, user_key, selection=selection, nsigs=nsigs, same_plate=same_plate, quiet=quiet)
    assert len(data_control) > 0
    ## 4. Collect all IDs of treated and control experimental profiles 
    sig_ids = [str(x[cid]) for x in data_treated+data_control]
    sigs = create_restricted_drug_signatures(sig_ids, genes if (not trim_w_interference_scale) else genes_controls, path_to_lincs, which_lvl=which_lvl, strict=True)
    assert len(sigs) > 0
    ## 5. Get annotations (control/treated, profile/signature IDs) for each experimental profile
    signame_trt, signame_ctl = [[str(s[cid]) for s in x] for x in [data_treated, data_control]]
    samples = [2*int(c in signame_trt)+int(c in signame_ctl) for c in sigs.columns]
    assert all([s in [1,2] for s in samples])
    sigs.loc["annotation"] = samples
    ## 6. If returning other metrics, append them to the Pandas DataFrame
    if (len(return_metrics) > 0):
        dtc = data_treated+data_control
        metric_vals = pd.DataFrame([[x.get(metric, np.nan) for metric in return_metrics] for x in dtc], columns=sig_ids, index=return_metrics)
        sigs = pd.concat((sigs, metric_vals), axis=0)
    ## 7. Compute interference scale for genetic experiments if needed for further trimming
    if (trim_w_interference_scale):
        if (entrez_id not in sigs.index):
            return None
        select = [col for col in sigs.columns if (sigs.loc["annotation"][col] in [1, 2])]
        iscale = compute_interference_scale(sigs[select].drop(["annotation"]), sigs[select].loc["annotation"], entrez_id, (pert_type in ["trt_oe"]), taxon_id, lincs_specific_ctl_genes, quiet=quiet)
        sigs.loc["interference_scale"] = [iscale]*sigs.shape[1] ## common to all profiles from the same experiment (treated/control)
    sigs = sigs.loc[[g for g in genes if (g in sigs.index)]+["annotation"]+return_metrics+([] if (not trim_w_interference_scale) else ["interference_scale"])]
    return sigs

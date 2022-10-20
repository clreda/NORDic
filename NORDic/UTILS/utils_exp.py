# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import os

from NORDic.UTILS.LINCS_utils import *
from NORDic.UTILS.utils_state import binarize_experiments

def profiles2signatures(profiles_df, user_key, path_to_lincs, save_fname, backgroundfile=False, selection="distil_ss", thres=0.5, bin_method="binary", nbackground_limits=(4,30), quiet=False):
    '''
        Convert experimental profiles into signatures (1 for control samples, 1 for treated ones)
        @param\tprofiles_df\tPandas DataFrame: rows/[genes+annotations] x columns/[samples]
        @param\tuser_key\tPython character string: LINCS L1000 user API key
        @param\tpath_to_lincs\tPython character string: path to local LINCS L1000 files
        @param\tsave_fname\tPython character string: path to save normalized expression profiles per cell line
        @param\tbackground_file\tPython bool[default=False]: retrieves from LINCS L1000 supplementary expression values if set to True to compute more precise basal gene expression levels
        @param\tselection\tPython character string[default="distil_ss"]: LINCS L1000 metric to maximize for the "background" data
        @param\tthres\tPython float[default=0.5]: threshold for cutoff normalized gene expression values (in [0,0.5])
        @param\tbin_method\tPython character string[default="binary"]: binarization approach
        @param\tnbackground_limits\tPython integer tuple[default=(4,30)]: lower and upper bounds on the number of profiles for the background expression data
        @param\tquiet\tPython bool[default=False]
        @return\tsignatures_df\tPandas DataFrame: rows/[genes] x columns/[signature ID]
    '''
    assert thres >= 0 and thres <= 0.5
    assert path_to_lincs
    assert user_key
    assert bin_method in ["binary", "binary_CD"]
    selection_min, selection_max = nbackground_limits
    assert selection_min > 0
    assert selection_max >= selection_min
    cell_lines = list(set([x.split("_")[1] for x in list(profiles_df.loc["sigid"])]))
    signatures_list = []
    for icell, cell in enumerate(cell_lines):
        cell_save_fname = save_fname+"_"+cell+".csv"
        ## 1. For each cell line, split control and treated samples into two Pandas DataFrames
        profiles__df = profiles_df[[profiles_df.columns[ix] for ix, x in enumerate(profiles_df.loc["sigid"]) if (x.split("_")[1]==cell)]]
        initial_cols = profiles__df.loc["annotation"].apply(pd.to_numeric).values==1
        final_profiles = profiles__df[profiles__df.columns[~initial_cols]]
        conditions = list(set(final_profiles.loc["signame"]))
        initial_profiles = profiles__df[profiles__df.columns[initial_cols]].iloc[:-3,:].apply(pd.to_numeric)
        initial_profiles.columns = ["Ctrl_rep%d" % (i+1) for i in range(initial_profiles.shape[1])]
        final_profiles.columns = [x+"_%d" % (ix+1) for ix, x in enumerate(list(final_profiles.loc["signame"]))]
        final_profiles = final_profiles.iloc[:-3,:].apply(pd.to_numeric)
        assert final_profiles.shape[1]==np.sum(profiles__df.loc["annotation"].apply(pd.to_numeric).values==2)
        assert initial_profiles.shape[1]==np.sum(profiles__df.loc["annotation"].apply(pd.to_numeric).values==1)
        if (not os.path.exists(cell_save_fname)):
            if (not quiet):
                print("<UTILS_EXP> Cell %s (%d/%d)" % (cell, icell+1, len(cell_lines)), end="... ")
            ## 1'. If required, retrieve background expression data corresponding to the considered cell line
            if (backgroundfile):
                endpoint = "sigs"
                method = "filter"
                params = {
                        "where": {"cell_id": cell, "pert_type": "trt_sh"}, 
                        "fields": ["distil_cc_q75", selection, "pct_self_rank_q25", "distil_id", "brew_prefix"]
                }
                request_url = build_url(endpoint, method, params=params, user_key=user_key)
                response = requests.get(request_url)
                assert response.status_code == 200
                data = json.loads(response.text)
                data = [dt for dt in data if (("distil_id" in dt) and ("distil_cc_q75" in dt) and ("pct_self_rank_q25" in dt))]
                ## Select only "gold" signatures, as defined by LINCS L1000 
                ## https://clue.io/connectopedia/glossary#is_gold
                data = [dt for dt in data if ((len(dt["distil_id"])>1) and (dt["distil_cc_q75"]>=0.2) and (dt["pct_self_rank_q25"]<=0.05))]
                assert len(data)>0
                ## Select profiles maximizing the "selection" criterion
                mselection = np.min([dt[selection] for dt in data])
                max_selection = np.argsort(np.array([dt[selection] if (dt[selection]>=selection_min) else mselection for dt in data]))
                max_selection = max_selection[-min(selection_max,len(max_selection)):]
                vals_selection = [dt[selection] for dt in [data[i] for i in max_selection]]
                if (not quiet):
                    print("<UTILS_EXP> %d (good) profiles | %d (best) profiles (capped at 50 or min>=%d) (%s max=%.3f, min=%.3f)" % (len(data), selection_min, len(max_selection), selection, np.max(vals_selection), np.min(vals_selection)))
                bkgrnd = create_restricted_drug_signatures([did for dt in [data[i] for i in max_selection] for did in dt["distil_id"]], [int(g) for g in list(profiles__df.index)[:-3]], path_to_lincs, which_lvl=[3], strict=False)
                bkgrnd.index = [int(g) for g in bkgrnd.index]
            ## 2. Aggregate replicates by median values for signature ~ initial condition
            initial_profile = initial_profiles.median(axis=1)
            initial_profile = initial_profile.loc[~initial_profile.duplicated()]
            ## 3. Aggregate values across probes of the same gene
            final_profile = final_profiles.T
            final_profile.index = ["_".join(idx.split("_")[:-1]) for idx in final_profile.index]
            data = final_profile.groupby(level=0).median().T
            data.columns = conditions
            data = data.loc[~data.index.duplicated()]
            data["initial"] = initial_profile.loc[[i for i in data.index if (i in initial_profile.index)]]
            data.index = data.index.astype(int)
            if (backgroundfile):
                data = data.join(bkgrnd, how="inner")
            data.to_csv(cell_save_fname)
        data = pd.read_csv(cell_save_fname, index_col=0)
        ## 4. Binarize profiles
        signatures = binarize_experiments(data, thres=thres, method=bin_method.split("_CD")[0], strict=not ('CD' in bin_method))
        signatures = signatures[conditions+["initial"]]
        if ("_CD" in bin_method):
            for c in conditions:
                df = pd.concat((final_profiles[[col for col in final_profiles.columns if (c in col)]], initial_profiles), axis=1)
                samples = [int("Ctrl_" in col)+1 for col in df.columns]
                sigs = binarize_via_CD(df, samples=samples, binarize=1, nperm=10000)
                signatures[c] = list(sigs["aggregated"])
        signatures.columns = [s+"_"+cell for s in signatures.columns]
        signatures_list.append(signatures)
    signatures_df = signatures_list[0].join(signatures_list[1:], how="outer")
    return signatures_df

def get_experimental_constraints(cell_lines, pert_types, pert_di, taxon_id, selection, user_key, path_to_lincs, thres_iscale=None, nsigs=2, quiet=False):
    '''
    Retrieve experimental profiles from the provided cell lines, perturbation types, list of genes, in the given species (taxon ID)
    @param\tcell_lines\tPython character string list: cell lines present in LINCS L1000
    @param\tpert_types\tPython character string list: types of perturbations as supported by LINCS L1000
    @param\tpert_di\tPython dictionary (keys=Python character string, values=Python integer): associates HUGO gene symbols to their EntrezGene IDs
    @param\ttaxon_id\tPython integer: NCBI taxonomy ID
    @param\tselection\tPython character string: LINCS L1000 metric to maximize
    @param\tuser_key\tPython character string: LINCS L1000 user API key
    @param\tpath_to_lincs\tPython character string: path to local LINCS L1000 files
    @param\tthres_iscale\tPython float or None[default=None]: lower threshold on the interference scale which quantifies the success of a genetic experiment
    @param\tnsigs\tPython integer[default=2]: minimal number of profiles per experiment and condition
    @param\tquiet\tPython bool[default=False]
    @return\tsignatures\tPandas DataFrame: rows/[genes+annotations] x columns/[profile/signature IDs]
    '''
    assert str(thres_iscale) == "None" or thres_iscale >= 0
    assert len(cell_lines) > 0
    assert len(pert_types) > 0
    assert nsigs > 1
    entrez_ids, pert_inames = list(pert_di.values()), list(pert_di.keys())
    signatures, perturbed_genes = [], []
    ## 1. For each cell line
    for il, line in enumerate(cell_lines):
        ids = {}
        pert_types_ = [p for p in pert_types if (p != "ctl_untrt")]
        for ip, pert_type in enumerate(pert_types_):
            endpoint = "sigs"
            method = "filter"
            ## 2. Build the additional parameters for the LINCS L1000 request
            ## 2a. Select the type of perturbation and cell line
            where = {"pert_type": pert_type, "cell_id": line}
            ## 2b. Return the HUGO gene symbol of the perturbed gene, the perturbagen dose/type/cell line/exposure time
            ## and distil_id (unique to profile), brew_prefix (unique to set of replicates)
            params = { "where": where,
                    "fields": ["pert_iname", "pert_idose", "pert_type", "cell_id", "pert_itime", "distil_id", "brew_prefix"] }
            ## 3. Create URL based on those parameters
            #request_url = build_url(endpoint, method, params=params, user_key=user_key)
            ## 4. Retrieve the set of experiments matching those constraints, with enough replicates (>=nsigs)
            ## /!\ restriction due to limit=1000 in LINCS L1000 requests
            #data_pert_ = post_request(request_url, quiet=True)
            #data_pert_ = [dt for dt in data_pert_ if ((dt["pert_iname"] in pert_inames) and (len(dt["distil_id"])>=nsigs))]
            data_pert_ = []
            for ig, gene in enumerate(pert_inames):
                if (not quiet):
                    print("<UTILS_EXP> Gene %d/%d Cell %d/%d Type %d/%d" % (ig+1, len(pert_inames), il+1, len(cell_lines), ip+1, len(pert_types_)))
                p = {'where': {"pert_type": pert_type, "cell_id": line, "pert_iname": gene}, "fields": params["fields"]}
                res_pert_ = post_request(build_url(endpoint, method, params=p, user_key=user_key), quiet=True, pause_time=0.3)
                if (len(res_pert_)>0 and not quiet):
                    print("\t<UTILS_EXP> (%s,%s,%s): %d" % (gene, line, pert_type, len(res_pert_)))
                data_pert_ += [dt for dt in res_pert_ if (len(dt["distil_id"])>=nsigs)]
            for data in data_pert_:
                entrez_id = pert_di[data["pert_iname"]]
                if (not quiet):
                    print("<UTILS_EXP> %d experiments so far" % len(signatures))
                signame = "_".join([str(data["pert_iname"]), "OE" if ("_oe" in pert_type) else "KD", data["pert_idose"] if (data["pert_idose"]!="-666") else "NA"])
                treatment = str(data["pert_iname"])
                ## avoid duplicates
                if (treatment in perturbed_genes and not quiet):
                    print("\t<UTILS_EXP> Duplicated treatment:%s, cell:%s, type:%s" % (treatment, str(data["cell_id"]), str(data["pert_type"])))
                    continue
                elif (not quiet):
                    print("\t<UTILS_EXP> Treatment %s (entrez_id %d)... " % (treatment, entrez_id), end="")
                ## 6. Returns control & treated profiles from LINCS L1000
                sigs = get_treated_control_dataset(treatment, pert_type, line, {}, entrez_ids, taxon_id, user_key, path_to_lincs, entrez_id=entrez_id,
                        which_lvl=[3], nsigs=nsigs, same_plate=True, selection=selection, quiet=quiet, trim_w_interference_scale=(str(thres_iscale)!="None"))
                if (not quiet):
                    print("... %d genes, %d profiles" % (sigs.shape[0]-3, sigs.shape[1]))
                if (str(sigs)=="None" or len(sigs)==0):
                    continue
                perturbed_genes.append(treatment)
                sigs.loc["signame"] = [signame]*sigs.shape[1]
                sigs.loc["sigid"] = list(sigs.columns)
                nexp = len(signatures)+1
                sigs.columns = ["Exp"+str(nexp)+":"+str(i)+"-rep"+str(ai+1) for ai, i in enumerate(list(sigs.loc["annotation"]))]
                signatures.append(sigs)
    signatures = signatures[0].join(signatures[1:], how="outer")
    signatures = signatures.loc[~signatures.index.duplicated()]
    return signatures

#coding:utf-8

import imports

from NORDic.NORDic_NI.functions import network_identification, solution2cytoscape
from NORDic.UTILS.utils_grn import load_grn

seed_number=0

DISGENET_args = {
        "credentials": "credentials_DISGENET.txt",
        "disease_cids": ["C0014544"],
        "min_score":0,
        "min_ei":0,
        "min_dsi":0.25,
        "min_dpi":0,
}

STRING_args = {
        "credentials": "credentials_STRING.txt",
        "score": 0,
        "version": "v10.5",
}

LINCS_args = {
        "path_to_lincs": "../lincs/",
        "credentials": "credentials_LINCS.txt",
        "cell_lines": ["NPC", "SHSY5Y"],
        "pert_types": ["trt_sh", "trt_oe", "trt_xpr"],
        "selection": "distil_ss", 
        "thres_iscale": 0,
        "nsigs": 2,
}

EDGE_args = {
        "beta": 1,
        "tau": 0,
        "cor_method": "pearson",
        "filter": True,
	"connected": True,
}

SIG_args = {
        "bin_thres": 0.5,
        "bin_method": "binary",
}

DESIRABILITY = {"DS": 3, "CL": 3, "Centr": 3, "GT": 1}

BONESIS_args = {
        "exp_ids": [],
        "use_diverse": True,
        "limit": 1,
        "niterations": 1,
        "exact": True,
        "max_maxclause": 3,
}

STRING_args.update({"score": 0.31, "beta": 1})
EDGE_args.update({"tau": 0, "filter": True, "connected": True})
SIG_args.update({"bin_thres": 0.265})
LINCS_args.update({"thres_iscale": 0})
BONESIS_args.update({"limit": 100, "exact": False, "max_maxclause":4})

force_experiments=True
accept_nonRNA=False

## Download epilepsy-related data
from download_Refractory_Epilepsy_Data import get_EPILEPSY_genes, file_folder, path_to_genes
get_EPILEPSY_genes()

taxon_id=9606 # human
from multiprocessing import cpu_count
njobs=max(1,cpu_count()-2)

NETWORK_fname = None
solution = network_identification(file_folder, taxon_id, path_to_genes, disgenet_args=DISGENET_args, string_args=STRING_args, lincs_args=LINCS_args, edge_args=EDGE_args, sig_args=SIG_args, bonesis_args=BONESIS_args, weights=DESIRABILITY, seed=seed_number, network_fname=NETWORK_fname, njobs=njobs, force_experiments=force_experiments, accept_nonRNA=accept_nonRNA)
if (solution is not None):
    solution = load_grn(file_folder+"solution.bnet")
    ## Convert to Cytoscape-readable file
    solution2cytoscape(solution, file_folder+"solution_cytoscape")

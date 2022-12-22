#coding:utf-8

import pandas as pd
import numpy as np
import json
import os
from scipy.stats.mstats import gmean
import random
from joblib import Parallel, delayed

import mpbn
import mpbn_sim
from tqdm import tqdm
import gc

from NORDic.UTILS.utils_state import compare_states

####################
## Spread process ##
####################

def compute_similarities(f, x0, A, A_WT, gene_outputs, nb_sims, experiments, repeat=1, exp_name="", quiet=False):
    '''
        Compute similarities between any attractor in WT and in mutants, weighted by their probabilities
        @param\tf\tBoolean Network (MPBN) object: the mutated network
        @param\tx0\tMPBN object: initial state
        @param\tA\tAttractor list: list of attractors in mutant
        @param\tA_WT\tAttractor list: list of attractors in WT
        @param\tgene_outputs\tPython character string list: list of node names to check
        @param\tnb_sims\tPython integer: number of iterations to compute the probabilities
        @param\texperiments\tPython dictionary list: list of experiments (different rates/depths)
        @param\trepeat\tPython integer[default=1]: how many times should these experiments be repeated
        @param\texp_name\tPython character string[default=""]: printed info about the experiment (if quiet=True)
        @param\tquiet\tPython bool[default=False]
        @return\tsim\tPython float: change in attractors induced by the mutation
    '''
    for exp in experiments:
        if "name" not in exp:
            continue
        dists = []
        for _ in range(repeat):
            rates = getattr(mpbn_sim, f"{exp['rates']}_rates")
            depth = getattr(mpbn_sim, f"{exp['depth']}_depth")
            rates_args = exp.get("rate_args", {})
            depth_args = exp.get("depth_args", {})
            if (not quiet):
                print(exp_name+" "*int(len(exp_name)>0)+(f"- {depth.__name__}{depth_args}\t{rates.__name__}{rates_args}"))
            probs = mpbn_sim.estimate_reachable_attractor_probabilities(f, x0, A, nb_sims, depth(f, **depth_args), rates(f, **rates_args))
            attrs = pd.DataFrame({"MUT_%d"%ia: a for ia, a in enumerate(A)}).replace("*",np.nan).astype(float)
            probs = {i: x for i,x in list(probs.items()) if (x>0)}
            attrs = attrs[[attrs.columns[i] for i in list(probs.keys())]]
            ## if too many attractors, select the most common ones
            if (attrs.shape[1]>45000):
                idx_common = np.argsort([probs[i] for i in range(len(probs))]).tolist()
                idx_common.reverse()
                idx_common = idx_common[:45000]
                attrs = attrs[[attrs.columns[i] for i in idx_common]]
                probs = {i:probs[i] for i in idx_common}
                sprobs = sum(list(probs.values()))
                probs = {i:probs[i]*100/sprobs for i in probs}
            probs = np.array([probs[ia]/100 for ia in probs])
            attrs_init = pd.DataFrame({"WT_%d"%ia: a for ia, a in enumerate(A_WT)}).replace("*",np.nan).astype(float)
            sims, nb_gene = compare_states(attrs, attrs_init, gene_outputs)
            assert nb_gene == len(gene_outputs)
            sims = probs.T.dot(sims)
            dt = 1-np.max(sims) #max: minimum of change in (*different*) attractors induced by the subset S
            dists.append(dt)
    return np.mean(dists) if (repeat>1) else dists[0]

def run_experiments(network_name, spreader, gene_list, state, gene_outputs, simu_params, quiet=False):
    ## Create file
    import json
    from subprocess import call as sbcall
    perts, perts_S = [(state.loc[[g for g in lst if (g in state.index)]]+1)%2 for lst in [gene_list, spreader]]
    experiments, nb_sims = [{"name": "mpsim", "rates": simu_params.get("rates", "fully_asynchronous"), "depth": simu_params.get("depth", "constant_unitary")}], simu_params["nb_sims"]
    experiments_di = {
        "bnet_file": network_name,
        "init_active": list(state.loc[state[state.columns[0]]==1].index),
        "nb_sims": nb_sims,
        "mutants": {g+"_"+("KO" if (perts.loc[g][perts.columns[0]]==0) else "OE")+" {"+str(spreader)+"}": {gx: str(pd.concat((perts,perts_S),axis=0).loc[gx][perts.columns[0]]) for gx in [g]+spreader} for g in list(perts.index)},
        "experiments": experiments,
    }
    with open("experiments.json", "w") as f:
        json.dump(experiments_di)
    sbcall("mpbn_sim --save experiments.json", shell=True)

def spread(network_name, spreader, gene_list, state, gene_outputs, simu_params, seednb=0, quiet=False):
    '''
        Compute the spread of each gene in @gene_inputs+@spreader with initial state @state on genes @gene_outputs
        Here, the (single state) spread is defined as the indicator of the emptyness of the intersection between WT and mutant attractors
        @param\tnetwork_name\tPython character string: filename of the network in .bnet (needs to be pickable)
        @param\tspreader\tPython character string list: subset of node names
        @param\tgene_list\tPython character string list: list of node names to perturb in addition to the spreader
        @param\tstate\tPandas DataFrame: binary initial state rows/[genes] x columns/[values in {-1,0,1}]
        @param\tgene_outputs\tPython character string list: list of node names to check
        @param\tsimu_params\tPython dictionary: arguments to MPBN-SIM
        @param\tseednb\tPython integer[default=0]
        @param\tquiet\tPython bool[default=False]
        @return\tspds\tPython float dictionary: change in mutant attractor states for each gene in @gene_list
        that is, the similarity between any attractor reachable from @state in WT and any in mutant spreader+{g} where g in gene_list
    '''
    random.seed(seednb)
    np.random.seed(seednb)
    ## 1. Load the Boolean network
    f = mpbn.load(network_name)
    ## 2. Create the initial profile
    if (not quiet):
        print("\t<NORD_PMR> Initial state %s (gene(s):%s)" % (state.columns[0], gene_list[0]))
    x0 = f.zero()
    for i in list(state.loc[state[state.columns[0]]==1].index):
        x0[i] = 1
    ## 3. Get the reachable attractors from initial state in the absence of perturbation ("wild type")
    experiments, nb_sims = [{"name": "mpsim", "rates": simu_params.get("rates", "fully_asynchronous"), "depth": simu_params.get("depth", "constant_unitary")}], simu_params["nb_sims"]
    A_WT = [a for a in tqdm(list(f.attractors(reachable_from=x0)))]
    exp = experiments[0]
    rates = getattr(mpbn_sim, f"{exp['rates']}_rates")
    depth = getattr(mpbn_sim, f"{exp['depth']}_depth")
    rates_args = exp.get("rate_args", {})
    depth_args = exp.get("depth_args", {})
    probs_WT = mpbn_sim.estimate_reachable_attractor_probabilities(f, x0, A_WT, nb_sims, depth(f, **depth_args), rates(f, **rates_args))
    probs_WT = {i: x for i,x in list(probs_WT.items()) if (x>0)}
    A_WT = [A_WT[i] for i in list(probs_WT.keys())]
    if (not quiet):
        print("%d wild type attractors with proba > 0 (initial state %s)" % (len(A_WT), state.columns[0]))
    ## if too many attractors, select the most common ones
    if (len(A_WT)>45000):
        idx_common_WT = np.argsort([probs_WT[i] for i in range(len(probs_WT))]).tolist()
        idx_common_WT.reverse()
        idx_common_WT = idx_common_WT[:45000]
        A_WT = [A_WT[i] for i in idx_common_WT]
        if (not quiet):
            print("> reduced to %d wild type attractors (initial state %s, %d perc. of all attractors)" % (len(A_WT), np.sum([probs_WT[i] for i in idx_common_WT])))
    ## 4. Create the mutated networks
    def patch_model(f, patch):
        f = mpbn.MPBooleanNetwork(f)
        for i, fi in patch.items():
            f[i] = fi
        return f
    perts, perts_S = [(state.loc[[g for g in lst if (g in state.index)]]+1)%2 for lst in [gene_list, spreader]]
    mutants = {g: {gx: str(pd.concat((perts,perts_S),axis=0).loc[gx][perts.columns[0]]) for gx in [g]+spreader} for g in list(perts.index)}
    f_mutants = {name: patch_model(f, patch) for name, patch in mutants.items()}
    ## 5. Get the reachable attractors from initial state in the presence of mutations ("mutants" KO/OE)
    ## 6. Estimate probabilities of attractors from Mutants and compute similarities
    spds = [0 if (name_g not in f_mutants) else compute_similarities(f_mutants[name_g], x0, [a for a in tqdm(list(f_mutants[name_g].attractors(reachable_from=x0)))], A_WT, [g for g in gene_outputs if (g not in [name_g]+spreader)], nb_sims, experiments, exp_name="Gene %s (%d/%d) in state %s" % (name_g, ig+1, len(gene_list), state.columns[0]), quiet=quiet) for ig, name_g in enumerate(gene_list)]
    return spds

def spread_multistate(network_name, spreader, gene_list, states, gene_outputs, im_params, simu_params, quiet=False):
    '''
        Compute the spread of each gene in @gene_inputs+@spreader with initial states in @states on genes @gene_outputs
        Here, the (single state) spread is defined as the indicator of the emptyness of the intersection between WT and mutant attractors
        @param\tnetwork_name\tPython character string: filename of the network in .bnet (needs to be pickable)
        @param\tspreader\tPython character string list: subset of node names
        @param\tgene_list\tPython character string list: list of node names to perturb in addition to the spreader
        @param\tstates\tPandas DataFrame: binary initial state rows/[genes] x columns/[state ID]
        @param\tgene_outputs\tPython character string list: list of node names to check
        @param\tim_params\tPython dictionary: arguments to Influence Maximization
        @param\tsimu_params\tPython dictionary: arguments to MPBN-SIM
        @param\tquiet\tPython bool[default=False]
        @return\tspds\tPython float dictionary: change in mutant attractor states for each gene in @gene_list
        that is, the geometric mean of similarities between any attractor reachable from state in @states in WT and any in mutant spreader+{g} where g in gene_list
    '''
    from multiprocessing import cpu_count
    assert simu_params.get('thread_count', 1)>=1 and simu_params.get('thread_count', 1)<=max(1,cpu_count()-2)
    if (simu_params.get('thread_count', 1)==1):
        sprds_multistate = [spread(network_name, spreader, gene_list, states[[col]], gene_outputs, simu_params, seednb=im_params.get("seed", 0)) for col in states.columns]
    else:
        if (states.shape[1]>1):
            sprds_multistate = Parallel(n_jobs=simu_params['thread_count'], backend='loky')(delayed(spread)(network_name, spreader, gene_list, states[[col]], gene_outputs, simu_params, seednb=im_params.get("seed", 0)) for col in states.columns)
        else:
            sprds_multistate = Parallel(n_jobs=simu_params['thread_count'], backend='loky')(delayed(spread)(network_name, spreader, [gene], states[[col]], gene_outputs, simu_params, seednb=im_params.get("seed", 0)) for col in states.columns for gene in gene_list)
    ## Aggregate the values across the set of initial states
    ## Genes which are not measured in the states are assigned value 0
    if (states.shape[1]>1):
        spds = [(gmean([(s[ig]+1) for s in sprds_multistate])-1) for ig, g in enumerate(gene_list)]
    else:
        spds = [sprds_multistate[ig][0] for ig, g in enumerate(gene_list)]
    return spds

#######################################
## INFLUENCE MAXIMIZATION ALGORITHM  ##
#######################################

def greedy(network_name, k, states, im_params, simu_params, save_folder=None, quiet=False):
    '''
        Greedy Influence Maximization Algorithm [Kempe et al., 2003]
        Finds iteratively the maximum spreader and adds it to the list until the list is of size k
        @param\tnetwork_name\tPython character string: bnet network
        @param\tk\tPython integer: maximum size of the spreader
        @param\tim_params\tPython dictionary or None[default=None]: parameters of the influence maximization
        @param\tstates\tPandas DataFrame or None[default=None]: list of initial states to consider
        @param\tsave_folder\tPython character string[default=None]: where to save intermediary results (if None: do not save intermediary results)
        @param\tquiet\tPython bool[default=False]
        @return\tS, spreads\tPython character string list: nodes in the spreader set, Python dictionary: spread value associated with every tested subset of nodes
    '''
    random.seed(im_params.get("seed", 0))
    np.random.seed(im_params.get("seed", 0))
    if (im_params.get("gene_inputs", None) is None):
        with open(solution_fname, "r") as f:
            network = str(f.read())
        genes = [x.split(", ")[0] for x in network.split("\n")[:-1]]
        gene_inputs = genes
    else:
        gene_inputs = im_params["gene_inputs"]
    if (im_params.get("gene_outputs", None) is None):
        with open(solution_fname, "r") as f:
            network = str(f.read())
        gene_outputs = [x.split(", ")[0] for x in network.split("\n")[:-1] if (x.split(", ")[1] not in [x.split(", ")[0], "0", "1"])]
    else:
        gene_outputs = im_params["gene_outputs"]
    S, S_unfold, spreads, start_k = [], [], {}, 0
    if (save_folder is not None and os.path.exists(save_folder+"application_regulators.json")):
        with open(save_folder+"application_regulators.json", "r") as f:
            res = json.loads(f.read())
        if (len(res)>0):
            spreads, S, S_unfold, start_k = res["spreads"], res["S"], res["S_unfold"], res["k"]
    S_spread = -1 if (len(spreads)==0 and len(S)==0) else np.max(list(spreads[str(S)].values()))
    for current_k in range(start_k+1, k+1):
        if (not quiet):
            print("<NORD_PMR> Iteration k=%d" % (current_k))
        gene_list = [g for g in gene_inputs if (g not in S_unfold)]
        ## List of Python floats in the order gene_list
        sprds_lst = spread_multistate(network_name, S_unfold, gene_list, states, gene_outputs, im_params, simu_params, quiet=quiet)
        ## can't find spreader that strictly increases the spread of the previous spreader
        sprds_arr = np.array(sprds_lst)
        if (np.max(sprds_arr)<=S_spread):
            if (not quiet):
                print("\n<NORD_PMR> /!\ Can't find a more performant spreader for k=%d" % current_k)
            break
        ## Keep ex-aequo
        ex_aequo = np.argwhere(sprds_arr==np.max(sprds_arr)).flatten().tolist()
        S_spread, nodes = np.max(sprds_arr), [gene_list[x] for x in ex_aequo]
        if (len(nodes)>1):
            nodes = tuple(nodes)
            S_unfold += nodes
            S += [nodes]
        else:
            nodes = nodes[0]
            S_unfold += [nodes]
            S += [[nodes]]
        spreads.update({str(S):{g: sprds_lst[ig] if (g not in S_unfold) else S_spread for ig, g in enumerate(gene_list)}})
        with open(save_folder+"application_regulators.json", "w") as f:
            json.dump({"spreads": spreads, "S": S, "S_unfold": S_unfold, "k": current_k}, f)
            pd.DataFrame(spreads).to_csv(save_folder+"application_regulators.csv")
    spreads_df = pd.DataFrame(spreads)
    return S, spreads_df

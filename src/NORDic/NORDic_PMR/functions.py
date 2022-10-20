#coding:utf-8

#from glob import glob
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

from NORDic.UTILS.utils_state import compare_states

####################
## Spread process ##
####################

def compute_similarities(f, x0, A, A_WT, gene_outputs, nb_sims, experiments, repeat=1, quiet=False):
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
                print(f"- {depth.__name__}{depth_args}\t{rates.__name__}{rates_args}")
            probs = mpbn_sim.estimate_reachable_attractor_probabilities(f, x0, A, nb_sims, depth(f, **depth_args), rates(f, **rates_args))
            attrs = pd.DataFrame({ia: a for ia, a in enumerate(A) if ("__masked__" not in a)})
            probs = np.array([probs[ia]/100. for ia in attrs.columns])
            attrs.columns = ["MUT_%d" % i for i in range(attrs.shape[1])]
            attrs_init = pd.DataFrame({ia: a for ia, a in enumerate(A_WT) if ("__masked__" not in a)})
            attrs_init.columns = ["WT_%d" % i for i in range(attrs_init.shape[1])]
            #print(pd.concat((attrs, attrs_init), axis=1).loc[gene_outputs])
            sims, nb_gene = compare_states(attrs, attrs_init, gene_outputs)
            assert nb_gene == len(gene_outputs)
            sims = np.round(float(probs.reshape(sims.shape).T.dot(sims)), 2)
            dt = 1-np.max(sims) #max: minimum of change in (*different*) attractors induced by the subset S
            dists.append(dt)
    return np.mean(dists)

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
    perts, perts_S = [(state.loc[[g for g in lst if (g in state.index)]]+1)%2 for lst in [gene_list, spreader]]
    mutants = {g+"_"+("KO" if (perts.loc[g][perts.columns[0]]==0) else "OE")+" {"+str(spreader)+"}": {gx: str(pd.concat((perts,perts_S),axis=0).loc[gx][perts.columns[0]]) for gx in [g]+spreader} for g in list(perts.index)}
    experiments = [{"name": "mpsim", "rates": simu_params.get("rates", "fully_asynchronous"), "depth": simu_params.get("depth", "constant_unitary")}]
    ## 1. Load the Boolean network
    f = mpbn.load(network_name)
    ## 2. Create the initial profile
    if (not quiet):
        print("\t<NORD_PMR> Initial state %s" % state.columns[0])
    x0 = f.zero()
    for i in list(state.loc[state[state.columns[0]]==1].index):
        x0[i] = 1
    ## 3. Get the reachable attractors from initial state in the absence of perturbation ("wild type")
    A = [a for a in tqdm(list(f.attractors(reachable_from=x0)))]
    if (not quiet):
        print("%d wild type attractors (initial state %s)" % (len(A), state.columns[0]))
    ## 4. Create the mutated networks
    def patch_model(f, patch):
        f = mpbn.MPBooleanNetwork(f)
        for i, fi in patch.items():
            f[i] = fi
        return f
    f_mutants = {name: patch_model(f, patch) for name, patch in mutants.items()}
    ## 5. Get the reachable attractors from initial state in the presence of mutations ("mutants" KO/OE)
    for ni, name in enumerate(f_mutants):
        f_muted, NnewA = f_mutants[name], 0
        for a in tqdm(list(f_muted.attractors(reachable_from=x0))):
            if a not in A:
                NnewA += 1
                A.append(a)
        if (not quiet):
            print("%d new mutant %s attractors (%d/%d) (initial state %s)" % (NnewA, name, ni+1, len(f_mutants), state.columns[0]))
    ## 6. Estimate probabilities of attractors from Mutants and compute similarities
    nb_sims = simu_params["nb_sims"]
    A_WT = list(f.attractors(reachable_from=x0))
    spds = {}
    for name, f_mut in list(f_mutants.items()):
        if (not quiet):
            print("* %s" % name, end="\t")
        B = list(f_mut.attractors(reachable_from=x0))
        myA = [a if (a in B) else {"__masked__": '*'} for a in A]
        sim = compute_similarities(f_mut, x0, myA, A_WT, [g for g in gene_outputs if (g not in [name.split("_")[0]]+spreader)], nb_sims, experiments, quiet=quiet)
        spds.setdefault(name.split("_")[0], sim)
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
    if (im_params.get("njobs", 1)==1):
        sprds_multistate = [spread(network_name, spreader, gene_list, states[[col]], gene_outputs, simu_params, seednb=im_params.get("seed", 0)) for col in states.columns]
    else:
        sprds_multistate = Parallel(n_jobs=im_params["njobs"], backend='loky')(delayed(spread)(network_name, spreader, gene_list, states[[col]], gene_outputs, simu_params, seednb=im_params.get("seed", 0)) for col in states.columns)
    all_dfs = [pd.DataFrame({states.columns[i]: di}) for i, di in enumerate(sprds_multistate)]
    spds = {g: (gmean([(s[g]+1) for s in sprds_multistate])-1) for g in gene_list} # aggregate the values across the set of initial states
    all_dfs += [pd.DataFrame({"aggregated": spds})]
    all_spds = all_dfs[0].join(all_dfs[1:], how="outer")
    if (not quiet):
        print("* S = %s (all states)\n%s" % (str(spreader), str(all_spds)))
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
        genes = [x.split(", ")[0] for x in network.split("\n")[:-1]]
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
            print("<NORD_PMR> Iteration k=%d" % (current_k+1))
        gene_list = [g for g in gene_inputs if (g not in S_unfold)]
        sprds = spread_multistate(network_name, S_unfold, gene_list, states, gene_outputs, im_params, simu_params, quiet=quiet)
        ## can't find spreader that strictly increases the spread of the previous spreader
        sprds_arr = np.array([sprds[g] for g in gene_list])
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
        spreads.update({str(S):{g: sprds[g] if (g not in S_unfold) else S_spread for g in gene_list+S_unfold}})
        with open(save_folder+"application_regulators.json", "w") as f:
            json.dump({"spreads": spreads, "S": S, "S_unfold": S_unfold, "k": current_k}, f)
            pd.DataFrame(spreads).to_csv(save_folder+"application_regulators.csv")
    spreads_df = pd.DataFrame(spreads)
    return S, spreads_df

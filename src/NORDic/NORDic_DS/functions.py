# coding:utf-8

#from glob import glob
import pandas as pd
import numpy as np
#import subprocess as sb
#import sys
#import os
#from copy import deepcopy
#import maboss

from joblib import Parallel, delayed
import mpbn
import mpbn_sim
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import fbeta_score as Fbeta
from random import seed as rseed
from scipy.stats import ks_2samp, spearmanr, kendalltau

#from NORDic.UTILS.utils_grn import load_grn

#from params import *
#from utils_grn import predict, load_grn, load_ensemble_grn, set_initial_states, get_states_proba, solution2influences
#from utils_state import binarize_experiments, compare_states
#sys.path.insert(1, root_folder+'utils/')
#sys.path.insert(1, root_folder+'utils/credentials/')
#from LINCS_utils import *
#prediction_score = lambda C1, C2, S : compare_states(C1, C2, S)[0]
#from sklearn.metrics.pairwise import cosine_similarity

#################################################
## Baseline (L1000 CDS^2) [Duan et al., 2016]  ##
#################################################
def baseline(signatures, phenotype, is_binary=False):
    '''
        Compute the cosine scores between a set of signatures and a differential phenotype
        @param\tS\tPandas DataFrame: signatures rows/[genes] x columns/[drug names] [Treated || Control]
        @param\tP\tPandas DataFrame: differential phenotype rows/[genes] x column/[disease name] [Diseased || Healthy]
        @param\tis_binary\tPython bool[default=False]: if set to True, the signatures and phenotypes might be binary (that is, with values in {0,1})
        @return\tscores\tPython float dictionary: dictionary (keys=drug names, values=scores), the higher the score, the higher the repurposing power
    '''
    SC, PC = signatures.dropna(), phenotype.dropna()
    if (is_binary):
        SC[SC>0] = 1
        PC[PC>0] = 1
        SC[SC<0] = -1
        PC[PC<0] = -1
        SC[pd.isnull(SC)] = 0
        PC[pd.isnull(PC)] = 0
    C = SC.join(PC, how="inner")
    cosscores = pairwise_distances(C[signatures.columns].T,C[phenotype.columns].T, metric="cosine")
    baseline_df = pd.DataFrame(cosscores, columns=["Cosine Score"], index=signatures.columns)
    return baseline_df

#######################################
## Compute accuracy of the predictor ##
#######################################
def empirical_pvalue(sorted_rewards, sorted_ground_truth, nperms, method="ks_2samp"):
    '''
        Compute an empirical p-value corresponding to testing whether the distributions in the predicted scores and the ground truth are similar
        by randomly permuting the values of the predictions and averaging the number of significant statistics across all permutations
        @param\tsorted_rewards\tPython float list: predicted scores sorted by genes of increasing predicted value
        @param\tsorted_ground_truth\tPython float list: ground truth scores sorted by genes of increasing predicted value
        @param\tnperms\tPython integer: number of permutations to perform
        @param\tmethod\tPython character string[default="ks_2samp"]: statistical test to perform at each permutation (should belong to scipy.stats)
        @return\tpvalue\tPython float: empirical p-value corresponding to the test across all @nperms permutations
    '''
    assert method in ["ks_2samp", "spearmanr", "kendalltau"]
    val_true, _ = eval(method)(sorted_rewards, sorted_ground_truth)
    p_cumsum, N = 0, len(sorted_rewards)
    for nperm in range(nperms):
        rseed(nperm)
        ## Sample a random ordering of elements
        ids__ = np.random.choice(range(N), size=N, replace=False, p=[1/N]*N)
        new_rewards = [sorted_rewards[i] for i in ids__]
        val_new, _ = eval(method)(new_rewards, sorted_ground_truth)
        p_cumsum += int(val_new >= val_true)
    return p_cumsum/nperms

def compute_metrics(rewards, ground_truth, K=[2,5,10], use_negative_class=False, nperms=10000, thres=0., beta=1.):
    '''
        Compute AUC, Hit Ratio @ k of method for positive/negative class with p-value
        @param\trewards\tPython float list: predicted scores
        @param\tground_truth\tPython float list: ground truth scores
        @param\tK\tPython integer list[default=[2,5,10]]: ranks at which the hit ratio should be computed
        @param\tuse_negative_class\tPython bool[default=False]: if set to True, compute the performance with respect to the negative class instead of the positive class
        @param\tnperms\tPython integer[default=10000]: number of permutations to perform
        @param\tthres\tPython float[default=0.]: decision threshold to determine the positive (resp. negative) class
        @param\tbeta\tPython float[default=1.]: value of the coefficient in the F-measure
        @return\tres_di\tPython dictionary: (keys=metrics, values=values of the metrics)
    '''
    ids = np.argsort(rewards).tolist()
    ids.reverse()
    sorted_rewards = [rewards[i] for i in ids]
    sorted_decisions = [int(rewards[i]>thres) for i in ids]
    sorted_ground_truth = [int(ground_truth[i]<0) if (use_negative_class) else int(ground_truth[i]>0) for i in ids]
    hr = [ACC(sorted_ground_truth[:k], sorted_decisions[:k]) for k in K]
    auc = AUC(sorted_ground_truth, sorted_rewards)
    acc = ACC(sorted_ground_truth, sorted_decisions)
    f_score = Fbeta(sorted_ground_truth, sorted_decisions, average='weighted', beta=beta)
    p = empirical_pvalue(sorted_rewards, sorted_ground_truth, nperms)
    res_di = {'AUC': np.round(auc,3), "ACC": np.round(acc,3), "p": np.round(p,3), ("F_%s") % beta: f_score}
    res_di.update({'HR@%d' % k : hr[ik] for ik, k in enumerate(K)})
    return res_di

######################
## DRUG SIMULATOR   ##
######################

def simulate(network_fname, targets, phenotypes, simu_params, nbseed=0, quiet=False):
    '''
        Simulate and score the individual effects of drugs on patient phenotypes, compared to controls
        @param\tnetwork_fname\tPython character string: (relative) path to a network .BNET file
        @param\ttargets\tPandas DataFrame: rows/[genes+annotation patient/control] x columns/[drugs to test] (either 1: activatory, -1: inhibitory, 0: no regulation).
        @param\tphenotypes\tPandas DataFrame: rows/[genes] x columns/[drugs to test] (either 1: active expression, -1: inactive expression, 0: undetermined expression)
        The last line "annotation" is 1 (healthy sample) or 2 (patient sample).
        @param\tsimu_params\tTODO
        @param\tnbseed\tPython integer[default=0]
        @param\tquiet\tPython bool[default=False]
        @return\tscores\tPandas DataFrame: rows/[patient phenotypes] x columns/[drug names], values are the associated scores
    '''
    from random import seed as rseed
    rseed(nbseed)
    np.random.seed(nbseed)
    ## 1. Classification model b/w healthy and patient phenotypes (to classify final attractor states from treated patients)
    frontier = compute_frontier(phenotypes)
    patients = phenotypes[[c for c in phenotypes.columns if (phenotypes.loc["annotation"][c]==2)]].loc[[g for g in phenotypes.index if (g != "annotation")]]
    ## 2. Compute one score per drug and per patients
    scores = [simulate_treatment(network_fname, targets, frontier, patients[[Patient]], simu_params, quiet=quiet) for Patient in patients.columns]
    scores = pd.DataFrame(scores, index=patients.columns, columns=targets.columns)
    return scores

def compute_frontier(phenotypes, nbseed=0, quiet=False):
    '''
        Fit a model to classify control/treated phenotypes
        @param\tphenotypes\tPandas DataFrame: rows/[genes] x columns/[drugs to test] (either 1: active expression, -1: inactive expression, 0: undetermined expression)
        The last line "annotation" is 1 (healthy sample) or 2 (patient sample).
        @param\tnbseed\tPython integer[default=0]
        @param\tquiet\tPython bool[default=False]
        @return\tmodel\tPython object with a function "predict" that returns predictions (1: control, or 2: treated) on phenotypes
    '''
    from sklearn import svm
    model = svm.SVC(random_state=nbseed)
    model.fit(phenotypes.loc[[g for g in phenotypes.index if (g != "annotation")]].values.T, list(phenotypes.loc["annotation"]))
    acc = np.mean([int(x==y) for x,y in zip(list(model.predict(phenotypes.loc[[g for g in phenotypes.index if (g != "annotation")]].values.T)), list(phenotypes.loc["annotation"]))])
    if (not quiet):
        print("<NORD_DS> Accuracy of the model %.2f" % acc)
    return model

def compute_score(f, x0, A, frontier, nb_sims, experiments, repeat=1, exp_name="", quiet=False):
    '''
        Compute similarities between any attractor in WT and in mutants, weighted by their probabilities
        @param\tf\tBoolean Network (MPBN) object: the mutated network
        @param\tx0\tMPBN object: initial state
        @param\tA\tAttractor list: list of attractors in mutant network
        @param\tfrontier\tPython object: model to classify phenotypes
        @param\tnb_sims\tPython integer: number of iterations to compute the probabilities
        @param\texperiments\tPython dictionary list: list of experiments (different rates/depths)
        @param\trepeat\tPython integer[default=1]: how many times should these experiments be repeated
        @param\texp_name\tPython character string[default=""]: printable for an experiment
        @param\tquiet\tPython bool[default=False]
        @return\tscore\tPython float: change in attractors induced by the mutation
    '''
    for exp in experiments:
        if "name" not in exp:
            continue
        d_scores = []
        for _ in range(repeat):
            rates = getattr(mpbn_sim, f"{exp['rates']}_rates")
            depth = getattr(mpbn_sim, f"{exp['depth']}_depth")
            rates_args = exp.get("rate_args", {})
            depth_args = exp.get("depth_args", {})
            if (not quiet):
                print(exp_name+" "*int(len(exp_name)>0)+(f"- {depth.__name__}{depth_args}\t{rates.__name__}{rates_args}"))
            probs = mpbn_sim.estimate_reachable_attractor_probabilities(f, x0, A, nb_sims, depth(f, **depth_args), rates(f, **rates_args))
            attrs = pd.DataFrame({"MUT_%d"%ia: a for ia, a in enumerate(A)}).replace("*",np.nan).astype(float)
            probs = np.array([probs[ia]/100. for ia in range(attrs.shape[1])])
            classification_attrs = (frontier.predict(attrs.values.T)==1).astype(int) #classifies into 1:control, 2:patient
            drug_score = probs.T.dot(classification_attrs)
            d_scores.append(drug_score)
    return np.mean(d_scores) if (repeat>1) else d_scores[0]

def simulate_treatment(network_name, targets, frontier, state, simu_params, quiet=False):
    '''
        Compute the score assigned to a drug with targets in @targets[[drug]] in network
        @param\tnetwork_name\tPython character string: filename of the network in .bnet (needs to be pickable)
        @param\ttargets\tPandas DataFrame: rows/[genes] x columns/[columns]
        @param\tfrontier\tPython object: model to classify phenotypes
        @param\tstate\tPandas DataFrame: binary patient initial state rows/[genes] x columns/[values in {-1,0,1}]
        @param\tsimu_params\tPython dictionary: arguments to MPBN-SIM
        @param\tseednb\tPython integer[default=0]
        @param\tquiet\tPython bool[default=False]
        @return\teffects\tPython float dictionary: distance from attractors from treated networks to control profiles
    '''
    ## 1. Load the Boolean network
    f = mpbn.load(network_name)
    ## 2. Create the initial profile
    if (not quiet):
        print("\t<NORD_DS> Initial state %s" % state.columns[0])
    x0 = f.zero()
    for i in list(state.loc[state[state.columns[0]]==1].index):
        x0[i] = 1
    ## 3. Create the mutated networks
    def patch_model(f, patch):
        f = mpbn.MPBooleanNetwork(f)
        for i, fi in patch.items():
            f[i] = fi
        return f
    experiments, nb_sims = [{"name": "mpsim", "rates": simu_params.get("rates", "fully_asynchronous"), "depth": simu_params.get("depth", "constant_unitary")}], simu_params["nb_sims"]
    mutants = {t: {gx: str(int(targets.loc[gx][t]>0)) for gx in targets[[t]].index if (targets.loc[gx][t]!=0)} for t in targets.columns}
    f_mutants = {name: patch_model(f, patch) for name, patch in mutants.items()}
    ## 4. Get the reachable attractors from initial state in the presence of mutations ("effect during drug exposure")
    ## 5. Estimate probabilities of attractors from Mutants and compute score
    effects = [0 if (t not in f_mutants) else compute_score(f_mutants[t], x0, [a for a in tqdm(list(f_mutants[t].attractors(reachable_from=x0)))], frontier, nb_sims, experiments, exp_name="Drug %s (%d/%d) in state %s" % (t, it+1, targets.shape[1], state.columns[0]), quiet=quiet) for it, t in enumerate(targets.columns)]
    assert len(effects)==targets.shape[1]
    return effects
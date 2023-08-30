# coding:utf-8

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
import mpbn
import mpbn.simulation as mpbn_sim
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import fbeta_score as Fbeta
from random import seed as rseed
from scipy.stats import ks_2samp, spearmanr, kendalltau

#################################################
## Baseline (L1000 CDS^2) [Duan et al., 2016]  ##
#################################################
def baseline(signatures, phenotype, is_binary=False):
    '''
    Compute the cosine scores between a set of signatures S and a differential phenotype P

    ...

    Parameters
    ----------
    S : Pandas DataFrame
        signatures rows/[genes] x columns/[drug names] [Treated || Control]
    P : Pandas DataFrame
        differential phenotype rows/[genes] x column/[disease name] [Diseased || Healthy]
    is_binary : Python bool
        [default=False] : if set to True, the signatures and phenotypes might be binary (that is, with values in {0,1})

    Returns
    -------
    scores : Python float dictionary
        dictionary (keys=drug names, values=scores), the higher the score, the higher the repurposing power
    '''
    SC, PC = signatures.dropna(), phenotype.dropna()
    if (is_binary):
        SC[SC>0] = 1
        PC[PC>0] = 1
        SC[SC<0] = -1
        PC[PC<0] = -1
        SC[pd.isnull(SC)] = 0
        PC[pd.isnull(PC)] = 0
    C = SC.join(PC, how="outer").fillna(0)
    cosscores = 1-pairwise_distances(C[signatures.columns].T,C[phenotype.columns].T, metric="cosine")
    baseline_df = pd.DataFrame(cosscores, columns=["Cosine Score"], index=signatures.columns)
    return baseline_df

#######################################
## Compute accuracy of the predictor ##
#######################################
def empirical_pvalue(sorted_rewards, sorted_ground_truth, nperms, method="ks_2samp"):
    '''
    Compute an empirical p-value corresponding to testing whether the distributions in the predicted scores and the ground truth are similar by randomly permuting the values of the predictions and averaging the number of significant statistics across all permutations

    ...

    Parameters
    ----------
    sorted_rewards : Python float list
        predicted scores sorted by genes of increasing predicted value
    sorted_ground_truth : Python float list
        ground truth scores sorted by genes of increasing predicted value
    nperms : Python integer
        number of permutations to perform
    method : Python character string
        [default="ks_2samp"] : statistical test to perform at each permutation (should belong to scipy.stats)

    Returns
    -------
    pvalue : Python float
        empirical p-value corresponding to the test across all @nperms permutations
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

    ...

    Parameters
    ----------
    rewards : Python float list
        predicted scores
    ground_truth : Python float list
        ground truth scores
    K : Python integer list
        [default=[2,5,10]] : ranks at which the hit ratio should be computed
    use_negative_class : Python bool
        [default=False] : if set to True, compute the performance with respect to the negative class instead of the positive class
    nperms : Python integer
        [default=10000] : number of permutations to perform
    thres : Python float
        [default=0.] : decision threshold to determine the positive (resp. negative) class
    beta : Python float
        [default=1.] : value of the coefficient in the F-measure

    Returns
    -------
    res_di : Python dictionary
        (keys=metrics, values=values of the metrics)
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

def simulate(network_fname, targets, patients, score, simu_params={}, nbseed=0, quiet=False):
    '''
    Simulate and score the individual effects of drugs on patient phenotypes, compared to controls

    ...

    Parameters
    ----------
    network_fname : Python character string
        (relative) path to a network .BNET file
    targets : Pandas DataFrame
        rows/[genes] x columns/[drugs to test] (either 1: active expression, -1: inactive expression, 0: undetermined expression)
    patients : Pandas DataFrame
        rows/[genes] x columns/[samples] (either 1: activatory, -1: inhibitory, 0: no regulation).
    score : Python object
        scoring of attractors
    simu_params : Python dictionary
        [default={}] : arguments to MPBN-SIM
    nbseed : Python integer
        [default=0] : random seed
    quiet : Python bool
        [default=False] : prints out verbose

    Returns
    ----------
    scores : Pandas DataFrame
        rows/[patient phenotypes] x columns/[drug names], values are the associated scores
    '''
    from multiprocessing import cpu_count
    assert simu_params.get('thread_count', 1)>=1 and simu_params.get('thread_count', 1)<=max(1,cpu_count()-2)
    ## Get M30 genes
    with open(network_fname, "r") as f:
        network = str(f.read())
    genes = [x.split(", ")[0] for x in network.split("\n") if (len(x)>0)]
    from random import seed as rseed
    rseed(nbseed)
    np.random.seed(nbseed)
    ## 1. Classification model b/w healthy and patient phenotypes (to classify final attractor states from treated patients)
    ## 2. Compute one score per drug and per patients
    if (simu_params.get('thread_count', 1)==1):
        scores = [simulate_treatment(network_fname, targets.loc[[g for g in targets.index if (g in genes)]], score, patients[[Patient]], simu_params, quiet=quiet) for Patient in patients.columns]
    else:
        scores = Parallel(n_jobs=simu_params['thread_count'], backend='loky')(delayed(simulate_treatment)(network_fname, targets.loc[[g for g in targets.index if (g in genes)]], score, patients[[Patient]], simu_params, quiet=quiet) for Patient in patients.columns)
    scores = pd.DataFrame(scores, index=patients.columns, columns=targets.columns)
    return scores

def compute_frontier(df, samples, nbseed=0, quiet=False):
    '''
    Fit a model to classify control/treated phenotypes

    ...

    Parameters
    ----------
    df : Pandas DataFrame
        rows/[genes] x columns/[samples] (either 1: active expression, -1: inactive expression, 0: undetermined expression)
    samples : Pandas DataFrame
        rows/["annotation"] x columns/[samples ], values are 1 (healthy sample) or 2 (patient sample).
    nbseed : Python integer
        [default=0] : random seed
    quiet : Python bool
        [default=False] : prints out verbose

    Returns
    ----------
    model : Python object
        object with a function "predict" that returns predictions (1: control, or 2: treated) on phenotypes
    '''
    from sklearn import svm
    model = svm.SVC(random_state=nbseed)
    X = df.values.T
    y = samples.values.T
    model.fit(X, y)
    acc = np.mean([int(x==y) for x,y in zip(list(model.predict(X)), list(y.T))])
    if (not quiet):
        print("<NORD_DS> Accuracy of the model %.2f" % acc)
    return model

def compute_score(f, x0, A, score, genes, nb_sims, experiments, repeat=1, exp_name="", quiet=False):
    '''
    Compute similarities between any attractor in WT and in mutants, weighted by their probabilities

    ...

    Parameters
    ----------
    f : Boolean Network (MPBN) object
        the mutated network
    x0 : MPBN object
        initial state
    A : Attractor list
        list of attractors in mutant network
    score : Python object
        scoring of attractors
    genes : Python character string list
        list of genes in the model frontier
    nb_sims : Python integer
        number of iterations to compute the probabilities
    experiments : Python dictionary list
        list of experiments (different rates/depths)
    repeat : Python integer
        [default=1] : how many times should these experiments be repeated
    exp_name : Python character string
        [default=""] : printable for an experiment
    quiet : Python bool
        [default=False] : prints out verbose

    Returns
    ----------
    score : Python float
        change in attractors induced by the mutation
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
            probs = mpbn_sim.estimate_reachable_attractors_probabilities(f, x0, A, nb_sims, depth(f, **depth_args), rates(f, **rates_args))
            attrs = pd.DataFrame({"MUT_%d"%ia: a for ia, a in enumerate(A)}).replace("*",np.nan).astype(float).loc[genes]
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
            classification_attrs = score(attrs)
            drug_score = probs.T.dot(classification_attrs)
            d_scores.append(drug_score)
    return np.mean(d_scores) if (repeat>1) else d_scores[0]

def simulate_treatment(network_name, targets, score, state, simu_params={}, quiet=False):
    '''
    Compute the score assigned to a drug with targets in @targets[[drug]] in network

    ...

    Parameters
    ----------
    network_name : Python character string
        filename of the network in .bnet (needs to be pickable)
    targets : Pandas DataFrame
        rows/[genes] x columns/[columns]
    score : Python object
        scoring of attractors
    state : Pandas DataFrame
        binary patient initial state rows/[genes] x columns/[values in {-1,0,1}]
    simu_params : Python dictionary
        [default={}] : arguments to MPBN-SIM
    seednb : Python integer
        [default=0] : random seed
    quiet : Python bool
        [default=False] : prints out verbose

    Returns
    ----------
    effects : Python float dictionary
        distance from attractors from treated networks to control profiles
    '''
    ## 1. Load the Boolean network
    f = mpbn.load(network_name)
    genes = list(state.index)
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
    experiments, nb_sims = [{"name": "mpsim", "rates": simu_params.get("rates", "fully_asynchronous"), "depth": simu_params.get("depth", "constant_unitary")}], simu_params.get("nb_sims", 100)
    mutants = {t: {gx: str(int(targets.loc[gx][t]>0)) for gx in targets[[t]].index if (targets.loc[gx][t]!=0)} for t in targets.columns}
    f_mutants = {name: patch_model(f, patch) for name, patch in mutants.items()}
    ## 4. Get the reachable attractors from initial state in the presence of mutations ("effect during drug exposure")
    ## 5. Estimate probabilities of attractors from Mutants and compute score
    effects = [0 if (t not in f_mutants) else compute_score(f_mutants[t], x0, [a for a in tqdm(list(f_mutants[t].attractors(reachable_from=x0)))], score, genes, nb_sims, experiments, exp_name="Drug %s (%d/%d) in state %s" % (t, it+1, targets.shape[1], state.columns[0]), quiet=quiet) for it, t in enumerate(targets.columns)]
    assert len(effects)==targets.shape[1]
    return effects

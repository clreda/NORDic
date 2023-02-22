# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from qnorm import quantile_normalize as qnm

def quantile_normalize(df, njobs=1):
    return qnm(df, axis=1, ncpus=njobs)

def binarize_experiments(data, thres=0.5, method="binary", strict=True, njobs=1):
    '''
        Binarize experimental profiles
        @param\tdata\tPandas DataFrame: rows/[genes] x columns/[samples]
        @param\tthres\tPython float[default=0.5]: threshold for @method="binary" (in [0,0.5])
        @param\tmethod\tPython character string[default="binary"]: binarization method in ["binary","probin"]
        @param\tstrict\tPython bool[default=True]: takes into account equalities (if set to True, value=thres will lead to undefined for the corresponding gene)
        @param\tnjobs\tPython integer[default=1]: parallelism if needed
        @return\tsignatures\tPandas DataFrame: rows/[genes] x columns[samples] with values in [0,1,NaN]
    '''
    assert method in ["binary","probin"]
    assert thres <= 0.5 and thres >= 0
    from multiprocessing import cpu_count
    assert njobs <= max(1,cpu_count()-2) and njobs >= 1
    if (method == "probin"):
        ## /!\ needs to install profilebinR (see install_env_synthesis.sh)
        from profile_binr import ProfileBin
        probin = ProfileBin(data.T)
        probin.fit(njobs)
        data = probin.binarize().T
        return data
    else:
        assert thres <= 0.5 and thres >= 0
        signatures = quantile_normalize(data, njobs=njobs)
        min_mat = pd.concat([signatures.min(axis=1)] * signatures.shape[1], axis=1, ignore_index=True).values
        sigs = pd.DataFrame(signatures.values-min_mat, index=signatures.index, columns=signatures.columns)
        max_mat = pd.concat([sigs.max(axis=1)] * sigs.shape[1], axis=1, ignore_index=True).values
        signatures = pd.DataFrame(sigs.values/max_mat, index=signatures.index, columns=signatures.columns)
        if (strict):
            signatures[signatures < thres] = 0
            signatures[signatures > 1-thres] = 1
            signatures[(signatures>=thres)&(signatures<=1-thres)] = np.nan
        else:
            signatures[signatures<=thres] = 0
            signatures[signatures>=1-thres] = 1
            signatures[(signatures>thres)&(signatures<1-thres)] = np.nan
        return signatures

def compare_states(x, y, genes=None):
    '''
        Computes the similarity between two sets of Boolean states
        @param\tx\tPandas DataFrame: rows/[genes] x columns/[state IDs] contains (0, 1, NaN)
        @param\ty\tPandas DataFrame: rows/[genes] x columns/[state IDs] contains (0, 1, NaN)
        @param\tgenes\tPython character string list: list of gene symbols 
        @returns\tsims, N\tSimilarities between each column of x and each columns of y, on the list of N present genes in @genes (if provided) 
        otherwise on the union of N genes in x and y
    '''
    assert all([(zx in [0,1] or np.isnan(zx)) for zx in np.unique(x.values)])
    assert all([(zy in [0,1] or np.isnan(zy)) for zy in np.unique(y.values)])
    xx = pd.DataFrame(x.values, index=x.index, columns=["X%d" %d for d in range(x.shape[1])])
    yy = pd.DataFrame(y.values, index=y.index, columns=["Y%d" %d for d in range(y.shape[1])])
    z = xx.join(yy, how="outer")
    if (genes is not None):
        gene_list = [g for g in genes if (g in z.index)]
        z = z.loc[gene_list]
        if (len(gene_list)==0):
            raise ValueError("None of the genes in the list is present in any of the vectors!")
    N = z.shape[0]
    x_, y_ = [z[u.columns] for u in [xx,yy]]
    ## Compute separately distances between 1's and 0's
    X_pos, Y_pos, X_neg, Y_neg = [np.mod(u.fillna(v).T.values+1, 2).astype(float) for v in [0,1] for u in [x_,y_]]
    dists_pos, dists_neg = [pairwise_distances(X, Y, metric="cityblock")/N for X,Y in [[X_pos,Y_pos],[X_neg,Y_neg]]]
    dists_pos[np.isnan(dists_pos)] = 0
    dists_neg[np.isnan(dists_neg)] = 0
    dists = np.power(np.multiply(dists_pos,dists_neg), 0.5) # geometric mean
    ## Numerical approximations
    dists[np.isclose(dists, 0)] = 0
    dists[np.isclose(dists, 1)] = 1
    sims = 1-dists
    return sims, N

def finetune_binthres(df, samples, network_fname, mutation, step=0.005, maxt=0.5, mint=0, score_binthres=lambda itc,ita_c,ita_t:(1-itc)*ita_c*ita_t, njobs=1, verbose=True):
    '''
        Select the binarization threshold (in function @binarize_experiments) which maximize the dissimilarity interconditions and the similarity intracondition
        @param\tdf\t
        @param\tsamples\t
        @param\tnetwork_fname\t
        @param\tmutation\t
        @param\tstep\t
        @param\tmaxt\t
        @param\tmint\t
        @param\tscore_binthres\t
        @param\tnjobs\t
        @param\tverbose\t
        @returns\tmax_thres\t
    '''
    assert mint>=0 and maxt<=0.5
    with open(network_fname, "r") as f:
        gene_list = [x.split(", ")[0] for x in f.read().split("\n") if (len(x)>0)]
    gene_outputs = [g for g in gene_list if (g not in mutation)]
    max_thres, max_score = None, -float("inf")
    scale=float("0."+("0"*(len(str(step).split(".")[-1])-1))+"1")
    for bt in [x*scale for x in range(int(mint/scale),int(maxt/scale)+1,int(step/scale))]:
        if (bt>maxt):
            break
        df_b = binarize_experiments(df, thres=bt, method="binary", strict=True, njobs=njobs)
        cp = df_b[[s for s in samples if (samples[s]=="control")]]
        tps = df_b[[s for s in samples if (samples[s]=="treated")]]
        sims_interconditions, _ = compare_states(cp, tps, gene_outputs)
        sims_intractrl, _ = compare_states(cp, cp, gene_outputs)
        sims_intratps, _ = compare_states(tps, tps, gene_outputs)
        sc = score_binthres(np.max(sims_interconditions), np.min(sims_intractrl), np.min(sims_intratps))
        #sc = score_binthres(np.max(sims_interconditions), np.max(sims_intractrl), np.max(sims_intratps))
        if (sc > max_score):
            max_thres, max_score = bt, sc
        if (verbose):
            print("BinThres=%.3f\tMax.sim CTRL||TRT: %.3f\tMin.sim CTRL||CTRL: %.3f\tMin.sim TRT||TRT: %.3f\tScore: %.3f"%(bt, np.max(sims_interconditions), np.min(sims_intractrl), np.min(sims_intratps), sc))
    if (verbose):
        print("Final bin_thres value: %.3f (score %.3f)" % (max_thres, max_score))
    return max_thres

## Tests
if __name__ == "__main__":
    ## Length of vectors
    N=10
    P=2
    ## All 1
    ones = pd.DataFrame([1]*N, index=range(N))
    ## Half 1, Half 0
    half = pd.DataFrame([1]*(N//P)+[0]*(N-N//P), index=range(N))
    ## Half 1, Half NaN
    halfnan = pd.DataFrame([1]*(N//P)+[np.nan]*(N-N//P), index=range(N))
    ## Half 0, Half NaN
    nanhalf = pd.DataFrame([0]*(N//P)+[np.nan]*(N-N//P), index=range(N))
    ## All 0
    zeros = pd.DataFrame([0]*N, index=range(N))
    tests = [ones, half, halfnan, nanhalf, zeros]
    res_mat = np.zeros((len(tests), len(tests)))
    all_pairs = [[tests[i], tests[j]] for i in range(len(tests)) for j in range(len(tests))]
    idx_lst = ["%s" % (list(u[u.columns[0]].value_counts(dropna=False).index.astype(float))) for u in tests]
    res_mat = np.array([float(compare_states(c1, c2)[0]) for c1, c2 in all_pairs]).reshape((len(tests), len(tests)))
    res_df = pd.DataFrame(res_mat, index=idx_lst, columns=idx_lst)
    print(res_df.T)

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
    ## numerical approximations
    dists[np.isclose(dists, 0)] = 0
    dists[np.isclose(dists, 1)] = 1
    sims = 1-dists
    return sims, N

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

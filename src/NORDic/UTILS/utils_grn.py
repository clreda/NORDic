# coding: utf-8

import bonesis
import mpbn
import numpy as np
import pandas as pd
from itertools import product
from functools import reduce
import sys
from zipfile import ZipFile
import os
from glob import glob
from tqdm import tqdm
from subprocess import call as sbcall
from copy import deepcopy
from sklearn.linear_model import LogisticRegression as Logit

from NORDic.UTILS.utils_state import quantile_normalize

def desirability(x, f_weight_di, A=0, B=1):
    '''
        Harrington's desirability function, used by [1]
        Convert a list of functions to maximize into a single scalar function to maximize with values in [@A,@B]
        @param\tx\tPoint
        @param\tf_weight_di\tPython dictionary: function with arguments as the same type as x, and associated weight
        @param\tA\tPython float[default=0]: lower bound of the function interval
        @param\tB\tPython float[default=1]: upper bound of the function interval
        @return\tdes(x)\tvalue of the desirability function at point x
        [1] http://ceur-ws.org/Vol-2488/paper17.pdf
        https://cran.r-project.org/web/packages/desirability/vignettes/desirability.pdf
    '''
    values = [np.exp(-np.exp(-((B-A)+weight/(B-A)*funct(x)))) for funct, weight in f_weight_di.items()]
    return float(np.exp(np.mean(np.log(values))))

def DS(influences):
    '''
        Computes the number of edges over the maximum number of possible connections between the nodes in the network
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes]
        @return\tDS\tPython float: network density
    '''
    n = influences.shape[0]
    return 2*np.sum(np.sum(np.triu(np.abs(influences.values)), axis=1), axis=0)/float(n*(n-1))

def CL(influences):
    '''
        Computes the average of node-wise clustering coefficients
        The clustering coefficient of a node is the ratio of the degree of the considered node and the maximum possible number of connections such that this node and its current neighbors form a clique
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes]
        @return\tCL\tPython float: network clustering coefficient
    '''
    n = influences.shape[0]
    network_outdegree = np.sum(np.abs(influences.values), axis=1)
    connect = np.sum(np.triu((np.abs(influences)+np.abs(influences).T).values),axis=1)
    max_connect = 0.5*np.multiply(network_outdegree,network_outdegree-1)
    max_connect[max_connect<=0] = 0.5
    numCL = np.sum(np.divide(connect, max_connect),axis=0)
    return numCL/n

def Centr(influences):
    '''
        Computes the network centralization, which is correlated with the similarity of the network to a graph with a star topology
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes]
        @return\tCentr\tPython float: network centralization
    '''
    n = influences.shape[0]
    network_outdegree = np.sum(np.abs(influences.values),axis=1)
    k_max = np.max(network_outdegree)
    return n/(n-2)*(k_max/(n-1)-DS(influences))

def GT(influences):
    '''
        Computes the network heterogeneity, which quantifies the non-uniformity of the node degrees across the network
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes]
        @return\tGT\tPython float: network heterogeneity
    '''
    network_outdegree = np.sum(np.abs(influences.values), axis=1)
    k_mean = np.mean(network_outdegree)
    return np.sqrt(np.var(network_outdegree))/max(k_mean,1)

def general_topological_parameter(influences, weights):
    '''
        Computes the general topological parameter (GTP) associated with the input network
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes]
        @param\tweights\tPython dictionary of (Python character string x Python float): all keys must be in ["DS","CL","Centr","GT"]
        @return\tscore\tPython float: return the score using the Harrington's desirability function
    '''
    assert all([w in ["DS","CL","Centr","GT"] for w in weights])
    missing_index = [x for x in influences.index if (x not in influences.columns)]
    missing_cols = [x for x in influences.columns if (x not in influences.index)]
    if (len(missing_index)>0):
        influences = pd.concat((influences, pd.DataFrame(0, index=missing_index, columns=influences.columns)), axis=0)
    if (len(missing_cols)>0):
        influences = pd.concat((influences, pd.DataFrame(0, index=list(influences.index)+missing_index, columns=missing_cols)), axis=1)
    return desirability(influences, {DS: weights.get("DS",0), CL: weights.get("CL",0), Centr: weights.get("Centr",0), GT: weights.get("GT", 0)})

def get_weakly_connected(network_df, gene_list, index_col="preferredName_A", column_col="preferredName_B", score_col="sscore"):
    '''
        Depth-first search (DFS) on undirected network
        @param\tnetwork_df\tPandas DataFrame: rows/[index] x columns/[["Input","Output"]]
        @param\tgene_list\tPython character string list: list of genes (needed to take into account isolated genes in the network)
        @param\tindex_col\tPython character string[default="preferredName_A"]: column in network_df (input gene)
        @param\tcolumn_col\tPython character string[default="preferredName_B"]: column in network_df (output gene)
        @param\tscore_col\tPython character string[default="sscore"]: column in network_df (edge weight)
        @return\tcomponents\tType of @network_df.loc[network_df.index[0]]["Input"] Python list of Python list: list of weakly connected components in the network, ordered by decreasing size
    '''
    adjacency = network_df.pivot_table(index=index_col, columns=column_col, values=score_col, aggfunc="mean")
    ## Undirected adjacency matrix
    adjacency[~np.isnan(adjacency)] = 1
    adjacency = adjacency.fillna(0)
    missing_index = [g for g in gene_list if (g not in adjacency.index)]
    if (len(missing_index)>0):
        missing_index_df = pd.DataFrame(0, index=missing_index, columns=adjacency.columns)
        adjacency = pd.concat((adjacency, missing_index_df), axis=0)
    missing_columns = [g for g in gene_list if (g not in adjacency.columns)]
    if (len(missing_columns)>0):
        missing_columns_df = pd.DataFrame(0, index=adjacency.index, columns=missing_columns)
        adjacency = pd.concat((adjacency, missing_columns_df), axis=1)
    assert adjacency.shape[0]==adjacency.shape[1]==len(gene_list)
    N = len(gene_list)
    components = []
    to_visit = [0]
    visited = [False]*N
    while (not all(visited)):
        component = []
        while (True):
            node = to_visit.pop()
            visited[node] = True
            if (node not in component):
                component.append(node)
            children = [list(adjacency.index).index(list(adjacency.columns)[i]) for i in np.argwhere(adjacency.values[node, :] != 0).flatten().tolist()]
            to_visit = [child for child in children if (not visited[child])]+to_visit
            if (len(to_visit)==0):
                break
        components.append(component)
        to_visit = [np.argmin(visited)]
    components = list(sorted(components, key=lambda c : len(c), reverse=True))
    components = [[list(adjacency.index)[n] for n in c] for c in components]
    return components

def get_genes_interactions_from_PPI(ppi, connected=False, score=0, filtering=True, quiet=False):
    '''
        Filtering edges to decrease computational cost while preserving network connectivity (if needed)
        @param\tppi\tPandas DataFrame: rows/[index] x columns[{"preferredName_A", "preferredName_B", "sign", "directed", "score"]]; sign in {-1,1,2}, directed in {0,1}, score in [0,1]
        @param\tconnected\tPython bool[default=True]: if set to True, preserve/enforce connectivity on the final network
        @param\tscore\tPython float[default=0]: Lower bound on the edge-associated score
        @param\tfiltering\tPython bool[default=True]: Whether to filter out edges by a correlation threshold
        @param\tquiet\tPython bool[default=False]
        @return\tppi_accepted\tPandas DataFrame: rows/[index] x columns/[["Input", "Output"]]
    '''
    assert all([x in [-1,1,2] for x in list(ppi["sign"])])
    assert all([x in [0,1] for x in list(ppi["directed"])])
    assert all([x <= 1 and x >= 0 for x in list(ppi["score"])])
    cols = ['preferredName_A', 'preferredName_B', 'sign', 'directed', 'score']
    assert all([col in cols for col in ppi.columns])
    assert ppi.shape[1]==len(cols)
    ppi = ppi[['preferredName_B', 'preferredName_A', 'sign', 'directed', 'score']]
    ## Remove duplicate edges (preserve the duplicates with highest score)
    ppi.index = ["--".join(x) for x in zip(ppi["preferredName_A"], ppi["preferredName_B"])]
    ppi = ppi.sort_values(by="score", ascending=False)
    ppi = ppi.loc[~ppi.index.duplicated(keep="first")]
    ppi = ppi[["preferredName_A","preferredName_B", 'sign', 'directed', 'score']]
    ppi.index = range(ppi.shape[0])
    Ntotal_edges = ppi.shape[0]
    ## 1. Double edges depending on whether they are marked as "directed"
    undirected_edges = ppi.loc[ppi["directed"]==0]
    Nundirected_edges = undirected_edges.shape[0]
    undirected_edges.columns = ['preferredName_B', 'preferredName_A', 'sign', 'directed', 'score']
    undirected_edges = undirected_edges[['preferredName_A', 'preferredName_B', 'sign', 'directed', 'score']]
    assert all([c==['preferredName_A', 'preferredName_B', 'sign', 'directed', 'score'][i] for i, c in enumerate(undirected_edges.columns)])
    assert undirected_edges.loc[undirected_edges.index[0]]['preferredName_B']==ppi.loc[ppi["directed"]==0].loc[ppi.loc[ppi["directed"]==0].index[0]]['preferredName_A']
    ppi = pd.concat((ppi, undirected_edges),axis=0)[["preferredName_A","preferredName_B","sign","score"]]
    ppi.index = range(ppi.shape[0])
    assert ppi.shape[0]==Ntotal_edges+Nundirected_edges
    ## 2. Double edges depending on whether they have a specific sign
    nonmonotonic_edges = ppi.loc[ppi["sign"]==2]
    Nnonmonotonic_edges = nonmonotonic_edges.shape[0]
    ppi_sign_sum = sum([abs(x) for x in list(ppi["sign"])])
    nonmonotonic_edges = pd.concat((nonmonotonic_edges[[c for c in nonmonotonic_edges.columns if (c!="sign")]], pd.DataFrame([[-1]]*nonmonotonic_edges.shape[0], index=nonmonotonic_edges.index, columns=["sign"])), axis=1)
    nonmonotonic_edges = nonmonotonic_edges[["preferredName_A","preferredName_B","sign","score"]]
    ppi = pd.concat((ppi, nonmonotonic_edges), axis=0)
    ppi["sign"] = [1 if (s==2) else s for s in list(ppi["sign"])]
    ppi.index = range(ppi.shape[0])
    assert ppi.shape[0]==Ntotal_edges+Nundirected_edges+Nnonmonotonic_edges
    assert ppi["sign"].abs().sum().sum()==ppi_sign_sum
    ## 3. Concatenate columns "score" and "sign" as score in [0,1]
    ppi["sscore"] = np.multiply(ppi["score"], ppi["sign"])
    ppi = ppi[["preferredName_A","preferredName_B","sscore"]]
    if (connected):
        components = get_weakly_connected(ppi, list(set(list(ppi["preferredName_A"])+list(ppi["preferredName_B"]))))
        main_component = components[0]
        ## 5. Remove genes which are isolated in the full PPI
        isolated_genes = [g for c in components[1:] for g in c]
        if (len(isolated_genes)>0):
            test_isolated = np.vectorize(lambda x : x in isolated_genes)
            ppi = ppi.loc[~test_isolated(ppi["preferredName_A"])&~test_isolated(ppi["preferredName_B"])]
        t = min(score, ppi["sscore"].abs().max()) if (filtering) else ppi["sscore"].abs().min()
        ppi_accepted = ppi.loc[ppi["sscore"].abs()>=t]
        components = get_weakly_connected(ppi_accepted, main_component)
        isolated_genes = [g for c in components[1:] for g in c if (g not in components[0])]
        assert len(isolated_genes)+len(components[0])==len(main_component)
        ## 6. Add edges to/from isolated genes in decreasing (absolute) score order until the PPI is connected and all genes in the connected PPI are present
        if (len(isolated_genes)>0):
            ppi_rejected = ppi.loc[ppi["sscore"].abs()<t]
            test_isolated = np.vectorize(lambda x : x in isolated_genes)
            while ((len(isolated_genes)>0) and (ppi_rejected.shape[0]>0)):
                ppi_isolated = ppi_rejected.loc[(test_isolated(ppi_rejected["preferredName_A"])&~test_isolated(ppi_rejected["preferredName_B"]))|(test_isolated(ppi_rejected["preferredName_B"])&~test_isolated(ppi_rejected["preferredName_A"]))]
                t = np.max(ppi_isolated["sscore"].abs())
                add_ppi_isolated = ppi_isolated.loc[ppi_isolated["sscore"].abs()>=t]
                ppi_accepted = pd.concat((ppi_accepted, add_ppi_isolated), axis=0)
                components = get_weakly_connected(ppi_accepted, main_component)
                if (not quiet):
                    print("adding %d edges at t=%f (%d edges in total)" % (add_ppi_isolated.shape[0], t, ppi_accepted.shape[0]))
                isolated_genes = [g for c in components[1:] for g in c if (g not in components[0])]
                if (not quiet):
                    print("Components %s... (%d isolated genes)" % (str([len(c) for c in components[:3]]),len(isolated_genes)))
                assert len(isolated_genes)+len(components[0])==len(main_component)
                test_isolated = np.vectorize(lambda x : x in isolated_genes)
                ppi_rejected = ppi_rejected.loc[(ppi_rejected["sscore"].abs()<t)|((test_isolated(ppi_rejected["preferredName_A"])|(test_isolated(ppi_rejected["preferredName_B"]))))]
        assert len(components)==1 and len(components[0])==len(main_component)
    else:
        t = min(score, ppi["sscore"].abs().max()) if (filtering) else ppi["sscore"].abs().min()
        ppi_accepted = ppi.loc[ppi["sscore"].abs()>=t]
    ppi_accepted.columns = ["Input","Output","SSign"]
    return ppi_accepted

def build_influences(network_df, tau, beta=1, cor_method="pearson", expr_df=None, accept_nonRNA=False, quiet=False):
    '''
        Filters out (and signs of unsigned) edges based on gene expression 
        @param\tnetwork_df\tPandas DataFrame: rows/[index] x columns/[["Input", "Output", "SSign"]] interactions
        @param\ttau\tPython float: threshold on genepairwise expression correlation
        @param\tbeta\tPython integer[default=1]: power applied to the adjacency matrix
        @param\tcor_method\tPython character string[default="pearson"]: type of correlation
        @param\texpr_df\tPandas DataFrame[default=None]: rows/[genes] x columns/[samples] gene expression data
        @param\taccept_nonRNA\tPython bool[default=False]: if set to False, ignores gene names which are not present in expr_df
        @param\tquiet\tPython bool[default=False]
        @return\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes] signed adjacency matrix with only interactions s.t. corr^beta>=tau
    '''
    assert network_df.shape[1]==3 and all([c in ["Input","Output", "SSign"] for c in network_df.columns])
    network = network_df.pivot_table(index="Input", columns="Output", values="SSign", aggfunc="mean")
    missing_index = [g for g in network.columns if (g not in network.index)]
    if (len(missing_index)>0):
        missing_index_df = pd.DataFrame([], index=missing_index, columns=network.columns)
        network = pd.concat((network, missing_index_df), axis=0)
    missing_columns = [g for g in network.index if (g not in network.columns)]
    if (len(missing_columns)>0):
        missing_columns_df = pd.DataFrame([], index=network.index, columns=missing_columns)
        network = pd.concat((network, missing_columns_df), axis=1)
    assert network.shape[0]==network.shape[1]
    network = network[network.index]
    network = network.fillna(-666) #missing value
    network[network==0] = 2 #means that the edge is present both as activatory and inhibitory
    network[network==-666] = 0
    network[network<0] = -1 #only inhibitory
    network[(network>0)&(network<2)] = 1 #only activatory
    network = network.astype(int)
    network_unsigned, network_signed = [deepcopy(network) for _ in range(2)] 
    network_unsigned[network_unsigned!=2] = 0
    network_unsigned = network_unsigned/2 #1 if unsigned edge exists, 0 otherwise
    network_signed[network_signed==2] = 0 #1 if signed activatory edge exists, -1 if signed inhibitory edge exists, 0 otherwise
    # Correlation matrix
    if (expr_df is not None):
        df = quantile_normalize(expr_df)
        df = df.loc[[g for g in list(network.index) if (g in df.index)]]
        coexpr = np.power(df.T.corr(method=cor_method), beta)
        coexpr[coexpr.abs()<tau] = 0
        if (accept_nonRNA):
            missing_genes = [g for g in list(network.index) if (g not in df.index)]
            for g in missing_genes:
                coexpr[g] = 1
                coexpr.loc[g] = 1
            coexpr = coexpr.loc[list(network.index)][list(network.index)]
    else:
        coexpr = pd.DataFrame(np.ones(network.shape), index=list(network.index), columns=list(network.index)) #default: positive interactions
    assert all([coexpr.shape[i]==s for i,s in enumerate(network_unsigned.shape)])
    assert all([coexpr.index[i]==s for i,s in enumerate(network_unsigned.index)])
    assert all([coexpr.columns[i]==s for i,s in enumerate(network_unsigned.columns)])
    net_mat = np.multiply((-1)**(coexpr.values<0).astype(int), network_unsigned.values).astype(float)#equal to 1 if strong correlation and activatory, -1 if strong correlation and inhibitory, 0 otherwise
    influences = pd.DataFrame(net_mat+network_signed.values, index=network.index, columns=network.columns)
    assert not pd.isnull(influences).any().any()
    return influences

def create_grn(influences, exact=False, max_maxclause=3, quiet=False):
    '''
        Create a BoneSiS InfluenceGraph
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes] of interactions, values in {-1,1,0} -1:negative,1:positive,0:absent
        @param\texact\tPython bool[default=False]: should all interactions be preserved?
        @param\tmax_maxclause\tPython integer[default=3]: upper bound on the number of clauses in DNF form
        @param\tquiet\tPython bool[default=False]
        @return\tgrn\tBoneSiS InfluenceGraph class object
    '''
    assert all([n in [-1,1,0] for n in np.unique(influences.values)])
    maxclause = min(get_maxdegree(influences, quiet=quiet), max_maxclause)
    influences_list = influences.melt(ignore_index=False)
    influences_list["id"] = influences_list.index
    influences_list = influences_list[influences_list["value"]!=0]
    influences_list = [(x,y,dict(sign=s)) for [y,s,x] in influences_list.values.tolist()]
    if (not quiet):
        print("<BONESIS> %d interactions (maximum # of clauses = %d)" % (len(influences_list), maxclause))
    grn = bonesis.InfluenceGraph(graph=influences_list, maxclause=maxclause, exact=exact)
    return grn

def get_maxdegree(influences, activatory=True, quiet=False):
    '''
        Computes the maximum ingoing degree (or the maximum number of potential activatory regulators) in a graph
        @param\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes] of interactions: -1:negative,1:positive,0:absent
        @param\tactivatory\tPython bool[default=True]: computes the maximum number of potential activatory regulators instead
        @param\tquiet\tPython bool[default=False]
        @return\tmaxindegree\tPython integer
    '''
    assert all([n in [-1,0,1] for n in np.unique(influences.values)])
    if (not activatory):
        maxindegree=int(np.max((influences!=0).astype(float).sum(axis=0)))
    else:
        maxindegree=int(np.max((influences>0).astype(float).sum(axis=0)))
    if (not quiet):
        print("<UTILS_GRN> Maximum "+("ingoing degree" if (not activatory) else "possible #activators")+"=%d" % maxindegree)
    return maxindegree

def build_observations(grn, signatures, quiet=False):
    '''
        Implement experimental constraints from perturbation experiments in signatures
        Experimental constraints are of the form...
        @param\tgrn\tInfluenceGraph (from BoneSiS): contains topological constraints
        @param\tsignatures\tPandas DataFrame: rows/[genes] x columns/[experiment IDs]. Experiment IDs is of the form "<pert. gene>_<pert. type>_<...>_<cell line>" (treated) or "initial_<cell line>" (control)
        @param\tquiet\tPython bool[default=False]
        @return\tBO\tBoNesis object (from BoneSiS)
    '''
    data_exps = {}
    if (len(signatures)==0):
        BO = bonesis.BoNesis(grn, data_exps)
        return BO
    signatures = signatures.loc[[g for g in signatures.index if (g in grn.nodes)]]
    ## 1. Add signatures of experimental states
    exps = [x for x in signatures.columns if ("initial" not in x)]
    exps_ids = range(len(exps))
    ## 1a. Instantiate experiments
    ## zero state to ignore
    #data_exps.update({"zero": {g: 0 for g in grn.nodes}})
    ## 1b. For each experiment (1 initial state, 1 final state, 1 perturbation)
    for exp_nb in exps_ids:
        cell = exps[exp_nb].split("_")[-1]
        cols = ["Exp%d_"%(exp_nb+1)+x for x in ["init", "final"]]
        data_df = signatures[["initial_"+cell, exps[exp_nb-1]]]

        ## Compatible with perturbation experiment
        data_df = data_df.T
        pert, sign = exps[exp_nb].split("_")[0], 0 if (exps[exp_nb].split("_")[1]=="KD") else 1
        data_df[pert] = sign 
        data_df = data_df.T

        data_df.columns = cols
        for col in cols:
            data_exps.update(data_df[[col]].dropna().astype(int).to_dict())
    if (not quiet):
        print_exps = pd.DataFrame.from_dict(data_exps, orient="index").fillna(-1).astype(int)
        print_exps[print_exps==-1] = ""
        print("\n<UTILS_GRN> %d experiments\n%s" % (len(exps_ids), str(print_exps)))
    BO = bonesis.BoNesis(grn, data_exps)
    ## 2. Instantiate reachability & fixed point constraints
    for exp_nb in exps_ids:
        exp = exps[exp_nb]
        pert, sign = exp.split("_")[:2]
        cell = exp.split("_")[-1]
        sign = int(sign!="KD")
        with BO.mutant({pert: sign}) as m:
            final_FP = m.fixed(~m.obs("Exp%d_final" % (exp_nb+1)))
            ## 2a. There exists a trajectory : init -> trapspace
            ~m.obs("Exp%d_init" % (exp_nb+1)) >= final_FP #~m.obs("Exp%d_final" % (exp_nb+1)) ##
            ## MEMO: Universal reachable fixed points: ~m.obs("Exp%d_init" % (exp_nb+1)) >> "fixpoints" ^ {m.obs("Exp%d_final" % (exp_nb+1))}
        ## 2b. No trivial trajectory from zero state : zero -> trapspace(unless the experiment actually involves it)
        #state_di = data_exps["Exp%d_init" % (exp_nb+1)]
        #state_di.update({pert: sign})
        #if (all([state_di.get(g, 1)==0 for g in grn.nodes])):
        #    continue
        #~BO.obs("zero") / ~BO.obs("Exp%d_final" % (exp_nb+1))
    ## MEMO: Fixed point: BO.fixed(~BO.obs("attractor_%d" % (state_id+1)))
    ## MEMO: Trap space: BO.fixed(BO.obs("attractor_%d" % (state_id+1)))
    return BO

def solution2influences(solution):
    '''
        Converts a solution object into a influences object
        @param\tsolution\tPandas Series: rows/[genes]
        @return\tinfluences\tPandas DataFrame: rows/[genes] x columns/[genes] contains values {-1,1,0,2} -1: negative, 1: positive, 0: absent, 2: non monotonic
    '''
    gene_list = list(solution.index)
    N = len(gene_list)
    infl_mat = np.zeros((N,N))
    grfs = get_grfs_from_solution(solution)
    for gene in grfs:
        ig = gene_list.index(gene)
        for regulator in grfs[gene]:
            if (regulator in gene_list):
                ir = gene_list.index(regulator)
                infl_mat[ir,ig] = grfs[gene][regulator]
            else:
                infl_mat = np.concatenate((infl_mat, np.zeros((1,N))), axis=0)
                infl_mat = np.concatenate((infl_mat, np.zeros((N+1,1))), axis=1)
                infl_mat[N,ig] = grfs[gene][regulator]
                gene_list.append(regulator)
                N += 1
    influences = pd.DataFrame(infl_mat, index=gene_list, columns=gene_list)
    return influences

def zip2df(fname):
    '''
        Extract solutions in ZIP file as DataFrames
    '''
    with ZipFile(fname, "r") as zip:
        zip.extractall()
    grn_fnames = glob("*.bnet")
    grns = []
    for grn_fname in grn_fnames:
        grn = load_grn(grn_fname)
        grns.append(grn)
    solutions = pd.DataFrame(grns)
    sbcall("rm -f *.bnet", shell=True)
    return solutions

def load_grn(fname):
    '''
        Loads GRN as MPBN class element
    '''
    BN = mpbn.MPBooleanNetwork(fname)
    sep=" <- " if (" <- " in str(BN)) else ", "
    solution = pd.Series([x.split(sep)[-1] for x in str(BN).split("\n")], index=[x.split(sep)[0] for x in str(BN).split("\n")])
    return solution

def get_minimal_edges(R, maximal=False):
    '''
       Return one of the solutions with the smallest (or greatest) number of edges
       @param\tR\tPandas DataFrame: rows/[genes] x columns/[solution IDs]
       @param\tconnected\tPython bool[default=False]: if set to True, return the CONNECTED solution which satisfies those constraints
       @param\tmaximal\tPython bool[default=False]: if set to True, return the solution with the greatest number of edges
       @return\tsolution, nedges\tPython integer x Python integer: solution and corresponding number of edges
    '''
    nedges_list = [int(solution2influences(R[col]).abs().sum().sum()) for col in R.columns]
    solution_id = (np.argmax if (maximal) else np.argmin)(nedges_list)
    return R[R.columns[solution_id]], nedges_list[solution_id]

def get_grfs_from_solution(solution):
    '''
        Retrieve all gene regulatory functions (GRFs) from a given solution
        @param\tsolution\tPandas Series: rows/[genes] 
        @return\tgrfs\tPython dictionary: {gene: {regulator: sign, ...}, ...} where sign in {-1,1} -1: inhibitor, 1: activator
    '''
    sol_dict, grfs = solution.to_dict(), {}
    for gene in sol_dict:
        grf = "".join("".join(" ".join(" ".join(str(sol_dict[gene]).split(", ")[-1].split("|")).split("&")).split("(")).split(")"))
        if (len(str(grf))==0):
            continue
        regulators = {}
        if (str(grf) not in ["0", "1"]):
            reg_names = list(set(grf.split(" ")))
            regulators_df = pd.DataFrame([[(-1)**int(regulator[0]=="!")] for regulator in reg_names], index=[regulator[int(regulator[0]=="!"):] for regulator in reg_names], columns=["regulators"])
            regulators_df = regulators_df.groupby(level=0).sum()
            regulators_df[regulators_df==0] = 2
            regulators = regulators_df.to_dict()["regulators"]
        grfs.setdefault(gene, regulators)
    return grfs

def save_grn(solution, fname, sep=", ", quiet=False, max_show=5, write=True):
    '''
        Write and/or print .bnet file
        @param\tsolution\tPandas Series: rows/[genes] contains gene regulatory functions (GRF)
        @param\tfname\tPython character string: where to write the file (w/o .bnet extension)
        @param\tsep\tPython character string: what separates regulators from regulated genes
        @param\tquiet\tPython bool[default=False]
        @param\tmax_show\tPython integer[default=5]: maximum number of printed GRFs
        @param\twrite\tPython bool[default=True]: if set to True, write to a .bnet file
        @return\tNone\t
    '''
    sol = solution.to_dict()
    print_sol = ["".join([str(x) for x in [gene, sep, sol[gene]]]) for gene in sol if (len(gene)>0)]
    if (not quiet):
        print("\n"+("\n".join(print_sol[:max_show]+["..." if (len(print_sol)>max_show) else ""])))
    print_sol = "\n".join(print_sol)
    if (write):
        with open(fname+".bnet", "w+") as f:
            f.write(print_sol)
    return None

def save_solutions(bnetworks, fname, limit):
    '''
        Enumerate and save solutions
        @param\tbnetworks\tOutput of the inference
        @param\tfname\t
        @param\tlimit\tPython integer: maximum number of solutions to enumerate
        @return\tn\tPython integer: number of enumerated solutions
    '''
    with ZipFile(fname, "w") as bundle:
        n = 0
        for i, bn in enumerate(bnetworks):
            with bundle.open(f"bn{i}.bnet", "w") as fp:
                fp.write(bn.source().encode())
            n += 1
            if n == limit:
                break
    return n

def infer_network(BO, njobs=1, fname="solutions", use_diverse=True, limit=50, niterations=1):
    '''
        Infer solutions matching topological & experimental constraints
        @param\tBO\t Bonesis object (from BoneSiS): contains topological & experimental constraints
        @param\tfname\tPython character string[default="solutions"]: path to solution files
        @param\tuse_diverse\tPython bool[default=True]: use the "diverse" procedure in BoneSiS
        @param\tlimit\tPython integer[default=50]: maximum number of solutions to generate per interation
        @param\tniterations\tPython integer[default=1]: maximum number of iterations
        @return list of # solutions per iteration
    '''
    bonesis.settings["parallel"] = njobs
    #bonesis.settings["solutions"] = "subset-minimal"
    if (use_diverse):
        infer = lambda bo : bo.diverse_boolean_networks
    else:
        infer = lambda bo : bo.boolean_networks
    param_di = {"skip_supersets": True}
    if (str(limit)!="None"):
        param_di.setdefault("limit",limit)
    bnetworks = infer(BO)(**param_di)
    nsolutions = []
    for niter in tqdm(range(niterations)):
        if (not os.path.exists(fname+"_"+str(niter+1)+".zip")):
            nsolutions.append(save_solutions(bnetworks, fname+"_"+str(niter+1)+".zip", limit))
    return nsolutions

def get_genes_downstream(network_fname, gene, n=-1):
    '''
        Get the list of genes downstream of a gene in a network
        @param\tnetwork_fname\tPython character string: path to the .BNET file associated with the network
        @param\tgene\tPython character string: gene name in the network
        @param\tn\tPython integer[default=-1]: number of recursions (if<0, recursively get all downstream genes)
        @return\tlst_downstream\tPython character string list: list of nodes downstream of @gene
    '''   
    with open(network_fname, "r") as f:
        grf_list = f.read().split("\n")
    grfs = dict([x.split(", ") for x in grf_list if (len(x)>0)]) 
    set_downstream = set([])
    if (gene not in grfs):
        return []
    count_n = 0
    while ((n<0) or (count_n<n)):
        l = len(set_downstream)
        for g in grfs:
            if (any(x in grfs[g] for x in set_downstream.union({gene}))):
                set_downstream = set_downstream.union({g})
        if (l==len(set_downstream)):
            break
        count_n += 1
    return list(set_downstream)

def get_genes_most_variable(control_profiles, treated_profiles, p=0.8):
    '''
        Get the list of genes which contribute most to the variation between two conditions (in the @pth percentile of change)
        @param\tcontrol_profiles\tPandas DataFrame: rows/[genes] x columns/[samples] profiles from condition 1
        @param\ttreated_profiles\tPandas DataFrame: rows/[genes] x columns/[samples] profiles from condition 1
        @param\tp\tPython float: 100*p th percentile to consider 
        @return\tlst_genes\tPython character string list: list of nodes which contribute most to the variation between conditions
    '''  
    assert p>=0 and p<=1
    model = Logit(penalty="l1",solver="saga",fit_intercept=False,random_state=0,max_iter=1000,n_jobs=njobs)
    df = control_profiles.join(treated_profiles, how="inner").fillna(0.5)
    X = df.T.values
    y = np.ravel(np.array([0]*control_profiles.shape[1]+[1]*treated_profiles.shape[1]))
    model.fit(X, y)
    q = np.quantile(np.abs(model.coef_.flatten()), p)
    genes = {g: model.coef_[0,ig] for ig, g in enumerate(list(df.index)) if (abs(model.coef_[0,ig])>q)}
    return list(genes.keys())

def reconnect_network(network_fname):
    '''
        Write the network with all isolated nodes (no ingoing/outgoing edges) filtered out
        @param\tnetwork_fname\tPython character string: path to the .BNET associated with the network
        @return\tfname\tPython character string: path to the .BNET associated with the reconnected network
    '''  
    with open(network_fname, "r") as f:
        network = pd.DataFrame({"Solution": dict([["_".join(g.split("-")) for g in x.split(", ")] for x in f.read().split("\n") if (len(x)>0)])})
    influences = solution2influences(network["Solution"])
    assert influences.shape[0]==influences.shape[1]
    assert all([influences.index[i]==influences.columns[i] for i in range(influences.shape[0])])
    influences = influences.loc[(influences.abs().sum(axis=1)>0)&(influences.abs().sum(axis=0)>0)]
    gene_list = list(influences.index)
    network_connected = network.loc[gene_list]
    def get_all_genes(network):
        all_genes = list(network.index)
        for idx in network.index:
            grf = str(network.loc[idx]["Solution"])
            for symb in ["!","&","|","(",")"]:
                grf = " ".join(grf.split(symb))
            gene_lst = [g for g in grf.split(" ") if (g not in [" ", "0","1"] and len(g)>0)]
            all_genes += gene_lst
        return list(set(all_genes))
    full_lst = get_all_genes(network_connected)
    network_connected = network.loc[full_lst].to_dict()["Solution"]
    network_connected = "\n".join([", ".join(["_".join(g.split("-")) for g in [x,y]]) for x,y in network_connected.items()])
    network_fname_connected = network_fname.split(".bnet")[0]+"_connected.bnet"
    with open(network_fname_connected, "w") as f:
        f.write(network_connected) 
    return network_fname_connected

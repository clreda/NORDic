#coding:utf-8

from subprocess import call as sbcall
import pandas as pd
import numpy as np
import omnipath as op

from NORDic.UTILS.STRING_utils import get_interactions_from_STRING, get_network_from_STRING
from NORDic.UTILS.utils_data import request_biodbnet
from NORDic.UTILS.utils_grn import get_weakly_connected

import contextlib
@contextlib.contextmanager
def capture():
    import sys
    from io import StringIO
    oldout,olderr = sys.stdout, sys.stderr
    try:
        out=[StringIO(), StringIO()]
        sys.stdout,sys.stderr = out
        yield out
    finally:
        sys.stdout,sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()

def determine_edge_threshold(network, core_gene_set, quiet=True):
    '''
        Determine the greatest threshold on the edge score which allows all of the core gene set to be connected (binary search)
        @param\tnetwork\tPandas DataFrame: rows/[interactions] x at least three columns "preferredName_A" (input node), "preferredName_B" (output node), "score" (edge score)
        @param\tcore_gene_set\tPython character string list: list of genes that should remain connected
        @param\tquiet\tPython bool[default=None]: 
        @return\tt\tPython float: maximum threshold which allows the connection of all genes in the core set
    '''
    ppi = network[["preferredName_A","preferredName_B","score"]]
    ppi.columns = ["A","B","score"]
    glist = list(set(list(ppi["A"])+list(ppi["B"])))
    components = get_weakly_connected(ppi, glist, index_col="A", column_col="B", score_col="score")
    main_component = [g for g in components[0] if (g in core_gene_set)]
    if (not quiet):
        print("Connected core gene set of size %d" % len(main_component))
    isolated_genes = [g for c in components[1:] for g in c]
    if (len(isolated_genes)>0):
        test_notisolated = np.vectorize(lambda x : x not in isolated_genes)
        ppi = ppi.loc[test_notisolated(ppi["A"])&test_notisolated(ppi["B"])]
    t = ppi["score"].max()
    ppi_accepted = ppi.loc[ppi["score"]>=t]
    glist = [g for g in glist if (g not in isolated_genes)]
    components = get_weakly_connected(ppi_accepted, glist, index_col="A", column_col="B", score_col="score")
    if (not quiet):
        print("%d core genes in largest CC" % len([g for g in components[0] if (g in core_gene_set)]))
    isolated_genes = [g for c in components[1:] for g in c if (g not in components[0])]
    isolated_coregenes = [g for g in isolated_genes if (g in core_gene_set)]
    ## Add edges to/from isolated genes in decreasing score order
    ## until the PPI is connected and all genes in the connected PPI are present
    if (len(isolated_coregenes)>0):
        test_isolated = np.vectorize(lambda x : x in isolated_genes)
        ppi_rejected = ppi.loc[ppi["score"]<t]
        t_array = np.unique(ppi_rejected[["score"]].values)
        m, M = 0, len(t_array)
        while ((m!=M) or (M!=0) or (m!=len(t_array))):
            t = t_array[(M+m)//2]
            add_ppi = ppi_rejected.loc[ppi_rejected["score"]>=t]
            ppi_accepted_test = pd.concat((ppi_accepted, add_ppi), axis=0)
            components = get_weakly_connected(ppi_accepted_test, glist, index_col="A", column_col="B", score_col="score")
            if (not quiet):
                print("adding %d edges at t=%.6f (%d edges in total)" % (add_ppi.shape[0], t, ppi_accepted_test.shape[0]))
            isolated_genes = [g for c in components[1:] for g in c if (g not in components[0])]
            isolated_coregenes = [g for g in isolated_genes if (g in core_gene_set)]
            if (not quiet):
                print("%d components %s... (%d isolated genes of the core set)" % (len(components), str([len(c) for c in components[:3]]),len(isolated_coregenes)))
            if (len(isolated_coregenes)>0):
                M = (M+m)//2+1
            else:
                m = (M+m)//2
                break
        t = t_array[m]
    assert len([g for g in components[0] if (g in core_gene_set)])==len(main_component)
    return t

def merge_network_PPI(network, PPI, quiet=True):
    '''
        Merge two network while solving all inconsistencies (duplicates, paradoxes, etc.) in signs, directions, scores
        @param\tnetwork\tPandas DataFrame: rows/[interactions] x at least three columns "preferredName_A" (input node), "preferredName_B" (output node), "score" (edge score)
        @param\tPPI\tPandas DataFrame: rows/[interactions] x at least three columns "preferredName_A" (input node), "preferredName_B" (output node), "score" (edge score)
        @param\tquiet\tPython bool[default=None]: 
        @return\tfinal_network\tPandas DataFrame: rows/[interactions] x columns/[["preferredName_A", "preferredName_B", "sign", "directed", "score"]]
    '''
    edges_PPI = ["--".join(PPI.loc[x][["preferredName_A","preferredName_B"]]) for x in PPI.index]
    edges_network = ["--".join(network.loc[x][["preferredName_A","preferredName_B"]]) for x in network.index]
    if (not quiet):
        print("%d edges in network 1, %d in network 2, intersection=%d" % (len(edges_PPI),len(edges_network),len([x for ix, x in enumerate(network.index) if (edges_network[ix] in edges_PPI or "--".join([edges_network[ix].split("--")[-1], edges_network[ix].split("--")[0]]) in edges_PPI)])))
    add_indices = [x for ix, x in enumerate(network.index) if (edges_network[ix] not in edges_PPI and "--".join([edges_network[ix].split("--")[-1], edges_network[ix].split("--")[0]]) not in edges_PPI)]
    final_network = pd.concat((PPI, network.loc[add_indices]), axis=0)
    edges_final_network = ["--".join(final_network.loc[x][["preferredName_A","preferredName_B"]]) for x in final_network.index]
    scores = []
    for ix, x in enumerate(list(final_network["score"])):
        edge = edges_final_network[ix]
        inv_edge = "--".join([edge.split("--")[-1], edge.split("--")[0]])
        if (edge not in edges_network and inv_edge not in edges_network):
            scores.append(x)
        else:
            if (edge in edges_network):
                idx = edges_network.index(edge)
            elif (inv_edge in edges_network):
                idx = edges_network.index(inv_edge)
            x = network.iloc[idx,:]["score"]
            scores.append(x)
    final_network["score"] = scores
    return final_network

## https://workflows.omnipathdb.org/tissue-hpa.pdf
## https://workflows.omnipathdb.org/networks-r.html
def get_network_from_OmniPath(gene_list=None, disease_name=None, species="human", sources_int=None, domains_int=None, types_int=None, min_curation_effort=-1, domains_annot='HPA_tissue', quiet=False):
    '''
        Retrieve a network from OmniPath
        @param\tgene_list\tPython character string[default=None]: List of genes to consider (or do not filter the interactions from Omnipath if =None)
        @param\tdisease_name\tPython character string[default=None]: Disease name (in letters) to consider
        @param\tspecies\tPython character string[default=None]: Species to consider (either "human", "mouse", or "rat")
        @param\tsources_int\tPython character string[default=None]: Which databases for interactions to consider (if =None, consider them all)
        @param\tdomains_int\tPython character string[default=None]:
        @param\ttypes_int\tPython character string[default=None]: Types of interactions, e.g., "post_translational", "transcriptional", "post_transcriptional", "mirna_transcriptional"
        @param\tmin_curation_effort\tPython integer[default=-1]: if positive, select edges based on that criteria (the higher, the better). Counts the unique database-citation pairs, i.e. how many times was an interaction described in a paper and mentioned in a database
        @param\tdomain_annot\tPython integer[default='HPA_tissue']:
        @param\tquiet\tPython bool[default=None]:  
        @return\tfinal_network\tPandas DataFrame: rows/[interactions] x columns/[["preferredName_A", "preferredName_B", "sign", "directed", "score"]]
                \tannot_wide\tPandas DataFrame: rows/[gene symbols] x columns/[annotations from the database @domains_annot]
    '''
    assert species in ['human', 'mouse', 'rat']
    assert types_int in [None, "post_translational", "transcriptional", "post_transcriptional", "mirna_transcriptional"]
    assert sources_int in [None]+list(op.interactions.AllInteractions.resources())
    assert domains_int in [None, "DOROTHEA", "KINASE_EXTRA", "LIGREC_EXTRA", "LNCRNA_MRNA", "MIRNA_TARGET","OMNIPATH",
        "PATHWAY_EXTRA", "SMALL_MOLECULE", "TF_MIRNA", "TF_REGULONS", "TF_TARGET"]
    assert domains_annot in [None]+list(op.requests.Annotations.resources())
    assert (gene_list is not None) or (disease_name is not None)
    params = {"organisms": species, "include": domains_int, "sources": sources_int, "fields": ['curation_effort'], #'references', 'sources', 'type'], 
              "genesymbols": True, "types": types_int, "directed": False, "signed": False} 
    with capture() as out:
        interactions = op.interactions.AllInteractions.get(**params).squeeze('columns')
    if (not quiet):
        print("Getting all interactions from OmniPath with the following parameters:%s..." % str(params))
    interactions = interactions[interactions["curation_effort"]>min_curation_effort]
    interactions["directed"] = interactions["consensus_direction"].astype(int)
    interactions["sign"] = interactions["consensus_stimulation"].astype(int)-interactions["consensus_inhibition"].astype(int)+2*(interactions["consensus_stimulation"].astype(int)*interactions["consensus_inhibition"].astype(int)+(1-interactions["consensus_stimulation"].astype(int))*(1-interactions["consensus_inhibition"].astype(int)))
    interactions = interactions[['source_genesymbol', 'target_genesymbol', 'directed', 'sign']]
    interactions.columns = ["preferredName_A","preferredName_B","directed","sign"]
    if (not quiet):
        print("Define a network-formatted dataframe from these interactions...")
    if (gene_list is not None):
        interactions = interactions[interactions.preferredName_A.isin(gene_list)&interactions.preferredName_B.isin(gene_list)]
    else:
        gene_list = list(set(list(interactions["preferredName_A"])+list(interactions["preferredName_B"])))
    if (not quiet):
        print("Obtain the gene list or filter interactions using the given gene list (%d genes)..." % len(gene_list))
    annot_ls = []
    chunks = np.array_split(gene_list, len(gene_list)//500+1)
    for i, gene_ls in enumerate(chunks):
        if (not quiet):
            print("%d/%d gene annotation chunks" % (i+1, len(chunks)))
        with capture() as out:
            annot_ls.append(op.requests.Annotations.get(proteins=gene_ls,resources=domains_annot).squeeze("columns"))
        annot = pd.concat(tuple(annot_ls), axis=0)
    annot = annot[["genesymbol", "label", "value"]]
    annot_wide = pd.pivot_table(annot, index='genesymbol', columns='label', values='value', aggfunc='first')
    return interactions, annot_wide

def remove_isolated(network, quiet=False):
    '''
        Remove all nodes which do not belong to the largest connected component from the network
        @param\tnetwork\tPandas DataFrame: rows/[interactions] x columns/[["preferredName_A", "preferredName_B", "sign", "directed", "score"]]
        @param\tquiet\tPython bool[default=None]:  
        @return\ttrimmed_network\tPandas DataFrame: rows/[interactions] x columns/[["preferredName_A", "preferredName_B", "sign", "directed", "score"]]
    '''
    glist = list(set(list(network["preferredName_A"])+list(network["preferredName_B"])))
    components = get_weakly_connected(network, glist, score_col="score")
    is_comp0 = np.vectorize(lambda x : x in components[0])
    if (not quiet):
        print("%d genes in total in the network (%d genes in the largest connected component)" % (len(glist), len(components[0])))
    network_trimmed = network.loc[is_comp0(network["preferredName_A"])&is_comp0(network["preferredName_B"])]
    if (not quiet):
        print("%d edges in trimmed network (down from %d edges)" % (network_trimmed.shape[0], network.shape[0]))
    return network_trimmed.sort_values(by="score", ascending=False)

def aggregate_networks(file_folder, gene_list, taxon_id, min_score, network_type, app_name, version_net="11.5", version_act="11.0", quiet=0):
    '''
        This function performs the following pipeline to build a prior knowledge network based on a subset of genes
	- Retrieve protein actions and predicted PPIs from STRING
	- Merge the two networks while solving all inconsistencies (duplicates, paradoxes, etc.) in signs, directions, scores
	- Determine the greatest threshold on the edge score which allows all of the core gene set to be connected (binary search)
	- Trim out edges which scores are below the threshold, and remove all isolated nodes
        @param\tfile_folder\tPython character string: relative path where to store files
        @param\tgene_list\tPython character string list: list of core gene symbols to preserve in the network
        @param\ttaxon_id\tPython integer: NCBI taxonomy ID
        @param\tmin_score\tPython integer: minimum score on edges retrieved from the STRING database
        @param\tapp_name\tPython character string: Identifier for STRING requests
        @param\tversion_net\tPython character string[default="11.5"]: Number of version for interaction data in the STRING database. To avoid compatibility issues, it is strongly advised not to change this parameter
        @param\tversion_act\tPython character string[default="11.0"]: Number of version for protein action data in the STRING database. To avoid compatibility issues, it is strongly advised not to change this parameter
        @param\tquiet\tPython bool[default=None]: 
        @return\tfinal_network\tPandas DataFrame: rows/[interactions] x columns/[["preferredName_A", "preferredName_B", "sign", "directed", "score"]]
    '''
    network = get_network_from_STRING(gene_list, taxon_id, min_score=min_score, network_type=network_type, add_nodes=0, app_name=app_name, version=version_net, quiet=quiet)
    PPI = get_interactions_from_STRING(gene_list, taxon_id, min_score=min_score, strict=False, version=version_act, app_name=app_name, file_folder=file_folder)
    final_network = merge_network_PPI(network, PPI)
    genes = list(set(list(final_network["preferredName_A"])+list(final_network["preferredName_B"])))
    threshold = determine_edge_threshold(final_network, genes)
    final_network_thres = final_network.loc[final_network["score"]>=threshold]
    ## remove redundant edges
    final_network_trimmed_thres = final_network_thres.sort_values(by="score", ascending=False).drop_duplicates(keep="first")
    ## remove isolated nodes
    final_network_trimmed_thresf = remove_isolated(final_network_trimmed_thres)
    return final_network_trimmed_thresf.sort_values(by="score", ascending=False)
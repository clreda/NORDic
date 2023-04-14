#coding: utf-8

import NORDic
import os
import pandas as pd

from NORDic.UTILS.STRING_utils import get_interactions_from_STRING, get_network_from_STRING
from NORDic.UTILS.utils_network import merge_network_PPI

file_folder="ToyOndine/"

from NORDic.UTILS.STRING_utils import get_interactions_from_STRING, get_network_from_STRING
from NORDic.UTILS.utils_network import merge_network_PPI

## Retrieve networks from the STRING database

core_genes = ["PHOX2B", "RET", "BDNF", "ASCL1", "EDN3", "GDNF"]
taxon_id = 9606

if (not os.path.exists(file_folder+"network.csv")):
    network = get_network_from_STRING(core_genes, taxon_id, min_score=0., network_type="functional",
            add_nodes=0, app_name="NORDic PKN", version="11.5", quiet=0)
    network.to_csv(file_folder+"network.csv")
network = pd.read_csv(file_folder+"network.csv", index_col=0)

if (not os.path.exists(file_folder+"PPI.csv")):
    PPI = get_interactions_from_STRING(core_genes, taxon_id, min_score=0., strict=False,
            version="11.0", app_name="NORDic PKN PPI", file_folder=file_folder)
    PPI.to_csv(file_folder+"PPI.csv")
PPI = pd.read_csv(file_folder+"PPI.csv", index_col=0)

if (not os.path.exists(file_folder+"final_network.csv")):
    final_network = merge_network_PPI(network, PPI)
    final_network.to_csv(file_folder+"final_network.csv")
final_network = pd.read_csv(file_folder+"final_network.csv", index_col=0)

from NORDic.UTILS.utils_network import determine_edge_threshold, remove_isolated

threshold = determine_edge_threshold(final_network, core_genes)
assert 0 < threshold and threshold < 1

from NORDic.UTILS.utils_grn import get_weakly_connected
from functools import reduce

final_network = remove_isolated(final_network.loc[final_network["score"]>=threshold])
glist = list(set(reduce(lambda x,y: x+y, [list(final_network[c]) for c in ["preferredName_A", "preferredName_B"]])))
components = get_weakly_connected(final_network, glist, score_col="score")
assert len(components)==1 and len(glist)==len(components[0])
assert all([g in components[0] for g in core_genes])

final_network.index = range(final_network.shape[0])
NETWORK_fname = file_folder+"network.tsv"
final_network.to_csv(NETWORK_fname, sep="\t", index=None)

from NORDic.UTILS.utils_plot import plot_influence_graph

plot_influence_graph(final_network, "preferredName_A", "preferredName_B", "sign", direction_col="directed", fname=file_folder+"graph_final", optional=True)

from NORDic.UTILS.utils_network import aggregate_networks

help(aggregate_networks)

final_network2 = aggregate_networks(file_folder, core_genes, taxon_id, 0, "functional", "NORDic whole PKN", quiet=1)
glist2 = list(set(reduce(lambda x,y: x+y, [list(final_network2[c]) for c in ["preferredName_A", "preferredName_B"]])))
components2 = get_weakly_connected(final_network2, glist2, score_col="score")
assert len(components2)==1 and len(glist2)==len(components2[0])
assert all([g in components2[0] for g in core_genes])

from NORDic.UTILS.utils_grn import reconnect_network

help(reconnect_network)

## Retrieve networks from OmniPath

from NORDic.UTILS.utils_network import get_network_from_OmniPath

#interactions, annot_wide = get_network_from_OmniPath(gene_list=core_genes, species="human", min_curation_effort=-1, quiet=False)
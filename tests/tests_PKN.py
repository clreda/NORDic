#coding: utf-8

import NORDic
import os
import pandas as pd
import unittest
from subprocess import Popen

from functools import reduce

class TestPKN(unittest.TestCase):

    def create_network(self):
        ## Retrieve networks from the STRING database
        from NORDic.UTILS.STRING_utils import get_network_from_STRING
        file_folder="ToyOndine/"
        core_genes = ["PHOX2B", "RET", "BDNF", "ASCL1", "EDN3", "GDNF"]
        taxon_id = 9606
        if (not os.path.exists(file_folder+"network.csv")):
            network = get_network_from_STRING(core_genes, taxon_id, min_score=0., network_type="functional",
                add_nodes=0, app_name="NORDic PKN", version="11.5", quiet=0)
            network.to_csv(file_folder+"network.csv")
        network = pd.read_csv(file_folder+"network.csv", index_col=0)
        return network

    def create_PPI(self):
        ## Retrieve networks from the STRING database
        from NORDic.UTILS.STRING_utils import get_interactions_from_STRING
        file_folder="ToyOndine/"
        core_genes = ["PHOX2B", "RET", "BDNF", "ASCL1", "EDN3", "GDNF"]
        taxon_id = 9606
        if (not os.path.exists(file_folder+"PPI.csv")):
            PPI = get_interactions_from_STRING(core_genes, taxon_id, min_score=0., strict=False,
                    version="11.0", app_name="NORDic PKN PPI", file_folder=file_folder)
            PPI.to_csv(file_folder+"PPI.csv")
        PPI = pd.read_csv(file_folder+"PPI.csv", index_col=0)
        return PPI

    def test_merge_network_PPI(self):
        ## Retrieve networks from the STRING database
        from NORDic.UTILS.utils_network import merge_network_PPI
        file_folder="ToyOndine/"
        core_genes = ["PHOX2B", "RET", "BDNF", "ASCL1", "EDN3", "GDNF"]
        taxon_id = 9606
        network = self.create_network()
        PPI = self.create_PPI()
        if (not os.path.exists(file_folder+"final_network.csv")):
            final_network = merge_network_PPI(network, PPI)
            final_network.to_csv(file_folder+"final_network.csv")
        final_network = pd.read_csv(file_folder+"final_network.csv", index_col=0)
        return final_network, core_genes

    def test_determine_threshold(self):
        from NORDic.UTILS.utils_network import determine_edge_threshold
        from NORDic.UTILS.utils_network import remove_isolated
        from NORDic.UTILS.utils_grn import get_weakly_connected
        from functools import reduce
        file_folder="ToyOndine/"
        final_network, core_genes = self.test_merge_network_PPI()
        threshold = determine_edge_threshold(final_network, core_genes)
        self.assertTrue(0 < threshold)
        self.assertTrue(threshold < 1)
        final_network = remove_isolated(final_network.loc[final_network["score"]>=threshold])
        glist = list(set(reduce(lambda x,y: x+y, [list(final_network[c]) for c in ["preferredName_A", "preferredName_B"]])))
        components = get_weakly_connected(final_network, glist, score_col="score")
        self.assertEqual(len(components),1)
        self.assertEqual(len(glist),len(components[0]))
        self.assertTrue(all([g in components[0] for g in core_genes]))
        final_network.index = range(final_network.shape[0])
        NETWORK_fname = file_folder+"network.tsv"
        final_network.to_csv(NETWORK_fname, sep="\t", index=None)
        return final_network

    def test_plot_influence_graph(self):
        from NORDic.UTILS.utils_plot import plot_influence_graph
        file_folder="ToyOndine/"
        final_network = self.test_determine_threshold()
        plot_influence_graph(final_network, "preferredName_A", "preferredName_B", "sign", direction_col="directed", fname=file_folder+"graph_final", optional=True)

    def test_full_pipeline(self):
        from NORDic.UTILS.STRING_utils import get_network_from_STRING
        from NORDic.UTILS.utils_network import aggregate_networks
        from NORDic.UTILS.utils_grn import get_weakly_connected
        file_folder="ToyOndine_full/"
        Popen(('cp -r ToyOndine/ '+file_folder).split(" "))
        core_genes = ["PHOX2B", "RET", "BDNF", "ASCL1", "EDN3", "GDNF"]
        taxon_id = 9606
        network = get_network_from_STRING(core_genes, taxon_id, min_score=0., network_type="functional",
                add_nodes=0, app_name="NORDic PKN", version="11.5", quiet=0)
        network.to_csv(file_folder+"network.csv")
        final_network2 = aggregate_networks(file_folder, core_genes, taxon_id, 0, "functional", "NORDic whole PKN", quiet=1)
        glist2 = list(set(reduce(lambda x,y: x+y, [list(final_network2[c]) for c in ["preferredName_A", "preferredName_B"]])))
        components2 = get_weakly_connected(final_network2, glist2, score_col="score")
        self.assertEqual(len(components2),1)
        self.assertEqual(len(glist2),len(components2[0]))
        self.assertTrue(all([g in components2[0] for g in core_genes]))

if __name__ == '__main__':
    unittest.main()
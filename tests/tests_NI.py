#coding: utf-8

import NORDic
import os
import pandas as pd
from multiprocessing import cpu_count
import numpy as np
from subprocess import call
import unittest

## I. Building a small regulatory model of CCHS (**NORDic NI**)

class TestNI(unittest.TestCase):

    def create_instance(self):
        ## Registration to databases
        DisGeNET_credentials = "../tests/credentials_DISGENET.txt"
        STRING_credentials = "../tests/credentials_STRING.txt"
        LINCS_credentials = "../tests/credentials_LINCS.txt"
        ## Parameters
        seed_number=123456
        njobs=max(1,cpu_count()-2) ## all available threads but 2
        file_folder="ToyOndine/"
        taxon_id=9606 # human species
        disease_cids=["C1275808"] ## Concept ID of Ondine syndrome
        cell_lines=["NPC", "SHSY5Y"] # brain cell lines in LINCS L1000
        ## Information about the disease
        DISGENET_args = {"credentials": DisGeNET_credentials, "disease_cids": disease_cids}
        ## Selection of parameters relative to the prior knowledge network 
        STRING_args = {"credentials": STRING_credentials, "score": 0}
        EDGE_args = {"tau": 0, "filter": True, "connected": True}
        accept_nonRNA=True
        preserve_network_sign=True
        ## Selection of parameters relative to experimental constraints
        LINCS_args = {"path_to_lincs": "../lincs/", "credentials": LINCS_credentials, "cell_lines": cell_lines, 
              "thres_iscale": None}
        SIG_args = {"bin_thres": 0.5}
        force_experiments=False
        ## Selection of parameters relative to the inference of networks
        BONESIS_args = {"limit": 1, "exact": True, "max_maxclause": 3}
        ## Advanced
        DESIRABILITY = {"DS": 3, "CL": 3, "Centr": 3, "GT": 1}
        call("mkdir -p "+file_folder, shell=True)

        ## The undirected, unsigned network from STRING
        network_content = pd.DataFrame([], index=["preferredName_A", "preferredName_B", "sign", "directed", "score"])
        network_content[0] = ["PHOX2B", "BDNF", 2, 0, 0.342]
        network_content[1] = ["PHOX2B", "GDNF", 2, 0, 0.572]
        network_content[2] = ["PHOX2B", "RET", 2, 0, 0.605]
        network_content[3] = ["PHOX2B", "EDN3", 2, 0, 0.607]
        network_content[4] = ["PHOX2B", "ASCL1", 2, 0, 0.676]
        network_content[5] = ["ASCL1", "RET", 2, 0, 0.397]
        network_content[6] = ["ASCL1", "EDN3", 2, 0, 0.433]
        network_content[7] = ["ASCL1", "GDNF", 2, 0, 0.47]
        network_content[8] = ["ASCL1", "BDNF", 2, 0, 0.519]
        network_content[9] = ["EDN3", "BDNF", 2, 0, 0.15]
        network_content[10] = ["EDN3", "RET", 2, 0, 0.622]
        network_content[11] = ["EDN3", "GDNF", 2, 0, 0.634]
        network_content[12] = ["RET", "BDNF", 2, 0, 0.438]
        network_content[12] = ["RET", "GDNF", 2, 0, 0.999]
        network_content[12] = ["GDNF", "BDNF", 2, 0, 0.95]
        network_content.T.to_csv(file_folder+"network.tsv", sep="\t", index=None)

        ## An experiment retrieved from LINCS L1000 involving those genes
        index=["PHOX2B","EDN3","RET","GDNF","BDNF","ASCL1","cell_line","annotation","perturbed","perturbation","sigid"]
        experiments_content = pd.DataFrame([], index=index)
        experiments_content["BDNF_KD_SHSY5Y"] = [1,1,1,0,0,0,"Cell","2","BDNF","KD","Sig1"] ## mutated profile
        experiments_content["initial_SHSY5Y"] = [1,1,1,1,0,0,"Cell","1","None","None","Sig2"] ## control/initial profile
        experiments_content.to_csv(file_folder+"experiments.csv")
        return file_folder, taxon_id, DISGENET_args, STRING_args, EDGE_args, LINCS_args, SIG_args, BONESIS_args,DESIRABILITY, seed_number, njobs, force_experiments, accept_nonRNA, preserve_network_sign

    def test_network_identification(self):
        file_folder, taxon_id, DISGENET_args, STRING_args, EDGE_args, LINCS_args, SIG_args, BONESIS_args,DESIRABILITY, seed_number, njobs, force_experiments, accept_nonRNA, preserve_network_sign = self.create_instance()
        ## Network identification
        from NORDic.NORDic_NI.functions import network_identification
        solution = network_identification(file_folder, taxon_id, path_to_genes=None, 
            network_fname=file_folder+"network.tsv", experiments_fname=file_folder+"experiments.csv", 
            disgenet_args=DISGENET_args, string_args=STRING_args, edge_args=EDGE_args, lincs_args=LINCS_args, 
            sig_args=SIG_args, bonesis_args=BONESIS_args, weights=DESIRABILITY, seed=seed_number, njobs=njobs,
            force_experiments=force_experiments, accept_nonRNA=accept_nonRNA, preserve_network_sign=preserve_network_sign)
        self.assertTrue(os.path.exists(file_folder+'inferred_max_criterion_solution.png'))
        with open(file_folder+"solution.bnet", "r") as f:
            network = f.read().split("\n")
        self.assertEqual(len(network),6)

if __name__ == '__main__':
    unittest.main()
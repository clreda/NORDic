#coding: utf-8

import NORDic
import os
import pandas as pd
from multiprocessing import cpu_count
import numpy as np
import mpbn
import unittest

## II. Prioritization of master regulators (**NORDic PMR**)

class TestPMR(unittest.TestCase):

    def create_instance(self):
        ## Path to files
        seed_number=123456
        njobs=min(5,max(1,cpu_count()-2))
        file_folder="ToyOndine/"
        ## Options for PMR
        np.random.seed(seed_number)
        with open(file_folder+"solution.bnet", "r") as f:
            genes = [line.split(", ")[0] for line in f.read().split("\n") if (len(line)>0)]
        k=1
        IM_params = {"seed": seed_number, "gene_inputs": genes, "gene_outputs": genes}
        SIMU_params = {'nb_sims': 1000, 'rates': "fully_asynchronous", 'thread_count': njobs, 'depth': "constant_unitary"}
        ## Generate "patient" states
        ### Create the mutated networks
        f_wildtype = mpbn.load(file_folder+"solution.bnet")
        f_mutant = mpbn.MPBooleanNetwork(f_wildtype)
        f_mutant["PHOX2B"] = 0 ## CCHS ~ PHOX2B mutant
        ### Initial states drawn at random
        state_len = 20
        states = pd.DataFrame(
              [np.random.choice([0,1], p=[0.5,0.5], size=len(genes)).tolist() for _ in range(state_len)]
              , columns=genes, index=range(state_len)).T
        A_WT, A_mut = [], []
        for state in states.columns:
            x0 = f_wildtype.zero()
            for i in list(states.loc[states[state]==1].columns):
                x0[i] = 1
            ### Get the reachable attractors from initial state in the presence/absence of X0 knockout
            A_WT += [a for a in list(f_wildtype.attractors(reachable_from=x0))]
            A_mut += [a for a in list(f_mutant.attractors(reachable_from=x0))]
        for a in A_WT:
            a.update({"annotation": 1})
        for a in A_mut:
            a.update({"annotation": 2})
        controls = {"Control%d" % (i+1): a for i,a in enumerate(A_WT)}
        patients = {"Case%d" % (i+1): a for i,a in enumerate(A_mut)}
        controls.update(patients)
        states = pd.DataFrame(controls)
        states.to_csv(file_folder+"phenotypes.csv")
        patients = pd.DataFrame(patients)
        patients = patients.loc[[i for i in patients.index if (i != "annotation")]]
        return file_folder, k, patients, IM_params, SIMU_params

    def test_prioritization_master_regulators(self):
        file_folder, k, patients, IM_params, SIMU_params = self.create_instance()
        ## Prioritization of Master Regulators
        from NORDic.NORDic_PMR.functions import greedy
        S, spreads = greedy(file_folder+"solution.bnet", k, patients, IM_params, SIMU_params, save_folder=file_folder)
        self.assertEqual(spreads.shape[0],6)
        self.assertEqual(np.max(spreads.values),spreads.loc[S[0][0]][str(S)])

if __name__ == '__main__':
    unittest.main()
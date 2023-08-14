#coding: utf-8

import NORDic
import os
import pandas as pd
from multiprocessing import cpu_count
import numpy as np
import unittest

## IV. Drug Repurposing (**NORDic DR**)

class TestDR(unittest.TestCase):

    def create_instance(self):
        file_folder="ToyOndine/"
        seed_number=12345
        njobs=min(5,max(1,cpu_count()-2))
        SIMU_params = {'nb_sims': 1000, 'rates': "fully_asynchronous", 'thread_count': njobs, 'depth': "constant_unitary"}

        np.random.seed(seed_number)
        with open(file_folder+"solution.bnet", "r") as f:
            genes = [line.split(", ")[0] for line in f.read().split("\n") if (len(line)>0)]

        ## Create a set of random "patient" and "control" states (see tests_PMR.py)
        states = pd.read_csv(file_folder+"phenotypes.csv", index_col=0)
        ### Patient profiles
        patients = states[[c for c in states.columns if ("Case" in c)]].loc[[i for i in states.index if (i != "annotation")]]

        ## Drug targets
        targets = pd.read_csv(file_folder+"targets.csv", index_col=0)

        ## Define the function which determines whether a state is closer to controls/patients
        dfdata = states.loc[[i for i in states.index if (i != "annotation")]]
        samples = states.loc["annotation"]
        return dfdata, samples, seed_number, SIMU_params, states, patients, targets, file_folder

    def test_compute_frontier(self):
        from NORDic.NORDic_DS.functions import compute_frontier
        dfdata, samples, seed_number, SIMU_params, states, patients, targets, file_folder = self.create_instance()
        ## compute_frontier fits a SVM model to the data
        frontier = compute_frontier(dfdata, samples)
        score = lambda attrs : (frontier.predict(attrs.values.T)==1).astype(int)
        return score, seed_number, SIMU_params, states, patients, targets, file_folder

    def test_drug_repurposing(self):
        score, seed_number, SIMU_params, states, patients, targets, file_folder = self.test_compute_frontier()
        ## Drug Repurposing through adaptive testing
        from NORDic.NORDic_DR.functions import adaptive_testing
        BANDIT_args = {
            'bandit': 'LinGapE', 
            'seed': seed_number,
            'delta': 0.1, 
            'nsimu': 1, 
            'm': 4, 
            'c': 0, 
            'sigma': 1,
            'beta': "heuristic",
            'epsilon': 0.1,
            'tracking_type': "D",
            'gain_type': "empirical",
            'learner': "AdaHedge"
        }
        recommendation = adaptive_testing(file_folder+"solution.bnet", targets, targets, score, patients, 
                SIMU_params, BANDIT_args, reward_fname=None, quiet=False).T
        self.assertEqual(recommendation.shape[0],targets.shape[1])
        self.assertTrue(np.isclose(recommendation.sum().sum(), BANDIT_args["m"]))

if __name__ == '__main__':
    unittest.main()
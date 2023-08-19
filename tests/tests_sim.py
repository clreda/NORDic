#coding: utf-8

import NORDic
import pandas as pd
import numpy as np
import os
from multiprocessing import cpu_count
from time import time
import unittest

class TestSIM(unittest.TestCase):

    def create_instance(self):
        taxon_id = 9606
        njobs=max(1,cpu_count()-2)
        seednb=12345
        file_folder="ToyOndine/"
        network_fname=file_folder+"solution.bnet"
        import mpbn
        net = mpbn.MPBooleanNetwork(mpbn.load(network_fname))
        x0 = net.zero() # consider initial state where all gene expression levels are set to 0...
        for g in ['BDNF', 'EDN3']: #... except for 'BDNF' and 'EDN3'
            x0[g] = 1
        return x0, network_fname, seednb, njobs

    def test_MABOSS_SIM(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        params = {'sample_count': 10000, 'max_time': 100}
        nee = MABOSS_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False)
        probs = nee.generate_trajectories(params=params, outputs=[]) ## generate trajectories
        self.assertTrue(all([c=="<nil>" for c in probs.columns]))
        self.assertTrue(np.isclose(probs.loc["prob"]["<nil>"],1))

    def test_MPBN_SIM(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MPBN_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        params = {'sample_count': 10000, 'max_time': 100}
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False) ## no mutations, initial state x0
        probs = nee.generate_trajectories(params=params, outputs=[]) ## generate trajectories
        self.assertTrue(all([c=="<nil>" for c in probs.columns]))
        self.assertTrue(np.isclose(probs.loc["prob"]["<nil>"],1))

    def test_BONESIS_SIM_FinalEmpty(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        final = pd.DataFrame([], index=[g for g in x0], columns=["final"]).fillna(0) 
        nee = BONESIS_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, final, verbose=False) ## no mutations, initial state x0
        attrs = nee.enumerate_attractors(verbose=False) 
        self.assertTrue("StateNotFound" not in attrs.columns[0])

    def test_BONESIS_SIM_FinalState(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        final = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["final"]).T 
        nee = BONESIS_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, final, verbose=False) ## no mutations, initial state x0
        attrs = nee.enumerate_attractors(verbose=False) 
        self.assertTrue("StateNotFound" in attrs.columns[0])

    def test_MABOSS_SIM_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        nee = MABOSS_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False)
        nee.enumerate_attractors() ## enumerate all attractors
        self.assertEqual(nee.attrs.shape[1], 1)
        self.assertTrue(all(["".join(list(nee.attrs.astype(str)[c])) in ["1"*len(nee.gene_list), "0"*len(nee.gene_list)] for c in nee.attrs]))

    def test_MPBN_SIM_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MPBN_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False)
        nee.enumerate_attractors() ## enumerate all attractors
        self.assertEqual(nee.attrs.shape[1],1)
        self.assertTrue(all(["".join(list(nee.attrs.astype(str)[c])) in ["1"*len(nee.gene_list), "0"*len(nee.gene_list)] for c in nee.attrs]))

    def test_MABOSS_Trajectories_Profiles(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        _, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
        res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(res.loc["score"]["initial->NotReachable"]<1)

    def test_MABOSS_Trajectories_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        _, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
        res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable"],1))

    def test_MPBN_Trajectories_Profiles(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(res.loc["score"]["initial->NotReachable"]<1)

    def test_MPBN_Trajectories_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable"],1))

    def test_BONESIS_Trajectories_Profiles(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        _, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
        res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable"],0))

    def test_BONESIS_Trajectories_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        _, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
        res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable"],0))

    def test_MPBN_Mutant_KO(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, mutation_permanent={'PHOX2B': 0}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('PHOX2B' in nee.all_mutants)

    def test_MPBN_Mutant_OE(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, mutation_permanent={'PHOX2B': 1}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('PHOX2B' in nee.all_mutants)

    def test_MPBN_Mutant_KO_transient(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, mutation_transient={'PHOX2B': 0}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('PHOX2B' in nee.all_mutants)

    def test_MPBN_Mutant_OE_transient(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, mutation_transient={'PHOX2B': 1}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('PHOX2B' in nee.all_mutants)

    def test_MPBN_Trajectory_Profiles_Mutant_KO(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'PHOX2B': 0},verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(res.loc["score"]["initial->NotReachable"]<1)

    def test_MPBN_Trajectory_Profiles_Mutant_KO_transient(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,1,1,1,1]], columns=[g for g in x0], index=["NotReachable"]).T 
        T = T.join(pd.DataFrame([], index=[g for g in x0], columns=["Reachable"]).fillna(0), how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_transient={'EDN3': 0},verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(res.loc["score"]["initial->NotReachable"]<1)

if __name__ == '__main__':
    unittest.main()
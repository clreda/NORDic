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
        network_fname=file_folder+"solution_small.bnet"
        import mpbn
        net = mpbn.MPBooleanNetwork(mpbn.load(network_fname))
        x0 = net.zero() # consider initial state where all gene expression levels are set to 0...
        for g in ['RET']: #... except for 'RET'
            x0[g] = 1
        #import networkx as nx
        #import matplotlib.pyplot as plt
        #G = net.dynamics(update_mode="mp", init=x0)
        #nx.draw(G, cmap = plt.get_cmap('jet'), with_labels=True)
        #plt.savefig("dynamics_mpbn.png", bbox_inches="tight")
        #plt.close()
        #G = net.dynamics(update_mode="asynchronous", init=x0)
        #nx.draw(G, cmap = plt.get_cmap('jet'), with_labels=True)
        #plt.savefig("dynamics_maboss.png", bbox_inches="tight")
        #plt.close()
        #x1 = net.zero() # consider initial state where all gene expression levels are set to 0
        #G = net.dynamics(update_mode="mp", init=x1)
        #nx.draw(G, cmap = plt.get_cmap('jet'), with_labels=True)
        #plt.savefig("dynamics_mpbn_000.png", bbox_inches="tight")
        #plt.close()
        return x0, network_fname, seednb, njobs
        
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
        final = pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["final"]).T 
        nee = BONESIS_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, final, verbose=False) ## no mutations, initial state x0
        attrs = nee.enumerate_attractors(verbose=False) 
        self.assertTrue("StateNotFound" in attrs.columns[0])
        
    def test_MABOSS_SIM(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        params = {'sample_count': 10000, 'max_time': 100}
        nee = MABOSS_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False)
        probs = nee.generate_trajectories(params=params, outputs=[]) ## generate trajectories
        self.assertTrue(any([c=="<nil>" for c in probs.columns]))

    def test_MPBN_SIM(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MPBN_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        params = {'sample_count': 10000, 'max_time': 100}
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False) ## no mutations, initial state x0
        probs = nee.generate_trajectories(params=params, outputs=[]) ## generate trajectories
        self.assertTrue(any([c=="<nil>" for c in probs.columns]))
        
    def test_MABOSS_SIM_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        nee = MABOSS_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False)
        nee.enumerate_attractors() ## enumerate all attractors
        self.assertTrue(all(["".join(list(nee.attrs.astype(str)[c])) in ["1"*len(nee.gene_list), "0"*len(nee.gene_list)] for c in nee.attrs]))
        
    def test_MPBN_SIM_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        from NORDic.UTILS.utils_sim import MPBN_SIM
        initial = pd.DataFrame(pd.Series(x0), columns=["initial"])
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, initial, verbose=False)
        nee.enumerate_attractors() ## enumerate all attractors
        self.assertTrue(all(["".join(list(nee.attrs.astype(str)[c])) in ["1"*len(nee.gene_list), "0"*len(nee.gene_list)] for c in nee.attrs]))
        
    def test_MABOSS_Trajectories_Profiles(self): ## compare (similarity-wise) attractors under potential mutations and profiles
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotAttractor1"]).T 
        T = T.join(pd.DataFrame([[1,1,1]], columns=[g for g in x0], index=["Attractor1"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,0]], columns=[g for g in x0], index=["Attractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,1,0]], columns=[g for g in x0], index=["NotAttractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,0]], columns=[g for g in x0], index=["NotAttractor3"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,1]], columns=[g for g in x0], index=["NotAttractor4"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotAttractor5"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,1]], columns=[g for g in x0], index=["NotAttractor6"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        _, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
        res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
        for i in [1,2]:
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->Attractor{i}"],1))
        for i in range(1,7):
        	self.assertTrue(res.loc["score"][f"initial->NotAttractor{i}"]<1)
        	
    def test_MABOSS_Trajectories_Attractors(self): ## compare (similarity-wise) attractors under potential mutations and attractors without mutations
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotAttractor1"]).T 
        T = T.join(pd.DataFrame([[1,1,1]], columns=[g for g in x0], index=["Attractor1"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,0]], columns=[g for g in x0], index=["Attractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,1,0]], columns=[g for g in x0], index=["NotAttractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,0]], columns=[g for g in x0], index=["NotAttractor3"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,1]], columns=[g for g in x0], index=["NotAttractor4"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotAttractor5"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,1]], columns=[g for g in x0], index=["NotAttractor6"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        _, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
        res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
        for i in [1,2]:
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->Attractor{i}"],1))
        for i in range(1,7):
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->NotAttractor{i}"],1)) ## we obtain the same attractor set
        	
    def test_MPBN_Trajectories_Profiles(self): ## same but with MP dynamics instead of fully asynchronous
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotAttractor1"]).T 
        T = T.join(pd.DataFrame([[1,1,1]], columns=[g for g in x0], index=["Attractor1"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,0]], columns=[g for g in x0], index=["Attractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,1,0]], columns=[g for g in x0], index=["NotAttractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,0]], columns=[g for g in x0], index=["NotAttractor3"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,1]], columns=[g for g in x0], index=["NotAttractor4"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotAttractor5"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,1]], columns=[g for g in x0], index=["NotAttractor6"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        for i in [1,2]:
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->Attractor{i}"],1))
        for i in range(1,7):
        	self.assertTrue(res.loc["score"][f"initial->NotAttractor{i}"]<1)
        	
    def test_MPBN_Trajectories_Attractors(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotAttractor1"]).T 
        T = T.join(pd.DataFrame([[1,1,1]], columns=[g for g in x0], index=["Attractor1"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,0]], columns=[g for g in x0], index=["Attractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,1,0]], columns=[g for g in x0], index=["NotAttractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,0]], columns=[g for g in x0], index=["NotAttractor3"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,1]], columns=[g for g in x0], index=["NotAttractor4"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotAttractor5"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,1]], columns=[g for g in x0], index=["NotAttractor6"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        for i in [1,2]:
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->Attractor{i}"],1))
        for i in range(1,7):
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->NotAttractor{i}"],1))
        	
    def test_BONESIS_Trajectories_Profiles(self): 
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotAttractor1"]).T 
        T = T.join(pd.DataFrame([[1,1,1]], columns=[g for g in x0], index=["Attractor1"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,0]], columns=[g for g in x0], index=["Attractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,1,0]], columns=[g for g in x0], index=["NotAttractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,0]], columns=[g for g in x0], index=["NotAttractor3"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,1]], columns=[g for g in x0], index=["NotAttractor4"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotAttractor5"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,1]], columns=[g for g in x0], index=["NotAttractor6"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        _, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"profiles",verbose=0)
        res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
        for i in [1,2]:
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->Attractor{i}"],1))
        for i in range(1,7):
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->NotAttractor{i}"],0))
        	
    def test_BONESIS_Trajectories_Attractors(self): ## same but with MP dynamics and comparing exact attractors instead of similarity, "profiles" gives the same thing (only comparing attractors)
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotAttractor1"]).T 
        T = T.join(pd.DataFrame([[1,1,1]], columns=[g for g in x0], index=["Attractor1"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,0]], columns=[g for g in x0], index=["Attractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,1,0]], columns=[g for g in x0], index=["NotAttractor2"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,0]], columns=[g for g in x0], index=["NotAttractor3"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,1,1]], columns=[g for g in x0], index=["NotAttractor4"]).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotAttractor5"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,1]], columns=[g for g in x0], index=["NotAttractor6"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        _, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"attractors",verbose=0)
        res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
        for i in [1,2]:
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->Attractor{i}"],1))
        for i in range(1,7):
        	self.assertTrue(np.isclose(res.loc["score"][f"initial->NotAttractor{i}"],0))
        	
    def test_MABOSS_Mutant_KO(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["Similar1"]).fillna(1).T  ## similarity is measured on gene outputs, i.e., outside of perturbed genes
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["Similar2"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["Dissimilar1"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["Dissimilar2"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        nee = MABOSS_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_permanent={'GDNF': 0}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'GDNF': 0},verbose=0)
        res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Similar1"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Similar2"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar1"],0.5))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar2"],0.5))
        
    def test_MABOSS_Mutant_OE(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["Reachable"]).fillna(1).T  ## similarity is measured on gene outputs, i.e., outside of perturbed genes
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["NotReachable"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["Dissimilar1"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["Dissimilar2"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MABOSS_SIM
        nee = MABOSS_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_permanent={'GDNF': 1}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_MABOSS_noMut = test(MABOSS_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'GDNF': 1},verbose=0)
        res = pd.DataFrame({"score": net_MABOSS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable"],0))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar1"],0.5))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar2"],0.5))
        
    def test_MPBN_Mutant_KO(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["Similar1"]).fillna(1).T  ## similarity is measured on gene outputs, i.e., outside of perturbed genes
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["Similar2"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["Dissimilar1"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["Dissimilar2"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_permanent={'GDNF': 0}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'GDNF': 0},verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Similar1"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Similar2"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar1"],0.5))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar2"],0.5))

    def test_MPBN_Mutant_OE(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["Reachable"]).fillna(1).T  ## similarity is measured on gene outputs, i.e., outside of perturbed genes
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["NotReachable"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["Dissimilar1"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["Dissimilar2"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_permanent={'GDNF': 1}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'GDNF': 1},verbose=0)
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable"],0))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar1"],0.5))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar2"],0.5))
        
    def test_BONESIS_Mutant_KO(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["NotReachable1"]).fillna(1).T  ## similarity is strict for bonesis
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["Reachable"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotReachable2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotReachable3"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        nee = BONESIS_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_permanent={'GDNF': 0}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'GDNF': 0},verbose=0)
        res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable1"],0))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable2"],0))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable3"],0))
        
    def test_BONESIS_Mutant_OE(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["Reachable"]).fillna(1).T  
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["NotReachable1"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["NotReachable2"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["NotReachable3"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import BONESIS_SIM
        nee = BONESIS_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_permanent={'GDNF': 1}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_BONESIS_noMut = test(BONESIS_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_permanent={'GDNF': 1},verbose=0)
        res = pd.DataFrame({"score": net_BONESIS_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable1"],0))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable2"],0))
        self.assertTrue(np.isclose(res.loc["score"]["initial->NotReachable3"],0))
        
    def test_MPBN_Mutant_KO_transient(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["Reachable1"]).fillna(1).T  ## similarity is measured on gene outputs, i.e., outside of perturbed genes
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["Reachable2"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["Dissimilar1"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["Dissimilar2"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_transient={'GDNF': 0}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_transient={'GDNF': 0},verbose=0) 
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable1"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable2"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar1"],0.5))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar2"],0.5))
        
    def test_MPBN_Mutant_OE_transient(self):
        x0, network_fname, seednb, njobs = self.create_instance()
        C = pd.DataFrame(pd.Series(x0), columns=["initial"])
        T = pd.DataFrame([], columns=[g for g in x0], index=["Reachable1"]).fillna(1).T  ## similarity is measured on gene outputs, i.e., outside of perturbed genes
        T = T.join(pd.DataFrame([], columns=[g for g in x0], index=["Reachable2"]).fillna(0).T, how="outer")
        T = T.join(pd.DataFrame([[0,0,1]], columns=[g for g in x0], index=["Dissimilar1"]).T, how="outer")
        T = T.join(pd.DataFrame([[1,0,0]], columns=[g for g in x0], index=["Dissimilar2"]).T, how="outer")
        from NORDic.UTILS.utils_sim import test
        from NORDic.UTILS.utils_sim import MPBN_SIM
        nee = MPBN_SIM(seednb,njobs)
        nee.update_network(network_fname, C, mutation_transient={'GDNF': 1}, verbose=False)
        self.assertEqual(len(nee.all_mutants),1)
        self.assertTrue('GDNF' in nee.all_mutants) ## test if taking into account gene properly
        _, net_MPBN_noMut = test(MPBN_SIM,seednb,njobs,network_fname,C,T,"profiles", mutation_transient={'GDNF': 1},verbose=0) ## changes initial state
        res = pd.DataFrame({"score": net_MPBN_noMut.max_values}).T
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable1"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Reachable2"],1))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar1"],0.5))
        self.assertTrue(np.isclose(res.loc["score"]["initial->Dissimilar2"],0.5))
        	
if __name__ == '__main__':
    unittest.main()
